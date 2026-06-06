"""Dependency-safe frame cache for Offline World Builder runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.frames import CameraModel, FramePacket
from atlas3r.contracts.truth import TruthBoundary
from atlas3r.input.metadata import camera_from_focal
from atlas3r.input.video import (
    DecodedFrame,
    VideoDependencyError,
    VideoInspection,
    decode_input_frames,
    inspect_video_input,
    write_ppm_image,
)
from atlas3r.offline.run_manifest import FailurePoint, relative_to_run, write_jsonl


class InputDataError(RuntimeError):
    """Raised for true input/IO failures."""


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float
    exposure_score: float
    visual_change_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "blur_score": self.blur_score,
            "exposure_score": self.exposure_score,
            "visual_change_score": self.visual_change_score,
        }


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_ns: int
    frame_path: str
    source_uri: str
    width: int
    height: int
    camera: CameraModel
    quality: FrameQuality
    truth_boundary: TruthBoundary
    original_frame_index: int | None = None
    decoder_name: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "original_frame_index": self.original_frame_index,
            "timestamp_ns": self.timestamp_ns,
            "frame_path": self.frame_path,
            "source_uri": self.source_uri,
            "width": self.width,
            "height": self.height,
            "camera": self.camera.to_dict(),
            "decoder_name": self.decoder_name,
            "quality": self.quality.to_dict(),
            "truth_boundary": self.truth_boundary.to_dict(),
        }


@dataclass(frozen=True)
class FrameCacheResult:
    status: str
    inspection: VideoInspection
    frames: tuple[FramePacket, ...]
    records: tuple[FrameRecord, ...]
    camera: CameraModel | None
    frame_index_path: str
    input_error: bool = False


def build_frame_cache(
    input_path: str | Path,
    run_dir: str | Path,
    *,
    max_frames: int,
    failure_points: list[FailurePoint],
    allow_missing: bool = False,
) -> FrameCacheResult:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    root = Path(run_dir)
    frame_index = root / "frames" / "frame_index.jsonl"
    inspection = inspect_video_input(input_path)
    if inspection.kind == "missing":
        failure = FailurePoint(
            module="frame_cache",
            code="input_missing",
            severity="error",
            status="unavailable",
            why="input path does not exist",
            input_missing=str(input_path),
            future_module="provide a readable MP4 or image sequence",
            artifact_path="frames/frame_index.jsonl",
        )
        failure_points.append(failure)
        write_jsonl(frame_index, [])
        if not allow_missing:
            raise InputDataError(f"input path does not exist: {input_path}")
        return FrameCacheResult(
            status="unavailable",
            inspection=inspection,
            frames=(),
            records=(),
            camera=None,
            frame_index_path="frames/frame_index.jsonl",
            input_error=True,
        )
    try:
        decoded_frames = decode_input_frames(input_path, max_frames=max_frames)
    except (VideoDependencyError, ValueError) as exc:
        failure_points.append(
            FailurePoint(
                module="frame_cache",
                code=_decoder_failure_code(inspection),
                severity="warning",
                status="unavailable",
                why=str(exc),
                dependency_missing=_decoder_dependency_name(inspection),
                future_module="dependency-safe image/video decoder",
                artifact_path="frames/frame_index.jsonl",
            )
        )
        write_jsonl(frame_index, [])
        return FrameCacheResult(
            status="unavailable",
            inspection=inspection,
            frames=(),
            records=(),
            camera=None,
            frame_index_path="frames/frame_index.jsonl",
        )
    if not decoded_frames:
        failure_points.append(
            FailurePoint(
                module="frame_cache",
                code="no_decoded_frames",
                severity="warning",
                status="unavailable",
                why="input exists but no frames could be decoded",
                input_missing="decodable frames",
                dependency_missing=_decoder_dependency_name(inspection),
                future_module="dependency-safe image/video decoder",
                artifact_path="frames/frame_index.jsonl",
            )
        )
        write_jsonl(frame_index, [])
        return FrameCacheResult(
            status="unavailable",
            inspection=inspection,
            frames=(),
            records=(),
            camera=None,
            frame_index_path="frames/frame_index.jsonl",
        )
    frames, records, camera = _load_decoded_frames(decoded_frames, root)
    write_jsonl(frame_index, [record.to_dict() for record in records])
    return FrameCacheResult(
        status="complete",
        inspection=inspection,
        frames=tuple(frames),
        records=tuple(records),
        camera=camera,
        frame_index_path="frames/frame_index.jsonl",
    )


def _load_decoded_frames(
    decoded_frames: tuple[DecodedFrame, ...], run_dir: Path
) -> tuple[list[FramePacket], list[FrameRecord], CameraModel]:
    first_rgb = decoded_frames[0].rgb_u8
    height, width = int(first_rgb.shape[0]), int(first_rgb.shape[1])
    focal = float(max(width, height))
    camera = camera_from_focal(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        source="guessed_from_image_size",
        confidence=0.25,
    )
    truth = TruthBoundary.unanchored_mp4("RGB frames have no physical scale anchor yet.")
    frames: list[FramePacket] = []
    records: list[FrameRecord] = []
    previous_rgb: NDArray[np.uint8] | None = None
    for frame_id, decoded in enumerate(decoded_frames):
        rgb = first_rgb if frame_id == 0 else decoded.rgb_u8
        if rgb.shape[:2] != (height, width):
            raise InputDataError(f"{decoded.source_uri} dimensions do not match first frame")
        target = run_dir / "frames" / "images" / f"frame_{frame_id:06d}.ppm"
        write_ppm_image(target, rgb)
        quality = _score_frame_quality(rgb, previous_rgb)
        relative_frame_path = relative_to_run(target, run_dir)
        frames.append(
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=decoded.timestamp_ns,
                rgb_u8=rgb,
                K_original=None,
                K_model=camera.K,
                resize_transform=np.eye(3, dtype=np.float32),
                camera_metadata={
                    "source_format": "normalized_ppm",
                    "camera_source": camera.source,
                    "decoder_name": decoded.decoder_name,
                    "original_frame_index": decoded.original_frame_index,
                },
                source_uri=decoded.source_uri,
            )
        )
        records.append(
            FrameRecord(
                frame_id=frame_id,
                timestamp_ns=decoded.timestamp_ns,
                frame_path=relative_frame_path,
                source_uri=decoded.source_uri,
                width=width,
                height=height,
                camera=camera,
                quality=quality,
                truth_boundary=truth,
                original_frame_index=decoded.original_frame_index,
                decoder_name=decoded.decoder_name,
            )
        )
        previous_rgb = rgb
    return frames, records, camera


def _decoder_failure_code(inspection: VideoInspection) -> str:
    if inspection.kind == "video_file":
        return "video_decoder_unavailable"
    if inspection.decoder == "optional_image":
        return "image_decoder_unavailable"
    return "frame_decoder_unavailable"


def _decoder_dependency_name(inspection: VideoInspection) -> str | None:
    if inspection.kind == "video_file":
        return "imageio or OpenCV video decoder"
    if inspection.decoder == "optional_image":
        return "Pillow, imageio, or OpenCV image decoder"
    return None


def _score_frame_quality(
    rgb: NDArray[np.uint8], previous_rgb: NDArray[np.uint8] | None
) -> FrameQuality:
    gray = rgb.astype(np.float32).mean(axis=2)
    gx = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0) if gray.shape[1] > 1 else 0.0
    gy = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0) if gray.shape[0] > 1 else 0.0
    blur_score = float(min(1.0, gx + gy))
    mean_luma = float(gray.mean() / 255.0)
    exposure_score = float(max(0.0, 1.0 - abs(mean_luma - 0.5) * 2.0))
    if previous_rgb is None:
        visual_change_score = 0.0
    else:
        visual_change_score = float(np.abs(rgb.astype(np.float32) - previous_rgb).mean() / 255.0)
    return FrameQuality(
        blur_score=blur_score,
        exposure_score=exposure_score,
        visual_change_score=visual_change_score,
    )
