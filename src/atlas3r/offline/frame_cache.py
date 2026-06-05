"""Dependency-safe frame cache for Offline World Builder runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.frames import CameraModel, FramePacket
from atlas3r.contracts.truth import TruthBoundary
from atlas3r.input.metadata import camera_from_focal
from atlas3r.input.video import VideoInspection, inspect_video_input, read_ppm_image
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

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "frame_path": self.frame_path,
            "source_uri": self.source_uri,
            "width": self.width,
            "height": self.height,
            "camera": self.camera.to_dict(),
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
            future_module="provide a readable MP4 or PPM image sequence",
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
    if inspection.kind == "video_file":
        failure_points.append(
            FailurePoint(
                module="frame_cache",
                code="video_decoder_unavailable",
                severity="warning",
                status="unavailable",
                why=inspection.message,
                dependency_missing="imageio or opencv video decoder",
                future_module="dependency-safe MP4 decoder adapter",
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
    ppm_paths = _resolve_ppm_inputs(input_path, inspection)[:max_frames]
    if not ppm_paths:
        failure_points.append(
            FailurePoint(
                module="frame_cache",
                code="no_dependency_free_frames",
                severity="warning",
                status="unavailable",
                why="input exists but contains no dependency-free PPM frames",
                input_missing="*.ppm frames",
                dependency_missing="PNG/JPEG image decoder adapter",
                future_module="image-folder decoder adapter for additional formats",
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
    frames, records, camera = _load_ppm_frames(ppm_paths, root)
    write_jsonl(frame_index, [record.to_dict() for record in records])
    return FrameCacheResult(
        status="complete",
        inspection=inspection,
        frames=tuple(frames),
        records=tuple(records),
        camera=camera,
        frame_index_path="frames/frame_index.jsonl",
    )


def _resolve_ppm_inputs(input_path: str | Path, inspection: VideoInspection) -> list[Path]:
    path = Path(input_path)
    if inspection.kind == "image_file" and path.suffix.lower() == ".ppm":
        return [path]
    if inspection.kind == "image_directory":
        return sorted(item for item in path.iterdir() if item.suffix.lower() == ".ppm")
    return []


def _load_ppm_frames(
    ppm_paths: list[Path], run_dir: Path
) -> tuple[list[FramePacket], list[FrameRecord], CameraModel]:
    first_rgb = read_ppm_image(ppm_paths[0])
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
    for frame_id, source_path in enumerate(ppm_paths):
        rgb = first_rgb if frame_id == 0 else read_ppm_image(source_path)
        if rgb.shape[:2] != (height, width):
            raise InputDataError(f"{source_path} dimensions do not match first frame")
        target = run_dir / "frames" / "images" / f"frame_{frame_id:06d}.ppm"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        timestamp_ns = frame_id * 33_333_333
        quality = _score_frame_quality(rgb, previous_rgb)
        relative_frame_path = relative_to_run(target, run_dir)
        frames.append(
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                rgb_u8=rgb,
                K_original=None,
                K_model=camera.K,
                resize_transform=np.eye(3, dtype=np.float32),
                camera_metadata={"source_format": "ppm", "camera_source": camera.source},
                source_uri=str(source_path),
            )
        )
        records.append(
            FrameRecord(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                frame_path=relative_frame_path,
                source_uri=str(source_path),
                width=width,
                height=height,
                camera=camera,
                quality=quality,
                truth_boundary=truth,
            )
        )
        previous_rgb = rgb
    return frames, records, camera


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
