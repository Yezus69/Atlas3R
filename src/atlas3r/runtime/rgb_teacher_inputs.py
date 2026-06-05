"""RGB input loading for teacher-assisted runtime mapping."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics
from atlas3r.data.frame_source import PPMSequenceFrameSource, load_npz_clip_frames
from atlas3r.data.image_runtime import PillowDependencyError, require_pillow_image
from atlas3r.recording.schema import (
    RECORDING_MANIFEST_FILENAME,
    Atlas3RRecording,
    RecordingFrame,
    load_recording,
    resolve_recording_path,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
INTRINSICS_SIDECAR = "intrinsics.npz"


@dataclass(frozen=True)
class RGBTeacherFrame:
    frame_id: int
    timestamp_s: float
    rgb_u8: npt.NDArray[np.uint8]
    K: npt.NDArray[np.float32]
    source_path: str
    source_metadata: dict[str, object]


@dataclass(frozen=True)
class RGBTeacherInput:
    root: Path
    source_type: str
    source_dataset: str
    source_sequence: str
    frames: tuple[RGBTeacherFrame, ...]
    recording: Atlas3RRecording | None = None

    @property
    def width(self) -> int:
        return int(self.frames[0].rgb_u8.shape[1])

    @property
    def height(self) -> int:
        return int(self.frames[0].rgb_u8.shape[0])

    @property
    def measured_depth_available(self) -> bool:
        return bool(self.recording is not None and self.recording.manifest["depth_present"])

    @property
    def measured_pose_available(self) -> bool:
        return bool(self.recording is not None and self.recording.manifest["pose_present"])


def load_rgb_teacher_input(path: str | Path) -> RGBTeacherInput:
    """Load RGB frames without consuming measured depth or pose for mapping."""

    source = Path(path)
    if source.is_dir() and (source / RECORDING_MANIFEST_FILENAME).is_file():
        return _load_recording_rgb_only(source)
    if source.is_file() and source.suffix.lower() == ".npz":
        packets = load_npz_clip_frames(source)
        frames = tuple(
            RGBTeacherFrame(
                frame_id=packet.frame_id,
                timestamp_s=float(packet.timestamp_ns) / 1_000_000_000.0,
                rgb_u8=packet.rgb_u8.copy(),
                K=packet.K_model.astype(np.float32, copy=True),
                source_path=str(source),
                source_metadata=dict(packet.camera_metadata),
            )
            for packet in packets
        )
        return _input_from_frames(source, "npz_clip", frames)
    if source.is_dir():
        return _load_image_folder(source)
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        return _load_video_file(source)
    raise ValueError(
        f"{source}: expected an atlas3r_recording, RGB image folder, NPZ clip, or video file"
    )


def select_rgb_teacher_frames(
    frames: Sequence[RGBTeacherFrame],
    *,
    max_frames: int | None,
    frame_stride: int,
) -> tuple[RGBTeacherFrame, ...]:
    if frame_stride <= 0:
        raise ValueError("frame_stride: must be positive")
    selected = tuple(frames[::frame_stride])
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames: must be positive when provided")
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("input: no RGB frames selected")
    return selected


def default_intrinsics(width: int, height: int) -> npt.NDArray[np.float32]:
    focal = float(max(width, height))
    return np.asarray(
        [[focal, 0.0, (width - 1.0) * 0.5], [0.0, focal, (height - 1.0) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _load_recording_rgb_only(path: Path) -> RGBTeacherInput:
    recording = load_recording(path)
    height = _manifest_int(recording, "height")
    width = _manifest_int(recording, "width")
    frames = tuple(
        RGBTeacherFrame(
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            rgb_u8=_load_recording_rgb(recording, frame, height=height, width=width),
            K=frame.K.astype(np.float32, copy=True),
            source_path=frame.rgb_path,
            source_metadata={
                **dict(frame.source_metadata),
                "recording_rgb_only": True,
                "measured_depth_available": frame.depth_path is not None,
                "measured_pose_available": frame.T_world_camera is not None,
            },
        )
        for frame in recording.frames
    )
    return RGBTeacherInput(
        root=path,
        source_type="atlas3r_recording_rgb_only",
        source_dataset=str(recording.manifest["source_dataset"]),
        source_sequence=str(recording.manifest["source_sequence"]),
        frames=frames,
        recording=recording,
    )


def _load_image_folder(path: Path) -> RGBTeacherInput:
    npz_paths = tuple(
        item for item in sorted(path.glob("*.npz")) if item.name != INTRINSICS_SIDECAR
    )
    if npz_paths:
        frames = _load_npz_image_folder(path, npz_paths)
        return _input_from_frames(path, "npz_image_folder", frames)

    ppm_paths = tuple(sorted(path.glob("*.ppm")))
    sidecar = path / INTRINSICS_SIDECAR
    if ppm_paths and sidecar.is_file():
        packets = tuple(PPMSequenceFrameSource(path).frames())
        frames = tuple(
            RGBTeacherFrame(
                frame_id=packet.frame_id,
                timestamp_s=float(packet.timestamp_ns) / 1_000_000_000.0,
                rgb_u8=packet.rgb_u8.copy(),
                K=packet.K_model.astype(np.float32, copy=True),
                source_path=str(packet.camera_metadata.get("source_file", path)),
                source_metadata=dict(packet.camera_metadata),
            )
            for packet in packets
        )
        return _input_from_frames(path, "ppm_image_folder", frames)

    image_paths = tuple(
        item for item in sorted(path.iterdir()) if item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"{path}: no RGB images found")
    frames = _load_pillow_image_folder(path, image_paths)
    return _input_from_frames(path, "image_folder", frames)


def _load_npz_image_folder(
    root: Path,
    paths: Sequence[Path],
) -> tuple[RGBTeacherFrame, ...]:
    first_rgb: npt.NDArray[np.uint8] | None = None
    rgb_frames: list[npt.NDArray[np.uint8]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if "rgb_u8" not in payload.files:
                raise ValueError(f"{path}: RGB NPZ must contain rgb_u8")
            rgb = _validate_rgb(str(path), np.asarray(payload["rgb_u8"]))
        if first_rgb is None:
            first_rgb = rgb
        elif rgb.shape != first_rgb.shape:
            raise ValueError(f"{path}: RGB frame shape must match first frame {first_rgb.shape}")
        rgb_frames.append(rgb)
    assert first_rgb is not None
    K = _folder_intrinsics(root, width=first_rgb.shape[1], height=first_rgb.shape[0])
    return tuple(
        RGBTeacherFrame(
            frame_id=index,
            timestamp_s=float(index),
            rgb_u8=rgb.copy(),
            K=K.copy(),
            source_path=str(path),
            source_metadata={"source_format": "npz_image_folder", "frame_index": index},
        )
        for index, (path, rgb) in enumerate(zip(paths, rgb_frames, strict=True))
    )


def _load_pillow_image_folder(
    root: Path,
    paths: Sequence[Path],
) -> tuple[RGBTeacherFrame, ...]:
    try:
        image_module = require_pillow_image()
    except PillowDependencyError as exc:
        raise ValueError(
            f"{root}: Pillow is required to load PNG/JPEG/BMP image folders; "
            "install with python -m pip install -e .[train]"
        ) from exc
    rgb_frames: list[npt.NDArray[np.uint8]] = []
    first_shape: tuple[int, ...] | None = None
    for path in paths:
        with image_module.open(path) as image:
            rgb = _validate_rgb(str(path), np.asarray(image.convert("RGB")))
        if first_shape is None:
            first_shape = rgb.shape
        elif rgb.shape != first_shape:
            raise ValueError(f"{path}: RGB image shape must match first frame {first_shape}")
        rgb_frames.append(rgb)
    if first_shape is None:
        raise ValueError(f"{root}: no image frames loaded")
    K = _folder_intrinsics(root, width=first_shape[1], height=first_shape[0])
    return tuple(
        RGBTeacherFrame(
            frame_id=index,
            timestamp_s=float(index),
            rgb_u8=rgb.copy(),
            K=K.copy(),
            source_path=str(path),
            source_metadata={"source_format": "image_folder", "frame_index": index},
        )
        for index, (path, rgb) in enumerate(zip(paths, rgb_frames, strict=True))
    )


def _load_video_file(path: Path) -> RGBTeacherInput:
    if importlib.util.find_spec("cv2") is not None:
        frames = _load_video_with_cv2(path)
        return _input_from_frames(path, "video_cv2", frames)
    if importlib.util.find_spec("imageio") is not None:
        frames = _load_video_with_imageio(path)
        return _input_from_frames(path, "video_imageio", frames)
    raise ValueError(
        f"{path}: video input requires optional opencv-python or imageio; "
        "install one of them in the active environment"
    )


def _load_video_with_cv2(path: Path) -> tuple[RGBTeacherFrame, ...]:
    cv2 = importlib.import_module("cv2")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path}: OpenCV failed to open video")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[npt.NDArray[np.uint8]] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(_validate_rgb(str(path), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return _frames_from_rgb_arrays(path, "video_cv2", frames, fps=fps)


def _load_video_with_imageio(path: Path) -> tuple[RGBTeacherFrame, ...]:
    imageio = importlib.import_module("imageio.v3")
    frames = [_validate_rgb(str(path), np.asarray(frame)) for frame in imageio.imiter(path)]
    return _frames_from_rgb_arrays(path, "video_imageio", frames, fps=30.0)


def _frames_from_rgb_arrays(
    path: Path,
    source_format: str,
    frames: Sequence[npt.NDArray[np.uint8]],
    *,
    fps: float,
) -> tuple[RGBTeacherFrame, ...]:
    if not frames:
        raise ValueError(f"{path}: no RGB frames decoded")
    first_shape = frames[0].shape
    for frame in frames:
        if frame.shape != first_shape:
            raise ValueError(f"{path}: decoded video frames must have a stable shape")
    K = default_intrinsics(width=first_shape[1], height=first_shape[0])
    return tuple(
        RGBTeacherFrame(
            frame_id=index,
            timestamp_s=float(index) / max(fps, 1e-6),
            rgb_u8=frame.copy(),
            K=K.copy(),
            source_path=str(path),
            source_metadata={"source_format": source_format, "frame_index": index},
        )
        for index, frame in enumerate(frames)
    )


def _load_recording_rgb(
    recording: Atlas3RRecording,
    frame: RecordingFrame,
    *,
    height: int,
    width: int,
) -> npt.NDArray[np.uint8]:
    path = resolve_recording_path(recording, recording.root, frame.rgb_path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "rgb_u8" not in payload.files:
                raise ValueError(f"{path}: RGB NPZ must contain rgb_u8")
            return _validate_rgb(
                str(path), np.asarray(payload["rgb_u8"]), height=height, width=width
            )
    try:
        image_module = require_pillow_image()
    except PillowDependencyError as exc:
        raise ValueError(f"{path}: Pillow is required to load recording RGB image files") from exc
    with image_module.open(path) as image:
        resampling = getattr(image_module, "Resampling", image_module)
        resized = image.convert("RGB").resize((width, height), resample=resampling.NEAREST)
        rgb = np.asarray(resized)
    return _validate_rgb(str(path), rgb, height=height, width=width)


def _folder_intrinsics(root: Path, *, width: int, height: int) -> npt.NDArray[np.float32]:
    sidecar = root / INTRINSICS_SIDECAR
    if not sidecar.is_file():
        return default_intrinsics(width, height)
    with np.load(sidecar, allow_pickle=False) as payload:
        if "K" not in payload.files:
            raise ValueError(f"{sidecar}: missing required K array")
        K = np.asarray(payload["K"], dtype=np.float32)
    if K.shape == (1, 3, 3):
        K = K[0]
    if K.shape != (3, 3):
        raise ValueError(f"{sidecar}: K must have shape 3x3")
    return validate_intrinsics(f"{sidecar}: K", K).astype(np.float32, copy=True)


def _input_from_frames(
    root: Path,
    source_type: str,
    frames: Sequence[RGBTeacherFrame],
) -> RGBTeacherInput:
    if not frames:
        raise ValueError(f"{root}: no RGB frames loaded")
    return RGBTeacherInput(
        root=root,
        source_type=source_type,
        source_dataset=source_type,
        source_sequence=root.stem or source_type,
        frames=tuple(frames),
    )


def _validate_rgb(
    field_name: str,
    value: npt.NDArray[Any],
    *,
    height: int | None = None,
    width: int | None = None,
) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{field_name}: rgb_u8 must be uint8 HxWx3")
    if height is not None and width is not None and array.shape[:2] != (height, width):
        raise ValueError(f"{field_name}: rgb_u8 must match recording size {(height, width)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name}: rgb_u8 must contain finite values")
    return array.astype(np.uint8, copy=True)


def _manifest_int(recording: Atlas3RRecording, key: str) -> int:
    value = recording.manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"manifest.{key}: must be an integer")
    return value


__all__ = [
    "RGBTeacherFrame",
    "RGBTeacherInput",
    "default_intrinsics",
    "load_rgb_teacher_input",
    "select_rgb_teacher_frames",
]
