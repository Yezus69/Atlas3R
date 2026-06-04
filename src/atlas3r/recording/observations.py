"""Convert validated Atlas3R recording frames into mapper observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.data.image_runtime import PillowDependencyError, require_pillow_image
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.recording.schema import Atlas3RRecording, RecordingFrame, resolve_recording_path


def observations_from_recording(
    recording: Atlas3RRecording,
    *,
    max_frames: int | None,
    keyframe_stride: int,
    depth_sigma_floor_m: float = 0.02,
) -> tuple[DepthObservation, ...]:
    """Load measured recording depth+pose frames as chronological DepthObservations."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    if keyframe_stride <= 0:
        raise ValueError("keyframe_stride: must be positive")
    if depth_sigma_floor_m <= 0.0:
        raise ValueError("depth_sigma_floor_m: must be positive")
    frames = recording.frames[::keyframe_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    observations = [
        observation_from_recording_frame(
            recording,
            frame,
            depth_sigma_floor_m=depth_sigma_floor_m,
        )
        for frame in frames
    ]
    if not observations:
        raise ValueError("recording: no frames selected for fusion")
    return tuple(observations)


def observation_from_recording_frame(
    recording: Atlas3RRecording,
    frame: RecordingFrame,
    *,
    depth_sigma_floor_m: float = 0.02,
) -> DepthObservation:
    """Load one measured depth+pose recording frame into the mapper boundary."""

    if frame.depth_path is None:
        raise ValueError(f"frame {frame.frame_id}: recording depth is required for fusion")
    if frame.T_world_camera is None:
        raise ValueError(f"frame {frame.frame_id}: recording pose is required for fusion")
    height = _int_manifest(recording, "height")
    width = _int_manifest(recording, "width")
    depth_m, valid_mask, confidence = _load_depth_payload(
        resolve_recording_path(recording, recording.root, frame.depth_path),
        height=height,
        width=width,
        depth_scale=frame.depth_scale,
    )
    rgb_u8 = _load_rgb_payload(recording, frame, height=height, width=width)
    depth_sigma = np.full(depth_m.shape, np.float32(depth_sigma_floor_m), dtype=np.float32)
    depth_sigma[~valid_mask] = np.float32(max(depth_sigma_floor_m, 1.0))
    T_world_camera = frame.T_world_camera.astype(np.float32, copy=True)
    camera_center = T_world_camera[:3, 3].astype(np.float32, copy=True)
    return DepthObservation(
        frame_id=frame.frame_id,
        camera=CameraModel(
            width=width,
            height=height,
            K=frame.K.astype(np.float32, copy=True),
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="atlas3r_recording",
        ),
        pose=PoseEstimate(
            frame_id=frame.frame_id,
            timestamp_ns=int(round(frame.timestamp_s * 1_000_000_000.0)),
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=camera_center,
            covariance_6x6=np.diag(np.full(6, 1e-6, dtype=np.float32)),
            confidence=1.0,
            tracking_state="OK",
            scale_source="external_pose",
            diagnostics={
                "coordinate_frame": str(recording.manifest["coordinate_frame"]),
                "depth_source": "recording",
                "pose_source": "recording",
                "source_metadata": dict(frame.source_metadata),
                "truth_boundary": dict(
                    cast(dict[str, object], recording.manifest["truth_boundary"])
                ),
            },
        ),
        depth_m=depth_m,
        depth_sigma_m=depth_sigma,
        confidence=confidence,
        static_mask=valid_mask,
        object_id=None,
        rgb_u8=rgb_u8,
        source="atlas3r_recording_measured_depth_pose",
    )


def _load_depth_payload(
    path: Path,
    *,
    height: int,
    width: int,
    depth_scale: float | None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "depth_m" not in payload.files:
                raise ValueError(f"{path}: depth NPZ must contain depth_m")
            depth_m = np.asarray(payload["depth_m"], dtype=np.float32)
            valid_mask = (
                np.asarray(payload["valid_depth_mask"], dtype=np.bool_)
                if "valid_depth_mask" in payload.files
                else depth_m > 0.0
            )
            confidence = (
                np.asarray(payload["confidence"], dtype=np.float32)
                if "confidence" in payload.files
                else valid_mask.astype(np.float32)
            )
    elif suffix == ".png":
        if depth_scale is None or depth_scale <= 0.0:
            raise ValueError(f"{path}: PNG depth requires positive depth_scale")
        raw = _load_png_array(path, height=height, width=width, mode=None)
        depth_m = np.asarray(raw, dtype=np.float32) / np.float32(depth_scale)
        valid_mask = depth_m > 0.0
        confidence = valid_mask.astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported depth extension {suffix!r}")
    _validate_hw("depth_m", depth_m, height=height, width=width)
    _validate_hw("valid_depth_mask", valid_mask, height=height, width=width)
    _validate_hw("confidence", confidence, height=height, width=width)
    depth_m = depth_m.astype(np.float32, copy=False)
    depth_m[~valid_mask] = np.float32(0.0)
    confidence = np.clip(confidence.astype(np.float32, copy=False), 0.0, 1.0)
    confidence[~valid_mask] = np.float32(0.0)
    valid_mask_bool: npt.NDArray[np.bool_] = valid_mask.astype(np.bool_, copy=False)
    return depth_m, valid_mask_bool, confidence


def _load_rgb_payload(
    recording: Atlas3RRecording,
    frame: RecordingFrame,
    *,
    height: int,
    width: int,
) -> npt.NDArray[np.uint8] | None:
    path = resolve_recording_path(recording, recording.root, frame.rgb_path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "rgb_u8" not in payload.files:
                return None
            rgb = np.asarray(payload["rgb_u8"], dtype=np.uint8)
    elif suffix in {".png", ".jpg", ".jpeg", ".ppm"}:
        rgb = _load_png_array(path, height=height, width=width, mode="RGB")
    else:
        return None
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"{path}: rgb_u8 must be uint8 HxWx3 at recording size")
    return rgb


def _load_png_array(path: Path, *, height: int, width: int, mode: str | None) -> npt.NDArray[Any]:
    try:
        image_module = require_pillow_image()
    except PillowDependencyError as exc:
        raise ValueError(f"{path}: Pillow is required to load PNG/JPEG recording frames") from exc
    resampling = getattr(image_module, "Resampling", image_module)
    with image_module.open(path) as image:
        loaded = image.convert(mode) if mode is not None else image
        resized = loaded.resize((width, height), resample=resampling.NEAREST)
        return np.asarray(resized)


def _validate_hw(name: str, array: npt.NDArray[Any], *, height: int, width: int) -> None:
    if array.shape != (height, width):
        raise ValueError(f"{name}: expected shape {(height, width)}, got {array.shape}")
    if array.size > 0 and not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: must contain finite values")


def _int_manifest(recording: Atlas3RRecording, key: str) -> int:
    value = recording.manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"manifest.{key}: must be an integer")
    return value


__all__ = [
    "observation_from_recording_frame",
    "observations_from_recording",
]
