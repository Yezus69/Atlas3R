"""Convert RGB teacher predictions into Atlas3R mapper observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherFrame
from atlas3r.teachers.external.vggt_arrays import VGGTRunStats, normalise_vggt_clip_prediction

RGB_TEACHER_TRUTH_FLAGS: dict[str, object] = {
    "accuracy_report": False,
    "diagnostic_only": True,
    "hidden_geometry_measured": False,
    "measured_depth_used": False,
    "measured_pose_used": False,
    "observed_only": True,
    "predicted_completion": False,
    "pseudo_depth_used": True,
    "pseudo_pose_used": True,
    "realtime_claim": False,
    "rgb_only_mapping_ready": False,
    "student_rgb_only_used": False,
    "teacher_geometry_used": True,
    "teacher_name": "vggt",
}

POSE_KEYS = ("T_world_camera", "extrinsic", "extrinsics", "T_camera_world")


@dataclass(frozen=True)
class TeacherObservationBatch:
    observations: tuple[DepthObservation, ...]
    payload: dict[str, npt.NDArray[Any] | np.generic]
    metadata: dict[str, object]


def frames_to_vggt_clip_payload(frames: Sequence[RGBTeacherFrame]) -> dict[str, npt.NDArray[Any]]:
    if not frames:
        raise ValueError("frames: at least one frame is required")
    height, width = frames[0].rgb_u8.shape[:2]
    for frame in frames:
        if frame.rgb_u8.shape[:2] != (height, width):
            raise ValueError("frames: all RGB frames in a teacher window must share H,W")
    images = np.stack([frame.rgb_u8 for frame in frames], axis=0).astype(np.uint8, copy=False)
    K = np.stack([frame.K for frame in frames], axis=0).astype(np.float32, copy=False)
    identities = np.repeat(
        np.eye(4, dtype=np.float32)[np.newaxis, :, :],
        len(frames),
        axis=0,
    )
    return {
        "images_rgb_u8": images,
        "depth_m": np.zeros((len(frames), height, width), dtype=np.float32),
        "valid_depth_mask": np.zeros((len(frames), height, width), dtype=np.bool_),
        "K": K,
        "T_world_camera": identities,
        "frame_ids": np.asarray([frame.frame_id for frame in frames], dtype=np.int32),
        "timestamps_s": np.asarray([frame.timestamp_s for frame in frames], dtype=np.float64),
        "center_index": np.asarray(0, dtype=np.int32),
    }


def teacher_prediction_to_observations(
    raw_prediction: object,
    *,
    frames: Sequence[RGBTeacherFrame],
    teacher_name: str,
    metric_scale_source: str,
) -> TeacherObservationBatch:
    """Normalize a VGGT-like prediction and build pseudo DepthObservation records."""

    clip_payload = frames_to_vggt_clip_payload(frames)
    raw, convention_metadata = _raw_prediction_with_T_world_camera(raw_prediction, len(frames))
    stats = VGGTRunStats()
    payload = normalise_vggt_clip_prediction(
        raw,
        clip_payload=clip_payload,
        align_to_source_pose="none",
        stats=stats,
    )
    observations: list[DepthObservation] = []
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32)
    timestamps_s = np.asarray(payload["timestamps_s"], dtype=np.float64)
    for frame_offset, frame_id in enumerate(frame_ids.tolist()):
        frame = frames[frame_offset]
        observations.append(
            _observation_from_payload_frame(
                payload=payload,
                frame=frame,
                frame_offset=frame_offset,
                frame_id=int(frame_id),
                timestamp_s=float(timestamps_s[frame_offset]),
                teacher_name=teacher_name,
                metric_scale_source=metric_scale_source,
                conversion_metadata=convention_metadata,
            )
        )
    metadata = {
        **stats.result_fields(),
        **convention_metadata,
        "depth_valid_pixel_ratio": _valid_ratio(np.asarray(payload["valid_mask"], dtype=np.bool_)),
        "metric_scale_source": metric_scale_source,
        "teacher_name": teacher_name,
    }
    return TeacherObservationBatch(
        observations=tuple(observations),
        payload=payload,
        metadata=metadata,
    )


def opencv_camera_from_world_to_T_world_camera(
    T_camera_world: object,
    *,
    clip_length: int | None = None,
) -> npt.NDArray[np.float32]:
    """Convert OpenCV-style camera-from-world extrinsics to `T_world_camera`."""

    array = _to_numpy(T_camera_world).astype(np.float64, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"T_camera_world: expected T,3,4 or T,4,4, got {array.shape}")
    if clip_length is not None and array.shape[0] != clip_length:
        raise ValueError(f"T_camera_world: expected {clip_length} transforms, got {array.shape[0]}")
    if array.shape[1:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[np.newaxis, :, :], array.shape[0], axis=0)
        padded[:, :3, :] = array
        array = padded
    if array.shape[1:] != (4, 4):
        raise ValueError(f"T_camera_world: expected T,3,4 or T,4,4, got {array.shape}")
    _validate_transform_sequence("T_camera_world", array)
    inverted = np.linalg.inv(array)
    _validate_transform_sequence("T_world_camera", inverted)
    return inverted.astype(np.float32, copy=False)


def _raw_prediction_with_T_world_camera(
    raw_prediction: object,
    clip_length: int,
) -> tuple[dict[str, object], dict[str, object]]:
    raw = _prediction_mapping(raw_prediction)
    key = next((candidate for candidate in POSE_KEYS if candidate in raw), None)
    if key is None:
        raise ValueError("teacher prediction: missing camera pose; refusing to map RGB as measured")
    metadata: dict[str, object] = {
        "coordinate_convention": "x_right_y_down_z_forward",
        "pose_input_key": key,
    }
    if key == "T_world_camera":
        transforms = _sequence_transforms(raw[key], key=key, clip_length=clip_length)
        raw["T_world_camera"] = transforms
        metadata["pose_conversion"] = "teacher_T_world_camera_validated"
    else:
        raw["T_world_camera"] = opencv_camera_from_world_to_T_world_camera(
            raw[key],
            clip_length=clip_length,
        )
        metadata["pose_conversion"] = "opencv_camera_from_world_inverted_to_T_world_camera"
    return raw, metadata


def _observation_from_payload_frame(
    *,
    payload: Mapping[str, Any],
    frame: RGBTeacherFrame,
    frame_offset: int,
    frame_id: int,
    timestamp_s: float,
    teacher_name: str,
    metric_scale_source: str,
    conversion_metadata: Mapping[str, object],
) -> DepthObservation:
    depth = np.asarray(payload["depth_m"][frame_offset], dtype=np.float32)
    sigma = np.asarray(payload["depth_sigma_m"][frame_offset], dtype=np.float32)
    confidence = np.asarray(payload["confidence"][frame_offset], dtype=np.float32)
    valid = np.asarray(payload["valid_mask"][frame_offset], dtype=np.bool_)
    T_world_camera = np.asarray(payload["T_world_camera"][frame_offset], dtype=np.float32)
    K = np.asarray(payload["K"][frame_offset], dtype=np.float32)
    _validate_transform_sequence("T_world_camera", T_world_camera[np.newaxis, :, :])
    mean_sigma = float(np.mean(sigma[valid])) if np.any(valid) else 1.0
    covariance = np.diag(np.full(6, max(mean_sigma * mean_sigma, 1e-8), dtype=np.float32))
    truth = {
        **RGB_TEACHER_TRUTH_FLAGS,
        "teacher_name": teacher_name,
        "metric_scale_source": metric_scale_source,
    }
    return DepthObservation(
        frame_id=frame_id,
        camera=CameraModel(
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            K=K,
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source=f"{teacher_name}_teacher_intrinsics_or_input_metadata",
        ),
        pose=PoseEstimate(
            frame_id=frame_id,
            timestamp_ns=int(round(timestamp_s * 1_000_000_000.0)),
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
            covariance_6x6=covariance.astype(np.float32),
            confidence=float(np.mean(confidence[valid])) if np.any(valid) else 0.0,
            tracking_state="OK" if np.any(valid) else "LOW_CONFIDENCE",
            scale_source="rgb_prior",
            diagnostics={
                **dict(conversion_metadata),
                "metric_scale_source": metric_scale_source,
                "source_rgb_path": frame.source_path,
                "truth_boundary": truth,
            },
        ),
        depth_m=depth,
        depth_sigma_m=sigma,
        confidence=confidence,
        static_mask=valid,
        object_id=None,
        rgb_u8=frame.rgb_u8.copy(),
        source=f"{teacher_name}_pseudo_depth_pose",
    )


def _prediction_mapping(raw_prediction: object) -> dict[str, object]:
    if isinstance(raw_prediction, Mapping):
        return dict(raw_prediction)
    keys = (
        "depth",
        "depth_m",
        "depth_conf",
        "confidence",
        "depth_sigma_m",
        "extrinsic",
        "extrinsics",
        "intrinsic",
        "intrinsics",
        "K",
        "T_world_camera",
        "T_camera_world",
        "world_points",
        "pointmap_world_m",
        "pointmap_camera_m",
        "normal_camera",
    )
    return {key: getattr(raw_prediction, key) for key in keys if hasattr(raw_prediction, key)}


def _sequence_transforms(
    value: object,
    *,
    key: str,
    clip_length: int,
) -> npt.NDArray[np.float32]:
    array = _to_numpy(value).astype(np.float64, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (clip_length, 4, 4):
        raise ValueError(f"{key}: expected shape T,4,4, got {array.shape}")
    _validate_transform_sequence(key, array)
    return array.astype(np.float32, copy=False)


def _validate_transform_sequence(name: str, transforms: npt.NDArray[np.floating[Any]]) -> None:
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{name}: expected T,4,4 transforms")
    if not np.all(np.isfinite(transforms)):
        raise ValueError(f"{name}: transforms must be finite")
    bottom = transforms[:, 3, :]
    expected = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(bottom, expected[None, :], atol=1e-5):
        raise ValueError(f"{name}: transform bottom row must be [0,0,0,1]")
    det = np.linalg.det(transforms[:, :3, :3])
    if np.any(np.abs(det) < 1e-6) or np.any(~np.isfinite(det)):
        raise ValueError(f"{name}: rotation block is degenerate")


def _valid_ratio(mask: npt.NDArray[np.bool_]) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def _to_numpy(value: object) -> npt.NDArray[Any]:
    if hasattr(value, "detach"):
        value = cast(Any, value).detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "RGB_TEACHER_TRUTH_FLAGS",
    "TeacherObservationBatch",
    "frames_to_vggt_clip_payload",
    "opencv_camera_from_world_to_T_world_camera",
    "teacher_prediction_to_observations",
]
