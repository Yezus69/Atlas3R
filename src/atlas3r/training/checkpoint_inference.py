"""Dependency-safe Phase 4B inference bridge for tiny training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, FramePacket, PoseEstimate
from atlas3r.data.student_clip import student_clip_from_frame_packets
from atlas3r.mapping.observations import DepthObservation
from atlas3r.models.student.contracts import CAMERA_COORDINATE_FRAME, StudentClipInput
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.training.torch_runtime import require_torch, select_device

TINY_DEPTH_POSE_CHECKPOINT_FORMAT = "atlas3r_tiny_depth_pose_checkpoint"
TINY_DEPTH_POSE_OBSERVATION_SOURCE = "tiny_depth_pose_checkpoint_inference"

_REQUIRED_TRUTH_BOUNDARY: Mapping[str, bool] = {
    "training_mvp": True,
    "synthetic_only": True,
    "real_capture_model": False,
    "usable_for_realtime_mapping": False,
    "usable_for_mapping": False,
    "accuracy_report": False,
    "performance_report": False,
    "learned_inference": True,
    "generalizes_to_real_world": False,
}


@dataclass(frozen=True)
class TinyDepthPoseCheckpoint:
    """Loaded Phase 4A tiny checkpoint plus model metadata."""

    path: Path
    model: Any
    step: int
    config: dict[str, object]
    model_config: dict[str, object]
    metrics: dict[str, object]
    truth_boundary: dict[str, object]
    device: str


@dataclass(frozen=True)
class TinyDepthPosePrediction:
    """NumPy prediction batch produced by `TinyDepthPoseNet`."""

    frame_ids: tuple[int, ...]
    depth_m: npt.NDArray[np.float32]
    depth_sigma_m: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    camera_center_world_m: npt.NDArray[np.float32]
    T_world_camera: npt.NDArray[np.float32]
    intrinsics: npt.NDArray[np.float32]
    truth_boundary: dict[str, object]
    coordinate_frame: str
    metadata: dict[str, object]


def load_tiny_depth_pose_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> TinyDepthPoseCheckpoint:
    """Load a Phase 4A tiny checkpoint and instantiate its Torch model lazily."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"{path}: checkpoint file does not exist")

    resolved_device = select_device(device)
    torch = require_torch()
    try:
        payload = torch.load(path, map_location=resolved_device)
    except Exception as exc:
        raise ValueError(f"{path}: failed to load checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload must be a dictionary")

    if payload.get("format_name") != TINY_DEPTH_POSE_CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: expected format_name={TINY_DEPTH_POSE_CHECKPOINT_FORMAT!r}")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"{path}: unsupported checkpoint format_version")

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path}: missing model_state_dict")
    truth_boundary = _validated_truth_boundary(path, payload.get("truth_boundary"))
    config = _dict_payload_field(path, payload, "config")
    model_config = _dict_payload_field(path, payload, "model_config")
    metrics = _dict_payload_field(path, payload, "metrics")
    hidden_channels = _hidden_channels(path, model_config)

    from atlas3r.training.tiny_depth_pose_model import TinyDepthPoseNet

    model = TinyDepthPoseNet(hidden_channels=hidden_channels).to(resolved_device)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise ValueError(f"{path}: model_state_dict is incompatible with TinyDepthPoseNet") from exc
    model.eval()

    return TinyDepthPoseCheckpoint(
        path=path,
        model=model,
        step=int(payload["step"]),
        config=config,
        model_config=model_config,
        metrics=metrics,
        truth_boundary=truth_boundary,
        device=resolved_device,
    )


def predict_tiny_depth_pose_student_clip(
    checkpoint: TinyDepthPoseCheckpoint,
    clip: StudentClipInput,
) -> TinyDepthPosePrediction:
    """Run `TinyDepthPoseNet` over an existing `StudentClipInput` batch."""

    if not isinstance(checkpoint, TinyDepthPoseCheckpoint):
        raise ValueError("checkpoint: must be a TinyDepthPoseCheckpoint")
    if not isinstance(clip, StudentClipInput):
        raise ValueError("clip: must be a StudentClipInput")
    if clip.coordinate_frame != CAMERA_COORDINATE_FRAME:
        raise ValueError(f"clip.coordinate_frame: must be {CAMERA_COORDINATE_FRAME}")

    torch = require_torch()
    images = _clip_images_float32(clip)
    intrinsics = clip.batched_intrinsics()
    batch_size, frame_count, _channels, height, width = images.shape
    flat_images = images.reshape(batch_size * frame_count, 3, height, width)
    flat_intrinsics = intrinsics.reshape(batch_size * frame_count, 3, 3)

    with torch.no_grad():
        output = checkpoint.model(
            torch.from_numpy(flat_images).to(checkpoint.device),
            torch.from_numpy(flat_intrinsics).to(checkpoint.device),
        )

    depth_m = _bthw_output(output, "depth_m", batch_size, frame_count, height, width)
    depth_sigma_m = _bthw_output(
        output,
        "depth_sigma_m",
        batch_size,
        frame_count,
        height,
        width,
    )
    confidence = _bthw_output(output, "confidence", batch_size, frame_count, height, width)
    validate_confidence = (confidence >= 0.0) & (confidence <= 1.0)
    if not bool(np.all(validate_confidence)):
        raise ValueError("confidence: model output must be in [0, 1]")
    centers = _camera_center_output(output, batch_size, frame_count)
    T_world_camera = _prediction_transforms_from_centers(clip, centers)

    return TinyDepthPosePrediction(
        frame_ids=clip.frame_ids,
        depth_m=depth_m,
        depth_sigma_m=depth_sigma_m,
        confidence=confidence,
        camera_center_world_m=centers,
        T_world_camera=T_world_camera,
        intrinsics=intrinsics,
        truth_boundary=dict(checkpoint.truth_boundary),
        coordinate_frame=clip.coordinate_frame,
        metadata={
            "source": TINY_DEPTH_POSE_OBSERVATION_SOURCE,
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_step": checkpoint.step,
            "model_name": checkpoint.model_config.get("model_name", "TinyDepthPoseNet"),
            "input_metadata": dict(clip.metadata),
            "pose_rotation_source": (
                "T_world_camera_prior" if clip.T_world_camera_prior is not None else "identity"
            ),
            "pose_translation_source": "tiny_depth_pose_checkpoint_camera_center",
        },
    )


def predict_tiny_depth_pose_frame_packets(
    checkpoint: TinyDepthPoseCheckpoint,
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "tiny-depth-pose-frame-packets",
) -> TinyDepthPosePrediction:
    """Convert ordered `FramePacket` records to `StudentClipInput` and run inference."""

    return predict_tiny_depth_pose_student_clip(
        checkpoint,
        student_clip_from_frame_packets(frames, batch_id=batch_id),
    )


def depth_observations_from_tiny_prediction(
    prediction: TinyDepthPosePrediction,
    *,
    rgb_u8_by_frame_id: Mapping[int, npt.NDArray[np.uint8]] | None = None,
    timestamp_ns_by_frame_id: Mapping[int, int] | None = None,
    camera_source: str = "input_intrinsics",
    metric_scale_source: str = "rgb_prior",
) -> tuple[DepthObservation, ...]:
    """Convert tiny-model predictions to validated mapper `DepthObservation` records."""

    if not isinstance(prediction, TinyDepthPosePrediction):
        raise ValueError("prediction: must be a TinyDepthPosePrediction")
    if prediction.depth_m.shape[:2] != (1, len(prediction.frame_ids)):
        raise ValueError("prediction: DepthObservation conversion expects B=1")
    observations: list[DepthObservation] = []
    for frame_index, frame_id in enumerate(prediction.frame_ids):
        depth = prediction.depth_m[0, frame_index].astype(np.float32, copy=True)
        sigma = prediction.depth_sigma_m[0, frame_index].astype(np.float32, copy=True)
        confidence = prediction.confidence[0, frame_index].astype(np.float32, copy=True)
        mean_confidence = float(np.mean(confidence.astype(np.float64, copy=False)))
        T_world_camera = prediction.T_world_camera[0, frame_index].astype(np.float32, copy=True)
        intrinsics = prediction.intrinsics[0, frame_index].astype(np.float32, copy=True)
        height, width = depth.shape
        timestamp_ns = (
            int(timestamp_ns_by_frame_id[frame_id])
            if timestamp_ns_by_frame_id is not None and frame_id in timestamp_ns_by_frame_id
            else 0
        )
        camera = CameraModel(
            width=width,
            height=height,
            K=intrinsics,
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source=camera_source,
        )
        pose = PoseEstimate(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=T_world_camera[:3, 3].astype(np.float32, copy=True),
            covariance_6x6=_pose_covariance(mean_confidence, sigma),
            confidence=mean_confidence,
            tracking_state="OK" if mean_confidence >= 0.25 else "LOW_CONFIDENCE",
            scale_source=metric_scale_source,
            diagnostics={
                "source": TINY_DEPTH_POSE_OBSERVATION_SOURCE,
                "coordinate_frame": prediction.coordinate_frame,
                "metric_scale_source": metric_scale_source,
                "truth_boundary": dict(prediction.truth_boundary),
                "checkpoint_path": prediction.metadata["checkpoint_path"],
                "checkpoint_step": prediction.metadata["checkpoint_step"],
                "pose_rotation_source": prediction.metadata["pose_rotation_source"],
                "pose_translation_source": prediction.metadata["pose_translation_source"],
                "depth_uncertainty_source": "tiny_depth_pose_checkpoint_depth_sigma_m",
                "confidence_mean": mean_confidence,
            },
        )
        observations.append(
            DepthObservation(
                frame_id=frame_id,
                camera=camera,
                pose=pose,
                depth_m=depth,
                depth_sigma_m=sigma,
                confidence=confidence,
                static_mask=np.ones(depth.shape, dtype=np.bool_),
                rgb_u8=(
                    rgb_u8_by_frame_id[frame_id].copy()
                    if rgb_u8_by_frame_id is not None and frame_id in rgb_u8_by_frame_id
                    else None
                ),
                source=TINY_DEPTH_POSE_OBSERVATION_SOURCE,
            )
        )
    return tuple(observations)


def predict_depth_observations_from_frame_packets(
    checkpoint: TinyDepthPoseCheckpoint,
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "tiny-depth-pose-frame-packets",
) -> tuple[DepthObservation, ...]:
    """Run inference on ordered `FramePacket` records and emit mapper observations."""

    packets = tuple(frames)
    prediction = predict_tiny_depth_pose_frame_packets(
        checkpoint,
        packets,
        batch_id=batch_id,
    )
    return depth_observations_from_tiny_prediction(
        prediction,
        rgb_u8_by_frame_id={packet.frame_id: packet.rgb_u8 for packet in packets},
        timestamp_ns_by_frame_id={packet.frame_id: packet.timestamp_ns for packet in packets},
    )


def _clip_images_float32(clip: StudentClipInput) -> npt.NDArray[np.float32]:
    images = np.asarray(clip.images_rgb)
    if images.dtype == np.uint8:
        return cast(npt.NDArray[np.float32], images.astype(np.float32) / np.float32(255.0))
    values = images.astype(np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("images_rgb: floating-point values must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("images_rgb: TinyDepthPoseNet expects float inputs in [0, 1]")
    return values


def _bthw_output(
    output: Mapping[str, Any],
    key: str,
    batch_size: int,
    frame_count: int,
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    value = _required_output(output, key).detach().cpu().numpy().astype(np.float32, copy=False)
    if value.shape != (batch_size * frame_count, 1, height, width):
        raise ValueError(f"{key}: expected model output shape B*T,1,H,W")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key}: model output must contain finite values")
    if key in {"depth_m", "depth_sigma_m"} and np.any(value < 0.0):
        raise ValueError(f"{key}: model output must be non-negative")
    return cast(npt.NDArray[np.float32], value.reshape(batch_size, frame_count, height, width))


def _camera_center_output(
    output: Mapping[str, Any],
    batch_size: int,
    frame_count: int,
) -> npt.NDArray[np.float32]:
    value = _required_output(output, "camera_center_world_m").detach().cpu().numpy()
    centers = np.asarray(value, dtype=np.float32)
    if centers.shape != (batch_size * frame_count, 3):
        raise ValueError("camera_center_world_m: expected model output shape B*T,3")
    if not np.all(np.isfinite(centers)):
        raise ValueError("camera_center_world_m: model output must contain finite values")
    return centers.reshape(batch_size, frame_count, 3)


def _required_output(output: Mapping[str, Any], key: str) -> Any:
    if key not in output:
        raise ValueError(f"model output missing required key {key}")
    return output[key]


def _prediction_transforms_from_centers(
    clip: StudentClipInput,
    centers: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    batch_size, frame_count, _dims = centers.shape
    transforms = np.broadcast_to(
        np.eye(4, dtype=np.float32),
        (batch_size, frame_count, 4, 4),
    ).copy()
    if clip.T_world_camera_prior is not None:
        transforms[:, :, :3, :3] = np.asarray(
            clip.T_world_camera_prior[:, :, :3, :3],
            dtype=np.float32,
        )
    transforms[:, :, :3, 3] = centers
    return transforms


def _pose_covariance(
    mean_confidence: float,
    sigma_m: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    sigma_values = sigma_m.astype(np.float64, copy=False)
    depth_variance = float(np.mean(np.square(sigma_values)))
    confidence_penalty = 1.0 + max(0.0, 1.0 - mean_confidence)
    translation_variance = max(depth_variance * confidence_penalty, 1e-8)
    rotation_variance = max(translation_variance * 0.01, 1e-8)
    diagonal = np.array(
        [
            translation_variance,
            translation_variance,
            translation_variance,
            rotation_variance,
            rotation_variance,
            rotation_variance,
        ],
        dtype=np.float32,
    )
    return cast(npt.NDArray[np.float32], np.diag(diagonal).astype(np.float32))


def _dict_payload_field(
    path: Path,
    payload: Mapping[str, Any],
    field_name: str,
) -> dict[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing {field_name}")
    return dict(cast(dict[str, object], value))


def _validated_truth_boundary(path: Path, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing truth_boundary")
    truth_boundary = dict(cast(dict[str, object], value))
    for flag_name, expected in _REQUIRED_TRUTH_BOUNDARY.items():
        observed = truth_boundary.get(flag_name)
        if not isinstance(observed, bool) or observed is not expected:
            raise ValueError(f"{path}: truth_boundary.{flag_name} must be {str(expected).lower()}")
    return truth_boundary


def _hidden_channels(path: Path, model_config: Mapping[str, object]) -> int:
    value = model_config.get("hidden_channels", 24)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: model_config.hidden_channels must be a positive integer")
    return value


__all__ = [
    "TINY_DEPTH_POSE_CHECKPOINT_FORMAT",
    "TINY_DEPTH_POSE_OBSERVATION_SOURCE",
    "TinyDepthPoseCheckpoint",
    "TinyDepthPosePrediction",
    "depth_observations_from_tiny_prediction",
    "load_tiny_depth_pose_checkpoint",
    "predict_depth_observations_from_frame_packets",
    "predict_tiny_depth_pose_frame_packets",
    "predict_tiny_depth_pose_student_clip",
]
