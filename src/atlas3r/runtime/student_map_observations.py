"""Observation and reference-frame helpers for Phase 5E student runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import (
    compose_transforms,
    invert_transform,
    quaternion_xyzw_from_rotation_matrix,
)
from atlas3r.runtime.student_map_reports import DIAGNOSTIC_TRUTH_FLAGS, TeacherReferenceFrame
from atlas3r.runtime.student_stream import StreamWindow
from atlas3r.teachers.signals import (
    load_teacher_signal_manifest,
    read_teacher_signal_payload_from_entry,
    teacher_signal_manifest_path_from_input,
    validate_payload_matches_signal_entry,
    validate_teacher_signal_payload,
)


@dataclass
class StudentOdometryState:
    """Roll out learned relative SE(3) transforms over a chronological stream."""

    last_stream_index: int | None = None
    last_T_world_camera: npt.NDArray[np.float32] | None = None

    def pose_for_window(
        self,
        *,
        prediction: Mapping[str, Any],
        window: StreamWindow,
    ) -> tuple[npt.NDArray[np.float32], str]:
        frame = window.output
        if frame.stream_index == 0:
            self.last_stream_index = frame.stream_index
            self.last_T_world_camera = frame.T_world_camera.astype(np.float32, copy=True)
            return (
                self.last_T_world_camera.copy(),
                "student-odometry anchor: first frame uses source clip-cache T_world_camera",
            )
        if self.last_stream_index is None or self.last_T_world_camera is None:
            raise ValueError("student-odometry: first stream frame must initialize the anchor pose")
        expected_previous = self.last_stream_index + 1
        if frame.stream_index != expected_previous:
            raise ValueError(
                "student-odometry: stream frames must be processed in chronological order"
            )
        previous_slot = _slot_for_stream_index(window, self.last_stream_index)
        T_current_previous = _predicted_relative_transform(prediction, previous_slot)
        T_previous_current = invert_transform(T_current_previous)
        T_world_camera = compose_transforms(
            self.last_T_world_camera,
            T_previous_current,
        ).astype(np.float32)
        self.last_stream_index = frame.stream_index
        self.last_T_world_camera = T_world_camera
        return (
            T_world_camera.copy(),
            "student-odometry rollout from learned T_current_previous relative SE(3)",
        )


def load_teacher_reference_frames(
    teacher_cache: str | Path,
    *,
    frame_ids: tuple[int, ...] | None = None,
) -> dict[int, TeacherReferenceFrame]:
    """Load first-occurrence teacher reference frames by frame id."""

    wanted = None if frame_ids is None else set(frame_ids)
    manifest_path = teacher_signal_manifest_path_from_input(teacher_cache)
    manifest = load_teacher_signal_manifest(manifest_path, validate_payloads=False)
    entries = _entries(manifest, "signals")
    clip_length = _int_field(manifest, "clip_length")
    height = _int_field(manifest, "image_height")
    width = _int_field(manifest, "image_width")
    truth = cast(dict[str, object], manifest["truth_boundary"])
    references: dict[int, TeacherReferenceFrame] = {}
    for signal_index, entry in enumerate(entries):
        payload = read_teacher_signal_payload_from_entry(manifest_path.parent, entry)
        validate_teacher_signal_payload(
            payload,
            clip_length=clip_length,
            height=height,
            width=width,
        )
        validate_payload_matches_signal_entry(payload, entry, index=signal_index)
        payload_frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
        timestamps = np.asarray(payload["timestamps_s"], dtype=np.float64)
        for offset, frame_id_value in enumerate(payload_frame_ids.tolist()):
            frame_id = int(frame_id_value)
            if wanted is not None and frame_id not in wanted:
                continue
            reference = _teacher_reference_from_payload(
                manifest=manifest,
                truth=truth,
                payload=payload,
                offset=offset,
                frame_id=frame_id,
                timestamp_s=float(timestamps[offset]),
            )
            existing = references.get(frame_id)
            if existing is None:
                references[frame_id] = reference
            else:
                _validate_duplicate_reference(existing, reference)
        if wanted is not None and wanted.issubset(references):
            break
    return references


def observation_from_prediction(
    *,
    prediction: Mapping[str, Any],
    window: StreamWindow,
    pose_mode: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    odometry_state: StudentOdometryState | None = None,
) -> tuple[DepthObservation, dict[str, object]]:
    """Convert one temporal-student output slot into a `DepthObservation`."""

    output_slot = window.center_index
    frame = window.output
    depth = _prediction_array(prediction, "depth_m", output_slot)
    sigma = _prediction_array(prediction, "depth_sigma_m", output_slot)
    confidence = _prediction_array(prediction, "confidence", output_slot)
    relative_translation = (
        prediction["relative_translation_center_from_camera"][0, output_slot]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    T_world_camera, pose_note = _pose_for_mode(
        prediction=prediction,
        window=window,
        pose_mode=pose_mode,
        relative_translation_center_from_camera=relative_translation,
        odometry_state=odometry_state,
    )
    mean_sigma = float(np.mean(sigma[confidence > 0.0])) if np.any(confidence > 0.0) else 0.0
    covariance = np.diag(np.full(6, max(mean_sigma * mean_sigma, 1e-8), dtype=np.float32))
    timestamp_ns = int(round(frame.timestamp_s * 1_000_000_000.0))
    truth_boundary = cast(dict[str, object], checkpoint["truth_boundary"])
    diagnostics = {
        "source": "phase5g_stream_student_map_runtime",
        "coordinate_frame": "x_right_y_down_z_forward",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_format_name": str(checkpoint["format_name"]),
        "stream_index": frame.stream_index,
        "window": window.metadata(),
        "pose_mode": pose_mode,
        "pose_note": pose_note,
        "truth_boundary": truth_boundary,
        **DIAGNOSTIC_TRUTH_FLAGS,
    }
    pose = PoseEstimate(
        frame_id=frame.frame_id,
        timestamp_ns=timestamp_ns,
        T_world_camera=T_world_camera,
        q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
            np.float32
        ),
        camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
        covariance_6x6=covariance.astype(np.float32),
        confidence=1.0 if pose_mode == "oracle" else 0.25,
        tracking_state="OK" if pose_mode == "oracle" else "LOW_CONFIDENCE",
        scale_source="external_pose" if pose_mode == "oracle" else "rgb_prior",
        diagnostics=diagnostics,
    )
    observation = DepthObservation(
        frame_id=frame.frame_id,
        camera=CameraModel(
            width=frame.rgb_u8.shape[1],
            height=frame.rgb_u8.shape[0],
            K=frame.K.astype(np.float32, copy=True),
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="clip_cache",
        ),
        pose=pose,
        depth_m=depth,
        depth_sigma_m=sigma,
        confidence=confidence,
        static_mask=(depth > 0.0) & (confidence > 0.0),
        rgb_u8=frame.rgb_u8.copy(),
        source="phase5g_temporal_student_checkpoint",
    )
    record = {
        "format_name": "atlas3r_phase5g_observation_summary",
        "format_version": 1,
        "frame_id": frame.frame_id,
        "timestamp_s": frame.timestamp_s,
        "pose_mode": pose_mode,
        "depth_mean_m": float(np.mean(depth)),
        "depth_sigma_mean_m": float(np.mean(sigma)),
        "confidence_mean": float(np.mean(confidence)),
        "source_frame": frame.metadata(),
        "window": window.metadata(),
        "pose_diagnostic_note": pose_note,
        **DIAGNOSTIC_TRUTH_FLAGS,
    }
    return observation, record


def window_arrays(window: StreamWindow) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build Torch-ready NumPy arrays for one temporal inference window."""

    rgb = np.stack([frame.rgb_u8 for frame in window.frames], axis=0).astype(np.float32) / 255.0
    images = np.transpose(rgb, (0, 3, 1, 2))[np.newaxis, ...].copy()
    intrinsics = np.stack([frame.K for frame in window.frames], axis=0)[np.newaxis, ...].copy()
    return images.astype(np.float32, copy=False), intrinsics.astype(np.float32, copy=False)


def _prediction_array(
    prediction: Mapping[str, Any],
    key: str,
    output_slot: int,
) -> npt.NDArray[np.float32]:
    return cast(
        npt.NDArray[np.float32],
        prediction[key][0, output_slot, 0].detach().cpu().numpy().astype(np.float32),
    )


def _pose_for_mode(
    *,
    prediction: Mapping[str, Any],
    window: StreamWindow,
    pose_mode: str,
    relative_translation_center_from_camera: npt.NDArray[np.float32],
    odometry_state: StudentOdometryState | None,
) -> tuple[npt.NDArray[np.float32], str]:
    frame = window.output
    T_world_camera = frame.T_world_camera.astype(np.float32, copy=True)
    if pose_mode == "oracle":
        return T_world_camera, "oracle source clip-cache T_world_camera"
    if pose_mode == "student-odometry":
        if odometry_state is None:
            raise ValueError("student-odometry: odometry state is required")
        return odometry_state.pose_for_window(prediction=prediction, window=window)
    delta_world = T_world_camera[:3, :3] @ relative_translation_center_from_camera.astype(
        np.float32,
        copy=False,
    )
    T_world_camera[:3, 3] = T_world_camera[:3, 3] + delta_world
    return (
        T_world_camera,
        "diagnostic student-relative translation around source anchor; source rotation kept",
    )


def _slot_for_stream_index(window: StreamWindow, stream_index: int) -> int:
    matches = [
        slot
        for slot, source_index in enumerate(window.source_stream_indices)
        if source_index == stream_index
    ]
    if not matches:
        raise ValueError(f"student-odometry: previous stream index {stream_index} not in window")
    return matches[-1]


def _predicted_relative_transform(
    prediction: Mapping[str, Any],
    slot: int,
) -> npt.NDArray[np.float32]:
    if "relative_rotation_6d_center_from_camera" not in prediction:
        raise ValueError("student-odometry: checkpoint prediction has no SE(3) rotation output")
    translation = (
        prediction["relative_translation_center_from_camera"][0, slot]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    rotation_6d = (
        prediction["relative_rotation_6d_center_from_camera"][0, slot]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    T_current_previous = np.eye(4, dtype=np.float32)
    T_current_previous[:3, :3] = _rotation_matrix_from_6d(rotation_6d)
    T_current_previous[:3, 3] = translation
    return T_current_previous


def _rotation_matrix_from_6d(rotation_6d: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if rotation_6d.shape != (6,):
        raise ValueError("relative_rotation_6d_center_from_camera: expected shape 6")
    first = rotation_6d[:3].astype(np.float64, copy=False)
    second = rotation_6d[3:6].astype(np.float64, copy=False)
    norm_first = float(np.linalg.norm(first))
    if norm_first <= 1e-12:
        raise ValueError("relative_rotation_6d_center_from_camera: first basis vector is zero")
    basis_0 = first / norm_first
    second_orthogonal = second - float(np.dot(basis_0, second)) * basis_0
    norm_second = float(np.linalg.norm(second_orthogonal))
    if norm_second <= 1e-12:
        raise ValueError(
            "relative_rotation_6d_center_from_camera: second basis vector is degenerate"
        )
    basis_1 = second_orthogonal / norm_second
    basis_2 = np.cross(basis_0, basis_1)
    return cast(
        npt.NDArray[np.float32],
        np.stack([basis_0, basis_1, basis_2], axis=1).astype(np.float32),
    )


def _teacher_reference_from_payload(
    *,
    manifest: Mapping[str, object],
    truth: Mapping[str, object],
    payload: Mapping[str, Any],
    offset: int,
    frame_id: int,
    timestamp_s: float,
) -> TeacherReferenceFrame:
    return TeacherReferenceFrame(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        depth_m=np.asarray(payload["depth_m"][offset], dtype=np.float32).copy(),
        valid_mask=np.asarray(payload["valid_mask"][offset], dtype=np.bool_).copy(),
        K=np.asarray(payload["K"][offset], dtype=np.float32).copy(),
        T_world_camera=np.asarray(payload["T_world_camera"][offset], dtype=np.float32).copy(),
        confidence=np.asarray(payload["confidence"][offset], dtype=np.float32).copy(),
        depth_sigma_m=np.asarray(payload["depth_sigma_m"][offset], dtype=np.float32).copy(),
        teacher_name=str(manifest["teacher_name"]),
        measured_geometry=bool(truth.get("measured_geometry")),
    )


def _validate_duplicate_reference(
    first: TeacherReferenceFrame,
    duplicate: TeacherReferenceFrame,
) -> None:
    checks = (
        ("depth_m", first.depth_m, duplicate.depth_m),
        ("K", first.K, duplicate.K),
        ("T_world_camera", first.T_world_camera, duplicate.T_world_camera),
    )
    for field_name, left, right in checks:
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=1e-5):
            raise ValueError(f"duplicate teacher frame_id {first.frame_id}: {field_name} differs")


def _entries(manifest: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    entries: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.{key}[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], item))
    return entries


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "StudentOdometryState",
    "load_teacher_reference_frames",
    "observation_from_prediction",
    "window_arrays",
]
