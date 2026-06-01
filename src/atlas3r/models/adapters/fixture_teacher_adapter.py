"""Dependency-free synthetic fixture teacher adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, FramePrediction, PoseEstimate, ScaleSource
from atlas3r.api.validation import validate_confidence_array, validate_same_shape
from atlas3r.camera.pinhole import unproject_depth_map
from atlas3r.io.session import LoadedSession, load_depth_npz, validate_session
from atlas3r.models.adapters.contracts import (
    AdapterAvailability,
    AdapterCapabilities,
    AdapterStatus,
    FrameBatch,
    TeacherPrediction,
)
from atlas3r.pose.transforms import transform_points

ADAPTER_NAME = "fixture-cube-room"
ADAPTER_DISPLAY_NAME = "Synthetic Cube-Room Fixture Teacher"
INSTALL_HINT = "No install required; only Phase 0B synthetic cube-room sessions are supported."
COORDINATE_FRAME = "synthetic_world"
_SYNTHETIC_SESSION_TYPE = "synthetic_cube_room"
_DEPTH_RE = re.compile(r"^frame_(\d+)\.npz$")
_REQUIRED_DEPTH_KEYS = (
    "depth_m",
    "depth_sigma_m",
    "confidence",
    "object_mask",
    "object_id",
)

CAPABILITIES = AdapterCapabilities(
    predicts_camera=True,
    predicts_pose=True,
    predicts_depth=True,
    predicts_normals=True,
    predicts_points=True,
    predicts_dense_matches=False,
    predicts_objects=True,
    supports_batch=True,
    supports_streaming=False,
    notes=(
        "Deterministic fixture adapter for Phase 0B synthetic cube-room sessions only.",
        "No external model, neural inference, downloaded weights, or measured real-world geometry.",
    ),
)


def get_adapter_status() -> AdapterStatus:
    return AdapterStatus(
        name=ADAPTER_NAME,
        display_name=ADAPTER_DISPLAY_NAME,
        availability=AdapterAvailability.AVAILABLE.value,
        capabilities=CAPABILITIES,
        install_hint=INSTALL_HINT,
        reason="Synthetic fixture/test adapter only; not an external teacher model.",
    )


class FixtureCubeRoomTeacherAdapter:
    """Reconstruct TeacherPrediction records from synthetic cube-room sidecars."""

    name = ADAPTER_NAME

    @property
    def status(self) -> AdapterStatus:
        return get_adapter_status()

    def predict(self, frames: FrameBatch) -> TeacherPrediction:
        session_path = frames.metadata.get("session_path")
        if not isinstance(session_path, str | Path):
            raise ValueError(
                "FrameBatch.metadata['session_path'] must point to a Phase 0B "
                "synthetic cube-room `.atlas3r` session"
            )
        prediction = self.predict_from_session(session_path)
        expected_frame_ids = tuple(frame.frame_id for frame in frames.frames)
        actual_frame_ids = tuple(
            frame_prediction.pose.frame_id for frame_prediction in prediction.frame_predictions
        )
        if expected_frame_ids != actual_frame_ids:
            raise ValueError(
                "frames: frame_id values must match the synthetic session depth sidecars"
            )
        return prediction

    def predict_from_session(self, session: str | Path | LoadedSession) -> TeacherPrediction:
        loaded_session = (
            validate_session(session) if not isinstance(session, LoadedSession) else session
        )
        _validate_fixture_session(loaded_session)
        frame_predictions = tuple(
            _frame_prediction_from_sidecars(
                pose=pose,
                camera=camera,
                depth_path=depth_path,
            )
            for pose, camera, depth_path in zip(
                loaded_session.poses,
                loaded_session.cameras,
                loaded_session.depth_files,
                strict=True,
            )
        )
        frame_ids = [prediction.pose.frame_id for prediction in frame_predictions]
        return TeacherPrediction(
            adapter_name=ADAPTER_NAME,
            frame_predictions=frame_predictions,
            capabilities=CAPABILITIES,
            metadata={
                "coordinate_frame": COORDINATE_FRAME,
                "fixture": True,
                "fixture_session_type": _SYNTHETIC_SESSION_TYPE,
                "source": "phase_0b_synthetic_cube_room_sidecars",
                "source_session": str(loaded_session.root),
                "source_frame_ids": frame_ids,
                "scale_source": ScaleSource.KNOWN_ANCHOR.value,
                "geometry_truth_boundary": (
                    "Synthetic analytic fixture only; this is not measured real-world geometry "
                    "and not neural model output."
                ),
            },
        )


def _validate_fixture_session(session: LoadedSession) -> None:
    metadata_path = session.root / "metadata.json"
    if session.metadata.get("session_type") != _SYNTHETIC_SESSION_TYPE:
        raise ValueError(
            f"{metadata_path}: fixture-cube-room requires session_type {_SYNTHETIC_SESSION_TYPE!r}"
        )
    if len(session.poses) != len(session.cameras):
        raise ValueError(
            f"{session.root}: pose count {len(session.poses)} does not match "
            f"camera count {len(session.cameras)}"
        )
    if len(session.depth_files) != len(session.poses):
        raise ValueError(f"{session.root / 'depth'}: expected one depth sidecar per pose frame")
    pose_frame_ids = [pose.frame_id for pose in session.poses]
    depth_frame_ids = [_depth_frame_id(path) for path in session.depth_files]
    if depth_frame_ids != pose_frame_ids:
        raise ValueError(
            f"{session.root / 'depth'}: depth frame IDs {depth_frame_ids} do not "
            f"match pose frame IDs {pose_frame_ids}"
        )


def _frame_prediction_from_sidecars(
    *,
    pose: PoseEstimate,
    camera: CameraModel,
    depth_path: Path,
) -> FramePrediction:
    depth_record = load_depth_npz(depth_path)
    _validate_depth_keys(depth_record, depth_path)
    depth_m = _image_array(depth_record["depth_m"], "depth_m", depth_path, camera)
    depth_sigma_m = _image_array(depth_record["depth_sigma_m"], "depth_sigma_m", depth_path, camera)
    if np.any(depth_sigma_m < 0.0):
        raise ValueError(f"{depth_path}: depth_sigma_m must be non-negative")
    confidence = _image_array(depth_record["confidence"], "confidence", depth_path, camera)
    try:
        validate_confidence_array("confidence", confidence)
    except ValueError as exc:
        raise ValueError(f"{depth_path}: malformed confidence: {exc}") from exc
    object_mask = _bool_image(depth_record["object_mask"], "object_mask", depth_path, camera)
    object_id = _image_array(depth_record["object_id"], "object_id", depth_path, camera)
    object_present = object_id > 0
    if not np.array_equal(object_mask, object_present):
        raise ValueError(f"{depth_path}: object_mask must match object_id > 0")
    static_mask = np.ones_like(object_mask, dtype=np.bool_)
    object_mask_logits = object_present.astype(np.float32)[..., None]

    try:
        points_camera = unproject_depth_map(depth_m, camera.K)
    except ValueError as exc:
        raise ValueError(f"{depth_path}: malformed depth_m: {exc}") from exc
    point_world = transform_points(
        pose.T_world_camera,
        points_camera.reshape(-1, 3),
    ).reshape(camera.height, camera.width, 3)
    normal_camera = _estimate_normals_camera(points_camera)

    return FramePrediction(
        pose=pose,
        camera=camera,
        depth_m=depth_m.astype(np.float32, copy=False),
        depth_sigma_m=depth_sigma_m.astype(np.float32, copy=False),
        normal_camera=normal_camera,
        point_world=point_world.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        static_mask=static_mask,
        object_embeddings=None,
        object_mask_logits=object_mask_logits,
        dense_matches=None,
    )


def _validate_depth_keys(
    depth_record: dict[str, npt.NDArray[Any]],
    depth_path: Path,
) -> None:
    missing = [key for key in _REQUIRED_DEPTH_KEYS if key not in depth_record]
    if missing:
        raise ValueError(f"{depth_path}: missing depth sidecar fields {missing}")


def _image_array(
    value: npt.NDArray[Any],
    field_name: str,
    depth_path: Path,
    camera: CameraModel,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    try:
        return validate_same_shape(field_name, array, (camera.height, camera.width))
    except ValueError as exc:
        raise ValueError(f"{depth_path}: malformed {field_name}: {exc}") from exc


def _bool_image(
    value: npt.NDArray[Any],
    field_name: str,
    depth_path: Path,
    camera: CameraModel,
) -> npt.NDArray[np.bool_]:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        array = array.astype(np.bool_)
    try:
        return validate_same_shape(field_name, array, (camera.height, camera.width))
    except ValueError as exc:
        raise ValueError(f"{depth_path}: malformed {field_name}: {exc}") from exc


def _estimate_normals_camera(
    points_camera: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    height, width, _channels = points_camera.shape
    if height < 2 or width < 2:
        normals = np.zeros((height, width, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        return normals

    points = points_camera.astype(np.float64, copy=False)
    dx = np.empty_like(points)
    dy = np.empty_like(points)
    dx[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
    dx[:, 0, :] = points[:, 1, :] - points[:, 0, :]
    dx[:, -1, :] = points[:, -1, :] - points[:, -2, :]
    dy[1:-1, :, :] = points[2:, :, :] - points[:-2, :, :]
    dy[0, :, :] = points[1, :, :] - points[0, :, :]
    dy[-1, :, :] = points[-1, :, :] - points[-2, :, :]
    normals = np.cross(dx, dy)
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    valid = norms > 1e-12
    normals = np.divide(normals, np.where(valid, norms, 1.0))
    normals[~valid[..., 0]] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normals.astype(np.float32)


def _depth_frame_id(path: Path) -> int:
    match = _DEPTH_RE.match(path.name)
    if match is None:
        raise ValueError(f"{path}: expected depth file name frame_<id>.npz")
    return int(match.group(1))


__all__ = [
    "ADAPTER_NAME",
    "FixtureCubeRoomTeacherAdapter",
    "get_adapter_status",
]
