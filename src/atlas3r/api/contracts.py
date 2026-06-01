"""Public NumPy data contracts for Atlas3R."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import (
    validate_array_shape,
    validate_choice,
    validate_confidence,
    validate_confidence_array,
    validate_covariance_6x6,
    validate_faces,
    validate_finite_numeric_array,
    validate_image_hw,
    validate_int_sequence,
    validate_intrinsics,
    validate_mapping,
    validate_matrix_nx2,
    validate_matrix_nx3,
    validate_non_negative_int,
    validate_non_negative_scalar,
    validate_nonempty_str,
    validate_positive_int,
    validate_positive_scalar,
    validate_rotation_matrix,
    validate_same_shape,
    validate_str_sequence,
    validate_transform,
    validate_vector,
)

Array = npt.NDArray[Any]


class DistortionModel(str, Enum):
    NONE = "none"
    BROWN_CONRADY = "brown_conrady"
    FISHEYE = "fisheye"


class TrackingState(str, Enum):
    OK = "OK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    RELOCALIZING = "RELOCALIZING"
    LOST = "LOST"
    NEW_SUBMAP = "NEW_SUBMAP"


class ScaleSource(str, Enum):
    RGB_PRIOR = "rgb_prior"
    CALIBRATED_RGB = "calibrated_rgb"
    KNOWN_ANCHOR = "known_anchor"
    EXTERNAL_POSE = "external_pose"


class SurfaceSource(IntEnum):
    OBSERVED_SURFACE = 0
    SINGLE_VIEW_PRIOR = 1
    COMPLETED_SURFACE = 2
    DYNAMIC_SURFACE = 3
    LOW_CONFIDENCE = 4


DISTORTION_MODEL_VALUES = {item.value for item in DistortionModel}
TRACKING_STATE_VALUES = {item.value for item in TrackingState}
SCALE_SOURCE_VALUES = {item.value for item in ScaleSource}
SURFACE_SOURCE_VALUES = {int(item.value) for item in SurfaceSource}


def _set(instance: object, field_name: str, value: object) -> None:
    object.__setattr__(instance, field_name, value)


def _require_type(field_name: str, value: object, expected_type: type[object]) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name}: must be a {expected_type.__name__}")


def _validate_label_candidates(candidates: list[tuple[str, float]]) -> None:
    if not isinstance(candidates, list):
        raise ValueError("label_candidates: must be a list")
    for index, (label, confidence) in enumerate(candidates):
        validate_nonempty_str(f"label_candidates[{index}][0]", label)
        validate_confidence(f"label_candidates[{index}][1]", confidence)


@dataclass(frozen=True)
class CameraDistortion:
    model: str
    params: npt.NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        _set(self, "model", validate_choice("model", self.model, DISTORTION_MODEL_VALUES))
        if self.params is not None:
            params = validate_finite_numeric_array("params", self.params)
            if params.ndim != 1:
                raise ValueError("params: must be a 1D array")
            _set(self, "params", params)


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ns: int
    rgb_u8: npt.NDArray[np.uint8]
    rgb_model: Array
    K_original: npt.NDArray[np.float32] | None
    K_model: npt.NDArray[np.float32]
    distortion: CameraDistortion | None
    resize_transform: npt.NDArray[np.float32]
    camera_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        validate_non_negative_int("frame_id", self.frame_id)
        validate_non_negative_int("timestamp_ns", self.timestamp_ns)
        rgb_u8 = validate_finite_numeric_array("rgb_u8", self.rgb_u8)
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError("rgb_u8: must be a uint8 HxWx3 array")
        rgb_model = validate_finite_numeric_array("rgb_model", self.rgb_model)
        if rgb_model.ndim != 3 or rgb_model.shape[0] != 3:
            raise ValueError("rgb_model: must have shape 3xHmxWm")
        if self.K_original is not None:
            _set(self, "K_original", validate_intrinsics("K_original", self.K_original))
        _set(self, "K_model", validate_intrinsics("K_model", self.K_model))
        if self.distortion is not None:
            _require_type("distortion", self.distortion, CameraDistortion)
        _set(
            self,
            "resize_transform",
            validate_array_shape("resize_transform", self.resize_transform, (3, 3)),
        )
        validate_mapping("camera_metadata", self.camera_metadata)
        _set(self, "rgb_u8", rgb_u8)
        _set(self, "rgb_model", rgb_model)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    K: npt.NDArray[np.float32]
    distortion_model: str
    distortion_params: npt.NDArray[np.float32] | None
    rolling_shutter_row_time_s: float | None
    confidence: float
    source: str

    def __post_init__(self) -> None:
        validate_positive_int("width", self.width)
        validate_positive_int("height", self.height)
        _set(self, "K", validate_intrinsics("K", self.K))
        _set(
            self,
            "distortion_model",
            validate_choice("distortion_model", self.distortion_model, DISTORTION_MODEL_VALUES),
        )
        if self.distortion_params is not None:
            params = validate_finite_numeric_array("distortion_params", self.distortion_params)
            if params.ndim != 1:
                raise ValueError("distortion_params: must be a 1D array")
            _set(self, "distortion_params", params)
        if self.rolling_shutter_row_time_s is not None:
            validate_non_negative_scalar(
                "rolling_shutter_row_time_s", self.rolling_shutter_row_time_s
            )
        validate_confidence("confidence", self.confidence)
        validate_nonempty_str("source", self.source)


@dataclass(frozen=True)
class PoseEstimate:
    frame_id: int
    timestamp_ns: int
    T_world_camera: npt.NDArray[np.float32]
    q_world_camera_xyzw: npt.NDArray[np.float32]
    camera_center_world_m: npt.NDArray[np.float32]
    covariance_6x6: npt.NDArray[np.float32] | None
    confidence: float
    tracking_state: str
    scale_source: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        validate_non_negative_int("frame_id", self.frame_id)
        validate_non_negative_int("timestamp_ns", self.timestamp_ns)
        T_world_camera = validate_transform("T_world_camera", self.T_world_camera)
        q_world_camera_xyzw = validate_vector("q_world_camera_xyzw", self.q_world_camera_xyzw, 4)
        q_norm = float(np.linalg.norm(q_world_camera_xyzw.astype(np.float64, copy=False)))
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            raise ValueError("q_world_camera_xyzw: must be unit length")
        camera_center_world_m = validate_vector(
            "camera_center_world_m", self.camera_center_world_m, 3
        )
        if not np.allclose(camera_center_world_m, T_world_camera[:3, 3], atol=1e-4):
            raise ValueError("camera_center_world_m: must match T_world_camera[:3, 3]")
        _set(self, "T_world_camera", T_world_camera)
        _set(self, "q_world_camera_xyzw", q_world_camera_xyzw)
        _set(self, "camera_center_world_m", camera_center_world_m)
        _set(self, "covariance_6x6", validate_covariance_6x6("covariance_6x6", self.covariance_6x6))
        validate_confidence("confidence", self.confidence)
        _set(
            self,
            "tracking_state",
            validate_choice("tracking_state", self.tracking_state, TRACKING_STATE_VALUES),
        )
        _set(
            self,
            "scale_source",
            validate_choice("scale_source", self.scale_source, SCALE_SOURCE_VALUES),
        )
        validate_mapping("diagnostics", self.diagnostics)


@dataclass(frozen=True)
class DenseMatchSet:
    source_frame_id: int
    target_frame_id: int
    source_pixels_uv: npt.NDArray[np.float32]
    target_pixels_uv: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        validate_non_negative_int("source_frame_id", self.source_frame_id)
        validate_non_negative_int("target_frame_id", self.target_frame_id)
        source_pixels_uv = validate_matrix_nx2("source_pixels_uv", self.source_pixels_uv)
        target_pixels_uv = validate_matrix_nx2("target_pixels_uv", self.target_pixels_uv)
        confidence = validate_confidence_array("confidence", self.confidence)
        if target_pixels_uv.shape != source_pixels_uv.shape:
            raise ValueError("target_pixels_uv: must match source_pixels_uv shape")
        if confidence.shape != (source_pixels_uv.shape[0],):
            raise ValueError("confidence: must have length N")
        _set(self, "source_pixels_uv", source_pixels_uv)
        _set(self, "target_pixels_uv", target_pixels_uv)
        _set(self, "confidence", confidence)


@dataclass
class FramePrediction:
    pose: PoseEstimate
    camera: CameraModel
    depth_m: Array
    depth_sigma_m: Array
    normal_camera: Array
    point_world: Array
    confidence: Array
    static_mask: Array
    object_embeddings: Array | None
    object_mask_logits: Array | None
    dense_matches: DenseMatchSet | None

    def __post_init__(self) -> None:
        _require_type("pose", self.pose, PoseEstimate)
        _require_type("camera", self.camera, CameraModel)
        depth_m = validate_image_hw("depth_m", self.depth_m)
        if np.any(depth_m < 0.0):
            raise ValueError("depth_m: must be non-negative")
        expected_hw = depth_m.shape
        if expected_hw != (self.camera.height, self.camera.width):
            raise ValueError("depth_m: must match camera height and width")
        depth_sigma_m = validate_same_shape("depth_sigma_m", self.depth_sigma_m, expected_hw)
        if np.any(depth_sigma_m < 0.0):
            raise ValueError("depth_sigma_m: must be non-negative")
        normal_camera = validate_same_shape("normal_camera", self.normal_camera, (*expected_hw, 3))
        point_world = validate_same_shape("point_world", self.point_world, (*expected_hw, 3))
        confidence = validate_same_shape("confidence", self.confidence, expected_hw)
        validate_confidence_array("confidence", confidence)
        static_mask = validate_same_shape("static_mask", self.static_mask, expected_hw)
        if static_mask.dtype != np.bool_:
            validate_confidence_array("static_mask", static_mask)
        if self.object_embeddings is not None:
            object_embeddings = validate_finite_numeric_array(
                "object_embeddings", self.object_embeddings
            )
            if object_embeddings.ndim < 2 or object_embeddings.shape[:2] != expected_hw:
                raise ValueError("object_embeddings: first dimensions must be HxW")
        if self.object_mask_logits is not None:
            object_mask_logits = validate_finite_numeric_array(
                "object_mask_logits", self.object_mask_logits
            )
            if object_mask_logits.ndim < 2 or object_mask_logits.shape[:2] != expected_hw:
                raise ValueError("object_mask_logits: first dimensions must be HxW")
        if self.dense_matches is not None:
            _require_type("dense_matches", self.dense_matches, DenseMatchSet)
        self.depth_m = depth_m
        self.depth_sigma_m = depth_sigma_m
        self.normal_camera = normal_camera
        self.point_world = point_world
        self.confidence = confidence
        self.static_mask = static_mask


@dataclass
class ObjectInstance:
    object_id: int
    label_candidates: list[tuple[str, float]]
    T_world_object: npt.NDArray[np.float32]
    oriented_bbox_center_m: npt.NDArray[np.float32]
    oriented_bbox_axes: npt.NDArray[np.float32]
    oriented_bbox_extents_m: npt.NDArray[np.float32]
    mesh_chunk_ids: list[str]
    is_dynamic: bool
    observed_coverage_ratio: float
    confidence: float
    uncertainty_m: float
    first_seen_frame_id: int
    last_seen_frame_id: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        validate_non_negative_int("object_id", self.object_id)
        _validate_label_candidates(self.label_candidates)
        self.T_world_object = validate_transform("T_world_object", self.T_world_object)
        self.oriented_bbox_center_m = validate_vector(
            "oriented_bbox_center_m", self.oriented_bbox_center_m, 3
        )
        self.oriented_bbox_axes = validate_rotation_matrix(
            "oriented_bbox_axes", self.oriented_bbox_axes
        )
        extents = validate_vector("oriented_bbox_extents_m", self.oriented_bbox_extents_m, 3)
        if np.any(extents <= 0.0):
            raise ValueError("oriented_bbox_extents_m: must contain positive extents")
        self.oriented_bbox_extents_m = extents
        validate_str_sequence("mesh_chunk_ids", self.mesh_chunk_ids)
        if not isinstance(self.is_dynamic, bool):
            raise ValueError("is_dynamic: must be a bool")
        validate_confidence("observed_coverage_ratio", self.observed_coverage_ratio)
        validate_confidence("confidence", self.confidence)
        validate_non_negative_scalar("uncertainty_m", self.uncertainty_m)
        validate_non_negative_int("first_seen_frame_id", self.first_seen_frame_id)
        validate_non_negative_int("last_seen_frame_id", self.last_seen_frame_id)
        if self.first_seen_frame_id > self.last_seen_frame_id:
            raise ValueError("first_seen_frame_id: must be <= last_seen_frame_id")
        validate_mapping("metadata", self.metadata)


@dataclass
class MeshChunk:
    chunk_id: str
    version: int
    T_world_chunk: npt.NDArray[np.float32]
    vertices_m: npt.NDArray[np.float32]
    faces: npt.NDArray[np.int32]
    normals: npt.NDArray[np.float32] | None
    colors: npt.NDArray[np.uint8] | None
    uvs: npt.NDArray[np.float32] | None
    object_id_per_face: npt.NDArray[np.int32] | None
    surface_source_per_face: npt.NDArray[np.int8] | None
    voxel_size_m: float
    mean_uncertainty_m: float
    p95_uncertainty_m: float
    source_frame_ids: list[int]
    scale_source: str
    flags: list[str]

    def __post_init__(self) -> None:
        validate_nonempty_str("chunk_id", self.chunk_id)
        validate_positive_int("version", self.version)
        self.T_world_chunk = validate_transform("T_world_chunk", self.T_world_chunk)
        self.vertices_m = validate_matrix_nx3("vertices_m", self.vertices_m)
        self.faces = validate_faces("faces", self.faces, self.vertices_m.shape[0])
        face_count = self.faces.shape[0]
        if self.normals is not None:
            self.normals = validate_same_shape("normals", self.normals, self.vertices_m.shape)
        if self.colors is not None:
            colors = np.asarray(self.colors)
            if (
                colors.dtype != np.uint8
                or colors.ndim != 2
                or colors.shape[0] != self.vertices_m.shape[0]
            ):
                raise ValueError("colors: must be a uint8 Nx3 or Nx4 array")
            if colors.shape[1] not in (3, 4):
                raise ValueError("colors: must be a uint8 Nx3 or Nx4 array")
        if self.uvs is not None:
            uvs = validate_finite_numeric_array("uvs", self.uvs)
            if uvs.ndim != 2 or uvs.shape != (self.vertices_m.shape[0], 2):
                raise ValueError("uvs: must have shape Nx2")
            self.uvs = uvs
        if self.object_id_per_face is not None:
            object_ids = np.asarray(self.object_id_per_face)
            if object_ids.shape != (face_count,) or not np.issubdtype(object_ids.dtype, np.integer):
                raise ValueError("object_id_per_face: must be an integer array of length M")
        if self.surface_source_per_face is not None:
            surface_source = np.asarray(self.surface_source_per_face)
            if surface_source.shape != (face_count,) or not np.issubdtype(
                surface_source.dtype, np.integer
            ):
                raise ValueError("surface_source_per_face: must be an integer array of length M")
            invalid = set(int(value) for value in surface_source.tolist()) - SURFACE_SOURCE_VALUES
            if invalid:
                raise ValueError("surface_source_per_face: contains unknown surface source values")
        validate_positive_scalar("voxel_size_m", self.voxel_size_m)
        validate_non_negative_scalar("mean_uncertainty_m", self.mean_uncertainty_m)
        validate_non_negative_scalar("p95_uncertainty_m", self.p95_uncertainty_m)
        validate_int_sequence("source_frame_ids", self.source_frame_ids, non_empty=True)
        self.scale_source = validate_choice("scale_source", self.scale_source, SCALE_SOURCE_VALUES)
        validate_str_sequence("flags", self.flags)


@dataclass
class WorldMap:
    map_id: str
    world_frame_name: str
    created_at_ns: int
    mesh_chunks: dict[str, MeshChunk]
    objects: dict[int, ObjectInstance]
    keyframes: dict[int, PoseEstimate]
    scale_source: str
    global_confidence: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        validate_nonempty_str("map_id", self.map_id)
        validate_nonempty_str("world_frame_name", self.world_frame_name)
        validate_non_negative_int("created_at_ns", self.created_at_ns)
        for chunk_id, chunk in self.mesh_chunks.items():
            _require_type("mesh_chunks", chunk, MeshChunk)
            if chunk_id != chunk.chunk_id:
                raise ValueError("mesh_chunks: keys must match chunk_id")
        for object_id, instance in self.objects.items():
            _require_type("objects", instance, ObjectInstance)
            if object_id != instance.object_id:
                raise ValueError("objects: keys must match object_id")
        for frame_id, pose in self.keyframes.items():
            _require_type("keyframes", pose, PoseEstimate)
            if frame_id != pose.frame_id:
                raise ValueError("keyframes: keys must match frame_id")
        self.scale_source = validate_choice("scale_source", self.scale_source, SCALE_SOURCE_VALUES)
        validate_confidence("global_confidence", self.global_confidence)
        validate_mapping("metadata", self.metadata)


__all__ = [
    "CameraDistortion",
    "CameraModel",
    "DenseMatchSet",
    "DistortionModel",
    "FramePacket",
    "FramePrediction",
    "MeshChunk",
    "ObjectInstance",
    "PoseEstimate",
    "ScaleSource",
    "SurfaceSource",
    "TrackingState",
    "WorldMap",
]
