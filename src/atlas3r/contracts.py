"""Binding runtime contracts for the Atlas3R architecture.

These types validate boundary data only. They do not reconstruct, segment,
optimize, map, validate, export, or invoke external model code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Protocol, runtime_checkable


class ContractValidationError(ValueError):
    """Raised when data violates an Atlas3R architecture contract."""


class MetricAcceptanceStatus(str, Enum):
    MEASURED_METRIC = "measured_metric"
    METRIC_PSEUDO_LABEL = "metric_pseudo_label"
    NON_METRIC_PSEUDO_LABEL = "non_metric_pseudo_label"
    REJECTED = "rejected"


class DepthConvention(str, Enum):
    RADIAL_RANGE = "radial_range"
    OPTICAL_Z = "optical_z"
    INVERSE_DEPTH = "inverse_depth"
    DISPARITY = "disparity"
    UNKNOWN = "unknown"


class ScaleEvidenceType(str, Enum):
    MEASURED_DEPTH = "measured_depth"
    MEASURED_POSE = "measured_pose"
    MANUAL_DISTANCE = "manual_distance"
    KNOWN_MARKER = "known_marker"
    BENCHMARK_GT = "benchmark_gt"
    OBJECT_SIZE_PRIOR = "object_size_prior"
    ARCHITECTURE_PRIOR = "architecture_prior"
    LEARNED_METRIC_DEPTH_PRIOR = "learned_metric_depth_prior"
    SCENE_LAYOUT_PRIOR = "scene_layout_prior"


class StaticDynamicLabel(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


class OccupancyChannel(str, Enum):
    FREE = "free"
    OCCUPIED_STATIC = "occupied_static"
    MOVABLE_STATIC = "movable_static"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"
    PREDICTED = "predicted"
    MEASURED = "measured"


class BackboneAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class TrackType(str, Enum):
    REFERENCE_METRIC = "reference_metric"
    PHONE_ROOM = "phone_room"
    EXTERNAL_CANDIDATE = "external_candidate"


class VideoAssetStatus(str, Enum):
    AVAILABLE = "available"
    MISSING_ASSET = "missing_asset"
    CORRUPT = "corrupt"
    UNSUPPORTED = "unsupported"
    UNCHECKED = "unchecked"


_MEASURED_EVIDENCE_TYPES = frozenset(
    {
        ScaleEvidenceType.MEASURED_DEPTH,
        ScaleEvidenceType.MEASURED_POSE,
        ScaleEvidenceType.MANUAL_DISTANCE,
        ScaleEvidenceType.KNOWN_MARKER,
        ScaleEvidenceType.BENCHMARK_GT,
    }
)


@runtime_checkable
class CameraModel(Protocol):
    """Projection support required by visibility and reprojection contracts."""

    def unproject(self, pixel_uv: object, radial_depth_m: float) -> object: ...

    def project(self, X_camera: object) -> object: ...


@runtime_checkable
class VideoGeometryBackbone(Protocol):
    """Boundary for external geometry engines such as ViPE/DA3 or MegaSaM."""

    backend_name: str

    def predict(self, video_or_keyframes: object) -> "GeometryBackbonePrediction": ...


class ExternalBackboneUnavailableError(RuntimeError):
    """Raised when an external model boundary is called without dependencies."""


def _as_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ContractValidationError(
            f"{field_name} must be one of {[item.value for item in enum_type]}"
        ) from exc


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be a bool")


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")


def _validate_positive_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive finite number")


def _validate_non_negative_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative finite number"
        )


def _validate_probability(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
        raise ContractValidationError(f"{field_name} must be a probability in [0, 1]")


def _to_plain(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _shape(value: Any) -> tuple[int, ...]:
    value = _to_plain(value)
    if not _is_sequence(value):
        return ()
    length = len(value)
    if length == 0:
        return (0,)
    child_shapes = [_shape(child) for child in value]
    first = child_shapes[0]
    if any(child != first for child in child_shapes[1:]):
        raise ContractValidationError("array-like values must not be ragged")
    return (length, *first)


def _validate_shape(value: Any, expected: tuple[int | None, ...], field_name: str) -> tuple[int, ...]:
    actual = _shape(value)
    if len(actual) != len(expected):
        raise ContractValidationError(
            f"{field_name} must have shape {expected}, got {actual}"
        )
    for actual_dim, expected_dim in zip(actual, expected):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ContractValidationError(
                f"{field_name} must have shape {expected}, got {actual}"
            )
    return actual


def _iter_scalars(value: Any) -> Iterable[Any]:
    value = _to_plain(value)
    if _is_sequence(value):
        for item in value:
            yield from _iter_scalars(item)
    else:
        yield value


def _iter_pairs(first: Any, second: Any) -> Iterable[tuple[Any, Any]]:
    first = _to_plain(first)
    second = _to_plain(second)
    if _is_sequence(first) and _is_sequence(second):
        for left, right in zip(first, second):
            yield from _iter_pairs(left, right)
    else:
        yield first, second


def _iter_triples(first: Any, second: Any, third: Any) -> Iterable[tuple[Any, Any, Any]]:
    first = _to_plain(first)
    second = _to_plain(second)
    third = _to_plain(third)
    if _is_sequence(first) and _is_sequence(second) and _is_sequence(third):
        for left, middle, right in zip(first, second, third):
            yield from _iter_triples(left, middle, right)
    else:
        yield first, second, third


def _validate_all_finite(value: Any, field_name: str) -> None:
    for scalar in _iter_scalars(value):
        if not isinstance(scalar, (int, float)) or not isfinite(scalar):
            raise ContractValidationError(f"{field_name} must contain only finite numbers")


def _validate_all_non_negative(value: Any, field_name: str) -> None:
    for scalar in _iter_scalars(value):
        _validate_non_negative_number(scalar, field_name)


def _validate_all_positive(value: Any, field_name: str) -> None:
    for scalar in _iter_scalars(value):
        _validate_positive_number(scalar, field_name)


def _validate_all_probability(value: Any, field_name: str) -> None:
    for scalar in _iter_scalars(value):
        _validate_probability(scalar, field_name)


def _validate_rays_unit(rays_camera: Any, field_name: str, tolerance: float = 1e-5) -> None:
    rays = _to_plain(rays_camera)
    for row in rays:
        for ray in row:
            if not _is_sequence(ray) or len(ray) != 3:
                raise ContractValidationError(f"{field_name} must contain 3-vectors")
            _validate_all_finite(ray, field_name)
            norm = sqrt(sum(float(component) ** 2 for component in ray))
            if abs(norm - 1.0) > tolerance:
                raise ContractValidationError(
                    f"{field_name} must contain unit camera rays"
                )


def _validate_mapping(value: Mapping[str, Any], field_name: str, *, non_empty: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be a mapping")
    if non_empty and not value:
        raise ContractValidationError(f"{field_name} must not be empty")


def _validate_sequence(value: Sequence[Any], field_name: str, *, non_empty: bool = False) -> tuple[Any, ...]:
    if not _is_sequence(value):
        raise ContractValidationError(f"{field_name} must be a sequence")
    result = tuple(value)
    if non_empty and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return result


def _validate_frame_ids(frame_ids: Sequence[int], field_name: str, *, non_empty: bool = True) -> tuple[int, ...]:
    ids = _validate_sequence(frame_ids, field_name, non_empty=non_empty)
    for frame_id in ids:
        _validate_non_negative_int(frame_id, field_name)
    return ids


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None:
        _validate_positive_int(value, field_name)


def _validate_optional_non_negative_number(value: float | None, field_name: str) -> None:
    if value is not None:
        _validate_non_negative_number(value, field_name)


def _validate_string_sequence(value: Sequence[str], field_name: str, *, non_empty: bool = False) -> tuple[str, ...]:
    items = _validate_sequence(value, field_name, non_empty=non_empty)
    for item in items:
        _validate_non_empty_string(item, field_name)
    return items


def _validate_same_shape(fields: Mapping[str, Any]) -> tuple[int, ...]:
    iterator = iter(fields.items())
    first_name, first_value = next(iterator)
    first_shape = _shape(first_value)
    for name, value in iterator:
        actual = _shape(value)
        if actual != first_shape:
            raise ContractValidationError(
                f"{name} must have shape {first_shape} to match {first_name}, got {actual}"
            )
    return first_shape


def _validate_probability_simplex(fields: Mapping[str, Any], tolerance: float = 1e-6) -> None:
    values = tuple(fields.items())
    for name, value in values:
        _validate_all_probability(value, name)
    for scalars in zip(*(_iter_scalars(value) for _, value in values)):
        total = sum(float(scalar) for scalar in scalars)
        if abs(total - 1.0) > tolerance:
            names = ", ".join(name for name, _ in values)
            raise ContractValidationError(f"{names} probabilities must sum to 1")


@dataclass(frozen=True)
class VideoAsset:
    asset_id: str
    track_type: TrackType
    source_uri_or_path: str
    expected_modalities: Sequence[str]
    status: VideoAssetStatus = VideoAssetStatus.UNCHECKED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.asset_id, "asset_id")
        track_type = _as_enum(self.track_type, TrackType, "track_type")
        object.__setattr__(self, "track_type", track_type)
        _validate_non_empty_string(self.source_uri_or_path, "source_uri_or_path")
        _validate_string_sequence(
            self.expected_modalities, "expected_modalities", non_empty=True
        )
        status = _as_enum(self.status, VideoAssetStatus, "status")
        object.__setattr__(self, "status", status)
        _validate_mapping(self.metadata, "metadata")


@dataclass(frozen=True)
class VideoInspectionReport:
    asset_id: str
    frame_count: int
    fps_or_frame_timestamps: Mapping[str, Any]
    width_px: int | None
    height_px: int | None
    duration_s: float | None
    codec_or_container_optional: str | None
    sampled_frame_ids: Sequence[int]
    blur_summary: Mapping[str, Any]
    exposure_summary: Mapping[str, Any]
    motion_summary: Mapping[str, Any]
    scene_change_summary: Mapping[str, Any]
    usable_frame_ratio: float
    hard_rejection_reasons: Sequence[str]
    soft_risk_flags: Sequence[str]
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.asset_id, "asset_id")
        _validate_non_negative_int(self.frame_count, "frame_count")
        _validate_mapping(self.fps_or_frame_timestamps, "fps_or_frame_timestamps")
        _validate_optional_positive_int(self.width_px, "width_px")
        _validate_optional_positive_int(self.height_px, "height_px")
        _validate_optional_non_negative_number(self.duration_s, "duration_s")
        if self.codec_or_container_optional is not None:
            _validate_non_empty_string(
                self.codec_or_container_optional, "codec_or_container_optional"
            )
        _validate_frame_ids(
            self.sampled_frame_ids, "sampled_frame_ids", non_empty=False
        )
        for field_name in (
            "blur_summary",
            "exposure_summary",
            "motion_summary",
            "scene_change_summary",
            "metadata",
        ):
            _validate_mapping(getattr(self, field_name), field_name)
        _validate_probability(self.usable_frame_ratio, "usable_frame_ratio")
        _validate_string_sequence(
            self.hard_rejection_reasons, "hard_rejection_reasons"
        )
        _validate_string_sequence(self.soft_risk_flags, "soft_risk_flags")
        _validate_probability(self.confidence, "confidence")
        if self.frame_count == 0 and not self.hard_rejection_reasons:
            raise ContractValidationError(
                "empty inspection reports must explain why no frames were inspected"
            )


@dataclass(frozen=True)
class KeyframeProposal:
    asset_id: str
    selected_frame_ids: Sequence[int]
    timestamps_s: Sequence[float | None]
    selection_reasons: Sequence[str]
    sharpness_scores: Sequence[float]
    scene_change_scores: Sequence[float]
    motion_or_baseline_proxy_scores: Sequence[float]
    coverage_or_overlap_proxy_scores: Sequence[float]
    risk_flags: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.asset_id, "asset_id")
        frame_ids = _validate_frame_ids(
            self.selected_frame_ids, "selected_frame_ids", non_empty=False
        )
        count = len(frame_ids)
        sequence_fields = {
            "timestamps_s": self.timestamps_s,
            "selection_reasons": self.selection_reasons,
            "sharpness_scores": self.sharpness_scores,
            "scene_change_scores": self.scene_change_scores,
            "motion_or_baseline_proxy_scores": self.motion_or_baseline_proxy_scores,
            "coverage_or_overlap_proxy_scores": self.coverage_or_overlap_proxy_scores,
        }
        for field_name, value in sequence_fields.items():
            sequence = _validate_sequence(value, field_name)
            if len(sequence) != count:
                raise ContractValidationError(
                    f"{field_name} length must match selected_frame_ids"
                )
        for timestamp in self.timestamps_s:
            if timestamp is not None:
                _validate_non_negative_number(timestamp, "timestamps_s")
        _validate_string_sequence(self.selection_reasons, "selection_reasons")
        for field_name in (
            "sharpness_scores",
            "scene_change_scores",
            "motion_or_baseline_proxy_scores",
            "coverage_or_overlap_proxy_scores",
        ):
            for score in getattr(self, field_name):
                _validate_probability(score, field_name)
        _validate_string_sequence(self.risk_flags, "risk_flags")
        _validate_mapping(self.metadata, "metadata")


@dataclass(frozen=True)
class VideoInput:
    source_uri: str
    frame_count: int
    fps: float
    width_px: int
    height_px: int
    timestamp_s: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.source_uri, "source_uri")
        _validate_positive_int(self.frame_count, "frame_count")
        _validate_positive_number(self.fps, "fps")
        _validate_positive_int(self.width_px, "width_px")
        _validate_positive_int(self.height_px, "height_px")
        timestamps = _validate_sequence(self.timestamp_s, "timestamp_s")
        if len(timestamps) != self.frame_count:
            raise ContractValidationError("timestamp_s length must match frame_count")
        previous = None
        for timestamp in timestamps:
            _validate_non_negative_number(timestamp, "timestamp_s")
            if previous is not None and timestamp <= previous:
                raise ContractValidationError("timestamp_s must be strictly increasing")
            previous = timestamp
        _validate_mapping(self.metadata, "metadata")


@dataclass(frozen=True)
class ReconstructabilityReport:
    accepted_for_reconstruction: bool
    rejection_reasons: Sequence[str]
    parallax_score: float
    blur_score: float
    dynamic_foreground_ratio: float
    zoom_or_stabilization_score: float
    static_structure_score: float
    confidence: float

    def __post_init__(self) -> None:
        _validate_bool(self.accepted_for_reconstruction, "accepted_for_reconstruction")
        reasons = _validate_sequence(self.rejection_reasons, "rejection_reasons")
        for reason in reasons:
            _validate_non_empty_string(reason, "rejection_reasons")
        if not self.accepted_for_reconstruction and not reasons:
            raise ContractValidationError("rejected videos must carry rejection reasons")
        for field_name in (
            "parallax_score",
            "blur_score",
            "dynamic_foreground_ratio",
            "zoom_or_stabilization_score",
            "static_structure_score",
            "confidence",
        ):
            _validate_probability(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class KeyframeSet:
    selected_frame_ids: Sequence[int]
    selection_reasons: Sequence[str]
    baseline_scores: Sequence[float]
    overlap_scores: Sequence[float]
    sharpness_scores: Sequence[float]
    dynamic_ratio_scores: Sequence[float]
    timestamps_s: Sequence[float]

    def __post_init__(self) -> None:
        frame_ids = _validate_frame_ids(self.selected_frame_ids, "selected_frame_ids")
        count = len(frame_ids)
        fields = {
            "selection_reasons": self.selection_reasons,
            "baseline_scores": self.baseline_scores,
            "overlap_scores": self.overlap_scores,
            "sharpness_scores": self.sharpness_scores,
            "dynamic_ratio_scores": self.dynamic_ratio_scores,
            "timestamps_s": self.timestamps_s,
        }
        for field_name, value in fields.items():
            sequence = _validate_sequence(value, field_name)
            if len(sequence) != count:
                raise ContractValidationError(
                    f"{field_name} length must match selected_frame_ids"
                )
        for field_name in (
            "baseline_scores",
            "overlap_scores",
            "sharpness_scores",
            "dynamic_ratio_scores",
        ):
            for score in getattr(self, field_name):
                _validate_probability(score, field_name)
        for reason in self.selection_reasons:
            _validate_non_empty_string(reason, "selection_reasons")
        for timestamp in self.timestamps_s:
            _validate_non_negative_number(timestamp, "timestamps_s")


@dataclass(frozen=True)
class FrameRayPacket:
    asset_id: str
    frame_id: int
    T_world_camera: Sequence[Sequence[float]]
    rays_camera: Any
    radial_depth_m: Any
    confidence: Any
    camera_model: CameraModel
    source: str
    uncertainty: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_depth_convention: DepthConvention
    intrinsics: Mapping[str, Any] | None = None
    rolling_shutter_model: Mapping[str, Any] | None = None
    depth_residual_field: Any | None = None
    camera_confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.asset_id, "asset_id")
        _validate_non_negative_int(self.frame_id, "frame_id")
        _validate_shape(self.T_world_camera, (4, 4), "T_world_camera")
        _validate_all_finite(self.T_world_camera, "T_world_camera")
        last_row = tuple(float(value) for value in _to_plain(self.T_world_camera)[3])
        if last_row != (0.0, 0.0, 0.0, 1.0):
            raise ContractValidationError(
                "T_world_camera must be a homogeneous camera-to-world transform"
            )
        ray_shape = _validate_shape(self.rays_camera, (None, None, 3), "rays_camera")
        height, width, _ = ray_shape
        _validate_shape(self.radial_depth_m, (height, width), "radial_depth_m")
        _validate_shape(self.confidence, (height, width), "confidence")
        _validate_all_positive(self.radial_depth_m, "radial_depth_m")
        _validate_all_probability(self.confidence, "confidence")
        _validate_rays_unit(self.rays_camera, "rays_camera")
        if not isinstance(self.camera_model, CameraModel):
            raise ContractValidationError(
                "camera_model must support unproject(pixel_uv, radial_depth_m) and project(X_camera)"
            )
        _validate_non_empty_string(self.source, "source")
        _validate_mapping(self.uncertainty, "uncertainty", non_empty=True)
        _validate_mapping(self.provenance, "provenance", non_empty=True)
        convention = _as_enum(
            self.source_depth_convention,
            DepthConvention,
            "source_depth_convention",
        )
        object.__setattr__(self, "source_depth_convention", convention)
        if convention is DepthConvention.UNKNOWN:
            raise ContractValidationError(
                "source_depth_convention=unknown is not allowed for FrameRayPacket"
            )
        if self.intrinsics is not None:
            _validate_mapping(self.intrinsics, "intrinsics")
        if self.rolling_shutter_model is not None:
            _validate_mapping(self.rolling_shutter_model, "rolling_shutter_model")
        if self.camera_confidence is not None:
            _validate_probability(self.camera_confidence, "camera_confidence")


@dataclass(frozen=True)
class ExternalDependencyStatus:
    backend_name: str
    availability: BackboneAvailability
    dependency_paths: Sequence[str] = field(default_factory=tuple)
    artifact_paths: Sequence[str] = field(default_factory=tuple)
    missing_dependency_paths: Sequence[str] = field(default_factory=tuple)
    missing_artifact_paths: Sequence[str] = field(default_factory=tuple)
    message: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.backend_name, "backend_name")
        availability = _as_enum(self.availability, BackboneAvailability, "availability")
        object.__setattr__(self, "availability", availability)
        for field_name in (
            "dependency_paths",
            "artifact_paths",
            "missing_dependency_paths",
            "missing_artifact_paths",
        ):
            values = _validate_sequence(getattr(self, field_name), field_name)
            for value in values:
                _validate_non_empty_string(value, field_name)
        if availability is BackboneAvailability.UNAVAILABLE and not (
            self.missing_dependency_paths or self.missing_artifact_paths or self.message
        ):
            raise ContractValidationError(
                "unavailable external backbones must explain missing dependencies or artifacts"
            )


@dataclass(frozen=True)
class GeometryBackbonePrediction:
    frame_ray_packets: Sequence[FrameRayPacket]
    camera_confidence: float
    depth_confidence: float
    dependency_provenance: ExternalDependencyStatus

    def __post_init__(self) -> None:
        _validate_sequence(self.frame_ray_packets, "frame_ray_packets", non_empty=True)
        _validate_probability(self.camera_confidence, "camera_confidence")
        _validate_probability(self.depth_confidence, "depth_confidence")
        if not isinstance(self.dependency_provenance, ExternalDependencyStatus):
            raise ContractValidationError(
                "dependency_provenance must be an ExternalDependencyStatus"
            )


@dataclass(frozen=True)
class UnavailableVideoGeometryBackbone:
    backend_name: str
    dependency_status: ExternalDependencyStatus

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.backend_name, "backend_name")
        if not isinstance(self.dependency_status, ExternalDependencyStatus):
            raise ContractValidationError(
                "dependency_status must be an ExternalDependencyStatus"
            )
        if self.dependency_status.availability is not BackboneAvailability.UNAVAILABLE:
            raise ContractValidationError(
                "UnavailableVideoGeometryBackbone requires unavailable dependency_status"
            )

    def predict(self, video_or_keyframes: object) -> GeometryBackbonePrediction:
        del video_or_keyframes
        raise ExternalBackboneUnavailableError(
            f"{self.backend_name} is unavailable: {self.dependency_status.message}"
        )


@dataclass(frozen=True)
class MaskTrack:
    track_id: str
    frame_ids: Sequence[int]
    mask_rle_or_bitmap: Any
    mask_confidence: float
    prompt_or_source: str
    semantic_label_optional: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.track_id, "track_id")
        _validate_frame_ids(self.frame_ids, "frame_ids")
        if self.mask_rle_or_bitmap is None:
            raise ContractValidationError("mask_rle_or_bitmap must be present")
        _validate_probability(self.mask_confidence, "mask_confidence")
        _validate_non_empty_string(self.prompt_or_source, "prompt_or_source")
        if self.semantic_label_optional is not None:
            _validate_non_empty_string(
                self.semantic_label_optional, "semantic_label_optional"
            )


@dataclass(frozen=True)
class MaskTrackSet:
    tracks: Sequence[MaskTrack]

    def __post_init__(self) -> None:
        tracks = _validate_sequence(self.tracks, "tracks", non_empty=True)
        for track in tracks:
            if not isinstance(track, MaskTrack):
                raise ContractValidationError("tracks must contain MaskTrack objects")
        track_ids = [track.track_id for track in tracks]
        if len(set(track_ids)) != len(track_ids):
            raise ContractValidationError("track_id values must be unique")


@dataclass(frozen=True)
class MaskTrackDecision:
    track_id: str
    label: StaticDynamicLabel
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.track_id, "track_id")
        label = _as_enum(self.label, StaticDynamicLabel, "label")
        object.__setattr__(self, "label", label)
        _validate_probability(self.confidence, "confidence")
        _validate_non_empty_string(self.reason, "reason")


@dataclass(frozen=True)
class StaticDynamicState:
    frame_id: int
    static_probability: Any
    dynamic_probability: Any
    unknown_probability: Any
    mask_track_decisions: Sequence[MaskTrackDecision]
    residual_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.frame_id, "frame_id")
        _validate_same_shape(
            {
                "static_probability": self.static_probability,
                "dynamic_probability": self.dynamic_probability,
                "unknown_probability": self.unknown_probability,
            }
        )
        _validate_probability_simplex(
            {
                "static_probability": self.static_probability,
                "dynamic_probability": self.dynamic_probability,
                "unknown_probability": self.unknown_probability,
            }
        )
        decisions = _validate_sequence(
            self.mask_track_decisions, "mask_track_decisions"
        )
        for decision in decisions:
            if not isinstance(decision, MaskTrackDecision):
                raise ContractValidationError(
                    "mask_track_decisions must contain MaskTrackDecision objects"
                )
        _validate_mapping(self.residual_summary, "residual_summary", non_empty=True)


@dataclass(frozen=True)
class VisibilityEdge:
    source_frame_id: int
    target_frame_id: int
    measurements: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.source_frame_id, "source_frame_id")
        _validate_non_negative_int(self.target_frame_id, "target_frame_id")
        if self.source_frame_id == self.target_frame_id:
            raise ContractValidationError("visibility edges must connect two frames")
        _validate_mapping(self.measurements, "measurements", non_empty=True)
        for name, value in self.measurements.items():
            _validate_non_empty_string(name, "measurements")
            _validate_non_negative_number(value, "measurements")


@dataclass(frozen=True)
class VisibilityGraph:
    nodes: Sequence[int]
    temporal_edges: Sequence[VisibilityEdge]
    overlap_edges: Sequence[VisibilityEdge]
    loop_edges: Sequence[VisibilityEdge]
    scale_edges: Sequence[VisibilityEdge]
    edge_measurements: Mapping[str, Any]

    def __post_init__(self) -> None:
        nodes = _validate_frame_ids(self.nodes, "nodes")
        if len(set(nodes)) != len(nodes):
            raise ContractValidationError("visibility graph nodes must be unique")
        node_set = set(nodes)
        for field_name in (
            "temporal_edges",
            "overlap_edges",
            "loop_edges",
            "scale_edges",
        ):
            edges = _validate_sequence(getattr(self, field_name), field_name)
            for edge in edges:
                if not isinstance(edge, VisibilityEdge):
                    raise ContractValidationError(
                        f"{field_name} must contain VisibilityEdge objects"
                    )
                if (
                    edge.source_frame_id not in node_set
                    or edge.target_frame_id not in node_set
                ):
                    raise ContractValidationError(
                        f"{field_name} must only reference visibility graph nodes"
                    )
        _validate_mapping(self.edge_measurements, "edge_measurements")


@dataclass(frozen=True)
class ScaleEvidence:
    evidence_id: str
    evidence_type: ScaleEvidenceType
    measured: bool
    source: str
    frame_ids: Sequence[int]
    confidence: float
    provenance: Mapping[str, Any]
    object_or_region_id: str | None = None
    length_mean_m: float | None = None
    length_std_m: float | None = None
    scale_mean: float | None = None
    scale_std: float | None = None
    residual_after_optimization: float | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.evidence_id, "evidence_id")
        evidence_type = _as_enum(self.evidence_type, ScaleEvidenceType, "evidence_type")
        object.__setattr__(self, "evidence_type", evidence_type)
        _validate_bool(self.measured, "measured")
        if self.measured and evidence_type not in _MEASURED_EVIDENCE_TYPES:
            raise ContractValidationError(
                "measured=true is allowed only for real metric measurement evidence"
            )
        _validate_non_empty_string(self.source, "source")
        _validate_frame_ids(self.frame_ids, "frame_ids")
        _validate_probability(self.confidence, "confidence")
        _validate_mapping(self.provenance, "provenance", non_empty=True)
        if self.object_or_region_id is not None:
            _validate_non_empty_string(self.object_or_region_id, "object_or_region_id")
        for field_name in ("length_mean_m", "length_std_m", "scale_mean", "scale_std"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_positive_number(value, field_name)
        if self.residual_after_optimization is not None:
            _validate_non_negative_number(
                self.residual_after_optimization, "residual_after_optimization"
            )


@dataclass(frozen=True)
class ScalePosterior:
    scale_mean: float
    scale_std: float
    relative_scale_uncertainty: float
    scale_sources: Sequence[ScaleEvidence]
    anchor_residuals: Sequence[float]
    metric_acceptance_status: MetricAcceptanceStatus

    def __post_init__(self) -> None:
        _validate_positive_number(self.scale_mean, "scale_mean")
        _validate_positive_number(self.scale_std, "scale_std")
        _validate_positive_number(
            self.relative_scale_uncertainty, "relative_scale_uncertainty"
        )
        expected_relative = float(self.scale_std) / float(self.scale_mean)
        if abs(float(self.relative_scale_uncertainty) - expected_relative) > 1e-9:
            raise ContractValidationError(
                "relative_scale_uncertainty must equal scale_std / scale_mean"
            )
        sources = _validate_sequence(self.scale_sources, "scale_sources")
        for source in sources:
            if not isinstance(source, ScaleEvidence):
                raise ContractValidationError(
                    "scale_sources must contain ScaleEvidence objects"
                )
        residuals = _validate_sequence(self.anchor_residuals, "anchor_residuals")
        for residual in residuals:
            _validate_non_negative_number(residual, "anchor_residuals")
        status = _as_enum(
            self.metric_acceptance_status,
            MetricAcceptanceStatus,
            "metric_acceptance_status",
        )
        object.__setattr__(self, "metric_acceptance_status", status)
        if status is MetricAcceptanceStatus.MEASURED_METRIC and not any(
            source.measured for source in sources
        ):
            raise ContractValidationError(
                "measured_metric requires at least one measured scale evidence source"
            )
        if status is MetricAcceptanceStatus.METRIC_PSEUDO_LABEL and not sources:
            raise ContractValidationError(
                "metric_pseudo_label requires at least one scale evidence source"
            )


@dataclass(frozen=True)
class OptimizedSceneState:
    frame_ray_packets: Sequence[FrameRayPacket]
    scale_posterior: ScalePosterior
    static_dynamic_state: Sequence[StaticDynamicState]
    visibility_graph: VisibilityGraph
    optimizer_trace: Mapping[str, Any]
    validation_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        packets = _validate_sequence(self.frame_ray_packets, "frame_ray_packets", non_empty=True)
        for packet in packets:
            if not isinstance(packet, FrameRayPacket):
                raise ContractValidationError(
                    "frame_ray_packets must contain FrameRayPacket objects"
                )
        if not isinstance(self.scale_posterior, ScalePosterior):
            raise ContractValidationError("scale_posterior must be a ScalePosterior")
        states = _validate_sequence(
            self.static_dynamic_state, "static_dynamic_state", non_empty=True
        )
        packet_frame_ids = {packet.frame_id for packet in packets}
        for state in states:
            if not isinstance(state, StaticDynamicState):
                raise ContractValidationError(
                    "static_dynamic_state must contain StaticDynamicState objects"
                )
            if state.frame_id not in packet_frame_ids:
                raise ContractValidationError(
                    "static_dynamic_state frame_id must match a FrameRayPacket"
                )
        if not isinstance(self.visibility_graph, VisibilityGraph):
            raise ContractValidationError("visibility_graph must be a VisibilityGraph")
        _validate_mapping(self.optimizer_trace, "optimizer_trace")
        _validate_mapping(self.validation_inputs, "validation_inputs")


@dataclass(frozen=True)
class VoxelMapState:
    voxel_size_m: float
    coordinate_frame: str
    tsdf_value: Any
    tsdf_weight: Any
    occupancy_log_odds: Any
    free_space_count: Any
    surface_count: Any
    dynamic_count: Any
    uncertainty: Any

    def __post_init__(self) -> None:
        _validate_positive_number(self.voxel_size_m, "voxel_size_m")
        _validate_non_empty_string(self.coordinate_frame, "coordinate_frame")
        if self.coordinate_frame.lower() in {"wc", "cw", "world_camera", "camera_world"}:
            raise ContractValidationError(
                "coordinate_frame must be an explicit frame name, not a transform abbreviation"
            )
        shape = _validate_same_shape(
            {
                "tsdf_value": self.tsdf_value,
                "tsdf_weight": self.tsdf_weight,
                "occupancy_log_odds": self.occupancy_log_odds,
                "free_space_count": self.free_space_count,
                "surface_count": self.surface_count,
                "dynamic_count": self.dynamic_count,
                "uncertainty": self.uncertainty,
            }
        )
        if len(shape) != 3:
            raise ContractValidationError("voxel map tensors must be 3D")
        _validate_all_finite(self.tsdf_value, "tsdf_value")
        _validate_all_non_negative(self.tsdf_weight, "tsdf_weight")
        _validate_all_finite(self.occupancy_log_odds, "occupancy_log_odds")
        for field_name in ("free_space_count", "surface_count", "dynamic_count"):
            _validate_all_non_negative(getattr(self, field_name), field_name)
        _validate_all_non_negative(self.uncertainty, "uncertainty")


@dataclass(frozen=True)
class MeshChunkMetadata:
    chunk_id: str
    source_frame_ids: Sequence[int]
    observed_coverage_estimate: float
    voxel_size_m: float
    coordinate_frame: str
    metric_scale_source: str
    mean_uncertainty_m: float
    p50_uncertainty_m: float
    p95_uncertainty_m: float
    observed_only: bool
    predicted_completion: bool

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.chunk_id, "chunk_id")
        _validate_frame_ids(self.source_frame_ids, "source_frame_ids")
        _validate_probability(
            self.observed_coverage_estimate, "observed_coverage_estimate"
        )
        _validate_positive_number(self.voxel_size_m, "voxel_size_m")
        _validate_non_empty_string(self.coordinate_frame, "coordinate_frame")
        _validate_non_empty_string(self.metric_scale_source, "metric_scale_source")
        for field_name in (
            "mean_uncertainty_m",
            "p50_uncertainty_m",
            "p95_uncertainty_m",
        ):
            _validate_non_negative_number(getattr(self, field_name), field_name)
        _validate_bool(self.observed_only, "observed_only")
        _validate_bool(self.predicted_completion, "predicted_completion")
        if self.observed_only and self.predicted_completion:
            raise ContractValidationError(
                "observed_only and predicted_completion must remain distinct"
            )


@dataclass(frozen=True)
class OccupancyGrid2D:
    """Floor-aligned robot grid.

    As of the 3D occupancy work this is the PURE top-down projection of
    :class:`VoxelOccupancyGrid3D` (the single source of truth). The projection
    collapses the collision band along the floor axis, so it drops the *height
    within the band* at which an obstacle occurs (that height is preserved in
    ``height_min_m`` / ``height_max_m``). It must never be fused independently of
    the 3D field.
    """

    grid_frame: str
    resolution_m: float
    origin_world: Sequence[float]
    P_free: Any
    P_occupied_static: Any
    P_movable_static: Any
    P_dynamic: Any
    P_unknown: Any
    height_min_m: Any
    height_max_m: Any
    scale_uncertainty: float
    map_confidence: float

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.grid_frame, "grid_frame")
        _validate_positive_number(self.resolution_m, "resolution_m")
        _validate_shape(self.origin_world, (3,), "origin_world")
        _validate_all_finite(self.origin_world, "origin_world")
        channel_fields = {
            "P_free": self.P_free,
            "P_occupied_static": self.P_occupied_static,
            "P_movable_static": self.P_movable_static,
            "P_dynamic": self.P_dynamic,
            "P_unknown": self.P_unknown,
            "height_min_m": self.height_min_m,
            "height_max_m": self.height_max_m,
        }
        shape = _validate_same_shape(channel_fields)
        if len(shape) != 2:
            raise ContractValidationError("occupancy grid channels must be 2D")
        for field_name in (
            "P_free",
            "P_occupied_static",
            "P_movable_static",
            "P_dynamic",
            "P_unknown",
        ):
            _validate_all_probability(getattr(self, field_name), field_name)
        _validate_all_finite(self.height_min_m, "height_min_m")
        _validate_all_finite(self.height_max_m, "height_max_m")
        for min_height, max_height in _iter_pairs(self.height_min_m, self.height_max_m):
            if float(max_height) < float(min_height):
                raise ContractValidationError("height_max_m must be >= height_min_m")
        _validate_non_negative_number(self.scale_uncertainty, "scale_uncertainty")
        _validate_probability(self.map_confidence, "map_confidence")
        self._validate_grid_channels_are_not_collapsed()

    def _validate_grid_channels_are_not_collapsed(self) -> None:
        for free, unknown in _iter_pairs(self.P_free, self.P_unknown):
            if float(free) + float(unknown) > 1.0 + 1e-6:
                raise ContractValidationError("P_unknown must remain distinct from P_free")
        for dynamic, occupied in _iter_pairs(self.P_dynamic, self.P_occupied_static):
            if float(dynamic) + float(occupied) > 1.0 + 1e-6:
                raise ContractValidationError(
                    "P_dynamic must remain distinct from P_occupied_static"
                )
        for movable, free in _iter_pairs(self.P_movable_static, self.P_free):
            if float(movable) + float(free) > 1.0 + 1e-6:
                raise ContractValidationError(
                    "P_movable_static must remain distinct from P_free"
                )


@dataclass(frozen=True)
class VoxelOccupancyGrid3D:
    """Primary robot-facing output: a per-voxel multichannel occupancy field
    bounded to the robot's vertical collision envelope.

    The field lives in the floor-aligned metric (or reconstruction) frame. It is
    cropped along ``floor_axis`` to the band ``[band_min_m, band_max_m]`` =
    ``[floor, floor + collision_height + margin]`` -- resolution is spent only
    inside the collision envelope; the ceiling / full room volume is never
    modelled.

    Probability convention mirrors :class:`OccupancyGrid2D`: each channel is a
    probability in ``[0, 1]`` with PAIRWISE non-collapse (NOT a strict sum-to-one
    simplex). The honest per-voxel invariants are non-negotiable:

    - unknown is never free (``P_free + P_unknown <= 1``);
    - dynamic is never static (``P_dynamic + P_occupied_static <= 1``);
    - movable_static is never free (``P_movable_static + P_free <= 1``);
    - free space comes from RAY TRAVERSAL only; space behind a surface along a ray
      stays unknown (enforced by the fuser, not here).

    All channel volumes plus ``map_confidence`` share one 3D shape ``(A0, A1, B)``
    where ``B`` is the number of band slices along ``floor_axis``. ``origin_world``
    is the world coordinate of the ``(0, 0, 0)`` voxel corner (the cropped band
    ``grid_min``); voxel center ``(i, j, k)`` is
    ``origin_world + (index + 0.5) * voxel_size_m`` along the three world axes.
    """

    grid_frame: str
    voxel_size_m: float
    origin_world: Sequence[float]
    floor_axis: int
    band_min_m: float
    band_max_m: float
    P_free: Any
    P_occupied_static: Any
    P_movable_static: Any
    P_dynamic: Any
    P_unknown: Any
    map_confidence: Any
    scale_uncertainty: float
    acceptance_category: MetricAcceptanceStatus

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.grid_frame, "grid_frame")
        _validate_positive_number(self.voxel_size_m, "voxel_size_m")
        _validate_shape(self.origin_world, (3,), "origin_world")
        _validate_all_finite(self.origin_world, "origin_world")
        if not isinstance(self.floor_axis, int) or self.floor_axis not in (0, 1, 2):
            raise ContractValidationError("floor_axis must be one of 0, 1, 2")
        # band_min_m / band_max_m are world coordinates along the floor axis and
        # may be negative (the floor can sit below the world origin) -- require
        # finite, not non-negative.
        for field_name in ("band_min_m", "band_max_m"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ContractValidationError(f"{field_name} must be a finite number")
        if float(self.band_max_m) < float(self.band_min_m):
            raise ContractValidationError("band_max_m must be >= band_min_m")
        channel_fields = {
            "P_free": self.P_free,
            "P_occupied_static": self.P_occupied_static,
            "P_movable_static": self.P_movable_static,
            "P_dynamic": self.P_dynamic,
            "P_unknown": self.P_unknown,
            "map_confidence": self.map_confidence,
        }
        shape = _validate_same_shape(channel_fields)
        if len(shape) != 3:
            raise ContractValidationError("voxel occupancy channels must be 3D")
        for field_name in channel_fields:
            _validate_all_probability(getattr(self, field_name), field_name)
        self._validate_voxel_channels_are_not_collapsed()
        _validate_non_negative_number(self.scale_uncertainty, "scale_uncertainty")
        status = _as_enum(
            self.acceptance_category, MetricAcceptanceStatus, "acceptance_category"
        )
        object.__setattr__(self, "acceptance_category", status)

    def _validate_voxel_channels_are_not_collapsed(self) -> None:
        for free, unknown in _iter_pairs(self.P_free, self.P_unknown):
            if float(free) + float(unknown) > 1.0 + 1e-6:
                raise ContractValidationError("P_unknown must remain distinct from P_free")
        for dynamic, occupied in _iter_pairs(self.P_dynamic, self.P_occupied_static):
            if float(dynamic) + float(occupied) > 1.0 + 1e-6:
                raise ContractValidationError(
                    "P_dynamic must remain distinct from P_occupied_static"
                )
        for movable, free in _iter_pairs(self.P_movable_static, self.P_free):
            if float(movable) + float(free) > 1.0 + 1e-6:
                raise ContractValidationError(
                    "P_movable_static must remain distinct from P_free"
                )


@dataclass(frozen=True)
class ValidationReport:
    held_out_render_error: float
    free_space_contradiction_rate: float
    scale_posterior: ScalePosterior
    floor_wall_consistency: float
    dynamic_leakage_score: float
    accepted_for_metric_training: bool
    rejection_reasons: Sequence[str]
    # Optional per-voxel agreement of the monocular 3D field vs the measured 3D
    # field, computed inside the collision band. Reportage only -- it never gates
    # ``accepted_for_metric_training`` (the category is driven by the scale
    # posterior). ``None`` when no measured 3D reference exists (e.g. phone_room).
    band3d_agreement: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_number(
            self.held_out_render_error, "held_out_render_error"
        )
        _validate_probability(
            self.free_space_contradiction_rate, "free_space_contradiction_rate"
        )
        if not isinstance(self.scale_posterior, ScalePosterior):
            raise ContractValidationError("scale_posterior must be a ScalePosterior")
        _validate_probability(self.floor_wall_consistency, "floor_wall_consistency")
        _validate_probability(self.dynamic_leakage_score, "dynamic_leakage_score")
        _validate_bool(
            self.accepted_for_metric_training, "accepted_for_metric_training"
        )
        reasons = _validate_sequence(self.rejection_reasons, "rejection_reasons")
        for reason in reasons:
            _validate_non_empty_string(reason, "rejection_reasons")
        status = self.scale_posterior.metric_acceptance_status
        if self.accepted_for_metric_training:
            if status not in {
                MetricAcceptanceStatus.MEASURED_METRIC,
                MetricAcceptanceStatus.METRIC_PSEUDO_LABEL,
            }:
                raise ContractValidationError(
                    "metric training acceptance requires metric ScalePosterior status"
                )
            if reasons:
                raise ContractValidationError(
                    "accepted metric training reports must not carry rejection reasons"
                )
        elif not reasons:
            raise ContractValidationError(
                "rejected validation reports must carry rejection reasons"
            )
        if self.band3d_agreement is not None:
            _validate_mapping(self.band3d_agreement, "band3d_agreement", non_empty=True)
