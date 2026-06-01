"""Teacher model adapter contracts for Atlas3R."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from atlas3r.api import FramePacket, FramePrediction
from atlas3r.api.validation import validate_mapping, validate_nonempty_str


class AdapterAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STUB_ONLY = "stub-only"


ADAPTER_AVAILABILITY_VALUES = {item.value for item in AdapterAvailability}


def _set(instance: object, field_name: str, value: object) -> None:
    object.__setattr__(instance, field_name, value)


def _validate_bool(field_name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name}: must be a bool")
    return value


def _validate_optional_str(field_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return validate_nonempty_str(field_name, value)


def _validate_availability(value: str) -> str:
    if value not in ADAPTER_AVAILABILITY_VALUES:
        allowed = ", ".join(sorted(ADAPTER_AVAILABILITY_VALUES))
        raise ValueError(f"availability: must be one of: {allowed}")
    return value


@dataclass(frozen=True)
class AdapterCapabilities:
    predicts_camera: bool
    predicts_pose: bool
    predicts_depth: bool
    predicts_normals: bool
    predicts_points: bool
    predicts_dense_matches: bool
    predicts_objects: bool
    supports_batch: bool
    supports_streaming: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "predicts_camera",
            "predicts_pose",
            "predicts_depth",
            "predicts_normals",
            "predicts_points",
            "predicts_dense_matches",
            "predicts_objects",
            "supports_batch",
            "supports_streaming",
        ):
            _validate_bool(field_name, getattr(self, field_name))
        notes = tuple(self.notes)
        for index, note in enumerate(notes):
            validate_nonempty_str(f"notes[{index}]", note)
        _set(self, "notes", notes)


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    display_name: str
    availability: str
    capabilities: AdapterCapabilities
    install_hint: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _set(self, "name", validate_nonempty_str("name", self.name))
        _set(self, "display_name", validate_nonempty_str("display_name", self.display_name))
        _set(self, "availability", _validate_availability(self.availability))
        if not isinstance(self.capabilities, AdapterCapabilities):
            raise ValueError("capabilities: must be AdapterCapabilities")
        _set(self, "install_hint", _validate_optional_str("install_hint", self.install_hint))
        _set(self, "reason", _validate_optional_str("reason", self.reason))


@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[FramePacket, ...]
    batch_id: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("frames: must contain at least one FramePacket")
        seen_frame_ids: set[int] = set()
        for index, frame in enumerate(frames):
            if not isinstance(frame, FramePacket):
                raise ValueError(f"frames[{index}]: must be FramePacket")
            if frame.frame_id in seen_frame_ids:
                raise ValueError("frames: frame_id values must be unique")
            seen_frame_ids.add(frame.frame_id)
        _set(self, "frames", frames)
        _set(self, "batch_id", validate_nonempty_str("batch_id", self.batch_id))
        _set(self, "metadata", validate_mapping("metadata", self.metadata))


@dataclass(frozen=True)
class TeacherPrediction:
    adapter_name: str
    frame_predictions: tuple[FramePrediction, ...]
    capabilities: AdapterCapabilities
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set(self, "adapter_name", validate_nonempty_str("adapter_name", self.adapter_name))
        frame_predictions = tuple(self.frame_predictions)
        if not frame_predictions:
            raise ValueError("frame_predictions: must contain at least one FramePrediction")
        seen_frame_ids: set[int] = set()
        for index, prediction in enumerate(frame_predictions):
            if not isinstance(prediction, FramePrediction):
                raise ValueError(f"frame_predictions[{index}]: must be FramePrediction")
            frame_id = prediction.pose.frame_id
            if frame_id in seen_frame_ids:
                raise ValueError("frame_predictions: pose.frame_id values must be unique")
            seen_frame_ids.add(frame_id)
        _set(self, "frame_predictions", frame_predictions)
        if not isinstance(self.capabilities, AdapterCapabilities):
            raise ValueError("capabilities: must be AdapterCapabilities")
        _set(self, "metadata", validate_mapping("metadata", self.metadata))


class AdapterDependencyError(RuntimeError):
    """Raised when a teacher adapter is requested without optional dependencies."""

    def __init__(
        self,
        adapter_name: str,
        missing_dependencies: tuple[str, ...],
        install_hint: str,
    ) -> None:
        missing = ", ".join(missing_dependencies) if missing_dependencies else "unknown"
        super().__init__(
            f"{adapter_name} adapter is unavailable: missing optional dependency "
            f"{missing}. {install_hint}"
        )
        self.adapter_name = adapter_name
        self.missing_dependencies = missing_dependencies
        self.install_hint = install_hint


class AdapterNotImplementedError(RuntimeError):
    """Raised when a dependency-present adapter remains a Phase 0E stub."""


class GeometryTeacherAdapter(Protocol):
    def predict(self, frames: FrameBatch) -> TeacherPrediction: ...


__all__ = [
    "AdapterAvailability",
    "AdapterCapabilities",
    "AdapterDependencyError",
    "AdapterNotImplementedError",
    "AdapterStatus",
    "FrameBatch",
    "GeometryTeacherAdapter",
    "TeacherPrediction",
]
