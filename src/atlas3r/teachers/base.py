"""Dependency-safe teacher adapter protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from atlas3r.contracts.frames import FramePacket
from atlas3r.contracts.geometry import TeacherProposal


@dataclass(frozen=True)
class AdapterCapabilities:
    depth: bool = False
    intrinsics: bool = False
    pose: bool = False
    point_tracks: bool = False
    object_masks: bool = False
    streaming: bool = False
    metric_depth: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "intrinsics": self.intrinsics,
            "pose": self.pose,
            "point_tracks": self.point_tracks,
            "object_masks": self.object_masks,
            "streaming": self.streaming,
            "metric_depth": self.metric_depth,
        }


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    display_name: str
    available: bool
    capabilities: AdapterCapabilities
    install_hint: str
    reason: str

    def __post_init__(self) -> None:
        if not self.name or not self.display_name:
            raise ValueError("adapter names must be non-empty")
        if not self.available and not self.install_hint:
            raise ValueError("unavailable adapters must provide install_hint")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.available,
            "capabilities": self.capabilities.to_dict(),
            "install_hint": self.install_hint,
            "reason": self.reason,
        }


class TeacherUnavailableError(RuntimeError):
    """Raised by placeholder adapters that need external model installation."""


class GeometryTeacherAdapter(Protocol):
    name: str

    def status(self) -> AdapterStatus: ...

    def predict(self, frames: Sequence[FramePacket]) -> TeacherProposal: ...


@dataclass(frozen=True)
class UnavailableTeacherAdapter:
    name: str
    status_record: AdapterStatus

    def status(self) -> AdapterStatus:
        return self.status_record

    def predict(self, frames: Sequence[FramePacket]) -> TeacherProposal:
        raise TeacherUnavailableError(
            f"{self.name} is unavailable: {self.status_record.reason}. "
            f"{self.status_record.install_hint}"
        )
