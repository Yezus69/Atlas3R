"""Minimal recording manifest used as an input boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from atlas3r.contracts._arrays import as_int
from atlas3r.contracts.frames import CameraModel
from atlas3r.contracts.truth import TruthBoundary


@dataclass(frozen=True)
class RecordingFrame:
    frame_id: int
    timestamp_ns: int
    rgb_path: str
    camera: CameraModel
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame_id and timestamp_ns must be non-negative")
        _safe_relative_path(self.rgb_path)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "rgb_path": self.rgb_path,
            "camera": self.camera.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RecordingFrame:
        return cls(
            frame_id=as_int(data["frame_id"], "frame_id"),
            timestamp_ns=as_int(data["timestamp_ns"], "timestamp_ns"),
            rgb_path=str(data["rgb_path"]),
            camera=CameraModel.from_dict(_mapping(data["camera"], "camera")),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class RecordingManifest:
    recording_id: str
    frames: tuple[RecordingFrame, ...]
    truth_boundary: TruthBoundary
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.recording_id:
            raise ValueError("recording_id must be non-empty")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("recording frame IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_name": "atlas3r_recording_manifest",
            "format_version": 1,
            "recording_id": self.recording_id,
            "frames": [frame.to_dict() for frame in self.frames],
            "truth_boundary": self.truth_boundary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RecordingManifest:
        if data.get("format_name") != "atlas3r_recording_manifest":
            raise ValueError("not an atlas3r recording manifest")
        frames_value = data.get("frames", ())
        if not isinstance(frames_value, (list, tuple)):
            raise ValueError("frames must be a list")
        return cls(
            recording_id=str(data["recording_id"]),
            frames=tuple(
                RecordingFrame.from_dict(_mapping(item, "frame")) for item in frames_value
            ),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )


def write_recording(path: str | Path, manifest: RecordingManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_recording(path: str | Path) -> RecordingManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("recording manifest must be a JSON object")
    return RecordingManifest.from_dict(data)


def _safe_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"path must be safe and relative: {value}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
