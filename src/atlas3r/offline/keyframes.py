"""Keyframe selection for dependency-safe offline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.frame_cache import FrameRecord
from atlas3r.offline.run_manifest import FailurePoint, write_json


@dataclass(frozen=True)
class KeyframeRecord:
    keyframe_id: int
    frame_id: int
    timestamp_ns: int
    frame_path: str
    quality: dict[str, object]
    selection_rank: int
    reasons: tuple[str, ...]
    pose_assumption: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "frame_path": self.frame_path,
            "quality": self.quality,
            "selection_rank": self.selection_rank,
            "reasons": list(self.reasons),
            "pose_assumption": self.pose_assumption,
        }


@dataclass(frozen=True)
class KeyframeSelectionResult:
    status: str
    keyframes: tuple[KeyframeRecord, ...]
    keyframes_path: str


def select_keyframes(
    records: tuple[FrameRecord, ...],
    run_dir: str | Path,
    *,
    keyframe_stride: int,
    keyframe_max_count: int,
    failure_points: list[FailurePoint],
) -> KeyframeSelectionResult:
    if keyframe_stride <= 0:
        raise ValueError("keyframe_stride must be positive")
    if keyframe_max_count <= 0:
        raise ValueError("keyframe_max_count must be positive")
    target = Path(run_dir) / "keyframes" / "keyframes.json"
    if not records:
        failure_points.append(
            FailurePoint(
                module="keyframe_selector",
                code="no_frames",
                severity="warning",
                status="unavailable",
                why="no frames were available for keyframe selection",
                input_missing="frame records",
                future_module="frame cache with decodable input",
                artifact_path="keyframes/keyframes.json",
            )
        )
        unavailable_payload = {
            "status": "unavailable",
            "keyframes": [],
            "why": "no frames available",
        }
        write_json(target, unavailable_payload)
        return KeyframeSelectionResult("unavailable", (), "keyframes/keyframes.json")
    selected: list[KeyframeRecord] = []
    for record in records:
        reasons = _selection_reasons(record, keyframe_stride)
        if reasons and len(selected) < keyframe_max_count:
            keyframe_id = len(selected)
            selected.append(
                KeyframeRecord(
                    keyframe_id=keyframe_id,
                    frame_id=record.frame_id,
                    timestamp_ns=record.timestamp_ns,
                    frame_path=record.frame_path,
                    quality=record.quality.to_dict(),
                    selection_rank=keyframe_id,
                    reasons=tuple(reasons),
                )
            )
    if not selected:
        first = records[0]
        selected.append(
            KeyframeRecord(
                keyframe_id=0,
                frame_id=first.frame_id,
                timestamp_ns=first.timestamp_ns,
                frame_path=first.frame_path,
                quality=first.quality.to_dict(),
                selection_rank=0,
                reasons=("fallback_first_frame", "no_pose_assumed"),
            )
        )
    payload: dict[str, object] = {
        "status": "complete",
        "selection": {
            "keyframe_stride": keyframe_stride,
            "keyframe_max_count": keyframe_max_count,
            "pose_assumption": "none",
        },
        "keyframes": [item.to_dict() for item in selected],
    }
    write_json(target, payload)
    return KeyframeSelectionResult("complete", tuple(selected), "keyframes/keyframes.json")


def _selection_reasons(record: FrameRecord, keyframe_stride: int) -> list[str]:
    reasons: list[str] = []
    if record.frame_id % keyframe_stride == 0:
        reasons.append(f"stride_{keyframe_stride}")
    if record.quality.visual_change_score >= 0.08:
        reasons.append("visual_change")
    if record.quality.exposure_score >= 0.05:
        reasons.append("usable_exposure")
    if record.quality.blur_score >= 0.01:
        reasons.append("nonflat_texture")
    if reasons:
        reasons.append("no_pose_assumed")
    return reasons
