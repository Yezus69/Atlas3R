"""Shared report helpers for student map runtime diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

DIAGNOSTIC_TRUTH_FLAGS: dict[str, bool] = {
    "diagnostic_only": True,
    "accuracy_report": False,
    "performance_report": False,
    "realtime_claim": False,
    "mapping_ready": False,
}


@dataclass(frozen=True)
class TeacherReferenceFrame:
    """One measured/reference teacher frame for runtime quality comparison."""

    frame_id: int
    timestamp_s: float
    depth_m: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]
    K: npt.NDArray[np.float32]
    T_world_camera: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    depth_sigma_m: npt.NDArray[np.float32]
    teacher_name: str
    measured_geometry: bool


def write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(record), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(dict(record), handle, sort_keys=True)
            handle.write("\n")


def mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def percentile_or_none(values: npt.NDArray[np.float64], percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


__all__ = [
    "DIAGNOSTIC_TRUTH_FLAGS",
    "TeacherReferenceFrame",
    "mean_or_none",
    "percentile_or_none",
    "write_json",
    "write_jsonl",
]
