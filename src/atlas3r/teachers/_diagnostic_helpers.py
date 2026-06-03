"""Small helpers for teacher-signal diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt


def rmse(values: npt.NDArray[np.float64]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return float(np.mean(values)) if values.size else None


def max_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return float(np.max(values)) if values.size else None


def within_percent(values: npt.NDArray[np.float64], threshold: float) -> float | None:
    if not values.size:
        return None
    return float(np.count_nonzero(values <= threshold) / values.size * 100.0)


def aggregate_within_percent(
    within: dict[float, int],
    threshold: float,
    count: int,
) -> float | None:
    return float(within[threshold] / count * 100.0) if count else None


def mean_metric(records: list[dict[str, object]], key: str) -> float | None:
    values = [record[key] for record in records if isinstance(record.get(key), int | float)]
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def unique_ints(values: list[int]) -> list[int]:
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
