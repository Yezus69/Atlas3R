"""Small helpers for teacher-signal diagnostic artifacts."""

from __future__ import annotations

import html
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


def write_preview_html(
    path: Path,
    summary: dict[str, object],
    metrics: list[dict[str, object]],
) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{record['signal_index']}</td>"
        f"<td>{record['overlap_valid_pixel_count']}</td>"
        f"<td>{record['depth_rmse_m']}</td>"
        f"<td>{record['confidence_mean']}</td>"
        "</tr>"
        for record in metrics[:64]
    )
    title = html.escape(str(summary["teacher_name"]))
    summary_json = html.escape(json.dumps(summary, indent=2, sort_keys=True))
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Teacher Signal Inspection</title>"
        f"<h1>{title}</h1><pre>{summary_json}</pre>"
        "<table><thead><tr><th>signal</th><th>overlap</th><th>rmse</th>"
        f"<th>confidence</th></tr></thead><tbody>{rows}</tbody></table>",
        encoding="utf-8",
        newline="\n",
    )


def write_preview_svg(path: Path, metrics: list[dict[str, object]]) -> None:
    bar_count = min(len(metrics), 64)
    width = max(160, bar_count * 6 + 20)
    bars: list[str] = []
    for index, record in enumerate(metrics[:bar_count]):
        rmse_value = record.get("depth_rmse_m")
        height = (
            min(90, int(float(rmse_value) * 1000.0)) if isinstance(rmse_value, int | float) else 0
        )
        x = 10 + index * 6
        y = 100 - height
        bars.append(f"<rect x='{x}' y='{y}' width='4' height='{height}' fill='#2563eb'/>")
    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width}' height='110' viewBox='0 0 {width} 110'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<line x1='10' y1='100' x2='{width - 10}' y2='100' stroke='black'/>"
        f"{''.join(bars)}</svg>",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "aggregate_within_percent",
    "max_or_none",
    "mean_metric",
    "mean_or_none",
    "rmse",
    "unique_ints",
    "within_percent",
    "write_json",
    "write_jsonl",
    "write_preview_html",
    "write_preview_svg",
]
