"""Comparison summaries for Atlas3R ViPE imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas3r.offline.run_manifest import utc_now_iso, write_json


def write_vipe_comparison(
    output: Path,
    vipe_quality: dict[str, object],
    previous_map: Path | None,
    *,
    truth_boundary: dict[str, object],
) -> Path | None:
    if previous_map is None:
        return None
    previous = _load_map_summary(previous_map)
    vipe = _summary_from_quality(output, vipe_quality)
    less_collapsed = _less_collapsed_by_bbox(vipe.get("bbox_size_m"), previous.get("bbox_size_m"))
    payload = {
        "format_name": "atlas3r_vipe_map_comparison",
        "format_version": 1,
        "created_utc": utc_now_iso(),
        "comparison_basis": "artifact counts, finite bbox extents, and trajectory counts only",
        "ground_truth_available": False,
        "physical_accuracy_claim": False,
        "training_quality_claim": False,
        "vipe_map": vipe,
        "previous_map": previous,
        "less_collapsed_than_previous_by_bbox": less_collapsed,
        "truth_boundary": truth_boundary,
    }
    path = output / "comparison_against_previous_map.json"
    write_json(path, payload)
    (output / "comparison_against_previous_map.md").write_text(
        _comparison_markdown(payload), encoding="utf-8"
    )
    return path


def _summary_from_quality(output: Path, quality: dict[str, object]) -> dict[str, object]:
    return {
        "path": str(output),
        "fused_point_count": quality["fused_point_count"],
        "occupied_voxel_count": quality["occupied_voxel_count"],
        "observed_mesh_triangle_count": quality["observed_mesh_triangle_count"],
        "camera_trajectory_count": quality["camera_trajectory_count"],
        "bbox_size_m": quality["bbox_size_m"],
        "inspectable_map_available": quality["inspectable_map_available"],
        "truth_boundary": quality["truth_boundary"],
    }


def _load_map_summary(map_dir: Path) -> dict[str, object]:
    quality_path = map_dir / "map_quality.json"
    if not quality_path.is_file():
        return {"path": str(map_dir), "available": False, "missing": str(quality_path)}
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    return {
        "path": str(map_dir),
        "available": True,
        "fused_point_count": quality.get("fused_point_count", 0),
        "occupied_voxel_count": quality.get("occupied_voxel_count", 0),
        "observed_mesh_triangle_count": quality.get("observed_mesh_triangle_count", 0),
        "camera_trajectory_count": quality.get("camera_trajectory_count", 0),
        "bbox_size_m": quality.get("bbox_size_m"),
        "inspectable_map_available": quality.get("inspectable_map_available", False),
        "truth_boundary": quality.get("truth_boundary", {}),
    }


def _comparison_markdown(payload: dict[str, object]) -> str:
    less_collapsed = payload["less_collapsed_than_previous_by_bbox"]
    return "\n".join(
        [
            "# ViPE Map Comparison",
            "",
            f"- Ground truth available: {payload['ground_truth_available']}",
            f"- Physical accuracy claim: {payload['physical_accuracy_claim']}",
            f"- Less collapsed by bbox heuristic: {less_collapsed}",
            "",
            "This comparison is an artifact-shape diagnostic only. It does not prove physical "
            "accuracy or training quality.",
            "",
        ]
    )


def _less_collapsed_by_bbox(value: object, previous: object) -> bool | None:
    if (
        not isinstance(value, list)
        or not isinstance(previous, list)
        or len(value) != 3
        or len(previous) != 3
    ):
        return None
    size = np.asarray(value, dtype=np.float32)
    prev = np.asarray(previous, dtype=np.float32)
    if not np.all(np.isfinite(size)) or not np.all(np.isfinite(prev)):
        return None
    volume = float(np.prod(np.maximum(size, 0.0)))
    prev_volume = float(np.prod(np.maximum(prev, 0.0)))
    return bool(volume > prev_volume * 1.25 and float(size.min()) > float(prev.min()) * 0.8)


__all__ = ["write_vipe_comparison"]
