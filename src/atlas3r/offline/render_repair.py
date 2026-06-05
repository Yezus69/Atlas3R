"""Render-and-repair diagnostic artifact skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.geometry_preview import GeometryPreviewResult
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import FailurePoint, write_json


@dataclass(frozen=True)
class RenderRepairResult:
    status: str
    diagnostics_path: str


def write_render_repair_diagnostics(
    run_dir: str | Path,
    *,
    geometry: GeometryPreviewResult,
    keyframes: tuple[KeyframeRecord, ...],
    failure_points: list[FailurePoint],
) -> RenderRepairResult:
    if geometry.point_count == 0:
        status = "unavailable"
        why = "geometry preview is empty, so render/projection diagnostics cannot run"
        failure_points.append(
            FailurePoint(
                module="render_repair_diagnostics",
                code="geometry_missing",
                severity="warning",
                status=status,
                why=why,
                input_missing="geometry preview points",
                future_module="geometry lifter with real teacher proposals",
                artifact_path="diagnostics/render_repair_diagnostics.json",
            )
        )
        coverage = 0.0
        projection_count = 0
    else:
        status = "partial"
        why = "projected coverage placeholder only; no renderer or optimizer has run"
        coverage = min(1.0, float(geometry.point_count) / max(1.0, float(len(keyframes) * 100)))
        projection_count = min(geometry.point_count, len(keyframes) * 100)
    payload = {
        "status": status,
        "why": why,
        "geometry_point_count": geometry.point_count,
        "keyframe_count": len(keyframes),
        "projection_sample_count": projection_count,
        "coverage_placeholder": coverage,
        "render_mismatch": None,
        "repair_hooks": {
            "pose_repair_needed": True,
            "depth_repair_needed": True,
            "object_track_repair_needed": True,
            "scale_repair_needed": True,
        },
    }
    write_json(Path(run_dir) / "diagnostics" / "render_repair_diagnostics.json", payload)
    return RenderRepairResult(
        status=status, diagnostics_path="diagnostics/render_repair_diagnostics.json"
    )
