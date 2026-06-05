"""Quality report writer for the offline vertical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import CameraScaleLedgerResult
from atlas3r.offline.geometry_preview import GeometryPreviewResult
from atlas3r.offline.object_ledger import ObjectLedgerResult
from atlas3r.offline.render_repair import RenderRepairResult
from atlas3r.offline.run_manifest import FailurePoint, write_json
from atlas3r.offline.training_cache import TrainingCacheResult
from atlas3r.teachers.base import AdapterStatus


@dataclass(frozen=True)
class QualityReportResult:
    json_path: str
    markdown_path: str


def write_quality_report(
    run_dir: str | Path,
    *,
    teacher_statuses: tuple[AdapterStatus, ...],
    ledgers: CameraScaleLedgerResult,
    geometry: GeometryPreviewResult,
    objects: ObjectLedgerResult,
    render: RenderRepairResult,
    training: TrainingCacheResult,
    failure_points: list[FailurePoint],
) -> QualityReportResult:
    teachers = [
        {
            "name": status.name,
            "available": status.available,
            "reason": status.reason,
            "install_hint": status.install_hint,
        }
        for status in teacher_statuses
    ]
    payload = {
        "status": "partial",
        "teacher_availability": teachers,
        "teacher_disagreement": "unavailable until at least two proposal streams exist",
        "scale_source": ledgers.scale_source,
        "physical_accuracy": False,
        "physical_accuracy_reason": "no measured scale anchor or evaluation report",
        "geometry": {
            "status": geometry.status,
            "point_count": geometry.point_count,
            "observed_only": geometry.observed_only,
            "predicted_completion": geometry.predicted_completion,
            "measured_geometry": geometry.measured_geometry,
            "metric_scale_source": geometry.metric_scale_source,
        },
        "object_tracking_status": objects.status,
        "render_diagnostic_status": render.status,
        "training_cache": {
            "status": training.status,
            "usable_for_training": training.usable_for_training,
        },
        "failure_points": [failure.to_dict() for failure in failure_points],
    }
    root = Path(run_dir)
    write_json(root / "quality_report.json", payload)
    markdown = _markdown_report(payload)
    (root / "quality_report.md").write_text(markdown, encoding="utf-8")
    return QualityReportResult("quality_report.json", "quality_report.md")


def _markdown_report(payload: dict[str, object]) -> str:
    geometry = payload["geometry"]
    if not isinstance(geometry, dict):
        raise ValueError("geometry payload must be a dict")
    training_cache = payload["training_cache"]
    if not isinstance(training_cache, dict):
        raise ValueError("training payload must be a dict")
    failure_points = payload["failure_points"]
    if not isinstance(failure_points, list):
        raise ValueError("failure_points payload must be a list")
    lines = [
        "# Offline World Builder Quality Report",
        "",
        f"- Status: {payload['status']}",
        f"- Scale source: {payload['scale_source']}",
        f"- Physically accurate: {payload['physical_accuracy']}",
        f"- Geometry points: {geometry['point_count']}",
        f"- Geometry measured: {geometry['measured_geometry']}",
        f"- Object tracking: {payload['object_tracking_status']}",
        f"- Render diagnostics: {payload['render_diagnostic_status']}",
        f"- Training usable: {training_cache['usable_for_training']}",
        f"- Failure points: {len(failure_points)}",
        "",
        "This report is a tracer report, not an accuracy report.",
    ]
    return "\n".join(lines) + "\n"
