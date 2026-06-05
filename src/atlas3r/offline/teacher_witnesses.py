"""Teacher witness status artifacts for offline runs."""

from __future__ import annotations

from pathlib import Path

from atlas3r.offline.run_manifest import FailurePoint, write_json
from atlas3r.teachers.base import AdapterStatus
from atlas3r.teachers.registry import list_teacher_statuses


def write_teacher_statuses(
    run_dir: str | Path, failure_points: list[FailurePoint]
) -> tuple[AdapterStatus, ...]:
    statuses = list_teacher_statuses()
    rows: list[dict[str, object]] = []
    for status in statuses:
        row = status.to_dict()
        row["status"] = "available" if status.available else "unavailable"
        row["proposal_stream_path"] = f"proposals/{status.name}_proposals.jsonl"
        rows.append(row)
        if not status.available:
            failure_points.append(
                FailurePoint(
                    module="teacher_witness_layer",
                    code=f"{status.name}_unavailable",
                    severity="warning",
                    status="unavailable",
                    why=status.reason,
                    dependency_missing=status.display_name,
                    future_module=status.install_hint,
                    artifact_path="teachers/teacher_status.json",
                )
            )
    payload = {
        "status": "partial",
        "format_name": "atlas3r_teacher_witness_status",
        "format_version": 1,
        "teachers": rows,
    }
    write_json(Path(run_dir) / "teachers" / "teacher_status.json", payload)
    return statuses
