"""Teacher witness status artifacts for offline runs."""

from __future__ import annotations

from pathlib import Path

from atlas3r.offline.depth_pro_witness import DepthProWitnessResult
from atlas3r.offline.run_manifest import FailurePoint, write_json
from atlas3r.offline.vggt_witness import VggtWitnessResult
from atlas3r.teachers.base import AdapterStatus
from atlas3r.teachers.registry import list_teacher_statuses


def write_teacher_statuses(
    run_dir: str | Path,
    failure_points: list[FailurePoint],
    *,
    vggt_result: VggtWitnessResult | None = None,
    depth_pro_result: DepthProWitnessResult | None = None,
) -> tuple[AdapterStatus, ...]:
    statuses = list_teacher_statuses()
    rows: list[dict[str, object]] = []
    for status in statuses:
        row = status.to_dict()
        row["status"] = "available" if status.available else "unavailable"
        row["proposal_stream_path"] = f"proposals/{status.name}_proposals.jsonl"
        if status.name == "vggt" and vggt_result is not None:
            row["available"] = vggt_result.available
            row["status"] = vggt_result.runtime_status
            row["reason"] = vggt_result.reason
            row["install_hint"] = vggt_result.install_hint
            row["proposal_stream_path"] = (
                "proposals/vggt_cameras.jsonl" if vggt_result.has_geometry else None
            )
            row["proposal_counts"] = {
                "cameras": len(vggt_result.camera_records),
                "depths": len(vggt_result.depth_records),
                "windows": len(vggt_result.window_records),
            }
            row["model_metadata"] = vggt_result.metadata
            row["replay_source"] = vggt_result.replay_source
        if status.name == "depth_pro" and depth_pro_result is not None:
            row["available"] = depth_pro_result.available
            row["status"] = depth_pro_result.runtime_status
            row["reason"] = depth_pro_result.reason
            row["install_hint"] = depth_pro_result.install_hint
            row["proposal_stream_path"] = (
                "proposals/depth_pro_depths.npz" if depth_pro_result.has_depth else None
            )
            row["proposal_counts"] = {
                "cameras": len(depth_pro_result.camera_records),
                "depths": len(depth_pro_result.depth_records),
                "frames": len(depth_pro_result.frame_records),
            }
            row["model_metadata"] = depth_pro_result.metadata
            row["replay_source"] = depth_pro_result.replay_source
        rows.append(row)
        already_reported = (
            status.name == "vggt"
            and vggt_result is not None
            and vggt_result.failure_code is not None
        ) or (
            status.name == "depth_pro"
            and depth_pro_result is not None
            and depth_pro_result.failure_code is not None
        )
        row_available = bool(row["available"])
        if not row_available and not already_reported:
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
