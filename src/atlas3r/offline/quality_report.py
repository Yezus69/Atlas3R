"""Quality report writer for the offline vertical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.best_map_selection import BestMapSelectionResult
from atlas3r.offline.camera_scale_ledger import CameraScaleLedgerResult
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.fused_world_map import FusedWorldMapResult
from atlas3r.offline.geometry_preview import GeometryPreviewResult
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.object_ledger import ObjectLedgerResult
from atlas3r.offline.proposal_cache import ProposalCacheResult
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
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    ledgers: CameraScaleLedgerResult,
    geometry: GeometryPreviewResult,
    world_map: FusedWorldMapResult,
    objects: ObjectLedgerResult,
    render: RenderRepairResult,
    training: TrainingCacheResult,
    failure_points: list[FailurePoint],
    optimizer: MapConsistencyOptimizerResult | None = None,
    best_map: BestMapSelectionResult | None = None,
    classical_result: object | None = None,
    classical_alignment: object | None = None,
    classical_comparison: object | None = None,
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
        "teacher_proposal_counts": {
            "vggt_cameras": len(proposal_cache.vggt_camera_records),
            "vggt_depths": len(proposal_cache.vggt_depth_records),
            "depth_pro_cameras": len(proposal_cache.depth_pro_camera_records),
            "depth_pro_depths": len(proposal_cache.depth_pro_depth_records),
        },
        "depth_pro": {
            "status": "available" if proposal_cache.depth_pro_depth_records else "unavailable",
            "depth_proposals": len(proposal_cache.depth_pro_depth_records),
            "camera_proposals": len(proposal_cache.depth_pro_camera_records),
            "global_pose_available": False,
        },
        "teacher_disagreement": {
            "status": "unavailable" if disagreement is None else disagreement.status,
            "path": None if disagreement is None else disagreement.json_path,
            "maps_path": None if disagreement is None else disagreement.maps_npz_path,
            "summary": {} if disagreement is None else disagreement.summary,
            "diagnostic_only": True,
            "optimized_consensus": False,
        },
        "consensus_preview": {
            "status": "unavailable" if disagreement is None else disagreement.consensus_status,
            "path": None if disagreement is None else disagreement.consensus_npz_path,
            "diagnostic_only": True,
            "optimized_consensus": False,
        },
        "scale_source": ledgers.scale_source,
        "physical_accuracy": False,
        "physical_accuracy_reason": (
            "no measured scale anchor, calibration target, measured depth, external pose, "
            "or named evaluation report"
        ),
        "geometry": {
            "status": geometry.status,
            "point_count": geometry.point_count,
            "source_teacher": geometry.source_teacher,
            "teacher_proposed_geometry_available": bool(
                geometry.source_teacher in {"vggt", "vggt_pose_consensus_depth_diagnostic"}
                and geometry.point_count > 0
            ),
            "observed_only": geometry.observed_only,
            "predicted_completion": geometry.predicted_completion,
            "measured_geometry": geometry.measured_geometry,
            "metric_scale_source": geometry.metric_scale_source,
            "valid_point_ratio": geometry.valid_point_ratio,
            "per_frame_point_counts": geometry.per_frame_point_counts,
        },
        "world_map": {
            "status": world_map.status,
            "depth_source": world_map.depth_source,
            "point_count": world_map.point_count,
            "occupied_voxel_count": world_map.occupied_voxel_count,
            "observed_mesh_vertex_count": world_map.mesh_vertex_count,
            "observed_mesh_triangle_count": world_map.mesh_triangle_count,
            "camera_trajectory_count": world_map.trajectory_count,
            "inspectable_map_available": world_map.inspectable_map_available,
            "manifest": world_map.manifest_path,
            "fused_points_ply": world_map.fused_points_ply_path,
            "occupancy_grid": world_map.occupancy_grid_npz_path,
            "observed_voxel_mesh": world_map.observed_voxel_mesh_ply_path,
            "map_quality": world_map.map_quality_json_path,
            "physical_accuracy_claim": False,
            "training_quality_claim": False,
        },
        "world_map_best": None
        if best_map is None
        else {
            "status": best_map.status,
            "selected_source": best_map.selected_source,
            "point_count": best_map.point_count,
            "occupied_voxel_count": best_map.occupied_voxel_count,
            "observed_mesh_triangle_count": best_map.mesh_triangle_count,
            "camera_trajectory_count": best_map.trajectory_count,
            "manifest": best_map.manifest_path,
            "fused_points_ply": best_map.fused_points_ply_path,
            "observed_voxel_mesh": best_map.observed_voxel_mesh_ply_path,
            "topdown_preview": best_map.topdown_preview_path,
            "map_quality": best_map.map_quality_json_path,
            "physical_accuracy_claim": False,
            "training_quality_claim": False,
        },
        "map_consistency_optimizer": {
            "status": "disabled" if optimizer is None else optimizer.status,
            "manifest": None if optimizer is None else optimizer.manifest_path,
            "before_metrics": None if optimizer is None else optimizer.before_metrics,
            "after_metrics": None if optimizer is None else optimizer.after_metrics,
            "improvement_passed": False if optimizer is None else optimizer.improvement_passed,
            "improvement_summary": None if optimizer is None else optimizer.improvement_summary,
            "optimized_world_map": None
            if optimizer is None
            else {
                "status": optimizer.optimized_world_map.status,
                "manifest": optimizer.optimized_world_map.manifest_path,
                "point_count": optimizer.optimized_world_map.point_count,
                "occupied_voxel_count": optimizer.optimized_world_map.occupied_voxel_count,
                "observed_mesh_triangle_count": optimizer.optimized_world_map.mesh_triangle_count,
                "inspectable_map_available": (
                    optimizer.optimized_world_map.inspectable_map_available
                ),
            },
            "physical_accuracy_claim": False,
            "training_quality_claim": False,
        },
        "classical_geometry_witness": _classical_payload(
            classical_result, classical_alignment, classical_comparison
        ),
        "missing_blockers": [
            "scale_anchor",
            "render_repair_optimizer",
            "named_evaluation_report",
        ],
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
    optimizer = payload["map_consistency_optimizer"]
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer payload must be a dict")
    best_map = payload.get("world_map_best")
    best_status = "disabled" if best_map is None else _world_map_best_status(best_map)
    failure_points = payload["failure_points"]
    if not isinstance(failure_points, list):
        raise ValueError("failure_points payload must be a list")
    lines = [
        "# Offline World Builder Quality Report",
        "",
        f"- Status: {payload['status']}",
        f"- Scale source: {payload['scale_source']}",
        f"- Physically accurate: {payload['physical_accuracy']}",
        f"- Physical accuracy reason: {payload['physical_accuracy_reason']}",
        f"- Geometry points: {geometry['point_count']}",
        f"- Geometry source: {geometry['source_teacher']}",
        f"- Teacher-proposed geometry: {geometry['teacher_proposed_geometry_available']}",
        f"- Geometry measured: {geometry['measured_geometry']}",
        f"- Fused world map: {_world_map_status(payload)}",
        f"- Best world map: {best_status}",
        f"- Map optimizer: {optimizer.get('status', 'disabled')}",
        f"- Optimizer improvement passed: {optimizer.get('improvement_passed', False)}",
        f"- Classical geometry witness: {_classical_status(payload)}",
        f"- Teacher disagreement: {_disagreement_status(payload)}",
        f"- Consensus preview: {_consensus_status(payload)}",
        f"- Object tracking: {payload['object_tracking_status']}",
        f"- Render diagnostics: {payload['render_diagnostic_status']}",
        f"- Training usable: {training_cache['usable_for_training']}",
        "- Missing blockers: scale anchor, render repair optimizer, named evaluation report",
        f"- Failure points: {len(failure_points)}",
        "",
        "This report is a tracer report, not an accuracy report. VGGT and Depth Pro output, "
        "when present, are teacher-proposed geometry rather than measured geometry or "
        "training-quality labels.",
    ]
    return "\n".join(lines) + "\n"


def _disagreement_status(payload: dict[str, object]) -> str:
    disagreement = payload["teacher_disagreement"]
    if not isinstance(disagreement, dict):
        return "unavailable"
    return str(disagreement.get("status", "unavailable"))


def _consensus_status(payload: dict[str, object]) -> str:
    consensus = payload["consensus_preview"]
    if not isinstance(consensus, dict):
        return "unavailable"
    return str(consensus.get("status", "unavailable"))


def _world_map_status(payload: dict[str, object]) -> str:
    world_map = payload["world_map"]
    if not isinstance(world_map, dict):
        return "unavailable"
    status = str(world_map.get("status", "unavailable"))
    points = int(world_map.get("point_count", 0))
    voxels = int(world_map.get("occupied_voxel_count", 0))
    inspectable = bool(world_map.get("inspectable_map_available", False))
    return f"{status}, points={points}, voxels={voxels}, inspectable={inspectable}"


def _world_map_best_status(value: object) -> str:
    if not isinstance(value, dict):
        return "unavailable"
    status = str(value.get("status", "unavailable"))
    source = str(value.get("selected_source", "none"))
    points = int(value.get("point_count", 0))
    voxels = int(value.get("occupied_voxel_count", 0))
    triangles = int(value.get("observed_mesh_triangle_count", 0))
    return f"{status}, source={source}, points={points}, voxels={voxels}, triangles={triangles}"


def _classical_payload(
    classical_result: object | None,
    classical_alignment: object | None,
    classical_comparison: object | None,
) -> dict[str, object]:
    result = _object_to_dict(classical_result)
    alignment = _object_to_dict(classical_alignment)
    comparison = _object_to_dict(classical_comparison)
    return {
        "status": result.get("status", "disabled"),
        "source": result.get("source", "none"),
        "available": bool(result.get("available", False)),
        "registered_image_count": _int_value(result.get("registered_image_count")),
        "sparse_point_count": _int_value(result.get("sparse_point_count")),
        "stage_failed": result.get("stage_failed"),
        "likely_reason": result.get("likely_reason"),
        "alignment": {
            "status": alignment.get("status", "unavailable"),
            "common_frame_count": _int_value(alignment.get("common_frame_count")),
            "camera_center_rmse_m": alignment.get("camera_center_rmse_m"),
            "camera_center_p95_m": alignment.get("camera_center_p95_m"),
            "sim3_scale": alignment.get("sim3_scale"),
        },
        "comparison": {
            "status": comparison.get("status", "unavailable"),
            "trajectory_agreement_status": comparison.get(
                "trajectory_agreement_status", "unavailable"
            ),
            "map_agreement_status": comparison.get("map_agreement_status", "unavailable"),
            "world_map_best": _comparison_best_map(comparison),
        },
        "physical_accuracy_claim": False,
        "training_quality_claim": False,
    }


def _object_to_dict(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return value if isinstance(value, dict) else {}


def _comparison_best_map(comparison: dict[str, object]) -> dict[str, object]:
    maps = comparison.get("maps", {})
    if not isinstance(maps, dict):
        return {}
    best = maps.get("world_map_best", {})
    return best if isinstance(best, dict) else {}


def _classical_status(payload: dict[str, object]) -> str:
    classical = payload.get("classical_geometry_witness", {})
    if not isinstance(classical, dict):
        return "disabled"
    status = classical.get("status", "disabled")
    registered = classical.get("registered_image_count", 0)
    points = classical.get("sparse_point_count", 0)
    alignment = classical.get("alignment", {})
    common = alignment.get("common_frame_count", 0) if isinstance(alignment, dict) else 0
    return f"{status}, registered={registered}, points={points}, common_frames={common}"


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return 0
