"""Room-walk run diagnostics and Markdown report."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from atlas3r.offline.best_map_selection import BestMapSelectionResult
from atlas3r.offline.camera_scale_ledger import CameraScaleLedgerResult
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult, FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import write_json


def write_room_walk_diagnostics(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    optimizer: MapConsistencyOptimizerResult,
    best_map: BestMapSelectionResult,
    ledgers: CameraScaleLedgerResult,
    soft_metric_ledger: dict[str, object],
    classical_result: object | None = None,
    classical_alignment: object | None = None,
    classical_comparison: object | None = None,
) -> tuple[str, str]:
    root = Path(run_dir)
    payload = _payload(
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        disagreement=disagreement,
        optimizer=optimizer,
        best_map=best_map,
        ledgers=ledgers,
        soft_metric_ledger=soft_metric_ledger,
        classical_result=classical_result,
        classical_alignment=classical_alignment,
        classical_comparison=classical_comparison,
    )
    write_json(root / "diagnostics" / "room_walk_001_diagnostics.json", payload)
    (root / "room_walk_001_report.md").write_text(_markdown(payload), encoding="utf-8")
    return "diagnostics/room_walk_001_diagnostics.json", "room_walk_001_report.md"


def _payload(
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    optimizer: MapConsistencyOptimizerResult,
    best_map: BestMapSelectionResult,
    ledgers: CameraScaleLedgerResult,
    soft_metric_ledger: dict[str, object],
    classical_result: object | None,
    classical_alignment: object | None,
    classical_comparison: object | None,
) -> dict[str, object]:
    bbox_min = best_map.bbox_world_min_m
    bbox_max = best_map.bbox_world_max_m
    bbox_size = (
        None
        if bbox_min is None or bbox_max is None
        else [float(bbox_max[index] - bbox_min[index]) for index in range(3)]
    )
    trajectory_bbox = _trajectory_bbox(best_map)
    return {
        "format_name": "atlas3r_room_walk_diagnostics",
        "format_version": 1,
        "input_path": input_path,
        "frame_count": len(frame_cache.records),
        "decoded_frame_dimensions": _dimensions(frame_cache.records),
        "keyframe_ids": [item.frame_id for item in keyframes],
        "frame_quality": _quality_summary(frame_cache.records),
        "exif_metadata_summary": _compact_metadata_summary(frame_cache.metadata_summary),
        "teacher_status": {
            "vggt_available": bool(proposal_cache.vggt_depth_records),
            "depth_pro_available": bool(proposal_cache.depth_pro_depth_records),
        },
        "vggt": {
            "camera_proposals": len(proposal_cache.vggt_camera_records),
            "depth_proposals": len(proposal_cache.vggt_depth_records),
            "window_count": len(proposal_cache.vggt_window_records),
            "stitching": proposal_cache.vggt_metadata.get("stitching", {}),
        },
        "depth_pro": {
            "camera_proposals": len(proposal_cache.depth_pro_camera_records),
            "depth_proposals": len(proposal_cache.depth_pro_depth_records),
            "status": "available" if proposal_cache.depth_pro_depth_records else "unavailable",
            "metadata": _compact_depth_pro_metadata(proposal_cache.depth_pro_metadata),
        },
        "disagreement_stats": {} if disagreement is None else disagreement.summary,
        "optimizer": {
            "status": optimizer.status,
            "before_metrics": optimizer.before_metrics,
            "after_metrics": optimizer.after_metrics,
            "improvement_passed": optimizer.improvement_passed,
        },
        "selected_best_map_source": best_map.selected_source,
        "world_map_best": {
            "status": best_map.status,
            "point_count": best_map.point_count,
            "occupied_voxel_count": best_map.occupied_voxel_count,
            "observed_mesh_triangle_count": best_map.mesh_triangle_count,
            "bbox_min_m": bbox_min,
            "bbox_max_m": bbox_max,
            "bbox_size_m": bbox_size,
            "camera_trajectory_count": best_map.trajectory_count,
            "camera_trajectory_bbox": trajectory_bbox,
            "map_density_points_per_voxel": (
                float(best_map.point_count / best_map.occupied_voxel_count)
                if best_map.occupied_voxel_count
                else 0.0
            ),
            "connected_component_count": (
                None
                if best_map.cleanup is None
                else best_map.cleanup.get("connected_component_count")
            ),
            "cleanup": best_map.cleanup,
        },
        "scale": {
            "scale_mode": ledgers.selected_scale_mode,
            "scale_confidence": str(
                soft_metric_ledger.get("scale_confidence", ledgers.scale_confidence)
            ),
            "soft_metric_ledger": soft_metric_ledger,
        },
        "classical_geometry_witness": _classical_payload(
            classical_result, classical_alignment, classical_comparison
        ),
        "physical_accuracy_claim": False,
        "training_quality_claim": False,
        "files_to_open": [
            "world_map_best/fused_points.ply",
            "world_map_best/observed_voxel_mesh.ply",
            "world_map_best/topdown_preview.svg",
            "world_map_best/camera_trajectory.json",
            "world_map_best/map_quality.md",
        ],
    }


def _dimensions(records: tuple[FrameRecord, ...]) -> dict[str, object]:
    values = sorted({(record.width, record.height) for record in records})
    return {"unique_width_height": [[w, h] for w, h in values]}


def _quality_summary(records: tuple[FrameRecord, ...]) -> dict[str, object]:
    blur = [record.quality.blur_score for record in records]
    exposure = [record.quality.exposure_score for record in records]
    change = [record.quality.visual_change_score for record in records]
    return {
        "blur": _stats(blur),
        "exposure": _stats(exposure),
        "visual_change": _stats(change),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "mean": mean(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _compact_metadata_summary(summary: dict[str, object]) -> dict[str, object]:
    records = summary.get("records")
    compact = {key: value for key, value in summary.items() if key != "records"}
    compact["record_count"] = len(records) if isinstance(records, list) else 0
    compact["records_path"] = "frames/metadata_summary.json"
    return compact


def _compact_depth_pro_metadata(metadata: dict[str, object]) -> dict[str, object]:
    frame_ids = metadata.get("frame_ids")
    output_shapes = metadata.get("output_shapes")
    return {
        "teacher_name": metadata.get("teacher_name"),
        "model_source": metadata.get("model_source"),
        "checkpoint": metadata.get("checkpoint"),
        "device": metadata.get("device"),
        "image_size": metadata.get("image_size"),
        "runtime_ms": metadata.get("runtime_ms"),
        "frame_count": len(frame_ids) if isinstance(frame_ids, list) else 0,
        "output_shape_count": len(output_shapes) if isinstance(output_shapes, dict) else 0,
        "coordinate_convention": metadata.get("coordinate_convention"),
    }


def _trajectory_bbox(best_map: BestMapSelectionResult) -> dict[str, object]:
    path = best_map.camera_trajectory_path
    if path is None:
        return {"available": False}
    return {
        "available": True,
        "note": "camera trajectory bbox is inspectable in world_map_best/camera_trajectory.json",
    }


def _markdown(payload: dict[str, object]) -> str:
    best = payload["world_map_best"]
    scale = payload["scale"]
    if not isinstance(best, dict) or not isinstance(scale, dict):
        raise ValueError("invalid room diagnostics payload")
    files = payload["files_to_open"]
    if not isinstance(files, list):
        files = []
    keyframe_ids = payload["keyframe_ids"]
    keyframe_count = len(keyframe_ids) if isinstance(keyframe_ids, list) else 0
    vggt = payload["vggt"]
    depth_pro = payload["depth_pro"]
    optimizer = payload["optimizer"]
    classical = payload["classical_geometry_witness"]
    if (
        not isinstance(vggt, dict)
        or not isinstance(depth_pro, dict)
        or not isinstance(optimizer, dict)
    ):
        raise ValueError("invalid room diagnostics payload")
    if not isinstance(classical, dict):
        classical = {}
    classical_alignment = classical.get("alignment", {})
    classical_comparison = classical.get("comparison", {})
    if not isinstance(classical_alignment, dict):
        classical_alignment = {}
    if not isinstance(classical_comparison, dict):
        classical_comparison = {}
    stitching = vggt.get("stitching", {})
    if not isinstance(stitching, dict):
        stitching = {}
    before = optimizer.get("before_metrics", {})
    after = optimizer.get("after_metrics", {})
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    lines = [
        "# Room Walk 001 Report",
        "",
        f"- Input path: {payload['input_path']}",
        f"- Frames decoded: {payload['frame_count']}",
        f"- Keyframes selected: {keyframe_count}",
        f"- VGGT proposals: cameras={vggt.get('camera_proposals')}, "
        f"depths={vggt.get('depth_proposals')}, windows={vggt.get('window_count')}",
        f"- VGGT stitching: mode={stitching.get('stitch_mode')}, "
        f"accepted_edges={stitching.get('accepted_edge_count')}, "
        f"rejected_edges={stitching.get('rejected_edge_count')}, "
        f"overlap_center_rmse_m={stitching.get('overlap_center_rmse_m')}",
        f"- Depth Pro proposals: cameras={depth_pro.get('camera_proposals')}, "
        f"depths={depth_pro.get('depth_proposals')}, status={depth_pro.get('status')}",
        f"- Optimizer status: {optimizer.get('status')}, "
        f"improvement_passed={optimizer.get('improvement_passed')}",
        f"- Optimizer rel diff mean: {before.get('vggt_depthpro_rel_diff_mean')} -> "
        f"{after.get('vggt_depthpro_rel_diff_mean')}",
        f"- Optimizer rel diff p95: {before.get('vggt_depthpro_rel_diff_p95')} -> "
        f"{after.get('vggt_depthpro_rel_diff_p95')}",
        f"- Optimizer abs diff mean m: {before.get('vggt_depthpro_abs_diff_mean_m')} -> "
        f"{after.get('vggt_depthpro_abs_diff_mean_m')}",
        f"- Optimizer abs diff p95 m: {before.get('vggt_depthpro_abs_diff_p95_m')} -> "
        f"{after.get('vggt_depthpro_abs_diff_p95_m')}",
        f"- Optimizer projection residual mean m: "
        f"{before.get('cross_view_depth_residual_mean_m')} -> "
        f"{after.get('cross_view_depth_residual_mean_m')}",
        f"- Optimizer projection residual p95 m: "
        f"{before.get('cross_view_depth_residual_p95_m')} -> "
        f"{after.get('cross_view_depth_residual_p95_m')}",
        f"- Selected best map source: {payload['selected_best_map_source']}",
        f"- COLMAP/GLOMAP status: {classical.get('status')}, "
        f"registered_images={classical.get('registered_image_count')}, "
        f"sparse_points={classical.get('sparse_point_count')}",
        f"- Classical alignment: status={classical_alignment.get('status')}, "
        f"common_frames={classical_alignment.get('common_frame_count')}, "
        f"rmse_m={classical_alignment.get('camera_center_rmse_m')}, "
        f"p95_m={classical_alignment.get('camera_center_p95_m')}",
        "- Classical map agreement: "
        f"trajectory={classical_comparison.get('trajectory_agreement_status')}, "
        f"map={classical_comparison.get('map_agreement_status')}",
        f"- Best map points: {best['point_count']}",
        f"- Best map occupied voxels: {best['occupied_voxel_count']}",
        f"- Best map observed mesh triangles: {best['observed_mesh_triangle_count']}",
        f"- Best map bbox size m: {best['bbox_size_m']}",
        f"- Camera trajectory count: {best['camera_trajectory_count']}",
        f"- Scale mode: {scale['scale_mode']}",
        f"- Scale confidence: {scale['scale_confidence']}",
        "- Physical accuracy claim: false",
        "- Training-quality claim: false",
        "",
        "Open these files first:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in files)
    lines.extend(
        [
            "",
            "This is an unanchored soft-metric teacher-consensus map. It is observed-only, "
            "does not fill hidden space, and is not physical ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return 0
