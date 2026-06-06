"""Core math and artifact helpers for map consistency optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.fused_world_map import FusedWorldMapResult
from atlas3r.offline.projection_diagnostics import (
    ProjectionDiagnostics,
    compute_projection_consistency,
)
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import write_json

_EPS = 1e-6
_ABS_HIGH_THRESHOLD_M = 0.5
_REL_HIGH_THRESHOLD = 0.25


@dataclass(frozen=True)
class MapConsistencyOptimizerOptions:
    enabled: bool = False
    export_optimized_world_map: bool = False
    max_iterations: int = 5
    depth_scale_min: float = 0.5
    depth_scale_max: float = 2.0
    depth_bias_max_m: float = 1.0
    enable_intrinsics_scale: bool = False
    focal_scale_min: float = 0.8
    focal_scale_max: float = 1.25
    cross_view_pairs: int = 3
    min_overlap_pixels: int = 512
    min_improvement_ratio: float = 0.05


@dataclass(frozen=True)
class ScaleBiasEstimate:
    frame_id: int
    keyframe_id: int
    scale: float
    bias_m: float
    robust_loss_before: float
    robust_loss_after: float
    valid_overlap_pixels: int
    accepted: bool
    rejection_reason: str | None = None


def validate_options(options: MapConsistencyOptimizerOptions) -> None:
    if options.max_iterations <= 0:
        raise ValueError("optimizer max iterations must be positive")
    if options.depth_scale_min <= 0.0 or options.depth_scale_max < options.depth_scale_min:
        raise ValueError("optimizer depth scale bounds are invalid")
    if options.depth_bias_max_m < 0.0:
        raise ValueError("optimizer depth bias max must be non-negative")
    if options.focal_scale_min <= 0.0 or options.focal_scale_max < options.focal_scale_min:
        raise ValueError("optimizer focal scale bounds are invalid")
    if options.cross_view_pairs <= 0:
        raise ValueError("optimizer cross-view pairs must be positive")
    if options.min_overlap_pixels <= 0:
        raise ValueError("optimizer min overlap pixels must be positive")


def paired_records(
    proposal_cache: ProposalCacheResult,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    depth_pro = {int(str(row["frame_id"])): row for row in proposal_cache.depth_pro_depth_records}
    return [
        (row, depth_pro[int(str(row["frame_id"]))])
        for row in proposal_cache.vggt_depth_records
        if int(str(row["frame_id"])) in depth_pro
    ]


def fit_depth_scale_bias(
    vggt_depth: NDArray[np.float32],
    depth_pro_depth: NDArray[np.float32],
    valid_overlap: NDArray[np.bool_],
    *,
    options: MapConsistencyOptimizerOptions,
    frame_id: int = 0,
    keyframe_id: int = 0,
) -> ScaleBiasEstimate:
    valid = (
        valid_overlap
        & np.isfinite(vggt_depth)
        & np.isfinite(depth_pro_depth)
        & (vggt_depth > 0.0)
        & (depth_pro_depth > 0.0)
    )
    count = int(valid.sum())
    before_loss = robust_log_loss(vggt_depth, depth_pro_depth, valid)
    if count < options.min_overlap_pixels:
        return ScaleBiasEstimate(
            frame_id,
            keyframe_id,
            1.0,
            0.0,
            before_loss,
            before_loss,
            count,
            False,
            "valid_overlap_below_minimum",
        )
    x = depth_pro_depth[valid].astype(np.float64)
    y = vggt_depth[valid].astype(np.float64)
    weights = np.ones_like(x)
    scale = 1.0
    bias = 0.0
    for _ in range(max(1, options.max_iterations)):
        scale, bias = weighted_lstsq_scale_bias(x, y, weights, options)
        aligned = scale * x + bias
        positive = aligned > _EPS
        if int(positive.sum()) < options.min_overlap_pixels:
            break
        residual = np.abs(np.log(np.maximum(y[positive], _EPS)) - np.log(aligned[positive]))
        cutoff = max(float(np.percentile(residual, 70)), 1e-3)
        next_weights = np.zeros_like(weights)
        next_weights[positive] = 1.0 / (1.0 + (residual / cutoff) ** 2)
        weights = np.maximum(next_weights, 1e-3)
    aligned_depth = (float(scale) * depth_pro_depth + float(bias)).astype(np.float32)
    after_loss = robust_log_loss(vggt_depth, aligned_depth, valid & (aligned_depth > 0.0))
    accepted = bool(np.isfinite(after_loss) and after_loss <= before_loss + 1e-8)
    return ScaleBiasEstimate(
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        scale=float(scale),
        bias_m=float(bias),
        robust_loss_before=before_loss,
        robust_loss_after=after_loss,
        valid_overlap_pixels=count,
        accepted=accepted,
        rejection_reason=None if accepted else "alignment_did_not_reduce_robust_log_loss",
    )


def build_optimized_consensus(
    root: Path,
    proposal_cache: ProposalCacheResult,
    *,
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    options: MapConsistencyOptimizerOptions,
    estimates: list[ScaleBiasEstimate],
    trace_rows: list[dict[str, object]],
) -> DisagreementResult:
    frame_ids: list[int] = []
    abs_maps: list[NDArray[np.float32]] = []
    rel_maps: list[NDArray[np.float32]] = []
    log_maps: list[NDArray[np.float32]] = []
    valid_maps: list[NDArray[np.float32]] = []
    high_maps: list[NDArray[np.float32]] = []
    depths: list[NDArray[np.float32]] = []
    confidences: list[NDArray[np.float32]] = []
    source_masks: list[NDArray[np.int16]] = []
    records: list[dict[str, object]] = []
    arrays: dict[str, NDArray[np.float32]] = {}
    summaries: list[dict[str, object]] = []
    for index, (vggt, depth_pro) in enumerate(pairs):
        frame_id = int(str(vggt["frame_id"]))
        keyframe_id = int(str(vggt["keyframe_index"]))
        vggt_depth, vggt_valid = depth_and_valid(vggt, proposal_cache)
        depth_pro_depth, depth_pro_valid = depth_and_valid(depth_pro, proposal_cache)
        target_shape = (int(depth_pro_depth.shape[0]), int(depth_pro_depth.shape[1]))
        vggt_depth = resize_nearest(vggt_depth, target_shape)
        vggt_valid = resize_nearest_bool(vggt_valid, target_shape)
        valid = overlap(vggt_depth, vggt_valid, depth_pro_depth, depth_pro_valid)
        estimate = fit_depth_scale_bias(
            vggt_depth,
            depth_pro_depth,
            valid,
            options=options,
            frame_id=frame_id,
            keyframe_id=keyframe_id,
        )
        estimates.append(estimate)
        aligned = aligned_depth(depth_pro_depth, estimate)
        abs_diff, rel_diff, log_diff, high = diff_maps(vggt_depth, aligned, valid)
        source_mask, consensus_depth, confidence = optimized_consensus_frame(
            vggt_depth,
            vggt_valid,
            aligned,
            depth_pro_valid & (aligned > 0.0),
            abs_diff,
            rel_diff,
            high,
        )
        frame_ids.append(frame_id)
        abs_maps.append(abs_diff)
        rel_maps.append(rel_diff)
        log_maps.append(log_diff)
        valid_maps.append(valid.astype(np.float32))
        high_maps.append(high.astype(np.float32))
        depths.append(consensus_depth)
        confidences.append(confidence)
        source_masks.append(source_mask)
        summaries.append(frame_metric_payload(frame_id, valid, abs_diff, rel_diff, log_diff, high))
        append_consensus_record(
            index,
            frame_id=frame_id,
            keyframe_id=keyframe_id,
            consensus_depth=consensus_depth,
            consensus_confidence=confidence,
            source_mask=source_mask,
            records=records,
            arrays=arrays,
        )
        trace_rows.append(estimate_row_for_trace(len(trace_rows), estimate))
    result = DisagreementResult(
        status="available" if records else "no_valid_overlap",
        json_path="optimizer/optimized_disagreement.json",
        maps_npz_path="optimizer/optimized_disagreement_maps.npz",
        consensus_npz_path="optimizer/optimized_consensus_preview.npz",
        summary=summary_from_maps(valid_maps, abs_maps, rel_maps, log_maps, high_maps),
        frame_summaries=tuple(summaries),
        valid_overlap_count=int(sum(int(item.sum()) for item in valid_maps)),
        consensus_status="optimized" if records else "unavailable",
        consensus_depth_records=tuple(records),
        consensus_depth_arrays=arrays,
    )
    write_optimized_disagreement_files(
        root,
        result,
        frame_ids,
        abs_maps,
        rel_maps,
        valid_maps,
        high_maps,
        depths,
        confidences,
        source_masks,
    )
    return result


def depth_metrics_from_records(
    proposal_cache: ProposalCacheResult,
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    estimates: list[ScaleBiasEstimate] | None,
) -> dict[str, object]:
    by_frame = {} if estimates is None else {item.frame_id: item for item in estimates}
    abs_values: list[NDArray[np.float32]] = []
    rel_values: list[NDArray[np.float32]] = []
    log_values: list[NDArray[np.float32]] = []
    high_values: list[NDArray[np.float32]] = []
    overlap_count = 0
    for vggt, depth_pro in pairs:
        frame_id = int(str(vggt["frame_id"]))
        vggt_depth, vggt_valid = depth_and_valid(vggt, proposal_cache)
        depth_pro_depth, depth_pro_valid = depth_and_valid(depth_pro, proposal_cache)
        target_shape = (int(depth_pro_depth.shape[0]), int(depth_pro_depth.shape[1]))
        vggt_depth = resize_nearest(vggt_depth, target_shape)
        vggt_valid = resize_nearest_bool(vggt_valid, target_shape)
        compare = (
            aligned_depth(depth_pro_depth, by_frame[frame_id])
            if frame_id in by_frame
            else depth_pro_depth
        )
        valid = overlap(vggt_depth, vggt_valid, compare, depth_pro_valid)
        abs_diff, rel_diff, log_diff, high = diff_maps(vggt_depth, compare, valid)
        overlap_count += int(valid.sum())
        if np.any(valid):
            abs_values.append(abs_diff[valid])
            rel_values.append(rel_diff[valid])
            log_values.append(log_diff[valid])
            high_values.append(high[valid].astype(np.float32))
    abs_all = concat(abs_values)
    rel_all = concat(rel_values)
    log_all = concat(log_values)
    high_all = concat(high_values)
    return {
        "vggt_depthpro_abs_diff_mean_m": mean(abs_all),
        "vggt_depthpro_abs_diff_p50_m": percentile(abs_all, 50),
        "vggt_depthpro_abs_diff_p95_m": percentile(abs_all, 95),
        "vggt_depthpro_rel_diff_mean": mean(rel_all),
        "vggt_depthpro_rel_diff_p95": percentile(rel_all, 95),
        "log_depth_diff_mean": mean(log_all),
        "high_disagreement_ratio": mean(high_all),
        "common_valid_overlap_pixels": overlap_count,
    }


def select_focal_scale(
    proposal_cache: ProposalCacheResult,
    optimized: DisagreementResult,
    options: MapConsistencyOptimizerOptions,
) -> float:
    if not options.enable_intrinsics_scale:
        return 1.0
    best_scale = 1.0
    best_loss = float("inf")
    for scale in np.linspace(options.focal_scale_min, options.focal_scale_max, 7):
        diagnostics = compute_projection_consistency(
            proposal_cache=proposal_cache,
            depth_records=optimized.consensus_depth_records,
            depth_arrays=optimized.consensus_depth_arrays,
            focal_scale=float(scale),
            max_neighbor_pairs=options.cross_view_pairs,
            min_overlap_pixels=options.min_overlap_pixels,
        )
        loss = diagnostics.residual_mean_m if diagnostics.projection_count else float("inf")
        if loss < best_loss:
            best_loss = loss
            best_scale = float(scale)
    return best_scale


def with_scaled_vggt_intrinsics(
    proposal_cache: ProposalCacheResult, focal_scale: float
) -> ProposalCacheResult:
    if abs(focal_scale - 1.0) < 1e-8:
        return proposal_cache
    cameras = []
    for record in proposal_cache.vggt_camera_records:
        row = dict(record)
        K = np.asarray(row["K"], dtype=np.float32).copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
        row["K"] = K.astype(float).tolist()
        row["intrinsics_optimizer_focal_scale"] = focal_scale
        cameras.append(row)
    return replace(proposal_cache, vggt_camera_records=tuple(cameras))


def with_map_counts(
    metrics: dict[str, object], world_map: FusedWorldMapResult, retained_ratio: float
) -> dict[str, object]:
    updated = dict(metrics)
    updated.update(
        {
            "cross_view_projection_count": 0,
            "cross_view_depth_residual_mean_m": 0.0,
            "cross_view_depth_residual_p95_m": 0.0,
            "fused_point_count": world_map.point_count,
            "occupied_voxel_count": world_map.occupied_voxel_count,
            "observed_mesh_triangle_count": world_map.mesh_triangle_count,
            "retained_point_ratio": retained_ratio,
            "rejected_low_confidence_ratio": world_map.rejected_low_confidence_ratio,
            "rejected_high_disagreement_ratio": world_map.rejected_high_disagreement_ratio,
            "inspectable_map_available": world_map.inspectable_map_available,
        }
    )
    return updated


def projection_metrics(diagnostics: ProjectionDiagnostics) -> dict[str, object]:
    return {
        "cross_view_projection_count": diagnostics.projection_count,
        "cross_view_depth_residual_mean_m": diagnostics.residual_mean_m,
        "cross_view_depth_residual_p95_m": diagnostics.residual_p95_m,
    }


def improvement_summary(before: dict[str, object], after: dict[str, object]) -> dict[str, float]:
    keys = (
        "vggt_depthpro_rel_diff_mean",
        "vggt_depthpro_rel_diff_p95",
        "cross_view_depth_residual_mean_m",
        "cross_view_depth_residual_p95_m",
    )
    summary: dict[str, float] = {}
    for key in keys:
        before_value = _float_metric(before.get(key, 0.0))
        after_value = _float_metric(after.get(key, 0.0))
        summary[key] = 0.0 if before_value <= _EPS else (before_value - after_value) / before_value
    return summary


def anti_cheat_passed(raw_map: FusedWorldMapResult, optimized_map: FusedWorldMapResult) -> bool:
    if raw_map.point_count <= 0:
        return False
    retained = optimized_map.point_count / raw_map.point_count
    return bool(
        retained >= 0.5
        and optimized_map.occupied_voxel_count > 0
        and optimized_map.mesh_triangle_count > 0
    )


def write_projection_diagnostics(
    root: Path, diagnostics: ProjectionDiagnostics, *, suffix: str
) -> None:
    write_json(
        root / "diagnostics" / f"projection_consistency_{suffix}.json",
        {
            "format_name": "atlas3r_projection_consistency",
            "format_version": 1,
            "projection_count": diagnostics.projection_count,
            "cross_view_depth_residual_mean_m": diagnostics.residual_mean_m,
            "cross_view_depth_residual_p95_m": diagnostics.residual_p95_m,
            "truth_boundary": optimized_truth_boundary(),
        },
    )
    np.savez_compressed(
        root / "diagnostics" / f"projection_residuals_{suffix}.npz",
        residuals_m=diagnostics.residuals_m,
        frame_i=diagnostics.frame_i,
        frame_j=diagnostics.frame_j,
    )


def estimate_row(item: ScaleBiasEstimate) -> dict[str, object]:
    row = {
        "frame_id": item.frame_id,
        "keyframe_id": item.keyframe_id,
        "depth_source": "depth_pro",
        "scale": item.scale,
        "bias_m": item.bias_m,
        "robust_loss_before": item.robust_loss_before,
        "robust_loss_after": item.robust_loss_after,
        "valid_overlap_pixels": item.valid_overlap_pixels,
        "accepted": item.accepted,
    }
    if item.rejection_reason is not None:
        row["rejection_reason"] = item.rejection_reason
    return row


def optimizer_markdown(
    status: str,
    before: dict[str, object],
    after: dict[str, object],
    improvement: dict[str, float],
) -> str:
    return "\n".join(
        [
            "# Map Consistency Optimizer Report",
            "",
            f"- Status: {status}",
            f"- Improvement happened: {status == 'improved'}",
            f"- Before relative diff mean: {before['vggt_depthpro_rel_diff_mean']}",
            f"- After relative diff mean: {after['vggt_depthpro_rel_diff_mean']}",
            f"- Before projection residual mean m: {before['cross_view_depth_residual_mean_m']}",
            f"- After projection residual mean m: {after['cross_view_depth_residual_mean_m']}",
            f"- Retained point ratio: {after['retained_point_ratio']}",
            f"- Improvement ratios: {json.dumps(improvement, sort_keys=True)}",
            "- Physical accuracy claim: false",
            "- Training-quality claim: false",
            "",
            "The optimizer improves diagnostic teacher-consensus consistency only. It does "
            "not create measured geometry, hidden completion, or training-quality labels.",
            "",
        ]
    )


def optimized_truth_boundary() -> dict[str, object]:
    return {
        "label_type": "teacher_pseudo_optimized_map",
        "measured_geometry": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "metric_scale_source": "unanchored_vggt_depthpro_teacher_consensus",
        "physical_accuracy_claim": False,
        "training_quality": False,
        "realtime_claim": False,
        "optimized_world_state": "diagnostic_depth_consistency_only",
        "accuracy_report": False,
        "usable_for_training": False,
    }


def empty_metrics() -> dict[str, object]:
    return {
        "vggt_depthpro_abs_diff_mean_m": 0.0,
        "vggt_depthpro_abs_diff_p50_m": 0.0,
        "vggt_depthpro_abs_diff_p95_m": 0.0,
        "vggt_depthpro_rel_diff_mean": 0.0,
        "vggt_depthpro_rel_diff_p95": 0.0,
        "log_depth_diff_mean": 0.0,
        "high_disagreement_ratio": 0.0,
        "common_valid_overlap_pixels": 0,
    }


def weighted_lstsq_scale_bias(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    options: MapConsistencyOptimizerOptions,
) -> tuple[float, float]:
    sqrt_w = np.sqrt(np.maximum(weights, 1e-6))
    A = np.stack([x, np.ones_like(x)], axis=1)
    ridge = 1e-6 * max(1, x.shape[0])
    A_aug = np.vstack([A * sqrt_w[:, None], [[ridge, 0.0], [0.0, ridge]]])
    y_aug = np.concatenate([y * sqrt_w, [ridge, 0.0]])
    scale, bias = np.linalg.lstsq(A_aug, y_aug, rcond=None)[0].tolist()
    scale = float(np.clip(scale, options.depth_scale_min, options.depth_scale_max))
    bias = float(np.clip(bias, -options.depth_bias_max_m, options.depth_bias_max_m))
    return scale, bias


def robust_log_loss(
    reference_depth: NDArray[np.float32],
    compare_depth: NDArray[np.float32],
    valid: NDArray[np.bool_],
) -> float:
    valid = valid & np.isfinite(reference_depth) & np.isfinite(compare_depth) & (compare_depth > 0)
    if not np.any(valid):
        return float("inf")
    residual = np.abs(
        np.log(np.maximum(reference_depth[valid], _EPS))
        - np.log(np.maximum(compare_depth[valid], _EPS))
    )
    cutoff = max(float(np.percentile(residual, 80)), 1e-3)
    return float(np.minimum(residual, cutoff).mean())


def depth_and_valid(
    record: dict[str, object], proposal_cache: ProposalCacheResult
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    depth = proposal_cache.depth_arrays[str(record["depth_key"])].astype(np.float32)
    valid = proposal_cache.depth_arrays[str(record["valid_mask_key"])] > 0.0
    return depth, valid


def overlap(
    vggt_depth: NDArray[np.float32],
    vggt_valid: NDArray[np.bool_],
    depth_pro_depth: NDArray[np.float32],
    depth_pro_valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    mask = (
        vggt_valid
        & depth_pro_valid
        & np.isfinite(vggt_depth)
        & np.isfinite(depth_pro_depth)
        & (vggt_depth > 0.0)
        & (depth_pro_depth > 0.0)
    )
    return cast(NDArray[np.bool_], mask)


def aligned_depth(
    depth_pro_depth: NDArray[np.float32], estimate: ScaleBiasEstimate
) -> NDArray[np.float32]:
    if not estimate.accepted:
        return depth_pro_depth.astype(np.float32)
    return (estimate.scale * depth_pro_depth + estimate.bias_m).astype(np.float32)


def diff_maps(
    vggt_depth: NDArray[np.float32],
    depth_pro_depth: NDArray[np.float32],
    valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    abs_diff = np.where(valid, np.abs(vggt_depth - depth_pro_depth), 0.0).astype(np.float32)
    rel_diff = np.where(valid, abs_diff / np.maximum(depth_pro_depth, _EPS), 0.0).astype(np.float32)
    log_diff = np.where(
        valid,
        np.abs(np.log(np.maximum(vggt_depth, _EPS)) - np.log(np.maximum(depth_pro_depth, _EPS))),
        0.0,
    ).astype(np.float32)
    high = valid & ((abs_diff > _ABS_HIGH_THRESHOLD_M) | (rel_diff > _REL_HIGH_THRESHOLD))
    return abs_diff, rel_diff, log_diff, high


def optimized_consensus_frame(
    vggt_depth: NDArray[np.float32],
    vggt_valid: NDArray[np.bool_],
    aligned: NDArray[np.float32],
    depth_pro_valid: NDArray[np.bool_],
    abs_diff: NDArray[np.float32],
    rel_diff: NDArray[np.float32],
    high: NDArray[np.bool_],
) -> tuple[NDArray[np.int16], NDArray[np.float32], NDArray[np.float32]]:
    both = vggt_valid & depth_pro_valid & (vggt_depth > 0.0) & (aligned > 0.0)
    vggt_only = vggt_valid & ~depth_pro_valid
    depth_pro_only = depth_pro_valid & ~vggt_valid
    source_mask = np.zeros(vggt_depth.shape, dtype=np.int16)
    source_mask[vggt_only] = 1
    source_mask[depth_pro_only] = 2
    source_mask[both & ~high] = 3
    source_mask[both & high] = 4
    depth = np.zeros(vggt_depth.shape, dtype=np.float32)
    depth[vggt_only] = vggt_depth[vggt_only]
    depth[depth_pro_only] = aligned[depth_pro_only]
    depth[both] = (0.65 * vggt_depth[both] + 0.35 * aligned[both]).astype(np.float32)
    confidence = np.zeros(vggt_depth.shape, dtype=np.float32)
    confidence[vggt_only | depth_pro_only] = 0.4
    penalty = np.minimum(0.85, 1.8 * rel_diff + 0.2 * abs_diff)
    confidence[both] = np.clip(0.95 - penalty[both], 0.05, 0.95)
    return source_mask, depth, confidence


def append_consensus_record(
    index: int,
    *,
    frame_id: int,
    keyframe_id: int,
    consensus_depth: NDArray[np.float32],
    consensus_confidence: NDArray[np.float32],
    source_mask: NDArray[np.int16],
    records: list[dict[str, object]],
    arrays: dict[str, NDArray[np.float32]],
) -> None:
    depth_key = f"optimized_consensus_depth_{index:06d}"
    sigma_key = f"optimized_consensus_sigma_{index:06d}"
    confidence_key = f"optimized_consensus_confidence_{index:06d}"
    valid_key = f"optimized_consensus_valid_mask_{index:06d}"
    source_key = f"optimized_consensus_source_mask_{index:06d}"
    valid = source_mask > 0
    arrays[depth_key] = consensus_depth.astype(np.float32)
    arrays[sigma_key] = np.where(valid, 1.0 - consensus_confidence, 0.0).astype(np.float32)
    arrays[confidence_key] = consensus_confidence.astype(np.float32)
    arrays[valid_key] = valid.astype(np.float32)
    arrays[source_key] = source_mask.astype(np.float32)
    records.append(
        {
            "teacher_name": "optimized_consensus",
            "frame_id": frame_id,
            "keyframe_index": keyframe_id,
            "depth_key": depth_key,
            "depth_sigma_key": sigma_key,
            "confidence_key": confidence_key,
            "valid_mask_key": valid_key,
            "source_mask_key": source_key,
            "depth_shape": list(consensus_depth.shape),
            "depth_source": "optimized_vggt_depth_pro_consensus",
            "metric_scale_source": "unanchored_vggt_depthpro_teacher_consensus",
            "coordinate_convention": "x_right_y_down_z_forward",
            "measured_geometry": False,
            "truth_boundary": optimized_truth_boundary(),
            "diagnostic_only": False,
            "optimized_consensus": True,
        }
    )


def write_optimized_disagreement_files(
    root: Path,
    result: DisagreementResult,
    frame_ids: list[int],
    abs_maps: list[NDArray[np.float32]],
    rel_maps: list[NDArray[np.float32]],
    valid_maps: list[NDArray[np.float32]],
    high_maps: list[NDArray[np.float32]],
    depths: list[NDArray[np.float32]],
    confidences: list[NDArray[np.float32]],
    source_masks: list[NDArray[np.int16]],
) -> None:
    optimizer_dir = root / "optimizer"
    np.savez_compressed(
        optimizer_dir / "optimized_disagreement_maps.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        abs_depth_diff_m=np.stack(abs_maps).astype(np.float32),
        rel_depth_diff=np.stack(rel_maps).astype(np.float32),
        valid_overlap_mask=np.stack(valid_maps).astype(np.float32),
        high_disagreement_mask=np.stack(high_maps).astype(np.float32),
    )
    np.savez_compressed(
        optimizer_dir / "optimized_consensus_preview.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        consensus_depth_m=np.stack(depths).astype(np.float32),
        consensus_confidence=np.stack(confidences).astype(np.float32),
        source_mask=np.stack(source_masks).astype(np.int16),
        metadata_json=json.dumps(
            {
                "status": result.consensus_status,
                "diagnostic_only": False,
                "optimized_consensus": True,
                "truth_boundary": optimized_truth_boundary(),
            },
            sort_keys=True,
        ),
    )
    write_json(
        optimizer_dir / "optimized_disagreement.json",
        {
            "status": result.status,
            "optimized_consensus": True,
            "summary": result.summary,
            "frames": list(result.frame_summaries),
            "maps_path": result.maps_npz_path,
            "consensus_preview_path": result.consensus_npz_path,
            "truth_boundary": optimized_truth_boundary(),
        },
    )


def estimate_row_for_trace(iteration: int, estimate: ScaleBiasEstimate) -> dict[str, object]:
    row = estimate_row(estimate)
    row["iteration"] = iteration
    return row


def frame_metric_payload(
    frame_id: int,
    valid: NDArray[np.bool_],
    abs_diff: NDArray[np.float32],
    rel_diff: NDArray[np.float32],
    log_diff: NDArray[np.float32],
    high: NDArray[np.bool_],
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "overlap_pixel_count": int(valid.sum()),
        "abs_depth_diff_mean_m": mean(abs_diff[valid]),
        "rel_depth_diff_mean": mean(rel_diff[valid]),
        "rel_depth_diff_p95": percentile(rel_diff[valid], 95),
        "log_depth_diff_mean": mean(log_diff[valid]),
        "disagreement_high_ratio": mean(high[valid].astype(np.float32)),
    }


def summary_from_maps(
    valid_maps: list[NDArray[np.float32]],
    abs_maps: list[NDArray[np.float32]],
    rel_maps: list[NDArray[np.float32]],
    log_maps: list[NDArray[np.float32]],
    high_maps: list[NDArray[np.float32]],
) -> dict[str, object]:
    valid = np.concatenate([item.reshape((-1,)) > 0.0 for item in valid_maps])
    abs_all = np.concatenate([item.reshape((-1,)) for item in abs_maps])[valid]
    rel_all = np.concatenate([item.reshape((-1,)) for item in rel_maps])[valid]
    log_all = np.concatenate([item.reshape((-1,)) for item in log_maps])[valid]
    high_all = np.concatenate([item.reshape((-1,)) for item in high_maps])[valid]
    return {
        "overlap_pixel_count": int(valid.sum()),
        "abs_depth_diff_mean_m": mean(abs_all),
        "abs_depth_diff_p50_m": percentile(abs_all, 50),
        "abs_depth_diff_p95_m": percentile(abs_all, 95),
        "rel_depth_diff_mean": mean(rel_all),
        "rel_depth_diff_p95": percentile(rel_all, 95),
        "log_depth_diff_mean": mean(log_all),
        "disagreement_high_ratio": mean(high_all),
    }


def resize_nearest(array: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float32]:
    if array.shape == shape:
        return array.astype(np.float32)
    y_idx = np.clip(
        np.round(np.linspace(0, array.shape[0] - 1, shape[0])).astype(np.int32),
        0,
        array.shape[0] - 1,
    )
    x_idx = np.clip(
        np.round(np.linspace(0, array.shape[1] - 1, shape[1])).astype(np.int32),
        0,
        array.shape[1] - 1,
    )
    return cast(NDArray[np.float32], array[y_idx[:, None], x_idx[None, :]].astype(np.float32))


def resize_nearest_bool(array: NDArray[np.bool_], shape: tuple[int, int]) -> NDArray[np.bool_]:
    return resize_nearest(array.astype(np.float32), shape) > 0.0


def concat(values: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    return np.concatenate(values).astype(np.float32) if values else np.zeros((0,), np.float32)


def mean(values: NDArray[np.float32]) -> float:
    return float(values.mean()) if values.size else 0.0


def percentile(values: NDArray[np.float32], value: float) -> float:
    return float(np.percentile(values, value)) if values.size else 0.0


def _float_metric(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))
