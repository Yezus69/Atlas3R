"""Map consistency optimizer orchestration for offline world builds."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.fused_world_map import (
    FusedWorldMapOptions,
    FusedWorldMapResult,
    write_fused_world_map,
)
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.map_consistency_core import (
    MapConsistencyOptimizerOptions,
    ScaleBiasEstimate,
    anti_cheat_passed,
    build_optimized_consensus,
    depth_metrics_from_records,
    empty_metrics,
    estimate_row,
    fit_depth_scale_bias,
    improvement_summary,
    optimized_truth_boundary,
    optimizer_markdown,
    paired_records,
    projection_metrics,
    select_focal_scale,
    validate_options,
    with_map_counts,
    with_scaled_vggt_intrinsics,
    write_projection_diagnostics,
)
from atlas3r.offline.projection_diagnostics import (
    ProjectionDiagnostics,
    compute_projection_consistency,
)
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint, write_json, write_jsonl


@dataclass(frozen=True)
class MapConsistencyOptimizerResult:
    status: str
    manifest_path: str | None = None
    before_metrics_path: str | None = None
    after_metrics_path: str | None = None
    depth_scale_bias_path: str | None = None
    intrinsics_adjustments_path: str | None = None
    projection_residuals_path: str | None = None
    optimization_trace_path: str | None = None
    optimizer_report_path: str | None = None
    optimized_world_map: FusedWorldMapResult = FusedWorldMapResult(status="disabled")
    before_metrics: dict[str, object] | None = None
    after_metrics: dict[str, object] | None = None
    improvement_passed: bool = False
    improvement_summary: dict[str, float] | None = None
    optimized_disagreement: DisagreementResult | None = None

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        paths = (
            self.manifest_path,
            self.before_metrics_path,
            self.after_metrics_path,
            self.depth_scale_bias_path,
            self.intrinsics_adjustments_path,
            self.projection_residuals_path,
            self.optimization_trace_path,
            self.optimizer_report_path,
            *self.optimized_world_map.artifact_paths,
        )
        return tuple(path for path in paths if path is not None)


def write_map_consistency_optimizer(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    raw_world_map: FusedWorldMapResult,
    map_options: FusedWorldMapOptions,
    optimizer_options: MapConsistencyOptimizerOptions,
    failure_points: list[FailurePoint],
) -> MapConsistencyOptimizerResult:
    if not optimizer_options.enabled:
        return MapConsistencyOptimizerResult(status="disabled")
    validate_options(optimizer_options)
    root = Path(run_dir)
    (root / "optimizer").mkdir(parents=True, exist_ok=True)
    pairs = paired_records(proposal_cache)
    if not pairs:
        result = _write_insufficient(root, input_path, raw_world_map, "no overlapping witnesses")
        failure_points.append(
            FailurePoint(
                module="map_consistency_optimizer",
                code="optimizer_insufficient_witnesses",
                severity="warning",
                status="unavailable",
                why="VGGT and Depth Pro depth proposals do not overlap by frame ID",
                input_missing="overlapping VGGT and Depth Pro proposals",
                artifact_path="optimizer/optimizer_manifest.json",
            )
        )
        return result

    estimates: list[ScaleBiasEstimate] = []
    trace_rows: list[dict[str, object]] = []
    optimized_disagreement = build_optimized_consensus(
        root,
        proposal_cache,
        pairs=pairs,
        options=optimizer_options,
        estimates=estimates,
        trace_rows=trace_rows,
    )
    before_metrics = depth_metrics_from_records(proposal_cache, pairs, estimates=None)
    after_metrics = depth_metrics_from_records(proposal_cache, pairs, estimates=estimates)
    before_projection = compute_projection_consistency(
        proposal_cache=proposal_cache,
        depth_records=disagreement.consensus_depth_records if disagreement is not None else (),
        depth_arrays=disagreement.consensus_depth_arrays if disagreement is not None else {},
        max_neighbor_pairs=optimizer_options.cross_view_pairs,
        min_overlap_pixels=optimizer_options.min_overlap_pixels,
    )
    focal_scale = select_focal_scale(proposal_cache, optimized_disagreement, optimizer_options)
    optimized_proposals = with_scaled_vggt_intrinsics(proposal_cache, focal_scale)
    after_projection = compute_projection_consistency(
        proposal_cache=optimized_proposals,
        depth_records=optimized_disagreement.consensus_depth_records,
        depth_arrays=optimized_disagreement.consensus_depth_arrays,
        max_neighbor_pairs=optimizer_options.cross_view_pairs,
        min_overlap_pixels=optimizer_options.min_overlap_pixels,
    )
    write_projection_diagnostics(root, before_projection, suffix="before")
    write_projection_diagnostics(root, after_projection, suffix="after")
    optimized_map = _write_optimized_map(
        root,
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=optimized_proposals,
        optimized_disagreement=optimized_disagreement,
        map_options=map_options,
        optimizer_options=optimizer_options,
        failure_points=failure_points,
    )
    before_metrics = with_map_counts(before_metrics, raw_world_map, retained_ratio=1.0)
    retained_ratio = (
        float(optimized_map.point_count / raw_world_map.point_count)
        if raw_world_map.point_count
        else 0.0
    )
    after_metrics = with_map_counts(after_metrics, optimized_map, retained_ratio=retained_ratio)
    before_metrics.update(projection_metrics(before_projection))
    after_metrics.update(projection_metrics(after_projection))
    improvements = improvement_summary(before_metrics, after_metrics)
    improved = any(
        ratio >= optimizer_options.min_improvement_ratio for ratio in improvements.values()
    )
    anti_cheat = anti_cheat_passed(raw_world_map, optimized_map)
    status = "improved" if improved and anti_cheat else "failed_no_improvement"
    if not anti_cheat:
        status = "failed_anti_cheat"
    _write_optimizer_outputs(
        root,
        status=status,
        options=optimizer_options,
        estimates=estimates,
        trace_rows=trace_rows,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        before_projection=before_projection,
        after_projection=after_projection,
        focal_scale=focal_scale,
        improvements=improvements,
        improved=improved,
        anti_cheat=anti_cheat,
        optimized_map=optimized_map,
    )
    return MapConsistencyOptimizerResult(
        status=status,
        manifest_path="optimizer/optimizer_manifest.json",
        before_metrics_path="optimizer/before_metrics.json",
        after_metrics_path="optimizer/after_metrics.json",
        depth_scale_bias_path="optimizer/depth_scale_bias.jsonl",
        intrinsics_adjustments_path="optimizer/intrinsics_adjustments.json",
        projection_residuals_path="optimizer/projection_residuals.npz",
        optimization_trace_path="optimizer/optimization_trace.jsonl",
        optimizer_report_path="optimizer/optimizer_report.md",
        optimized_world_map=optimized_map,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        improvement_passed=improved and anti_cheat,
        improvement_summary=improvements,
        optimized_disagreement=optimized_disagreement,
    )


def _write_optimized_map(
    root: Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    optimized_disagreement: DisagreementResult,
    map_options: FusedWorldMapOptions,
    optimizer_options: MapConsistencyOptimizerOptions,
    failure_points: list[FailurePoint],
) -> FusedWorldMapResult:
    if not optimizer_options.export_optimized_world_map:
        return FusedWorldMapResult(status="disabled")
    return write_fused_world_map(
        root,
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        disagreement=optimized_disagreement,
        options=replace(
            map_options,
            export_world_map=True,
            depth_source="consensus",
            output_dir_name="world_map_optimized",
            optimized_map=True,
        ),
        failure_points=failure_points,
    )


def _write_optimizer_outputs(
    root: Path,
    *,
    status: str,
    options: MapConsistencyOptimizerOptions,
    estimates: list[ScaleBiasEstimate],
    trace_rows: list[dict[str, object]],
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    before_projection: ProjectionDiagnostics,
    after_projection: ProjectionDiagnostics,
    focal_scale: float,
    improvements: dict[str, float],
    improved: bool,
    anti_cheat: bool,
    optimized_map: FusedWorldMapResult,
) -> None:
    optimizer_dir = root / "optimizer"
    write_json(optimizer_dir / "before_metrics.json", before_metrics)
    write_json(optimizer_dir / "after_metrics.json", after_metrics)
    write_jsonl(optimizer_dir / "depth_scale_bias.jsonl", [estimate_row(e) for e in estimates])
    write_json(
        optimizer_dir / "intrinsics_adjustments.json",
        {
            "enabled": options.enable_intrinsics_scale,
            "focal_scale": focal_scale,
            "focal_scale_min": options.focal_scale_min,
            "focal_scale_max": options.focal_scale_max,
            "adjustment_type": "global_focal_scale",
        },
    )
    np.savez_compressed(
        optimizer_dir / "projection_residuals.npz",
        before_residuals_m=before_projection.residuals_m,
        after_residuals_m=after_projection.residuals_m,
        before_frame_i=before_projection.frame_i,
        before_frame_j=before_projection.frame_j,
        after_frame_i=after_projection.frame_i,
        after_frame_j=after_projection.frame_j,
    )
    write_jsonl(optimizer_dir / "optimization_trace.jsonl", trace_rows)
    write_json(
        optimizer_dir / "optimizer_manifest.json",
        {
            "format_name": "atlas3r_map_consistency_optimizer",
            "format_version": 1,
            "status": status,
            "optimized_world_state": "diagnostic_depth_consistency_only",
            "truth_boundary": optimized_truth_boundary(),
            "options": options.__dict__,
            "improvement_passed": improved,
            "anti_cheat_passed": anti_cheat,
            "improvement_summary": improvements,
            "optimized_world_map_manifest": optimized_map.manifest_path,
        },
    )
    (optimizer_dir / "optimizer_report.md").write_text(
        optimizer_markdown(status, before_metrics, after_metrics, improvements),
        encoding="utf-8",
    )


def _write_insufficient(
    root: Path,
    input_path: str,
    raw_world_map: FusedWorldMapResult,
    reason: str,
) -> MapConsistencyOptimizerResult:
    empty_projection = ProjectionDiagnostics(
        0,
        0.0,
        0.0,
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int32),
    )
    before = with_map_counts(empty_metrics(), raw_world_map, retained_ratio=1.0)
    after = with_map_counts(empty_metrics(), FusedWorldMapResult(status="unavailable"), 0.0)
    before.update(projection_metrics(empty_projection))
    after.update(projection_metrics(empty_projection))
    optimizer_dir = root / "optimizer"
    write_json(optimizer_dir / "before_metrics.json", before)
    write_json(optimizer_dir / "after_metrics.json", after)
    write_jsonl(optimizer_dir / "depth_scale_bias.jsonl", [])
    write_json(
        optimizer_dir / "intrinsics_adjustments.json", {"enabled": False, "focal_scale": 1.0}
    )
    np.savez_compressed(
        optimizer_dir / "projection_residuals.npz",
        before_residuals_m=empty_projection.residuals_m,
        after_residuals_m=empty_projection.residuals_m,
    )
    write_jsonl(optimizer_dir / "optimization_trace.jsonl", [])
    write_json(
        optimizer_dir / "optimizer_manifest.json",
        {
            "format_name": "atlas3r_map_consistency_optimizer",
            "format_version": 1,
            "status": "insufficient_witnesses",
            "why": reason,
            "input_path": input_path,
            "truth_boundary": optimized_truth_boundary(),
        },
    )
    (optimizer_dir / "optimizer_report.md").write_text(
        "# Map Consistency Optimizer Report\n\n"
        f"- Status: insufficient_witnesses\n- Why: {reason}\n"
        "- Physical accuracy claim: false\n- Training-quality claim: false\n",
        encoding="utf-8",
    )
    return MapConsistencyOptimizerResult(
        status="insufficient_witnesses",
        manifest_path="optimizer/optimizer_manifest.json",
        before_metrics_path="optimizer/before_metrics.json",
        after_metrics_path="optimizer/after_metrics.json",
        depth_scale_bias_path="optimizer/depth_scale_bias.jsonl",
        intrinsics_adjustments_path="optimizer/intrinsics_adjustments.json",
        projection_residuals_path="optimizer/projection_residuals.npz",
        optimization_trace_path="optimizer/optimization_trace.jsonl",
        optimizer_report_path="optimizer/optimizer_report.md",
        before_metrics=before,
        after_metrics=after,
    )


__all__ = [
    "MapConsistencyOptimizerOptions",
    "MapConsistencyOptimizerResult",
    "ScaleBiasEstimate",
    "compute_projection_consistency",
    "fit_depth_scale_bias",
    "write_map_consistency_optimizer",
]
