"""End-to-end Offline World Builder vertical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import write_camera_scale_ledgers
from atlas3r.offline.consensus_world import write_consensus_world_state
from atlas3r.offline.depth_pro_witness import DepthProRuntimeOptions, run_depth_pro_witness
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.fused_world_map import (
    FusedWorldMapOptions,
    FusedWorldMapResult,
    MapDepthSource,
    write_fused_world_map,
)
from atlas3r.offline.geometry_preview import write_geometry_preview
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.map_consistency_optimizer import (
    MapConsistencyOptimizerOptions,
    MapConsistencyOptimizerResult,
    write_map_consistency_optimizer,
)
from atlas3r.offline.object_ledger import write_object_ledger
from atlas3r.offline.proposal_cache import DebugGeometryMode, write_proposal_cache
from atlas3r.offline.quality_report import write_quality_report
from atlas3r.offline.render_repair import write_render_repair_diagnostics
from atlas3r.offline.run_manifest import (
    FailurePoint,
    ensure_run_tree,
    utc_now_iso,
    write_json,
)
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.training_cache import write_training_cache_manifest
from atlas3r.offline.vggt_witness import VggtRuntimeOptions, VggtStitchMode, run_vggt_witness


@dataclass(frozen=True)
class BuildWorldOptions:
    input_path: str
    output_path: str
    max_frames: int = 120
    keyframe_stride: int = 5
    keyframe_max_count: int = 32
    debug_geometry_mode: DebugGeometryMode = "none"
    write_ply: bool = False
    enable_vggt: bool = False
    vggt_repo: str | None = None
    vggt_checkpoint: str | None = None
    vggt_device: str = "cuda:0"
    vggt_image_size: int = 518
    vggt_window_size: int = 24
    vggt_window_overlap: int = 8
    vggt_max_keyframes: int | None = None
    vggt_proposal_cache: str | None = None
    vggt_stitch_mode: VggtStitchMode = "overlap-sim3"
    enable_depth_pro: bool = False
    depth_pro_repo: str | None = None
    depth_pro_checkpoint: str | None = None
    depth_pro_device: str = "cuda:0"
    depth_pro_image_size: int | None = None
    depth_pro_max_keyframes: int | None = None
    depth_pro_proposal_cache: str | None = None
    export_world_map: bool = False
    map_point_stride: int = 8
    map_max_points: int = 2_000_000
    map_min_confidence: float = 0.25
    map_max_relative_disagreement: float = 0.25
    map_voxel_size_m: float = 0.05
    map_depth_source: MapDepthSource | None = None
    map_write_observed_mesh: bool = False
    map_write_occupancy: bool = False
    optimize_map_consistency: bool = False
    optimizer_max_iterations: int = 5
    optimizer_depth_scale_min: float = 0.5
    optimizer_depth_scale_max: float = 2.0
    optimizer_depth_bias_max_m: float = 1.0
    optimizer_enable_intrinsics_scale: bool = False
    optimizer_focal_scale_min: float = 0.8
    optimizer_focal_scale_max: float = 1.25
    optimizer_cross_view_pairs: int = 3
    optimizer_min_overlap_pixels: int = 512
    optimizer_min_improvement_ratio: float = 0.05
    export_optimized_world_map: bool = False


@dataclass(frozen=True)
class BuildWorldResult:
    exit_code: int
    run_dir: str
    artifact_paths: tuple[str, ...]
    geometry_point_count: int
    failure_count: int
    world_map_point_count: int = 0
    occupied_voxel_count: int = 0
    inspectable_map_available: bool = False
    optimized_world_map_point_count: int = 0
    optimized_occupied_voxel_count: int = 0
    optimized_inspectable_map_available: bool = False


def build_world(options: BuildWorldOptions) -> BuildWorldResult:
    started_at = utc_now_iso()
    run_dir = ensure_run_tree(options.output_path)
    failure_points: list[FailurePoint] = []
    frame_cache = build_frame_cache(
        options.input_path,
        run_dir,
        max_frames=options.max_frames,
        failure_points=failure_points,
        allow_missing=True,
    )
    keyframes = select_keyframes(
        frame_cache.records,
        run_dir,
        keyframe_stride=options.keyframe_stride,
        keyframe_max_count=options.keyframe_max_count,
        failure_points=failure_points,
    )
    vggt_result = run_vggt_witness(
        run_dir,
        frame_cache=frame_cache,
        keyframes=keyframes.keyframes,
        options=VggtRuntimeOptions(
            enabled=options.enable_vggt,
            proposal_cache=options.vggt_proposal_cache,
            repo_path=options.vggt_repo,
            checkpoint=options.vggt_checkpoint,
            device=options.vggt_device,
            image_size=options.vggt_image_size,
            window_size=options.vggt_window_size,
            window_overlap=options.vggt_window_overlap,
            max_keyframes=options.vggt_max_keyframes,
            stitch_mode=options.vggt_stitch_mode,
        ),
        failure_points=failure_points,
    )
    depth_pro_result = run_depth_pro_witness(
        run_dir,
        frame_cache=frame_cache,
        keyframes=keyframes.keyframes,
        options=DepthProRuntimeOptions(
            enabled=options.enable_depth_pro,
            proposal_cache=options.depth_pro_proposal_cache,
            repo_path=options.depth_pro_repo,
            checkpoint=options.depth_pro_checkpoint,
            device=options.depth_pro_device,
            image_size=options.depth_pro_image_size,
            max_keyframes=options.depth_pro_max_keyframes,
        ),
        failure_points=failure_points,
    )
    teacher_statuses = write_teacher_statuses(
        run_dir, failure_points, vggt_result=vggt_result, depth_pro_result=depth_pro_result
    )
    proposals = write_proposal_cache(
        run_dir,
        teacher_statuses=teacher_statuses,
        frame_records=frame_cache.records,
        keyframes=keyframes.keyframes,
        debug_geometry_mode=options.debug_geometry_mode,
        vggt_result=vggt_result,
        depth_pro_result=depth_pro_result,
    )
    disagreement = write_teacher_disagreement(run_dir, proposal_cache=proposals)
    ledgers = write_camera_scale_ledgers(
        run_dir,
        camera=frame_cache.camera,
        frame_count=len(frame_cache.records),
        debug_geometry_mode=options.debug_geometry_mode,
        proposal_cache=proposals,
    )
    consensus = write_consensus_world_state(
        run_dir,
        frame_records=frame_cache.records,
        keyframes=keyframes.keyframes,
        proposal_cache=proposals,
        ledgers=ledgers,
        disagreement=disagreement,
    )
    geometry = write_geometry_preview(
        run_dir,
        frame_cache=frame_cache,
        proposal_cache=proposals,
        write_ply=options.write_ply,
        failure_points=failure_points,
        disagreement=disagreement,
    )
    map_options = FusedWorldMapOptions(
        export_world_map=options.export_world_map or options.optimize_map_consistency,
        point_stride=options.map_point_stride,
        max_points=options.map_max_points,
        min_confidence=options.map_min_confidence,
        max_relative_disagreement=options.map_max_relative_disagreement,
        voxel_size_m=options.map_voxel_size_m,
        depth_source=options.map_depth_source,
        write_observed_mesh=options.map_write_observed_mesh,
        write_occupancy=options.map_write_occupancy,
    )
    world_map = write_fused_world_map(
        run_dir,
        input_path=options.input_path,
        frame_cache=frame_cache,
        keyframes=keyframes.keyframes,
        proposal_cache=proposals,
        disagreement=disagreement,
        options=map_options,
        failure_points=failure_points,
    )
    optimizer = write_map_consistency_optimizer(
        run_dir,
        input_path=options.input_path,
        frame_cache=frame_cache,
        keyframes=keyframes.keyframes,
        proposal_cache=proposals,
        disagreement=disagreement,
        raw_world_map=world_map,
        map_options=map_options,
        optimizer_options=MapConsistencyOptimizerOptions(
            enabled=options.optimize_map_consistency,
            export_optimized_world_map=options.export_optimized_world_map
            or options.optimize_map_consistency,
            max_iterations=options.optimizer_max_iterations,
            depth_scale_min=options.optimizer_depth_scale_min,
            depth_scale_max=options.optimizer_depth_scale_max,
            depth_bias_max_m=options.optimizer_depth_bias_max_m,
            enable_intrinsics_scale=options.optimizer_enable_intrinsics_scale,
            focal_scale_min=options.optimizer_focal_scale_min,
            focal_scale_max=options.optimizer_focal_scale_max,
            cross_view_pairs=options.optimizer_cross_view_pairs,
            min_overlap_pixels=options.optimizer_min_overlap_pixels,
            min_improvement_ratio=options.optimizer_min_improvement_ratio,
        ),
        failure_points=failure_points,
    )
    objects = write_object_ledger(run_dir, proposal_cache=proposals, failure_points=failure_points)
    render = write_render_repair_diagnostics(
        run_dir,
        geometry=geometry,
        keyframes=keyframes.keyframes,
        failure_points=failure_points,
        disagreement=disagreement,
    )
    training = write_training_cache_manifest(
        run_dir,
        frame_index_path=frame_cache.frame_index_path,
        keyframes_path=keyframes.keyframes_path,
        world_state_path=consensus.world_state_path,
        geometry=geometry,
        disagreement=disagreement,
        world_map=world_map,
        optimized_world_map=optimizer.optimized_world_map,
        optimizer=optimizer,
    )
    write_json(
        run_dir / "diagnostics" / "failure_points.json",
        {"failure_points": [failure.to_dict() for failure in failure_points]},
    )
    quality = write_quality_report(
        run_dir,
        teacher_statuses=teacher_statuses,
        proposal_cache=proposals,
        disagreement=disagreement,
        ledgers=ledgers,
        geometry=geometry,
        world_map=world_map,
        optimizer=optimizer,
        objects=objects,
        render=render,
        training=training,
        failure_points=failure_points,
    )
    artifact_paths = _artifact_paths(geometry.geometry_ply_path, world_map, optimizer)
    manifest = {
        "format_name": "atlas3r_offline_world_builder_run_manifest",
        "format_version": 1,
        "run_id": Path(options.output_path).name,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "input_path": options.input_path,
        "output_path": str(options.output_path),
        "command": {
            "max_frames": options.max_frames,
            "keyframe_stride": options.keyframe_stride,
            "keyframe_max_count": options.keyframe_max_count,
            "debug_geometry_mode": options.debug_geometry_mode,
            "write_ply": options.write_ply,
            "enable_vggt": options.enable_vggt,
            "vggt_repo": options.vggt_repo,
            "vggt_checkpoint": options.vggt_checkpoint,
            "vggt_device": options.vggt_device,
            "vggt_image_size": options.vggt_image_size,
            "vggt_window_size": options.vggt_window_size,
            "vggt_window_overlap": options.vggt_window_overlap,
            "vggt_max_keyframes": options.vggt_max_keyframes,
            "vggt_proposal_cache": options.vggt_proposal_cache,
            "vggt_stitch_mode": options.vggt_stitch_mode,
            "enable_depth_pro": options.enable_depth_pro,
            "depth_pro_repo": options.depth_pro_repo,
            "depth_pro_checkpoint": options.depth_pro_checkpoint,
            "depth_pro_device": options.depth_pro_device,
            "depth_pro_image_size": options.depth_pro_image_size,
            "depth_pro_max_keyframes": options.depth_pro_max_keyframes,
            "depth_pro_proposal_cache": options.depth_pro_proposal_cache,
            "export_world_map": options.export_world_map,
            "map_point_stride": options.map_point_stride,
            "map_max_points": options.map_max_points,
            "map_min_confidence": options.map_min_confidence,
            "map_max_relative_disagreement": options.map_max_relative_disagreement,
            "map_voxel_size_m": options.map_voxel_size_m,
            "map_depth_source": options.map_depth_source,
            "map_write_observed_mesh": options.map_write_observed_mesh,
            "map_write_occupancy": options.map_write_occupancy,
            "optimize_map_consistency": options.optimize_map_consistency,
            "optimizer_max_iterations": options.optimizer_max_iterations,
            "optimizer_depth_scale_min": options.optimizer_depth_scale_min,
            "optimizer_depth_scale_max": options.optimizer_depth_scale_max,
            "optimizer_depth_bias_max_m": options.optimizer_depth_bias_max_m,
            "optimizer_enable_intrinsics_scale": options.optimizer_enable_intrinsics_scale,
            "optimizer_focal_scale_min": options.optimizer_focal_scale_min,
            "optimizer_focal_scale_max": options.optimizer_focal_scale_max,
            "optimizer_cross_view_pairs": options.optimizer_cross_view_pairs,
            "optimizer_min_overlap_pixels": options.optimizer_min_overlap_pixels,
            "optimizer_min_improvement_ratio": options.optimizer_min_improvement_ratio,
            "export_optimized_world_map": options.export_optimized_world_map,
        },
        "module_status": {
            "frame_cache": frame_cache.status,
            "keyframes": keyframes.status,
            "teacher_witnesses": "partial",
            "proposal_cache": proposals.status,
            "teacher_disagreement": disagreement.status,
            "camera_scale_ledger": "partial",
            "consensus_world": consensus.status,
            "geometry_preview": geometry.status,
            "fused_world_map": world_map.status,
            "map_consistency_optimizer": optimizer.status,
            "optimized_world_map": optimizer.optimized_world_map.status,
            "object_ledger": objects.status,
            "render_repair": render.status,
            "training_cache": training.status,
            "quality_report": "complete",
        },
        "artifact_paths": list(artifact_paths),
        "failure_points": "diagnostics/failure_points.json",
        "quality_report": {
            "json": quality.json_path,
            "markdown": quality.markdown_path,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return BuildWorldResult(
        exit_code=1 if frame_cache.input_error else 0,
        run_dir=str(run_dir),
        artifact_paths=artifact_paths,
        geometry_point_count=geometry.point_count,
        failure_count=len(failure_points),
        world_map_point_count=world_map.point_count,
        occupied_voxel_count=world_map.occupied_voxel_count,
        inspectable_map_available=world_map.inspectable_map_available,
        optimized_world_map_point_count=optimizer.optimized_world_map.point_count,
        optimized_occupied_voxel_count=optimizer.optimized_world_map.occupied_voxel_count,
        optimized_inspectable_map_available=optimizer.optimized_world_map.inspectable_map_available,
    )


def _artifact_paths(
    geometry_ply_path: str | None,
    world_map: FusedWorldMapResult,
    optimizer: MapConsistencyOptimizerResult,
) -> tuple[str, ...]:
    paths = [
        "run_manifest.json",
        "frames/frame_index.jsonl",
        "keyframes/keyframes.json",
        "teachers/teacher_status.json",
        "proposals/proposal_manifest.json",
        "world/world_state.json",
        "world/camera_ledger.json",
        "world/scale_ledger.json",
        "geometry/geometry_preview.npz",
        "objects/object_ledger.json",
        "diagnostics/teacher_disagreement.json",
        "diagnostics/disagreement_maps.npz",
        "diagnostics/consensus_preview.npz",
        "diagnostics/render_repair_diagnostics.json",
        "diagnostics/failure_points.json",
        "quality_report.json",
        "quality_report.md",
        "training_cache/training_cache_manifest.json",
    ]
    if geometry_ply_path is not None:
        paths.insert(9, geometry_ply_path)
    paths.extend(world_map.artifact_paths)
    paths.extend(optimizer.artifact_paths)
    return tuple(paths)
