"""Runtime bridge from RGB teacher geometry to sparse TSDF mesh chunks."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping.mesh_artifacts import MeshChunkArtifactWriter
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper, SparseTSDFConfig
from atlas3r.mapping.sparse_tsdf_artifacts import write_sparse_tsdf_outputs
from atlas3r.mapping.sparse_tsdf_meshing import SparseTSDFMesherConfig
from atlas3r.recording.schema import RECORDING_COORDINATE_FRAME
from atlas3r.runtime.live_replay_types import LiveReplayEventRecorder
from atlas3r.runtime.rgb_teacher_config import RGBTeacherMapConfig
from atlas3r.runtime.rgb_teacher_conversion import (
    frames_to_vggt_clip_payload,
    teacher_prediction_to_observations,
)
from atlas3r.runtime.rgb_teacher_eval import write_rgb_teacher_eval_if_available
from atlas3r.runtime.rgb_teacher_inputs import (
    RGBTeacherFrame,
    RGBTeacherInput,
    load_rgb_teacher_input,
    select_rgb_teacher_frames,
)
from atlas3r.runtime.rgb_teacher_outputs import (
    MAPPER_BACKEND,
    empty_mesh_status,
    extract_or_empty_surface,
    live_replay_summary,
    require_nonzero_mesh,
    rgb_teacher_summary,
    rgb_teacher_truth_boundary,
    sparse_metrics,
    write_prediction_payload,
    write_pseudo_recording,
    write_rgb_frames,
    write_rgb_teacher_report,
)
from atlas3r.runtime.student_map_report_common import write_json, write_jsonl
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply
from atlas3r.teachers.external.contracts import ExternalTeacherDependencyError
from atlas3r.teachers.external.vggt import get_vggt_status
from atlas3r.teachers.external.vggt_model import (
    VGGTRawClipPredictor,
    load_vggt_predictor,
)


def run_rgb_teacher_mapping(
    config: RGBTeacherMapConfig,
    *,
    predictor: VGGTRawClipPredictor | None = None,
) -> dict[str, object]:
    """Run RGB-only teacher pseudo geometry through sparse TSDF and mesh chunks."""

    start_ns = time.perf_counter_ns()
    source = load_rgb_teacher_input(config.input)
    frames = select_rgb_teacher_frames(
        source.frames,
        max_frames=config.max_frames,
        frame_stride=config.frame_stride,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    write_rgb_frames(config.output / "rgb_teacher_frames", frames)
    attempts = (
        _attempt_configs(config) if predictor is None and config.teacher == "vggt" else (config,)
    )
    errors: list[str] = []
    for attempt in attempts:
        try:
            return _run_attempt(
                attempt,
                source=source,
                frames=frames[: attempt.max_frames] if attempt.max_frames is not None else frames,
                pipeline_start_ns=start_ns,
                injected_predictor=predictor,
            )
        except RuntimeError as exc:
            if predictor is not None or not _is_oom_error(exc):
                raise
            errors.append(str(exc))
            _empty_cuda_cache()
    raise RuntimeError("VGGT CUDA OOM after required retries: " + " | ".join(errors))


def _run_attempt(
    config: RGBTeacherMapConfig,
    *,
    source: RGBTeacherInput,
    frames: Sequence[RGBTeacherFrame],
    pipeline_start_ns: int,
    injected_predictor: VGGTRawClipPredictor | None,
) -> dict[str, object]:
    recorder = LatencyRecorder()
    events = LiveReplayEventRecorder()
    events.emit(
        "runtime_start",
        metadata={
            **rgb_teacher_truth_boundary(metric_scale_source=config.metric_scale_source),
            "input": str(config.input),
            "teacher": config.teacher,
            "rgb_only": config.rgb_only,
        },
    )
    teacher_start_ns = time.perf_counter_ns()
    predictor, predictor_metadata = _resolve_predictor(config, injected_predictor)
    observations, prediction_metadata = _predict_observations(
        config,
        frames=frames,
        predictor=predictor,
        output=config.output / "rgb_teacher_predictions",
    )
    teacher_runtime_ns = time.perf_counter_ns() - teacher_start_ns
    recorder.add("teacher_inference", teacher_runtime_ns)
    events.emit(
        "teacher_prediction_complete",
        latency_ns=teacher_runtime_ns,
        metadata={
            "frame_count": len(frames),
            "observation_count": len(observations),
            "teacher_name": "vggt",
        },
    )
    write_pseudo_recording(
        config.output / "rgb_teacher_recording",
        source,
        observations,
        metric_scale_source=config.metric_scale_source,
    )
    mapper, mesh_writer, source_observations, mesh_manifest = _map_observations(
        config,
        observations=observations,
        events=events,
        recorder=recorder,
    )
    surface = extract_or_empty_surface(mapper, metric_scale_source=config.metric_scale_source)
    write_sparse_tsdf_outputs(
        config.output / "sparse_tsdf",
        mapper=mapper,
        surface=surface,
        metrics=sparse_metrics(
            mapper,
            surface,
            metric_scale_source=config.metric_scale_source,
        ),
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(config.output / "surface_points.ply", surface)
    mesh_status = (
        mesh_writer.status()
        if mesh_writer is not None
        else empty_mesh_status(
            mesh_format=config.mesh_format,
            metric_scale_source=config.metric_scale_source,
        )
    )
    write_json(config.output / "mesh_status.json", mesh_status)
    recorder.add("total_pipeline", time.perf_counter_ns() - pipeline_start_ns)
    latency_report = recorder.report()
    latency_report["format_name"] = "atlas3r_rgb_teacher_mapping_latency_report"
    summary = rgb_teacher_summary(
        metric_scale_source=config.metric_scale_source,
        input_path=config.input,
        output=config.output,
        source=source,
        frames=frames,
        observations=source_observations,
        mapper=mapper,
        surface=surface,
        mesh_status=mesh_status,
        mesh_manifest=mesh_manifest,
        point_cloud_path=point_cloud_path,
        teacher_runtime_ns=teacher_runtime_ns,
        predictor_metadata=predictor_metadata,
        prediction_metadata=prediction_metadata,
        latency_report=latency_report,
    )
    eval_summary = write_rgb_teacher_eval_if_available(
        config.output,
        source=source,
        observations=source_observations,
        metric_scale_source=config.metric_scale_source,
    )
    summary["eval"] = eval_summary
    if eval_summary is not None:
        artifacts = cast(dict[str, object], summary["artifacts"])
        artifacts["rgb_teacher_eval"] = "rgb_teacher_eval.json"
        artifacts["rgb_teacher_eval_report"] = "rgb_teacher_eval.md"
    if config.export_mesh_chunks:
        require_nonzero_mesh(summary)
    events.emit(
        "runtime_complete",
        metadata={
            "mesh_chunk_count": summary["mesh_chunk_count"],
            "total_triangle_count": summary["total_triangle_count"],
            "total_vertex_count": summary["total_vertex_count"],
        },
        paths={
            "summary": "rgb_teacher_summary.json",
            "mesh_chunks": "mesh_chunks/mesh_chunk_manifest.json",
        },
    )
    write_jsonl(config.output / "live_replay_events.jsonl", events.events)
    write_json(config.output / "live_replay_latency_report.json", latency_report)
    write_json(config.output / "live_replay_summary.json", live_replay_summary(summary))
    write_json(config.output / "rgb_teacher_summary.json", summary)
    write_rgb_teacher_report(config.output / "rgb_teacher_report.md", summary)
    write_rgb_teacher_report(config.output / "live_replay_report.md", summary)
    return summary


def _predict_observations(
    config: RGBTeacherMapConfig,
    *,
    frames: Sequence[RGBTeacherFrame],
    predictor: VGGTRawClipPredictor,
    output: Path,
) -> tuple[tuple[DepthObservation, ...], dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    observations: list[DepthObservation] = []
    seen_frame_ids: set[int] = set()
    window_records: list[dict[str, object]] = []
    for window_index, window_frames in enumerate(
        _windows(
            frames,
            window_size=config.teacher_window_size,
            overlap=config.teacher_window_overlap,
        )
    ):
        raw_prediction = predictor(frames_to_vggt_clip_payload(window_frames))
        batch = teacher_prediction_to_observations(
            raw_prediction,
            frames=window_frames,
            teacher_name="vggt",
            metric_scale_source=config.metric_scale_source,
        )
        write_prediction_payload(output / f"window_{window_index:06d}.npz", batch.payload)
        new_count = 0
        for observation in batch.observations:
            if observation.frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(observation.frame_id)
            observations.append(observation)
            new_count += 1
        window_records.append(
            {
                **batch.metadata,
                "window_index": window_index,
                "frame_ids": [frame.frame_id for frame in window_frames],
                "accepted_observation_count": new_count,
            }
        )
    if not observations:
        raise ValueError("teacher prediction produced no pseudo observations")
    write_jsonl(output / "window_summaries.jsonl", window_records)
    valid_pixels = sum(int(np.count_nonzero(obs.confidence > 0.0)) for obs in observations)
    total_pixels = sum(int(obs.confidence.size) for obs in observations)
    return tuple(observations), {
        "prediction_windows": window_records,
        "pseudo_depth_valid_pixel_ratio": valid_pixels / max(total_pixels, 1),
        "window_count": len(window_records),
    }


def _map_observations(
    config: RGBTeacherMapConfig,
    *,
    observations: Sequence[DepthObservation],
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
) -> tuple[
    SparseBlockTSDFMapper,
    MeshChunkArtifactWriter | None,
    list[DepthObservation],
    dict[str, object] | None,
]:
    sparse_config = SparseTSDFConfig(
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=config.voxel_size_m * config.truncation_voxels,
        coordinate_frame=RECORDING_COORDINATE_FRAME,
        metric_scale_source=config.metric_scale_source,
        pixel_stride=config.pixel_stride,
    )
    mapper = SparseBlockTSDFMapper(sparse_config)
    mesh_writer = (
        MeshChunkArtifactWriter(
            config.output / "mesh_chunks",
            mesh_format=config.mesh_format,
            mesher_config=SparseTSDFMesherConfig(min_weight=config.mesh_min_weight),
            truth_flags=rgb_teacher_truth_boundary(
                metric_scale_source=config.metric_scale_source,
            ),
            truth_boundary=rgb_teacher_truth_boundary(
                metric_scale_source=config.metric_scale_source,
            ),
        )
        if config.export_mesh_chunks
        else None
    )
    source_observations: list[DepthObservation] = []
    pending_dirty: set[tuple[int, int, int]] = set()
    for observation in observations:
        start_ns = time.perf_counter_ns()
        stats = mapper.integrate(observation)
        latency_ns = time.perf_counter_ns() - start_ns
        recorder.add("map_update", latency_ns)
        for stage_name, stage_latency_ns in stats.stage_timings_ns.items():
            recorder.add(stage_name, stage_latency_ns)
        pending_dirty.update(stats.changed_block_coords_xyz)
        source_observations.append(observation)
        events.emit(
            "map_update",
            frame_id=observation.frame_id,
            timestamp_ns=observation.pose.timestamp_ns,
            latency_ns=latency_ns,
            metadata={
                "active_block_count": stats.active_block_count,
                "active_voxel_count": stats.active_voxel_count,
                "dirty_block_count": stats.dirty_block_count,
                "pseudo_depth_used": True,
                "pseudo_pose_used": True,
                "stage_timings_ns": stats.stage_timings_ns,
            },
        )
        if mesh_writer is not None and pending_dirty:
            _process_mesh_blocks(
                mesh_writer=mesh_writer,
                mapper=mapper,
                events=events,
                recorder=recorder,
                pending_dirty=pending_dirty,
                observation=observation,
                map_update_latency_ns=latency_ns,
            )
    mesh_manifest = None
    if mesh_writer is not None:
        mesh_manifest = mesh_writer.write_manifest(
            mapper=mapper,
            source_frame_ids_mapped=[obs.frame_id for obs in source_observations],
            mapper_backend=MAPPER_BACKEND,
        )
    return mapper, mesh_writer, source_observations, mesh_manifest


def _process_mesh_blocks(
    *,
    mesh_writer: MeshChunkArtifactWriter,
    mapper: SparseBlockTSDFMapper,
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
    pending_dirty: set[tuple[int, int, int]],
    observation: DepthObservation,
    map_update_latency_ns: int,
) -> None:
    dirty = tuple(sorted(pending_dirty))
    pending_dirty.clear()
    mesh_events, mesh_latency_ns = mesh_writer.process_dirty_blocks(
        mapper=mapper,
        dirty_block_coords_xyz=dirty,
        frame_id=observation.frame_id,
        timestamp_ns=observation.pose.timestamp_ns,
        dirty_reason="rgb_teacher_sparse_tsdf_dirty_block",
        capture_queue_depth=0,
        map_queue_depth=0,
        max_dirty_chunks=None,
    )
    recorder.add("mesh_update", mesh_latency_ns)
    recorder.add("map_mesh_update", map_update_latency_ns + mesh_latency_ns)
    for mesh_event in mesh_events:
        events.emit(
            "mesh_chunk_update",
            frame_id=observation.frame_id,
            timestamp_ns=observation.pose.timestamp_ns,
            latency_ns=cast(int, mesh_event["mesh_latency_ns"]),
            metadata=mesh_event,
            paths={
                key: str(mesh_event[key])
                for key in ("payload_npz", "payload_ply")
                if mesh_event.get(key) is not None
            },
        )


def _resolve_predictor(
    config: RGBTeacherMapConfig,
    injected_predictor: VGGTRawClipPredictor | None,
) -> tuple[VGGTRawClipPredictor, dict[str, object]]:
    if injected_predictor is not None:
        return injected_predictor, {
            "model_source": "injected_test_predictor",
            "resolved_device": config.device,
            "checkpoint": config.checkpoint,
            "image_size": config.image_size,
        }
    if config.teacher == "fixture-vggt":
        return _fixture_vggt_predictor, {
            "model_source": "fixture_vggt",
            "resolved_device": "cpu",
            "checkpoint": None,
            "image_size": config.image_size,
        }
    status = get_vggt_status(vggt_repo=config.vggt_repo, checkpoint=config.checkpoint)
    if status.availability != "available":
        raise ExternalTeacherDependencyError(
            status.display_name,
            status.reason or "VGGT is not importable or configured",
            status.install_hint,
        )
    loaded = load_vggt_predictor(
        vggt_repo=config.vggt_repo,
        checkpoint=config.checkpoint,
        device=config.device,
        image_size=config.image_size,
    )
    return loaded.predictor, {
        "model_source": loaded.model_source,
        "resolved_device": loaded.resolved_device,
        "checkpoint": loaded.checkpoint,
        "image_size": loaded.input_resolution,
    }


def _fixture_vggt_predictor(clip_payload: Mapping[str, Any]) -> Mapping[str, object]:
    images = np.asarray(clip_payload["images_rgb_u8"], dtype=np.uint8)
    frame_count, height, width = images.shape[:3]
    depth = np.ones((frame_count, height, width), dtype=np.float32)
    confidence = np.full_like(depth, 2.0, dtype=np.float32)
    T_world_camera = np.repeat(np.eye(4, dtype=np.float32)[np.newaxis, :, :], frame_count, axis=0)
    for index in range(frame_count):
        T_world_camera[index, 0, 3] = np.float32(index * 0.05)
    return {
        "depth": depth,
        "depth_conf": confidence,
        "extrinsic": np.linalg.inv(T_world_camera).astype(np.float32),
        "intrinsic": np.asarray(clip_payload["K"], dtype=np.float32),
    }


def _windows(
    frames: Sequence[RGBTeacherFrame],
    *,
    window_size: int,
    overlap: int,
) -> Iterator[tuple[RGBTeacherFrame, ...]]:
    step = window_size - overlap
    start = 0
    while start < len(frames):
        end = min(start + window_size, len(frames))
        yield tuple(frames[start:end])
        if end == len(frames):
            break
        start += step


def _attempt_configs(config: RGBTeacherMapConfig) -> tuple[RGBTeacherMapConfig, ...]:
    attempts = [config]
    if config.image_size > 384:
        attempts.append(replace(config, image_size=384))
    if config.teacher_window_size > 8:
        attempts.append(replace(config, teacher_window_size=8, teacher_window_overlap=0))
    if config.max_frames is None or config.max_frames > 16:
        attempts.append(
            replace(config, max_frames=16, teacher_window_size=8, teacher_window_overlap=0)
        )
    deduped: list[RGBTeacherMapConfig] = []
    keys: set[tuple[int | None, int, int, int]] = set()
    for attempt in attempts:
        key = (
            attempt.max_frames,
            attempt.image_size,
            attempt.teacher_window_size,
            attempt.teacher_window_overlap,
        )
        if key not in keys:
            keys.add(key)
            deduped.append(attempt)
    return tuple(deduped)


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas" in text


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
