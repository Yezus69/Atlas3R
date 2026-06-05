"""Runtime bridge from learned SMGT-tiny RGB predictions to sparse TSDF chunks."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.mesh_artifacts import MeshChunkArtifactWriter
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper, SparseTSDFConfig
from atlas3r.mapping.sparse_tsdf_artifacts import write_sparse_tsdf_outputs
from atlas3r.mapping.sparse_tsdf_meshing import SparseTSDFMesherConfig
from atlas3r.models.smgt import load_smgt_tiny_checkpoint, smgt_tiny_truth_boundary
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.recording.schema import RECORDING_COORDINATE_FRAME
from atlas3r.runtime.live_replay_types import LiveReplayEventRecorder
from atlas3r.runtime.rgb_student_eval import write_rgb_student_eval_if_available
from atlas3r.runtime.rgb_student_gating import (
    StudentMapGateConfig,
    aggregate_student_gate_stats,
    resolve_student_map_valid_policy,
    sanitize_student_confidence,
    sanitize_student_depth,
    sanitize_student_sigma,
    student_mapping_valid_mask,
)
from atlas3r.runtime.rgb_student_outputs import (
    MAPPER_BACKEND,
    empty_student_mesh_status,
    extract_or_empty_student_surface,
    require_nonzero_student_mesh,
    student_live_replay_summary,
    student_sparse_metrics,
    student_summary,
    write_student_report,
)
from atlas3r.runtime.rgb_student_preprocess import resize_nearest, scale_intrinsics
from atlas3r.runtime.rgb_teacher_inputs import (
    RGBTeacherFrame,
    load_rgb_teacher_input,
    select_rgb_teacher_frames,
)
from atlas3r.runtime.student_map_report_common import write_json, write_jsonl
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply
from atlas3r.training.torch_runtime import require_torch


@dataclass(frozen=True)
class RGBStudentMapConfig:
    input: Path
    output: Path
    checkpoint: Path
    device: str = "cuda"
    max_frames: int | None = 64
    frame_stride: int = 1
    clip_length: int = 8
    clip_overlap: int = 4
    image_size: tuple[int, int] | None = None
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0
    pixel_stride: int = 12
    export_mesh_chunks: bool = False
    mesh_format: str = "npz"
    mesh_min_weight: float = 0.0
    export_point_cloud: bool = False
    rgb_only: bool = False
    student_confidence_threshold: float = 0.30
    student_min_depth_m: float | None = None
    student_max_depth_m: float | None = None
    student_max_sigma_m: float | None = None
    student_dynamic_threshold: float = 0.50
    student_map_valid_policy: str | None = None


def run_rgb_student_mapping(config: RGBStudentMapConfig) -> dict[str, object]:
    """Run learned SMGT-tiny RGB predictions through sparse TSDF mapping."""

    _validate_config(config)
    torch = require_torch()
    truth = smgt_tiny_truth_boundary(pseudo_training_used=True)
    start_ns = time.perf_counter_ns()
    loaded = load_smgt_tiny_checkpoint(config.checkpoint, device=config.device)
    model = loaded.model
    checkpoint = loaded.checkpoint
    model_config = cast(dict[str, Any], checkpoint["model_config"])["config"]
    gate_config = _resolved_gate_config(config, cast(Mapping[str, object], model_config))
    target_size = config.image_size or (
        int(model_config["image_height"]),
        int(model_config["image_width"]),
    )
    source = load_rgb_teacher_input(config.input)
    frames = select_rgb_teacher_frames(
        source.frames,
        max_frames=config.max_frames,
        frame_stride=config.frame_stride,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    recorder = LatencyRecorder()
    events = LiveReplayEventRecorder()
    events.emit(
        "runtime_start",
        metadata={
            **truth,
            "input": str(config.input),
            "checkpoint": str(config.checkpoint),
            "device": loaded.device,
            "rgb_only": config.rgb_only,
        },
    )
    observations, prediction_metadata = _predict_observations(
        model=model,
        torch=torch,
        device=loaded.device,
        frames=frames,
        target_size=target_size,
        clip_length=config.clip_length,
        clip_overlap=config.clip_overlap,
        output=config.output / "rgb_student_predictions",
        recorder=recorder,
        truth_boundary=truth,
        gate_config=gate_config,
    )
    mapper, mesh_writer, mesh_manifest = _map_observations(
        config,
        observations=observations,
        events=events,
        recorder=recorder,
        truth_boundary=truth,
    )
    surface = extract_or_empty_student_surface(mapper, truth_boundary=truth)
    write_sparse_tsdf_outputs(
        config.output / "sparse_tsdf",
        mapper=mapper,
        surface=surface,
        metrics=student_sparse_metrics(mapper, surface, truth_boundary=truth),
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(config.output / "surface_points.ply", surface)
    mesh_status = (
        mesh_writer.status()
        if mesh_writer is not None
        else empty_student_mesh_status(mesh_format=config.mesh_format, truth_boundary=truth)
    )
    write_json(config.output / "mesh_status.json", mesh_status)
    recorder.add("total_pipeline", time.perf_counter_ns() - start_ns)
    latency_report = recorder.report()
    latency_report["format_name"] = "atlas3r_rgb_student_mapping_latency_report"
    eval_summary = write_rgb_student_eval_if_available(
        config.output,
        source=source,
        observations=observations,
        truth_boundary=truth,
    )
    summary = student_summary(
        config=config,
        source_frame_count=len(source.frames),
        frames=frames,
        observations=observations,
        mapper=mapper,
        surface=surface,
        mesh_status=mesh_status,
        mesh_manifest=mesh_manifest,
        point_cloud_path=point_cloud_path,
        prediction_metadata=prediction_metadata,
        latency_report=latency_report,
        checkpoint=checkpoint,
        resolved_device=loaded.device,
        eval_summary=eval_summary,
        truth_boundary=truth,
    )
    if config.export_mesh_chunks:
        require_nonzero_student_mesh(config.output, summary, config.mesh_format)
    events.emit(
        "runtime_complete",
        metadata={
            "mesh_chunk_count": summary["mesh_chunk_count"],
            "total_vertex_count": summary["total_vertex_count"],
            "total_triangle_count": summary["total_triangle_count"],
        },
        paths={
            "summary": "rgb_student_summary.json",
            "mesh_chunks": "mesh_chunks/mesh_chunk_manifest.json",
        },
    )
    write_jsonl(config.output / "live_replay_events.jsonl", events.events)
    write_json(config.output / "live_replay_latency_report.json", latency_report)
    write_json(config.output / "live_replay_summary.json", student_live_replay_summary(summary))
    write_json(config.output / "rgb_student_summary.json", summary)
    write_student_report(config.output / "rgb_student_report.md", summary)
    write_student_report(config.output / "live_replay_report.md", summary)
    return summary


def _predict_observations(
    *,
    model: Any,
    torch: Any,
    device: str,
    frames: Sequence[RGBTeacherFrame],
    target_size: tuple[int, int],
    clip_length: int,
    clip_overlap: int,
    output: Path,
    recorder: LatencyRecorder,
    truth_boundary: Mapping[str, object],
    gate_config: StudentMapGateConfig,
) -> tuple[tuple[DepthObservation, ...], dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    observations: list[DepthObservation] = []
    seen: set[int] = set()
    global_poses: dict[int, npt.NDArray[np.float32]] = {}
    gate_stats: list[dict[str, object]] = []
    for window_index, window_frames in enumerate(_windows(frames, clip_length, clip_overlap)):
        images_np, K_np, resized_rgb = _window_arrays(window_frames, target_size)
        images = torch.from_numpy(images_np).to(device)
        K = torch.from_numpy(K_np).to(device)
        start_ns = time.perf_counter_ns()
        with torch.no_grad():
            prediction = model(images, K)
        inference_ns = time.perf_counter_ns() - start_ns
        recorder.add("student_inference", inference_ns)
        payload = _prediction_payload(prediction)
        np.savez_compressed(output / f"window_{window_index:06d}.npz", **payload)
        anchor_id = window_frames[0].frame_id
        anchor_pose = global_poses.get(anchor_id)
        if anchor_pose is None:
            anchor_pose = payload["T_world_camera"][0, 0].astype(np.float32, copy=True)
        T_window_anchor_inv = np.linalg.inv(payload["T_world_camera"][0, 0]).astype(np.float32)
        for slot, frame in enumerate(window_frames):
            if frame.frame_id in seen:
                continue
            T_global = (
                anchor_pose @ T_window_anchor_inv @ payload["T_world_camera"][0, slot]
            ).astype(np.float32)
            global_poses[frame.frame_id] = T_global
            seen.add(frame.frame_id)
            observation = _observation_from_prediction_slot(
                frame=frame,
                rgb_u8=resized_rgb[slot],
                K=K_np[0, slot],
                depth=payload["depth_m"][0, slot],
                sigma=payload["depth_sigma_m"][0, slot],
                confidence=payload["confidence"][0, slot],
                dynamic_probability=payload["dynamic_probability"][0, slot],
                T_world_camera=T_global,
                checkpoint_truth=truth_boundary,
                gate_config=gate_config,
            )
            observations.append(observation)
            gate_stats.append(
                cast(dict[str, object], observation.pose.diagnostics["student_mapping_gate"])
            )
    if not observations:
        raise ValueError("SMGTTiny prediction produced no observations")
    aggregate_gate = aggregate_student_gate_stats(gate_stats)
    return tuple(observations), {
        "prediction_window_count": len(tuple(_windows(frames, clip_length, clip_overlap))),
        "student_depth_valid_pixel_ratio": aggregate_gate["mapped_pixel_ratio"],
        "student_mapping_gate": aggregate_gate,
        **aggregate_gate,
        "student_pose_count": len(observations),
        "student_depth_count": len(observations),
        "student_intrinsics_count": len(observations),
    }


def _map_observations(
    config: RGBStudentMapConfig,
    *,
    observations: Sequence[DepthObservation],
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
    truth_boundary: Mapping[str, object],
) -> tuple[SparseBlockTSDFMapper, MeshChunkArtifactWriter | None, dict[str, object] | None]:
    mapper = SparseBlockTSDFMapper(
        SparseTSDFConfig(
            voxel_size_m=config.voxel_size_m,
            truncation_distance_m=config.voxel_size_m * config.truncation_voxels,
            coordinate_frame=RECORDING_COORDINATE_FRAME,
            metric_scale_source="student_rgb_prior_unverified",
            pixel_stride=config.pixel_stride,
        )
    )
    mesh_writer = (
        MeshChunkArtifactWriter(
            config.output / "mesh_chunks",
            mesh_format=config.mesh_format,
            mesher_config=SparseTSDFMesherConfig(min_weight=config.mesh_min_weight),
            truth_flags=truth_boundary,
            truth_boundary=truth_boundary,
        )
        if config.export_mesh_chunks
        else None
    )
    pending_dirty: set[tuple[int, int, int]] = set()
    for observation in observations:
        start_ns = time.perf_counter_ns()
        stats = mapper.integrate(observation)
        map_latency_ns = time.perf_counter_ns() - start_ns
        recorder.add("map_update", map_latency_ns)
        for stage_name, stage_latency_ns in stats.stage_timings_ns.items():
            recorder.add(stage_name, stage_latency_ns)
        pending_dirty.update(stats.changed_block_coords_xyz)
        events.emit(
            "map_update",
            frame_id=observation.frame_id,
            timestamp_ns=observation.pose.timestamp_ns,
            latency_ns=map_latency_ns,
            metadata={
                **truth_boundary,
                "active_block_count": stats.active_block_count,
                "active_voxel_count": stats.active_voxel_count,
                "dirty_block_count": stats.dirty_block_count,
                "stage_timings_ns": stats.stage_timings_ns,
                "mapped_pixel_ratio": observation.pose.diagnostics.get("mapped_pixel_ratio"),
                "pose_translation_norm_m": float(
                    np.linalg.norm(observation.pose.camera_center_world_m.astype(np.float64))
                ),
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
                map_update_latency_ns=map_latency_ns,
            )
    mesh_manifest = None
    if mesh_writer is not None:
        mesh_manifest = mesh_writer.write_manifest(
            mapper=mapper,
            source_frame_ids_mapped=[obs.frame_id for obs in observations],
            mapper_backend=MAPPER_BACKEND,
        )
    return mapper, mesh_writer, mesh_manifest


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
        dirty_reason="rgb_student_sparse_tsdf_dirty_block",
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


def _observation_from_prediction_slot(
    *,
    frame: RGBTeacherFrame,
    rgb_u8: npt.NDArray[np.uint8],
    K: npt.NDArray[np.float32],
    depth: npt.NDArray[np.float32],
    sigma: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    dynamic_probability: npt.NDArray[np.float32],
    T_world_camera: npt.NDArray[np.float32],
    checkpoint_truth: Mapping[str, object],
    gate_config: StudentMapGateConfig,
) -> DepthObservation:
    sanitized_depth = sanitize_student_depth(depth)
    sanitized_sigma = sanitize_student_sigma(sigma, fallback_m=gate_config.max_sigma_m or 1.0)
    sanitized_confidence = sanitize_student_confidence(confidence)
    sanitized_dynamic = sanitize_student_confidence(dynamic_probability)
    gate = student_mapping_valid_mask(
        depth_m=depth,
        depth_sigma_m=sigma,
        confidence=confidence,
        dynamic_probability=sanitized_dynamic,
        config=gate_config,
    )
    valid = gate.valid_mask
    mean_sigma = float(np.mean(sanitized_sigma[valid])) if np.any(valid) else 1.0
    covariance = np.diag(np.full(6, max(mean_sigma * mean_sigma, 1e-8), dtype=np.float32))
    diagnostics = dict(gate.stats)
    return DepthObservation(
        frame_id=frame.frame_id,
        camera=CameraModel(
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            K=K.astype(np.float32, copy=True),
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="smgt_tiny_student_or_input_intrinsics",
        ),
        pose=PoseEstimate(
            frame_id=frame.frame_id,
            timestamp_ns=int(round(frame.timestamp_s * 1_000_000_000.0)),
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
            covariance_6x6=covariance.astype(np.float32),
            confidence=float(np.mean(sanitized_confidence[valid])) if np.any(valid) else 0.0,
            tracking_state="OK" if np.any(valid) else "LOW_CONFIDENCE",
            scale_source="rgb_prior",
            diagnostics={
                "source": "smgt_tiny_student_rgb_checkpoint",
                "truth_boundary": dict(checkpoint_truth),
                "source_rgb_path": frame.source_path,
                "student_mapping_gate": diagnostics,
                "mapped_pixel_ratio": diagnostics["mapped_pixel_ratio"],
            },
        ),
        depth_m=sanitized_depth,
        depth_sigma_m=sanitized_sigma,
        confidence=sanitized_confidence,
        static_mask=valid,
        object_id=None,
        rgb_u8=rgb_u8.copy(),
        source="smgt_tiny_student_rgb_checkpoint",
    )


def _window_arrays(
    frames: Sequence[RGBTeacherFrame],
    target_size: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], tuple[npt.NDArray[np.uint8], ...]]:
    height, width = target_size
    rgbs: list[npt.NDArray[np.uint8]] = []
    intrinsics: list[npt.NDArray[np.float32]] = []
    for frame in frames:
        resized = resize_nearest(frame.rgb_u8, height=height, width=width)
        rgbs.append(resized)
        original_size = (int(frame.rgb_u8.shape[0]), int(frame.rgb_u8.shape[1]))
        intrinsics.append(scale_intrinsics(frame.K, original_size, target_size))
    rgb_float = np.stack(rgbs, axis=0).astype(np.float32) / 255.0
    images = np.transpose(rgb_float, (0, 3, 1, 2))[np.newaxis, ...].copy()
    return images, np.stack(intrinsics, axis=0)[np.newaxis, ...].astype(np.float32), tuple(rgbs)


def _windows(
    frames: Sequence[RGBTeacherFrame],
    clip_length: int,
    clip_overlap: int,
) -> tuple[tuple[RGBTeacherFrame, ...], ...]:
    stride = clip_length - clip_overlap
    windows: list[tuple[RGBTeacherFrame, ...]] = []
    for start in range(0, len(frames), stride):
        window = tuple(frames[start : start + clip_length])
        if window:
            windows.append(window)
        if start + clip_length >= len(frames):
            break
    return tuple(windows)


def _prediction_payload(prediction: Mapping[str, Any]) -> dict[str, npt.NDArray[Any]]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in prediction.items()
        if hasattr(value, "detach")
    }


def _resolved_gate_config(
    config: RGBStudentMapConfig,
    model_config: Mapping[str, object],
) -> StudentMapGateConfig:
    if config.student_min_depth_m is None:
        raw_min_depth = model_config.get("min_depth_m", 0.05)
        if not isinstance(raw_min_depth, int | float) or isinstance(raw_min_depth, bool):
            raise ValueError("checkpoint model_config.min_depth_m must be numeric")
        min_depth = float(raw_min_depth)
    else:
        min_depth = float(config.student_min_depth_m)
    policy = resolve_student_map_valid_policy(
        config.student_map_valid_policy,
        config.student_max_sigma_m,
    )
    return StudentMapGateConfig(
        min_depth_m=min_depth,
        max_depth_m=config.student_max_depth_m,
        confidence_threshold=config.student_confidence_threshold,
        max_sigma_m=config.student_max_sigma_m,
        dynamic_threshold=config.student_dynamic_threshold,
        policy=policy,
    )


def _validate_config(config: RGBStudentMapConfig) -> None:
    if not config.rgb_only:
        raise ValueError("--rgb-only is required for runtime map-rgb-student")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    if config.frame_stride <= 0:
        raise ValueError("frame_stride: must be positive")
    if config.clip_length <= 0:
        raise ValueError("clip_length: must be positive")
    if config.clip_overlap < 0 or config.clip_overlap >= config.clip_length:
        raise ValueError("clip_overlap: must be in [0, clip_length)")
    if config.image_size is not None and (config.image_size[0] <= 0 or config.image_size[1] <= 0):
        raise ValueError("image_size: dimensions must be positive")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if config.truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")
    if config.pixel_stride <= 0:
        raise ValueError("pixel_stride: must be positive")
    if config.mesh_format not in {"npz", "ply", "both"}:
        raise ValueError("mesh_format: must be npz, ply, or both")
    if config.mesh_min_weight < 0.0:
        raise ValueError("mesh_min_weight: must be non-negative")
    if config.student_min_depth_m is not None and config.student_min_depth_m <= 0.0:
        raise ValueError("student_min_depth_m: must be positive when provided")
    if config.student_max_depth_m is not None and config.student_max_depth_m <= 0.0:
        raise ValueError("student_max_depth_m: must be positive when provided")
    if not 0.0 <= config.student_confidence_threshold <= 1.0:
        raise ValueError("student_confidence_threshold: must be in [0, 1]")
    if config.student_max_sigma_m is not None and config.student_max_sigma_m <= 0.0:
        raise ValueError("student_max_sigma_m: must be positive when provided")
    if not 0.0 <= config.student_dynamic_threshold <= 1.0:
        raise ValueError("student_dynamic_threshold: must be in [0, 1]")
    if config.student_map_valid_policy is not None:
        resolve_student_map_valid_policy(
            config.student_map_valid_policy,
            config.student_max_sigma_m,
        )


__all__ = ["RGBStudentMapConfig", "run_rgb_student_mapping"]
