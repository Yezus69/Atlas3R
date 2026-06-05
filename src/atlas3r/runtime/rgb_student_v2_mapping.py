"""Runtime bridge from SMGT-small-v2 RGB predictions to sparse TSDF chunks."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from atlas3r.models.smgt import load_smgt_small_v2_checkpoint
from atlas3r.runtime.live_replay_types import LiveReplayEventRecorder
from atlas3r.runtime.rgb_student_eval import write_rgb_student_eval_if_available
from atlas3r.runtime.rgb_student_gating import StudentMapGateConfig
from atlas3r.runtime.rgb_student_mapping import (
    RGBStudentMapConfig,
    _map_observations,
    _predict_observations,
    _resolved_gate_config,
    _validate_config,
)
from atlas3r.runtime.rgb_student_outputs import (
    empty_student_mesh_status,
    extract_or_empty_student_surface,
    require_nonzero_student_mesh,
    student_live_replay_summary,
    student_sparse_metrics,
    student_summary,
    write_student_report,
)
from atlas3r.runtime.rgb_teacher_inputs import load_rgb_teacher_input, select_rgb_teacher_frames
from atlas3r.runtime.student_map_report_common import write_json, write_jsonl
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply
from atlas3r.training.torch_runtime import require_torch


@dataclass(frozen=True)
class RGBStudentV2MapConfig:
    input: Path
    output: Path
    checkpoint: Path
    device: str = "cuda"
    max_frames: int | None = 120
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
    calibration: Path | None = None
    student_confidence_threshold: float = 0.30
    student_min_depth_m: float | None = None
    student_max_depth_m: float | None = None
    student_max_sigma_m: float | None = None
    student_dynamic_threshold: float = 0.50
    student_map_valid_policy: str | None = None


def run_rgb_student_v2_mapping(config: RGBStudentV2MapConfig) -> dict[str, object]:
    torch = require_torch()
    base_config = _base_config_with_calibration(config, calibration=None)
    _validate_config(base_config)
    start_ns = time.perf_counter_ns()
    loaded = load_smgt_small_v2_checkpoint(config.checkpoint, device=config.device)
    checkpoint = loaded.checkpoint
    truth = dict(cast(dict[str, object], checkpoint["truth_boundary"]))
    calibration = _load_calibration(config, checkpoint)
    base_config = _base_config_with_calibration(config, calibration=calibration)
    model_config = cast(dict[str, Any], checkpoint["model_config"])["config"]
    gate_config = _resolved_v2_gate_config(
        base_config,
        cast(Mapping[str, object], model_config),
        calibration,
    )
    target_size = base_config.image_size or (
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
            "student_model_family": "SMGTSmallV2",
        },
    )
    observations, prediction_metadata = _predict_observations(
        model=loaded.model,
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
        student_source_name="smgt_small_v2_student_rgb_checkpoint",
        camera_source_name="smgt_small_v2_student_or_input_intrinsics",
    )
    prediction_metadata["student_model_family"] = "SMGTSmallV2"
    prediction_metadata["calibration"] = calibration
    mapper, mesh_writer, mesh_manifest = _map_observations(
        base_config,
        observations=observations,
        events=events,
        recorder=recorder,
        truth_boundary=truth,
    )
    surface = extract_or_empty_student_surface(mapper, truth_boundary=truth)
    from atlas3r.mapping.sparse_tsdf_artifacts import write_sparse_tsdf_outputs

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
    latency_report["format_name"] = "atlas3r_rgb_student_v2_mapping_latency_report"
    eval_summary = write_rgb_student_eval_if_available(
        config.output,
        source=source,
        observations=observations,
        truth_boundary=truth,
    )
    summary = student_summary(
        config=base_config,
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
    summary["format_name"] = "atlas3r_rgb_student_v2_mapping_summary"
    summary["student_model_family"] = "SMGTSmallV2"
    summary["calibration"] = calibration
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


def _base_config_with_calibration(
    config: RGBStudentV2MapConfig,
    *,
    calibration: Mapping[str, object] | None,
) -> RGBStudentMapConfig:
    confidence = config.student_confidence_threshold
    max_sigma = config.student_max_sigma_m
    policy = config.student_map_valid_policy
    if calibration:
        confidence = float(cast(float, calibration["confidence_threshold"]))
        max_sigma = float(cast(float, calibration["max_sigma_m"]))
        policy = "confidence_sigma"
    return RGBStudentMapConfig(
        input=config.input,
        output=config.output,
        checkpoint=config.checkpoint,
        device=config.device,
        max_frames=config.max_frames,
        frame_stride=config.frame_stride,
        clip_length=config.clip_length,
        clip_overlap=config.clip_overlap,
        image_size=config.image_size,
        voxel_size_m=config.voxel_size_m,
        truncation_voxels=config.truncation_voxels,
        pixel_stride=config.pixel_stride,
        export_mesh_chunks=config.export_mesh_chunks,
        mesh_format=config.mesh_format,
        mesh_min_weight=config.mesh_min_weight,
        export_point_cloud=config.export_point_cloud,
        rgb_only=config.rgb_only,
        student_confidence_threshold=confidence,
        student_min_depth_m=config.student_min_depth_m,
        student_max_depth_m=config.student_max_depth_m,
        student_max_sigma_m=max_sigma,
        student_dynamic_threshold=config.student_dynamic_threshold,
        student_map_valid_policy=policy,
    )


def _resolved_v2_gate_config(
    config: RGBStudentMapConfig,
    model_config: Mapping[str, object],
    calibration: Mapping[str, object] | None,
) -> StudentMapGateConfig:
    if calibration:
        return StudentMapGateConfig(
            min_depth_m=float(cast(Any, model_config.get("min_depth_m", 0.05))),
            max_depth_m=config.student_max_depth_m,
            confidence_threshold=float(cast(float, calibration["confidence_threshold"])),
            max_sigma_m=float(cast(float, calibration["max_sigma_m"])),
            dynamic_threshold=config.student_dynamic_threshold,
            policy="confidence_sigma",
        )
    return _resolved_gate_config(config, model_config)


def _load_calibration(
    config: RGBStudentV2MapConfig,
    checkpoint: Mapping[str, Any],
) -> dict[str, object] | None:
    if config.calibration is not None:
        with config.calibration.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{config.calibration}: calibration must be a JSON object")
        return cast(dict[str, object], payload)
    payload = checkpoint.get("calibration")
    if isinstance(payload, dict) and payload.get("confidence_threshold") is not None:
        return cast(dict[str, object], payload)
    return None


__all__ = ["RGBStudentV2MapConfig", "run_rgb_student_v2_mapping"]
