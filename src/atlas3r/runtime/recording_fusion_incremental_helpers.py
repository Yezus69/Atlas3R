"""Shared helpers for measured-recording incremental TSDF backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import TSDFSurface, TSDFVolume, extract_tsdf_surface
from atlas3r.mapping.incremental_tsdf import PERSISTENT_CPU_UPDATE_IMPLEMENTATION
from atlas3r.mapping.mesh_sidecar import write_tsdf_surface_mesh_sidecar_from_artifacts
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.world_map_sidecar import write_tsdf_world_map_sidecar_from_artifacts
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import Atlas3RRecording, RecordingFrame
from atlas3r.runtime.recording_fusion import (
    FuseRecordingConfig,
    _tsdf_metrics,
    _with_recording_metadata,
)
from atlas3r.runtime.student_map_report_common import DIAGNOSTIC_TRUTH_FLAGS
from atlas3r.runtime.student_map_reports import LatencyRecorder

CPU_REBUILD_UPDATE_IMPLEMENTATION = "cpu_tsdf_full_rebuild_per_selected_keyframe"
CPU_REBUILD_BACKEND = "cpu-rebuild"
CPU_PERSISTENT_BACKEND = "cpu-persistent"


@dataclass(frozen=True)
class _MapUpdate:
    volume: TSDFVolume
    surface: TSDFSurface
    latency_ns: int


@dataclass(frozen=True)
class _PreloadedObservation:
    frame: RecordingFrame
    observation: DepthObservation
    load_latency_ns: int


def _selected_and_dropped_frames(
    recording: Atlas3RRecording,
    config: FuseRecordingConfig,
) -> tuple[tuple[RecordingFrame, bool], ...]:
    selected_count = 0
    records: list[tuple[RecordingFrame, bool]] = []
    for index, frame in enumerate(recording.frames):
        if config.max_frames is not None and selected_count >= config.max_frames:
            break
        selected = index % config.keyframe_stride == 0
        records.append((frame, selected))
        if selected:
            selected_count += 1
    return tuple(records)


def _preload_selected_observations(
    *,
    recording: Atlas3RRecording,
    selected_records: tuple[tuple[RecordingFrame, bool], ...],
    config: FuseRecordingConfig,
    recorder: LatencyRecorder,
) -> tuple[_PreloadedObservation, ...]:
    preloaded: list[_PreloadedObservation] = []
    for frame, selected in selected_records:
        if not selected:
            continue
        load_start_ns = time.perf_counter_ns()
        observation = observation_from_recording_frame(
            recording,
            frame,
            depth_sigma_floor_m=max(0.01, config.voxel_size_m * 0.25),
        )
        load_latency_ns = time.perf_counter_ns() - load_start_ns
        recorder.add("per_frame_observation_load", load_latency_ns)
        preloaded.append(
            _PreloadedObservation(
                frame=frame,
                observation=observation,
                load_latency_ns=load_latency_ns,
            )
        )
    return tuple(preloaded)


def _rebuild_cpu_tsdf(
    config: FuseRecordingConfig,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
    backend: str,
) -> _MapUpdate:
    start_ns = time.perf_counter_ns()
    truncation_distance_m = config.voxel_size_m * config.truncation_voxels
    grid_min, grid_max = grid_bounds_from_observations(
        observations,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    volume = integrate_observations_to_volume(
        observations,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    surface = _with_incremental_metadata(
        extract_tsdf_surface(volume),
        recording=recording,
        observations=observations,
        backend=backend,
        update_implementation=CPU_REBUILD_UPDATE_IMPLEMENTATION,
    )
    return _MapUpdate(volume=volume, surface=surface, latency_ns=time.perf_counter_ns() - start_ns)


def _write_final_tsdf_artifacts(
    config: FuseRecordingConfig,
    output: Path,
    update: _MapUpdate,
    *,
    backend: str,
    update_implementation: str,
) -> Path:
    tsdf_dir = output / "tsdf"
    metrics = _tsdf_metrics(update.surface)
    metrics.update(
        {
            "backend": backend,
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "update_implementation": update_implementation,
        }
    )
    write_tsdf_outputs(tsdf_dir, update.volume, update.surface, metrics)
    chunk_id = (
        "phase6c_persistent_recording_cpu_tsdf_surface"
        if backend == CPU_PERSISTENT_BACKEND
        else "phase6b_incremental_recording_cpu_tsdf_surface"
    )
    write_tsdf_surface_mesh_sidecar_from_artifacts(tsdf_dir, chunk_id=chunk_id)
    write_tsdf_world_map_sidecar_from_artifacts(tsdf_dir)
    return tsdf_dir


def _compare_persistent_to_batch_cpu_tsdf(
    *,
    backend: str,
    observations: tuple[DepthObservation, ...],
    persistent_volume: TSDFVolume,
    persistent_surface: TSDFSurface,
    grid_min_world_m: npt.NDArray[np.float64],
    grid_max_world_m: npt.NDArray[np.float64],
    voxel_size_m: float,
    truncation_distance_m: float,
) -> dict[str, object]:
    batch_volume = integrate_observations_to_volume(
        observations,
        grid_min_world_m=grid_min_world_m,
        grid_max_world_m=grid_max_world_m,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    batch_surface = extract_tsdf_surface(batch_volume)
    common_observed = (persistent_volume.weight > 0.0) & (batch_volume.weight > 0.0)
    tsdf_abs_delta = np.abs(
        persistent_volume.tsdf[common_observed].astype(np.float64)
        - batch_volume.tsdf[common_observed].astype(np.float64)
    )
    weight_abs_delta = np.abs(
        persistent_volume.weight.astype(np.float64) - batch_volume.weight.astype(np.float64)
    )
    observed_persistent = int(np.count_nonzero(persistent_volume.weight > 0.0))
    observed_batch = int(np.count_nonzero(batch_volume.weight > 0.0))
    surface_count_persistent = int(persistent_surface.points_world_m.shape[0])
    surface_count_batch = int(batch_surface.points_world_m.shape[0])
    max_tsdf_delta = _max_or_none(tsdf_abs_delta)
    max_weight_delta = _max_or_none(weight_abs_delta)
    same_grid_shape = persistent_volume.tsdf.shape == batch_volume.tsdf.shape
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "backend": backend,
        "batch_backend": "batch_cpu_tsdf_reference",
        "common_observed_voxel_count": int(np.count_nonzero(common_observed)),
        "format_name": "atlas3r_phase6c_backend_comparison",
        "format_version": 1,
        "grid_shape": list(persistent_volume.tsdf.shape),
        "known_limitations": [
            "Comparison uses the same fixed offline grid and selected measured observations.",
            "This is a numerical backend-equivalence check, not an accuracy report.",
            "No realtime claim is made from this comparison.",
        ],
        "matches_batch_within_tolerance": bool(
            same_grid_shape
            and max_tsdf_delta is not None
            and max_tsdf_delta <= 1.0e-6
            and max_weight_delta is not None
            and max_weight_delta <= 1.0e-5
            and observed_persistent == observed_batch
            and surface_count_persistent == surface_count_batch
        ),
        "metric_family": "phase6c_persistent_vs_batch_cpu_tsdf",
        "mean_abs_tsdf_delta_common_observed": _mean_or_none(tsdf_abs_delta),
        "mean_abs_weight_delta": _mean_or_none(weight_abs_delta),
        "max_abs_tsdf_delta_common_observed": max_tsdf_delta,
        "max_abs_weight_delta": max_weight_delta,
        "observed_voxel_count_batch": observed_batch,
        "observed_voxel_count_delta": observed_persistent - observed_batch,
        "observed_voxel_count_persistent": observed_persistent,
        "point_cloud_count_delta": surface_count_persistent - surface_count_batch,
        "same_grid_shape": same_grid_shape,
        "source_frame_ids": [observation.frame_id for observation in observations],
        "surface_point_count_batch": surface_count_batch,
        "surface_point_count_delta": surface_count_persistent - surface_count_batch,
        "surface_point_count_persistent": surface_count_persistent,
        "tolerances": {
            "tsdf_abs_atol_common_observed": 1.0e-6,
            "weight_abs_atol": 1.0e-5,
        },
        "update_implementation": PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
    }


def _with_incremental_metadata(
    surface: TSDFSurface,
    *,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
    backend: str,
    update_implementation: str,
) -> TSDFSurface:
    base = _with_recording_metadata(surface, recording, observations)
    metadata = dict(base.metadata)
    metadata["artifact_type"] = (
        "phase6c_recording_cpu_persistent_tsdf_surface_points"
        if backend == CPU_PERSISTENT_BACKEND
        else metadata.get("artifact_type", "phase6b_recording_cpu_tsdf_surface_points")
    )
    metadata["backend"] = backend
    metadata["fixed_bounds_precomputed_offline"] = backend == CPU_PERSISTENT_BACKEND
    metadata["update_implementation"] = update_implementation
    metadata["flags"] = _stable_strings(
        [
            *[str(flag) for flag in metadata.get("flags", [])],
            "phase6c_true_persistent_incremental_tsdf"
            if backend == CPU_PERSISTENT_BACKEND
            else "phase6b_cpu_rebuild_incremental_tsdf",
            "observed_surface_points",
            "not_completed_surface",
            "not_accuracy_report",
            "not_performance_report",
            "not_realtime_claim",
        ]
    )
    return TSDFSurface(
        points_world_m=base.points_world_m,
        confidence=base.confidence,
        uncertainty_m=base.uncertainty_m,
        voxel_indices_xyz=base.voxel_indices_xyz,
        metadata=metadata,
    )


def _per_frame_event(
    *,
    backend: str,
    frame: RecordingFrame,
    selected_keyframe: bool,
    dropped_keyframe: bool,
    current_observation_count: int,
    observed_voxel_count: int,
    update_implementation: str,
    observation_load_latency_ns: int = 0,
    map_update_latency_ns: int = 0,
    observation_array_bytes: int = 0,
    tsdf_array_bytes: int = 0,
    centers_array_bytes: int = 0,
    surface_array_bytes: int = 0,
    surface_point_count: int | None = None,
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "backend": backend,
        "centers_array_bytes": centers_array_bytes,
        "current_observation_count": current_observation_count,
        "dropped_keyframe": dropped_keyframe,
        "format_name": "atlas3r_phase6c_incremental_frame_event",
        "format_version": 1,
        "frame_id": frame.frame_id,
        "map_update_latency_ns": map_update_latency_ns,
        "mode": "incremental",
        "observation_array_bytes": observation_array_bytes,
        "observation_load_latency_ns": observation_load_latency_ns,
        "observed_voxel_count": observed_voxel_count,
        "selected_keyframe": selected_keyframe,
        "surface_array_bytes": surface_array_bytes,
        "surface_point_count": surface_point_count,
        "timestamp_s": frame.timestamp_s,
        "tsdf_array_bytes": tsdf_array_bytes,
        "update_implementation": update_implementation,
    }


def _observation_array_bytes(observations: tuple[DepthObservation, ...]) -> int:
    return int(sum(_single_observation_array_bytes(observation) for observation in observations))


def _single_observation_array_bytes(observation: DepthObservation) -> int:
    return int(
        observation.depth_m.nbytes
        + observation.depth_sigma_m.nbytes
        + observation.confidence.nbytes
    )


def _volume_array_bytes(volume: TSDFVolume) -> int:
    return int(volume.tsdf.nbytes + volume.weight.nbytes)


def _surface_array_bytes(surface: TSDFSurface) -> int:
    return int(
        surface.points_world_m.nbytes + surface.confidence.nbytes + surface.uncertainty_m.nbytes
    )


def _observed_voxel_count(volume: TSDFVolume) -> int:
    return int(np.count_nonzero(volume.weight > 0.0))


def _incremental_limitations(backend: str) -> list[str]:
    shared = [
        "CPU TSDF is a diagnostic reference path and is not the final accelerated mapper.",
        "Only measured depth and pose inputs are fused; RGB-only geometry is not inferred here.",
        "Hidden or unobserved geometry is not emitted as measured geometry.",
        "No object-aware fusion is performed.",
        "No realtime, benchmark accuracy, or millimeter-level claim is made.",
    ]
    if backend == CPU_PERSISTENT_BACKEND:
        return [
            "Fixed TSDF bounds are precomputed offline from selected measured observations.",
            "Persistent dense CPU TSDF integrates only each new DepthObservation.",
            *shared,
        ]
    return [
        "Incremental mode with cpu-rebuild preserves the Phase 6B full-rebuild path.",
        "CPU TSDF state is rebuilt from all selected observations at each selected keyframe.",
        *shared,
    ]


def _resolved_backend(config: FuseRecordingConfig) -> str:
    return CPU_PERSISTENT_BACKEND if config.backend is None else config.backend


def _unique_or_mixed(values: list[str]) -> str:
    stable = _stable_strings(values)
    return stable[0] if len(stable) == 1 else "mixed"


def _stable_strings(values: list[str]) -> list[str]:
    stable: list[str] = []
    for value in values:
        if value not in stable:
            stable.append(value)
    return stable


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _max_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))


def _event_int(event: dict[str, object], key: str) -> int:
    value = event.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "CPU_PERSISTENT_BACKEND",
    "CPU_REBUILD_BACKEND",
    "CPU_REBUILD_UPDATE_IMPLEMENTATION",
]
