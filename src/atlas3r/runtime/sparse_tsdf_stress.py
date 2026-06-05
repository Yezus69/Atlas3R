"""Apartment-scale sparse TSDF memory stress diagnostic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import (
    SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    SparseBlockTSDFMapper,
    SparseTSDFConfig,
)
from atlas3r.mapping.sparse_tsdf_artifacts import (
    sparse_state_array_bytes,
    write_sparse_tsdf_outputs,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


@dataclass(frozen=True)
class SparseTSDFStressConfig:
    output: Path
    room_size_m: tuple[float, float, float] = (10.0, 10.0, 3.0)
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0
    observation_count: int = 3

    def __post_init__(self) -> None:
        if len(self.room_size_m) != 3 or any(value <= 0.0 for value in self.room_size_m):
            raise ValueError("room_size_m: must contain three positive values")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_voxels <= 0.0:
            raise ValueError("truncation_voxels: must be positive")
        if self.observation_count <= 0:
            raise ValueError("observation_count: must be positive")


def run_sparse_tsdf_stress(config: SparseTSDFStressConfig) -> dict[str, object]:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    truncation_distance_m = config.voxel_size_m * config.truncation_voxels
    mapper = SparseBlockTSDFMapper(
        SparseTSDFConfig(
            voxel_size_m=config.voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            coordinate_frame="x_right_y_down_z_forward",
            metric_scale_source="synthetic_measured_depth_pose",
            pixel_stride=4,
        )
    )
    recorder = LatencyRecorder()
    events: list[dict[str, object]] = []
    for observation in _stress_observations(config):
        start_ns = time.perf_counter_ns()
        stats = mapper.integrate(observation)
        update_latency_ns = time.perf_counter_ns() - start_ns
        recorder.add("sparse_map_update", update_latency_ns)
        events.append(
            {
                **DIAGNOSTIC_TRUTH_FLAGS,
                "active_block_count": stats.active_block_count,
                "active_voxel_count": stats.active_voxel_count,
                "approximate_state_bytes": stats.approximate_state_bytes,
                "frame_id": observation.frame_id,
                "map_update_latency_ns": update_latency_ns,
                "new_voxel_count": stats.new_voxel_count,
                "surface_point_count": None,
                "update_implementation": stats.update_implementation,
                "updated_voxel_count": stats.updated_voxel_count,
            }
        )
    surface = mapper.extract_surface()
    sparse_dir = write_sparse_tsdf_outputs(
        output / "sparse_tsdf",
        mapper=mapper,
        surface=surface,
        metrics=_stress_metrics(config, mapper, surface),
    )
    point_cloud_path = write_point_cloud_ply(output / "surface_points.ply", surface)
    dense_estimate = _dense_memory_estimate(config)
    sparse_output_bytes = sparse_state_array_bytes(mapper, surface)
    latency_report = recorder.report()
    latency_report.update(
        {
            "format_name": "atlas3r_phase6d_sparse_tsdf_stress_latency_report",
            "known_limitations": _stress_limitations(),
            "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        }
    )
    summary = {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "format_name": "atlas3r_phase6d_sparse_tsdf_stress_summary",
        "format_version": 1,
        "artifacts": {
            "latency_report": "latency_report.json",
            "per_frame_events": "per_frame_events.jsonl",
            "point_cloud": point_cloud_path.name,
            "sparse_tsdf": sparse_dir.name,
            "summary": "summary.json",
        },
        "dense_memory_estimate": dense_estimate,
        "known_limitations": _stress_limitations(),
        "observation_count": config.observation_count,
        "room_size_m": list(config.room_size_m),
        "sparse_memory_estimate": {
            "active_block_count": mapper.active_block_count,
            "active_voxel_count": mapper.active_voxel_count,
            "allocated_voxel_count": mapper.allocated_voxel_count,
            "approximate_state_bytes": mapper.approximate_state_bytes,
            "sparse_output_array_bytes": sparse_output_bytes,
        },
        "sparse_to_dense_state_memory_ratio": float(
            mapper.approximate_state_bytes
            / max(cast(int, dense_estimate["dense_persistent_estimated_bytes"]), 1)
        ),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "truncation_distance_m": truncation_distance_m,
        "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        "voxel_size_m": config.voxel_size_m,
    }
    write_json(output / "latency_report.json", latency_report)
    write_jsonl(output / "per_frame_events.jsonl", events)
    write_json(output / "summary.json", summary)
    return summary


def parse_room_size_m(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("room-size-m: expected L,W,H")
    try:
        room_size = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("room-size-m: expected numeric L,W,H") from exc
    if any(part <= 0.0 for part in room_size):
        raise ValueError("room-size-m: values must be positive")
    return (room_size[0], room_size[1], room_size[2])


def _stress_observations(config: SparseTSDFStressConfig) -> tuple[DepthObservation, ...]:
    width, height = 64, 48
    K = np.asarray([[48.0, 0.0, 31.5], [0.0, 48.0, 23.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera = CameraModel(
        width=width,
        height=height,
        K=K,
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source="synthetic_stress_camera",
    )
    observations = []
    depth_m = np.full((height, width), min(2.0, config.room_size_m[2] * 0.75), dtype=np.float32)
    sigma = np.full_like(depth_m, config.voxel_size_m * 0.25)
    confidence = np.ones_like(depth_m, dtype=np.float32)
    for frame_id in range(config.observation_count):
        T_world_camera = np.eye(4, dtype=np.float32)
        fraction = (frame_id + 1.0) / (config.observation_count + 1.0)
        T_world_camera[0, 3] = np.float32(config.room_size_m[0] * fraction)
        T_world_camera[1, 3] = np.float32(config.room_size_m[1] * 0.5)
        T_world_camera[2, 3] = np.float32(0.25)
        pose = PoseEstimate(
            frame_id=frame_id,
            timestamp_ns=frame_id,
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            camera_center_world_m=T_world_camera[:3, 3].copy(),
            covariance_6x6=None,
            confidence=1.0,
            tracking_state="OK",
            scale_source="external_pose",
            diagnostics={"coordinate_frame": "x_right_y_down_z_forward"},
        )
        observations.append(
            DepthObservation(
                frame_id=frame_id,
                camera=camera,
                pose=pose,
                depth_m=depth_m,
                depth_sigma_m=sigma,
                confidence=confidence,
                source="synthetic_sparse_tsdf_stress",
            )
        )
    return tuple(observations)


def _dense_memory_estimate(config: SparseTSDFStressConfig) -> dict[str, object]:
    shape = tuple(int(np.ceil(size_m / config.voxel_size_m)) for size_m in config.room_size_m)
    dense_voxel_count = int(shape[0] * shape[1] * shape[2])
    tsdf_weight_bytes = dense_voxel_count * np.dtype(np.float32).itemsize * 2
    centers_bytes = dense_voxel_count * 3 * np.dtype(np.float64).itemsize
    return {
        "dense_grid_shape_xyz": list(shape),
        "dense_voxel_count": dense_voxel_count,
        "dense_tsdf_weight_array_bytes": int(tsdf_weight_bytes),
        "dense_precomputed_centers_array_bytes": int(centers_bytes),
        "dense_persistent_estimated_bytes": int(tsdf_weight_bytes + centers_bytes),
    }


def _stress_metrics(
    config: SparseTSDFStressConfig,
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "known_limitations": _stress_limitations(),
        "metric_family": "phase6d_apartment_scale_sparse_tsdf_stress_diagnostic",
        "room_size_m": list(config.room_size_m),
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }


def _stress_limitations() -> list[str]:
    return [
        "Stress input is synthetic measured-depth geometry for memory diagnostics only.",
        "Dense memory is an array-size estimate for a room bounding box, not process RSS.",
        "Sparse update timing is a local wall-clock diagnostic, not a performance report.",
        "No realtime, accuracy, mapping-readiness, or millimeter-level claim is made.",
    ]


__all__ = [
    "SparseTSDFStressConfig",
    "parse_room_size_m",
    "run_sparse_tsdf_stress",
]
