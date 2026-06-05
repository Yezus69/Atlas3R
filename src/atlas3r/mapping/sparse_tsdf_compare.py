"""Diagnostic comparison between sparse block TSDF and dense persistent TSDF."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import grid_bounds_from_observations
from atlas3r.mapping.cpu_tsdf import FLOAT64, TSDFSurface, extract_tsdf_surface
from atlas3r.mapping.incremental_tsdf import IncrementalTSDFConfig, PersistentIncrementalTSDFMapper
from atlas3r.mapping.observations import DepthObservation

DIAGNOSTIC_TRUTH_FLAGS: dict[str, bool] = {
    "diagnostic_only": True,
    "accuracy_report": False,
    "performance_report": False,
    "realtime_claim": False,
    "mapping_ready": False,
}


@dataclass(frozen=True)
class SparseDenseComparisonResult:
    comparison: dict[str, object]
    dense_surface: TSDFSurface
    dense_state_bytes: int


def compare_sparse_to_dense_persistent(
    *,
    observations: tuple[DepthObservation, ...],
    sparse_surface: TSDFSurface,
    sparse_state_bytes: int,
    voxel_size_m: float,
    truncation_distance_m: float,
    sparse_update_latencies_ns: tuple[int, ...],
    coordinate_frame: str,
    metric_scale_source: str,
) -> SparseDenseComparisonResult:
    """Run dense persistent baseline after sparse replay and compare point sets."""

    grid_min, grid_max = grid_bounds_from_observations(
        observations,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    mapper = PersistentIncrementalTSDFMapper(
        IncrementalTSDFConfig(
            grid_min_world_m=grid_min,
            grid_max_world_m=grid_max,
            voxel_size_m=voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            coordinate_frame=coordinate_frame,
            metric_scale_source=metric_scale_source,
        )
    )
    dense_update_latencies: list[int] = []
    for observation in observations:
        start_ns = time.perf_counter_ns()
        mapper.integrate(observation)
        dense_update_latencies.append(time.perf_counter_ns() - start_ns)
    dense_surface = extract_tsdf_surface(mapper.volume())
    dense_state_bytes = int(mapper.tsdf_array_bytes + mapper.centers_array_bytes)
    point_metrics = _point_set_metrics(
        sparse_surface.points_world_m,
        dense_surface.points_world_m,
    )
    comparison = {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "format_name": "atlas3r_phase6d_sparse_vs_dense_persistent_comparison",
        "format_version": 1,
        "metric_family": "phase6d_sparse_surface_vs_dense_persistent_surface_diagnostic",
        "backend": "cpu-sparse",
        "dense_backend": "cpu-persistent",
        "source_frame_ids": [observation.frame_id for observation in observations],
        "voxel_size_m": voxel_size_m,
        "truncation_distance_m": truncation_distance_m,
        "dense_surface_point_count": int(dense_surface.points_world_m.shape[0]),
        "sparse_surface_point_count": int(sparse_surface.points_world_m.shape[0]),
        "sparse_state_bytes": sparse_state_bytes,
        "dense_persistent_state_bytes": dense_state_bytes,
        "sparse_to_dense_state_memory_ratio": float(sparse_state_bytes / max(dense_state_bytes, 1)),
        "latency_units": "milliseconds",
        "sparse_update_latency": _latency_stats(sparse_update_latencies_ns),
        "dense_persistent_update_latency": _latency_stats(tuple(dense_update_latencies)),
        "point_set": point_metrics,
        "known_limitations": [
            "Sparse and dense backends use different representations and are not expected to "
            "match TSDF values exactly.",
            "Nearest-neighbor surface distances are sampled diagnostic metrics, not benchmark "
            "accuracy metrics.",
            "Dense persistent comparison is run after sparse replay and is not part of the "
            "online sparse update loop.",
            "No realtime, performance, mapping-readiness, or millimeter-accuracy claim is made.",
        ],
    }
    return SparseDenseComparisonResult(
        comparison=comparison,
        dense_surface=dense_surface,
        dense_state_bytes=dense_state_bytes,
    )


def _point_set_metrics(
    sparse_points: npt.NDArray[np.float32],
    dense_points: npt.NDArray[np.float32],
    *,
    max_points: int = 2048,
) -> dict[str, object]:
    sparse_sample = _sample_points(sparse_points, max_points=max_points)
    dense_sample = _sample_points(dense_points, max_points=max_points)
    sparse_to_dense = _nearest_distances(sparse_sample, dense_sample)
    dense_to_sparse = _nearest_distances(dense_sample, sparse_sample)
    thresholds = (0.01, 0.05, 0.10)
    metrics: dict[str, object] = {
        "sampled_sparse_point_count": int(sparse_sample.shape[0]),
        "sampled_dense_point_count": int(dense_sample.shape[0]),
        "sparse_to_dense_mean_m": float(np.mean(sparse_to_dense)),
        "sparse_to_dense_p95_m": float(np.percentile(sparse_to_dense, 95.0)),
        "dense_to_sparse_mean_m": float(np.mean(dense_to_sparse)),
        "dense_to_sparse_p95_m": float(np.percentile(dense_to_sparse, 95.0)),
        "symmetric_chamfer_like_mean_m": float(
            (np.mean(sparse_to_dense) + np.mean(dense_to_sparse)) / 2.0
        ),
    }
    for threshold in thresholds:
        suffix = _threshold_suffix(threshold)
        metrics[f"precision_like_within_{suffix}_percent"] = _within_percent(
            sparse_to_dense,
            threshold,
        )
        metrics[f"recall_like_within_{suffix}_percent"] = _within_percent(
            dense_to_sparse,
            threshold,
        )
    return metrics


def _sample_points(
    points: npt.NDArray[np.float32],
    *,
    max_points: int,
) -> npt.NDArray[np.float32]:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.floor(
        (np.arange(max_points, dtype=FLOAT64) + 0.5) * points.shape[0] / max_points
    ).astype(np.int64)
    return cast(npt.NDArray[np.float32], points[indices].astype(np.float32, copy=False))


def _nearest_distances(
    query: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    distances: list[npt.NDArray[np.float64]] = []
    target64 = target.astype(FLOAT64, copy=False)
    for start in range(0, query.shape[0], 256):
        chunk = query[start : start + 256].astype(FLOAT64, copy=False)
        delta = chunk[:, None, :] - target64[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return np.concatenate(distances, axis=0)


def _latency_stats(samples_ns: tuple[int, ...]) -> dict[str, object]:
    if not samples_ns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    samples_ms = np.asarray(samples_ns, dtype=FLOAT64) / 1_000_000.0
    return {
        "count": int(samples_ms.size),
        "mean_ms": float(np.mean(samples_ms)),
        "p50_ms": float(np.percentile(samples_ms, 50.0)),
        "p95_ms": float(np.percentile(samples_ms, 95.0)),
        "max_ms": float(np.max(samples_ms)),
    }


def _within_percent(values: npt.NDArray[np.float64], threshold_m: float) -> float:
    return float(np.count_nonzero(values <= threshold_m) / max(values.size, 1) * 100.0)


def _threshold_suffix(threshold_m: float) -> str:
    return {0.01: "1cm", 0.05: "5cm", 0.10: "10cm"}[threshold_m]


__all__ = [
    "SparseDenseComparisonResult",
    "compare_sparse_to_dense_persistent",
]
