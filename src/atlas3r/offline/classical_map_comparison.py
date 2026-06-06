"""Compare aligned classical sparse geometry with Atlas3R fused maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.best_map_selection import BestMapSelectionResult
from atlas3r.offline.fused_world_map import FusedWorldMapResult
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.run_manifest import write_json
from atlas3r.offline.trajectory_alignment import TrajectoryAlignmentResult


@dataclass(frozen=True)
class ClassicalMapComparisonResult:
    status: str
    reason: str
    json_path: str = "diagnostics/classical_map_comparison.json"
    markdown_path: str = "diagnostics/classical_map_comparison.md"
    aligned_sparse_points_path: str | None = None
    trajectory_agreement_status: str = "unavailable"
    map_agreement_status: str = "unavailable"
    maps: dict[str, dict[str, object]] = field(default_factory=dict)
    dense: dict[str, object] = field(default_factory=dict)

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return (self.json_path, self.markdown_path)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "aligned_sparse_points_path": self.aligned_sparse_points_path,
            "trajectory_agreement_status": self.trajectory_agreement_status,
            "map_agreement_status": self.map_agreement_status,
            "maps": self.maps,
            "dense": self.dense,
            "truth_boundary": {
                "label_type": "classical_sfm_proposal",
                "measured_geometry": False,
                "observed_only": True,
                "predicted_completion": False,
                "hidden_geometry_measured": False,
                "physical_accuracy_claim": False,
                "training_quality": False,
            },
        }


def write_classical_map_comparison(
    run_dir: str | Path,
    *,
    alignment: TrajectoryAlignmentResult,
    raw_world_map: FusedWorldMapResult,
    optimizer: MapConsistencyOptimizerResult,
    best_map: BestMapSelectionResult,
    voxel_size_m: float,
) -> ClassicalMapComparisonResult:
    root = Path(run_dir)
    if alignment.status != "available" or alignment.aligned_points_npz_path is None:
        result = ClassicalMapComparisonResult(
            status="unavailable",
            reason=f"classical alignment unavailable: {alignment.reason}",
            trajectory_agreement_status="unavailable",
            map_agreement_status="unavailable",
        )
        _write(root, result)
        return result
    classical_points = _read_points(root / alignment.aligned_points_npz_path)
    if classical_points.shape[0] == 0:
        result = ClassicalMapComparisonResult(
            status="unavailable",
            reason="aligned classical sparse point cloud is empty",
            aligned_sparse_points_path=alignment.aligned_points_npz_path,
        )
        _write(root, result)
        return result
    maps: dict[str, dict[str, object]] = {}
    candidates: list[tuple[str, str | None]] = [
        ("raw_world_map", raw_world_map.fused_points_npz_path),
        (
            "optimized_world_map",
            optimizer.optimized_world_map.fused_points_npz_path,
        ),
        ("world_map_best", best_map.fused_points_npz_path),
    ]
    for name, relative_path in candidates:
        maps[name] = _compare_one(root, relative_path, classical_points, voxel_size_m)
    best_metrics = maps.get("world_map_best", {})
    trajectory_status = _trajectory_status(alignment)
    map_status = _map_status(best_metrics)
    result = ClassicalMapComparisonResult(
        status="available",
        reason="computed nearest-neighbor and bbox agreement against aligned sparse SfM points",
        aligned_sparse_points_path=alignment.aligned_points_npz_path,
        trajectory_agreement_status=trajectory_status,
        map_agreement_status=map_status,
        maps=maps,
        dense=_dense_summary(root),
    )
    _write(root, result)
    return result


def compare_point_sets(
    map_points: NDArray[np.float32],
    classical_points: NDArray[np.float32],
    *,
    near_threshold_m: float,
    sample_limit: int = 20_000,
) -> dict[str, object]:
    map_sample = _sample(map_points, sample_limit)
    classical_sample = _sample(classical_points, sample_limit)
    if map_sample.shape[0] == 0 or classical_sample.shape[0] == 0:
        return {"status": "unavailable", "reason": "empty map or classical point set"}
    classical_to_map = nearest_neighbor_distances(classical_sample, map_sample)
    map_to_classical = nearest_neighbor_distances(map_sample, classical_sample)
    map_bbox = _bbox(map_sample)
    classical_bbox = _bbox(classical_sample)
    return {
        "status": "available",
        "map_point_count_compared": int(map_sample.shape[0]),
        "classical_point_count_compared": int(classical_sample.shape[0]),
        "nearest_neighbor_distance_mean_m": _mean(classical_to_map),
        "nearest_neighbor_distance_p50_m": _percentile(classical_to_map, 50.0),
        "nearest_neighbor_distance_p95_m": _percentile(classical_to_map, 95.0),
        "map_points_near_classical_ratio": _near_ratio(map_to_classical, near_threshold_m),
        "classical_points_near_map_ratio": _near_ratio(classical_to_map, near_threshold_m),
        "bbox_overlap_ratio": _bbox_overlap_ratio(map_bbox, classical_bbox),
        "near_threshold_m": near_threshold_m,
    }


def nearest_neighbor_distances(
    query_points: NDArray[np.float32],
    target_points: NDArray[np.float32],
    *,
    chunk_size: int = 1024,
) -> NDArray[np.float32]:
    query = np.asarray(query_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("point arrays must be shaped N,3")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(target)):
        raise ValueError("point arrays must be finite")
    if query.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    distances = np.empty((query.shape[0],), dtype=np.float32)
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        squared = np.sum(diff * diff, axis=2)
        distances[start : start + chunk.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return distances.astype(np.float32)


def _compare_one(
    root: Path,
    relative_path: str | None,
    classical_points: NDArray[np.float32],
    voxel_size_m: float,
) -> dict[str, object]:
    if relative_path is None:
        return {"status": "unavailable", "reason": "map point artifact is missing"}
    path = root / relative_path
    if not path.is_file():
        return {"status": "unavailable", "reason": f"map point artifact missing: {relative_path}"}
    map_points = _read_points(path)
    threshold = max(0.10, 2.0 * voxel_size_m)
    return compare_point_sets(map_points, classical_points, near_threshold_m=threshold)


def _read_points(path: Path) -> NDArray[np.float32]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["points_world_m"], dtype=np.float32)


def _sample(points: NDArray[np.float32], limit: int) -> NDArray[np.float32]:
    finite = np.asarray(points, dtype=np.float32)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if finite.shape[0] <= limit:
        return cast(NDArray[np.float32], finite)
    indices = np.linspace(0, finite.shape[0] - 1, num=limit, dtype=np.int64)
    return cast(NDArray[np.float32], finite[indices])


def _trajectory_status(alignment: TrajectoryAlignmentResult) -> str:
    if alignment.common_frame_count < 8:
        return "weak_common_frame_support"
    rmse = alignment.camera_center_rmse_m
    p95 = alignment.camera_center_p95_m
    if rmse is not None and p95 is not None and rmse <= 0.15 and p95 <= 0.35:
        return "agrees"
    if rmse is not None and p95 is not None and rmse <= 0.35 and p95 <= 0.75:
        return "weak_agreement"
    return "disagrees"


def _map_status(metrics: dict[str, object]) -> str:
    if metrics.get("status") != "available":
        return "unavailable"
    p95 = _float_or_none(metrics.get("nearest_neighbor_distance_p95_m"))
    classical_near = _float_or_none(metrics.get("classical_points_near_map_ratio"))
    bbox_overlap = _float_or_none(metrics.get("bbox_overlap_ratio"))
    if p95 is not None and classical_near is not None and p95 <= 0.25 and classical_near >= 0.5:
        return "agrees"
    if (
        p95 is not None
        and classical_near is not None
        and bbox_overlap is not None
        and p95 <= 0.75
        and classical_near >= 0.2
        and bbox_overlap > 0.0
    ):
        return "weak_agreement"
    return "disagrees"


def _dense_summary(root: Path) -> dict[str, object]:
    dense = root / "classical" / "colmap_dense_fused.ply"
    mesh = root / "classical" / "colmap_dense_mesh.ply"
    return {
        "colmap_dense_fused": _ply_counts(dense) if dense.is_file() else None,
        "colmap_dense_mesh": _ply_counts(mesh) if mesh.is_file() else None,
    }


def _ply_counts(path: Path) -> dict[str, object]:
    vertices = None
    faces = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("element vertex "):
            vertices = int(line.split()[2])
        if line.startswith("element face "):
            faces = int(line.split()[2])
        if line == "end_header":
            break
    return {"path": f"classical/{path.name}", "vertex_count": vertices, "face_count": faces}


def _bbox(points: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return points.min(axis=0).astype(np.float32), points.max(axis=0).astype(np.float32)


def _bbox_overlap_ratio(
    first: tuple[NDArray[np.float32], NDArray[np.float32]],
    second: tuple[NDArray[np.float32], NDArray[np.float32]],
) -> float:
    min_a, max_a = first
    min_b, max_b = second
    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_volume = float(np.prod(overlap_size))
    volume_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    volume_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union = volume_a + volume_b - overlap_volume
    return float(overlap_volume / union) if union > 0.0 else 0.0


def _write(root: Path, result: ClassicalMapComparisonResult) -> None:
    write_json(root / result.json_path, result.to_dict())
    (root / result.markdown_path).write_text(_markdown(result), encoding="utf-8")


def _markdown(result: ClassicalMapComparisonResult) -> str:
    best = result.maps.get("world_map_best", {})
    return "\n".join(
        [
            "# Classical Map Comparison",
            "",
            f"- Status: {result.status}",
            f"- Reason: {result.reason}",
            f"- Trajectory agreement: {result.trajectory_agreement_status}",
            f"- Map agreement: {result.map_agreement_status}",
            f"- Best-map NN mean m: {best.get('nearest_neighbor_distance_mean_m')}",
            f"- Best-map NN p50 m: {best.get('nearest_neighbor_distance_p50_m')}",
            f"- Best-map NN p95 m: {best.get('nearest_neighbor_distance_p95_m')}",
            f"- Best-map points near classical: {best.get('map_points_near_classical_ratio')}",
            f"- Classical points near best map: {best.get('classical_points_near_map_ratio')}",
            f"- Best-map bbox overlap: {best.get('bbox_overlap_ratio')}",
            "- Physical accuracy claim: false",
            "- Training-quality claim: false",
            "",
            "Classical sparse geometry is an unanchored proposal used only as a "
            "consistency witness.",
            "",
        ]
    )


def _mean(values: NDArray[np.float32]) -> float:
    return float(values.mean()) if values.size else 0.0


def _percentile(values: NDArray[np.float32], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _near_ratio(values: NDArray[np.float32], threshold: float) -> float:
    return float(np.mean(values <= threshold)) if values.size else 0.0


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if np.isfinite(result) else None
