"""Small pure-NumPy CPU TSDF reference integrator for synthetic smoke tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.data.synthetic_cube_room import (
    AxisAlignedBox,
    SyntheticCubeRoomFrame,
    SyntheticCubeRoomScene,
    create_synthetic_cube_room_scene,
    write_synthetic_cube_room_session,
)
from atlas3r.pose.transforms import invert_transform, transform_points

FLOAT32 = np.float32
FLOAT64 = np.float64


@dataclass(frozen=True)
class TSDFVolume:
    """A deterministic dense TSDF grid in the Atlas3R world frame."""

    grid_min_corner_world_m: npt.NDArray[np.float32]
    voxel_size_m: float
    truncation_distance_m: float
    tsdf: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]
    source_frame_ids: tuple[int, ...]
    coordinate_frame: str
    metric_scale_source: str

    def centers_world_m(self) -> npt.NDArray[np.float64]:
        """Return voxel centers as an Nx3 world-space array."""
        nx, ny, nz = self.tsdf.shape
        origin = self.grid_min_corner_world_m.astype(FLOAT64, copy=False)
        x = origin[0] + (np.arange(nx, dtype=FLOAT64) + 0.5) * self.voxel_size_m
        y = origin[1] + (np.arange(ny, dtype=FLOAT64) + 0.5) * self.voxel_size_m
        z = origin[2] + (np.arange(nz, dtype=FLOAT64) + 0.5) * self.voxel_size_m
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


@dataclass(frozen=True)
class TSDFSurface:
    """Observed surface points extracted from a Phase 0D TSDF volume."""

    points_world_m: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    uncertainty_m: npt.NDArray[np.float32]
    voxel_indices_xyz: npt.NDArray[np.int32]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TSDFCubeRoomSmokeResult:
    """In-memory result for the `atlas3r smoke tsdf-cube-room` command."""

    session_path: Path
    volume: TSDFVolume
    surface: TSDFSurface
    metrics: dict[str, Any]


def integrate_synthetic_cube_room_scene(
    scene: SyntheticCubeRoomScene,
    *,
    voxel_size_m: float = 0.1,
    truncation_voxels: float = 3.0,
) -> TSDFVolume:
    """Fuse the Phase 0B analytic depth frames into a tiny CPU TSDF volume."""
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")

    grid_min = scene.room_bounds_m.min_corner_m.astype(FLOAT64, copy=True)
    grid_max = scene.room_bounds_m.max_corner_m.astype(FLOAT64, copy=False)
    shape_xyz = _grid_shape_xyz(grid_min, grid_max, voxel_size_m)
    centers_world = _voxel_centers(grid_min, shape_xyz, voxel_size_m)
    voxel_count = centers_world.shape[0]
    tsdf_flat = np.ones(voxel_count, dtype=FLOAT64)
    weight_flat = np.zeros(voxel_count, dtype=FLOAT64)
    truncation_distance_m = voxel_size_m * truncation_voxels

    for frame in scene.frames:
        _integrate_frame(
            frame=frame,
            centers_world_m=centers_world,
            voxel_size_m=voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            tsdf_flat=tsdf_flat,
            weight_flat=weight_flat,
        )

    return TSDFVolume(
        grid_min_corner_world_m=grid_min.astype(FLOAT32),
        voxel_size_m=float(voxel_size_m),
        truncation_distance_m=float(truncation_distance_m),
        tsdf=tsdf_flat.reshape(shape_xyz).astype(FLOAT32),
        weight=weight_flat.reshape(shape_xyz).astype(FLOAT32),
        source_frame_ids=tuple(frame.frame_id for frame in scene.frames),
        coordinate_frame="synthetic_world",
        metric_scale_source=scene.world_map.scale_source,
    )


def extract_tsdf_surface(volume: TSDFVolume, *, surface_band: float = 1.0 / 3.0) -> TSDFSurface:
    """Extract voxel-center surface points from near-zero observed TSDF values."""
    if surface_band <= 0.0:
        raise ValueError("surface_band: must be positive")
    observed_mask = volume.weight > 0.0
    surface_mask = observed_mask & (np.abs(volume.tsdf) <= surface_band)
    voxel_indices = np.argwhere(surface_mask).astype(np.int32)
    if voxel_indices.size == 0:
        raise ValueError("TSDF surface extraction produced no observed surface voxels")

    points = (
        volume.grid_min_corner_world_m.astype(FLOAT64, copy=False)
        + (voxel_indices.astype(FLOAT64) + 0.5) * volume.voxel_size_m
    )
    surface_weight = volume.weight[surface_mask].astype(FLOAT64, copy=False)
    tsdf_values = volume.tsdf[surface_mask].astype(FLOAT64, copy=False)
    confidence = np.clip(surface_weight / max(float(len(volume.source_frame_ids)), 1.0), 0.0, 1.0)
    uncertainty = (
        volume.voxel_size_m / np.sqrt(np.maximum(surface_weight, 1.0))
        + np.abs(tsdf_values) * volume.truncation_distance_m
    )
    metadata = _surface_metadata(
        volume=volume,
        surface_count=int(voxel_indices.shape[0]),
        observed_count=int(np.count_nonzero(observed_mask)),
        uncertainty_m=uncertainty,
        surface_band=surface_band,
    )
    return TSDFSurface(
        points_world_m=points.astype(FLOAT32),
        confidence=confidence.astype(FLOAT32),
        uncertainty_m=uncertainty.astype(FLOAT32),
        voxel_indices_xyz=voxel_indices,
        metadata=metadata,
    )


def evaluate_surface_against_synthetic_cube_room(
    scene: SyntheticCubeRoomScene, surface: TSDFSurface
) -> dict[str, Any]:
    """Compute conservative voxel-scale metrics against the synthetic box mesh."""
    points = surface.points_world_m.astype(FLOAT64, copy=False)
    if points.size == 0:
        raise ValueError("surface.points_world_m: expected at least one point")
    voxel_size_m = float(surface.metadata["voxel_size_m"])
    room_distance = _distance_to_box_surface(points, scene.room_bounds_m)
    object_distance = _distance_to_box_surface(points, scene.object_bounds_m)
    gt_surface_distance = np.minimum(room_distance, object_distance)
    inside_room = _points_inside_box(points, scene.room_bounds_m, tolerance_m=voxel_size_m)
    voxel_tolerance_m = voxel_size_m + 1e-6
    object_surface = object_distance <= voxel_tolerance_m
    room_surface = room_distance <= voxel_tolerance_m
    surface_min = points.min(axis=0)
    surface_max = points.max(axis=0)

    return {
        "metric_family": "phase_0d_synthetic_axis_aligned_box_reference",
        "ground_truth_mesh_chunk_id": scene.mesh_chunk.chunk_id,
        "voxel_size_m": voxel_size_m,
        "truncation_distance_m": float(surface.metadata["truncation_distance_m"]),
        "voxel_scale_tolerance_m": voxel_size_m,
        "surface_point_count": int(points.shape[0]),
        "room_surface_point_count": int(np.count_nonzero(room_surface)),
        "object_surface_point_count": int(np.count_nonzero(object_surface)),
        "surface_points_inside_room_bounds_ratio": _ratio(inside_room),
        "surface_points_within_one_voxel_of_gt_surface_ratio": _ratio(
            gt_surface_distance <= voxel_tolerance_m
        ),
        "surface_to_gt_box_surface_mean_distance_m": float(np.mean(gt_surface_distance)),
        "surface_to_gt_box_surface_p95_distance_m": float(np.percentile(gt_surface_distance, 95.0)),
        "surface_to_gt_box_surface_max_distance_m": float(np.max(gt_surface_distance)),
        "surface_bounds_min_m": _json_array(surface_min),
        "surface_bounds_max_m": _json_array(surface_max),
        "known_limitations": [
            "Reference uses nearest-pixel TSDF integration on the analytic Phase 0B fixture.",
            "Extracted points are voxel centers, so bounds and distances are voxel-scale checks.",
            "Only observed surfaces are represented; hidden or completed geometry is not emitted.",
            "This smoke metric is not an accuracy report and makes no millimeter-level claim.",
        ],
    }


def run_tsdf_cube_room_smoke(output_folder: str | Path) -> TSDFCubeRoomSmokeResult:
    """Generate the synthetic session, run TSDF fusion, and write smoke outputs."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    session_path = write_synthetic_cube_room_session(output_path / "synthetic_cube_room.atlas3r")
    scene = create_synthetic_cube_room_scene()
    volume = integrate_synthetic_cube_room_scene(scene)
    surface = extract_tsdf_surface(volume)
    metrics = evaluate_surface_against_synthetic_cube_room(scene, surface)

    _write_npz(
        output_path / "tsdf_grid.npz",
        grid_min_corner_world_m=volume.grid_min_corner_world_m,
        voxel_size_m=np.array(volume.voxel_size_m, dtype=FLOAT32),
        truncation_distance_m=np.array(volume.truncation_distance_m, dtype=FLOAT32),
        tsdf=volume.tsdf,
        weight=volume.weight,
    )
    _write_npz(
        output_path / "surface_points.npz",
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
    )
    _write_json(output_path / "metadata.json", surface.metadata)
    _write_json(output_path / "metrics.json", metrics)
    return TSDFCubeRoomSmokeResult(
        session_path=session_path,
        volume=volume,
        surface=surface,
        metrics=metrics,
    )


def write_tsdf_cube_room_smoke(
    output_folder: str | Path,
    *,
    write_mesh_sidecar: bool = False,
    write_world_map_sidecar: bool = False,
) -> tuple[Path, ...]:
    """Write TSDF smoke artifacts and return the deterministic top-level paths."""
    output_path = Path(output_folder)
    result = run_tsdf_cube_room_smoke(output_path)
    written_paths: tuple[Path, ...] = (
        result.session_path,
        output_path / "tsdf_grid.npz",
        output_path / "surface_points.npz",
        output_path / "metadata.json",
        output_path / "metrics.json",
    )
    if write_mesh_sidecar or write_world_map_sidecar:
        from atlas3r.mapping.mesh_sidecar import write_tsdf_surface_mesh_sidecar_from_artifacts

        sidecar_path = write_tsdf_surface_mesh_sidecar_from_artifacts(
            output_path,
            chunk_id="phase_0d_cpu_tsdf_surface_reference",
        )
        written_paths = (*written_paths, sidecar_path)
    if write_world_map_sidecar:
        from atlas3r.mapping.world_map_sidecar import write_tsdf_world_map_sidecar_from_artifacts

        map_sidecar_path = write_tsdf_world_map_sidecar_from_artifacts(output_path)
        written_paths = (*written_paths, map_sidecar_path)
    return written_paths


def _grid_shape_xyz(
    grid_min_world_m: npt.NDArray[np.float64],
    grid_max_world_m: npt.NDArray[np.float64],
    voxel_size_m: float,
) -> tuple[int, int, int]:
    extent = grid_max_world_m - grid_min_world_m
    if np.any(extent <= 0.0):
        raise ValueError("grid bounds: max corner must be greater than min corner")
    shape = np.ceil((extent / voxel_size_m) - 1e-9).astype(np.int64)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _voxel_centers(
    grid_min_world_m: npt.NDArray[np.float64],
    shape_xyz: tuple[int, int, int],
    voxel_size_m: float,
) -> npt.NDArray[np.float64]:
    nx, ny, nz = shape_xyz
    x = grid_min_world_m[0] + (np.arange(nx, dtype=FLOAT64) + 0.5) * voxel_size_m
    y = grid_min_world_m[1] + (np.arange(ny, dtype=FLOAT64) + 0.5) * voxel_size_m
    z = grid_min_world_m[2] + (np.arange(nz, dtype=FLOAT64) + 0.5) * voxel_size_m
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _integrate_frame(
    *,
    frame: SyntheticCubeRoomFrame,
    centers_world_m: npt.NDArray[np.float64],
    voxel_size_m: float,
    truncation_distance_m: float,
    tsdf_flat: npt.NDArray[np.float64],
    weight_flat: npt.NDArray[np.float64],
) -> None:
    T_camera_world = invert_transform(frame.pose.T_world_camera)
    centers_camera_m = transform_points(T_camera_world, centers_world_m)
    z_camera_m = centers_camera_m[:, 2]
    valid_z = z_camera_m > 0.0
    if not np.any(valid_z):
        return

    valid_indices = np.flatnonzero(valid_z)
    valid_points_camera = centers_camera_m[valid_indices]
    projected_u = (
        frame.camera.K[0, 0] * valid_points_camera[:, 0] / valid_points_camera[:, 2]
        + frame.camera.K[0, 2]
    )
    projected_v = (
        frame.camera.K[1, 1] * valid_points_camera[:, 1] / valid_points_camera[:, 2]
        + frame.camera.K[1, 2]
    )
    pixel_u = np.rint(projected_u).astype(np.int64)
    pixel_v = np.rint(projected_v).astype(np.int64)
    inside_image = (
        (pixel_u >= 0)
        & (pixel_u < frame.camera.width)
        & (pixel_v >= 0)
        & (pixel_v < frame.camera.height)
    )
    if not np.any(inside_image):
        return

    voxel_indices = valid_indices[inside_image]
    pixel_u = pixel_u[inside_image]
    pixel_v = pixel_v[inside_image]
    z_camera_m = valid_points_camera[inside_image, 2]
    measured_depth_m = frame.depth_m[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    signed_distance_m = measured_depth_m - z_camera_m
    in_truncation_band = np.abs(signed_distance_m) <= truncation_distance_m
    if not np.any(in_truncation_band):
        return

    voxel_indices = voxel_indices[in_truncation_band]
    pixel_u = pixel_u[in_truncation_band]
    pixel_v = pixel_v[in_truncation_band]
    signed_distance_m = signed_distance_m[in_truncation_band]
    normalized_tsdf = np.clip(signed_distance_m / truncation_distance_m, -1.0, 1.0)
    depth_sigma_m = frame.depth_sigma_m[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    frame_confidence = frame.confidence[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    measurement_weight = frame_confidence / (1.0 + depth_sigma_m / voxel_size_m)
    positive_weight = measurement_weight > 0.0
    if not np.any(positive_weight):
        return

    voxel_indices = voxel_indices[positive_weight]
    normalized_tsdf = normalized_tsdf[positive_weight]
    measurement_weight = measurement_weight[positive_weight]
    old_weight = weight_flat[voxel_indices]
    new_weight = old_weight + measurement_weight
    tsdf_flat[voxel_indices] = (
        tsdf_flat[voxel_indices] * old_weight + normalized_tsdf * measurement_weight
    ) / new_weight
    weight_flat[voxel_indices] = new_weight


def _surface_metadata(
    *,
    volume: TSDFVolume,
    surface_count: int,
    observed_count: int,
    uncertainty_m: npt.NDArray[np.float64],
    surface_band: float,
) -> dict[str, Any]:
    total_voxel_count = int(volume.tsdf.size)
    return {
        "artifact_type": "phase_0d_cpu_tsdf_surface_points",
        "coordinate_frame": volume.coordinate_frame,
        "metric_scale_source": volume.metric_scale_source,
        "source_frame_ids": list(volume.source_frame_ids),
        "voxel_size_m": volume.voxel_size_m,
        "truncation_distance_m": volume.truncation_distance_m,
        "surface_band_abs_normalized_tsdf": surface_band,
        "surface_voxel_count": surface_count,
        "observed_tsdf_voxel_count": observed_count,
        "total_voxel_count": total_voxel_count,
        "observed_coverage_estimate": float(observed_count / max(total_voxel_count, 1)),
        "surface_coverage_estimate": float(surface_count / max(total_voxel_count, 1)),
        "uncertainty_summary_m": {
            "mean": float(np.mean(uncertainty_m)),
            "p50": float(np.percentile(uncertainty_m, 50.0)),
            "p95": float(np.percentile(uncertainty_m, 95.0)),
            "max": float(np.max(uncertainty_m)),
        },
        "flags": [
            "phase_0d_reference_only",
            "observed_surface_points",
            "voxel_center_surface_approximation",
        ],
    }


def _distance_to_box_surface(
    points_world_m: npt.NDArray[np.float64], bounds: AxisAlignedBox
) -> npt.NDArray[np.float64]:
    min_corner = bounds.min_corner_m.astype(FLOAT64, copy=False)
    max_corner = bounds.max_corner_m.astype(FLOAT64, copy=False)
    clamped = np.minimum(np.maximum(points_world_m, min_corner), max_corner)
    outside_distance = np.linalg.norm(points_world_m - clamped, axis=1)
    inside = np.all((points_world_m >= min_corner) & (points_world_m <= max_corner), axis=1)
    distance_to_min_face = points_world_m - min_corner
    distance_to_max_face = max_corner - points_world_m
    inside_distance = np.min(np.minimum(distance_to_min_face, distance_to_max_face), axis=1)
    return np.where(inside, inside_distance, outside_distance)


def _points_inside_box(
    points_world_m: npt.NDArray[np.float64], bounds: AxisAlignedBox, *, tolerance_m: float
) -> npt.NDArray[np.bool_]:
    min_corner = bounds.min_corner_m.astype(FLOAT64, copy=False) - tolerance_m
    max_corner = bounds.max_corner_m.astype(FLOAT64, copy=False) + tolerance_m
    return cast(
        npt.NDArray[np.bool_],
        np.all((points_world_m >= min_corner) & (points_world_m <= max_corner), axis=1),
    )


def _ratio(mask: npt.NDArray[np.bool_]) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def _json_array(array: npt.NDArray[Any]) -> list[Any]:
    return cast(list[Any], array.tolist())


def _write_npz(path: Path, **arrays: npt.NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "TSDFCubeRoomSmokeResult",
    "TSDFSurface",
    "TSDFVolume",
    "evaluate_surface_against_synthetic_cube_room",
    "extract_tsdf_surface",
    "integrate_synthetic_cube_room_scene",
    "run_tsdf_cube_room_smoke",
    "write_tsdf_cube_room_smoke",
]
