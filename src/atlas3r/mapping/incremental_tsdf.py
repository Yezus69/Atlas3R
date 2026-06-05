"""Persistent incremental TSDF mapper backed by the validated CPU integrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import (
    FLOAT32,
    FLOAT64,
    TSDFSurface,
    TSDFVolume,
    extract_tsdf_surface,
    integrate_depth_observation,
)
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.tsdf_grid import compute_tsdf_grid_shape, voxel_centers_world

PERSISTENT_CPU_UPDATE_IMPLEMENTATION = "persistent_cpu_tsdf_single_observation"


@dataclass(frozen=True)
class IncrementalTSDFConfig:
    """Fixed dense-grid configuration for offline incremental TSDF diagnostics."""

    grid_min_world_m: npt.NDArray[np.float64]
    grid_max_world_m: npt.NDArray[np.float64]
    voxel_size_m: float
    truncation_distance_m: float
    coordinate_frame: str
    metric_scale_source: str

    def __post_init__(self) -> None:
        grid_min = _validate_corner("grid_min_world_m", self.grid_min_world_m)
        grid_max = _validate_corner("grid_max_world_m", self.grid_max_world_m)
        if np.any(grid_max <= grid_min):
            raise ValueError("grid bounds: max corner must be greater than min corner")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_distance_m <= 0.0:
            raise ValueError("truncation_distance_m: must be positive")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame: must be non-empty")
        if not self.metric_scale_source:
            raise ValueError("metric_scale_source: must be non-empty")
        object.__setattr__(self, "grid_min_world_m", grid_min)
        object.__setattr__(self, "grid_max_world_m", grid_max)
        object.__setattr__(self, "voxel_size_m", float(self.voxel_size_m))
        object.__setattr__(self, "truncation_distance_m", float(self.truncation_distance_m))
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame))
        object.__setattr__(self, "metric_scale_source", str(self.metric_scale_source))


@dataclass(frozen=True)
class IncrementalTSDFUpdateStats:
    """Deterministic counters for one persistent TSDF map update."""

    frame_id: int
    integrated_observation_count: int
    unique_source_frame_count: int
    observed_voxel_count: int
    newly_observed_voxel_count: int
    voxel_count: int
    tsdf_array_bytes: int
    centers_array_bytes: int
    update_implementation: str


class PersistentIncrementalTSDFMapper:
    """Dense CPU TSDF state that integrates one new observation per update."""

    def __init__(self, config: IncrementalTSDFConfig) -> None:
        self._config = config
        self._shape_xyz = compute_tsdf_grid_shape(
            config.grid_min_world_m,
            config.grid_max_world_m,
            config.voxel_size_m,
        )
        self._centers_world_m = voxel_centers_world(
            config.grid_min_world_m,
            self._shape_xyz,
            config.voxel_size_m,
        )
        voxel_count = int(self._centers_world_m.shape[0])
        self._tsdf_flat = np.ones(voxel_count, dtype=FLOAT64)
        self._weight_flat = np.zeros(voxel_count, dtype=FLOAT64)
        self._source_frame_ids: list[int] = []
        self._seen_frame_ids: set[int] = set()
        self._integrated_observation_count = 0

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return self._shape_xyz

    @property
    def voxel_count(self) -> int:
        return int(self._tsdf_flat.size)

    @property
    def source_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._source_frame_ids)

    @property
    def tsdf_array_bytes(self) -> int:
        return int(self._tsdf_flat.nbytes + self._weight_flat.nbytes)

    @property
    def centers_array_bytes(self) -> int:
        return int(self._centers_world_m.nbytes)

    @property
    def observed_voxel_count(self) -> int:
        return int(np.count_nonzero(self._weight_flat > 0.0))

    def integrate(self, observation: DepthObservation) -> IncrementalTSDFUpdateStats:
        """Integrate exactly one validated observation into persistent TSDF state."""

        observed_before = self.observed_voxel_count
        integrate_depth_observation(
            observation=observation,
            centers_world_m=self._centers_world_m,
            voxel_size_m=self._config.voxel_size_m,
            truncation_distance_m=self._config.truncation_distance_m,
            tsdf_flat=self._tsdf_flat,
            weight_flat=self._weight_flat,
        )
        self._integrated_observation_count += 1
        if observation.frame_id not in self._seen_frame_ids:
            self._seen_frame_ids.add(observation.frame_id)
            self._source_frame_ids.append(observation.frame_id)
        observed_after = self.observed_voxel_count
        return IncrementalTSDFUpdateStats(
            frame_id=observation.frame_id,
            integrated_observation_count=self._integrated_observation_count,
            unique_source_frame_count=len(self._source_frame_ids),
            observed_voxel_count=observed_after,
            newly_observed_voxel_count=max(0, observed_after - observed_before),
            voxel_count=self.voxel_count,
            tsdf_array_bytes=self.tsdf_array_bytes,
            centers_array_bytes=self.centers_array_bytes,
            update_implementation=PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
        )

    def volume(self) -> TSDFVolume:
        """Return the current persistent TSDF state as a public TSDF volume."""

        return TSDFVolume(
            grid_min_corner_world_m=self._config.grid_min_world_m.astype(FLOAT32),
            voxel_size_m=self._config.voxel_size_m,
            truncation_distance_m=self._config.truncation_distance_m,
            tsdf=self._tsdf_flat.reshape(self._shape_xyz).astype(FLOAT32),
            weight=self._weight_flat.reshape(self._shape_xyz).astype(FLOAT32),
            source_frame_ids=self.source_frame_ids,
            coordinate_frame=self._config.coordinate_frame,
            metric_scale_source=self._config.metric_scale_source,
        )

    def extract_surface(self) -> TSDFSurface:
        """Extract final surface points from the current persistent TSDF state."""

        return extract_tsdf_surface(self.volume())


def _validate_corner(name: str, value: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name}: must be a finite 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: must be a finite 3-vector")
    return array.astype(np.float64, copy=True)


__all__ = [
    "IncrementalTSDFConfig",
    "IncrementalTSDFUpdateStats",
    "PERSISTENT_CPU_UPDATE_IMPLEMENTATION",
    "PersistentIncrementalTSDFMapper",
]
