"""Sparse block TSDF mapper for measured incremental recording diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import FLOAT32, FLOAT64, TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf_math import (
    project_sparse_candidates,
    sparse_candidate_offsets,
    sparse_candidate_voxel_coords,
    sparse_surface_samples,
    voxel_centers_from_sparse_coords,
)

SPARSE_TSDF_UPDATE_IMPLEMENTATION = "sparse_block_tsdf_surface_centric_single_observation"


@dataclass(frozen=True)
class SparseTSDFConfig:
    """Online sparse TSDF configuration with no fixed dense world bounds."""

    voxel_size_m: float
    truncation_distance_m: float
    block_size_voxels: int = 8
    max_active_blocks: int | None = None
    coordinate_frame: str = "world"
    metric_scale_source: str = "measured_depth_pose"
    pixel_stride: int = 8

    def __post_init__(self) -> None:
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_distance_m <= 0.0:
            raise ValueError("truncation_distance_m: must be positive")
        if self.block_size_voxels <= 0:
            raise ValueError("block_size_voxels: must be positive")
        if self.max_active_blocks is not None and self.max_active_blocks <= 0:
            raise ValueError("max_active_blocks: must be positive when provided")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame: must be non-empty")
        if not self.metric_scale_source:
            raise ValueError("metric_scale_source: must be non-empty")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride: must be positive")
        object.__setattr__(self, "voxel_size_m", float(self.voxel_size_m))
        object.__setattr__(self, "truncation_distance_m", float(self.truncation_distance_m))
        object.__setattr__(self, "block_size_voxels", int(self.block_size_voxels))
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame))
        object.__setattr__(self, "metric_scale_source", str(self.metric_scale_source))
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))


@dataclass(frozen=True)
class SparseTSDFUpdateStats:
    """Counters for one online sparse TSDF update."""

    frame_id: int
    integrated_observation_count: int
    unique_source_frame_count: int
    active_block_count: int
    active_voxel_count: int
    allocated_voxel_count: int
    approximate_state_bytes: int
    valid_depth_sample_count: int
    candidate_voxel_count: int
    new_voxel_count: int
    updated_voxel_count: int
    skipped_duplicate_frame: bool
    update_implementation: str


@dataclass
class _SparseBlock:
    tsdf: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]


class SparseBlockTSDFMapper:
    """Sparse block TSDF state that grows from measured observations online.

    Duplicate frame IDs are skipped deterministically. The first observation for
    a frame ID owns the state update and source-frame record.
    """

    def __init__(self, config: SparseTSDFConfig) -> None:
        self._config = config
        self._blocks: dict[tuple[int, int, int], _SparseBlock] = {}
        self._source_frame_ids: list[int] = []
        self._seen_frame_ids: set[int] = set()
        self._integrated_observation_count = 0
        self._active_voxel_count = 0
        self._offsets_xyz = sparse_candidate_offsets(
            voxel_size_m=config.voxel_size_m,
            truncation_distance_m=config.truncation_distance_m,
        )

    @property
    def source_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._source_frame_ids)

    @property
    def active_block_count(self) -> int:
        return len(self._blocks)

    @property
    def active_voxel_count(self) -> int:
        return int(self._active_voxel_count)

    @property
    def allocated_voxel_count(self) -> int:
        return int(self.active_block_count * self._config.block_size_voxels**3)

    @property
    def approximate_state_bytes(self) -> int:
        bytes_per_block = self._config.block_size_voxels**3 * (np.dtype(np.float32).itemsize * 2)
        block_key_bytes = self.active_block_count * 3 * np.dtype(np.int64).itemsize
        return int(self.active_block_count * bytes_per_block + block_key_bytes)

    def integrate(self, observation: DepthObservation) -> SparseTSDFUpdateStats:
        """Integrate one measured observation into sparse block state."""

        if observation.frame_id in self._seen_frame_ids:
            return self._stats(
                observation,
                valid_depth_sample_count=0,
                candidate_voxel_count=0,
                new_voxel_count=0,
                updated_voxel_count=0,
                skipped_duplicate_frame=True,
            )

        samples = sparse_surface_samples(observation, pixel_stride=self._config.pixel_stride)
        if samples.valid_depth_sample_count == 0:
            self._record_source_frame(observation.frame_id)
            return self._stats(
                observation,
                valid_depth_sample_count=0,
                candidate_voxel_count=0,
                new_voxel_count=0,
                updated_voxel_count=0,
                skipped_duplicate_frame=False,
            )

        candidate_coords = sparse_candidate_voxel_coords(
            samples.points_world_m,
            voxel_size_m=self._config.voxel_size_m,
            offsets_xyz=self._offsets_xyz,
        )
        updates = project_sparse_candidates(
            observation=observation,
            voxel_coords_xyz=candidate_coords,
            voxel_size_m=self._config.voxel_size_m,
            truncation_distance_m=self._config.truncation_distance_m,
        )
        new_count, updated_count = self._apply_updates(
            updates.voxel_coords_xyz,
            updates.tsdf,
            updates.weight,
        )
        self._record_source_frame(observation.frame_id)
        self._integrated_observation_count += 1
        return self._stats(
            observation,
            valid_depth_sample_count=samples.valid_depth_sample_count,
            candidate_voxel_count=int(candidate_coords.shape[0]),
            new_voxel_count=new_count,
            updated_voxel_count=updated_count,
            skipped_duplicate_frame=False,
        )

    def extract_surface(self, *, surface_band: float = 1.0 / 3.0) -> TSDFSurface:
        """Extract observed near-zero TSDF voxel centers from sparse blocks."""

        if surface_band <= 0.0:
            raise ValueError("surface_band: must be positive")
        voxel_coords, tsdf, weight = self.active_voxel_arrays()
        if voxel_coords.size == 0:
            raise ValueError("sparse TSDF surface extraction produced no observed voxels")
        surface_mask = np.abs(tsdf.astype(FLOAT64, copy=False)) <= surface_band
        if not np.any(surface_mask):
            raise ValueError("sparse TSDF surface extraction produced no observed surface voxels")
        surface_coords = voxel_coords[surface_mask]
        surface_weight = weight[surface_mask].astype(FLOAT64, copy=False)
        surface_tsdf = tsdf[surface_mask].astype(FLOAT64, copy=False)
        points = voxel_centers_from_sparse_coords(surface_coords, self._config.voxel_size_m)
        confidence = np.clip(
            surface_weight / max(float(len(self._source_frame_ids)), 1.0),
            0.0,
            1.0,
        )
        uncertainty = (
            self._config.voxel_size_m / np.sqrt(np.maximum(surface_weight, 1.0))
            + np.abs(surface_tsdf) * self._config.truncation_distance_m
        )
        metadata = self.surface_metadata(
            surface_count=int(surface_coords.shape[0]),
            surface_band=surface_band,
            uncertainty_m=uncertainty,
        )
        return TSDFSurface(
            points_world_m=points.astype(FLOAT32),
            confidence=confidence.astype(FLOAT32),
            uncertainty_m=uncertainty.astype(FLOAT32),
            voxel_indices_xyz=surface_coords.astype(np.int32),
            metadata=metadata,
        )

    def active_voxel_arrays(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return sparse active voxel coordinates, TSDF values, and weights."""

        coords_parts: list[npt.NDArray[np.int64]] = []
        tsdf_parts: list[npt.NDArray[np.float32]] = []
        weight_parts: list[npt.NDArray[np.float32]] = []
        block_size = self._config.block_size_voxels
        for block_key in sorted(self._blocks):
            block = self._blocks[block_key]
            active_local = np.argwhere(block.weight > 0.0)
            if active_local.size == 0:
                continue
            block_origin = np.asarray(block_key, dtype=np.int64) * block_size
            coords = block_origin[None, :] + active_local.astype(np.int64, copy=False)
            coords_parts.append(coords)
            local = tuple(active_local[:, axis] for axis in range(3))
            tsdf_parts.append(block.tsdf[local].astype(FLOAT32, copy=False))
            weight_parts.append(block.weight[local].astype(FLOAT32, copy=False))
        if not coords_parts:
            return (
                np.empty((0, 3), dtype=np.int64),
                np.empty((0,), dtype=FLOAT32),
                np.empty((0,), dtype=FLOAT32),
            )
        return (
            np.concatenate(coords_parts, axis=0),
            np.concatenate(tsdf_parts, axis=0),
            np.concatenate(weight_parts, axis=0),
        )

    def block_coordinates(self) -> npt.NDArray[np.int64]:
        """Return allocated block coordinates sorted lexicographically."""

        if not self._blocks:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(sorted(self._blocks), dtype=np.int64)

    def surface_metadata(
        self,
        *,
        surface_count: int,
        surface_band: float,
        uncertainty_m: npt.NDArray[np.float64],
    ) -> dict[str, Any]:
        return {
            "artifact_type": "phase6d_sparse_block_tsdf_surface_points",
            "coordinate_frame": self._config.coordinate_frame,
            "metric_scale_source": self._config.metric_scale_source,
            "source_frame_ids": list(self._source_frame_ids),
            "voxel_size_m": self._config.voxel_size_m,
            "truncation_distance_m": self._config.truncation_distance_m,
            "block_size_voxels": self._config.block_size_voxels,
            "pixel_stride": self._config.pixel_stride,
            "active_block_count": self.active_block_count,
            "active_voxel_count": self.active_voxel_count,
            "allocated_voxel_count": self.allocated_voxel_count,
            "approximate_state_bytes": self.approximate_state_bytes,
            "surface_band_abs_normalized_tsdf": surface_band,
            "surface_voxel_count": surface_count,
            "observed_coverage_estimate": float(
                self.active_voxel_count / max(self.allocated_voxel_count, 1)
            ),
            "surface_coverage_estimate": float(surface_count / max(self.allocated_voxel_count, 1)),
            "uncertainty_summary_m": _uncertainty_summary(uncertainty_m),
            "duplicate_frame_policy": "skip",
            "flags": [
                "phase6d_sparse_block_tsdf",
                "observed_surface_points",
                "not_completed_surface",
                "not_accuracy_report",
                "not_performance_report",
                "not_realtime_claim",
            ],
        }

    def _apply_updates(
        self,
        voxel_coords_xyz: npt.NDArray[np.int64],
        tsdf: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
    ) -> tuple[int, int]:
        if voxel_coords_xyz.size == 0:
            return 0, 0
        block_size = self._config.block_size_voxels
        block_coords = np.floor_divide(voxel_coords_xyz, block_size)
        local_coords = voxel_coords_xyz - block_coords * block_size
        unique_blocks, inverse = np.unique(block_coords, axis=0, return_inverse=True)
        new_voxels = 0
        updated_voxels = 0
        for block_index, block_coord in enumerate(unique_blocks):
            block_key = (
                int(block_coord[0]),
                int(block_coord[1]),
                int(block_coord[2]),
            )
            block = self._block_for_update(block_key)
            mask = inverse == block_index
            local = local_coords[mask]
            local_index = tuple(local[:, axis] for axis in range(3))
            old_weight = block.weight[local_index]
            update_weight = weight[mask]
            new_weight = old_weight + update_weight
            block.tsdf[local_index] = (
                block.tsdf[local_index] * old_weight + tsdf[mask] * update_weight
            ) / new_weight
            block.weight[local_index] = new_weight
            newly_active = (old_weight <= 0.0) & (new_weight > 0.0)
            new_voxels += int(np.count_nonzero(newly_active))
            updated_voxels += int(np.count_nonzero(~newly_active & (update_weight > 0.0)))
        self._active_voxel_count += new_voxels
        return new_voxels, updated_voxels

    def _block_for_update(self, block_key: tuple[int, int, int]) -> _SparseBlock:
        block = self._blocks.get(block_key)
        if block is not None:
            return block
        if (
            self._config.max_active_blocks is not None
            and len(self._blocks) >= self._config.max_active_blocks
        ):
            raise RuntimeError("max_active_blocks: sparse TSDF block limit exceeded")
        shape = (self._config.block_size_voxels,) * 3
        block = _SparseBlock(
            tsdf=np.ones(shape, dtype=FLOAT32),
            weight=np.zeros(shape, dtype=FLOAT32),
        )
        self._blocks[block_key] = block
        return block

    def _record_source_frame(self, frame_id: int) -> None:
        self._seen_frame_ids.add(frame_id)
        self._source_frame_ids.append(frame_id)

    def _stats(
        self,
        observation: DepthObservation,
        *,
        valid_depth_sample_count: int,
        candidate_voxel_count: int,
        new_voxel_count: int,
        updated_voxel_count: int,
        skipped_duplicate_frame: bool,
    ) -> SparseTSDFUpdateStats:
        return SparseTSDFUpdateStats(
            frame_id=observation.frame_id,
            integrated_observation_count=self._integrated_observation_count,
            unique_source_frame_count=len(self._source_frame_ids),
            active_block_count=self.active_block_count,
            active_voxel_count=self.active_voxel_count,
            allocated_voxel_count=self.allocated_voxel_count,
            approximate_state_bytes=self.approximate_state_bytes,
            valid_depth_sample_count=valid_depth_sample_count,
            candidate_voxel_count=candidate_voxel_count,
            new_voxel_count=new_voxel_count,
            updated_voxel_count=updated_voxel_count,
            skipped_duplicate_frame=skipped_duplicate_frame,
            update_implementation=SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        )


def _uncertainty_summary(values: npt.NDArray[np.float64]) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


__all__ = [
    "SPARSE_TSDF_UPDATE_IMPLEMENTATION",
    "SparseBlockTSDFMapper",
    "SparseTSDFConfig",
    "SparseTSDFUpdateStats",
]
