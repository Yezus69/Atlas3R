"""Configuration for RGB teacher-assisted runtime mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RGBTeacherMapConfig:
    input: Path
    output: Path
    teacher: str = "vggt"
    device: str = "auto"
    max_frames: int | None = 64
    frame_stride: int = 2
    teacher_window_size: int = 24
    teacher_window_overlap: int = 8
    image_size: int = 518
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0
    pixel_stride: int = 12
    export_mesh_chunks: bool = False
    mesh_format: str = "npz"
    mesh_min_weight: float = 0.0
    export_point_cloud: bool = False
    rgb_only: bool = False
    sim3_align_for_eval_only: bool = False
    stitch_windows: str = "sim3-overlap"
    stitch_min_overlap_frames: int = 4
    stitch_max_center_rmse_m: float = 0.25
    stitch_max_scale_ratio: float = 2.0
    stitch_min_inliers: int = 4
    export_teacher_cache: bool = False
    teacher_cache_format: str = "atlas3r_teacher_temporal_cache_v1"
    cache_clip_length: int = 8
    cache_clip_stride: int = 4
    remap_from_teacher_cache: Path | None = None
    vggt_repo: Path | None = None
    checkpoint: str | None = None
    metric_scale_source: str = "teacher_scale_unverified"

    def __post_init__(self) -> None:
        if self.teacher not in {"vggt", "fixture-vggt"}:
            raise ValueError("teacher: must be vggt or fixture-vggt")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames: must be positive when provided")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride: must be positive")
        if self.teacher_window_size <= 0:
            raise ValueError("teacher_window_size: must be positive")
        if self.teacher_window_overlap < 0:
            raise ValueError("teacher_window_overlap: must be non-negative")
        if self.teacher_window_overlap >= self.teacher_window_size:
            raise ValueError("teacher_window_overlap: must be smaller than teacher_window_size")
        if self.image_size <= 0:
            raise ValueError("image_size: must be positive")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_voxels <= 0.0:
            raise ValueError("truncation_voxels: must be positive")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride: must be positive")
        if self.mesh_format not in {"npz", "ply", "both"}:
            raise ValueError("mesh_format: must be npz, ply, or both")
        if self.mesh_min_weight < 0.0:
            raise ValueError("mesh_min_weight: must be non-negative")
        if self.stitch_windows not in {"none", "sim3-overlap"}:
            raise ValueError("stitch_windows: must be none or sim3-overlap")
        if self.stitch_min_overlap_frames <= 0:
            raise ValueError("stitch_min_overlap_frames: must be positive")
        if self.stitch_max_center_rmse_m <= 0.0:
            raise ValueError("stitch_max_center_rmse_m: must be positive")
        if self.stitch_max_scale_ratio < 1.0:
            raise ValueError("stitch_max_scale_ratio: must be >= 1")
        if self.stitch_min_inliers <= 0:
            raise ValueError("stitch_min_inliers: must be positive")
        if self.teacher_cache_format != "atlas3r_teacher_temporal_cache_v1":
            raise ValueError("teacher_cache_format: must be atlas3r_teacher_temporal_cache_v1")
        if self.cache_clip_length <= 0:
            raise ValueError("cache_clip_length: must be positive")
        if self.cache_clip_stride <= 0:
            raise ValueError("cache_clip_stride: must be positive")
        if not self.metric_scale_source:
            raise ValueError("metric_scale_source: must be non-empty")


__all__ = ["RGBTeacherMapConfig"]
