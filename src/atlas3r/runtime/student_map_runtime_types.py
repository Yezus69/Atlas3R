"""Shared types for Phase 5E student map runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

POSE_MODES = ("oracle", "student-relative", "student-odometry", "both")


@dataclass(frozen=True)
class StudentMapRuntimeConfig:
    checkpoint: Path
    clip_cache: Path
    teacher_cache: Path
    output: Path
    device: str = "auto"
    max_frames: int = 60
    window_size: int = 5
    pose_mode: str = "oracle"
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0


__all__ = [
    "POSE_MODES",
    "StudentMapRuntimeConfig",
]
