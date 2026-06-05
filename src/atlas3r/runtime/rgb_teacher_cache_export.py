"""Runtime bridge for exporting RGB teacher observations as temporal caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from atlas3r.mapping.observations import DepthObservation
from atlas3r.training.teacher_temporal_cache import write_teacher_temporal_cache


def export_rgb_teacher_temporal_cache(
    cache_dir: Path,
    observations: Sequence[DepthObservation],
    *,
    source_rgb_teacher_run: Path,
    teacher_metadata: Mapping[str, object],
    stitch_mode: str,
    metric_scale_source: str,
    clip_length: int,
    clip_stride: int,
) -> dict[str, object]:
    return write_teacher_temporal_cache(
        cache_dir,
        observations,
        source_rgb_teacher_run=source_rgb_teacher_run,
        teacher_metadata=teacher_metadata,
        stitch_mode=stitch_mode,
        metric_scale_source=metric_scale_source,
        clip_length=clip_length,
        clip_stride=clip_stride,
    )


__all__ = ["export_rgb_teacher_temporal_cache"]
