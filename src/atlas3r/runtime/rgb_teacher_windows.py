"""Windowing and retry helpers for RGB teacher mapping."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

from atlas3r.runtime.rgb_teacher_config import RGBTeacherMapConfig
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherFrame


def rgb_teacher_windows(
    frames: Sequence[RGBTeacherFrame],
    *,
    window_size: int,
    overlap: int,
) -> Iterator[tuple[RGBTeacherFrame, ...]]:
    step = window_size - overlap
    start = 0
    while start < len(frames):
        end = min(start + window_size, len(frames))
        yield tuple(frames[start:end])
        if end == len(frames):
            break
        start += step


def rgb_teacher_attempt_configs(
    config: RGBTeacherMapConfig,
) -> tuple[RGBTeacherMapConfig, ...]:
    attempts = [config]
    if config.image_size > 384:
        attempts.append(replace(config, image_size=384))
    if config.teacher_window_size > 8:
        attempts.append(replace(config, teacher_window_size=8, teacher_window_overlap=4))
    if config.max_frames is None or config.max_frames > 64:
        attempts.append(
            replace(config, max_frames=64, teacher_window_size=8, teacher_window_overlap=4)
        )
    deduped: list[RGBTeacherMapConfig] = []
    keys: set[tuple[int | None, int, int, int]] = set()
    for attempt in attempts:
        key = (
            attempt.max_frames,
            attempt.image_size,
            attempt.teacher_window_size,
            attempt.teacher_window_overlap,
        )
        if key not in keys:
            keys.add(key)
            deduped.append(attempt)
    return tuple(deduped)


__all__ = ["rgb_teacher_attempt_configs", "rgb_teacher_windows"]
