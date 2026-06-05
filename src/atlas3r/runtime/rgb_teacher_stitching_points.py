"""Pointmap correspondence helpers for RGB teacher window stitching."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from atlas3r.runtime.rgb_teacher_stitching_types import (
    TeacherWindowNode,
    TeacherWindowPrediction,
)


def pointmap_correspondences(
    source_window: TeacherWindowPrediction,
    target_window: TeacherWindowPrediction,
    target_node: TeacherWindowNode,
    frame_id: int,
    *,
    max_points: int = 128,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if (
        "pointmap_world_m" not in source_window.payload
        or "pointmap_world_m" not in target_window.payload
    ):
        return _empty_points()
    source_index = _frame_offset(source_window, frame_id)
    target_index = _frame_offset(target_window, frame_id)
    source = np.asarray(source_window.payload["pointmap_world_m"][source_index], dtype=np.float32)
    target = np.asarray(target_window.payload["pointmap_world_m"][target_index], dtype=np.float32)
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        return _empty_points()
    finite = np.all(np.isfinite(source), axis=-1) & np.all(np.isfinite(target), axis=-1)
    coords = np.argwhere(finite)
    if coords.size == 0:
        return _empty_points()
    stride = max(int(math.ceil(coords.shape[0] / max_points)), 1)
    coords = coords[::stride][:max_points]
    yy = coords[:, 0]
    xx = coords[:, 1]
    source_points = source[yy, xx].reshape(-1, 3)
    target_points = target_node.sim3_local_to_global.apply_points(target[yy, xx].reshape(-1, 3))
    return source_points.astype(np.float32, copy=False), target_points


def _empty_points() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    empty = np.empty((0, 3), dtype=np.float32)
    return empty, empty


def _frame_offset(window: TeacherWindowPrediction, frame_id: int) -> int:
    try:
        return window.frame_ids.index(frame_id)
    except ValueError as exc:
        raise ValueError(f"frame_id {frame_id}: missing from window {window.window_index}") from exc
