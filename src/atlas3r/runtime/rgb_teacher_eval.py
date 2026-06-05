"""Eval-only diagnostics for RGB teacher mapping on measured recordings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherInput
from atlas3r.runtime.rgb_teacher_outputs import rgb_teacher_truth_boundary
from atlas3r.runtime.student_map_report_common import write_json


def write_rgb_teacher_eval_if_available(
    output: Path,
    *,
    source: RGBTeacherInput,
    observations: Sequence[DepthObservation],
    metric_scale_source: str,
) -> dict[str, object] | None:
    """Write eval-only metrics when source measured depth/pose exist."""

    if source.recording is None:
        return None
    if not source.measured_depth_available or not source.measured_pose_available:
        return None
    recording = source.recording
    frames_by_id = {frame.frame_id: frame for frame in recording.frames}
    matched: list[tuple[DepthObservation, DepthObservation]] = []
    for observation in observations:
        frame = frames_by_id.get(observation.frame_id)
        if frame is None or frame.depth_path is None or frame.T_world_camera is None:
            continue
        matched.append(
            (
                observation,
                observation_from_recording_frame(recording, frame, depth_sigma_floor_m=0.01),
            )
        )
    if not matched:
        return None
    depth_summary = _depth_eval(matched)
    pose_summary, centers_csv = _pose_eval(matched)
    summary = {
        **rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source),
        "format_name": "atlas3r_rgb_teacher_eval_summary",
        "format_version": 1,
        "comparison_type": "diagnostic_eval_only",
        "measured_depth_used_for_mapping": False,
        "measured_pose_used_for_mapping": False,
        "measured_depth_used_for_eval": True,
        "measured_pose_used_for_eval": True,
        "matched_frame_count": len(matched),
        "depth": depth_summary,
        "pose": pose_summary,
        "camera_centers_csv": "rgb_teacher_eval_camera_centers.csv",
        "truth_boundary": {
            **rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source),
            "measured_depth_used_for_eval": True,
            "measured_pose_used_for_eval": True,
            "measured_depth_used_for_mapping": False,
            "measured_pose_used_for_mapping": False,
        },
        "known_limitations": [
            "This diagnostic compares teacher pseudo labels to source recording measurements.",
            "Depth is scale-aligned for evaluation only and mapping outputs are not changed.",
            "Pose ATE is Sim3-aligned camera-center error for diagnostics, not a benchmark claim.",
        ],
    }
    write_json(output / "rgb_teacher_eval.json", summary)
    (output / "rgb_teacher_eval_camera_centers.csv").write_text(centers_csv, encoding="utf-8")
    _write_eval_report(output / "rgb_teacher_eval.md", summary)
    return summary


def _depth_eval(
    matched: Sequence[tuple[DepthObservation, DepthObservation]],
) -> dict[str, object]:
    abs_errors: list[npt.NDArray[np.float64]] = []
    abs_rels: list[npt.NDArray[np.float64]] = []
    valid_ratios: list[float] = []
    scales: list[float] = []
    for pseudo, measured in matched:
        pred = pseudo.depth_m.astype(np.float64, copy=False)
        gt = measured.depth_m.astype(np.float64, copy=False)
        valid = (
            np.isfinite(pred)
            & np.isfinite(gt)
            & (pred > 0.0)
            & (gt > 0.0)
            & (pseudo.confidence > 0.0)
        )
        valid_ratios.append(float(np.count_nonzero(valid) / valid.size))
        if not np.any(valid):
            continue
        scale = float(np.median(gt[valid] / np.maximum(pred[valid], 1e-12)))
        scales.append(scale)
        aligned = pred[valid] * scale
        error = np.abs(aligned - gt[valid])
        abs_errors.append(error)
        abs_rels.append(error / np.maximum(gt[valid], 1e-12))
    errors = _concat(abs_errors)
    rels = _concat(abs_rels)
    return {
        "alignment": "median_scale_eval_only",
        "scale_median": _median_or_none(np.asarray(scales, dtype=np.float64)),
        "valid_depth_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
        "overlap_valid_pixel_count": int(errors.size),
        "absrel": _mean_or_none(rels),
        "rmse_m": _rmse(errors),
    }


def _pose_eval(
    matched: Sequence[tuple[DepthObservation, DepthObservation]],
) -> tuple[dict[str, object], str]:
    pseudo_centers = np.asarray(
        [pseudo.pose.camera_center_world_m for pseudo, _measured in matched],
        dtype=np.float64,
    )
    measured_centers = np.asarray(
        [measured.pose.camera_center_world_m for _pseudo, measured in matched],
        dtype=np.float64,
    )
    aligned = _sim3_align_points(pseudo_centers, measured_centers)
    errors = np.linalg.norm(aligned - measured_centers, axis=1)
    csv_lines = [
        "frame_id,pseudo_x,pseudo_y,pseudo_z,measured_x,measured_y,measured_z,"
        "aligned_x,aligned_y,aligned_z,error_m"
    ]
    for (pseudo, _measured), raw, measured, aligned_point, error in zip(
        matched,
        pseudo_centers,
        measured_centers,
        aligned,
        errors,
        strict=True,
    ):
        values = [
            str(pseudo.frame_id),
            *[f"{value:.9g}" for value in raw],
            *[f"{value:.9g}" for value in measured],
            *[f"{value:.9g}" for value in aligned_point],
            f"{float(error):.9g}",
        ]
        csv_lines.append(",".join(values))
    return (
        {
            "alignment": "sim3_camera_center_eval_only",
            "ate_rmse_m": _rmse(errors),
            "ate_mean_m": _mean_or_none(errors),
            "ate_max_m": float(np.max(errors)) if errors.size else None,
            "relative_pose_error": None,
            "relative_pose_error_note": "RPE is not implemented in this Phase 6G diagnostic.",
        },
        "\n".join(csv_lines) + "\n",
    )


def _sim3_align_points(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if source.shape[0] < 2:
        aligned = source + (target.mean(axis=0, keepdims=True) - source.mean(axis=0, keepdims=True))
        return cast(npt.NDArray[np.float64], aligned)
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)
    src_centered = source - src_mean[None, :]
    tgt_centered = target - tgt_mean[None, :]
    covariance = (tgt_centered.T @ src_centered) / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = 1.0 if variance <= 1e-12 else float(np.sum(singular * np.diag(correction)) / variance)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return cast(npt.NDArray[np.float64], scale * (source @ rotation.T) + translation[None, :])


def _write_eval_report(path: Path, summary: dict[str, object]) -> None:
    depth = cast(dict[str, Any], summary["depth"])
    pose = cast(dict[str, Any], summary["pose"])
    lines = [
        "# Phase 6G RGB Teacher Diagnostic Eval",
        "",
        f"- Matched frames: `{summary['matched_frame_count']}`",
        f"- Pose ATE RMSE after Sim3: `{pose['ate_rmse_m']}` m",
        f"- Depth AbsRel after eval-only median scale: `{depth['absrel']}`",
        f"- Depth RMSE after eval-only median scale: `{depth['rmse_m']}` m",
        "",
        "This is an eval-only diagnostic comparison against source recording measurements. "
        "It is not a benchmark accuracy report and does not alter mapping outputs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _concat(arrays: Sequence[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    if not arrays:
        return np.zeros((0,), dtype=np.float64)
    return cast(npt.NDArray[np.float64], np.concatenate(arrays))


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _median_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _rmse(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(math.sqrt(float(np.mean(values * values))))


__all__ = ["write_rgb_teacher_eval_if_available"]
