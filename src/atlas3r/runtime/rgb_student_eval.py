"""Eval-only diagnostics for RGB student mapping on measured recordings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherInput
from atlas3r.runtime.student_map_report_common import write_json


def write_rgb_student_eval_if_available(
    output: Path,
    *,
    source: RGBTeacherInput,
    observations: Sequence[DepthObservation],
    truth_boundary: dict[str, object],
) -> dict[str, object] | None:
    if (
        source.recording is None
        or not source.measured_depth_available
        or not source.measured_pose_available
    ):
        return None
    frames_by_id = {frame.frame_id: frame for frame in source.recording.frames}
    matched: list[tuple[DepthObservation, DepthObservation]] = []
    for observation in observations:
        frame = frames_by_id.get(observation.frame_id)
        if frame is None or frame.depth_path is None or frame.T_world_camera is None:
            continue
        measured = observation_from_recording_frame(
            source.recording, frame, depth_sigma_floor_m=0.01
        )
        matched.append((observation, measured))
    if not matched:
        return None
    depth = _depth_eval(matched)
    pose, centers_csv = _pose_eval(matched)
    summary = {
        **truth_boundary,
        "format_name": "atlas3r_rgb_student_eval_summary",
        "format_version": 1,
        "comparison_type": "diagnostic_eval_only",
        "measured_depth_used_for_mapping": False,
        "measured_pose_used_for_mapping": False,
        "measured_depth_used_for_eval": True,
        "measured_pose_used_for_eval": True,
        "matched_frame_count": len(matched),
        "depth": depth,
        "pose": pose,
        "camera_centers_csv": "rgb_student_eval_camera_centers.csv",
        "truth_boundary": {
            **truth_boundary,
            "measured_depth_used_for_eval": True,
            "measured_pose_used_for_eval": True,
            "measured_depth_used_for_mapping": False,
            "measured_pose_used_for_mapping": False,
        },
        "known_limitations": [
            "This diagnostic compares student RGB predictions to source recording measurements.",
            "Depth is median-scale aligned for evaluation only; mapping outputs are unchanged.",
            "Pose ATE is Sim3-aligned camera-center error for diagnostics, not a benchmark claim.",
        ],
    }
    write_json(output / "rgb_student_eval.json", summary)
    (output / "rgb_student_eval_camera_centers.csv").write_text(centers_csv, encoding="utf-8")
    _write_report(output / "rgb_student_eval.md", summary)
    return summary


def _depth_eval(matched: Sequence[tuple[DepthObservation, DepthObservation]]) -> dict[str, object]:
    abs_errors: list[npt.NDArray[np.float64]] = []
    abs_rels: list[npt.NDArray[np.float64]] = []
    constant_abs_errors: list[npt.NDArray[np.float64]] = []
    constant_abs_rels: list[npt.NDArray[np.float64]] = []
    valid_ratios: list[float] = []
    scales: list[float] = []
    for student, measured in matched:
        pred = student.depth_m.astype(np.float64, copy=False)
        gt = measured.depth_m.astype(np.float64, copy=False)
        if gt.shape != pred.shape:
            gt = _resize_2d_nearest(gt, height=pred.shape[0], width=pred.shape[1])
        valid = (
            np.isfinite(pred)
            & np.isfinite(gt)
            & (pred > 0.0)
            & (gt > 0.0)
            & (student.confidence > 0.0)
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
        constant_depth = float(np.median(gt[valid]))
        constant_error = np.abs(constant_depth - gt[valid])
        constant_abs_errors.append(constant_error)
        constant_abs_rels.append(constant_error / np.maximum(gt[valid], 1e-12))
    errors = _concat(abs_errors)
    rels = _concat(abs_rels)
    constant_errors = _concat(constant_abs_errors)
    constant_rels = _concat(constant_abs_rels)
    absrel = _mean_or_none(rels)
    rmse_m = _rmse(errors)
    baseline_absrel = _mean_or_none(constant_rels)
    baseline_rmse_m = _rmse(constant_errors)
    return {
        "alignment": "median_scale_eval_only",
        "scale_median": _median_or_none(np.asarray(scales, dtype=np.float64)),
        "valid_depth_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
        "overlap_valid_pixel_count": int(errors.size),
        "absrel": absrel,
        "rmse_m": rmse_m,
        "constant_depth_baseline": "per_frame_median_depth_on_same_valid_pixels",
        "constant_depth_baseline_absrel": baseline_absrel,
        "constant_depth_baseline_rmse_m": baseline_rmse_m,
        "student_beats_constant_depth_baseline": (
            None if absrel is None or baseline_absrel is None else absrel < baseline_absrel
        ),
    }


def _pose_eval(
    matched: Sequence[tuple[DepthObservation, DepthObservation]],
) -> tuple[dict[str, object], str]:
    student_centers = np.asarray(
        [student.pose.camera_center_world_m for student, _ in matched], dtype=np.float64
    )
    measured_centers = np.asarray(
        [measured.pose.camera_center_world_m for _, measured in matched], dtype=np.float64
    )
    aligned = _sim3_align_points(student_centers, measured_centers)
    errors = np.linalg.norm(aligned - measured_centers, axis=1)
    no_motion_centers = np.repeat(measured_centers[:1], measured_centers.shape[0], axis=0)
    no_motion_errors = np.linalg.norm(no_motion_centers - measured_centers, axis=1)
    ate_rmse_m = _rmse(errors)
    no_motion_rmse_m = _rmse(no_motion_errors)
    csv_lines = [
        "frame_id,student_x,student_y,student_z,measured_x,measured_y,measured_z,"
        "aligned_x,aligned_y,aligned_z,error_m"
    ]
    for (student, _), raw, measured, aligned_point, error in zip(
        matched, student_centers, measured_centers, aligned, errors, strict=True
    ):
        values = [
            str(student.frame_id),
            *[f"{value:.9g}" for value in raw],
            *[f"{value:.9g}" for value in measured],
            *[f"{value:.9g}" for value in aligned_point],
            f"{float(error):.9g}",
        ]
        csv_lines.append(",".join(values))
    return (
        {
            "alignment": "sim3_camera_center_eval_only",
            "ate_rmse_m": ate_rmse_m,
            "ate_mean_m": _mean_or_none(errors),
            "ate_max_m": float(np.max(errors)) if errors.size else None,
            "no_motion_pose_baseline": "first_measured_camera_center_repeated_eval_only",
            "no_motion_pose_baseline_ate_rmse_m": no_motion_rmse_m,
            "student_beats_no_motion_pose_baseline": (
                None
                if ate_rmse_m is None or no_motion_rmse_m is None
                else ate_rmse_m < no_motion_rmse_m
            ),
        },
        "\n".join(csv_lines) + "\n",
    )


def _sim3_align_points(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if source.shape[0] < 2:
        return cast(
            npt.NDArray[np.float64],
            source + (target.mean(axis=0, keepdims=True) - source.mean(axis=0, keepdims=True)),
        )
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


def _write_report(path: Path, summary: dict[str, object]) -> None:
    depth = cast(dict[str, object], summary["depth"])
    pose = cast(dict[str, object], summary["pose"])
    path.write_text(
        "\n".join(
            [
                "# RGB Student Diagnostic Eval",
                "",
                f"- Matched frames: `{summary['matched_frame_count']}`",
                f"- Pose ATE RMSE after Sim3: `{pose['ate_rmse_m']}` m",
                f"- No-motion pose baseline ATE RMSE: "
                f"`{pose['no_motion_pose_baseline_ate_rmse_m']}` m",
                f"- Student beats no-motion pose baseline: "
                f"`{pose['student_beats_no_motion_pose_baseline']}`",
                f"- Depth AbsRel after eval-only median scale: `{depth['absrel']}`",
                f"- Depth RMSE after eval-only median scale: `{depth['rmse_m']}` m",
                f"- Constant-depth baseline AbsRel: `{depth['constant_depth_baseline_absrel']}`",
                f"- Student beats constant-depth baseline: "
                f"`{depth['student_beats_constant_depth_baseline']}`",
                "",
                "Eval-only measurements do not alter student mapping outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _concat(arrays: Sequence[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    if not arrays:
        return np.zeros((0,), dtype=np.float64)
    return cast(npt.NDArray[np.float64], np.concatenate(arrays))


def _resize_2d_nearest(
    array: npt.NDArray[np.float64],
    *,
    height: int,
    width: int,
) -> npt.NDArray[np.float64]:
    if array.shape == (height, width):
        return array.astype(np.float64, copy=True)
    y = np.linspace(0, array.shape[0] - 1, height).round().astype(np.int64)
    x = np.linspace(0, array.shape[1] - 1, width).round().astype(np.int64)
    return cast(
        npt.NDArray[np.float64], array[y[:, None], x[None, :]].astype(np.float64, copy=True)
    )


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _median_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def _rmse(values: npt.NDArray[np.float64]) -> float | None:
    return None if values.size == 0 else float(math.sqrt(float(np.mean(values * values))))


__all__ = ["write_rgb_student_eval_if_available"]
