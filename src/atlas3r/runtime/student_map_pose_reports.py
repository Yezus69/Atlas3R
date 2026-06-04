"""Pose quality reports for student map runtime diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import (
    compose_transforms,
    invert_transform,
    quaternion_xyzw_from_rotation_matrix,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    TeacherReferenceFrame,
    mean_or_none,
    percentile_or_none,
    write_json,
    write_jsonl,
)


def write_pose_quality_reports(
    output: Path,
    *,
    observations: tuple[DepthObservation, ...],
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
    pose_mode: str,
) -> dict[str, object]:
    per_frame = [
        _pose_quality_record(observation, references_by_frame_id, pose_mode)
        for observation in observations
    ]
    compared = [record for record in per_frame if record["reference_available"]]
    translation_errors = np.asarray(
        [record["translation_error_m"] for record in compared],
        dtype=np.float64,
    )
    rotation_errors = np.asarray(
        [record["rotation_error_deg"] for record in compared],
        dtype=np.float64,
    )
    rpe_translation, rpe_rotation = _rpe_pose_errors(observations, references_by_frame_id)
    report: dict[str, object] = {
        "format_name": "atlas3r_phase5g_pose_quality_report",
        "format_version": 1,
        **DIAGNOSTIC_TRUTH_FLAGS,
        "pose_mode": pose_mode,
        "observation_count": len(observations),
        "reference_frame_count": len(references_by_frame_id),
        "compared_frame_count": len(compared),
        "trajectory_format": "TUM timestamp tx ty tz qx qy qz qw",
        "aggregate": {
            "translation_mean_m": mean_or_none(translation_errors),
            "translation_median_m": percentile_or_none(translation_errors, 50.0),
            "translation_p95_m": percentile_or_none(translation_errors, 95.0),
            "rotation_mean_deg": mean_or_none(rotation_errors),
            "rotation_median_deg": percentile_or_none(rotation_errors, 50.0),
            "rotation_p95_deg": percentile_or_none(rotation_errors, 95.0),
            "ate_like_camera_center_mean_m": mean_or_none(translation_errors),
            "ate_like_camera_center_median_m": percentile_or_none(translation_errors, 50.0),
            "ate_like_camera_center_p95_m": percentile_or_none(translation_errors, 95.0),
            "rpe_like_translation_mean_m": mean_or_none(rpe_translation),
            "rpe_like_translation_median_m": percentile_or_none(rpe_translation, 50.0),
            "rpe_like_translation_p95_m": percentile_or_none(rpe_translation, 95.0),
            "rpe_like_rotation_mean_deg": mean_or_none(rpe_rotation),
            "rpe_like_rotation_median_deg": percentile_or_none(rpe_rotation, 50.0),
            "rpe_like_rotation_p95_deg": percentile_or_none(rpe_rotation, 95.0),
        },
        "known_limitations": [
            "Pose quality is diagnostic and uses the provided teacher cache as reference.",
            (
                "student-odometry anchors only the first frame to the source pose and "
                "rolls out relative SE(3)."
            ),
            (
                "No mapping-readiness, realtime, benchmark accuracy, or "
                "millimeter-accuracy claim is made."
            ),
        ],
    }
    write_json(output / "pose_quality_report.json", report)
    write_jsonl(output / "per_frame_pose_quality.jsonl", per_frame)
    _write_tum_trajectory(output / "trajectory_estimate_tum.txt", observations)
    _write_reference_tum_trajectory(
        output / "trajectory_groundtruth_tum.txt",
        observations,
        references_by_frame_id,
    )
    return report


def _pose_quality_record(
    observation: DepthObservation,
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
    pose_mode: str,
) -> dict[str, object]:
    reference = references_by_frame_id.get(observation.frame_id)
    base: dict[str, object] = {
        "frame_id": observation.frame_id,
        "timestamp_s": observation.pose.timestamp_ns / 1_000_000_000.0,
        "pose_mode": pose_mode,
        "reference_available": reference is not None,
    }
    if reference is None:
        return {**base, "translation_error_m": None, "rotation_error_deg": None}
    translation_error = float(
        np.linalg.norm(
            observation.pose.T_world_camera[:3, 3].astype(np.float64)
            - reference.T_world_camera[:3, 3].astype(np.float64)
        )
    )
    rotation_error = _rotation_error_deg(
        observation.pose.T_world_camera[:3, :3],
        reference.T_world_camera[:3, :3],
    )
    return {
        **base,
        "teacher_name": reference.teacher_name,
        "teacher_measured_geometry": reference.measured_geometry,
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
    }


def _write_tum_trajectory(path: Path, observations: tuple[DepthObservation, ...]) -> None:
    lines = [
        _tum_line(
            observation.pose.timestamp_ns / 1_000_000_000.0,
            observation.pose.T_world_camera,
        )
        for observation in observations
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_reference_tum_trajectory(
    path: Path,
    observations: tuple[DepthObservation, ...],
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
) -> None:
    lines = [
        _tum_line(reference.timestamp_s, reference.T_world_camera)
        for observation in observations
        for reference in [references_by_frame_id.get(observation.frame_id)]
        if reference is not None
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _tum_line(timestamp_s: float, T_world_camera: npt.NDArray[np.float32]) -> str:
    q = quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3])
    t = T_world_camera[:3, 3]
    return (
        f"{timestamp_s:.9f} {float(t[0]):.9g} {float(t[1]):.9g} {float(t[2]):.9g} "
        f"{float(q[0]):.9g} {float(q[1]):.9g} {float(q[2]):.9g} {float(q[3]):.9g}"
    )


def _rpe_pose_errors(
    observations: tuple[DepthObservation, ...],
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    previous_observation: DepthObservation | None = None
    previous_reference: TeacherReferenceFrame | None = None
    for observation in observations:
        reference = references_by_frame_id.get(observation.frame_id)
        if (
            previous_observation is not None
            and previous_reference is not None
            and reference is not None
        ):
            estimated_relative = compose_transforms(
                invert_transform(previous_observation.pose.T_world_camera),
                observation.pose.T_world_camera,
            )
            reference_relative = compose_transforms(
                invert_transform(previous_reference.T_world_camera),
                reference.T_world_camera,
            )
            translation_errors.append(
                float(
                    np.linalg.norm(
                        estimated_relative[:3, 3].astype(np.float64)
                        - reference_relative[:3, 3].astype(np.float64)
                    )
                )
            )
            rotation_errors.append(
                _rotation_error_deg(
                    estimated_relative[:3, :3],
                    reference_relative[:3, :3],
                )
            )
        previous_observation = observation
        previous_reference = reference
    return (
        np.asarray(translation_errors, dtype=np.float64),
        np.asarray(rotation_errors, dtype=np.float64),
    )


def _rotation_error_deg(
    estimated: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    reference: npt.NDArray[np.float32] | npt.NDArray[np.float64],
) -> float:
    relative = estimated.astype(np.float64).T @ reference.astype(np.float64)
    cos_theta = (float(np.trace(relative)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


__all__ = ["write_pose_quality_reports"]
