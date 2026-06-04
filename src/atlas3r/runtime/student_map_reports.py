"""Report, PLY, and latency helpers for student map runtime diagnostics."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.student_map_pose_reports import write_pose_quality_reports
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    TeacherReferenceFrame,
    write_json,
    write_jsonl,
)


class LatencyRecorder:
    """Collect wall-clock segment durations in nanoseconds."""

    def __init__(self) -> None:
        self._samples: dict[str, list[int]] = {}

    def add(self, stage_name: str, latency_ns: int) -> None:
        if latency_ns < 0:
            raise ValueError("latency_ns: must be non-negative")
        self._samples.setdefault(stage_name, []).append(int(latency_ns))

    def samples(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(values) for key, values in self._samples.items()}

    def report(self) -> dict[str, object]:
        return {
            "format_name": "atlas3r_phase5g_latency_report",
            "format_version": 1,
            **DIAGNOSTIC_TRUTH_FLAGS,
            "latency_units": "milliseconds",
            "segments": {
                key: _latency_stats(values) for key, values in sorted(self._samples.items())
            },
            "fps_equivalent": {
                "model_inference": _fps_equivalent(self._samples.get("model_inference", [])),
                "total_pipeline": _fps_equivalent(self._samples.get("total_pipeline", [])),
            },
            "known_limitations": [
                "This is a wall-clock diagnostic profile, not a performance report.",
                "No realtime claim is made from these measurements.",
            ],
        }


def write_latency_report(output: Path, recorder: LatencyRecorder) -> dict[str, object]:
    report = recorder.report()
    write_json(output / "latency_report.json", report)
    return report


def write_observation_summaries(
    output: Path,
    records: Sequence[Mapping[str, object]],
) -> Path:
    path = output / "observations_summary.jsonl"
    write_jsonl(path, records)
    return path


def write_quality_reports(
    output: Path,
    *,
    observations: tuple[DepthObservation, ...],
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
    pose_mode: str,
    tsdf_summary: Mapping[str, object],
    map_point_set_comparison: Mapping[str, object] | None,
) -> dict[str, object]:
    accumulator = _new_accumulator()
    per_frame = [
        _quality_record(observation, references_by_frame_id, pose_mode, accumulator)
        for observation in observations
    ]
    aggregate = _aggregate_record(accumulator)
    pose_quality = write_pose_quality_reports(
        output,
        observations=observations,
        references_by_frame_id=references_by_frame_id,
        pose_mode=pose_mode,
    )
    report: dict[str, object] = {
        "format_name": "atlas3r_phase5g_quality_report",
        "format_version": 1,
        **DIAGNOSTIC_TRUTH_FLAGS,
        "pose_mode": pose_mode,
        "observation_count": len(observations),
        "reference_frame_count": len(references_by_frame_id),
        "compared_frame_count": cast(int, accumulator["compared_frames"]),
        "aggregate": aggregate,
        "pose_quality": pose_quality,
        "cpu_tsdf": dict(tsdf_summary),
        "map_point_set_comparison": (
            None if map_point_set_comparison is None else dict(map_point_set_comparison)
        ),
        "known_limitations": [
            "Quality compares predicted depth to the provided teacher cache on overlapping pixels.",
            "This is not a benchmark accuracy report.",
            "No millimeter-level, realtime, or mapping-readiness claim is made.",
        ],
    }
    write_json(output / "quality_report.json", report)
    write_jsonl(output / "per_frame_quality.jsonl", per_frame)
    return report


def write_point_cloud_ply(path: Path, surface: TSDFSurface) -> Path:
    """Write an ASCII PLY point cloud from TSDF surface points."""

    points = np.asarray(surface.points_world_m, dtype=np.float32)
    confidence = np.asarray(surface.confidence, dtype=np.float32)
    uncertainty = np.asarray(surface.uncertainty_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface.points_world_m: expected Nx3 points")
    if confidence.shape != (points.shape[0],) or uncertainty.shape != (points.shape[0],):
        raise ValueError("surface attributes must match point count")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment Atlas3R Phase 5G diagnostic point cloud, not a triangle mesh\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float confidence\n")
        handle.write("property float uncertainty_m\n")
        handle.write("end_header\n")
        for point, conf, sigma in zip(points, confidence, uncertainty, strict=True):
            handle.write(
                f"{float(point[0]):.7g} {float(point[1]):.7g} {float(point[2]):.7g} "
                f"{float(conf):.7g} {float(sigma):.7g}\n"
            )
    return path


def write_map_preview(
    path: Path,
    *,
    pose_mode: str,
    summary: Mapping[str, object],
    quality_report: Mapping[str, object],
    latency_report: Mapping[str, object],
) -> Path:
    surface_points = html.escape(str(summary.get("surface_point_count", "unknown")))
    observed_coverage = html.escape(str(summary.get("observed_coverage_estimate", "unknown")))
    rmse = cast(Mapping[str, object], quality_report.get("aggregate", {})).get("depth_rmse_m")
    latency = cast(Mapping[str, object], latency_report.get("fps_equivalent", {})).get(
        "model_inference"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html><head><meta charset="utf-8"><title>Atlas3R Phase 5G</title>',
                "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}"
                "code{background:#eee;padding:.1rem .25rem}</style></head><body>",
                f"<h1>Atlas3R Phase 5G {html.escape(pose_mode)}</h1>",
                "<p>Diagnostic streaming student map runtime output.</p>",
                "<ul>",
                f"<li>Surface points: {surface_points}</li>",
                f"<li>Observed coverage estimate: {observed_coverage}</li>",
                f"<li>Depth RMSE vs teacher: {html.escape(str(rmse))}</li>",
                f"<li>Inference FPS-equivalent: {html.escape(str(latency))}</li>",
                "</ul>",
                "<p>Outputs are diagnostic only: no realtime, mapping-readiness, benchmark "
                "accuracy, or millimeter-accuracy claim is made.</p>",
                "</body></html>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def point_set_comparison(
    student_points: npt.NDArray[np.float32],
    teacher_points: npt.NDArray[np.float32],
    *,
    max_points: int = 2048,
) -> dict[str, object] | None:
    if student_points.size == 0 or teacher_points.size == 0:
        return None
    student_sample = _sample_points(student_points, max_points=max_points)
    teacher_sample = _sample_points(teacher_points, max_points=max_points)
    student_to_teacher = _nearest_distances(student_sample, teacher_sample)
    teacher_to_student = _nearest_distances(teacher_sample, student_sample)
    return {
        "diagnostic_only": True,
        "accuracy_report": False,
        "student_point_count": int(student_points.shape[0]),
        "teacher_point_count": int(teacher_points.shape[0]),
        "sampled_student_point_count": int(student_sample.shape[0]),
        "sampled_teacher_point_count": int(teacher_sample.shape[0]),
        "student_to_teacher_mean_m": float(np.mean(student_to_teacher)),
        "student_to_teacher_p95_m": float(np.percentile(student_to_teacher, 95.0)),
        "teacher_to_student_mean_m": float(np.mean(teacher_to_student)),
        "teacher_to_student_p95_m": float(np.percentile(teacher_to_student, 95.0)),
        "symmetric_mean_m": float(
            (np.mean(student_to_teacher) + np.mean(teacher_to_student)) / 2.0
        ),
    }


def _quality_record(
    observation: DepthObservation,
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
    pose_mode: str,
    accumulator: dict[str, object],
) -> dict[str, object]:
    reference = references_by_frame_id.get(observation.frame_id)
    base = {
        "frame_id": observation.frame_id,
        "pose_mode": pose_mode,
        "confidence_mean": float(np.mean(observation.confidence)),
        "uncertainty_mean_m": float(np.mean(observation.depth_sigma_m)),
        "reference_available": reference is not None,
    }
    if reference is None:
        return {**base, "overlap_valid_pixel_count": 0}
    valid = (
        reference.valid_mask
        & np.isfinite(reference.depth_m)
        & np.isfinite(observation.depth_m)
        & (reference.depth_m > 0.0)
        & (observation.depth_m > 0.0)
        & (observation.confidence > 0.0)
    )
    abs_error = np.abs(
        observation.depth_m.astype(np.float64, copy=False)[valid]
        - reference.depth_m.astype(np.float64, copy=False)[valid]
    )
    reference_depth = reference.depth_m.astype(np.float64, copy=False)[valid]
    _accumulate(accumulator, abs_error, reference_depth)
    if abs_error.size:
        accumulator["compared_frames"] = cast(int, accumulator["compared_frames"]) + 1
    pose_center_error_m = None
    if pose_mode != "oracle":
        pose_center_error_m = float(
            np.linalg.norm(
                observation.pose.T_world_camera[:3, 3].astype(np.float64)
                - reference.T_world_camera[:3, 3].astype(np.float64)
            )
        )
    return {
        **base,
        "teacher_name": reference.teacher_name,
        "teacher_measured_geometry": reference.measured_geometry,
        "overlap_valid_pixel_count": int(abs_error.size),
        "valid_pixel_overlap_percent": float(np.count_nonzero(valid) / valid.size * 100.0),
        "depth_rmse_m": _rmse(abs_error),
        "depth_mae_m": _mean_or_none(abs_error),
        "depth_absrel": _mean_or_none(abs_error / np.maximum(reference_depth, 1e-12)),
        "within_1mm_percent": _within(abs_error, 0.001),
        "within_5mm_percent": _within(abs_error, 0.005),
        "within_1cm_percent": _within(abs_error, 0.01),
        "within_5cm_percent": _within(abs_error, 0.05),
        "within_10cm_percent": _within(abs_error, 0.10),
        "pose_center_error_m": pose_center_error_m,
    }


def _new_accumulator() -> dict[str, object]:
    return {
        "count": 0,
        "compared_frames": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "abs_rel_sum": 0.0,
        "within": {0.001: 0, 0.005: 0, 0.01: 0, 0.05: 0, 0.10: 0},
    }


def _accumulate(
    accumulator: dict[str, object],
    abs_error: npt.NDArray[np.float64],
    reference_depth: npt.NDArray[np.float64],
) -> None:
    count = cast(int, accumulator["count"]) + int(abs_error.size)
    accumulator["count"] = count
    accumulator["abs_error_sum"] = cast(float, accumulator["abs_error_sum"]) + float(
        np.sum(abs_error)
    )
    accumulator["squared_error_sum"] = cast(float, accumulator["squared_error_sum"]) + float(
        np.sum(np.square(abs_error))
    )
    accumulator["abs_rel_sum"] = cast(float, accumulator["abs_rel_sum"]) + float(
        np.sum(abs_error / np.maximum(reference_depth, 1e-12))
    )
    within = cast(dict[float, int], accumulator["within"])
    for threshold in tuple(within):
        within[threshold] += int(np.count_nonzero(abs_error <= threshold))


def _aggregate_record(accumulator: Mapping[str, object]) -> dict[str, object]:
    count = cast(int, accumulator["count"])
    within = cast(dict[float, int], accumulator["within"])
    squared_error_sum = cast(float, accumulator["squared_error_sum"])
    abs_error_sum = cast(float, accumulator["abs_error_sum"])
    abs_rel_sum = cast(float, accumulator["abs_rel_sum"])
    return {
        "overlap_valid_pixel_count": count,
        "depth_rmse_m": (float(np.sqrt(squared_error_sum / count)) if count else None),
        "depth_mae_m": abs_error_sum / count if count else None,
        "depth_absrel": abs_rel_sum / count if count else None,
        "within_1mm_percent": _aggregate_within(within, 0.001, count),
        "within_5mm_percent": _aggregate_within(within, 0.005, count),
        "within_1cm_percent": _aggregate_within(within, 0.01, count),
        "within_5cm_percent": _aggregate_within(within, 0.05, count),
        "within_10cm_percent": _aggregate_within(within, 0.10, count),
    }


def _latency_stats(samples_ns: list[int]) -> dict[str, object]:
    if not samples_ns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(values_ms.size),
        "mean_ms": float(np.mean(values_ms)),
        "p50_ms": float(np.percentile(values_ms, 50.0)),
        "p95_ms": float(np.percentile(values_ms, 95.0)),
        "max_ms": float(np.max(values_ms)),
    }


def _fps_equivalent(samples_ns: list[int]) -> float | None:
    if not samples_ns:
        return None
    mean_seconds = float(np.mean(np.asarray(samples_ns, dtype=np.float64))) / 1_000_000_000.0
    if mean_seconds <= 0.0:
        return None
    return float(1.0 / mean_seconds)


def _sample_points(
    points: npt.NDArray[np.float32],
    *,
    max_points: int,
) -> npt.NDArray[np.float32]:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.floor(
        (np.arange(max_points, dtype=np.float64) + 0.5) * points.shape[0] / max_points
    ).astype(np.int64)
    return cast(npt.NDArray[np.float32], points[indices].astype(np.float32, copy=False))


def _nearest_distances(
    query: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    distances: list[npt.NDArray[np.float64]] = []
    target64 = target.astype(np.float64, copy=False)
    for start in range(0, query.shape[0], 256):
        chunk = query[start : start + 256].astype(np.float64, copy=False)
        delta = chunk[:, None, :] - target64[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return cast(npt.NDArray[np.float64], np.concatenate(distances, axis=0))


def _rmse(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _percentile_or_none(values: npt.NDArray[np.float64], percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _within(values: npt.NDArray[np.float64], threshold: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.count_nonzero(values <= threshold) / values.size * 100.0)


def _aggregate_within(within: Mapping[float, int], threshold: float, count: int) -> float | None:
    if count == 0:
        return None
    return float(within[threshold] / count * 100.0)


__all__ = [
    "DIAGNOSTIC_TRUTH_FLAGS",
    "LatencyRecorder",
    "TeacherReferenceFrame",
    "point_set_comparison",
    "write_json",
    "write_jsonl",
    "write_latency_report",
    "write_map_preview",
    "write_observation_summaries",
    "write_pose_quality_reports",
    "write_point_cloud_ply",
    "write_quality_reports",
]
