"""Focused VGGT teacher-signal evaluation against measured clip caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.pose.transforms import compose_transforms, invert_transform
from atlas3r.teachers._diagnostic_helpers import (
    aggregate_within_percent,
    mean_metric,
    mean_or_none,
    rmse,
    within_percent,
    write_json,
    write_jsonl,
)
from atlas3r.teachers.signals import (
    load_teacher_signal_manifest,
    read_teacher_signal_payload_from_entry,
    teacher_signal_manifest_path_from_input,
    validate_payload_matches_signal_entry,
    validate_teacher_signal_payload,
)


@dataclass(frozen=True)
class VGGTEvaluationConfig:
    clip_cache: Path
    teacher_cache: Path
    output: Path
    max_clips: int = 64


def evaluate_vggt_teacher_signals(config: VGGTEvaluationConfig) -> dict[str, object]:
    """Evaluate VGGT depth, pose, and optional pointmaps against source clip data."""

    if config.max_clips <= 0:
        raise ValueError("max_clips: must be positive")
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    teacher_manifest_path = teacher_signal_manifest_path_from_input(config.teacher_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    teacher_manifest = load_teacher_signal_manifest(
        teacher_manifest_path,
        clip_cache=clip_manifest_path,
        validate_payloads=False,
    )
    signal_entries = _entries(teacher_manifest, "signals")
    selected = signal_entries[: min(config.max_clips, len(signal_entries))]
    if not selected:
        raise ValueError(f"{teacher_manifest_path}: no VGGT teacher signals selected")

    clip_entries = _entries(clip_manifest, "clips")
    accumulator = _new_accumulator()
    per_clip: list[dict[str, object]] = []
    for index, signal_entry in enumerate(selected):
        source_clip_id = _int_field(signal_entry, "source_clip_id")
        clip_payload = read_clip_payload_from_entry(
            clip_manifest_path.parent,
            clip_entries[source_clip_id],
        )
        _validate_clip_payload_for_manifest(clip_payload, clip_manifest)
        teacher_payload = read_teacher_signal_payload_from_entry(
            teacher_manifest_path.parent,
            signal_entry,
        )
        _validate_signal_payload_for_manifest(teacher_payload, teacher_manifest)
        validate_payload_matches_signal_entry(teacher_payload, signal_entry, index=index)
        record = _per_clip_metrics(
            signal_index=index,
            signal_entry=signal_entry,
            clip_payload=clip_payload,
            teacher_payload=teacher_payload,
            accumulator=accumulator,
        )
        per_clip.append(record)

    summary = _summary(
        teacher_manifest=teacher_manifest,
        selected_count=len(selected),
        accumulator=accumulator,
        per_clip=per_clip,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    summary_path = config.output / "summary.json"
    per_clip_path = config.output / "per_clip_metrics.jsonl"
    report_path = config.output / "report.md"
    write_json(summary_path, summary)
    write_jsonl(per_clip_path, per_clip)
    report_path.write_text(_markdown_report(summary), encoding="utf-8", newline="\n")
    return {
        "format_name": "atlas3r_vggt_teacher_evaluation_result",
        "summary_path": str(summary_path),
        "per_clip_metrics_path": str(per_clip_path),
        "markdown_report_path": str(report_path),
        "summary": summary,
    }


def _per_clip_metrics(
    *,
    signal_index: int,
    signal_entry: Mapping[str, object],
    clip_payload: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    accumulator: dict[str, Any],
) -> dict[str, object]:
    clip_depth = np.asarray(clip_payload["depth_m"], dtype=np.float64)
    teacher_depth = np.asarray(teacher_payload["depth_m"], dtype=np.float64)
    overlap = np.asarray(clip_payload["valid_depth_mask"], dtype=np.bool_) & np.asarray(
        teacher_payload["valid_mask"], dtype=np.bool_
    )
    abs_error = np.abs(teacher_depth[overlap] - clip_depth[overlap])
    abs_rel = abs_error / np.maximum(clip_depth[overlap], 1e-12)
    _accumulate_depth(accumulator, abs_error, abs_rel)

    pose = _pose_metrics(clip_payload, teacher_payload)
    pointmap = _pointmap_metrics(clip_payload, teacher_payload, overlap)
    confidence = np.asarray(teacher_payload["confidence"], dtype=np.float64)
    confidence_valid = confidence[np.asarray(teacher_payload["valid_mask"], dtype=np.bool_)]
    record = {
        "signal_index": signal_index,
        "source_clip_id": _int_field(signal_entry, "source_clip_id"),
        "frame_ids": list(cast(list[int], signal_entry["frame_ids"])),
        "overlap_valid_pixel_count": int(abs_error.size),
        "valid_pixel_overlap_percent": float(np.count_nonzero(overlap) / overlap.size * 100.0),
        "teacher_valid_pixel_percent": float(
            np.count_nonzero(teacher_payload["valid_mask"]) / overlap.size * 100.0
        ),
        "depth_rmse_m": rmse(abs_error),
        "depth_mae_m": mean_or_none(abs_error),
        "depth_absrel": mean_or_none(abs_rel),
        "within_1mm_percent": within_percent(abs_error, 0.001),
        "within_5mm_percent": within_percent(abs_error, 0.005),
        "within_1cm_percent": within_percent(abs_error, 0.01),
        "within_5cm_percent": within_percent(abs_error, 0.05),
        "within_10cm_percent": within_percent(abs_error, 0.10),
        "confidence_mean_valid": mean_or_none(confidence_valid),
        **pose,
        **pointmap,
    }
    _accumulate_optional(accumulator, record)
    return record


def _pose_metrics(
    clip_payload: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
) -> dict[str, object]:
    source_T = np.asarray(clip_payload["T_world_camera"], dtype=np.float64)
    teacher_T = np.asarray(teacher_payload["T_world_camera"], dtype=np.float64)
    center_delta = np.linalg.norm(teacher_T[:, :3, 3] - source_T[:, :3, 3], axis=1)
    rpe_translation: list[float] = []
    rpe_rotation_deg: list[float] = []
    for index in range(source_T.shape[0] - 1):
        source_rel = compose_transforms(invert_transform(source_T[index]), source_T[index + 1])
        teacher_rel = compose_transforms(invert_transform(teacher_T[index]), teacher_T[index + 1])
        delta = compose_transforms(invert_transform(source_rel), teacher_rel)
        rpe_translation.append(float(np.linalg.norm(delta[:3, 3])))
        rpe_rotation_deg.append(_rotation_angle_deg(delta[:3, :3]))
    return {
        "ate_center_error_m_mean": mean_or_none(center_delta),
        "ate_center_error_m_median": _median_or_none(center_delta),
        "ate_center_error_m_p95": _percentile_or_none(center_delta, 95.0),
        "rpe_translation_m_mean": mean_or_none(np.asarray(rpe_translation, dtype=np.float64)),
        "rpe_rotation_deg_mean": mean_or_none(np.asarray(rpe_rotation_deg, dtype=np.float64)),
    }


def _pointmap_metrics(
    clip_payload: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    overlap: npt.NDArray[np.bool_],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "pointmap_world_rmse_m": None,
        "pointmap_depth_consistency_mae_m": None,
    }
    if "pointmap_world_m" in clip_payload and "pointmap_world_m" in teacher_payload:
        source = np.asarray(clip_payload["pointmap_world_m"], dtype=np.float64)
        teacher = np.asarray(teacher_payload["pointmap_world_m"], dtype=np.float64)
        diff = np.linalg.norm(teacher[overlap] - source[overlap], axis=1)
        metrics["pointmap_world_rmse_m"] = rmse(diff)
    pointmap_camera = teacher_payload.get("pointmap_camera_m")
    if pointmap_camera is not None:
        z = np.asarray(pointmap_camera, dtype=np.float64)[..., 2]
        depth = np.asarray(teacher_payload["depth_m"], dtype=np.float64)
        diff = np.abs(z[overlap] - depth[overlap])
        metrics["pointmap_depth_consistency_mae_m"] = mean_or_none(diff)
    return metrics


def _summary(
    *,
    teacher_manifest: Mapping[str, object],
    selected_count: int,
    accumulator: dict[str, Any],
    per_clip: list[dict[str, object]],
) -> dict[str, object]:
    count = int(accumulator["count"])
    within = cast(dict[float, int], accumulator["within"])
    return {
        "format_name": "atlas3r_vggt_teacher_evaluation_summary",
        "format_version": 1,
        "teacher_name": str(teacher_manifest["teacher_name"]),
        "teacher_source_type": str(teacher_manifest["teacher_source_type"]),
        "source_dataset_name": str(teacher_manifest["source_dataset_name"]),
        "source_sequence_name": str(teacher_manifest["source_sequence_name"]),
        "split": str(teacher_manifest["split"]),
        "selected_signal_count": selected_count,
        "overlap_valid_pixel_count": count,
        "aggregate": {
            "depth_rmse_m": (
                float(np.sqrt(float(accumulator["squared_error_sum"]) / count)) if count else None
            ),
            "depth_mae_m": float(accumulator["abs_error_sum"] / count) if count else None,
            "depth_absrel": float(accumulator["abs_rel_sum"] / count) if count else None,
            "within_1mm_percent": aggregate_within_percent(within, 0.001, count),
            "within_5mm_percent": aggregate_within_percent(within, 0.005, count),
            "within_1cm_percent": aggregate_within_percent(within, 0.01, count),
            "within_5cm_percent": aggregate_within_percent(within, 0.05, count),
            "within_10cm_percent": aggregate_within_percent(within, 0.10, count),
            "valid_pixel_overlap_percent_mean": mean_metric(
                per_clip, "valid_pixel_overlap_percent"
            ),
            "teacher_valid_pixel_percent_mean": mean_metric(
                per_clip, "teacher_valid_pixel_percent"
            ),
            "confidence_mean_valid": mean_metric(per_clip, "confidence_mean_valid"),
            "ate_center_error_m_mean": mean_metric(per_clip, "ate_center_error_m_mean"),
            "rpe_translation_m_mean": mean_metric(per_clip, "rpe_translation_m_mean"),
            "rpe_rotation_deg_mean": mean_metric(per_clip, "rpe_rotation_deg_mean"),
            "pointmap_world_rmse_m": mean_metric(per_clip, "pointmap_world_rmse_m"),
            "pointmap_depth_consistency_mae_m": mean_metric(
                per_clip, "pointmap_depth_consistency_mae_m"
            ),
        },
        "quality_gate_inputs": {
            "compare_depth_rmse_to_phase5g1_student": True,
            "compare_rpe_to_phase5g1_student_odometry": True,
            "train_student_only_if_external_teacher_improves_a_gate": True,
        },
        "truth_boundary": dict(cast(dict[str, object], teacher_manifest["truth_boundary"])),
        "source_metadata": dict(cast(dict[str, object], teacher_manifest["source_metadata"])),
        "known_limitations": [
            "This is a diagnostic comparison against local source clip depth/pose.",
            "It is not a benchmark accuracy report or performance report.",
            "Pseudo-label and aligned pseudo-label geometry remains non-measured.",
        ],
    }


def _markdown_report(summary: Mapping[str, object]) -> str:
    aggregate = cast(Mapping[str, object], summary["aggregate"])
    truth = cast(Mapping[str, object], summary["truth_boundary"])
    lines = [
        "# Phase 5H VGGT Teacher Evaluation",
        "",
        f"Teacher: `{summary['teacher_name']}`",
        f"Sequence: `{summary['source_sequence_name']}` / split `{summary['split']}`",
        f"Signals evaluated: {summary['selected_signal_count']}",
        "",
        "## Aggregate Metrics",
        "",
        f"- Depth RMSE: {_fmt(aggregate['depth_rmse_m'])} m",
        f"- Depth MAE: {_fmt(aggregate['depth_mae_m'])} m",
        f"- Depth AbsRel: {_fmt(aggregate['depth_absrel'])}",
        f"- ATE center mean: {_fmt(aggregate['ate_center_error_m_mean'])} m",
        f"- RPE translation mean: {_fmt(aggregate['rpe_translation_m_mean'])} m",
        f"- RPE rotation mean: {_fmt(aggregate['rpe_rotation_deg_mean'])} deg",
        f"- Pointmap world RMSE: {_fmt(aggregate['pointmap_world_rmse_m'])} m",
        "",
        "## Truth Boundary",
        "",
        f"- measured_geometry: `{truth.get('measured_geometry')}`",
        f"- pseudo_label: `{truth.get('pseudo_label')}`",
        f"- accuracy_report: `{truth.get('accuracy_report')}`",
        "",
        "Conclusion: diagnostic-only teacher quality evidence.",
        "",
    ]
    return "\n".join(lines)


def _new_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "abs_rel_sum": 0.0,
        "within": {0.001: 0, 0.005: 0, 0.01: 0, 0.05: 0, 0.10: 0},
    }


def _accumulate_depth(
    accumulator: dict[str, Any],
    abs_error: npt.NDArray[np.float64],
    abs_rel: npt.NDArray[np.float64],
) -> None:
    accumulator["count"] = int(accumulator["count"]) + int(abs_error.size)
    accumulator["abs_error_sum"] = float(accumulator["abs_error_sum"]) + float(np.sum(abs_error))
    accumulator["squared_error_sum"] = float(accumulator["squared_error_sum"]) + float(
        np.sum(np.square(abs_error))
    )
    accumulator["abs_rel_sum"] = float(accumulator["abs_rel_sum"]) + float(np.sum(abs_rel))
    within = cast(dict[float, int], accumulator["within"])
    for threshold in tuple(within):
        within[threshold] += int(np.count_nonzero(abs_error <= threshold))


def _accumulate_optional(_accumulator: dict[str, Any], _record: Mapping[str, object]) -> None:
    return None


def _rotation_angle_deg(R: npt.NDArray[np.float64]) -> float:
    trace = float(np.trace(R))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _median_or_none(values: npt.NDArray[np.float64]) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def _percentile_or_none(values: npt.NDArray[np.float64], q: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, q))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.6f}"


def _validate_clip_payload_for_manifest(
    payload: dict[str, Any],
    manifest: Mapping[str, object],
) -> None:
    validate_clip_payload(
        payload,
        clip_length=_int_field(manifest, "clip_length"),
        height=_int_field(manifest, "image_height"),
        width=_int_field(manifest, "image_width"),
    )


def _validate_signal_payload_for_manifest(
    payload: dict[str, Any],
    manifest: Mapping[str, object],
) -> None:
    validate_teacher_signal_payload(
        payload,
        clip_length=_int_field(manifest, "clip_length"),
        height=_int_field(manifest, "image_height"),
        width=_int_field(manifest, "image_width"),
    )


def _entries(manifest: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    entries: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.{key}[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], item))
    return entries


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "VGGTEvaluationConfig",
    "evaluate_vggt_teacher_signals",
]
