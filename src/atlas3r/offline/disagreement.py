"""Teacher-depth disagreement and diagnostic consensus preview artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import write_json

_EPS = 1e-6
_ABS_HIGH_THRESHOLD_M = 0.5
_REL_HIGH_THRESHOLD = 0.25


@dataclass(frozen=True)
class DisagreementResult:
    status: str
    json_path: str
    maps_npz_path: str | None
    consensus_npz_path: str | None
    summary: dict[str, object] = field(default_factory=dict)
    frame_summaries: tuple[dict[str, object], ...] = ()
    valid_overlap_count: int = 0
    consensus_status: str = "unavailable"
    consensus_depth_records: tuple[dict[str, object], ...] = ()
    consensus_depth_arrays: dict[str, NDArray[np.float32]] = field(default_factory=dict)


def write_teacher_disagreement(
    run_dir: str | Path, *, proposal_cache: ProposalCacheResult
) -> DisagreementResult:
    root = Path(run_dir)
    missing = _missing_witnesses(proposal_cache)
    if missing:
        result = _write_insufficient(root, missing)
        return result
    pairs = _paired_records(proposal_cache)
    if not pairs:
        return _write_insufficient(root, ["matching_frame_overlap"])
    frame_ids: list[int] = []
    abs_maps: list[NDArray[np.float32]] = []
    rel_maps: list[NDArray[np.float32]] = []
    valid_maps: list[NDArray[np.float32]] = []
    high_maps: list[NDArray[np.float32]] = []
    consensus_depths: list[NDArray[np.float32]] = []
    consensus_confidences: list[NDArray[np.float32]] = []
    source_masks: list[NDArray[np.int16]] = []
    frame_summaries: list[dict[str, object]] = []
    consensus_records: list[dict[str, object]] = []
    consensus_arrays: dict[str, NDArray[np.float32]] = {}
    total_overlap = 0
    total_pixels = 0
    all_abs_values: list[NDArray[np.float32]] = []
    all_rel_values: list[NDArray[np.float32]] = []
    all_log_values: list[NDArray[np.float32]] = []
    all_high_values: list[NDArray[np.float32]] = []
    for index, (vggt, depth_pro) in enumerate(pairs):
        frame_id = int(str(vggt["frame_id"]))
        vggt_depth, vggt_valid = _depth_and_valid(vggt, proposal_cache)
        depth_pro_depth, depth_pro_valid = _depth_and_valid(depth_pro, proposal_cache)
        depth_pro_depth = depth_pro_depth.astype(np.float32)
        depth_pro_valid = depth_pro_valid.astype(np.bool_)
        target_shape = (int(depth_pro_depth.shape[0]), int(depth_pro_depth.shape[1]))
        vggt_depth = _resize_nearest(vggt_depth, target_shape)
        vggt_valid = _resize_nearest_bool(vggt_valid, target_shape)
        overlap = (
            vggt_valid
            & depth_pro_valid
            & np.isfinite(vggt_depth)
            & np.isfinite(depth_pro_depth)
            & (vggt_depth > 0.0)
            & (depth_pro_depth > 0.0)
        )
        abs_diff = np.where(overlap, np.abs(vggt_depth - depth_pro_depth), 0.0).astype(np.float32)
        rel_diff = np.where(overlap, abs_diff / np.maximum(depth_pro_depth, _EPS), 0.0).astype(
            np.float32
        )
        log_diff = np.where(
            overlap,
            np.abs(
                np.log(np.maximum(vggt_depth, _EPS)) - np.log(np.maximum(depth_pro_depth, _EPS))
            ),
            0.0,
        ).astype(np.float32)
        high = overlap & ((abs_diff > _ABS_HIGH_THRESHOLD_M) | (rel_diff > _REL_HIGH_THRESHOLD))
        source_mask, consensus_depth, consensus_confidence = _consensus_preview(
            vggt_depth, vggt_valid, depth_pro_depth, depth_pro_valid, high
        )
        frame_summary = _frame_summary(
            frame_id=frame_id,
            overlap=overlap,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            log_diff=log_diff,
            high=high,
        )
        frame_summaries.append(frame_summary)
        total_overlap += int(overlap.sum())
        total_pixels += int(overlap.size)
        if np.any(overlap):
            all_abs_values.append(abs_diff[overlap])
            all_rel_values.append(rel_diff[overlap])
            all_log_values.append(log_diff[overlap])
            all_high_values.append(high[overlap].astype(np.float32))
        frame_ids.append(frame_id)
        abs_maps.append(abs_diff)
        rel_maps.append(rel_diff)
        valid_maps.append(overlap.astype(np.float32))
        high_maps.append(high.astype(np.float32))
        consensus_depths.append(consensus_depth)
        consensus_confidences.append(consensus_confidence)
        source_masks.append(source_mask)
        _append_consensus_record(
            index,
            frame_id=frame_id,
            keyframe_index=int(str(vggt["keyframe_index"])),
            consensus_depth=consensus_depth,
            consensus_confidence=consensus_confidence,
            source_mask=source_mask,
            consensus_records=consensus_records,
            consensus_arrays=consensus_arrays,
        )
    summary = _summary(
        total_overlap, total_pixels, all_abs_values, all_rel_values, all_log_values, all_high_values
    )
    status = "available" if total_overlap > 0 else "no_valid_overlap"
    maps_path = root / "diagnostics" / "disagreement_maps.npz"
    consensus_path = root / "diagnostics" / "consensus_preview.npz"
    _write_maps_npz(maps_path, frame_ids, abs_maps, rel_maps, valid_maps, high_maps)
    _write_consensus_npz(
        consensus_path, frame_ids, consensus_depths, consensus_confidences, source_masks, status
    )
    payload = {
        "status": status,
        "diagnostic_only": True,
        "optimized_consensus": False,
        "maps_path": "diagnostics/disagreement_maps.npz",
        "consensus_preview_path": "diagnostics/consensus_preview.npz",
        "summary": summary,
        "frames": frame_summaries,
        "missing_witnesses": [],
        "truth_boundary": {
            "label_type": "teacher_pseudo",
            "measured_geometry": False,
            "observed_only": True,
            "predicted_completion": False,
            "hidden_geometry_measured": False,
            "accuracy_report": False,
            "realtime_claim": False,
            "usable_for_training": False,
        },
    }
    write_json(root / "diagnostics" / "teacher_disagreement.json", payload)
    return DisagreementResult(
        status=status,
        json_path="diagnostics/teacher_disagreement.json",
        maps_npz_path="diagnostics/disagreement_maps.npz",
        consensus_npz_path="diagnostics/consensus_preview.npz",
        summary=summary,
        frame_summaries=tuple(frame_summaries),
        valid_overlap_count=total_overlap,
        consensus_status="diagnostic" if total_overlap > 0 else "unavailable",
        consensus_depth_records=tuple(consensus_records),
        consensus_depth_arrays=consensus_arrays,
    )


def _missing_witnesses(proposal_cache: ProposalCacheResult) -> list[str]:
    missing = []
    if not proposal_cache.vggt_depth_records:
        missing.append("vggt")
    if not proposal_cache.depth_pro_depth_records:
        missing.append("depth_pro")
    return missing


def _paired_records(
    proposal_cache: ProposalCacheResult,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    depth_pro_by_frame = {
        int(str(record["frame_id"])): record for record in proposal_cache.depth_pro_depth_records
    }
    pairs = []
    for vggt in proposal_cache.vggt_depth_records:
        frame_id = int(str(vggt["frame_id"]))
        if frame_id in depth_pro_by_frame:
            pairs.append((vggt, depth_pro_by_frame[frame_id]))
    return pairs


def _depth_and_valid(
    record: dict[str, object], proposal_cache: ProposalCacheResult
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    depth = proposal_cache.depth_arrays[str(record["depth_key"])].astype(np.float32)
    valid = proposal_cache.depth_arrays[str(record["valid_mask_key"])] > 0.0
    return depth, valid


def _frame_summary(
    *,
    frame_id: int,
    overlap: NDArray[np.bool_],
    abs_diff: NDArray[np.float32],
    rel_diff: NDArray[np.float32],
    log_diff: NDArray[np.float32],
    high: NDArray[np.bool_],
) -> dict[str, object]:
    values_abs = abs_diff[overlap]
    values_rel = rel_diff[overlap]
    values_log = log_diff[overlap]
    overlap_count = int(overlap.sum())
    return {
        "frame_id": frame_id,
        "overlap_pixel_count": overlap_count,
        "valid_overlap_ratio": float(overlap_count / overlap.size) if overlap.size else 0.0,
        "abs_depth_diff_mean_m": _mean(values_abs),
        "abs_depth_diff_p50_m": _percentile(values_abs, 50),
        "abs_depth_diff_p95_m": _percentile(values_abs, 95),
        "rel_depth_diff_mean": _mean(values_rel),
        "rel_depth_diff_p95": _percentile(values_rel, 95),
        "log_depth_diff_mean": _mean(values_log),
        "disagreement_high_ratio": float(high[overlap].mean()) if overlap_count else 0.0,
    }


def _summary(
    total_overlap: int,
    total_pixels: int,
    abs_values: list[NDArray[np.float32]],
    rel_values: list[NDArray[np.float32]],
    log_values: list[NDArray[np.float32]],
    high_values: list[NDArray[np.float32]],
) -> dict[str, object]:
    abs_all = np.concatenate(abs_values) if abs_values else np.zeros((0,), dtype=np.float32)
    rel_all = np.concatenate(rel_values) if rel_values else np.zeros((0,), dtype=np.float32)
    log_all = np.concatenate(log_values) if log_values else np.zeros((0,), dtype=np.float32)
    high_all = np.concatenate(high_values) if high_values else np.zeros((0,), dtype=np.float32)
    return {
        "overlap_pixel_count": total_overlap,
        "valid_overlap_ratio": float(total_overlap / total_pixels) if total_pixels else 0.0,
        "abs_depth_diff_mean_m": _mean(abs_all),
        "abs_depth_diff_p50_m": _percentile(abs_all, 50),
        "abs_depth_diff_p95_m": _percentile(abs_all, 95),
        "rel_depth_diff_mean": _mean(rel_all),
        "rel_depth_diff_p95": _percentile(rel_all, 95),
        "log_depth_diff_mean": _mean(log_all),
        "disagreement_high_ratio": float(high_all.mean()) if high_all.size else 0.0,
    }


def _consensus_preview(
    vggt_depth: NDArray[np.float32],
    vggt_valid: NDArray[np.bool_],
    depth_pro_depth: NDArray[np.float32],
    depth_pro_valid: NDArray[np.bool_],
    high: NDArray[np.bool_],
) -> tuple[NDArray[np.int16], NDArray[np.float32], NDArray[np.float32]]:
    overlap = vggt_valid & depth_pro_valid & (vggt_depth > 0.0) & (depth_pro_depth > 0.0)
    agree = overlap & ~high
    disagree = overlap & high
    vggt_only = vggt_valid & ~depth_pro_valid
    depth_pro_only = depth_pro_valid & ~vggt_valid
    source_mask = np.zeros(vggt_depth.shape, dtype=np.int16)
    source_mask[vggt_only] = 1
    source_mask[depth_pro_only] = 2
    source_mask[agree] = 3
    source_mask[disagree] = 4
    consensus_depth = np.zeros(vggt_depth.shape, dtype=np.float32)
    consensus_depth[vggt_only] = vggt_depth[vggt_only]
    consensus_depth[depth_pro_only] = depth_pro_depth[depth_pro_only]
    consensus_depth[overlap] = 0.5 * (vggt_depth[overlap] + depth_pro_depth[overlap])
    confidence = np.zeros(vggt_depth.shape, dtype=np.float32)
    confidence[vggt_only | depth_pro_only] = 0.4
    confidence[agree] = 0.85
    confidence[disagree] = 0.2
    return source_mask, consensus_depth, confidence


def _append_consensus_record(
    index: int,
    *,
    frame_id: int,
    keyframe_index: int,
    consensus_depth: NDArray[np.float32],
    consensus_confidence: NDArray[np.float32],
    source_mask: NDArray[np.int16],
    consensus_records: list[dict[str, object]],
    consensus_arrays: dict[str, NDArray[np.float32]],
) -> None:
    depth_key = f"consensus_preview_depth_{index:06d}"
    sigma_key = f"consensus_preview_sigma_{index:06d}"
    confidence_key = f"consensus_preview_confidence_{index:06d}"
    valid_key = f"consensus_preview_valid_mask_{index:06d}"
    source_key = f"consensus_preview_source_mask_{index:06d}"
    valid = source_mask > 0
    consensus_arrays[depth_key] = consensus_depth.astype(np.float32)
    consensus_arrays[sigma_key] = np.where(valid, 1.0 - consensus_confidence, 0.0).astype(
        np.float32
    )
    consensus_arrays[confidence_key] = consensus_confidence.astype(np.float32)
    consensus_arrays[valid_key] = valid.astype(np.float32)
    consensus_arrays[source_key] = source_mask.astype(np.float32)
    consensus_records.append(
        {
            "teacher_name": "consensus_preview",
            "frame_id": frame_id,
            "keyframe_index": keyframe_index,
            "depth_key": depth_key,
            "depth_sigma_key": sigma_key,
            "confidence_key": confidence_key,
            "valid_mask_key": valid_key,
            "source_mask_key": source_key,
            "depth_shape": list(consensus_depth.shape),
            "depth_source": "diagnostic_vggt_depth_pro_consensus",
            "metric_scale_source": "diagnostic_unoptimized_teacher_consensus",
            "coordinate_convention": "x_right_y_down_z_forward",
            "measured_geometry": False,
            "truth_boundary": {
                "label_type": "teacher_pseudo",
                "metric_scale_source": "diagnostic_unoptimized_teacher_consensus",
                "measured_geometry": False,
                "observed_only": True,
                "predicted_completion": False,
                "hidden_geometry_measured": False,
                "accuracy_report": False,
                "realtime_claim": False,
                "usable_for_training": False,
            },
            "diagnostic_only": True,
            "optimized_consensus": False,
        }
    )


def _write_maps_npz(
    path: Path,
    frame_ids: list[int],
    abs_maps: list[NDArray[np.float32]],
    rel_maps: list[NDArray[np.float32]],
    valid_maps: list[NDArray[np.float32]],
    high_maps: list[NDArray[np.float32]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        abs_depth_diff_m=np.stack(abs_maps).astype(np.float32),
        rel_depth_diff=np.stack(rel_maps).astype(np.float32),
        valid_overlap_mask=np.stack(valid_maps).astype(np.float32),
        high_disagreement_mask=np.stack(high_maps).astype(np.float32),
    )


def _write_consensus_npz(
    path: Path,
    frame_ids: list[int],
    depths: list[NDArray[np.float32]],
    confidences: list[NDArray[np.float32]],
    source_masks: list[NDArray[np.int16]],
    status: str,
) -> None:
    metadata = {
        "status": "diagnostic" if status == "available" else "unavailable",
        "diagnostic_only": True,
        "optimized_consensus": False,
    }
    np.savez_compressed(
        path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        consensus_depth_m=np.stack(depths).astype(np.float32),
        consensus_confidence=np.stack(confidences).astype(np.float32),
        source_mask=np.stack(source_masks).astype(np.int16),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _write_insufficient(root: Path, missing: list[str]) -> DisagreementResult:
    maps_path = root / "diagnostics" / "disagreement_maps.npz"
    consensus_path = root / "diagnostics" / "consensus_preview.npz"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        maps_path,
        frame_ids=np.zeros((0,), dtype=np.int32),
        abs_depth_diff_m=np.zeros((0, 0, 0), dtype=np.float32),
        rel_depth_diff=np.zeros((0, 0, 0), dtype=np.float32),
        valid_overlap_mask=np.zeros((0, 0, 0), dtype=np.float32),
        high_disagreement_mask=np.zeros((0, 0, 0), dtype=np.float32),
    )
    np.savez_compressed(
        consensus_path,
        frame_ids=np.zeros((0,), dtype=np.int32),
        consensus_depth_m=np.zeros((0, 0, 0), dtype=np.float32),
        consensus_confidence=np.zeros((0, 0, 0), dtype=np.float32),
        source_mask=np.zeros((0, 0, 0), dtype=np.int16),
        metadata_json=json.dumps(
            {
                "status": "unavailable",
                "diagnostic_only": True,
                "optimized_consensus": False,
            },
            sort_keys=True,
        ),
    )
    summary: dict[str, object] = {
        "overlap_pixel_count": 0,
        "valid_overlap_ratio": 0.0,
    }
    payload = {
        "status": "insufficient_witnesses",
        "diagnostic_only": True,
        "optimized_consensus": False,
        "missing_witnesses": missing,
        "summary": summary,
        "frames": [],
        "maps_path": "diagnostics/disagreement_maps.npz",
        "consensus_preview_path": "diagnostics/consensus_preview.npz",
    }
    write_json(root / "diagnostics" / "teacher_disagreement.json", payload)
    return DisagreementResult(
        status="insufficient_witnesses",
        json_path="diagnostics/teacher_disagreement.json",
        maps_npz_path="diagnostics/disagreement_maps.npz",
        consensus_npz_path="diagnostics/consensus_preview.npz",
        summary=summary,
        valid_overlap_count=0,
    )


def _resize_nearest(array: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float32]:
    if array.shape == shape:
        return array.astype(np.float32)
    y_idx = np.clip(
        np.round(np.linspace(0, array.shape[0] - 1, shape[0])).astype(np.int32),
        0,
        array.shape[0] - 1,
    )
    x_idx = np.clip(
        np.round(np.linspace(0, array.shape[1] - 1, shape[1])).astype(np.int32),
        0,
        array.shape[1] - 1,
    )
    return cast(NDArray[np.float32], array[y_idx[:, None], x_idx[None, :]].astype(np.float32))


def _resize_nearest_bool(array: NDArray[np.bool_], shape: tuple[int, int]) -> NDArray[np.bool_]:
    return _resize_nearest(array.astype(np.float32), shape) > 0.0


def _mean(values: NDArray[np.float32]) -> float | None:
    return float(values.mean()) if values.size else None


def _percentile(values: NDArray[np.float32], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None
