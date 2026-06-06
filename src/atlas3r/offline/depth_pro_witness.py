"""Depth Pro witness orchestration and normalized proposal records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME
from atlas3r.contracts.truth import TruthBoundary
from atlas3r.models.adapters.depth_pro_adapter import (
    DepthProAdapterConfig,
    DepthProBatchPrediction,
    DepthProRuntimeError,
    probe_depth_pro_runtime,
    run_depth_pro_frames,
)
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import FailurePoint

_DEPTH_PRO_SCALE_SOURCE = "depth_pro_metric_proposal_unanchored"


@dataclass(frozen=True)
class DepthProRuntimeOptions:
    enabled: bool = False
    proposal_cache: str | None = None
    repo_path: str | None = None
    checkpoint: str | None = None
    device: str = "cuda:0"
    image_size: int | None = None
    max_keyframes: int | None = None


@dataclass(frozen=True)
class DepthProWitnessResult:
    runtime_status: str
    available: bool
    reason: str
    install_hint: str
    camera_records: tuple[dict[str, object], ...] = ()
    depth_records: tuple[dict[str, object], ...] = ()
    depth_arrays: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    frame_records: tuple[dict[str, object], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    replay_source: str | None = None
    failure_code: str | None = None

    @property
    def has_depth(self) -> bool:
        return bool(self.available and self.depth_records)


def run_depth_pro_witness(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    options: DepthProRuntimeOptions,
    failure_points: list[FailurePoint],
) -> DepthProWitnessResult:
    if options.proposal_cache:
        try:
            return load_depth_pro_proposal_cache(options.proposal_cache)
        except Exception as exc:
            return _unavailable_with_failure(
                failure_points,
                code="depth_pro_replay_failed",
                why=f"Depth Pro proposal-cache replay failed: {exc}",
                dependency_missing=None,
            )
    if not options.enabled:
        return DepthProWitnessResult(
            runtime_status="disabled",
            available=False,
            reason="Depth Pro was not enabled for this run",
            install_hint=(
                "Use --enable-depth-pro or --depth-pro-proposal-cache to provide "
                "Depth Pro proposals."
            ),
            failure_code=None,
        )
    if not frame_cache.frames or not keyframes:
        return _unavailable_with_failure(
            failure_points,
            code="depth_pro_no_keyframes",
            why="Depth Pro needs decoded frames and selected keyframes",
            dependency_missing=None,
        )
    runtime_available, runtime_reason = probe_depth_pro_runtime(options.repo_path)
    if not runtime_available:
        return _unavailable_with_failure(
            failure_points,
            code="depth_pro_runtime_unavailable",
            why=runtime_reason,
            dependency_missing="torch and external depth_pro runtime",
        )
    try:
        return _run_external_depth_pro(
            run_dir, frame_cache=frame_cache, keyframes=keyframes, options=options
        )
    except (DepthProRuntimeError, ValueError, OSError) as exc:
        return _unavailable_with_failure(
            failure_points,
            code="depth_pro_runtime_failed",
            why=f"Depth Pro runtime failed: {exc}",
            dependency_missing="working Depth Pro runtime, checkpoint, and device",
        )


def load_depth_pro_proposal_cache(path: str | Path) -> DepthProWitnessResult:
    root = _resolve_proposal_root(Path(path))
    camera_records = tuple(_read_jsonl(root / "depth_pro_cameras.jsonl"))
    frame_records_path = root / "depth_pro_frames.jsonl"
    if not frame_records_path.is_file():
        frame_records_path = root / "depth_pro_windows.jsonl"
    frame_records = tuple(_read_jsonl(frame_records_path)) if frame_records_path.is_file() else ()
    depth_path = root / "depth_pro_depths.npz"
    if not depth_path.is_file():
        raise ValueError("Depth Pro replay cache must contain depth_pro_depths.npz")
    with np.load(depth_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key != "metadata_json"
        }
    if not isinstance(metadata, dict):
        raise ValueError("Depth Pro depth metadata must be a JSON object")
    depth_records_obj = metadata.get("depth_records", [])
    if not isinstance(depth_records_obj, list):
        raise ValueError("Depth Pro metadata must include depth_records")
    return DepthProWitnessResult(
        runtime_status="replayed",
        available=True,
        reason="Depth Pro proposals replayed from proposal cache",
        install_hint="Replay mode does not require importing Depth Pro.",
        camera_records=camera_records,
        depth_records=tuple(_require_dict(item) for item in depth_records_obj),
        depth_arrays=arrays,
        frame_records=frame_records,
        metadata=_require_dict(metadata.get("depth_pro_metadata", {})),
        replay_source=str(root),
    )


def truth_boundary_dict() -> dict[str, object]:
    return TruthBoundary.teacher_pseudo(
        _DEPTH_PRO_SCALE_SOURCE,
        notes="Depth Pro teacher proposal; unanchored RGB geometry is not measured truth.",
    ).to_dict() | {
        "usable_for_training": False,
    }


def focal_px_to_K(focal_px: float, *, width: int, height: int) -> NDArray[np.float32]:
    if not np.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("focal_px must be positive and finite")
    return np.array(
        [
            [focal_px, 0.0, float(width - 1) / 2.0],
            [0.0, focal_px, float(height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def normalize_depth_pro_prediction(
    prediction: DepthProBatchPrediction,
    *,
    keyframe_index: dict[int, int],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, NDArray[np.float32]],
    tuple[dict[str, object], ...],
]:
    camera_records: list[dict[str, object]] = []
    depth_records: list[dict[str, object]] = []
    depth_arrays: dict[str, NDArray[np.float32]] = {}
    frame_records: list[dict[str, object]] = []
    for proposal_id, frame_prediction in enumerate(prediction.frame_predictions):
        frame_id = int(frame_prediction.frame_id)
        depth_raw = np.asarray(frame_prediction.depth_m, dtype=np.float32)
        if depth_raw.ndim != 2:
            raise ValueError("Depth Pro depth must be HxW")
        valid_mask = np.isfinite(depth_raw) & (depth_raw > 0.0)
        depth = np.where(valid_mask, depth_raw, 0.0).astype(np.float32)
        confidence = _derive_confidence(depth, valid_mask)
        sigma = np.where(valid_mask, np.maximum(0.05, depth * (1.0 - confidence)), 0.0)
        sigma = sigma.astype(np.float32)
        depth_key = f"depth_pro_depth_{proposal_id:06d}"
        sigma_key = f"depth_pro_depth_sigma_{proposal_id:06d}"
        confidence_key = f"depth_pro_confidence_{proposal_id:06d}"
        valid_key = f"depth_pro_valid_mask_{proposal_id:06d}"
        depth_arrays[depth_key] = depth
        depth_arrays[sigma_key] = sigma
        depth_arrays[confidence_key] = confidence
        depth_arrays[valid_key] = valid_mask.astype(np.float32)
        height, width = int(depth.shape[0]), int(depth.shape[1])
        K = None
        focal_px = frame_prediction.focal_px
        if focal_px is not None:
            K = focal_px_to_K(float(focal_px), width=width, height=height)
        truth = truth_boundary_dict()
        camera_records.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": int(keyframe_index[frame_id]),
                "K": None if K is None else K.tolist(),
                "focal_px": focal_px,
                "fx": focal_px,
                "fy": focal_px,
                "intrinsics_source": "depth_pro" if K is not None else "unavailable",
                "metric_scale_source": _DEPTH_PRO_SCALE_SOURCE,
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "truth_boundary": truth,
                "depth_key": depth_key,
                "depth_shape": [height, width],
            }
        )
        depth_records.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": int(keyframe_index[frame_id]),
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": [height, width],
                "depth_source": "depth_pro",
                "metric_scale_source": _DEPTH_PRO_SCALE_SOURCE,
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "truth_boundary": truth,
                "confidence_derived": True,
            }
        )
        frame_records.append(
            {
                "teacher_name": "depth_pro",
                "frame_id": frame_id,
                "keyframe_index": int(keyframe_index[frame_id]),
                "status": "predicted",
                "runtime_ms": frame_prediction.runtime_ms,
                "output_shapes": frame_prediction.output_shapes,
                "truth_boundary": truth,
            }
        )
    return tuple(camera_records), tuple(depth_records), depth_arrays, tuple(frame_records)


def _run_external_depth_pro(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    options: DepthProRuntimeOptions,
) -> DepthProWitnessResult:
    selected = _select_depth_pro_keyframes(keyframes, options.max_keyframes)
    frame_paths = {
        record.frame_id: Path(run_dir) / record.frame_path for record in frame_cache.records
    }
    frame_ids = tuple(record.frame_id for record in selected)
    prediction = run_depth_pro_frames(
        [frame_paths[frame_id] for frame_id in frame_ids],
        frame_ids,
        DepthProAdapterConfig(
            repo_path=options.repo_path,
            checkpoint=options.checkpoint,
            device=options.device,
            image_size=options.image_size,
        ),
    )
    keyframe_index = {record.frame_id: record.keyframe_id for record in selected}
    camera_records, depth_records, depth_arrays, frame_records = normalize_depth_pro_prediction(
        prediction, keyframe_index=keyframe_index
    )
    metadata: dict[str, object] = {
        "teacher_name": "depth_pro",
        "model_source": prediction.model_source,
        "checkpoint": prediction.checkpoint,
        "device": prediction.device,
        "image_size": prediction.image_size,
        "frame_ids": list(frame_ids),
        "runtime_ms": prediction.runtime_ms,
        "output_shapes": {
            str(item.frame_id): item.output_shapes for item in prediction.frame_predictions
        },
        "coordinate_convention": COORDINATE_FRAME_NAME,
        "truth_boundary": truth_boundary_dict(),
    }
    return DepthProWitnessResult(
        runtime_status="running",
        available=True,
        reason="Depth Pro runtime produced normalized teacher proposals",
        install_hint="Depth Pro proposals were produced for this run.",
        camera_records=camera_records,
        depth_records=depth_records,
        depth_arrays=depth_arrays,
        frame_records=frame_records,
        metadata=metadata,
    )


def _derive_confidence(
    depth: NDArray[np.float32], valid_mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    if not np.any(valid_mask):
        return np.zeros(depth.shape, dtype=np.float32)
    finite_depth = depth[valid_mask]
    scale = float(np.median(finite_depth)) if finite_depth.size else 1.0
    scale = max(scale, 1e-3)
    grad_y = np.zeros_like(depth, dtype=np.float32)
    grad_x = np.zeros_like(depth, dtype=np.float32)
    grad_y[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
    grad_x[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    smoothness = 1.0 - np.clip((grad_x + grad_y) / scale, 0.0, 1.0)
    confidence = np.where(valid_mask, 0.25 + 0.5 * smoothness, 0.0)
    return confidence.astype(np.float32)


def _select_depth_pro_keyframes(
    keyframes: tuple[KeyframeRecord, ...], max_keyframes: int | None
) -> tuple[KeyframeRecord, ...]:
    if max_keyframes is None:
        return keyframes
    if max_keyframes <= 0:
        raise ValueError("depth_pro_max_keyframes must be positive when provided")
    return keyframes[:max_keyframes]


def _resolve_proposal_root(path: Path) -> Path:
    if (path / "proposals").is_dir():
        return path / "proposals"
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(_require_dict(json.loads(line)))
    return rows


def _require_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _unavailable_with_failure(
    failure_points: list[FailurePoint],
    *,
    code: str,
    why: str,
    dependency_missing: str | None,
) -> DepthProWitnessResult:
    install_hint = (
        "Install Depth Pro (Apple's ml-depth-pro) externally, pass --depth-pro-repo for "
        "a local checkout, provide --depth-pro-checkpoint if needed, or use "
        "--depth-pro-proposal-cache to replay normalized proposals."
    )
    failure_points.append(
        FailurePoint(
            module="teacher_witness_layer",
            code=code,
            severity="warning",
            status="unavailable",
            why=why,
            dependency_missing=dependency_missing,
            future_module=install_hint,
            artifact_path="teachers/teacher_status.json",
        )
    )
    return DepthProWitnessResult(
        runtime_status="unavailable",
        available=False,
        reason=why,
        install_hint=install_hint,
        failure_code=code,
    )
