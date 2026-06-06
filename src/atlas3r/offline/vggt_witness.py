"""VGGT witness orchestration and normalized proposal records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME, invert_T_A_B, validate_T_A_B
from atlas3r.contracts.truth import TruthBoundary
from atlas3r.models.adapters.vggt_adapter import (
    VggtAdapterConfig,
    VggtRuntimeError,
    VggtWindowPrediction,
    probe_vggt_runtime,
    run_vggt_window,
)
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import FailurePoint
from atlas3r.offline.stitching import (
    empty_stitch_stats,
    estimate_sim3_umeyama,
    none_stitch_stats,
    transform_pose_with_sim3,
)

VggtStitchMode = Literal["none", "overlap-sim3"]
_VGGT_SCALE_SOURCE = "vggt_unanchored_metric_proposal"


@dataclass(frozen=True)
class VggtRuntimeOptions:
    enabled: bool = False
    proposal_cache: str | None = None
    repo_path: str | None = None
    checkpoint: str | None = None
    device: str = "cuda:0"
    image_size: int = 518
    window_size: int = 24
    window_overlap: int = 8
    max_keyframes: int | None = None
    stitch_mode: VggtStitchMode = "overlap-sim3"


@dataclass(frozen=True)
class VggtWitnessResult:
    runtime_status: str
    available: bool
    reason: str
    install_hint: str
    camera_records: tuple[dict[str, object], ...] = ()
    depth_records: tuple[dict[str, object], ...] = ()
    depth_arrays: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    window_records: tuple[dict[str, object], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    replay_source: str | None = None
    failure_code: str | None = None

    @property
    def has_geometry(self) -> bool:
        return bool(self.available and self.camera_records and self.depth_records)


def run_vggt_witness(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    options: VggtRuntimeOptions,
    failure_points: list[FailurePoint],
) -> VggtWitnessResult:
    if options.proposal_cache:
        try:
            return load_vggt_proposal_cache(options.proposal_cache)
        except Exception as exc:
            return _unavailable_with_failure(
                failure_points,
                code="vggt_replay_failed",
                why=f"VGGT proposal-cache replay failed: {exc}",
                dependency_missing=None,
            )
    if not options.enabled:
        return VggtWitnessResult(
            runtime_status="disabled",
            available=False,
            reason="VGGT was not enabled for this run",
            install_hint="Use --enable-vggt or --vggt-proposal-cache to provide VGGT proposals.",
            failure_code=None,
        )
    if not frame_cache.frames or not keyframes:
        return _unavailable_with_failure(
            failure_points,
            code="vggt_no_keyframes",
            why="VGGT needs decoded frames and selected keyframes",
            dependency_missing=None,
        )
    runtime_available, runtime_reason = probe_vggt_runtime(options.repo_path)
    if not runtime_available:
        return _unavailable_with_failure(
            failure_points,
            code="vggt_runtime_unavailable",
            why=runtime_reason,
            dependency_missing="torch and VGGT external runtime",
        )
    try:
        return _run_external_vggt(
            run_dir, frame_cache=frame_cache, keyframes=keyframes, options=options
        )
    except (VggtRuntimeError, ValueError, OSError) as exc:
        return _unavailable_with_failure(
            failure_points,
            code="vggt_runtime_failed",
            why=f"VGGT runtime failed: {exc}",
            dependency_missing="working VGGT runtime, checkpoint, and device",
        )


def load_vggt_proposal_cache(path: str | Path) -> VggtWitnessResult:
    root = _resolve_proposal_root(Path(path))
    camera_records = tuple(_read_jsonl(root / "vggt_cameras.jsonl"))
    window_records = tuple(_read_jsonl(root / "vggt_windows.jsonl"))
    depth_path = root / "vggt_depths.npz"
    if not camera_records or not depth_path.is_file():
        raise ValueError("VGGT replay cache must contain cameras and depth NPZ")
    with np.load(depth_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key != "metadata_json"
        }
    if not isinstance(metadata, dict):
        raise ValueError("VGGT depth metadata must be a JSON object")
    depth_records_obj = metadata.get("depth_records", [])
    if not isinstance(depth_records_obj, list):
        raise ValueError("VGGT depth metadata must include depth_records")
    return VggtWitnessResult(
        runtime_status="replayed",
        available=True,
        reason="VGGT proposals replayed from proposal cache",
        install_hint="Replay mode does not require importing VGGT.",
        camera_records=camera_records,
        depth_records=tuple(_require_dict(item) for item in depth_records_obj),
        depth_arrays=arrays,
        window_records=window_records,
        metadata=_require_dict(metadata.get("vggt_metadata", {})),
        replay_source=str(root),
    )


def truth_boundary_dict() -> dict[str, object]:
    return TruthBoundary.teacher_pseudo(
        _VGGT_SCALE_SOURCE,
        notes="VGGT teacher proposal; unanchored RGB geometry is not measured truth.",
    ).to_dict() | {
        "usable_for_training": False,
    }


def stitch_vggt_records(
    camera_records: list[dict[str, object]],
    depth_records: list[dict[str, object]],
    depth_arrays: dict[str, NDArray[np.float32]],
    *,
    stitch_mode: VggtStitchMode,
) -> tuple[list[dict[str, object]], dict[str, NDArray[np.float32]], dict[str, object]]:
    return _stitch_records(camera_records, depth_records, depth_arrays, stitch_mode=stitch_mode)


def _run_external_vggt(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    options: VggtRuntimeOptions,
) -> VggtWitnessResult:
    selected = _select_vggt_keyframes(keyframes, options.max_keyframes)
    if options.window_size <= 0:
        raise ValueError("vggt_window_size must be positive")
    if options.window_overlap < 0 or options.window_overlap >= options.window_size:
        raise ValueError("vggt_window_overlap must be non-negative and smaller than window size")
    frame_paths = {
        record.frame_id: Path(run_dir) / record.frame_path for record in frame_cache.records
    }
    keyframe_index = {record.frame_id: record.keyframe_id for record in selected}
    config = VggtAdapterConfig(
        repo_path=options.repo_path,
        checkpoint=options.checkpoint,
        device=options.device,
        image_size=options.image_size,
    )
    camera_records: list[dict[str, object]] = []
    depth_records: list[dict[str, object]] = []
    depth_arrays: dict[str, NDArray[np.float32]] = {}
    window_records: list[dict[str, object]] = []
    output_shapes: dict[str, object] = {}
    runtime_ms = 0.0
    dtype = "unknown"
    model_source = options.repo_path or "python_package:vggt"
    checkpoint = options.checkpoint or "facebook/VGGT-1B"
    for window_id, window in enumerate(
        _build_windows(selected, options.window_size, options.window_overlap)
    ):
        frame_ids = tuple(record.frame_id for record in window)
        prediction = run_vggt_window(
            [frame_paths[frame_id] for frame_id in frame_ids],
            frame_ids,
            config,
        )
        runtime_ms += prediction.runtime_ms
        dtype = prediction.dtype
        model_source = prediction.model_source
        checkpoint = prediction.checkpoint
        output_shapes[f"window_{window_id}"] = prediction.output_shapes
        _append_prediction_records(
            prediction,
            window_id=window_id,
            keyframe_index=keyframe_index,
            camera_records=camera_records,
            depth_records=depth_records,
            depth_arrays=depth_arrays,
        )
        window_records.append(
            {
                "teacher_name": "vggt",
                "window_id": window_id,
                "frame_ids": list(frame_ids),
                "status": "raw_predicted",
                "runtime_ms": prediction.runtime_ms,
                "truth_boundary": truth_boundary_dict(),
            }
        )
    camera_records, depth_arrays, stitch_stats = _stitch_records(
        camera_records, depth_records, depth_arrays, stitch_mode=options.stitch_mode
    )
    window_records = _attach_window_stitch_stats(window_records, stitch_stats)
    metadata = {
        "teacher_name": "vggt",
        "model_source": model_source,
        "checkpoint": checkpoint,
        "device": options.device,
        "dtype": dtype,
        "image_size": options.image_size,
        "window_size": options.window_size,
        "window_overlap": options.window_overlap,
        "frame_ids": [record.frame_id for record in selected],
        "runtime_ms": runtime_ms,
        "coordinate_convention": COORDINATE_FRAME_NAME,
        "output_shapes": output_shapes,
        "stitching": stitch_stats,
        "truth_boundary": truth_boundary_dict(),
    }
    return VggtWitnessResult(
        runtime_status="running",
        available=True,
        reason="VGGT runtime produced normalized teacher proposals",
        install_hint="VGGT proposals were produced for this run.",
        camera_records=tuple(camera_records),
        depth_records=tuple(depth_records),
        depth_arrays=depth_arrays,
        window_records=tuple(window_records),
        metadata=metadata,
    )


def _append_prediction_records(
    prediction: VggtWindowPrediction,
    *,
    window_id: int,
    keyframe_index: dict[int, int],
    camera_records: list[dict[str, object]],
    depth_records: list[dict[str, object]],
    depth_arrays: dict[str, NDArray[np.float32]],
) -> None:
    for local_index, frame_id in enumerate(prediction.frame_ids):
        T_camera_window = prediction.extrinsics_camera_from_world[local_index]
        T_window_camera = invert_T_A_B(T_camera_window)
        if not np.all(np.isfinite(T_window_camera)):
            raise ValueError(f"VGGT emitted non-finite transform for frame {frame_id}")
        depth = prediction.depth_m[local_index].astype(np.float32)
        confidence = np.clip(prediction.depth_confidence[local_index], 0.0, 1.0).astype(np.float32)
        valid_mask = (np.isfinite(depth) & (depth > 0.0)).astype(np.float32)
        depth = np.where(valid_mask > 0.0, depth, 0.0).astype(np.float32)
        sigma = np.maximum(0.05, depth * (1.0 - confidence)).astype(np.float32)
        proposal_id = len(depth_records)
        depth_key = f"vggt_depth_{proposal_id:06d}"
        sigma_key = f"vggt_depth_sigma_{proposal_id:06d}"
        confidence_key = f"vggt_confidence_{proposal_id:06d}"
        valid_key = f"vggt_valid_mask_{proposal_id:06d}"
        depth_arrays[depth_key] = depth
        depth_arrays[sigma_key] = sigma
        depth_arrays[confidence_key] = confidence
        depth_arrays[valid_key] = valid_mask
        K = prediction.intrinsics[local_index].astype(np.float32)
        camera_records.append(
            {
                "teacher_name": "vggt",
                "frame_id": int(frame_id),
                "keyframe_index": int(keyframe_index[int(frame_id)]),
                "window_id": window_id,
                "K": K.tolist(),
                "T_window_camera": T_window_camera.tolist(),
                "T_world_camera": T_window_camera.tolist(),
                "camera_center_world_m": T_window_camera[:3, 3].tolist(),
                "pose_confidence": float(np.mean(confidence[valid_mask > 0.0]))
                if np.any(valid_mask > 0.0)
                else 0.0,
                "intrinsics_source": "vggt",
                "pose_source": "vggt",
                "metric_scale_source": _VGGT_SCALE_SOURCE,
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "truth_boundary": truth_boundary_dict(),
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": list(depth.shape),
                "pseudo_submap_id": 0,
                "depth_scale_applied": 1.0,
                "stitch_status": "raw",
            }
        )
        depth_records.append(
            {
                "teacher_name": "vggt",
                "frame_id": int(frame_id),
                "keyframe_index": int(keyframe_index[int(frame_id)]),
                "window_id": window_id,
                "depth_key": depth_key,
                "depth_sigma_key": sigma_key,
                "confidence_key": confidence_key,
                "valid_mask_key": valid_key,
                "depth_shape": list(depth.shape),
                "depth_source": "vggt",
                "metric_scale_source": _VGGT_SCALE_SOURCE,
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "truth_boundary": truth_boundary_dict(),
                "confidence_derived": False,
            }
        )


def _stitch_records(
    camera_records: list[dict[str, object]],
    depth_records: list[dict[str, object]],
    depth_arrays: dict[str, NDArray[np.float32]],
    *,
    stitch_mode: VggtStitchMode,
) -> tuple[list[dict[str, object]], dict[str, NDArray[np.float32]], dict[str, object]]:
    if not camera_records:
        return camera_records, depth_arrays, empty_stitch_stats(stitch_mode)
    if stitch_mode == "none":
        for record in camera_records:
            record["pseudo_submap_id"] = _int_field(record, "window_id")
            record["stitch_status"] = "not_fused"
        window_ids = {_int_field(record, "window_id") for record in camera_records}
        return camera_records, depth_arrays, none_stitch_stats(window_ids, has_records=True)
    records_by_window: dict[int, list[dict[str, object]]] = {}
    for record in camera_records:
        records_by_window.setdefault(_int_field(record, "window_id"), []).append(record)
    world_centers_by_frame: dict[int, NDArray[np.float32]] = {}
    accepted_scales: list[float] = []
    edge_rmses: list[float] = []
    accepted_edge_count = 0
    rejected_edge_count = 0
    rejected_windows: list[int] = []
    next_submap_id = 0
    for window_id in sorted(records_by_window):
        records = records_by_window[window_id]
        if window_id == 0:
            for record in records:
                _set_world_pose(record, validate_T_A_B(record["T_window_camera"]), 0, 1.0, "root")
                world_centers_by_frame[_int_field(record, "frame_id")] = _camera_center(record)
            accepted_scales.append(1.0)
            continue
        overlaps = [
            record for record in records if _int_field(record, "frame_id") in world_centers_by_frame
        ]
        if len(overlaps) >= 3:
            source = np.stack(
                [validate_T_A_B(record["T_window_camera"])[:3, 3] for record in overlaps]
            ).astype(np.float32)
            target = np.stack(
                [world_centers_by_frame[_int_field(record, "frame_id")] for record in overlaps]
            ).astype(np.float32)
            sim3 = estimate_sim3_umeyama(source, target)
        else:
            sim3 = estimate_sim3_umeyama(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
        if sim3.accepted:
            accepted_edge_count += 1
            edge_rmses.append(sim3.rmse_m)
            accepted_scales.append(sim3.scale)
            submap_id = 0
            for record in records:
                T_world_camera = transform_pose_with_sim3(record["T_window_camera"], sim3)
                _set_world_pose(
                    record, T_world_camera, submap_id, sim3.scale, "accepted_overlap_sim3"
                )
                _scale_depth_record_arrays(record, depth_arrays, sim3.scale)
                world_centers_by_frame[_int_field(record, "frame_id")] = _camera_center(record)
        else:
            rejected_edge_count += 1
            rejected_windows.append(window_id)
            next_submap_id += 1
            for record in records:
                _set_world_pose(
                    record,
                    validate_T_A_B(record["T_window_camera"]),
                    next_submap_id,
                    1.0,
                    f"new_pseudo_submap:{sim3.reason}",
                )
    stats = {
        "vggt_window_count": len(records_by_window),
        "stitch_mode": stitch_mode,
        "stitch_edge_count": max(0, len(records_by_window) - 1),
        "accepted_edge_count": accepted_edge_count,
        "rejected_edge_count": rejected_edge_count,
        "pseudo_submap_count": len(
            {_int_field(record, "pseudo_submap_id") for record in camera_records}
        ),
        "overlap_center_rmse_m": float(np.median(edge_rmses)) if edge_rmses else None,
        "scale_min": float(np.min(accepted_scales)) if accepted_scales else None,
        "scale_median": float(np.median(accepted_scales)) if accepted_scales else None,
        "scale_max": float(np.max(accepted_scales)) if accepted_scales else None,
        "rejected_windows": rejected_windows,
    }
    return camera_records, depth_arrays, stats


def _set_world_pose(
    record: dict[str, object],
    T_world_camera: NDArray[np.float32],
    pseudo_submap_id: int,
    depth_scale: float,
    stitch_status: str,
) -> None:
    if not np.all(np.isfinite(T_world_camera)):
        raise ValueError("T_world_camera must be finite")
    record["T_world_camera"] = T_world_camera.astype(np.float32).tolist()
    record["camera_center_world_m"] = T_world_camera[:3, 3].astype(np.float32).tolist()
    record["pseudo_submap_id"] = pseudo_submap_id
    record["depth_scale_applied"] = float(depth_scale)
    record["stitch_status"] = stitch_status


def _scale_depth_record_arrays(
    record: dict[str, object],
    depth_arrays: dict[str, NDArray[np.float32]],
    scale: float,
) -> None:
    depth_key = str(record["depth_key"])
    sigma_key = str(record["depth_sigma_key"])
    depth_arrays[depth_key] = (depth_arrays[depth_key] * scale).astype(np.float32)
    depth_arrays[sigma_key] = (depth_arrays[sigma_key] * scale).astype(np.float32)


def _camera_center(record: dict[str, object]) -> NDArray[np.float32]:
    return np.asarray(record["camera_center_world_m"], dtype=np.float32)


def _attach_window_stitch_stats(
    window_records: list[dict[str, object]], stitch_stats: dict[str, object]
) -> list[dict[str, object]]:
    rejected = {
        _int_value(item) for item in _object_sequence(stitch_stats.get("rejected_windows", []))
    }
    for record in window_records:
        window_id = _int_field(record, "window_id")
        record["stitch_mode"] = stitch_stats["stitch_mode"]
        record["stitch_status"] = (
            "rejected_new_submap" if window_id in rejected else "accepted_or_root"
        )
    return window_records


def _build_windows(
    keyframes: tuple[KeyframeRecord, ...], window_size: int, window_overlap: int
) -> list[tuple[KeyframeRecord, ...]]:
    if len(keyframes) <= window_size:
        return [keyframes]
    step = window_size - window_overlap
    windows: list[tuple[KeyframeRecord, ...]] = []
    start = 0
    while start < len(keyframes):
        window = keyframes[start : start + window_size]
        if not window:
            break
        windows.append(window)
        if start + window_size >= len(keyframes):
            break
        start += step
    return windows


def _select_vggt_keyframes(
    keyframes: tuple[KeyframeRecord, ...], max_keyframes: int | None
) -> tuple[KeyframeRecord, ...]:
    if max_keyframes is None:
        return keyframes
    if max_keyframes <= 0:
        raise ValueError("vggt_max_keyframes must be positive when provided")
    return keyframes[:max_keyframes]


def _resolve_proposal_root(path: Path) -> Path:
    if path.is_file() and path.name == "proposal_manifest.json":
        return path.parent
    if (path / "proposal_manifest.json").is_file():
        return path
    if (path / "proposals" / "proposal_manifest.json").is_file():
        return path / "proposals"
    raise ValueError(f"could not find proposal_manifest.json under {path}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL stream: {path}")
    rows: list[dict[str, object]] = []
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
) -> VggtWitnessResult:
    failure_points.append(
        FailurePoint(
            module="vggt_witness",
            code=code,
            severity="warning",
            status="unavailable",
            why=why,
            dependency_missing=dependency_missing,
            future_module="install VGGT externally or provide --vggt-proposal-cache",
            artifact_path="teachers/teacher_status.json",
        )
    )
    return VggtWitnessResult(
        runtime_status="unavailable",
        available=False,
        reason=why,
        install_hint=(
            "Install VGGT externally, pass --vggt-repo/--vggt-checkpoint as needed, "
            "or replay a previous cache with --vggt-proposal-cache."
        ),
        failure_code=code,
    )


def _int_field(record: dict[str, object], key: str) -> int:
    return _int_value(record[key])


def _int_value(value: object) -> int:
    return int(str(value))


def _object_sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
