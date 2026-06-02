"""TUM RGB-D checkpoint evaluation and mapping diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.eval._tum_rgbd_checkpoint_helpers import (
    predicted_observation,
    target_observation,
    write_mapping_diagnostics,
    write_tum_trajectory,
)
from atlas3r.mapping.observations import DepthObservation
from atlas3r.training.checkpoint_inference import (
    TinyDepthPoseCheckpoint,
    load_tiny_depth_pose_checkpoint,
)
from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_rgbd_dataset import TumRgbdDepthDataset

_THRESHOLDS_M = (
    ("1mm", 0.001),
    ("5mm", 0.005),
    ("1cm", 0.01),
    ("5cm", 0.05),
    ("10cm", 0.10),
)


@dataclass(frozen=True)
class TumRgbdCheckpointEvalConfig:
    checkpoint: Path
    manifest: Path
    output: Path
    split: str = "val"
    width: int = 160
    height: int = 120
    device: str = "cuda"
    max_frames: int | None = 64
    num_workers: int = 2
    write_tsdf: bool = False
    voxel_size_m: float = 0.1
    truncation_voxels: float = 3.0
    map_max_points: int = 3000


def run_tum_rgbd_checkpoint_eval(config: TumRgbdCheckpointEvalConfig) -> dict[str, object]:
    """Evaluate a tiny TUM RGB-D checkpoint on held-out manifest frames."""

    _validate_config(config)
    torch = require_torch()
    device = select_device(config.device)
    checkpoint = load_tiny_depth_pose_checkpoint(config.checkpoint, device=device)
    dataset = TumRgbdDepthDataset(
        config.manifest,
        width=config.width,
        height=config.height,
        split=config.split,
    )
    indices = list(range(len(dataset)))
    if config.max_frames is not None:
        indices = indices[: config.max_frames]
    if not indices:
        raise ValueError("max_frames selected no TUM RGB-D evaluation frames")
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=min(8, len(indices)),
        shuffle=False,
        num_workers=config.num_workers,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    per_frame_path = config.output / "per_frame_metrics.jsonl"
    per_frame_path.write_text("", encoding="utf-8")

    per_frame_metrics: list[dict[str, object]] = []
    predicted_observations: list[DepthObservation] = []
    target_observations: list[DepthObservation] = []
    trajectory_estimate: list[tuple[float, npt.NDArray[np.float32]]] = []
    trajectory_target: list[tuple[float, npt.NDArray[np.float32]]] = []
    sample_payload: dict[str, Any] | None = None

    checkpoint.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            output = checkpoint.model(batch["images_rgb"], batch["intrinsics"])
            batch_result = _evaluate_batch(
                batch=batch,
                output=output,
                checkpoint=checkpoint,
            )
            for metric in batch_result["per_frame_metrics"]:
                _append_jsonl(per_frame_path, cast(dict[str, object], metric))
                per_frame_metrics.append(cast(dict[str, object], metric))
            predicted_observations.extend(batch_result["predicted_observations"])
            target_observations.extend(batch_result["target_observations"])
            trajectory_estimate.extend(batch_result["trajectory_estimate"])
            trajectory_target.extend(batch_result["trajectory_target"])
            if sample_payload is None:
                sample_payload = cast(dict[str, Any], batch_result["sample_payload"])

    if sample_payload is None:
        raise ValueError("evaluation produced no prediction sample")
    summary = _summary_record(config, checkpoint, dataset.manifest, per_frame_metrics)
    np.savez(config.output / "prediction_sample.npz", **sample_payload)
    write_prediction_preview(
        output_dir=config.output,
        rgb_u8=sample_payload["rgb_u8"],
        target_depth_m=sample_payload["target_depth_m"],
        predicted_depth_m=sample_payload["predicted_depth_m"],
        abs_error_m=sample_payload["abs_depth_error_m"],
        valid_depth_mask=sample_payload["valid_depth_mask"],
        metrics=cast(dict[str, float], summary["depth_metrics"]),
        truth_boundary=cast(dict[str, object], summary["truth_boundary"]),
        title="Atlas3R TUM RGB-D Checkpoint Evaluation Preview",
    )
    write_tum_trajectory(config.output / "trajectory_estimate_tum.txt", trajectory_estimate)
    write_tum_trajectory(config.output / "trajectory_groundtruth_tum.txt", trajectory_target)
    if config.write_tsdf:
        map_record = write_mapping_diagnostics(
            output=config.output,
            checkpoint=checkpoint,
            split=config.split,
            voxel_size_m=config.voxel_size_m,
            truncation_voxels=config.truncation_voxels,
            map_max_points=config.map_max_points,
            predicted_observations=tuple(predicted_observations),
            target_observations=tuple(target_observations),
        )
        summary["artifacts"]["map_comparison"] = "map_comparison.json"
        summary["map_comparison"] = map_record
    _write_json(config.output / "summary.json", summary)
    return summary


def _evaluate_batch(
    *,
    batch: dict[str, Any],
    output: dict[str, Any],
    checkpoint: TinyDepthPoseCheckpoint,
) -> dict[str, Any]:
    images = _tensor_numpy(batch["images_rgb"]).astype(np.float32)
    intrinsics = _tensor_numpy(batch["intrinsics"]).astype(np.float32)
    target = cast(dict[str, Any], batch["target"])
    target_depth = _tensor_numpy(target["depth_m"])[:, 0].astype(np.float32)
    valid_mask = _tensor_numpy(target["valid_depth_mask"])[:, 0].astype(np.bool_)
    target_T = _tensor_numpy(target["T_world_camera"]).astype(np.float32)
    target_center = _tensor_numpy(target["camera_center_world_m"]).astype(np.float32)
    pred_depth = _tensor_numpy(output["depth_m"])[:, 0].astype(np.float32)
    pred_sigma = _tensor_numpy(output["depth_sigma_m"])[:, 0].astype(np.float32)
    pred_confidence = _tensor_numpy(output["confidence"])[:, 0].astype(np.float32)
    pred_center = _tensor_numpy(output["camera_center_world_m"]).astype(np.float32)
    frame_ids = [int(value) for value in _metadata_values(batch["metadata"], "frame_id")]
    timestamps_s = [float(value) for value in _metadata_values(batch["metadata"], "timestamp_s")]
    rgb_paths = [str(value) for value in _metadata_values(batch["metadata"], "rgb_path")]

    per_frame: list[dict[str, object]] = []
    predicted_observations: list[DepthObservation] = []
    target_observations: list[DepthObservation] = []
    trajectory_estimate: list[tuple[float, npt.NDArray[np.float32]]] = []
    trajectory_target: list[tuple[float, npt.NDArray[np.float32]]] = []
    sample_payload: dict[str, Any] | None = None
    for index, frame_id in enumerate(frame_ids):
        rgb_u8 = _rgb_from_model_tensor(images[index])
        pred_T = target_T[index].copy()
        pred_T[:3, 3] = pred_center[index]
        per_frame.append(
            _per_frame_metrics(
                frame_id=frame_id,
                rgb_path=rgb_paths[index],
                target_depth=target_depth[index],
                predicted_depth=pred_depth[index],
                valid_mask=valid_mask[index],
                target_center=target_center[index],
                predicted_center=pred_center[index],
            )
        )
        predicted_observations.append(
            predicted_observation(
                frame_id=frame_id,
                timestamp_s=timestamps_s[index],
                K=intrinsics[index],
                rgb_u8=rgb_u8,
                T_world_camera=pred_T,
                depth_m=pred_depth[index],
                sigma_m=pred_sigma[index],
                confidence=pred_confidence[index],
                checkpoint=checkpoint,
            )
        )
        target_observations.append(
            target_observation(
                frame_id=frame_id,
                timestamp_s=timestamps_s[index],
                K=intrinsics[index],
                rgb_u8=rgb_u8,
                T_world_camera=target_T[index],
                depth_m=target_depth[index],
                valid_mask=valid_mask[index],
            )
        )
        trajectory_estimate.append((timestamps_s[index], pred_T))
        trajectory_target.append((timestamps_s[index], target_T[index]))
        if sample_payload is None:
            abs_error = np.abs(pred_depth[index] - target_depth[index]).astype(np.float32)
            sample_payload = {
                "frame_id": np.array(frame_id, dtype=np.int32),
                "rgb_u8": rgb_u8,
                "target_depth_m": target_depth[index],
                "valid_depth_mask": valid_mask[index],
                "predicted_depth_m": pred_depth[index],
                "abs_depth_error_m": abs_error,
                "predicted_depth_sigma_m": pred_sigma[index],
                "predicted_confidence": pred_confidence[index],
                "K": intrinsics[index],
                "target_T_world_camera": target_T[index],
                "predicted_T_world_camera": pred_T,
                "target_camera_center_world_m": target_center[index],
                "predicted_camera_center_world_m": pred_center[index],
            }
    return {
        "per_frame_metrics": per_frame,
        "predicted_observations": predicted_observations,
        "target_observations": target_observations,
        "trajectory_estimate": trajectory_estimate,
        "trajectory_target": trajectory_target,
        "sample_payload": sample_payload,
    }


def _per_frame_metrics(
    *,
    frame_id: int,
    rgb_path: str,
    target_depth: npt.NDArray[np.float32],
    predicted_depth: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
    target_center: npt.NDArray[np.float32],
    predicted_center: npt.NDArray[np.float32],
) -> dict[str, object]:
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count <= 0:
        raise ValueError(f"frame_id {frame_id}: no valid target depth pixels")
    target_values = target_depth[valid_mask].astype(np.float64, copy=False)
    predicted_values = predicted_depth[valid_mask].astype(np.float64, copy=False)
    abs_error = np.abs(predicted_values - target_values)
    record: dict[str, object] = {
        "format_name": "atlas3r_tum_rgbd_checkpoint_eval_frame_metric",
        "metric_family": "real_rgbd_debug_eval",
        "diagnostic_only": True,
        "accuracy_report": False,
        "frame_id": frame_id,
        "rgb_path": rgb_path,
        "depth_mae_m": float(np.mean(abs_error)),
        "depth_rmse_m": float(np.sqrt(np.mean(np.square(abs_error)))),
        "depth_absrel": float(np.mean(abs_error / np.maximum(target_values, 1e-6))),
        "valid_depth_pixels": valid_count,
        "pose_center_error_m": float(np.linalg.norm(predicted_center - target_center)),
    }
    for label, threshold_m in _THRESHOLDS_M:
        record[f"depth_within_{label}_percent"] = float(np.mean(abs_error <= threshold_m) * 100.0)
    return record


def _summary_record(
    config: TumRgbdCheckpointEvalConfig,
    checkpoint: TinyDepthPoseCheckpoint,
    manifest: dict[str, object],
    per_frame_metrics: list[dict[str, object]],
) -> dict[str, Any]:
    if not per_frame_metrics:
        raise ValueError("summary: no per-frame metrics were produced")
    valid_counts = _metric_array(per_frame_metrics, "valid_depth_pixels")
    total_valid = float(np.sum(valid_counts))
    mae = _weighted_mean(per_frame_metrics, "depth_mae_m", valid_counts)
    rmse = np.sqrt(_weighted_mean_square(per_frame_metrics, "depth_rmse_m", valid_counts))
    absrel = _weighted_mean(per_frame_metrics, "depth_absrel", valid_counts)
    pose_errors = _metric_array(per_frame_metrics, "pose_center_error_m")
    depth_metrics: dict[str, float] = {
        "mae_m": float(mae),
        "rmse_m": float(rmse),
        "absrel": float(absrel),
        "valid_depth_pixels": total_valid,
    }
    for label, _threshold_m in _THRESHOLDS_M:
        key = f"depth_within_{label}_percent"
        depth_metrics[key] = float(_weighted_mean(per_frame_metrics, key, valid_counts))
    return {
        "format_name": "atlas3r_tum_rgbd_checkpoint_eval_summary",
        "format_version": 1,
        "metric_family": "real_rgbd_debug_eval",
        "diagnostic_only": True,
        "accuracy_report": False,
        "performance_report": False,
        "dataset_name": manifest["dataset_name"],
        "sequence_name": manifest["sequence_name"],
        "split": config.split,
        "manifest_path": str(config.manifest),
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_step": checkpoint.step,
        "model_name": str(checkpoint.model_config.get("model_name", "TinyDepthPoseNet")),
        "frame_count": len(per_frame_metrics),
        "width": config.width,
        "height": config.height,
        "device": checkpoint.device,
        "depth_metrics": depth_metrics,
        "pose_center_metrics": {
            "mean_error_m": float(np.mean(pose_errors)),
            "median_error_m": float(np.median(pose_errors)),
            "p95_error_m": float(np.percentile(pose_errors, 95.0)),
        },
        "pose_note": "Only camera-center translation is evaluated; no full pose accuracy claim.",
        "artifacts": {
            "summary": "summary.json",
            "per_frame_metrics": "per_frame_metrics.jsonl",
            "prediction_sample": "prediction_sample.npz",
            "preview_html": "prediction_preview.html",
            "preview_svg": "prediction_preview.svg",
            "trajectory_estimate_tum": "trajectory_estimate_tum.txt",
            "trajectory_groundtruth_tum": "trajectory_groundtruth_tum.txt",
        },
        "truth_boundary": dict(checkpoint.truth_boundary),
    }


def _metric_array(metrics: list[dict[str, object]], key: str) -> npt.NDArray[np.float64]:
    return np.asarray([_metric_float(record[key], key=key) for record in metrics], dtype=np.float64)


def _weighted_mean(
    metrics: list[dict[str, object]],
    key: str,
    weights: npt.NDArray[np.float64],
) -> float:
    values = _metric_array(metrics, key)
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1.0))


def _weighted_mean_square(
    metrics: list[dict[str, object]],
    key: str,
    weights: npt.NDArray[np.float64],
) -> float:
    values = _metric_array(metrics, key)
    return float(np.sum(np.square(values) * weights) / max(float(np.sum(weights)), 1.0))


def _metric_float(value: object, *, key: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{key}: expected numeric metric value")
    return float(value)


def _tensor_numpy(value: Any) -> npt.NDArray[Any]:
    return cast(npt.NDArray[Any], value.detach().cpu().numpy())


def _metadata_values(metadata: dict[str, Any], key: str) -> list[Any]:
    value = metadata[key]
    if hasattr(value, "detach"):
        return cast(list[Any], value.detach().cpu().tolist())
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _move_batch_to_device(value: Any, device: str) -> Any:
    torch = require_torch()
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    return value


def _rgb_from_model_tensor(image_chw: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    return np.clip(np.transpose(image_chw, (1, 2, 0)) * 255.0, 0.0, 255.0).astype(np.uint8)


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_config(config: TumRgbdCheckpointEvalConfig) -> None:
    for field_name in ("width", "height"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    if config.num_workers < 0:
        raise ValueError("num_workers: must be non-negative")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if config.truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")
    if config.map_max_points <= 0:
        raise ValueError("map_max_points: must be positive")


__all__ = [
    "TumRgbdCheckpointEvalConfig",
    "run_tum_rgbd_checkpoint_eval",
]
