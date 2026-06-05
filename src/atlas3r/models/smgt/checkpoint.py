"""Checkpoint helpers for SMGT-tiny."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from atlas3r.models.smgt.config import SMGTTinyConfig
from atlas3r.training.torch_runtime import require_torch, select_device

FORMAT_NAME = "atlas3r_smgt_tiny_checkpoint"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SMGTTinyCheckpoint:
    model: Any
    checkpoint: dict[str, Any]
    device: str


def smgt_tiny_truth_boundary(*, pseudo_training_used: bool = True) -> dict[str, object]:
    return {
        "diagnostic_only": True,
        "shape_only": False,
        "student_rgb_only_used": True,
        "learned_inference": True,
        "usable_for_mapping": False,
        "rgb_only_mapping_ready": False,
        "realtime_claim": False,
        "accuracy_report": False,
        "performance_report": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "teacher_geometry_used": False,
        "pseudo_training_used": bool(pseudo_training_used),
        "measured_depth_used": False,
        "measured_pose_used": False,
        "metric_scale_source": "student_rgb_prior_unverified",
        "final_smgt": False,
        "object_aware_fusion_implemented": False,
    }


def save_smgt_tiny_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any | None,
    step: int,
    config_record: dict[str, object],
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    truth_boundary: dict[str, object],
    loss_config: dict[str, object] | None = None,
) -> None:
    torch = require_torch()
    payload = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "model_name": "SMGTTiny",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "config": config_record,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "model_config": model.model_config(),
        "loss_config": {} if loss_config is None else dict(loss_config),
        "truth_boundary": truth_boundary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_smgt_tiny_checkpoint(path: str | Path, *, device: str = "auto") -> SMGTTinyCheckpoint:
    torch = require_torch()
    resolved_device = select_device(device)
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ValueError(f"{checkpoint_path}: SMGTTiny checkpoint file does not exist")
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint must be a mapping")
    if payload.get("format_name") != FORMAT_NAME:
        raise ValueError(f"{checkpoint_path}: unsupported SMGTTiny checkpoint format")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{checkpoint_path}: unsupported SMGTTiny checkpoint version")
    _validate_truth_boundary(checkpoint_path, payload.get("truth_boundary"))
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("model_name") != "SMGTTiny":
        raise ValueError(f"{checkpoint_path}: checkpoint model must be SMGTTiny")
    from atlas3r.models.smgt.tiny import SMGTTiny

    config = SMGTTinyConfig.from_json(cast(dict[str, Any], model_config["config"]))
    model = SMGTTiny(config).to(resolved_device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path}: model_state_dict must be a mapping")
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            f"{checkpoint_path}: incompatible SMGTTiny state dict; "
            f"missing={tuple(load_result.missing_keys)}, "
            f"unexpected={tuple(load_result.unexpected_keys)}"
        )
    model.eval()
    return SMGTTinyCheckpoint(
        model=model, checkpoint=cast(dict[str, Any], payload), device=resolved_device
    )


def _validate_truth_boundary(path: Path, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing truth_boundary")
    truth = dict(cast(dict[str, object], value))
    required_false = (
        "shape_only",
        "usable_for_mapping",
        "rgb_only_mapping_ready",
        "realtime_claim",
        "accuracy_report",
        "performance_report",
        "hidden_geometry_measured",
        "teacher_geometry_used",
        "measured_depth_used",
        "measured_pose_used",
        "final_smgt",
    )
    for key in required_false:
        if truth.get(key) is not False:
            raise ValueError(f"{path}: truth_boundary.{key} must be false")
    for key in ("diagnostic_only", "student_rgb_only_used", "learned_inference", "observed_only"):
        if truth.get(key) is not True:
            raise ValueError(f"{path}: truth_boundary.{key} must be true")
    if truth.get("metric_scale_source") != "student_rgb_prior_unverified":
        raise ValueError(f"{path}: truth_boundary.metric_scale_source is not conservative")
    return truth


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SMGTTinyCheckpoint",
    "load_smgt_tiny_checkpoint",
    "save_smgt_tiny_checkpoint",
    "smgt_tiny_truth_boundary",
]
