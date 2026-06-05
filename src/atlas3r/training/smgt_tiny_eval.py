"""Evaluation helpers for SMGT-tiny training loops."""

from __future__ import annotations

from typing import Any

from atlas3r.training.smgt_tiny_losses import SMGTTinyLossConfig, smgt_tiny_loss
from atlas3r.training.torch_runtime import require_torch


def evaluate_smgt_tiny(
    model: Any,
    loader: Any,
    *,
    device: str,
    loss_config: SMGTTinyLossConfig,
) -> dict[str, float]:
    """Return mean validation metrics for a SMGT-tiny dataloader."""

    torch = require_torch()
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["K"])
            _loss, metrics = smgt_tiny_loss(prediction, batch, config=loss_config)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    if batches <= 0:
        raise ValueError("validation: no batches were produced")
    return {key: value / float(batches) for key, value in totals.items()}


def move_batch_to_device(value: Any, device: str) -> Any:
    torch = require_torch()
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


__all__ = ["evaluate_smgt_tiny", "move_batch_to_device"]
