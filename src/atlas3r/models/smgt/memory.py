"""Streaming memory components for SMGT-tiny."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_NN: Any = _TORCH.nn


@dataclass(frozen=True)
class SMGTTinyMemoryState:
    """Recurrent ConvGRU state carried between streaming inference calls."""

    hidden: Any

    def detach(self) -> SMGTTinyMemoryState:
        return SMGTTinyMemoryState(hidden=self.hidden.detach())


class ConvGRUCell(_NN.Module):  # type: ignore[misc]
    """Small convolutional GRU cell for feature-map memory."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.update = _NN.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.reset = _NN.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.candidate = _NN.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, current: Any, hidden: Any | None) -> Any:
        if hidden is None:
            hidden = current.new_zeros(
                (current.shape[0], self.hidden_dim, current.shape[2], current.shape[3])
            )
        stacked = _TORCH.cat([current, hidden], dim=1)
        update_gate = _TORCH.sigmoid(self.update(stacked))
        reset_gate = _TORCH.sigmoid(self.reset(stacked))
        candidate = _TORCH.tanh(self.candidate(_TORCH.cat([current, reset_gate * hidden], dim=1)))
        return (1.0 - update_gate) * hidden + update_gate * candidate


__all__ = ["ConvGRUCell", "SMGTTinyMemoryState"]
