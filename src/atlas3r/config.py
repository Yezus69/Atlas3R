"""Robot collision-envelope configuration for Atlas3R.

The primary robot output -- ``VoxelOccupancyGrid3D`` -- is bounded to the robot's
vertical collision envelope. These values are CONFIG (a robot spec), not magic
numbers buried in code: they are loaded from ``configs/robot_envelope.json`` when
present, otherwise the documented defaults below are used. Units are meters.

The collision band spans ``[floor, floor + collision_height_m + margin_m]`` along
the estimated floor axis. Resolution is spent only inside that envelope; the
teacher never models the ceiling or the full room volume (the robot does not
collide with it).

This module imports nothing heavy; it is safe to import at package import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

# Documented defaults: a standard floor-cleaning robot. Overridable per-deployment
# via ``configs/robot_envelope.json``.
DEFAULT_COLLISION_HEIGHT_M = 0.20
DEFAULT_MARGIN_M = 0.05
DEFAULT_VOXEL_SIZE_M = 0.05

DEFAULT_CONFIG_PATH = Path("configs/robot_envelope.json")


class RobotEnvelopeConfigError(ValueError):
    """Raised when a robot-envelope config file is present but malformed.

    A *missing* config file is not an error (documented defaults are used); a
    *present but invalid* file is surfaced rather than silently ignored.
    """


@dataclass(frozen=True)
class RobotEnvelopeConfig:
    """The robot's vertical collision envelope + fusion resolution (meters)."""

    collision_height_m: float = DEFAULT_COLLISION_HEIGHT_M
    margin_m: float = DEFAULT_MARGIN_M
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M

    def __post_init__(self) -> None:
        if not _positive(self.collision_height_m):
            raise RobotEnvelopeConfigError("collision_height_m must be a positive finite number")
        if not _non_negative(self.margin_m):
            raise RobotEnvelopeConfigError("margin_m must be a non-negative finite number")
        if not _positive(self.voxel_size_m):
            raise RobotEnvelopeConfigError("voxel_size_m must be a positive finite number")

    @property
    def band_height_m(self) -> float:
        """Total vertical extent of the collision band above the floor."""
        return float(self.collision_height_m) + float(self.margin_m)

    def to_dict(self) -> dict[str, float]:
        return {
            "collision_height_m": float(self.collision_height_m),
            "margin_m": float(self.margin_m),
            "voxel_size_m": float(self.voxel_size_m),
            "band_height_m": self.band_height_m,
        }


def load_robot_envelope(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> tuple[RobotEnvelopeConfig, dict[str, Any]]:
    """Load a :class:`RobotEnvelopeConfig`, falling back to documented defaults.

    Returns ``(config, provenance)``. ``provenance`` records whether the file was
    present so the teacher report can state which envelope was used. A missing
    file yields the defaults; a present-but-malformed file raises
    :class:`RobotEnvelopeConfigError` (never a silent fallback).
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if root is not None and not target.is_absolute():
        target = Path(root) / target

    if not target.is_file():
        config = RobotEnvelopeConfig()
        return config, {
            "source": "defaults",
            "path": str(target),
            "present": False,
            **config.to_dict(),
        }

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobotEnvelopeConfigError(
            f"robot envelope config unreadable at {target}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RobotEnvelopeConfigError(
            f"robot envelope config must be a JSON object at {target}"
        )

    try:
        config = RobotEnvelopeConfig(
            collision_height_m=float(data.get("collision_height_m", DEFAULT_COLLISION_HEIGHT_M)),
            margin_m=float(data.get("margin_m", DEFAULT_MARGIN_M)),
            voxel_size_m=float(data.get("voxel_size_m", DEFAULT_VOXEL_SIZE_M)),
        )
    except (TypeError, ValueError) as exc:
        raise RobotEnvelopeConfigError(
            f"robot envelope config has invalid values at {target}: {exc}"
        ) from exc

    return config, {
        "source": "file",
        "path": str(target),
        "present": True,
        **config.to_dict(),
    }


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and isfinite(value) and value > 0


def _non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and isfinite(value) and value >= 0


__all__ = [
    "RobotEnvelopeConfig",
    "RobotEnvelopeConfigError",
    "load_robot_envelope",
    "DEFAULT_CONFIG_PATH",
]
