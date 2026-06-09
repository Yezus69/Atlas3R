"""Robot collision-envelope configuration for Atlas3R.

The primary robot output -- ``VoxelOccupancyGrid3D`` -- is bounded to the robot's
vertical collision envelope. These values are CONFIG (a robot spec), not magic
numbers buried in code: they are loaded from ``configs/robot_envelope.json`` when
present, otherwise the documented defaults below are used. Units are meters.

The collision band spans ``[floor, floor + collision_height_m + margin_m]`` along
the estimated floor axis. Resolution is spent only inside that envelope; the
teacher never models the ceiling or the full room volume (the robot does not
collide with it).

This config also carries the candidate-side occupancy-ESTIMATION policy applied
when fusing the monocular candidate map (never to the measured GT yardstick):

- ``free_carve_margin_m``: a DIRECTIONAL truncation band. Free space is retracted
  only in the column directly BELOW a confident fused surface (toward the floor)
  within this distance -- the grazing-ray flood that masks an obstacle's support
  column. Lateral free beside the obstacle is preserved. Removed-free becomes
  UNKNOWN, never occupied (honest: it only retracts an over-claim).
- ``occupancy_support_height_m``: a generic gravity/support prior. A real obstacle
  rests on the floor and occupies the whole column from its top down to the floor;
  this propagates occupancy DOWNWARD within the band by up to this height so a
  detected obstacle claims its support column (raising obstacle-base recall). It is
  HONEST: it only resolves UNKNOWN space below a CONFIDENT obstacle (occupied count
  >= ``occupancy_support_min_count``); it never overrides an observed-free voxel.
- ``occupancy_support_min_count``: minimum fused occupied-hit count for a voxel to
  act as a support SOURCE, so single-hit depth noise high in the band does not
  conjure a column of occupancy. Default 1 (any obstacle supports).
- ``occupancy_close_voxels``: in-plane morphological closing radius (voxels) that
  bridges small gaps between nearby candidate surface voxels.

All three default to OFF (0), so an unconfigured run reproduces the prior fuser
byte-for-byte. They are GENERIC (help any embodiment), config-driven, and apply
only to the candidate occupancy estimate -- the measured 3D GT is fused raw.

This module imports nothing heavy; it is safe to import at package import time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

# Documented defaults: a standard floor-cleaning robot. Overridable per-deployment
# via ``configs/robot_envelope.json``.
DEFAULT_COLLISION_HEIGHT_M = 0.20
DEFAULT_MARGIN_M = 0.05
DEFAULT_VOXEL_SIZE_M = 0.05
# Candidate occupancy-estimation policy. OFF by default -> prior fuser reproduced.
DEFAULT_FREE_CARVE_MARGIN_M = 0.0
DEFAULT_OCCUPANCY_SUPPORT_HEIGHT_M = 0.0
DEFAULT_OCCUPANCY_SUPPORT_MIN_COUNT = 1
DEFAULT_OCCUPANCY_CLOSE_VOXELS = 0

DEFAULT_CONFIG_PATH = Path("configs/robot_envelope.json")
# Optional env override pointing at an alternate config file. Lets a reversible
# sweep run many policies side by side (each process its own config + output dir)
# without mutating the committed config. Unset -> the canonical path is used.
ENV_CONFIG_PATH = "ATLAS3R_ROBOT_ENVELOPE_CONFIG"


class RobotEnvelopeConfigError(ValueError):
    """Raised when a robot-envelope config file is present but malformed.

    A *missing* config file is not an error (documented defaults are used); a
    *present but invalid* file is surfaced rather than silently ignored.
    """


@dataclass(frozen=True)
class RobotEnvelopeConfig:
    """The robot's vertical collision envelope + fusion resolution (meters).

    The last three fields are the candidate-side occupancy-estimation policy
    (see module docstring); all default to OFF so an unconfigured run reproduces
    the prior fuser exactly.
    """

    collision_height_m: float = DEFAULT_COLLISION_HEIGHT_M
    margin_m: float = DEFAULT_MARGIN_M
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M
    free_carve_margin_m: float = DEFAULT_FREE_CARVE_MARGIN_M
    occupancy_support_height_m: float = DEFAULT_OCCUPANCY_SUPPORT_HEIGHT_M
    occupancy_support_min_count: int = DEFAULT_OCCUPANCY_SUPPORT_MIN_COUNT
    occupancy_close_voxels: int = DEFAULT_OCCUPANCY_CLOSE_VOXELS

    def __post_init__(self) -> None:
        if not _positive(self.collision_height_m):
            raise RobotEnvelopeConfigError("collision_height_m must be a positive finite number")
        if not _non_negative(self.margin_m):
            raise RobotEnvelopeConfigError("margin_m must be a non-negative finite number")
        if not _positive(self.voxel_size_m):
            raise RobotEnvelopeConfigError("voxel_size_m must be a positive finite number")
        if not _non_negative(self.free_carve_margin_m):
            raise RobotEnvelopeConfigError("free_carve_margin_m must be a non-negative finite number")
        if not _non_negative(self.occupancy_support_height_m):
            raise RobotEnvelopeConfigError("occupancy_support_height_m must be a non-negative finite number")
        if not isinstance(self.occupancy_support_min_count, int) or isinstance(self.occupancy_support_min_count, bool) or self.occupancy_support_min_count < 1:
            raise RobotEnvelopeConfigError("occupancy_support_min_count must be an integer >= 1")
        if not isinstance(self.occupancy_close_voxels, int) or isinstance(self.occupancy_close_voxels, bool) or self.occupancy_close_voxels < 0:
            raise RobotEnvelopeConfigError("occupancy_close_voxels must be a non-negative integer")

    @property
    def band_height_m(self) -> float:
        """Total vertical extent of the collision band above the floor."""
        return float(self.collision_height_m) + float(self.margin_m)

    @property
    def policy_active(self) -> bool:
        """True when any candidate occupancy-estimation lever is engaged."""
        return (
            float(self.free_carve_margin_m) > 0.0
            or float(self.occupancy_support_height_m) > 0.0
            or int(self.occupancy_close_voxels) > 0
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "collision_height_m": float(self.collision_height_m),
            "margin_m": float(self.margin_m),
            "voxel_size_m": float(self.voxel_size_m),
            "band_height_m": self.band_height_m,
            "free_carve_margin_m": float(self.free_carve_margin_m),
            "occupancy_support_height_m": float(self.occupancy_support_height_m),
            "occupancy_support_min_count": int(self.occupancy_support_min_count),
            "occupancy_close_voxels": int(self.occupancy_close_voxels),
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

    When ``path`` is not given, an alternate config file may be selected via the
    ``ATLAS3R_ROBOT_ENVELOPE_CONFIG`` env var (used by reversible policy sweeps);
    unset -> the canonical ``configs/robot_envelope.json``.
    """
    env_override = os.environ.get(ENV_CONFIG_PATH) if path is None else None
    target = Path(path) if path is not None else Path(env_override) if env_override else DEFAULT_CONFIG_PATH
    if root is not None and not target.is_absolute():
        target = Path(root) / target

    if not target.is_file():
        config = RobotEnvelopeConfig()
        return config, {
            "source": "defaults",
            "path": str(target),
            "present": False,
            "env_override": env_override,
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
            free_carve_margin_m=float(data.get("free_carve_margin_m", DEFAULT_FREE_CARVE_MARGIN_M)),
            occupancy_support_height_m=float(
                data.get("occupancy_support_height_m", DEFAULT_OCCUPANCY_SUPPORT_HEIGHT_M)
            ),
            occupancy_support_min_count=int(
                data.get("occupancy_support_min_count", DEFAULT_OCCUPANCY_SUPPORT_MIN_COUNT)
            ),
            occupancy_close_voxels=int(data.get("occupancy_close_voxels", DEFAULT_OCCUPANCY_CLOSE_VOXELS)),
        )
    except (TypeError, ValueError) as exc:
        raise RobotEnvelopeConfigError(
            f"robot envelope config has invalid values at {target}: {exc}"
        ) from exc

    return config, {
        "source": "file",
        "path": str(target),
        "present": True,
        "env_override": env_override,
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
