"""Synthetic fixture conversions into mapper observation contracts."""

from __future__ import annotations

import numpy as np

from atlas3r.data.synthetic_cube_room import SyntheticCubeRoomFrame
from atlas3r.mapping.observations import DepthObservation


def depth_observation_from_synthetic_frame(frame: SyntheticCubeRoomFrame) -> DepthObservation:
    """Convert a deterministic synthetic frame into the public mapper contract."""
    return DepthObservation(
        frame_id=frame.frame_id,
        camera=frame.camera,
        pose=frame.pose,
        depth_m=frame.depth_m,
        depth_sigma_m=frame.depth_sigma_m,
        confidence=frame.confidence,
        static_mask=np.ones(frame.depth_m.shape, dtype=np.bool_),
        object_id=frame.object_id,
        source="synthetic_cube_room",
    )


__all__ = [
    "depth_observation_from_synthetic_frame",
]
