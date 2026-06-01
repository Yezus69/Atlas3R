"""Dataset adapters, unified data schemas, and deterministic fixtures."""

from typing import Any

from atlas3r.data.synthetic_cube_room import (
    AxisAlignedBox,
    SyntheticCubeRoomFrame,
    SyntheticCubeRoomScene,
    create_synthetic_cube_room_scene,
    write_synthetic_cube_room_session,
)


def __getattr__(name: str) -> Any:
    if name == "depth_observation_from_synthetic_frame":
        from atlas3r.data.synthetic_observations import depth_observation_from_synthetic_frame

        return depth_observation_from_synthetic_frame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AxisAlignedBox",
    "SyntheticCubeRoomFrame",
    "SyntheticCubeRoomScene",
    "create_synthetic_cube_room_scene",
    "depth_observation_from_synthetic_frame",
    "write_synthetic_cube_room_session",
]
