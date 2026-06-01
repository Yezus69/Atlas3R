"""Dataset adapters, unified data schemas, and deterministic fixtures."""

from atlas3r.data.synthetic_cube_room import (
    AxisAlignedBox,
    SyntheticCubeRoomFrame,
    SyntheticCubeRoomScene,
    create_synthetic_cube_room_scene,
    write_synthetic_cube_room_session,
)

__all__ = [
    "AxisAlignedBox",
    "SyntheticCubeRoomFrame",
    "SyntheticCubeRoomScene",
    "create_synthetic_cube_room_scene",
    "write_synthetic_cube_room_session",
]
