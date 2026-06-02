"""Dataset adapters, unified data schemas, and deterministic fixtures."""

from typing import Any

from atlas3r.data.frame_source import (
    FRAME_SOURCE_SMOKE_NPZ,
    PPM_INTRINSICS_SIDECAR,
    NPZFrameSource,
    PPMSequenceFrameSource,
    RGBFrameSource,
    load_npz_clip_frames,
    load_ppm_sequence_frames,
    write_frame_source_smoke_fixture,
)
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
    if name == "student_clip_from_frame_packets":
        from atlas3r.data.student_clip import student_clip_from_frame_packets

        return student_clip_from_frame_packets
    if name == "teacher_frame_batch_from_frame_packets":
        from atlas3r.data.teacher_batch import teacher_frame_batch_from_frame_packets

        return teacher_frame_batch_from_frame_packets
    if name == "teacher_frame_batch_from_rgb_source":
        from atlas3r.data.teacher_batch import teacher_frame_batch_from_rgb_source

        return teacher_frame_batch_from_rgb_source
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AxisAlignedBox",
    "FRAME_SOURCE_SMOKE_NPZ",
    "NPZFrameSource",
    "PPMSequenceFrameSource",
    "PPM_INTRINSICS_SIDECAR",
    "RGBFrameSource",
    "SyntheticCubeRoomFrame",
    "SyntheticCubeRoomScene",
    "create_synthetic_cube_room_scene",
    "depth_observation_from_synthetic_frame",
    "load_npz_clip_frames",
    "load_ppm_sequence_frames",
    "student_clip_from_frame_packets",
    "teacher_frame_batch_from_frame_packets",
    "teacher_frame_batch_from_rgb_source",
    "write_frame_source_smoke_fixture",
    "write_synthetic_cube_room_session",
]
