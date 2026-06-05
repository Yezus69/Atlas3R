"""Public Atlas3R Offline World Builder contracts."""

from atlas3r.contracts.coordinates import (
    CAMERA_FRAME,
    COORDINATE_FRAME_NAME,
    WORLD_FRAME_NAME,
    invert_T_A_B,
    project_points,
    transform_points,
    unproject_depth,
    validate_T_A_B,
)
from atlas3r.contracts.frames import CameraModel, FramePacket
from atlas3r.contracts.geometry import (
    DepthProposal,
    MapArtifact,
    ObjectMaskProposal,
    PoseEstimate,
    TeacherProposal,
    WorldState,
)
from atlas3r.contracts.truth import TruthBoundary, TruthLabelType

__all__ = [
    "CAMERA_FRAME",
    "COORDINATE_FRAME_NAME",
    "WORLD_FRAME_NAME",
    "CameraModel",
    "DepthProposal",
    "FramePacket",
    "MapArtifact",
    "ObjectMaskProposal",
    "PoseEstimate",
    "TeacherProposal",
    "TruthBoundary",
    "TruthLabelType",
    "WorldState",
    "invert_T_A_B",
    "project_points",
    "transform_points",
    "unproject_depth",
    "validate_T_A_B",
]
