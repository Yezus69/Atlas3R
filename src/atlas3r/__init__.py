"""Runtime contracts and teacher spine for Atlas3R.

The contract dataclasses keep the package import dependency-free. The teacher
spine functions (M3 -> M8) lazy-import numpy/cv2/PIL inside their own bodies, so
importing :mod:`atlas3r` never pulls heavy ML dependencies.
"""

from .config import (
    RobotEnvelopeConfig,
    RobotEnvelopeConfigError,
    load_robot_envelope,
)
from .contracts import (
    BackboneAvailability,
    CameraModel,
    ContractValidationError,
    DepthConvention,
    ExternalBackboneUnavailableError,
    ExternalDependencyStatus,
    FrameRayPacket,
    GeometryBackbonePrediction,
    KeyframeProposal,
    KeyframeSet,
    MaskTrack,
    MaskTrackDecision,
    MaskTrackSet,
    MeshChunkMetadata,
    MetricAcceptanceStatus,
    OccupancyChannel,
    OccupancyGrid2D,
    OptimizedSceneState,
    ReconstructabilityReport,
    ScaleEvidence,
    ScaleEvidenceType,
    ScalePosterior,
    StaticDynamicLabel,
    StaticDynamicState,
    TrackType,
    UnavailableVideoGeometryBackbone,
    ValidationReport,
    VideoAsset,
    VideoAssetStatus,
    VideoGeometryBackbone,
    VideoInput,
    VideoInspectionReport,
    VisibilityEdge,
    VisibilityGraph,
    VoxelMapState,
    VoxelOccupancyGrid3D,
)
from .export import export_teacher_artifacts
from .geometry_adapter import (
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from .mapping import fuse_static_map
from .refine import refine_scene
from .scale import estimate_scale_posterior
from .static_dynamic import infer_static_dynamic
from .teacher import run_teacher
from .validation import validate_and_accept
from .visibility import build_visibility_graph
from .visualize import write_visual_proof

__all__ = [
    "RobotEnvelopeConfig",
    "RobotEnvelopeConfigError",
    "load_robot_envelope",
    "BackboneAvailability",
    "CameraModel",
    "ContractValidationError",
    "DepthConvention",
    "ExternalBackboneUnavailableError",
    "ExternalDependencyStatus",
    "FrameRayPacket",
    "GeometryBackbonePrediction",
    "KeyframeProposal",
    "KeyframeSet",
    "MaskTrack",
    "MaskTrackDecision",
    "MaskTrackSet",
    "MeshChunkMetadata",
    "MetricAcceptanceStatus",
    "OccupancyChannel",
    "OccupancyGrid2D",
    "OptimizedSceneState",
    "ReconstructabilityReport",
    "ScaleEvidence",
    "ScaleEvidenceType",
    "ScalePosterior",
    "StaticDynamicLabel",
    "StaticDynamicState",
    "TrackType",
    "UnavailableVideoGeometryBackbone",
    "ValidationReport",
    "VideoAsset",
    "VideoAssetStatus",
    "VideoGeometryBackbone",
    "VideoInput",
    "VideoInspectionReport",
    "VisibilityEdge",
    "VisibilityGraph",
    "VoxelMapState",
    "VoxelOccupancyGrid3D",
    # Teacher spine (M3 -> M8) public entrypoints
    "load_geometry_artifacts",
    "load_measured_packets_from_m2",
    "build_visibility_graph",
    "estimate_scale_posterior",
    "infer_static_dynamic",
    "fuse_static_map",
    "refine_scene",
    "validate_and_accept",
    "run_teacher",
    "write_visual_proof",
    "export_teacher_artifacts",
]
