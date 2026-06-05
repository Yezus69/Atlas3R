"""Geometry, teacher proposal, world-state, and map-artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts._arrays import (
    array_to_list,
    as_float,
    as_float32_array,
    as_int,
    as_uint8_array,
    require_probability,
)
from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME, validate_T_A_B
from atlas3r.contracts.frames import CameraModel
from atlas3r.contracts.truth import TruthBoundary

TrackingState = Literal["OK", "LOW_CONFIDENCE", "RELOCALIZING", "LOST", "NEW_SUBMAP"]
MapArtifactType = Literal["mesh", "voxel", "occupancy", "training_cache", "quality_report"]


@dataclass(frozen=True)
class PoseEstimate:
    frame_id: int
    timestamp_ns: int
    T_world_camera: NDArray[np.float32]
    covariance_6x6: NDArray[np.float32] | None
    confidence: float
    tracking_state: TrackingState
    scale_source: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame_id and timestamp_ns must be non-negative")
        validate_T_A_B(self.T_world_camera, "T_world_camera")
        if self.covariance_6x6 is not None:
            cov = as_float32_array(self.covariance_6x6, (6, 6), "covariance_6x6")
            if np.any(np.diag(cov) < 0):
                raise ValueError("covariance_6x6 diagonal must be non-negative")
        require_probability(float(self.confidence), "confidence")
        if self.tracking_state not in {
            "OK",
            "LOW_CONFIDENCE",
            "RELOCALIZING",
            "LOST",
            "NEW_SUBMAP",
        }:
            raise ValueError(f"unknown tracking_state: {self.tracking_state}")
        if not self.scale_source:
            raise ValueError("scale_source must be non-empty")

    @property
    def camera_center_world_m(self) -> NDArray[np.float32]:
        return self.T_world_camera[:3, 3].astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "T_world_camera": array_to_list(self.T_world_camera),
            "camera_center_world_m": array_to_list(self.camera_center_world_m),
            "covariance_6x6": array_to_list(self.covariance_6x6),
            "confidence": self.confidence,
            "tracking_state": self.tracking_state,
            "scale_source": self.scale_source,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PoseEstimate:
        tracking_state = str(data.get("tracking_state", "LOW_CONFIDENCE"))
        return cls(
            frame_id=as_int(data["frame_id"], "frame_id"),
            timestamp_ns=as_int(data["timestamp_ns"], "timestamp_ns"),
            T_world_camera=validate_T_A_B(data["T_world_camera"], "T_world_camera"),
            covariance_6x6=(
                None
                if data.get("covariance_6x6") is None
                else as_float32_array(data["covariance_6x6"], (6, 6), "covariance_6x6")
            ),
            confidence=as_float(data.get("confidence", 0.0), "confidence"),
            tracking_state=cast(TrackingState, tracking_state),
            scale_source=str(data.get("scale_source", "unknown")),
            diagnostics=_mapping(data.get("diagnostics", {}), "diagnostics"),
        )


@dataclass(frozen=True)
class DepthProposal:
    frame_id: int
    teacher_name: str
    camera: CameraModel
    pose: PoseEstimate | None
    depth_m: NDArray[np.float32]
    depth_sigma_m: NDArray[np.float32]
    confidence: NDArray[np.float32]
    truth_boundary: TruthBoundary
    source: str = "teacher"

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.teacher_name:
            raise ValueError("teacher_name must be non-empty")
        depth = as_float32_array(self.depth_m, (self.camera.height, self.camera.width), "depth_m")
        sigma = as_float32_array(
            self.depth_sigma_m, (self.camera.height, self.camera.width), "depth_sigma_m"
        )
        confidence = as_float32_array(
            self.confidence, (self.camera.height, self.camera.width), "confidence"
        )
        if np.any(depth < 0) or np.any(sigma < 0):
            raise ValueError("depth_m and depth_sigma_m must be non-negative")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence must be in [0, 1]")
        if not self.source:
            raise ValueError("source must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "teacher_name": self.teacher_name,
            "camera": self.camera.to_dict(),
            "pose": None if self.pose is None else self.pose.to_dict(),
            "depth_m": array_to_list(self.depth_m),
            "depth_sigma_m": array_to_list(self.depth_sigma_m),
            "confidence": array_to_list(self.confidence),
            "truth_boundary": self.truth_boundary.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DepthProposal:
        camera = CameraModel.from_dict(_mapping(data["camera"], "camera"))
        return cls(
            frame_id=as_int(data["frame_id"], "frame_id"),
            teacher_name=str(data["teacher_name"]),
            camera=camera,
            pose=(
                None
                if data.get("pose") is None
                else PoseEstimate.from_dict(_mapping(data["pose"], "pose"))
            ),
            depth_m=as_float32_array(data["depth_m"], (camera.height, camera.width), "depth_m"),
            depth_sigma_m=as_float32_array(
                data["depth_sigma_m"], (camera.height, camera.width), "depth_sigma_m"
            ),
            confidence=as_float32_array(
                data["confidence"], (camera.height, camera.width), "confidence"
            ),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
            source=str(data.get("source", "teacher")),
        )


@dataclass(frozen=True)
class ObjectMaskProposal:
    frame_id: int
    teacher_name: str
    mask_u8: NDArray[np.uint8]
    confidence: float
    label: str
    object_id: int | None
    truth_boundary: TruthBoundary

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.teacher_name:
            raise ValueError("teacher_name must be non-empty")
        mask = as_uint8_array(self.mask_u8, (None, None), "mask_u8")
        if mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError("mask_u8 must have positive height and width")
        require_probability(float(self.confidence), "confidence")
        if not self.label:
            raise ValueError("label must be non-empty")
        if self.object_id is not None and self.object_id < 0:
            raise ValueError("object_id must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "teacher_name": self.teacher_name,
            "mask_u8": array_to_list(self.mask_u8),
            "confidence": self.confidence,
            "label": self.label,
            "object_id": self.object_id,
            "truth_boundary": self.truth_boundary.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ObjectMaskProposal:
        return cls(
            frame_id=as_int(data["frame_id"], "frame_id"),
            teacher_name=str(data["teacher_name"]),
            mask_u8=as_uint8_array(data["mask_u8"], (None, None), "mask_u8"),
            confidence=as_float(data.get("confidence", 0.0), "confidence"),
            label=str(data.get("label", "unknown")),
            object_id=None
            if data.get("object_id") is None
            else as_int(data["object_id"], "object_id"),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
        )


@dataclass(frozen=True)
class TeacherProposal:
    teacher_name: str
    frame_ids: tuple[int, ...]
    depth_proposals: tuple[DepthProposal, ...]
    object_masks: tuple[ObjectMaskProposal, ...]
    truth_boundary: TruthBoundary
    status: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.teacher_name:
            raise ValueError("teacher_name must be non-empty")
        if any(frame_id < 0 for frame_id in self.frame_ids):
            raise ValueError("frame_ids must be non-negative")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        if not self.status:
            raise ValueError("status must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "teacher_name": self.teacher_name,
            "frame_ids": list(self.frame_ids),
            "depth_proposals": [proposal.to_dict() for proposal in self.depth_proposals],
            "object_masks": [mask.to_dict() for mask in self.object_masks],
            "truth_boundary": self.truth_boundary.to_dict(),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TeacherProposal:
        return cls(
            teacher_name=str(data["teacher_name"]),
            frame_ids=tuple(
                as_int(item, "frame_id") for item in _sequence(data["frame_ids"], "frame_ids")
            ),
            depth_proposals=tuple(
                DepthProposal.from_dict(_mapping(item, "depth_proposal"))
                for item in _sequence(data.get("depth_proposals", ()), "depth_proposals")
            ),
            object_masks=tuple(
                ObjectMaskProposal.from_dict(_mapping(item, "object_mask"))
                for item in _sequence(data.get("object_masks", ()), "object_masks")
            ),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
            status=str(data.get("status", "unknown")),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class WorldState:
    world_id: str
    poses: tuple[PoseEstimate, ...]
    cameras: tuple[CameraModel, ...]
    teacher_proposals: tuple[TeacherProposal, ...]
    map_artifacts: tuple[MapArtifact, ...]
    truth_boundary: TruthBoundary
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.world_id:
            raise ValueError("world_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "world_id": self.world_id,
            "poses": [pose.to_dict() for pose in self.poses],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "teacher_proposals": [proposal.to_dict() for proposal in self.teacher_proposals],
            "map_artifacts": [artifact.to_dict() for artifact in self.map_artifacts],
            "truth_boundary": self.truth_boundary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WorldState:
        return cls(
            world_id=str(data["world_id"]),
            poses=tuple(
                PoseEstimate.from_dict(_mapping(item, "pose"))
                for item in _sequence(data.get("poses", ()), "poses")
            ),
            cameras=tuple(
                CameraModel.from_dict(_mapping(item, "camera"))
                for item in _sequence(data.get("cameras", ()), "cameras")
            ),
            teacher_proposals=tuple(
                TeacherProposal.from_dict(_mapping(item, "teacher_proposal"))
                for item in _sequence(data.get("teacher_proposals", ()), "teacher_proposals")
            ),
            map_artifacts=tuple(
                MapArtifact.from_dict(_mapping(item, "map_artifact"))
                for item in _sequence(data.get("map_artifacts", ()), "map_artifacts")
            ),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class MapArtifact:
    artifact_type: MapArtifactType
    path: str
    coordinate_frame: str
    source_frame_ids: tuple[int, ...]
    voxel_size_m: float | None
    observed_coverage_estimate: float
    mean_uncertainty_m: float
    p95_uncertainty_m: float
    truth_boundary: TruthBoundary
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.artifact_type not in {
            "mesh",
            "voxel",
            "occupancy",
            "training_cache",
            "quality_report",
        }:
            raise ValueError(f"unknown artifact_type: {self.artifact_type}")
        if not self.path or Path(self.path).is_absolute():
            raise ValueError("path must be a non-empty relative path")
        if self.coordinate_frame != COORDINATE_FRAME_NAME:
            raise ValueError(f"coordinate_frame must be {COORDINATE_FRAME_NAME}")
        if any(frame_id < 0 for frame_id in self.source_frame_ids):
            raise ValueError("source_frame_ids must be non-negative")
        if self.voxel_size_m is not None and self.voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        require_probability(float(self.observed_coverage_estimate), "observed_coverage_estimate")
        if self.mean_uncertainty_m < 0 or self.p95_uncertainty_m < 0:
            raise ValueError("uncertainty values must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "coordinate_frame": self.coordinate_frame,
            "source_frame_ids": list(self.source_frame_ids),
            "voxel_size_m": self.voxel_size_m,
            "observed_coverage_estimate": self.observed_coverage_estimate,
            "mean_uncertainty_m": self.mean_uncertainty_m,
            "p95_uncertainty_m": self.p95_uncertainty_m,
            "truth_boundary": self.truth_boundary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MapArtifact:
        artifact_type = str(data["artifact_type"])
        return cls(
            artifact_type=cast(MapArtifactType, artifact_type),
            path=str(data["path"]),
            coordinate_frame=str(data.get("coordinate_frame", COORDINATE_FRAME_NAME)),
            source_frame_ids=tuple(
                as_int(item, "source_frame_id")
                for item in _sequence(data.get("source_frame_ids", ()), "source_frame_ids")
            ),
            voxel_size_m=(
                None
                if data.get("voxel_size_m") is None
                else as_float(data["voxel_size_m"], "voxel_size_m")
            ),
            observed_coverage_estimate=as_float(
                data.get("observed_coverage_estimate", 0.0), "observed_coverage_estimate"
            ),
            mean_uncertainty_m=as_float(data.get("mean_uncertainty_m", 0.0), "mean_uncertainty_m"),
            p95_uncertainty_m=as_float(data.get("p95_uncertainty_m", 0.0), "p95_uncertainty_m"),
            truth_boundary=TruthBoundary.from_dict(
                _mapping(data["truth_boundary"], "truth_boundary")
            ),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return value
