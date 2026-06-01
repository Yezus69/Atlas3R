"""Deterministic synthetic cube-room scene for Phase 0 smoke tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import (
    CameraModel,
    MeshChunk,
    ObjectInstance,
    PoseEstimate,
    ScaleSource,
    SurfaceSource,
    TrackingState,
    WorldMap,
)
from atlas3r.camera.pinhole import unproject_pixels
from atlas3r.pose.transforms import (
    make_transform,
    quaternion_xyzw_from_rotation_matrix,
    transform_points,
)

FLOAT32 = np.float32
FLOAT64 = np.float64


@dataclass(frozen=True)
class AxisAlignedBox:
    """Axis-aligned bounds in the synthetic world frame."""

    min_corner_m: npt.NDArray[np.float64]
    max_corner_m: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticCubeRoomFrame:
    """One rendered synthetic frame and its ground-truth arrays."""

    frame_id: int
    timestamp_ns: int
    camera: CameraModel
    pose: PoseEstimate
    depth_m: npt.NDArray[np.float32]
    depth_sigma_m: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    object_mask: npt.NDArray[np.bool_]
    object_id: npt.NDArray[np.int32]
    point_world_m: npt.NDArray[np.float32]


@dataclass(frozen=True)
class SyntheticCubeRoomScene:
    """Complete deterministic cube-room fixture."""

    camera: CameraModel
    frames: tuple[SyntheticCubeRoomFrame, ...]
    room_bounds_m: AxisAlignedBox
    object_bounds_m: AxisAlignedBox
    mesh_chunk: MeshChunk
    objects: dict[int, ObjectInstance]
    world_map: WorldMap


def create_synthetic_cube_room_scene() -> SyntheticCubeRoomScene:
    """Create the fixed Phase 0B synthetic cube-room scene."""
    width = 32
    height = 24
    K = np.array(
        [
            [30.0, 0.0, (width - 1.0) / 2.0],
            [0.0, 30.0, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=FLOAT32,
    )
    camera = CameraModel(
        width=width,
        height=height,
        K=K,
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source="synthetic_calibrated",
    )
    room_bounds = AxisAlignedBox(
        min_corner_m=np.array([-1.5, -1.0, 0.0], dtype=FLOAT64),
        max_corner_m=np.array([1.5, 1.0, 4.0], dtype=FLOAT64),
    )
    object_bounds = AxisAlignedBox(
        min_corner_m=np.array([-0.35, -0.25, 1.85], dtype=FLOAT64),
        max_corner_m=np.array([0.35, 0.45, 2.55], dtype=FLOAT64),
    )
    camera_centers = (
        np.array([-0.25, 0.0, 0.4], dtype=FLOAT64),
        np.array([0.25, 0.0, 0.4], dtype=FLOAT64),
        np.array([0.0, -0.1, 0.55], dtype=FLOAT64),
    )
    poses = tuple(
        _make_pose(frame_id=index, timestamp_ns=index * 33_333_333, center_world_m=center)
        for index, center in enumerate(camera_centers)
    )
    frames = tuple(
        _render_frame(
            camera=camera, pose=pose, room_bounds=room_bounds, object_bounds=object_bounds
        )
        for pose in poses
    )
    mesh_chunk = _make_mesh_chunk(room_bounds, object_bounds, [frame.frame_id for frame in frames])
    objects = {1: _make_object_instance(object_bounds, mesh_chunk.chunk_id, frames)}
    world_map = WorldMap(
        map_id="synthetic_cube_room",
        world_frame_name="synthetic_world",
        created_at_ns=0,
        mesh_chunks={mesh_chunk.chunk_id: mesh_chunk},
        objects=objects,
        keyframes={pose.frame_id: pose for pose in poses},
        scale_source=ScaleSource.KNOWN_ANCHOR.value,
        global_confidence=1.0,
        metadata={
            "scene": "synthetic_cube_room",
            "deterministic": True,
            "unit_scale": "meters",
            "coordinate_convention": _coordinate_convention(),
        },
    )
    return SyntheticCubeRoomScene(
        camera=camera,
        frames=frames,
        room_bounds_m=room_bounds,
        object_bounds_m=object_bounds,
        mesh_chunk=mesh_chunk,
        objects=objects,
        world_map=world_map,
    )


def write_synthetic_cube_room_session(output_folder: str | Path) -> Path:
    """Write a tiny deterministic `.atlas3r` session folder."""
    output_path = Path(output_folder)
    scene = create_synthetic_cube_room_scene()
    depth_folder = output_path / "depth"
    mesh_folder = output_path / "mesh_chunks"
    logs_folder = output_path / "logs"
    depth_folder.mkdir(parents=True, exist_ok=True)
    mesh_folder.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    _write_json(output_path / "metadata.json", _metadata_record(scene))
    _write_jsonl(output_path / "poses.jsonl", (_pose_record(frame.pose) for frame in scene.frames))
    _write_jsonl(
        output_path / "cameras.jsonl",
        (
            {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                **_camera_record(frame.camera),
            }
            for frame in scene.frames
        ),
    )
    _write_jsonl(
        output_path / "objects.jsonl",
        (_object_record(instance) for instance in scene.objects.values()),
    )
    mesh_json = mesh_folder / f"chunk_{scene.mesh_chunk.chunk_id}_v{scene.mesh_chunk.version}.json"
    _write_json(mesh_json, _mesh_chunk_record(scene.mesh_chunk))
    _write_json(
        mesh_folder / "index.json",
        {
            "mesh_chunks": [
                {
                    "chunk_id": scene.mesh_chunk.chunk_id,
                    "version": scene.mesh_chunk.version,
                    "metadata_path": mesh_json.name,
                }
            ]
        },
    )
    for frame in scene.frames:
        np.savez(
            depth_folder / f"frame_{frame.frame_id:06d}.npz",
            depth_m=frame.depth_m,
            depth_sigma_m=frame.depth_sigma_m,
            confidence=frame.confidence,
            object_mask=frame.object_mask,
            object_id=frame.object_id,
        )
    _write_json(
        logs_folder / "runtime_profile.json",
        {
            "scene": "synthetic_cube_room",
            "runtime_capture": False,
            "neural_models": False,
            "frame_count": len(scene.frames),
        },
    )
    return output_path


def _make_pose(
    *, frame_id: int, timestamp_ns: int, center_world_m: npt.NDArray[np.float64]
) -> PoseEstimate:
    R_world_camera = np.eye(3, dtype=FLOAT64)
    T_world_camera = make_transform(R_world_camera, center_world_m)
    q_world_camera_xyzw = quaternion_xyzw_from_rotation_matrix(R_world_camera)
    return PoseEstimate(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        T_world_camera=T_world_camera.astype(FLOAT32),
        q_world_camera_xyzw=q_world_camera_xyzw.astype(FLOAT32),
        camera_center_world_m=center_world_m.astype(FLOAT32),
        covariance_6x6=(np.eye(6, dtype=FLOAT32) * FLOAT32(1e-6)),
        confidence=1.0,
        tracking_state=TrackingState.OK.value,
        scale_source=ScaleSource.KNOWN_ANCHOR.value,
        diagnostics={"synthetic": True},
    )


def _render_frame(
    *,
    camera: CameraModel,
    pose: PoseEstimate,
    room_bounds: AxisAlignedBox,
    object_bounds: AxisAlignedBox,
) -> SyntheticCubeRoomFrame:
    pixels_uv = _pixel_grid(camera.width, camera.height)
    unit_depth = np.ones((pixels_uv.shape[0],), dtype=FLOAT64)
    rays_camera = unproject_pixels(pixels_uv, unit_depth, camera.K)
    T_world_camera = pose.T_world_camera.astype(FLOAT64, copy=False)
    origin_world = T_world_camera[:3, 3]
    rays_world = (T_world_camera[:3, :3] @ rays_camera.T).T
    depth = np.empty((pixels_uv.shape[0],), dtype=FLOAT64)
    object_mask = np.zeros((pixels_uv.shape[0],), dtype=bool)
    object_id = np.zeros((pixels_uv.shape[0],), dtype=np.int32)
    for index, ray_world in enumerate(rays_world):
        room_depth = _intersect_box_exit(origin_world, ray_world, room_bounds)
        object_depth = _intersect_box_enter(origin_world, ray_world, object_bounds)
        if object_depth is not None and object_depth < room_depth:
            depth[index] = object_depth
            object_mask[index] = True
            object_id[index] = 1
        else:
            depth[index] = room_depth
    points_camera = rays_camera * depth[:, None]
    points_world = transform_points(T_world_camera, points_camera)
    shape = (camera.height, camera.width)
    return SyntheticCubeRoomFrame(
        frame_id=pose.frame_id,
        timestamp_ns=pose.timestamp_ns,
        camera=camera,
        pose=pose,
        depth_m=depth.reshape(shape).astype(FLOAT32),
        depth_sigma_m=np.full(shape, 0.002, dtype=FLOAT32),
        confidence=np.ones(shape, dtype=FLOAT32),
        object_mask=object_mask.reshape(shape),
        object_id=object_id.reshape(shape),
        point_world_m=points_world.reshape(camera.height, camera.width, 3).astype(FLOAT32),
    )


def _pixel_grid(width: int, height: int) -> npt.NDArray[np.float64]:
    u, v = np.meshgrid(np.arange(width, dtype=FLOAT64), np.arange(height, dtype=FLOAT64))
    return np.column_stack([u.reshape(-1), v.reshape(-1)])


def _intersect_box_enter(
    origin_world: npt.NDArray[np.float64],
    direction_world: npt.NDArray[np.float64],
    bounds: AxisAlignedBox,
) -> float | None:
    enter_depth, exit_depth = _intersect_box(origin_world, direction_world, bounds)
    if exit_depth <= 1e-9:
        return None
    if enter_depth <= 1e-9:
        return None
    return enter_depth


def _intersect_box_exit(
    origin_world: npt.NDArray[np.float64],
    direction_world: npt.NDArray[np.float64],
    bounds: AxisAlignedBox,
) -> float:
    enter_depth, exit_depth = _intersect_box(origin_world, direction_world, bounds)
    if enter_depth > 1e-9:
        return enter_depth
    if exit_depth <= 1e-9:
        raise RuntimeError("camera ray does not hit the synthetic room")
    return exit_depth


def _intersect_box(
    origin_world: npt.NDArray[np.float64],
    direction_world: npt.NDArray[np.float64],
    bounds: AxisAlignedBox,
) -> tuple[float, float]:
    enter_depth = -np.inf
    exit_depth = np.inf
    for axis in range(3):
        direction = float(direction_world[axis])
        origin = float(origin_world[axis])
        min_value = float(bounds.min_corner_m[axis])
        max_value = float(bounds.max_corner_m[axis])
        if abs(direction) < 1e-12:
            if origin < min_value or origin > max_value:
                return np.inf, -np.inf
            continue
        first = (min_value - origin) / direction
        second = (max_value - origin) / direction
        near_depth = min(first, second)
        far_depth = max(first, second)
        enter_depth = max(enter_depth, near_depth)
        exit_depth = min(exit_depth, far_depth)
        if enter_depth > exit_depth:
            return np.inf, -np.inf
    return float(enter_depth), float(exit_depth)


def _make_mesh_chunk(
    room_bounds: AxisAlignedBox, object_bounds: AxisAlignedBox, source_frame_ids: list[int]
) -> MeshChunk:
    room_vertices, room_faces = _box_mesh(room_bounds)
    object_vertices, object_faces = _box_mesh(object_bounds)
    vertices = np.vstack([room_vertices, object_vertices]).astype(FLOAT32)
    faces = np.vstack([room_faces, object_faces + room_vertices.shape[0]]).astype(np.int32)
    room_face_count = room_faces.shape[0]
    object_face_count = object_faces.shape[0]
    return MeshChunk(
        chunk_id="cube_room_ground_truth",
        version=1,
        T_world_chunk=np.eye(4, dtype=FLOAT32),
        vertices_m=vertices,
        faces=faces,
        normals=None,
        colors=None,
        uvs=None,
        object_id_per_face=np.array(
            [0] * room_face_count + [1] * object_face_count, dtype=np.int32
        ),
        surface_source_per_face=np.full(
            faces.shape[0], int(SurfaceSource.OBSERVED_SURFACE), dtype=np.int8
        ),
        voxel_size_m=0.05,
        mean_uncertainty_m=0.0,
        p95_uncertainty_m=0.0,
        source_frame_ids=source_frame_ids,
        scale_source=ScaleSource.KNOWN_ANCHOR.value,
        flags=["synthetic_ground_truth"],
    )


def _box_mesh(bounds: AxisAlignedBox) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    x0, y0, z0 = bounds.min_corner_m.tolist()
    x1, y1, z1 = bounds.max_corner_m.tolist()
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=FLOAT64,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _make_object_instance(
    bounds: AxisAlignedBox, mesh_chunk_id: str, frames: tuple[SyntheticCubeRoomFrame, ...]
) -> ObjectInstance:
    center = ((bounds.min_corner_m + bounds.max_corner_m) * 0.5).astype(FLOAT32)
    extents = (bounds.max_corner_m - bounds.min_corner_m).astype(FLOAT32)
    T_world_object = np.eye(4, dtype=FLOAT32)
    T_world_object[:3, 3] = center
    return ObjectInstance(
        object_id=1,
        label_candidates=[("synthetic_cube", 1.0)],
        T_world_object=T_world_object,
        oriented_bbox_center_m=center,
        oriented_bbox_axes=np.eye(3, dtype=FLOAT32),
        oriented_bbox_extents_m=extents,
        mesh_chunk_ids=[mesh_chunk_id],
        is_dynamic=False,
        observed_coverage_ratio=1.0,
        confidence=1.0,
        uncertainty_m=0.0,
        first_seen_frame_id=frames[0].frame_id,
        last_seen_frame_id=frames[-1].frame_id,
        metadata={
            "bounds_min_m": _json_array(bounds.min_corner_m),
            "bounds_max_m": _json_array(bounds.max_corner_m),
            "surface_source": SurfaceSource.OBSERVED_SURFACE.name,
        },
    )


def _metadata_record(scene: SyntheticCubeRoomScene) -> dict[str, object]:
    return {
        "session_type": "synthetic_cube_room",
        "coordinate_convention": _coordinate_convention(),
        "unit_scale": "meters",
        "scale_source": ScaleSource.KNOWN_ANCHOR.value,
        "camera_metadata_source": scene.camera.source,
        "model_checkpoint_hash": None,
        "voxel_size_m": scene.mesh_chunk.voxel_size_m,
        "accuracy_report_path": None,
        "warnings": ["Synthetic analytic ground truth only; no neural model or capture was run."],
        "frame_count": len(scene.frames),
        "object_count": len(scene.objects),
        "mesh_chunk_count": len(scene.world_map.mesh_chunks),
    }


def _coordinate_convention() -> dict[str, object]:
    return {
        "camera_frame": {"x": "right", "y": "down", "z": "forward"},
        "world_units": "meters",
        "transform_rule": "T_A_B maps homogeneous points from frame B into frame A",
    }


def _pose_record(pose: PoseEstimate) -> dict[str, object]:
    return {
        "frame_id": pose.frame_id,
        "timestamp_ns": pose.timestamp_ns,
        "T_world_camera": _json_array(pose.T_world_camera),
        "q_world_camera_xyzw": _json_array(pose.q_world_camera_xyzw),
        "camera_center_world_m": _json_array(pose.camera_center_world_m),
        "covariance_6x6": _json_array(pose.covariance_6x6)
        if pose.covariance_6x6 is not None
        else None,
        "confidence": pose.confidence,
        "tracking_state": pose.tracking_state,
        "scale_source": pose.scale_source,
        "diagnostics": pose.diagnostics,
    }


def _camera_record(camera: CameraModel) -> dict[str, object]:
    return {
        "width": camera.width,
        "height": camera.height,
        "K": _json_array(camera.K),
        "distortion_model": camera.distortion_model,
        "distortion_params": _json_array(camera.distortion_params)
        if camera.distortion_params is not None
        else None,
        "rolling_shutter_row_time_s": camera.rolling_shutter_row_time_s,
        "confidence": camera.confidence,
        "source": camera.source,
    }


def _object_record(instance: ObjectInstance) -> dict[str, object]:
    return {
        "object_id": instance.object_id,
        "label_candidates": instance.label_candidates,
        "T_world_object": _json_array(instance.T_world_object),
        "oriented_bbox_center_m": _json_array(instance.oriented_bbox_center_m),
        "oriented_bbox_axes": _json_array(instance.oriented_bbox_axes),
        "oriented_bbox_extents_m": _json_array(instance.oriented_bbox_extents_m),
        "mesh_chunk_ids": instance.mesh_chunk_ids,
        "is_dynamic": instance.is_dynamic,
        "observed_coverage_ratio": instance.observed_coverage_ratio,
        "confidence": instance.confidence,
        "uncertainty_m": instance.uncertainty_m,
        "first_seen_frame_id": instance.first_seen_frame_id,
        "last_seen_frame_id": instance.last_seen_frame_id,
        "metadata": instance.metadata,
    }


def _mesh_chunk_record(chunk: MeshChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "version": chunk.version,
        "T_world_chunk": _json_array(chunk.T_world_chunk),
        "vertices_m": _json_array(chunk.vertices_m),
        "faces": _json_array(chunk.faces),
        "normals": _json_array(chunk.normals) if chunk.normals is not None else None,
        "colors": _json_array(chunk.colors) if chunk.colors is not None else None,
        "uvs": _json_array(chunk.uvs) if chunk.uvs is not None else None,
        "object_id_per_face": _json_array(chunk.object_id_per_face)
        if chunk.object_id_per_face is not None
        else None,
        "surface_source_per_face": _json_array(chunk.surface_source_per_face)
        if chunk.surface_source_per_face is not None
        else None,
        "voxel_size_m": chunk.voxel_size_m,
        "mean_uncertainty_m": chunk.mean_uncertainty_m,
        "p95_uncertainty_m": chunk.p95_uncertainty_m,
        "source_frame_ids": chunk.source_frame_ids,
        "scale_source": chunk.scale_source,
        "flags": chunk.flags,
    }


def _json_array(array: npt.NDArray[Any]) -> object:
    return cast(object, array.tolist())


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


__all__ = [
    "AxisAlignedBox",
    "SyntheticCubeRoomFrame",
    "SyntheticCubeRoomScene",
    "create_synthetic_cube_room_scene",
    "write_synthetic_cube_room_session",
]
