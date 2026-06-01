"""Load and validate Phase 0B `.atlas3r` session folders."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, MeshChunk, ObjectInstance, PoseEstimate

_DEPTH_RE = re.compile(r"^frame_(\d+)\.npz$")


@dataclass(frozen=True)
class LoadedSession:
    """A Phase 0B session reconstructed from JSON sidecars."""

    root: Path
    metadata: dict[str, Any]
    poses: tuple[PoseEstimate, ...]
    cameras: tuple[CameraModel, ...]
    objects: tuple[ObjectInstance, ...]
    mesh_chunks: tuple[MeshChunk, ...]
    depth_files: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        """Number of pose frames present in the session."""
        return len(self.poses)


def load_session(path: str | Path) -> LoadedSession:
    """Load a Phase 0B `.atlas3r` folder without reading depth arrays."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{root}: expected a `.atlas3r` session directory")

    metadata_path = root / "metadata.json"
    poses_path = root / "poses.jsonl"
    cameras_path = root / "cameras.jsonl"
    objects_path = root / "objects.jsonl"
    mesh_index_path = root / "mesh_chunks" / "index.json"

    metadata = _read_json_object(metadata_path)
    poses = tuple(_pose_from_record(record, poses_path) for record in _read_jsonl(poses_path))
    cameras = tuple(
        _camera_from_record(record, cameras_path) for record in _read_jsonl(cameras_path)
    )
    objects = tuple(
        _object_from_record(record, objects_path) for record in _read_jsonl(objects_path)
    )
    mesh_chunks = _load_mesh_chunks(mesh_index_path)
    depth_files = _discover_depth_files(root / "depth")
    return LoadedSession(
        root=root,
        metadata=metadata,
        poses=poses,
        cameras=cameras,
        objects=objects,
        mesh_chunks=mesh_chunks,
        depth_files=depth_files,
    )


def validate_session(path: str | Path) -> LoadedSession:
    """Load a Phase 0B session and validate cross-file counts."""
    session = load_session(path)
    if not session.poses:
        raise ValueError(f"{session.root / 'poses.jsonl'}: expected at least one pose record")
    if not session.cameras:
        raise ValueError(f"{session.root / 'cameras.jsonl'}: expected at least one camera record")
    _validate_metadata_count(session.metadata, "frame_count", len(session.poses), session.root)
    _validate_metadata_count(session.metadata, "object_count", len(session.objects), session.root)
    _validate_metadata_count(
        session.metadata, "mesh_chunk_count", len(session.mesh_chunks), session.root
    )
    return session


def load_depth_npz(path: str | Path) -> dict[str, npt.NDArray[Any]]:
    """Load one depth `.npz` file into memory."""
    depth_path = Path(path)
    if not depth_path.is_file():
        raise ValueError(f"{depth_path}: missing depth file")
    with np.load(depth_path) as depth_file:
        return {name: np.asarray(depth_file[name]) for name in depth_file.files}


def _load_mesh_chunks(index_path: Path) -> tuple[MeshChunk, ...]:
    index = _read_json_object(index_path)
    mesh_records = _required(index, "mesh_chunks", index_path)
    if not isinstance(mesh_records, list):
        raise ValueError(f"{index_path}: mesh_chunks must be a list")

    chunks: list[MeshChunk] = []
    for item_index, item in enumerate(mesh_records):
        if not isinstance(item, dict):
            raise ValueError(f"{index_path}: mesh_chunks[{item_index}] must be an object")
        metadata_path = _required(item, "metadata_path", index_path)
        if not isinstance(metadata_path, str) or not metadata_path:
            raise ValueError(f"{index_path}: mesh_chunks[{item_index}].metadata_path is invalid")
        sidecar_path = index_path.parent / metadata_path
        chunk = _mesh_chunk_from_record(_read_json_object(sidecar_path), sidecar_path)
        if "chunk_id" in item and item["chunk_id"] != chunk.chunk_id:
            raise ValueError(f"{index_path}: mesh_chunks[{item_index}].chunk_id mismatch")
        if "version" in item and int(cast(Any, item["version"])) != chunk.version:
            raise ValueError(f"{index_path}: mesh_chunks[{item_index}].version mismatch")
        chunks.append(chunk)
    return tuple(chunks)


def _discover_depth_files(depth_dir: Path) -> tuple[Path, ...]:
    if not depth_dir.exists():
        return ()

    def frame_id(path: Path) -> int:
        match = _DEPTH_RE.match(path.name)
        if match is None:
            raise ValueError(f"{path}: expected depth file name frame_<id>.npz")
        return int(match.group(1))

    return tuple(sorted(depth_dir.glob("frame_*.npz"), key=frame_id))


def _pose_from_record(record: dict[str, Any], path: Path) -> PoseEstimate:
    try:
        return PoseEstimate(
            frame_id=int(_required(record, "frame_id", path)),
            timestamp_ns=int(_required(record, "timestamp_ns", path)),
            T_world_camera=_array(record, "T_world_camera", path, np.float32),
            q_world_camera_xyzw=_array(record, "q_world_camera_xyzw", path, np.float32),
            camera_center_world_m=_array(record, "camera_center_world_m", path, np.float32),
            covariance_6x6=_optional_array(record, "covariance_6x6", path, np.float32),
            confidence=float(_required(record, "confidence", path)),
            tracking_state=str(_required(record, "tracking_state", path)),
            scale_source=str(_required(record, "scale_source", path)),
            diagnostics=_dict_field(record, "diagnostics", path),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed pose record: {exc}") from exc


def _camera_from_record(record: dict[str, Any], path: Path) -> CameraModel:
    try:
        return CameraModel(
            width=int(_required(record, "width", path)),
            height=int(_required(record, "height", path)),
            K=_array(record, "K", path, np.float32),
            distortion_model=str(_required(record, "distortion_model", path)),
            distortion_params=_optional_array(record, "distortion_params", path, np.float32),
            rolling_shutter_row_time_s=_optional_float(record, "rolling_shutter_row_time_s", path),
            confidence=float(_required(record, "confidence", path)),
            source=str(_required(record, "source", path)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed camera record: {exc}") from exc


def _object_from_record(record: dict[str, Any], path: Path) -> ObjectInstance:
    try:
        return ObjectInstance(
            object_id=int(_required(record, "object_id", path)),
            label_candidates=_label_candidates(record, path),
            T_world_object=_array(record, "T_world_object", path, np.float32),
            oriented_bbox_center_m=_array(record, "oriented_bbox_center_m", path, np.float32),
            oriented_bbox_axes=_array(record, "oriented_bbox_axes", path, np.float32),
            oriented_bbox_extents_m=_array(record, "oriented_bbox_extents_m", path, np.float32),
            mesh_chunk_ids=_str_list(record, "mesh_chunk_ids", path),
            is_dynamic=_bool_field(record, "is_dynamic", path),
            observed_coverage_ratio=float(_required(record, "observed_coverage_ratio", path)),
            confidence=float(_required(record, "confidence", path)),
            uncertainty_m=float(_required(record, "uncertainty_m", path)),
            first_seen_frame_id=int(_required(record, "first_seen_frame_id", path)),
            last_seen_frame_id=int(_required(record, "last_seen_frame_id", path)),
            metadata=_dict_field(record, "metadata", path),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed object record: {exc}") from exc


def _mesh_chunk_from_record(record: dict[str, Any], path: Path) -> MeshChunk:
    try:
        return MeshChunk(
            chunk_id=str(_required(record, "chunk_id", path)),
            version=int(_required(record, "version", path)),
            T_world_chunk=_array(record, "T_world_chunk", path, np.float32),
            vertices_m=_array(record, "vertices_m", path, np.float32),
            faces=_array(record, "faces", path, np.int32),
            normals=_optional_array(record, "normals", path, np.float32),
            colors=_optional_array(record, "colors", path, np.uint8),
            uvs=_optional_array(record, "uvs", path, np.float32),
            object_id_per_face=_optional_array(record, "object_id_per_face", path, np.int32),
            surface_source_per_face=_optional_array(
                record, "surface_source_per_face", path, np.int8
            ),
            voxel_size_m=float(_required(record, "voxel_size_m", path)),
            mean_uncertainty_m=float(_required(record, "mean_uncertainty_m", path)),
            p95_uncertainty_m=float(_required(record, "p95_uncertainty_m", path)),
            source_frame_ids=_int_list(record, "source_frame_ids", path),
            scale_source=str(_required(record, "scale_source", path)),
            flags=_str_list(record, "flags", path),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed mesh chunk record: {exc}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], data)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required file")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(cast(dict[str, Any], data))
    return tuple(records)


def _required(record: dict[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in record:
        raise ValueError(f"{path}: missing field {field_name}")
    return record[field_name]


def _array(
    record: dict[str, Any], field_name: str, path: Path, dtype: npt.DTypeLike
) -> npt.NDArray[Any]:
    return np.asarray(_required(record, field_name, path), dtype=dtype)


def _optional_array(
    record: dict[str, Any], field_name: str, path: Path, dtype: npt.DTypeLike
) -> npt.NDArray[Any] | None:
    value = _required(record, field_name, path)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _optional_float(record: dict[str, Any], field_name: str, path: Path) -> float | None:
    value = _required(record, field_name, path)
    return None if value is None else float(value)


def _dict_field(record: dict[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    value = _required(record, field_name, path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field_name} must be an object")
    return cast(dict[str, Any], value)


def _str_list(record: dict[str, Any], field_name: str, path: Path) -> list[str]:
    value = _required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {field_name} must contain strings")
    return cast(list[str], value)


def _int_list(record: dict[str, Any], field_name: str, path: Path) -> list[int]:
    value = _required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    return [int(item) for item in value]


def _label_candidates(record: dict[str, Any], path: Path) -> list[tuple[str, float]]:
    value = _required(record, "label_candidates", path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: label_candidates must be a list")
    candidates: list[tuple[str, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"{path}: label_candidates[{index}] must contain label and confidence")
        candidates.append((str(item[0]), float(item[1])))
    return candidates


def _bool_field(record: dict[str, Any], field_name: str, path: Path) -> bool:
    value = _required(record, field_name, path)
    if not isinstance(value, bool):
        raise ValueError(f"{path}: {field_name} must be a bool")
    return value


def _validate_metadata_count(
    metadata: dict[str, Any], field_name: str, expected_count: int, root: Path
) -> None:
    if field_name not in metadata:
        return
    if int(metadata[field_name]) != expected_count:
        raise ValueError(
            f"{root / 'metadata.json'}: {field_name}={metadata[field_name]} "
            f"does not match loaded count {expected_count}"
        )


__all__ = ["LoadedSession", "load_depth_npz", "load_session", "validate_session"]
