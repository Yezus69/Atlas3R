"""Artifact writer for observed sparse mesh chunk updates."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from atlas3r.mapping.mesh_chunks import (
    MESH_CHUNK_FORMAT_VERSION,
    MESH_CHUNK_MANIFEST_FORMAT_NAME,
    MESH_TRUTH_FLAGS,
    ObservedMeshChunk,
    chunk_id_from_block_coord,
    save_mesh_chunk_npz,
    write_mesh_chunk_ply,
)
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper
from atlas3r.mapping.sparse_tsdf_meshing import (
    SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
    SparseTSDFMesherConfig,
    mesh_sparse_tsdf_snapshot,
    mesher_backend_status,
)

MESH_CHUNK_UPDATE_EVENT_FORMAT_NAME = "atlas3r_observed_mesh_chunk_update_event"


class MeshChunkArtifactWriter:
    """Write versioned observed mesh chunk payloads, updates, and manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        mesh_format: str,
        mesher_config: SparseTSDFMesherConfig,
        truth_flags: Mapping[str, object] | None = None,
        truth_boundary: Mapping[str, object] | None = None,
    ) -> None:
        if mesh_format not in {"npz", "ply", "both"}:
            raise ValueError("mesh_format: must be npz, ply, or both")
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        self.mesh_format = mesh_format
        self.mesher_config = mesher_config
        self.truth_flags = dict(MESH_TRUTH_FLAGS if truth_flags is None else truth_flags)
        self.truth_boundary = None if truth_boundary is None else dict(truth_boundary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, int] = {}
        self._active_chunks: dict[str, dict[str, object]] = {}
        self._deleted_chunk_ids: set[str] = set()
        self._updates: list[dict[str, object]] = []
        self._payload_bytes = 0

    @property
    def update_events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._updates)

    @property
    def mesh_payload_bytes(self) -> int:
        return int(self._payload_bytes)

    def process_dirty_blocks(
        self,
        *,
        mapper: SparseBlockTSDFMapper,
        dirty_block_coords_xyz: tuple[tuple[int, int, int], ...],
        frame_id: int,
        timestamp_ns: int,
        dirty_reason: str,
        capture_queue_depth: int,
        map_queue_depth: int,
        max_dirty_chunks: int | None,
    ) -> tuple[list[dict[str, object]], int]:
        unique_dirty = tuple(dict.fromkeys(sorted(dirty_block_coords_xyz)))
        if max_dirty_chunks is not None:
            unique_dirty = unique_dirty[:max_dirty_chunks]
        events: list[dict[str, object]] = []
        snapshot_start_ns = time.perf_counter_ns()
        snapshots = mapper.block_voxel_snapshots(unique_dirty, include_one_voxel_halo=True)
        total_latency_ns = time.perf_counter_ns() - snapshot_start_ns
        for block_coord in unique_dirty:
            start_ns = time.perf_counter_ns()
            chunk_id = chunk_id_from_block_coord(block_coord)
            version = self._versions.get(chunk_id, 0) + 1
            chunk = mesh_sparse_tsdf_snapshot(
                mapper,
                snapshots[block_coord],
                version=version,
                mesher_config=self.mesher_config,
            )
            latency_ns = time.perf_counter_ns() - start_ns
            total_latency_ns += latency_ns
            if chunk is None:
                delete_event = self._delete_event_if_active(
                    chunk_id=chunk_id,
                    block_coord=block_coord,
                    version=version,
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    dirty_reason=dirty_reason,
                    mesh_latency_ns=latency_ns,
                    capture_queue_depth=capture_queue_depth,
                    map_queue_depth=map_queue_depth,
                )
                if delete_event is not None:
                    events.append(delete_event)
                continue
            chunk.truth_flags = dict(self.truth_flags)
            event = self._write_upsert_event(
                chunk=chunk,
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                dirty_reason=dirty_reason,
                mesh_latency_ns=latency_ns,
                capture_queue_depth=capture_queue_depth,
                map_queue_depth=map_queue_depth,
            )
            events.append(event)
        return events, total_latency_ns

    def write_manifest(
        self,
        *,
        mapper: SparseBlockTSDFMapper,
        source_frame_ids_mapped: list[int],
        mapper_backend: str,
    ) -> dict[str, object]:
        chunks = [self._active_chunks[key] for key in sorted(self._active_chunks)]
        total_vertices = sum(_record_int(record, "vertex_count") for record in chunks)
        total_triangles = sum(_record_int(record, "triangle_count") for record in chunks)
        manifest = {
            **self.truth_flags,
            "active_chunk_count": len(chunks),
            "chunk_count": len(chunks) + len(self._deleted_chunk_ids),
            "chunks": chunks,
            "coordinate_frame": mapper.config.coordinate_frame,
            "deleted_chunk_count": len(self._deleted_chunk_ids),
            "format_name": MESH_CHUNK_MANIFEST_FORMAT_NAME,
            "format_version": MESH_CHUNK_FORMAT_VERSION,
            "mapper_backend": mapper_backend,
            "mesher_backend": SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
            "mesher_status": mesher_backend_status(),
            "mesh_format": self.mesh_format,
            "mesh_payload_bytes": int(self._payload_bytes),
            "source_frame_ids_mapped": source_frame_ids_mapped,
            "total_triangle_count": int(total_triangles),
            "total_vertex_count": int(total_vertices),
            "truth_boundary": {
                **self._manifest_truth_boundary(source_frame_ids_mapped),
                "diagnostic_only": True,
            },
            "update_count": len(self._updates),
            "updates": "mesh_chunk_updates.jsonl",
            "voxel_size_m": mapper.config.voxel_size_m,
        }
        _write_json(self.output_dir / "mesh_chunk_manifest.json", manifest)
        _write_jsonl(self.output_dir / "mesh_chunk_updates.jsonl", self._updates)
        return manifest

    def status(self) -> dict[str, object]:
        chunks = [self._active_chunks[key] for key in sorted(self._active_chunks)]
        return {
            **self.truth_flags,
            "active_mesh_chunk_count": len(chunks),
            "deleted_mesh_chunk_count": len(self._deleted_chunk_ids),
            "format_name": "atlas3r_phase6f_live_replay_mesh_status",
            "format_version": 1,
            "mesh_backend": SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
            "mesh_chunk_count": len(chunks) + len(self._deleted_chunk_ids),
            "mesh_chunk_update_count": len(self._updates),
            "mesh_exported": len(chunks) > 0,
            "mesh_format": self.mesh_format,
            "mesh_payload_bytes": int(self._payload_bytes),
            "total_triangle_count": sum(_record_int(record, "triangle_count") for record in chunks),
            "total_vertex_count": sum(_record_int(record, "vertex_count") for record in chunks),
        }

    def _write_upsert_event(
        self,
        *,
        chunk: ObservedMeshChunk,
        frame_id: int,
        timestamp_ns: int,
        dirty_reason: str,
        mesh_latency_ns: int,
        capture_queue_depth: int,
        map_queue_depth: int,
    ) -> dict[str, object]:
        npz_path = self.chunks_dir / f"{chunk.chunk_id}_v{chunk.version:06d}.npz"
        save_mesh_chunk_npz(npz_path, chunk)
        ply_rel: str | None = None
        if self.mesh_format in {"ply", "both"}:
            ply_path = self.chunks_dir / f"{chunk.chunk_id}_v{chunk.version:06d}.ply"
            write_mesh_chunk_ply(ply_path, chunk)
            ply_rel = _relative_to_mesh_dir(self.output_dir, ply_path)
        npz_rel = _relative_to_mesh_dir(self.output_dir, npz_path)
        self._payload_bytes += (
            chunk.vertices_world_m.nbytes + chunk.normals_world.nbytes + chunk.triangles.nbytes
        )
        record = chunk.metadata_record()
        record.update({"payload_npz": npz_rel, "payload_ply": ply_rel})
        self._active_chunks[chunk.chunk_id] = record
        self._deleted_chunk_ids.discard(chunk.chunk_id)
        self._versions[chunk.chunk_id] = chunk.version
        event = self._event_record(
            chunk_id=chunk.chunk_id,
            block_coord=chunk.chunk_coord_xyz,
            version=chunk.version,
            update_type="upsert",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            payload_npz=npz_rel,
            payload_ply=ply_rel,
            vertex_count=chunk.vertex_count,
            triangle_count=chunk.triangle_count,
            dirty_reason=dirty_reason,
            mesh_latency_ns=mesh_latency_ns,
            capture_queue_depth=capture_queue_depth,
            map_queue_depth=map_queue_depth,
        )
        self._updates.append(event)
        return event

    def _delete_event_if_active(
        self,
        *,
        chunk_id: str,
        block_coord: tuple[int, int, int],
        version: int,
        frame_id: int,
        timestamp_ns: int,
        dirty_reason: str,
        mesh_latency_ns: int,
        capture_queue_depth: int,
        map_queue_depth: int,
    ) -> dict[str, object] | None:
        if chunk_id not in self._active_chunks:
            return None
        self._active_chunks.pop(chunk_id)
        self._deleted_chunk_ids.add(chunk_id)
        self._versions[chunk_id] = version
        event = self._event_record(
            chunk_id=chunk_id,
            block_coord=block_coord,
            version=version,
            update_type="delete",
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            payload_npz=None,
            payload_ply=None,
            vertex_count=0,
            triangle_count=0,
            dirty_reason=dirty_reason,
            mesh_latency_ns=mesh_latency_ns,
            capture_queue_depth=capture_queue_depth,
            map_queue_depth=map_queue_depth,
        )
        self._updates.append(event)
        return event

    def _event_record(
        self,
        *,
        chunk_id: str,
        block_coord: tuple[int, int, int],
        version: int,
        update_type: str,
        frame_id: int,
        timestamp_ns: int,
        payload_npz: str | None,
        payload_ply: str | None,
        vertex_count: int,
        triangle_count: int,
        dirty_reason: str,
        mesh_latency_ns: int,
        capture_queue_depth: int,
        map_queue_depth: int,
    ) -> dict[str, object]:
        return {
            **self.truth_flags,
            "capture_queue_depth": int(capture_queue_depth),
            "chunk_coord_xyz": list(block_coord),
            "chunk_id": chunk_id,
            "dirty_reason": dirty_reason,
            "event_index": len(self._updates),
            "format_name": MESH_CHUNK_UPDATE_EVENT_FORMAT_NAME,
            "format_version": MESH_CHUNK_FORMAT_VERSION,
            "frame_id": int(frame_id),
            "map_queue_depth": int(map_queue_depth),
            "mesh_latency_ns": int(mesh_latency_ns),
            "payload_npz": payload_npz,
            "payload_ply": payload_ply,
            "timestamp_ns": int(timestamp_ns),
            "triangle_count": int(triangle_count),
            "update_type": update_type,
            "version": int(version),
            "vertex_count": int(vertex_count),
        }

    def _manifest_truth_boundary(self, source_frame_ids_mapped: list[int]) -> dict[str, object]:
        if self.truth_boundary is not None:
            return dict(self.truth_boundary)
        return {
            **MESH_TRUTH_FLAGS,
            "measured_depth_used": bool(source_frame_ids_mapped),
            "measured_pose_used": bool(source_frame_ids_mapped),
        }


def _relative_to_mesh_dir(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _record_int(record: dict[str, object], field_name: str) -> int:
    return int(cast(Any, record[field_name]))


def _write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


__all__ = [
    "MESH_CHUNK_UPDATE_EVENT_FORMAT_NAME",
    "MeshChunkArtifactWriter",
]
