"""Mesh chunk event helpers for RGB teacher mapping."""

from __future__ import annotations

from typing import cast

from atlas3r.mapping.mesh_artifacts import MeshChunkArtifactWriter
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper
from atlas3r.runtime.live_replay_types import LiveReplayEventRecorder
from atlas3r.runtime.student_map_reports import LatencyRecorder


def process_rgb_teacher_mesh_blocks(
    *,
    mesh_writer: MeshChunkArtifactWriter,
    mapper: SparseBlockTSDFMapper,
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
    pending_dirty: set[tuple[int, int, int]],
    observation: DepthObservation,
    map_update_latency_ns: int,
) -> None:
    dirty = tuple(sorted(pending_dirty))
    pending_dirty.clear()
    mesh_events, mesh_latency_ns = mesh_writer.process_dirty_blocks(
        mapper=mapper,
        dirty_block_coords_xyz=dirty,
        frame_id=observation.frame_id,
        timestamp_ns=observation.pose.timestamp_ns,
        dirty_reason="rgb_teacher_sparse_tsdf_dirty_block",
        capture_queue_depth=0,
        map_queue_depth=0,
        max_dirty_chunks=None,
    )
    recorder.add("mesh_update", mesh_latency_ns)
    recorder.add("map_mesh_update", map_update_latency_ns + mesh_latency_ns)
    for mesh_event in mesh_events:
        events.emit(
            "mesh_chunk_update",
            frame_id=observation.frame_id,
            timestamp_ns=observation.pose.timestamp_ns,
            latency_ns=cast(int, mesh_event["mesh_latency_ns"]),
            metadata=mesh_event,
            paths={
                key: str(mesh_event[key])
                for key in ("payload_npz", "payload_ply")
                if mesh_event.get(key) is not None
            },
        )
