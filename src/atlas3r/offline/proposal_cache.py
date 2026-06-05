"""Normalized proposal-cache skeleton for offline world builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from atlas3r.offline.frame_cache import FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import write_json, write_jsonl
from atlas3r.teachers.base import AdapterStatus

DebugGeometryMode = Literal["none", "flat-depth", "synthetic-known"]


@dataclass(frozen=True)
class ProposalCacheResult:
    status: str
    manifest_path: str
    streams: tuple[dict[str, object], ...]
    debug_depth_records: tuple[dict[str, object], ...]
    depth_proposal_available: bool
    debug_geometry_mode: DebugGeometryMode


def write_proposal_cache(
    run_dir: str | Path,
    *,
    teacher_statuses: tuple[AdapterStatus, ...],
    frame_records: tuple[FrameRecord, ...],
    keyframes: tuple[KeyframeRecord, ...],
    debug_geometry_mode: DebugGeometryMode,
) -> ProposalCacheResult:
    root = Path(run_dir)
    streams: list[dict[str, object]] = []
    for status in teacher_statuses:
        stream_path = root / "proposals" / f"{status.name}_proposals.jsonl"
        row = {
            "teacher_name": status.name,
            "status": "unavailable",
            "proposal_count": 0,
            "label_type": "teacher_pseudo",
            "measured_geometry": False,
            "usable_for_training": False,
            "why": status.reason,
            "install_hint": status.install_hint,
        }
        write_jsonl(stream_path, [row])
        streams.append(
            {
                "teacher_name": status.name,
                "status": "unavailable",
                "path": f"proposals/{status.name}_proposals.jsonl",
                "proposal_count": 0,
                "capabilities": status.capabilities.to_dict(),
            }
        )
    debug_records = _write_debug_flat_depth_stream(
        root, frame_records, keyframes, debug_geometry_mode
    )
    if debug_records:
        streams.append(
            {
                "teacher_name": "debug_flat_depth",
                "status": "available",
                "path": "proposals/debug_flat_depth_proposals.jsonl",
                "proposal_count": len(debug_records),
                "capabilities": {"depth": True, "intrinsics": True, "pose": True},
            }
        )
    payload = {
        "status": "partial" if streams else "unavailable",
        "format_name": "atlas3r_teacher_proposal_cache",
        "format_version": 1,
        "debug_geometry_mode": debug_geometry_mode,
        "streams": streams,
        "normalized_fields": [
            "depth",
            "intrinsics",
            "pose",
            "point_tracks",
            "masks",
            "features",
            "uncertainty",
            "coordinate_convention",
            "source",
        ],
    }
    write_json(root / "proposals" / "proposal_manifest.json", payload)
    return ProposalCacheResult(
        status=str(payload["status"]),
        manifest_path="proposals/proposal_manifest.json",
        streams=tuple(streams),
        debug_depth_records=tuple(debug_records),
        depth_proposal_available=bool(debug_records),
        debug_geometry_mode=debug_geometry_mode,
    )


def _write_debug_flat_depth_stream(
    run_dir: Path,
    frame_records: tuple[FrameRecord, ...],
    keyframes: tuple[KeyframeRecord, ...],
    debug_geometry_mode: DebugGeometryMode,
) -> list[dict[str, object]]:
    if debug_geometry_mode == "none" or not keyframes:
        return []
    records_by_id = {record.frame_id: record for record in frame_records}
    label_type = "synthetic_gt" if debug_geometry_mode == "synthetic-known" else "debug_synthetic"
    scale_source = (
        "synthetic_known" if debug_geometry_mode == "synthetic-known" else "debug_flat_depth"
    )
    usable_for_training = debug_geometry_mode == "synthetic-known"
    rows: list[dict[str, object]] = []
    for index, keyframe in enumerate(keyframes):
        frame = records_by_id[keyframe.frame_id]
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[0, 3] = float(index) * 0.05
        rows.append(
            {
                "teacher_name": "debug_flat_depth",
                "frame_id": keyframe.frame_id,
                "status": "available",
                "constant_depth_m": 2.0,
                "depth_sigma_m": 0.5,
                "confidence": 0.25,
                "camera": frame.camera.to_dict(),
                "T_world_camera": T_world_camera.tolist(),
                "label_type": label_type,
                "measured_geometry": debug_geometry_mode == "synthetic-known",
                "metric_scale_source": scale_source,
                "usable_for_training": usable_for_training,
                "coordinate_convention": "x_right_y_down_z_forward",
                "source": "debug_only_not_teacher_model",
            }
        )
    write_jsonl(run_dir / "proposals" / "debug_flat_depth_proposals.jsonl", rows)
    return rows
