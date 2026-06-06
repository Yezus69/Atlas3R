"""Normalized proposal-cache artifacts for offline world builds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.depth_pro_witness import (
    DepthProWitnessResult,
)
from atlas3r.offline.depth_pro_witness import (
    truth_boundary_dict as depth_pro_truth_boundary_dict,
)
from atlas3r.offline.frame_cache import FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import write_json, write_jsonl
from atlas3r.offline.vggt_witness import (
    VggtWitnessResult,
)
from atlas3r.offline.vggt_witness import (
    truth_boundary_dict as vggt_truth_boundary_dict,
)
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
    geometry_depth_records: tuple[dict[str, object], ...] = ()
    depth_arrays: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    geometry_source: str = "none"
    vggt_camera_records: tuple[dict[str, object], ...] = ()
    vggt_depth_records: tuple[dict[str, object], ...] = ()
    vggt_window_records: tuple[dict[str, object], ...] = ()
    vggt_depth_npz_path: str | None = None
    vggt_metadata: dict[str, object] = field(default_factory=dict)
    depth_pro_camera_records: tuple[dict[str, object], ...] = ()
    depth_pro_depth_records: tuple[dict[str, object], ...] = ()
    depth_pro_frame_records: tuple[dict[str, object], ...] = ()
    depth_pro_depth_npz_path: str | None = None
    depth_pro_metadata: dict[str, object] = field(default_factory=dict)


def write_proposal_cache(
    run_dir: str | Path,
    *,
    teacher_statuses: tuple[AdapterStatus, ...],
    frame_records: tuple[FrameRecord, ...],
    keyframes: tuple[KeyframeRecord, ...],
    debug_geometry_mode: DebugGeometryMode,
    vggt_result: VggtWitnessResult | None = None,
    depth_pro_result: DepthProWitnessResult | None = None,
) -> ProposalCacheResult:
    root = Path(run_dir)
    streams: list[dict[str, object]] = []
    for status in teacher_statuses:
        if status.name == "vggt" and vggt_result is not None and vggt_result.has_geometry:
            continue
        if (
            status.name == "depth_pro"
            and depth_pro_result is not None
            and depth_pro_result.has_depth
        ):
            continue
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
                "stream_type": "placeholder",
                "status": "unavailable",
                "path": f"proposals/{status.name}_proposals.jsonl",
                "proposal_count": 0,
                "capabilities": status.capabilities.to_dict(),
            }
        )
    debug_records = _write_debug_flat_depth_stream(
        root, frame_records, keyframes, debug_geometry_mode
    )
    geometry_records: list[dict[str, object]] = []
    depth_arrays: dict[str, NDArray[np.float32]] = {}
    geometry_source = "none"
    vggt_camera_records: tuple[dict[str, object], ...] = ()
    vggt_depth_records: tuple[dict[str, object], ...] = ()
    vggt_window_records: tuple[dict[str, object], ...] = ()
    vggt_depth_npz_path: str | None = None
    vggt_metadata: dict[str, object] = {}
    depth_pro_camera_records: tuple[dict[str, object], ...] = ()
    depth_pro_depth_records: tuple[dict[str, object], ...] = ()
    depth_pro_frame_records: tuple[dict[str, object], ...] = ()
    depth_pro_depth_npz_path: str | None = None
    depth_pro_metadata: dict[str, object] = {}
    if debug_records:
        geometry_records.extend(debug_records)
        geometry_source = "debug_flat_depth"
        streams.append(
            {
                "teacher_name": "debug_flat_depth",
                "stream_type": "depth_pose_intrinsics",
                "status": "available",
                "path": "proposals/debug_flat_depth_proposals.jsonl",
                "proposal_count": len(debug_records),
                "capabilities": {"depth": True, "intrinsics": True, "pose": True},
            }
        )
    if vggt_result is not None and vggt_result.has_geometry:
        vggt_camera_records, vggt_window_records, vggt_depth_npz_path = _write_vggt_streams(
            root, vggt_result
        )
        geometry_records.extend(vggt_result.depth_records)
        depth_arrays.update(vggt_result.depth_arrays)
        geometry_source = "vggt"
        vggt_depth_records = tuple(vggt_result.depth_records)
        vggt_metadata = vggt_result.metadata
        streams.extend(
            [
                {
                    "teacher_name": "vggt",
                    "stream_type": "cameras",
                    "status": vggt_result.runtime_status,
                    "path": "proposals/vggt_cameras.jsonl",
                    "proposal_count": len(vggt_camera_records),
                    "capabilities": {
                        "depth": True,
                        "intrinsics": True,
                        "pose": True,
                        "point_tracks": False,
                    },
                },
                {
                    "teacher_name": "vggt",
                    "stream_type": "depths",
                    "status": vggt_result.runtime_status,
                    "path": "proposals/vggt_depths.npz",
                    "proposal_count": len(vggt_result.depth_records),
                    "capabilities": {"depth": True, "uncertainty": True},
                },
                {
                    "teacher_name": "vggt",
                    "stream_type": "windows",
                    "status": vggt_result.runtime_status,
                    "path": "proposals/vggt_windows.jsonl",
                    "proposal_count": len(vggt_window_records),
                    "capabilities": {"window_stitching": True},
                },
            ]
        )
    if depth_pro_result is not None and depth_pro_result.has_depth:
        (
            depth_pro_camera_records,
            depth_pro_frame_records,
            depth_pro_depth_npz_path,
        ) = _write_depth_pro_streams(root, depth_pro_result)
        depth_pro_depth_records = tuple(depth_pro_result.depth_records)
        depth_arrays.update(depth_pro_result.depth_arrays)
        depth_pro_metadata = depth_pro_result.metadata
        streams.extend(
            [
                {
                    "teacher_name": "depth_pro",
                    "stream_type": "cameras",
                    "status": depth_pro_result.runtime_status,
                    "path": "proposals/depth_pro_cameras.jsonl",
                    "proposal_count": len(depth_pro_camera_records),
                    "capabilities": {
                        "depth": True,
                        "intrinsics": True,
                        "pose": False,
                        "point_tracks": False,
                    },
                },
                {
                    "teacher_name": "depth_pro",
                    "stream_type": "depths",
                    "status": depth_pro_result.runtime_status,
                    "path": "proposals/depth_pro_depths.npz",
                    "proposal_count": len(depth_pro_depth_records),
                    "capabilities": {"depth": True, "uncertainty": True},
                },
                {
                    "teacher_name": "depth_pro",
                    "stream_type": "frames",
                    "status": depth_pro_result.runtime_status,
                    "path": "proposals/depth_pro_frames.jsonl",
                    "proposal_count": len(depth_pro_frame_records),
                    "capabilities": {"per_frame_depth": True},
                },
            ]
        )
        if geometry_source == "none" and not debug_records:
            geometry_source = "depth_pro_diagnostic_no_global_pose"
    payload = {
        "status": "partial" if streams else "unavailable",
        "format_name": "atlas3r_teacher_proposal_cache",
        "format_version": 1,
        "debug_geometry_mode": debug_geometry_mode,
        "streams": streams,
        "teacher_counts": {
            "vggt_camera_proposals": len(vggt_camera_records),
            "vggt_depth_proposals": 0 if vggt_result is None else len(vggt_result.depth_records),
            "depth_pro_camera_proposals": len(depth_pro_camera_records),
            "depth_pro_depth_proposals": 0
            if depth_pro_result is None
            else len(depth_pro_result.depth_records),
            "debug_depth_proposals": len(debug_records),
        },
        "truth_boundary": vggt_truth_boundary_dict()
        if geometry_source == "vggt"
        else depth_pro_truth_boundary_dict()
        if geometry_source == "depth_pro_diagnostic_no_global_pose"
        else {
            "label_type": "debug_synthetic" if debug_records else "unknown",
            "measured_geometry": False,
            "metric_scale_source": "debug_flat_depth" if debug_records else "unknown",
            "observed_only": True,
            "predicted_completion": False,
            "hidden_geometry_measured": False,
            "accuracy_report": False,
            "realtime_claim": False,
            "usable_for_training": False,
        },
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
        depth_proposal_available=bool(geometry_records),
        debug_geometry_mode=debug_geometry_mode,
        geometry_depth_records=tuple(geometry_records),
        depth_arrays=depth_arrays,
        geometry_source=geometry_source,
        vggt_camera_records=vggt_camera_records,
        vggt_depth_records=vggt_depth_records,
        vggt_window_records=vggt_window_records,
        vggt_depth_npz_path=vggt_depth_npz_path,
        vggt_metadata=vggt_metadata,
        depth_pro_camera_records=depth_pro_camera_records,
        depth_pro_depth_records=depth_pro_depth_records,
        depth_pro_frame_records=depth_pro_frame_records,
        depth_pro_depth_npz_path=depth_pro_depth_npz_path,
        depth_pro_metadata=depth_pro_metadata,
    )


def _write_vggt_streams(
    run_dir: Path, vggt_result: VggtWitnessResult
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], str]:
    camera_records = tuple(vggt_result.camera_records)
    window_records = tuple(vggt_result.window_records)
    write_jsonl(run_dir / "proposals" / "vggt_cameras.jsonl", list(camera_records))
    write_jsonl(run_dir / "proposals" / "vggt_windows.jsonl", list(window_records))
    depth_path = run_dir / "proposals" / "vggt_depths.npz"
    metadata = {
        "teacher_name": "vggt",
        "depth_records": list(vggt_result.depth_records),
        "vggt_metadata": vggt_result.metadata,
        "truth_boundary": vggt_truth_boundary_dict(),
    }
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        depth_path,
        **{
            key: value.astype(np.float32) for key, value in sorted(vggt_result.depth_arrays.items())
        },
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return camera_records, window_records, "proposals/vggt_depths.npz"


def _write_depth_pro_streams(
    run_dir: Path, depth_pro_result: DepthProWitnessResult
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], str]:
    camera_records = tuple(depth_pro_result.camera_records)
    frame_records = tuple(depth_pro_result.frame_records)
    write_jsonl(run_dir / "proposals" / "depth_pro_cameras.jsonl", list(camera_records))
    write_jsonl(run_dir / "proposals" / "depth_pro_frames.jsonl", list(frame_records))
    depth_path = run_dir / "proposals" / "depth_pro_depths.npz"
    metadata = {
        "teacher_name": "depth_pro",
        "depth_records": list(depth_pro_result.depth_records),
        "depth_pro_metadata": depth_pro_result.metadata,
        "truth_boundary": depth_pro_truth_boundary_dict(),
    }
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        depth_path,
        **{
            key: value.astype(np.float32)
            for key, value in sorted(depth_pro_result.depth_arrays.items())
        },
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return camera_records, frame_records, "proposals/depth_pro_depths.npz"


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
                "keyframe_index": keyframe.keyframe_id,
                "status": "available",
                "constant_depth_m": 2.0,
                "depth_sigma_m": 0.5,
                "confidence": 0.25,
                "camera": frame.camera.to_dict(),
                "K": frame.camera.K.tolist(),
                "T_world_camera": T_world_camera.tolist(),
                "label_type": label_type,
                "measured_geometry": debug_geometry_mode == "synthetic-known",
                "metric_scale_source": scale_source,
                "usable_for_training": usable_for_training,
                "coordinate_convention": "x_right_y_down_z_forward",
                "source": "debug_only_not_teacher_model",
                "truth_boundary": {
                    "label_type": label_type,
                    "metric_scale_source": scale_source,
                    "measured_geometry": debug_geometry_mode == "synthetic-known",
                    "observed_only": True,
                    "predicted_completion": False,
                    "hidden_geometry_measured": False,
                    "accuracy_report": False,
                    "realtime_claim": False,
                    "usable_for_training": usable_for_training,
                },
            }
        )
    write_jsonl(run_dir / "proposals" / "debug_flat_depth_proposals.jsonl", rows)
    return rows
