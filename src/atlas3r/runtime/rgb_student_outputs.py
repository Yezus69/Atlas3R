"""Output helpers for RGB student mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping.cpu_tsdf import FLOAT32, TSDFSurface
from atlas3r.mapping.mesh_chunks import load_mesh_chunk_npz
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper

MAPPER_BACKEND = "cpu-sparse"
SUMMARY_FORMAT_NAME = "atlas3r_rgb_student_mapping_summary"


def student_summary(
    *,
    config: Any,
    source_frame_count: int,
    frames: Sequence[Any],
    observations: Sequence[DepthObservation],
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    mesh_status: Mapping[str, object],
    mesh_manifest: Mapping[str, object] | None,
    point_cloud_path: Path | None,
    prediction_metadata: Mapping[str, object],
    latency_report: Mapping[str, object],
    checkpoint: Mapping[str, Any],
    resolved_device: str,
    eval_summary: Mapping[str, object] | None,
    truth_boundary: dict[str, object],
) -> dict[str, object]:
    return {
        **truth_boundary,
        "format_name": SUMMARY_FORMAT_NAME,
        "format_version": 1,
        "input": str(config.input),
        "output": str(config.output),
        "checkpoint": str(config.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "device": resolved_device,
        "frame_count_available": source_frame_count,
        "frame_count_used": len(frames),
        "prediction_frame_count": len(observations),
        "sparse_map_update_count": len(observations),
        "active_blocks": mapper.active_block_count,
        "active_voxels": mapper.active_voxel_count,
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "mesh": dict(mesh_status),
        "mesh_chunk_count": int(cast(Any, mesh_status.get("mesh_chunk_count", 0))),
        "total_vertex_count": int(cast(Any, mesh_status.get("total_vertex_count", 0))),
        "total_triangle_count": int(cast(Any, mesh_status.get("total_triangle_count", 0))),
        "mesh_manifest": None if mesh_manifest is None else "mesh_chunks/mesh_chunk_manifest.json",
        "latency": cast(dict[str, object], latency_report["segments"]),
        "model_config": dict(cast(dict[str, object], checkpoint["model_config"])),
        "eval": None if eval_summary is None else dict(eval_summary),
        "artifacts": {
            "rgb_student_predictions": "rgb_student_predictions",
            "rgb_student_summary": "rgb_student_summary.json",
            "rgb_student_report": "rgb_student_report.md",
            "live_replay_events": "live_replay_events.jsonl",
            "live_replay_summary": "live_replay_summary.json",
            "live_replay_report": "live_replay_report.md",
            "sparse_tsdf": "sparse_tsdf",
            "mesh_chunks": "mesh_chunks" if mesh_manifest is not None else None,
            "point_cloud": None if point_cloud_path is None else point_cloud_path.name,
        },
        "truth_boundary": truth_boundary,
        "known_limitations": student_known_limitations(),
        **dict(prediction_metadata),
    }


def student_live_replay_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        **dict(summary),
        "format_name": "atlas3r_rgb_student_live_replay_summary",
        "measured_depth_used": False,
        "measured_pose_used": False,
    }


def write_student_report(path: Path, summary: Mapping[str, object]) -> None:
    truth = cast(Mapping[str, object], summary["truth_boundary"])
    lines = [
        "# Core SMGT Tiny RGB Student Mapping Report",
        "",
        f"- Input: `{summary['input']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Frames used: `{summary['frame_count_used']}`",
        f"- Student poses/depths/intrinsics: `{summary['student_pose_count']}` / "
        f"`{summary['student_depth_count']}` / `{summary['student_intrinsics_count']}`",
        f"- Student depth valid pixel ratio: `{summary['student_depth_valid_pixel_ratio']}`",
        f"- Sparse map updates: `{summary['sparse_map_update_count']}`",
        f"- Active blocks/voxels: `{summary['active_blocks']}` / `{summary['active_voxels']}`",
        f"- Mesh chunks: `{summary['mesh_chunk_count']}`",
        f"- Mesh vertices/triangles: `{summary['total_vertex_count']}` / "
        f"`{summary['total_triangle_count']}`",
        f"- VGGT used during student inference: `{truth['teacher_geometry_used']}`",
        f"- Measured depth used for mapping: `{truth['measured_depth_used']}`",
        f"- Measured pose used for mapping: `{truth['measured_pose_used']}`",
        f"- Metric scale source: `{truth['metric_scale_source']}`",
        "",
        "This is learned diagnostic RGB student mapping from an SMGT-tiny checkpoint. "
        "It is observed-only, not final SMGT, not realtime evidence, not an accuracy "
        "report, not object-aware fusion, and not RGB-only production readiness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_or_empty_student_surface(
    mapper: SparseBlockTSDFMapper,
    *,
    truth_boundary: Mapping[str, object],
) -> TSDFSurface:
    try:
        surface = mapper.extract_surface()
        return TSDFSurface(
            points_world_m=surface.points_world_m,
            confidence=surface.confidence,
            uncertainty_m=surface.uncertainty_m,
            voxel_indices_xyz=surface.voxel_indices_xyz,
            metadata={**surface.metadata, **truth_boundary},
        )
    except ValueError as exc:
        return TSDFSurface(
            points_world_m=np.zeros((0, 3), dtype=FLOAT32),
            confidence=np.zeros((0,), dtype=FLOAT32),
            uncertainty_m=np.zeros((0,), dtype=FLOAT32),
            voxel_indices_xyz=np.zeros((0, 3), dtype=np.int32),
            metadata={**truth_boundary, "empty_surface_reason": str(exc), "surface_count": 0},
        )


def student_sparse_metrics(
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    *,
    truth_boundary: Mapping[str, object],
) -> dict[str, object]:
    return {
        **truth_boundary,
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "mapper_backend": MAPPER_BACKEND,
        "metric_family": "core_smgt_tiny_rgb_student_sparse_tsdf_diagnostic",
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }


def empty_student_mesh_status(
    *,
    mesh_format: str,
    truth_boundary: Mapping[str, object],
) -> dict[str, object]:
    return {
        **truth_boundary,
        "active_mesh_chunk_count": 0,
        "deleted_mesh_chunk_count": 0,
        "mesh_chunk_count": 0,
        "mesh_chunk_update_count": 0,
        "mesh_exported": False,
        "mesh_format": mesh_format,
        "total_triangle_count": 0,
        "total_vertex_count": 0,
    }


def require_nonzero_student_mesh(
    output: Path,
    summary: Mapping[str, object],
    mesh_format: str,
) -> None:
    if int(cast(Any, summary["mesh_chunk_count"])) <= 0:
        raise RuntimeError("RGB student mapping produced no observed mesh chunks")
    vertices = int(cast(Any, summary["total_vertex_count"]))
    triangles = int(cast(Any, summary["total_triangle_count"]))
    if vertices <= 0 or triangles <= 0:
        raise RuntimeError("RGB student mapping produced empty mesh chunk geometry")
    manifest = cast(dict[str, Any], _read_json(output / "mesh_chunks" / "mesh_chunk_manifest.json"))
    chunks = cast(list[dict[str, Any]], manifest.get("chunks", []))
    if not chunks:
        raise RuntimeError("RGB student mapping mesh manifest has no active chunks")
    first = chunks[0]
    chunk = load_mesh_chunk_npz(output / "mesh_chunks" / str(first["payload_npz"]))
    if chunk.vertex_count <= 0 or chunk.triangle_count <= 0:
        raise RuntimeError("RGB student mapping first NPZ mesh chunk is empty")
    if mesh_format in {"ply", "both"}:
        ply_rel = first.get("payload_ply")
        if not isinstance(ply_rel, str) or not (output / "mesh_chunks" / ply_rel).is_file():
            raise RuntimeError("RGB student mapping did not write a loadable PLY chunk")


def student_known_limitations() -> list[str]:
    return [
        "SMGT-tiny is a diagnostic learned student, not the final SMGT architecture.",
        "Metric scale comes from an unverified RGB prior learned from pseudo labels.",
        "No measured depth or pose is consumed by mapping; measured truth is eval-only if present.",
        "No hidden geometry completion, object-aware fusion, loop closure, realtime, "
        "or accuracy claim is made.",
    ]


def _read_json(path: Path) -> object:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "MAPPER_BACKEND",
    "SUMMARY_FORMAT_NAME",
    "empty_student_mesh_status",
    "extract_or_empty_student_surface",
    "require_nonzero_student_mesh",
    "student_live_replay_summary",
    "student_sparse_metrics",
    "student_summary",
    "write_student_report",
]
