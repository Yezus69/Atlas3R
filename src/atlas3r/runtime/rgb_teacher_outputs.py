"""Artifact and report helpers for RGB teacher mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping.cpu_tsdf import FLOAT32, TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper
from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    recording_truth_boundary,
    write_recording_files,
)
from atlas3r.runtime.rgb_teacher_conversion import RGB_TEACHER_TRUTH_FLAGS
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherFrame, RGBTeacherInput
from atlas3r.runtime.student_map_report_common import write_jsonl
from atlas3r.teachers.external.vggt_model import VGGTRawClipPredictor

RGB_TEACHER_SUMMARY_FORMAT_NAME = "atlas3r_rgb_teacher_mapping_summary"
MAPPER_BACKEND = "cpu-sparse"


def rgb_teacher_truth_boundary(*, metric_scale_source: str) -> dict[str, object]:
    return {
        **RGB_TEACHER_TRUTH_FLAGS,
        "metric_scale_source": metric_scale_source,
        "teacher_name": "vggt",
    }


def write_rgb_frames(output: Path, frames: Sequence[RGBTeacherFrame]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        rel = f"frame_{frame.frame_id:06d}.npz"
        np.savez_compressed(output / rel, rgb_u8=frame.rgb_u8, K=frame.K)
        records.append(
            {
                "frame_id": frame.frame_id,
                "index": index,
                "path": rel,
                "source_path": frame.source_path,
                "timestamp_s": frame.timestamp_s,
            }
        )
    write_jsonl(output / "frames.jsonl", records)


def write_prediction_payload(path: Path, payload: Mapping[str, Any]) -> None:
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)


def write_pseudo_recording(
    output: Path,
    source: RGBTeacherInput,
    observations: Sequence[DepthObservation],
    *,
    metric_scale_source: str,
) -> None:
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    truth = {
        **recording_truth_boundary(depth_present=True, pose_present=True),
        **rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source),
    }
    frames: list[dict[str, object]] = []
    for observation in observations:
        rgb_path = f"rgb/frame_{observation.frame_id:06d}.npz"
        depth_path = f"depth/frame_{observation.frame_id:06d}.npz"
        if observation.rgb_u8 is None:
            raise ValueError(f"frame {observation.frame_id}: pseudo recording requires RGB")
        np.savez_compressed(output / rgb_path, rgb_u8=observation.rgb_u8)
        np.savez_compressed(
            output / depth_path,
            depth_m=observation.depth_m.astype(np.float32, copy=False),
            valid_depth_mask=np.asarray(observation.static_mask, dtype=np.bool_),
            confidence=observation.confidence.astype(np.float32, copy=False),
        )
        frames.append(
            {
                "K": observation.camera.K.astype(float).tolist(),
                "T_world_camera": observation.pose.T_world_camera.astype(float).tolist(),
                "camera_center_world_m": observation.pose.camera_center_world_m.astype(
                    float
                ).tolist(),
                "depth_path": depth_path,
                "frame_id": observation.frame_id,
                "rgb_path": rgb_path,
                "source_metadata": {
                    "source": observation.source,
                    "truth_boundary": truth,
                },
                "timestamp_s": observation.pose.timestamp_ns / 1_000_000_000.0,
            }
        )
    manifest = {
        "capture_metadata": {
            "source_input": str(source.root),
            "teacher_name": "vggt",
            "pseudo_recording": True,
        },
        "coordinate_frame": RECORDING_COORDINATE_FRAME,
        "depth_present": True,
        "format_name": RECORDING_FORMAT_NAME,
        "format_version": RECORDING_FORMAT_VERSION,
        "frame_count": len(frames),
        "height": source.height,
        "known_calibration_metadata": {},
        "pose_present": True,
        "source_dataset": "rgb_teacher_pseudo",
        "source_sequence": source.source_sequence,
        "truth_boundary": truth,
        "width": source.width,
    }
    write_recording_files(output, manifest=manifest, frames=frames)


def rgb_teacher_summary(
    *,
    metric_scale_source: str,
    input_path: Path,
    output: Path,
    source: RGBTeacherInput,
    frames: Sequence[RGBTeacherFrame],
    observations: Sequence[DepthObservation],
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    mesh_status: Mapping[str, object],
    mesh_manifest: Mapping[str, object] | None,
    point_cloud_path: Path | None,
    teacher_runtime_ns: int,
    predictor_metadata: Mapping[str, object],
    prediction_metadata: Mapping[str, object],
    latency_report: Mapping[str, object],
) -> dict[str, object]:
    truth = rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source)
    latency_segments = cast(Mapping[str, Any], latency_report["segments"])
    total_pipeline = cast(Mapping[str, Any], latency_segments["total_pipeline"])
    return {
        **truth,
        "active_blocks": mapper.active_block_count,
        "active_voxels": mapper.active_voxel_count,
        "artifacts": {
            "live_replay_events": "live_replay_events.jsonl",
            "live_replay_report": "live_replay_report.md",
            "live_replay_summary": "live_replay_summary.json",
            "mesh_chunks": "mesh_chunks",
            "mesh_manifest": (
                None if mesh_manifest is None else "mesh_chunks/mesh_chunk_manifest.json"
            ),
            "point_cloud": None if point_cloud_path is None else point_cloud_path.name,
            "rgb_teacher_predictions": "rgb_teacher_predictions",
            "rgb_teacher_recording": "rgb_teacher_recording",
            "sparse_tsdf": "sparse_tsdf",
        },
        "format_name": RGB_TEACHER_SUMMARY_FORMAT_NAME,
        "format_version": 1,
        "frame_count_available": len(source.frames),
        "frame_count_used": len(frames),
        "input": str(input_path),
        "input_source_type": source.source_type,
        "known_limitations": known_limitations(),
        "latency": dict(latency_segments),
        "map_uncertainty_summary_m": surface_uncertainty_summary(surface),
        "mapper_backend": MAPPER_BACKEND,
        "measured_depth_available_for_eval": source.measured_depth_available,
        "measured_pose_available_for_eval": source.measured_pose_available,
        "mesh": dict(mesh_status),
        "mesh_chunk_count": int(cast(Any, mesh_status.get("mesh_chunk_count", 0))),
        "mesh_uncertainty_summary_m": mesh_uncertainty_summary(mesh_manifest),
        "metric_scale_source": metric_scale_source,
        "output": str(output),
        "pseudo_depth_count": len(observations),
        "pseudo_depth_valid_pixel_ratio": prediction_metadata["pseudo_depth_valid_pixel_ratio"],
        "pseudo_intrinsics_count": len(observations),
        "pseudo_pose_count": len(observations),
        "sparse_map_update_count": len(observations),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "teacher_metadata": dict(predictor_metadata),
        "teacher_name": "vggt",
        "teacher_runtime_ms": teacher_runtime_ns / 1_000_000.0,
        "total_pipeline_time_ms": total_pipeline["max_ms"],
        "total_triangle_count": int(cast(Any, mesh_status.get("total_triangle_count", 0))),
        "total_vertex_count": int(cast(Any, mesh_status.get("total_vertex_count", 0))),
        "truth_boundary": truth,
        "voxel_size_m": mapper.config.voxel_size_m,
        **prediction_metadata,
    }


def live_replay_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        **dict(summary),
        "format_name": "atlas3r_rgb_teacher_live_replay_summary",
        "measured_depth_used": False,
        "measured_pose_used": False,
    }


def write_rgb_teacher_report(path: Path, summary: Mapping[str, object]) -> None:
    truth = cast(Mapping[str, object], summary["truth_boundary"])
    lines = [
        "# Phase 6G RGB Teacher Mapping Report",
        "",
        f"- Input: `{summary['input']}`",
        f"- Teacher: `{summary['teacher_name']}`",
        f"- Frames used: `{summary['frame_count_used']}`",
        f"- Teacher runtime ms: `{summary['teacher_runtime_ms']}`",
        f"- Pseudo poses/depths/intrinsics: `{summary['pseudo_pose_count']}` / "
        f"`{summary['pseudo_depth_count']}` / `{summary['pseudo_intrinsics_count']}`",
        f"- Pseudo depth valid pixel ratio: `{summary['pseudo_depth_valid_pixel_ratio']}`",
        f"- Sparse map updates: `{summary['sparse_map_update_count']}`",
        f"- Active blocks/voxels: `{summary['active_blocks']}` / `{summary['active_voxels']}`",
        f"- Mesh chunks: `{summary['mesh_chunk_count']}`",
        f"- Mesh vertices/triangles: `{summary['total_vertex_count']}` / "
        f"`{summary['total_triangle_count']}`",
        f"- Measured depth used for mapping: `{truth['measured_depth_used']}`",
        f"- Measured pose used for mapping: `{truth['measured_pose_used']}`",
        f"- Metric scale source: `{summary['metric_scale_source']}`",
        "",
        "This is offline teacher-pseudo RGB mapping evidence. Depth and pose are "
        "pseudo labels from the teacher, not measured physical truth. The output is "
        "diagnostic-only, observed-only, not RGB-only student mapping, not realtime, "
        "not hidden-geometry completion, and not an accuracy report.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_or_empty_surface(
    mapper: SparseBlockTSDFMapper,
    *,
    metric_scale_source: str,
) -> TSDFSurface:
    truth = rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source)
    try:
        surface = mapper.extract_surface()
        metadata = dict(surface.metadata)
        metadata.update(truth)
        return TSDFSurface(
            points_world_m=surface.points_world_m,
            confidence=surface.confidence,
            uncertainty_m=surface.uncertainty_m,
            voxel_indices_xyz=surface.voxel_indices_xyz,
            metadata=metadata,
        )
    except ValueError as exc:
        return TSDFSurface(
            points_world_m=np.zeros((0, 3), dtype=FLOAT32),
            confidence=np.zeros((0,), dtype=FLOAT32),
            uncertainty_m=np.zeros((0,), dtype=FLOAT32),
            voxel_indices_xyz=np.zeros((0, 3), dtype=np.int32),
            metadata={**truth, "empty_surface_reason": str(exc), "surface_count": 0},
        )


def sparse_metrics(
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    *,
    metric_scale_source: str,
) -> dict[str, object]:
    return {
        **rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source),
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "mapper_backend": MAPPER_BACKEND,
        "metric_family": "phase6g_rgb_teacher_sparse_tsdf_diagnostic",
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }


def empty_mesh_status(*, mesh_format: str, metric_scale_source: str) -> dict[str, object]:
    return {
        **rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source),
        "active_mesh_chunk_count": 0,
        "deleted_mesh_chunk_count": 0,
        "mesh_chunk_count": 0,
        "mesh_chunk_update_count": 0,
        "mesh_exported": False,
        "mesh_format": mesh_format,
        "total_triangle_count": 0,
        "total_vertex_count": 0,
    }


def require_nonzero_mesh(summary: Mapping[str, object]) -> None:
    if (
        int(cast(Any, summary["mesh_chunk_count"])) <= 0
        or int(cast(Any, summary["total_vertex_count"])) <= 0
        or int(cast(Any, summary["total_triangle_count"])) <= 0
    ):
        raise RuntimeError("RGB teacher mapping produced no nonzero observed mesh chunks")


def surface_uncertainty_summary(surface: TSDFSurface) -> dict[str, float | None]:
    return numeric_summary(surface.uncertainty_m.astype(np.float64, copy=False))


def mesh_uncertainty_summary(manifest: Mapping[str, object] | None) -> dict[str, float | None]:
    if manifest is None:
        return {"max": None, "mean": None, "p50": None, "p95": None}
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        return {"max": None, "mean": None, "p50": None, "p95": None}
    values: list[float] = []
    for item in chunks:
        if isinstance(item, Mapping):
            summary = item.get("uncertainty_summary_m")
            if isinstance(summary, Mapping) and isinstance(summary.get("p95"), int | float):
                values.append(float(summary["p95"]))
    return numeric_summary(np.asarray(values, dtype=np.float64))


def numeric_summary(values: np.ndarray[Any, Any]) -> dict[str, float | None]:
    if values.size == 0:
        return {"max": None, "mean": None, "p50": None, "p95": None}
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def known_limitations() -> list[str]:
    return [
        "Teacher depth and pose are pseudo labels, not measured geometry.",
        "Monocular teacher scale is recorded as unverified unless separately evaluated.",
        "Windowed teacher predictions are not loop-closed or globally optimized here.",
        "Sparse TSDF and fallback meshing are diagnostic CPU paths, not realtime runtime.",
        "No hidden geometry, object-aware fusion, benchmark accuracy, or millimeter claim is made.",
    ]


__all__ = [
    "MAPPER_BACKEND",
    "RGB_TEACHER_SUMMARY_FORMAT_NAME",
    "VGGTRawClipPredictor",
    "empty_mesh_status",
    "extract_or_empty_surface",
    "live_replay_summary",
    "require_nonzero_mesh",
    "rgb_teacher_summary",
    "rgb_teacher_truth_boundary",
    "sparse_metrics",
    "write_prediction_payload",
    "write_pseudo_recording",
    "write_rgb_frames",
    "write_rgb_teacher_report",
]
