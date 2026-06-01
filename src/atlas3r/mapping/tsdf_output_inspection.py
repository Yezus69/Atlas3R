"""Deterministic inspection for CPU TSDF output folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas3r.mapping._tsdf_output_inspection_helpers import (
    artifact_presence,
    inspect_mesh_sidecar,
    inspect_world_map_sidecar,
    metrics_record,
    numeric_summary,
    stable_flags,
    surface_bounds_summary,
    surface_record,
    truth_boundary,
    validate_complete_truth_boundary,
)
from atlas3r.mapping.mesh_sidecar import MESH_SIDECAR_FILENAME, load_tsdf_surface_artifacts
from atlas3r.mapping.world_map_sidecar import WORLD_MAP_SIDECAR_FILENAME

TSDF_OUTPUT_INSPECTION_FORMAT_NAME = "atlas3r_cpu_tsdf_output_folder_inspection"
TSDF_OUTPUT_INSPECTION_FORMAT_VERSION = 1
TSDF_OUTPUT_INSPECTION_MODES = ("surface", "mesh", "world-map", "complete")


def tsdf_output_folder_inspection_record(
    output_folder: str | Path,
    *,
    mode: str = "complete",
) -> dict[str, Any]:
    """Validate a CPU TSDF output folder and return deterministic inspection JSON."""
    if mode not in TSDF_OUTPUT_INSPECTION_MODES:
        raise ValueError(
            f"mode: expected one of {', '.join(TSDF_OUTPUT_INSPECTION_MODES)}, got {mode!r}"
        )
    output_path = Path(output_folder)
    surface = load_tsdf_surface_artifacts(output_path)
    surface_summary = surface_record(
        metadata=surface.metadata,
        point_count=int(surface.points_world_m.shape[0]),
        bounds_summary=surface_bounds_summary(surface.points_world_m),
        confidence_summary=numeric_summary(surface.confidence),
        uncertainty_summary=numeric_summary(surface.uncertainty_m),
        metadata_path=output_path / "metadata.json",
    )
    metrics_summary = metrics_record(output_path / "metrics.json")

    require_mesh = mode in {"mesh", "world-map", "complete"}
    require_world_map = mode in {"world-map", "complete"}
    mesh_path = output_path / MESH_SIDECAR_FILENAME
    world_map_path = output_path / WORLD_MAP_SIDECAR_FILENAME
    mesh_summary: dict[str, Any] | None = None
    world_map_summary: dict[str, Any] | None = None
    checked_artifacts = ["metadata.json", "surface_points.npz"]

    if mesh_path.is_file():
        mesh_summary = inspect_mesh_sidecar(mesh_path, surface_summary, surface.confidence)
        checked_artifacts.append(MESH_SIDECAR_FILENAME)
    elif require_mesh:
        raise ValueError(f"{mesh_path}: missing required MeshChunk sidecar for {mode} inspect mode")

    if world_map_path.is_file():
        world_map_summary = inspect_world_map_sidecar(world_map_path, surface_summary, mesh_summary)
        checked_artifacts.append(WORLD_MAP_SIDECAR_FILENAME)
    elif require_world_map:
        raise ValueError(
            f"{world_map_path}: missing required WorldMap sidecar for {mode} inspect mode"
        )

    flags = stable_flags(
        [
            *surface_summary["flags"],
            *(mesh_summary or {}).get("flags", []),
            *(world_map_summary or {}).get("flags", []),
        ]
    )
    inspection = {
        "format_name": TSDF_OUTPUT_INSPECTION_FORMAT_NAME,
        "format_version": TSDF_OUTPUT_INSPECTION_FORMAT_VERSION,
        "inspect_mode": mode,
        "artifacts": {
            "metadata_json": {
                "path": "metadata.json",
                "present": True,
                "artifact_type": surface_summary["artifact_type"],
            },
            "surface_points_npz": {
                "path": "surface_points.npz",
                "present": True,
                "point_count": surface_summary["point_count"],
            },
            "metrics_json": metrics_summary,
            "mesh_chunk_sidecar_json": artifact_presence(
                MESH_SIDECAR_FILENAME,
                mesh_summary,
                "chunk_id",
            ),
            "world_map_sidecar_json": artifact_presence(
                WORLD_MAP_SIDECAR_FILENAME,
                world_map_summary,
                "map_id",
            ),
        },
        "surface": surface_summary,
        "mesh_chunk": mesh_summary,
        "world_map": world_map_summary,
        "cross_checks": {
            "artifacts_checked": checked_artifacts,
            "coordinate_frame": surface_summary["coordinate_frame"],
            "source_frame_ids": surface_summary["source_frame_ids"],
            "metric_scale_source": surface_summary["metric_scale_source"],
            "voxel_size_m": surface_summary["voxel_size_m"],
            "observed_coverage_estimate": surface_summary["observed_coverage_estimate"],
            "mean_uncertainty_m": surface_summary["uncertainty_summary_m"]["mean"],
            "p95_uncertainty_m": surface_summary["uncertainty_summary_m"]["p95"],
            "passed": True,
        },
        "truth_boundary": truth_boundary(flags, metrics_summary),
    }
    validate_complete_truth_boundary(inspection, output_path)
    return inspection


def format_tsdf_output_folder_inspection(
    output_folder: str | Path,
    *,
    mode: str = "complete",
) -> str:
    """Format deterministic CPU TSDF output folder inspection JSON."""
    return (
        json.dumps(
            tsdf_output_folder_inspection_record(output_folder, mode=mode),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


__all__ = [
    "TSDF_OUTPUT_INSPECTION_FORMAT_NAME",
    "TSDF_OUTPUT_INSPECTION_FORMAT_VERSION",
    "TSDF_OUTPUT_INSPECTION_MODES",
    "format_tsdf_output_folder_inspection",
    "tsdf_output_folder_inspection_record",
]
