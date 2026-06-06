"""Quality and manifest writers for Atlas3R ViPE imports."""

from __future__ import annotations

from pathlib import Path

from atlas3r.offline.fused_world_map_artifacts import (
    FusedPointCloud,
    ObservedVoxelMesh,
    SparseOccupancyGrid,
    bbox,
    manifest_payload,
)
from atlas3r.offline.run_manifest import utc_now_iso, write_json


def write_vipe_quality_and_manifest(
    output: Path,
    *,
    frames: Path,
    vipe_output: Path,
    artifact_name: str,
    point_stride: int,
    max_points: int,
    voxel_size_m: float,
    write_observed_mesh: bool,
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    mesh: ObservedVoxelMesh,
    trajectory_count: int,
    rejected_pose_count: int,
    depth_frame_count: int,
    frame_count: int,
    paths: dict[str, str],
    failures: list[str],
    truth_boundary: dict[str, object],
) -> dict[str, object]:
    bbox_min, bbox_max = bbox(cloud.points_world_m)
    bbox_size = None if bbox_min is None or bbox_max is None else (bbox_max - bbox_min).tolist()
    inspectable = (
        "fused_points_ply" in paths
        and occupancy.occupied_voxel_count > 0
        and cloud.points_world_m.shape[0] > 0
        and (not write_observed_mesh or mesh.triangle_count > 0)
    )
    quality: dict[str, object] = {
        "format_name": "atlas3r_vipe_map_quality_report",
        "format_version": 1,
        "created_utc": utc_now_iso(),
        "input_source": str(frames),
        "vipe_output_source": str(vipe_output),
        "frames_available": frame_count,
        "vipe_depth_frame_count": depth_frame_count,
        "depth_source_used": cloud.depth_source,
        "fused_point_count": int(cloud.points_world_m.shape[0]),
        "occupied_voxel_count": occupancy.occupied_voxel_count,
        "observed_mesh_vertex_count": int(mesh.vertices_world_m.shape[0]),
        "observed_mesh_triangle_count": mesh.triangle_count,
        "camera_trajectory_count": trajectory_count,
        "rejected_nonfinite_pose_count": rejected_pose_count,
        "bbox_size_m": bbox_size,
        "valid_depth_ratio": cloud.valid_depth_ratio,
        "rejected_low_confidence_pixel_ratio": 0.0,
        "rejected_high_disagreement_pixel_ratio": 0.0,
        "mapped_disagreement": {"mean": None, "p50": None, "p95": None},
        "scale_source": truth_boundary["metric_scale_source"],
        "physical_accuracy_claim": False,
        "training_quality_claim": False,
        "known_failure_points": failures,
        "inspectable_map_available": inspectable,
        "truth_boundary": truth_boundary,
        "artifacts": paths,
    }
    write_json(output / "map_quality.json", quality)
    (output / "map_quality.md").write_text(_vipe_quality_markdown(quality), encoding="utf-8")
    _write_world_map_manifest(
        output,
        cloud=cloud,
        occupancy=occupancy,
        voxel_size_m=voxel_size_m,
        paths=paths,
        failures=failures,
        truth_boundary=truth_boundary,
    )
    _write_import_manifest(
        output,
        artifact_name=artifact_name,
        vipe_output=vipe_output,
        frames=frames,
        point_stride=point_stride,
        max_points=max_points,
        voxel_size_m=voxel_size_m,
        truth_boundary=truth_boundary,
        paths=paths,
        failures=failures,
    )
    return quality


def _write_world_map_manifest(
    output: Path,
    *,
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    voxel_size_m: float,
    paths: dict[str, str],
    failures: list[str],
    truth_boundary: dict[str, object],
) -> None:
    write_json(
        output / "world_map_manifest.json",
        manifest_payload(
            cloud,
            occupancy,
            voxel_size_m=voxel_size_m,
            min_confidence=1.0,
            max_relative_disagreement=-1.0,
            paths=paths,
            failure_reasons=failures,
            truth_boundary=truth_boundary,
        ),
    )


def _write_import_manifest(
    output: Path,
    *,
    artifact_name: str,
    vipe_output: Path,
    frames: Path,
    point_stride: int,
    max_points: int,
    voxel_size_m: float,
    truth_boundary: dict[str, object],
    paths: dict[str, str],
    failures: list[str],
) -> None:
    write_json(
        output / "vipe_import_manifest.json",
        {
            "format_name": "atlas3r_vipe_import",
            "format_version": 1,
            "created_utc": utc_now_iso(),
            "artifact_name": artifact_name,
            "vipe_output": str(vipe_output),
            "frames": str(frames),
            "output": str(output),
            "point_stride": point_stride,
            "max_points": max_points,
            "voxel_size_m": voxel_size_m,
            "truth_boundary": truth_boundary,
            "artifacts": paths,
            "known_failure_points": failures,
        },
    )


def _vipe_quality_markdown(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# ViPE Map Quality Report",
            "",
            f"- Inspectable map available: {payload['inspectable_map_available']}",
            f"- Depth source used: {payload['depth_source_used']}",
            f"- Source RGB frames available: {payload['frames_available']}",
            f"- ViPE depth frames: {payload['vipe_depth_frame_count']}",
            f"- Fused points: {payload['fused_point_count']}",
            f"- Occupied voxels: {payload['occupied_voxel_count']}",
            f"- Observed mesh vertices: {payload['observed_mesh_vertex_count']}",
            f"- Observed mesh triangles: {payload['observed_mesh_triangle_count']}",
            f"- Camera trajectory poses: {payload['camera_trajectory_count']}",
            f"- Valid depth ratio: {payload['valid_depth_ratio']}",
            f"- Physical accuracy claim: {payload['physical_accuracy_claim']}",
            f"- Training-quality claim: {payload['training_quality_claim']}",
            "",
            "This map is ViPE teacher-pseudo, observed-only, and not an accuracy or "
            "training-quality report.",
            "",
        ]
    )


__all__ = ["write_vipe_quality_and_manifest"]
