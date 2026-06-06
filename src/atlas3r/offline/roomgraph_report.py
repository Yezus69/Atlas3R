"""RoomGraph report rendering."""

from __future__ import annotations

from typing import Any, cast


def render_roomgraph_report_markdown(payload: dict[str, object]) -> str:
    before = cast(dict[str, Any], payload["before"])
    after = cast(dict[str, Any], payload["after"])
    improvement = cast(dict[str, float], payload["improvement"])
    trajectory_line = (
        f"- Camera trajectory length m: {before.get('camera_trajectory_length_m')} -> "
        f"{after['camera_trajectory_length_m']}"
    )
    camera_bbox_line = (
        f"- Camera bbox size m: {before.get('camera_bbox_size_m')} -> {after['camera_bbox_size_m']}"
    )
    mesh_line = (
        f"- Mesh triangles: {before['observed_mesh_triangle_count']} -> "
        f"{after['observed_mesh_triangle_count']}"
    )
    reproj_line = (
        f"- Reprojection error mean px: {before.get('reprojection_error_mean_px')} -> "
        f"{after['reprojection_error_mean_px']}"
    )
    depth_line = (
        f"- Cross-view depth residual mean m: "
        f"{before.get('cross_view_depth_residual_mean_m')} -> "
        f"{after['cross_view_depth_residual_mean_m']}"
    )
    teacher_line = (
        f"- Mapped teacher disagreement rel mean: "
        f"{before.get('mapped_teacher_disagreement_rel_mean')} -> "
        f"{after['mapped_teacher_disagreement_rel_mean']}"
    )
    collapse_delta = improvement.get("camera_collapse_score_m")
    collapse_summary = (
        "not improved" if isinstance(collapse_delta, float) and collapse_delta < 0.1 else "improved"
    )
    return "\n".join(
        [
            "# RoomGraph Report",
            "",
            f"- Status: {payload['status']}",
            f"- Track source used: {payload['track_source']}",
            f"- Selected optimizer variant: {payload['selected_variant']}",
            f"- Tracks kept as inliers: {after['track_inlier_ratio']}",
            trajectory_line,
            camera_bbox_line,
            f"- Map bbox size m: {before.get('map_bbox_size_m')} -> {after['map_bbox_size_m']}",
            f"- Point count: {before['fused_point_count']} -> {after['fused_point_count']}",
            f"- Voxel count: {before['occupied_voxel_count']} -> {after['occupied_voxel_count']}",
            mesh_line,
            reproj_line,
            depth_line,
            teacher_line,
            f"- Improvement ratios: {improvement}",
            f"- Collapse verdict: {collapse_summary}",
            "- Physical accuracy claim: false",
            "- Training-quality claim: false",
            "",
            "This is unanchored teacher/optimizer geometry, not measured physical truth.",
            "",
        ]
    )
