"""Dependency-free top-down preview and inspection note writers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.fused_world_map_artifacts import FusedPointCloud


def write_topdown_preview(
    path: Path, cloud: FusedPointCloud, trajectory: list[dict[str, object]]
) -> None:
    points = cloud.points_world_m
    centers = np.asarray(
        [pose.get("camera_center_world_m", [0.0, 0.0, 0.0]) for pose in trajectory],
        dtype=np.float32,
    )
    projected = _project_xz(points, centers)
    path.write_text(_svg_document(projected, points, centers), encoding="utf-8")


def write_inspection_instructions(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Room Map Inspection",
                "",
                "Open these first:",
                "",
                "- `world_map_best/fused_points.ply`",
                "- `world_map_best/observed_voxel_mesh.ply`",
                "- `world_map_best/topdown_preview.svg`",
                "- `world_map_best/camera_trajectory.json`",
                "- `world_map_best/map_quality.md`",
                "",
                "CloudCompare, MeshLab, or Blender can inspect the PLY files. The map is an "
                "unanchored soft-metric teacher-consensus map, not physical ground truth.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _project_xz(
    points: NDArray[np.float32], centers: NDArray[np.float32]
) -> dict[str, NDArray[np.float32]]:
    point_xz = points[:, [0, 2]] if points.size else np.zeros((0, 2), dtype=np.float32)
    center_xz = (
        centers[:, [0, 2]]
        if centers.ndim == 2 and centers.size
        else np.zeros((0, 2), dtype=np.float32)
    )
    chunks = [item for item in (point_xz, center_xz) if item.size]
    all_xz = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 2), dtype=np.float32)
    min_xy = all_xz.min(axis=0)
    max_xy = all_xz.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)

    def project(values: NDArray[np.float32]) -> NDArray[np.float32]:
        if values.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        normalized = (values - min_xy) / span
        out = np.empty_like(normalized)
        out[:, 0] = 40.0 + normalized[:, 0] * 720.0
        out[:, 1] = 560.0 - normalized[:, 1] * 520.0
        return cast(NDArray[np.float32], out.astype(np.float32))

    return {
        "points": project(point_xz),
        "centers": project(center_xz),
        "span": span,
    }


def _svg_document(
    projected: dict[str, NDArray[np.float32]],
    points: NDArray[np.float32],
    centers: NDArray[np.float32],
) -> str:
    projected_points = projected["points"]
    if projected_points.shape[0] > 2000:
        keep = np.linspace(0, projected_points.shape[0] - 1, 2000).round().astype(np.int64)
        projected_points = projected_points[keep]
    circles = "\n".join(
        f'<circle class="map-point" cx="{x:.2f}" cy="{y:.2f}" r="1.2" />'
        for x, y in projected_points
    )
    centers_xy = projected["centers"]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in centers_xy)
    bbox_size = " x ".join(f"{float(value):.2f}" for value in projected["span"])
    camera_count = centers.shape[0] if centers.ndim == 2 else 0
    header = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="620" viewBox="0 0 800 620">'
    )
    footer_label = (
        f"Top-down XZ preview; soft metric bbox {bbox_size} m; "
        f"points={points.shape[0]}; cameras={camera_count}"
    )
    return f"""{header}
<title>Atlas3R top-down soft-metric preview</title>
<style>
  .map-point {{ fill: #2f6f8f; opacity: 0.35; }}
  .trajectory {{ fill: none; stroke: #c2410c; stroke-width: 3; }}
  .bbox {{ fill: none; stroke: #111827; stroke-width: 1.5; stroke-dasharray: 6 4; }}
  .label {{ font: 14px sans-serif; fill: #111827; }}
  .warn {{ font: 16px sans-serif; fill: #991b1b; font-weight: 700; }}
</style>
<rect x="40" y="40" width="720" height="520" class="bbox" />
<g id="map-points">{circles}</g>
<polyline id="camera-trajectory" class="trajectory" points="{polyline}" />
<text x="40" y="24" class="warn">unanchored soft-metric map, not physical ground truth</text>
<text x="40" y="590" class="label">{footer_label}</text>
</svg>
"""
