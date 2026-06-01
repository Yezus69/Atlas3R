"""Write deterministic SVG/HTML previews for Phase 0B sessions."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api import ObjectInstance, PoseEstimate
from atlas3r.io.session import LoadedSession, load_depth_npz, validate_session


def write_session_preview(session_path: str | Path, output_dir: str | Path) -> tuple[Path, ...]:
    """Write deterministic inspection previews for a Phase 0B `.atlas3r` session."""
    session = validate_session(session_path)
    if not session.depth_files:
        raise ValueError(f"{session.root / 'depth'}: expected at least one frame_*.npz file")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    first_frame_id = _frame_id_from_depth_path(session.depth_files[0])
    depth_payload = load_depth_npz(session.depth_files[0])

    top_down = output / "top_down.svg"
    depth_svg = output / f"depth_frame_{first_frame_id:06d}.svg"
    mask_svg = output / f"object_mask_frame_{first_frame_id:06d}.svg"
    index_html = output / "index.html"

    _write_text(top_down, _top_down_svg(session))
    _write_text(depth_svg, _depth_svg(depth_payload, session.depth_files[0]))
    _write_text(mask_svg, _object_mask_svg(depth_payload, session.depth_files[0]))
    _write_text(index_html, _index_html(session, (top_down, depth_svg, mask_svg)))
    return (index_html, top_down, depth_svg, mask_svg)


@dataclass(frozen=True)
class _Projector:
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    width: int
    height: int
    margin: float

    def point(self, x_world: float, z_world: float) -> tuple[float, float]:
        x_range = max(self.max_x - self.min_x, 1e-6)
        z_range = max(self.max_z - self.min_z, 1e-6)
        x = self.margin + ((x_world - self.min_x) / x_range) * (self.width - 2.0 * self.margin)
        y = (
            self.height
            - self.margin
            - (((z_world - self.min_z) / z_range) * (self.height - 2.0 * self.margin))
        )
        return x, y


def _top_down_svg(session: LoadedSession) -> str:
    width = 640
    height = 480
    projector = _projector_for_session(session, width, height)
    elements = [
        '<rect x="0" y="0" width="640" height="480" fill="#fbfaf7"/>',
        '<text x="24" y="32" font-size="18" font-family="Arial">'
        "Atlas3R session top-down preview</text>",
    ]
    room = _mesh_bounds_xz(session)
    if room is not None:
        x0, y0 = projector.point(room[0], room[2])
        x1, y1 = projector.point(room[1], room[3])
        elements.append(
            "<rect "
            f'x="{_fmt(min(x0, x1))}" y="{_fmt(min(y0, y1))}" '
            f'width="{_fmt(abs(x1 - x0))}" height="{_fmt(abs(y1 - y0))}" '
            'fill="#eef6f1" stroke="#2f6f5e" stroke-width="2"/>'
        )

    for instance in sorted(session.objects, key=lambda item: item.object_id):
        footprint = _object_footprint_xz(instance)
        points = [_point_attr(projector, x_value, z_value) for x_value, z_value in footprint]
        elements.append(
            '<polygon points="'
            + " ".join(points)
            + '" fill="#f2c14e" fill-opacity="0.45" stroke="#9a6b00" stroke-width="2"/>'
        )
        label_x, label_y = projector.point(
            float(instance.oriented_bbox_center_m[0]), float(instance.oriented_bbox_center_m[2])
        )
        label = _object_label(instance)
        elements.append(
            f'<text x="{_fmt(label_x)}" y="{_fmt(label_y - 8.0)}" '
            f'font-size="14" font-family="Arial">{escape(label)}</text>'
        )

    for pose in sorted(session.poses, key=lambda item: item.frame_id):
        elements.extend(_camera_elements(projector, pose))

    return _svg_document(width, height, elements)


def _depth_svg(payload: dict[str, npt.NDArray[Any]], path: Path) -> str:
    if "depth_m" not in payload:
        raise ValueError(f"{path}: missing depth_m array")
    depth = np.asarray(payload["depth_m"], dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"{path}: depth_m must be a 2D array")
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError(f"{path}: depth_m contains no finite values")
    min_depth = float(np.min(finite))
    max_depth = float(np.max(finite))
    elements = _grid_rects(depth, lambda value: _depth_color(value, min_depth, max_depth))
    return _svg_document(depth.shape[1] * 12, depth.shape[0] * 12, elements)


def _object_mask_svg(payload: dict[str, npt.NDArray[Any]], path: Path) -> str:
    if "object_id" in payload:
        object_id = np.asarray(payload["object_id"], dtype=np.int32)
    elif "object_mask" in payload:
        object_id = np.asarray(payload["object_mask"], dtype=bool).astype(np.int32)
    else:
        raise ValueError(f"{path}: missing object_id or object_mask array")
    if object_id.ndim != 2:
        raise ValueError(f"{path}: object id grid must be a 2D array")
    elements = _grid_rects(object_id, lambda value: _object_color(int(value)))
    return _svg_document(object_id.shape[1] * 12, object_id.shape[0] * 12, elements)


def _index_html(session: LoadedSession, previews: tuple[Path, Path, Path]) -> str:
    warnings = session.metadata.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [warnings]
    metadata_rows = []
    for key in (
        "session_type",
        "unit_scale",
        "scale_source",
        "camera_metadata_source",
        "voxel_size_m",
        "accuracy_report_path",
    ):
        metadata_rows.append(
            f"<tr><th>{escape(key)}</th><td>{escape(str(session.metadata.get(key)))}</td></tr>"
        )
    warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in warnings)
    warning_items += (
        "<li>Diagnostic preview only; hidden or completed geometry is not measured geometry.</li>"
    )
    top_down, depth_svg, mask_svg = (path.name for path in previews)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Atlas3R Session Preview</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933;background:#fbfaf7}",
            "section{margin:24px 0} img{border:1px solid #c9d1d9;background:white;max-width:100%}",
            "table{border-collapse:collapse}th,td{border:1px solid #c9d1d9;padding:6px 10px}",
            ".warning{color:#7a4f01}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Atlas3R Session Preview</h1>",
            f"<p>Frames: {session.frame_count} | Objects: {len(session.objects)} | "
            f"Mesh chunks: {len(session.mesh_chunks)}</p>",
            "<table>",
            *metadata_rows,
            "</table>",
            '<section class="warning"><h2>Warnings</h2><ul>',
            warning_items,
            "</ul></section>",
            _html_image_section("Top Down", top_down, "top down preview"),
            _html_image_section("Depth Frame 000000", depth_svg, "depth preview"),
            _html_image_section("Object Mask Frame 000000", mask_svg, "object mask preview"),
            "</body>",
            "</html>",
            "",
        ]
    )


def _projector_for_session(session: LoadedSession, width: int, height: int) -> _Projector:
    bounds = _collect_xz_bounds(session)
    min_x, max_x, min_z, max_z = bounds
    x_pad = max((max_x - min_x) * 0.08, 0.25)
    z_pad = max((max_z - min_z) * 0.08, 0.25)
    return _Projector(
        min_x=min_x - x_pad,
        max_x=max_x + x_pad,
        min_z=min_z - z_pad,
        max_z=max_z + z_pad,
        width=width,
        height=height,
        margin=56.0,
    )


def _html_image_section(title: str, image_name: str, alt: str) -> str:
    return (
        f"<section><h2>{escape(title)}</h2>"
        f'<img src="{escape(image_name)}" alt="{escape(alt)}"></section>'
    )


def _collect_xz_bounds(session: LoadedSession) -> tuple[float, float, float, float]:
    x_values: list[float] = []
    z_values: list[float] = []
    room = _mesh_bounds_xz(session)
    if room is not None:
        x_values.extend([room[0], room[1]])
        z_values.extend([room[2], room[3]])
    for instance in session.objects:
        for x_value, z_value in _object_footprint_xz(instance):
            x_values.append(x_value)
            z_values.append(z_value)
    for pose in session.poses:
        x_values.append(float(pose.camera_center_world_m[0]))
        z_values.append(float(pose.camera_center_world_m[2]))
    if not x_values or not z_values:
        return (-1.0, 1.0, 0.0, 1.0)
    return (min(x_values), max(x_values), min(z_values), max(z_values))


def _mesh_bounds_xz(session: LoadedSession) -> tuple[float, float, float, float] | None:
    vertex_arrays = [chunk.vertices_m for chunk in session.mesh_chunks if chunk.vertices_m.size > 0]
    if not vertex_arrays:
        return None
    vertices = np.vstack(vertex_arrays).astype(np.float64, copy=False)
    return (
        float(np.min(vertices[:, 0])),
        float(np.max(vertices[:, 0])),
        float(np.min(vertices[:, 2])),
        float(np.max(vertices[:, 2])),
    )


def _object_footprint_xz(instance: ObjectInstance) -> tuple[tuple[float, float], ...]:
    center = instance.oriented_bbox_center_m.astype(np.float64, copy=False)
    axes = instance.oriented_bbox_axes.astype(np.float64, copy=False)
    half_extents = instance.oriented_bbox_extents_m.astype(np.float64, copy=False) * 0.5
    corners: list[tuple[float, float]] = []
    for x_sign, z_sign in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        corner = (
            center + axes[:, 0] * half_extents[0] * x_sign + axes[:, 2] * half_extents[2] * z_sign
        )
        corners.append((float(corner[0]), float(corner[2])))
    return tuple(corners)


def _camera_elements(projector: _Projector, pose: PoseEstimate) -> list[str]:
    center = pose.camera_center_world_m.astype(np.float64, copy=False)
    rotation = pose.T_world_camera[:3, :3].astype(np.float64, copy=False)
    forward = rotation[:, 2]
    right = rotation[:, 0]
    origin = np.array([center[0], center[2]], dtype=np.float64)
    ray_ends = (
        origin + np.array([forward[0], forward[2]]) * 0.55,
        origin + np.array([forward[0] + right[0] * 0.25, forward[2] + right[2] * 0.25]) * 0.45,
        origin + np.array([forward[0] - right[0] * 0.25, forward[2] - right[2] * 0.25]) * 0.45,
    )
    ox, oy = projector.point(float(origin[0]), float(origin[1]))
    elements = [
        f'<circle cx="{_fmt(ox)}" cy="{_fmt(oy)}" r="6" fill="#2563eb"/>',
        f'<text x="{_fmt(ox + 9.0)}" y="{_fmt(oy - 9.0)}" '
        f'font-size="13" font-family="Arial">{pose.frame_id}</text>',
    ]
    for end in ray_ends:
        ex, ey = projector.point(float(end[0]), float(end[1]))
        elements.append(
            f'<line x1="{_fmt(ox)}" y1="{_fmt(oy)}" x2="{_fmt(ex)}" y2="{_fmt(ey)}" '
            'stroke="#2563eb" stroke-width="1.5"/>'
        )
    return elements


def _grid_rects(array: npt.NDArray[Any], color_for_value: Any) -> list[str]:
    height, width = array.shape
    cell = 12
    elements = [
        f'<rect x="0" y="0" width="{width * cell}" height="{height * cell}" fill="#ffffff"/>'
    ]
    for row in range(height):
        for col in range(width):
            color = color_for_value(array[row, col])
            elements.append(
                f'<rect x="{col * cell}" y="{row * cell}" width="{cell}" height="{cell}" '
                f'fill="{color}"/>'
            )
    return elements


def _depth_color(value: float, min_depth: float, max_depth: float) -> str:
    if not np.isfinite(value):
        return "#202124"
    span = max(max_depth - min_depth, 1e-9)
    t = min(max((float(value) - min_depth) / span, 0.0), 1.0)
    red = int(round(46 + t * 190))
    green = int(round(88 + (1.0 - abs(2.0 * t - 1.0)) * 90))
    blue = int(round(210 - t * 155))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _object_color(object_id: int) -> str:
    if object_id == 0:
        return "#f3f4f6"
    red = (89 + object_id * 53) % 256
    green = (132 + object_id * 97) % 256
    blue = (63 + object_id * 193) % 256
    return f"#{red:02x}{green:02x}{blue:02x}"


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            *elements,
            "</svg>",
            "",
        ]
    )


def _point_attr(projector: _Projector, x_world: float, z_world: float) -> str:
    x_value, y_value = projector.point(x_world, z_world)
    return f"{_fmt(x_value)},{_fmt(y_value)}"


def _object_label(instance: ObjectInstance) -> str:
    if instance.label_candidates:
        return f"{instance.object_id} {instance.label_candidates[0][0]}"
    return f"object_{instance.object_id}"


def _frame_id_from_depth_path(path: Path) -> int:
    return int(path.stem.split("_", maxsplit=1)[1])


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


__all__ = ["write_session_preview"]
