"""Dependency-free HTML/SVG previews for synthetic training predictions."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def write_prediction_preview(
    *,
    output_dir: str | Path,
    rgb_u8: npt.NDArray[np.uint8],
    target_depth_m: npt.NDArray[np.float32],
    predicted_depth_m: npt.NDArray[np.float32],
    abs_error_m: npt.NDArray[np.float32],
    metrics: Mapping[str, float],
    truth_boundary: Mapping[str, object],
    valid_depth_mask: npt.NDArray[np.bool_] | None = None,
    title: str = "Atlas3R Training Preview",
) -> tuple[Path, Path]:
    """Write deterministic `prediction_preview.html` and `.svg` files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    svg_path = output_path / "prediction_preview.svg"
    html_path = output_path / "prediction_preview.html"
    svg_path.write_text(
        _preview_svg(
            rgb_u8=rgb_u8,
            target_depth_m=target_depth_m,
            predicted_depth_m=predicted_depth_m,
            abs_error_m=abs_error_m,
            valid_depth_mask=valid_depth_mask,
        ),
        encoding="utf-8",
        newline="\n",
    )
    html_path.write_text(
        _preview_html(svg_path.name, metrics, truth_boundary, title=title),
        encoding="utf-8",
        newline="\n",
    )
    return html_path, svg_path


def _preview_svg(
    *,
    rgb_u8: npt.NDArray[np.uint8],
    target_depth_m: npt.NDArray[np.float32],
    predicted_depth_m: npt.NDArray[np.float32],
    abs_error_m: npt.NDArray[np.float32],
    valid_depth_mask: npt.NDArray[np.bool_] | None,
) -> str:
    rgb = _validate_rgb(rgb_u8)
    target = _validate_hw("target_depth_m", target_depth_m)
    predicted = _validate_hw("predicted_depth_m", predicted_depth_m)
    error = _validate_hw("abs_error_m", abs_error_m)
    if target.shape != predicted.shape or target.shape != error.shape:
        raise ValueError("preview arrays: depth and error shapes must match")
    height, width = target.shape
    mask = (
        _validate_mask(valid_depth_mask, (height, width)) if valid_depth_mask is not None else None
    )
    pixel_size = 4
    panel_gap = 18
    label_height = 22
    panel_width = width * pixel_size
    panel_height = height * pixel_size + label_height
    panel_count = 5 if mask is not None else 4
    total_width = panel_width * panel_count + panel_gap * (panel_count - 1)
    total_height = panel_height
    depth_min = float(min(np.min(target), np.min(predicted)))
    depth_max = float(max(np.max(target), np.max(predicted)))
    error_max = max(float(np.percentile(error, 98)), 1e-6)
    panels = [
        _rgb_panel("input RGB", rgb, 0, label_height, pixel_size),
        _heatmap_panel(
            "target depth",
            target,
            panel_width + panel_gap,
            label_height,
            pixel_size,
            value_min=depth_min,
            value_max=depth_max,
        ),
        _heatmap_panel(
            "pred depth",
            predicted,
            (panel_width + panel_gap) * 2,
            label_height,
            pixel_size,
            value_min=depth_min,
            value_max=depth_max,
        ),
        _heatmap_panel(
            "abs error",
            error,
            (panel_width + panel_gap) * 3,
            label_height,
            pixel_size,
            value_min=0.0,
            value_max=error_max,
        ),
    ]
    labels_and_x = [
        ("input RGB", 0),
        ("target depth", panel_width + panel_gap),
        ("pred depth", (panel_width + panel_gap) * 2),
        ("abs error", (panel_width + panel_gap) * 3),
    ]
    if mask is not None:
        panels.append(
            _mask_panel(
                "valid mask",
                mask,
                (panel_width + panel_gap) * 4,
                label_height,
                pixel_size,
            )
        )
        labels_and_x.append(("valid mask", (panel_width + panel_gap) * 4))
    labels = "".join(
        f'<text x="{x}" y="15" font-family="Arial, sans-serif" font-size="12">{label}</text>'
        for label, x in labels_and_x
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" '
        f'height="{total_height}" viewBox="0 0 {total_width} {total_height}">'
        f'<rect width="100%" height="100%" fill="#f8f8f8"/>{labels}'
        f"{''.join(panels)}</svg>\n"
    )


def _preview_html(
    svg_name: str,
    metrics: Mapping[str, float],
    truth_boundary: Mapping[str, object],
    *,
    title: str,
) -> str:
    metrics_json = html.escape(json.dumps(dict(metrics), indent=2, sort_keys=True))
    truth_json = html.escape(json.dumps(dict(truth_boundary), indent=2, sort_keys=True))
    safe_svg_name = html.escape(svg_name)
    safe_title = html.escape(title)
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{safe_title}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}"
        "pre{background:#f4f4f4;padding:12px;overflow:auto}"
        "img{max-width:100%;height:auto;border:1px solid #ccc}</style></head><body>"
        f"<h1>{safe_title}</h1>"
        f'<img src="{safe_svg_name}" alt="Atlas3R training prediction preview">'
        "<h2>Metrics</h2><pre>"
        f"{metrics_json}</pre><h2>Truth Boundary</h2><pre>{truth_json}</pre></body></html>\n"
    )


def _rgb_panel(
    _label: str,
    rgb: npt.NDArray[np.uint8],
    x0: int,
    y0: int,
    pixel_size: int,
) -> str:
    rects: list[str] = []
    height, width, _channels = rgb.shape
    for y in range(height):
        for x in range(width):
            r, g, b = (int(v) for v in rgb[y, x])
            rects.append(
                f'<rect x="{x0 + x * pixel_size}" y="{y0 + y * pixel_size}" '
                f'width="{pixel_size}" height="{pixel_size}" fill="#{r:02x}{g:02x}{b:02x}"/>'
            )
    return "".join(rects)


def _heatmap_panel(
    _label: str,
    array: npt.NDArray[np.float32],
    x0: int,
    y0: int,
    pixel_size: int,
    *,
    value_min: float,
    value_max: float,
) -> str:
    denom = max(value_max - value_min, 1e-6)
    rects: list[str] = []
    height, width = array.shape
    for y in range(height):
        for x in range(width):
            ratio = float(np.clip((float(array[y, x]) - value_min) / denom, 0.0, 1.0))
            r, g, b = _heat_color(ratio)
            rects.append(
                f'<rect x="{x0 + x * pixel_size}" y="{y0 + y * pixel_size}" '
                f'width="{pixel_size}" height="{pixel_size}" fill="#{r:02x}{g:02x}{b:02x}"/>'
            )
    return "".join(rects)


def _mask_panel(
    _label: str,
    mask: npt.NDArray[np.bool_],
    x0: int,
    y0: int,
    pixel_size: int,
) -> str:
    rects: list[str] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            color = "#e9f6ee" if bool(mask[y, x]) else "#222222"
            rects.append(
                f'<rect x="{x0 + x * pixel_size}" y="{y0 + y * pixel_size}" '
                f'width="{pixel_size}" height="{pixel_size}" fill="{color}"/>'
            )
    return "".join(rects)


def _heat_color(ratio: float) -> tuple[int, int, int]:
    r = int(round(255.0 * ratio))
    g = int(round(180.0 * (1.0 - abs(ratio * 2.0 - 1.0))))
    b = int(round(255.0 * (1.0 - ratio)))
    return r, g, b


def _validate_rgb(value: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("rgb_u8: must have shape H,W,3 and dtype uint8")
    return array


def _validate_hw(field_name: str, value: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{field_name}: must have shape H,W")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{field_name}: must be floating point")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name}: must contain finite values")
    return array.astype(np.float32, copy=False)


def _validate_mask(value: npt.NDArray[Any], shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("valid_depth_mask: shape must match depth arrays")
    return array.astype(np.bool_, copy=False)


__all__ = [
    "write_prediction_preview",
]
