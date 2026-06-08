"""Atlas3R visual proof writer (dependency-light, no GUI/dashboard).

``write_visual_proof`` renders a minimal, honest picture of what the teacher
believes the world looks like: a floor-aligned top-down occupancy map (with the
camera trajectory overlaid), one grayscale heat-map per real occupancy channel,
and a compact Markdown index that links every artifact and states the final
status + provenance label.

Honesty rules (ARCHITECTURE.md):
  * Render ONLY real channels from a contract-valid ``OccupancyGrid2D``. Nothing
    is fabricated. ``unknown`` is drawn DISTINCTLY from ``free`` (never the same
    color, never collapsed).
  * A missing grid is NOT a blank picture -- it produces an ``index.md`` that
    states exactly which artifact is missing and the command to regenerate it.
  * ``measured_reference`` vs ``monocular_DA3`` (and the metric status) are
    surfaced verbatim from the provided ``provenance`` / ``status_label``; this
    writer never upgrades a label.

The package import stays dependency-free: ``numpy``/``matplotlib``/``PIL`` are
imported lazily INSIDE the functions only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Fixed legend (RGB 0-1). Distinct colors; unknown is gray, clearly != free.
CHANNEL_COLORS: dict[str, tuple[float, float, float]] = {
    "free": (0.20, 0.70, 0.25),            # green
    "occupied_static": (0.85, 0.15, 0.15),  # red
    "movable_static": (0.95, 0.55, 0.10),   # orange
    "dynamic": (0.85, 0.10, 0.80),          # magenta
    "unknown": (0.55, 0.55, 0.55),          # gray
}

# Channel -> OccupancyGrid2D attribute name.
CHANNEL_FIELDS: dict[str, str] = {
    "free": "P_free",
    "occupied_static": "P_occupied_static",
    "movable_static": "P_movable_static",
    "dynamic": "P_dynamic",
    "unknown": "P_unknown",
}

# Draw order for the combined top-down map: paint context first, hard surfaces
# last so occupied/dynamic stay visible over free/unknown.
_COMBINED_ORDER = ("unknown", "free", "movable_static", "occupied_static", "dynamic")


def write_visual_proof(
    asset_id: str,
    occupancy_grid: Any,
    trajectory_world: Any,
    out_dir: str | Path,
    status_label: str,
    provenance: Mapping[str, Any] | None = None,
    voxel_occupancy_3d: Any = None,
) -> dict[str, Any]:
    """Render the visual proof for one teacher track.

    Parameters
    ----------
    asset_id:
        Canonical asset id (e.g. ``"phone_room"``).
    occupancy_grid:
        A contract-valid :class:`atlas3r.contracts.OccupancyGrid2D`, or ``None``
        when fusion was blocked. ``None`` => a "missing artifact" ``index.md``.
    trajectory_world:
        Camera origins in world coords, shape ``[K,3]`` (or empty). These are the
        per-frame ``T_world_camera[:3,3]`` translations. Overlaid on the top-down
        map in plane coords. May be ``None``/empty -- the map still renders.
    out_dir:
        Output directory (``runs/teacher/<asset>/``; gitignored). Created if absent.
    status_label:
        The final honest category for this track (e.g. ``"metric_pseudo_label"``,
        ``"rejected"``). Surfaced verbatim; never upgraded here.
    provenance:
        Optional mapping. Recognized keys (all optional):
          * ``"provenance_label"``: one of ``measured_reference | monocular_DA3 |
            learned_metric_prior | manual_anchor | unavailable``. If absent it is
            derived from ``packet_source`` / ``metric_evidence`` /
            ``learned_metric_depth_prior``.
          * ``"plane_axes"``: ``(a0, a1)`` world axes the 2D grid spans (from
            ``map_report["occupancy_grid"]["plane_axes"]``). If absent, inferred.
          * ``"floor_axis"``: the collapsed world axis.
          * ``"packet_source"``: ``"measured_reference"`` | ``"monocular_artifact"``.
          * ``"track_type"``: ``"reference_metric"`` | ``"phone_room"`` (free-form).
          * ``"exports"``: mapping/sequence of exported artifact paths
            (ply/obj/npz) to link.
          * ``"teacher_report"``: path to the teacher report JSON to link.
          * ``"missing_command"``: command to regenerate a missing grid.

    Returns
    -------
    dict
        ``{"status", "out_dir", "files_written":[{path,bytes},...], "blockers", ...}``.
    """
    prov: Mapping[str, Any] = provenance or {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    provenance_label = _resolve_provenance_label(prov)
    files_written: list[dict[str, Any]] = []
    blockers: list[str] = []

    if occupancy_grid is None:
        index_path = _write_missing_index(
            asset_id, out_path, status_label, provenance_label, prov
        )
        files_written.append(_file_record(index_path))
        return {
            "status": "no_grid_missing_artifact_index",
            "asset_id": asset_id,
            "out_dir": str(out_path),
            "provenance_label": provenance_label,
            "status_label": status_label,
            "files_written": files_written,
            "rendered_channels": [],
            "blockers": ["occupancy_grid_missing"],
        }

    import numpy as np  # lazy: keep package import dependency-free

    # Pull channel arrays as float64; the grid is already contract-validated.
    channels = {
        name: np.asarray(getattr(occupancy_grid, field), dtype=np.float64)
        for name, field in CHANNEL_FIELDS.items()
    }
    grid_shape = channels["free"].shape  # (s0, s1) in plane-axis order
    resolution = float(occupancy_grid.resolution_m)
    origin_world = [float(v) for v in occupancy_grid.origin_world]

    plane_axes = _resolve_plane_axes(prov, grid_shape, resolution, origin_world, np)
    floor_axis = next(a for a in range(3) if a not in plane_axes)
    a0, a1 = plane_axes

    # Plane-coordinate extents (meters) for imshow. Cell (i,j) center is at
    # origin[a0] + (i+0.5)*res along axis a0, origin[a1] + (j+0.5)*res along a1.
    extent = (
        origin_world[a0],
        origin_world[a0] + grid_shape[0] * resolution,
        origin_world[a1],
        origin_world[a1] + grid_shape[1] * resolution,
    )

    traj_uv = _trajectory_plane_coords(trajectory_world, a0, a1, np)

    # --- combined top-down map -------------------------------------------------
    combined_path = out_path / "occupancy_topdown.png"
    rendered = _render_combined_topdown(
        combined_path, channels, extent, traj_uv, asset_id,
        status_label, provenance_label, resolution, plane_axes, floor_axis, np,
    )
    files_written.append(_file_record(combined_path))

    # --- per-channel grayscale heat maps --------------------------------------
    channel_files: dict[str, str] = {}
    for name in CHANNEL_FIELDS:
        png_path = out_path / f"occupancy_{name}.png"
        _render_channel_heat(png_path, channels[name], extent, name, np)
        files_written.append(_file_record(png_path))
        channel_files[name] = png_path.name

    # --- minimal 3D proof: a few argmax-colored band height slices -------------
    band_slice_files: list[tuple[str, float]] = []
    if voxel_occupancy_3d is not None:
        try:
            band_slice_files = _render_band_height_slices(
                out_path, voxel_occupancy_3d, np
            )
            for fname, _height in band_slice_files:
                files_written.append(_file_record(out_path / fname))
        except Exception:  # noqa: BLE001 -- slices are optional proof, never fatal
            band_slice_files = []

    # --- markdown index --------------------------------------------------------
    index_path = out_path / "index.md"
    _write_index(
        index_path, asset_id, status_label, provenance_label, prov,
        channel_files, channels, plane_axes, floor_axis, resolution,
        origin_world, grid_shape, traj_uv is not None, np,
        band_slice_files, voxel_occupancy_3d,
    )
    files_written.append(_file_record(index_path))

    return {
        "status": "rendered",
        "asset_id": asset_id,
        "out_dir": str(out_path),
        "provenance_label": provenance_label,
        "status_label": status_label,
        "plane_axes": list(plane_axes),
        "floor_axis": floor_axis,
        "grid_shape": list(grid_shape),
        "resolution_m": resolution,
        "rendered_channels": list(rendered),
        "trajectory_points": 0 if traj_uv is None else int(traj_uv.shape[0]),
        "band_height_slices": [name for name, _h in band_slice_files],
        "files_written": files_written,
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# provenance / axis resolution
# ---------------------------------------------------------------------------


def _resolve_provenance_label(prov: Mapping[str, Any]) -> str:
    """Derive the per-result provenance label.

    Prefer an explicit label; otherwise derive from packet_source +
    metric_evidence / learned_metric_depth_prior, per ARCHITECTURE.md.
    """
    explicit = prov.get("provenance_label")
    valid = {
        "measured_reference",
        "monocular_DA3",
        "learned_metric_prior",
        "manual_anchor",
        "unavailable",
    }
    if isinstance(explicit, str) and explicit in valid:
        return explicit

    source = str(prov.get("packet_source", "")).lower()
    if source == "measured_reference":
        return "measured_reference"
    if source in {"monocular_artifact", "external_artifact"}:
        if prov.get("metric_evidence") is True:
            return "measured_reference"
        if prov.get("learned_metric_depth_prior") is True:
            return "learned_metric_prior"
        return "monocular_DA3"
    if prov.get("metric_evidence") is True:
        return "measured_reference"
    if prov.get("learned_metric_depth_prior") is True:
        return "learned_metric_prior"
    if source in {"none", ""} and not prov:
        return "unavailable"
    return "monocular_DA3"


def _resolve_plane_axes(
    prov: Mapping[str, Any],
    grid_shape: tuple[int, ...],
    resolution: float,
    origin_world: Sequence[float],
    np: Any,
) -> tuple[int, int]:
    """Return the two world axes ``(a0, a1)`` the 2D grid spans.

    Prefer the explicit ``plane_axes`` from the fusion report. Otherwise infer
    from the grid shape: for each plane dimension pick the world axis whose
    expected cell count best matches. The 2D shape is ``(dims[a0], dims[a1])``
    where the 3D dims are ``ceil(extent/res)``; without grid_max we fall back to
    the standard floor-axis = vertical (axis 1, "y-up" reconstruction world) and
    finally a deterministic ``(0, 2)`` if all else is ambiguous.
    """
    explicit = prov.get("plane_axes")
    if (
        isinstance(explicit, (list, tuple))
        and len(explicit) == 2
        and all(isinstance(a, int) and 0 <= a < 3 for a in explicit)
        and explicit[0] != explicit[1]
    ):
        return (int(explicit[0]), int(explicit[1]))

    floor = prov.get("floor_axis")
    if isinstance(floor, int) and 0 <= floor < 3:
        axes = tuple(a for a in range(3) if a != floor)
        return (int(axes[0]), int(axes[1]))

    # Deterministic fallback: floor along axis 1 (the fusion default orientation
    # in reconstruction/metric world), so the plane is (0, 2). This matches the
    # canonical phone_room run; never silently mislabels because the index.md
    # records that the axes were inferred.
    return (0, 2)


def _trajectory_plane_coords(trajectory_world: Any, a0: int, a1: int, np: Any) -> Any:
    """Project world-frame camera origins to ``(axis a0, axis a1)`` plane coords.

    Returns an ``[K,2]`` float array, or ``None`` when there is no usable
    trajectory (never fabricates a path).
    """
    if trajectory_world is None:
        return None
    traj = np.asarray(trajectory_world, dtype=np.float64)
    if traj.ndim == 1 and traj.shape[0] == 3:
        traj = traj.reshape((1, 3))
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] == 0:
        return None
    finite = np.all(np.isfinite(traj), axis=1)
    traj = traj[finite]
    if traj.shape[0] == 0:
        return None
    return np.stack([traj[:, a0], traj[:, a1]], axis=1)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # headless, no GUI
    import matplotlib.pyplot as plt
    return plt


def _render_combined_topdown(
    path: Path,
    channels: dict[str, Any],
    extent: tuple[float, float, float, float],
    traj_uv: Any,
    asset_id: str,
    status_label: str,
    provenance_label: str,
    resolution: float,
    plane_axes: tuple[int, int],
    floor_axis: int,
    np: Any,
) -> list[str]:
    """Paint a single RGB top-down map combining real channels + trajectory."""
    plt = _setup_matplotlib()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    s0, s1 = channels["free"].shape
    # RGB canvas indexed [col(a1), row(a0)] for imshow with origin="lower".
    rgb = np.ones((s1, s0, 3), dtype=np.float64)  # white background
    alpha = np.zeros((s1, s0), dtype=np.float64)

    rendered: list[str] = []
    for name in _COMBINED_ORDER:
        prob = channels[name]
        if not np.any(prob > 0.0):
            continue
        rendered.append(name)
        color = np.asarray(CHANNEL_COLORS[name], dtype=np.float64)
        # weight = channel probability (real values), transposed to (s1, s0).
        w = np.clip(prob.T, 0.0, 1.0)
        mask = w > 0.0
        # Painter's blend: later channels overwrite where they have evidence.
        for c in range(3):
            rgb[..., c] = np.where(mask, (1.0 - w) * rgb[..., c] + w * color[c], rgb[..., c])
        alpha = np.where(mask, np.maximum(alpha, w), alpha)

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=110)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", aspect="equal")

    legend_handles = [
        Patch(facecolor=CHANNEL_COLORS[name], edgecolor="black", label=name)
        for name in ("free", "occupied_static", "movable_static", "dynamic", "unknown")
    ]

    if traj_uv is not None and traj_uv.shape[0] > 0:
        ax.plot(
            traj_uv[:, 0], traj_uv[:, 1],
            "-", color="black", linewidth=1.4, alpha=0.85, zorder=5,
        )
        ax.scatter(
            traj_uv[0, 0], traj_uv[0, 1],
            c="white", edgecolors="black", s=60, marker="o", zorder=6, label="cam start",
        )
        ax.scatter(
            traj_uv[-1, 0], traj_uv[-1, 1],
            c="black", edgecolors="white", s=60, marker="s", zorder=6, label="cam end",
        )
        legend_handles.append(
            Line2D([0], [0], color="black", lw=1.4, label="camera trajectory")
        )

    axis_names = {0: "X", 1: "Y", 2: "Z"}
    ax.set_xlabel(f"world axis {plane_axes[0]} ({axis_names[plane_axes[0]]}) [m]")
    ax.set_ylabel(f"world axis {plane_axes[1]} ({axis_names[plane_axes[1]]}) [m]")
    ax.set_title(
        f"{asset_id} top-down occupancy  |  status={status_label}\n"
        f"provenance={provenance_label}  |  floor axis={floor_axis} "
        f"({axis_names[floor_axis]})  |  res={resolution:.3f} m"
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return rendered


def _render_channel_heat(
    path: Path,
    prob: Any,
    extent: tuple[float, float, float, float],
    name: str,
    np: Any,
) -> None:
    """Grayscale heat map of one real probability channel (0..1)."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=100)
    # Transpose to (s1, s0) so axis a0 is horizontal under origin="lower".
    im = ax.imshow(
        np.clip(prob.T, 0.0, 1.0),
        origin="lower", extent=extent, cmap="gray",
        vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal",
    )
    ax.set_title(f"P_{name}  (max={float(np.max(prob)):.2f})")
    ax.set_xlabel("world axis a0 [m]")
    ax.set_ylabel("world axis a1 [m]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3D band height slices (minimal proof of the primary VoxelOccupancyGrid3D)
# ---------------------------------------------------------------------------


def _render_band_height_slices(
    out_path: Path,
    grid3d: Any,
    np: Any,
    max_slices: int = 4,
) -> list[tuple[str, float]]:
    """Render up to ``max_slices`` argmax-colored horizontal slices through the
    collision band of the ``VoxelOccupancyGrid3D``.

    Each cell is painted by its dominant (argmax) class color where the voxel was
    observed; unobserved voxels stay white (never colored as free). Returns a list
    of ``(filename, height_m)``.
    """
    plt = _setup_matplotlib()

    floor_axis = int(grid3d.floor_axis)
    plane_axes = tuple(a for a in range(3) if a != floor_axis)
    a0, a1 = plane_axes
    voxel = float(grid3d.voxel_size_m)
    origin = [float(v) for v in grid3d.origin_world]

    order = ("free", "occupied_static", "movable_static", "dynamic", "unknown")
    fields = [np.asarray(getattr(grid3d, CHANNEL_FIELDS[c]), dtype=np.float64) for c in order]
    stack = np.stack(fields, axis=-1)
    cls = np.argmax(stack, axis=-1)
    touched = np.asarray(grid3d.P_unknown, dtype=np.float64) < (1.0 - 1e-9)
    color_lut = np.asarray([CHANNEL_COLORS[c] for c in order], dtype=np.float64)

    nf = cls.shape[floor_axis]
    if nf <= max_slices:
        idxs = list(range(nf))
    else:
        idxs = sorted({int(round(k)) for k in np.linspace(0, nf - 1, num=max_slices)})

    extent = (
        origin[a0], origin[a0] + cls.shape[a0] * voxel,
        origin[a1], origin[a1] + cls.shape[a1] * voxel,
    )
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    files: list[tuple[str, float]] = []
    for k in idxs:
        cls_slice = np.take(cls, k, axis=floor_axis)
        touched_slice = np.take(touched, k, axis=floor_axis)
        rgb = np.ones((cls_slice.shape[1], cls_slice.shape[0], 3), dtype=np.float64)
        for ci in range(len(order)):
            mask = (cls_slice == ci) & touched_slice
            if not np.any(mask):
                continue
            mt = mask.T
            for c in range(3):
                rgb[..., c] = np.where(mt, color_lut[ci, c], rgb[..., c])
        height_m = origin[floor_axis] + (k + 0.5) * voxel
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=100)
        ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
        ax.set_title(
            f"band slice k={k}  height={height_m:.3f} m  (argmax class)\n"
            f"floor axis {floor_axis} ({axis_names[floor_axis]})"
        )
        ax.set_xlabel(f"world axis {a0} ({axis_names[a0]}) [m]")
        ax.set_ylabel(f"world axis {a1} ({axis_names[a1]}) [m]")
        fname = f"occupancy_band_slice_{k}.png"
        fig.tight_layout()
        fig.savefig(out_path / fname, facecolor="white")
        plt.close(fig)
        files.append((fname, float(height_m)))
    return files


# ---------------------------------------------------------------------------
# markdown index
# ---------------------------------------------------------------------------


def _write_index(
    path: Path,
    asset_id: str,
    status_label: str,
    provenance_label: str,
    prov: Mapping[str, Any],
    channel_files: Mapping[str, str],
    channels: Mapping[str, Any],
    plane_axes: tuple[int, int],
    floor_axis: int,
    resolution: float,
    origin_world: Sequence[float],
    grid_shape: tuple[int, ...],
    has_trajectory: bool,
    np: Any,
    band_slice_files: Sequence[tuple[str, float]] = (),
    voxel_occupancy_3d: Any = None,
) -> None:
    track_type = str(prov.get("track_type", "")) or "unknown"
    packet_source = str(prov.get("packet_source", "")) or "unknown"

    lines: list[str] = []
    lines.append(f"# Atlas3R teacher visual proof - `{asset_id}`")
    lines.append("")
    lines.append(f"- **Final status:** `{status_label}`")
    lines.append(f"- **Provenance label:** `{provenance_label}`")
    lines.append(f"- **Track type:** `{track_type}`")
    lines.append(f"- **Packet source:** `{packet_source}`")

    # Track-specific framing required by the architecture.
    if track_type in {"reference_metric", "reference-metric"} or packet_source in {
        "measured_reference", "monocular_artifact",
    }:
        if packet_source == "measured_reference":
            lines.append("- **Role:** `measured_baseline` (RGB-D/pose ground truth; "
                         "baseline/evaluation only).")
        elif track_type in {"reference_metric", "reference-metric"}:
            lines.append("- **Role:** `monocular_candidate` (DA3 monocular path, "
                         "proven separately against the measured baseline).")
    if track_type in {"phone_room", "phone-room"} or asset_id == "phone_room":
        lines.append("- **Role:** monocular teacher output (no measured reference "
                     f"expected); metric status: `{status_label}`.")
    lines.append("")

    # Grid geometry.
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    lines.append("## Map geometry")
    lines.append("")
    lines.append(f"- Plane axes: `{tuple(plane_axes)}` "
                 f"({axis_names[plane_axes[0]]}, {axis_names[plane_axes[1]]}); "
                 f"floor (collapsed) axis: `{floor_axis}` ({axis_names[floor_axis]}).")
    if not _axes_were_explicit(prov):
        lines.append("- _Plane axes were INFERRED (not supplied by the fuser); "
                     "treat the axis labels as best-effort._")
    lines.append(f"- Resolution: `{resolution:.4f}` m/cell; grid shape "
                 f"`{tuple(int(s) for s in grid_shape)}` cells.")
    lines.append(f"- Origin (world grid_min): "
                 f"`({origin_world[0]:.3f}, {origin_world[1]:.3f}, {origin_world[2]:.3f})`.")
    lines.append(f"- Camera trajectory overlaid: `{'yes' if has_trajectory else 'no'}`.")
    lines.append("")

    # Channel coverage (real fractions, never fabricated).
    lines.append("## Channels (real coverage)")
    lines.append("")
    lines.append("| channel | color | cells > 0 | max P |")
    lines.append("|---|---|---:|---:|")
    color_names = {
        "free": "green", "occupied_static": "red", "movable_static": "orange",
        "dynamic": "magenta", "unknown": "gray",
    }
    total_cells = int(grid_shape[0] * grid_shape[1])
    for name in CHANNEL_FIELDS:
        prob = channels[name]
        nz = int(np.count_nonzero(prob > 0.0))
        mx = float(np.max(prob)) if prob.size else 0.0
        lines.append(f"| `{name}` | {color_names[name]} | {nz} / {total_cells} | {mx:.2f} |")
    lines.append("")

    # Top-down + per-channel PNGs.
    lines.append("## Top-down occupancy (combined + trajectory)")
    lines.append("")
    lines.append("![top-down occupancy](occupancy_topdown.png)")
    lines.append("")
    lines.append("Legend: free=green, occupied_static=red, movable_static=orange, "
                 "dynamic=magenta, unknown=gray. **unknown is rendered distinctly "
                 "from free (unknown is never free).**")
    lines.append("")
    lines.append("_The top-down map is the PURE top-down projection of the 3D "
                 "collision-band field below (single source of truth)._")
    lines.append("")

    # 3D collision-band occupancy (primary output) -- height slices.
    if voxel_occupancy_3d is not None and band_slice_files:
        axis_names = {0: "X", 1: "Y", 2: "Z"}
        fa = int(getattr(voxel_occupancy_3d, "floor_axis", floor_axis))
        bmin = float(getattr(voxel_occupancy_3d, "band_min_m", 0.0))
        bmax = float(getattr(voxel_occupancy_3d, "band_max_m", 0.0))
        v = float(getattr(voxel_occupancy_3d, "voxel_size_m", resolution))
        cat = getattr(getattr(voxel_occupancy_3d, "acceptance_category", None), "value", "unknown")
        lines.append("## 3D collision-band occupancy field (primary output)")
        lines.append("")
        lines.append(f"- Floor axis: `{fa}` ({axis_names.get(fa, '?')}); collision band "
                     f"`[{bmin:.3f}, {bmax:.3f}]` m ({len(band_slice_files)} slice(s) "
                     f"shown); voxel `{v:.4f}` m; acceptance `{cat}`.")
        lines.append("- Each slice is colored by the per-voxel **argmax class**; "
                     "unobserved voxels are blank (unknown is never free).")
        lines.append("")
        for fname, height in band_slice_files:
            lines.append(f"### band slice @ `{height:.3f}` m")
            lines.append("")
            lines.append(f"![band slice {height:.3f} m]({fname})")
            lines.append("")

    lines.append("## Per-channel heat maps")
    lines.append("")
    for name in CHANNEL_FIELDS:
        fname = channel_files.get(name, f"occupancy_{name}.png")
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"![{name}]({fname})")
        lines.append("")

    # Linked exported artifacts (ply/obj/npz) + teacher report.
    lines.append("## Exported artifacts")
    lines.append("")
    export_lines = _export_links(prov, path.parent)
    if export_lines:
        lines.extend(export_lines)
    else:
        lines.append("- _No point-cloud / mesh / voxel exports were supplied to "
                     "the visualizer._")
    lines.append("")

    report_link = _relative_link(prov.get("teacher_report"), path.parent)
    lines.append("## Teacher report")
    lines.append("")
    if report_link:
        lines.append(f"- [`teacher_report.json`]({report_link})")
    else:
        lines.append("- _Teacher report path not supplied._")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by `atlas3r.visualize.write_visual_proof`. Only real, "
                 "contract-valid channels are rendered; nothing is fabricated._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_missing_index(
    asset_id: str,
    out_path: Path,
    status_label: str,
    provenance_label: str,
    prov: Mapping[str, Any],
) -> Path:
    """Write an index.md that names the missing artifact + regen command."""
    index_path = out_path / "index.md"
    command = str(
        prov.get("missing_command")
        or "python -m atlas3r.teacher"
    )
    missing = str(prov.get("missing_artifact") or "OccupancyGrid2D (fused static map)")
    map_status = str(prov.get("map_status") or "unknown")

    lines = [
        f"# Atlas3R teacher visual proof - `{asset_id}` (MISSING ARTIFACT)",
        "",
        f"- **Final status:** `{status_label}`",
        f"- **Provenance label:** `{provenance_label}`",
        "",
        "## No occupancy grid to render",
        "",
        f"The fused occupancy grid was not produced, so **no visual proof PNGs "
        f"were written** (nothing is fabricated).",
        "",
        f"- Missing artifact: `{missing}`",
        f"- Map/occupancy stage status: `{map_status}`",
    ]
    blockers = prov.get("blockers")
    if isinstance(blockers, (list, tuple)) and blockers:
        lines.append("- Exact blockers:")
        for b in blockers:
            lines.append(f"  - `{b}`")
    lines.append("")
    lines.append("## To regenerate")
    lines.append("")
    lines.append(f"```\n{command}\n```")
    lines.append("")
    report_link = _relative_link(prov.get("teacher_report"), out_path)
    if report_link:
        lines.append(f"See the teacher report for the exact stage blockers: "
                     f"[`teacher_report.json`]({report_link}).")
        lines.append("")
    lines.append("_Generated by `atlas3r.visualize.write_visual_proof`._")
    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _axes_were_explicit(prov: Mapping[str, Any]) -> bool:
    explicit = prov.get("plane_axes")
    if (
        isinstance(explicit, (list, tuple)) and len(explicit) == 2
        and all(isinstance(a, int) for a in explicit) and explicit[0] != explicit[1]
    ):
        return True
    return isinstance(prov.get("floor_axis"), int)


def _export_links(prov: Mapping[str, Any], base: Path) -> list[str]:
    """Build markdown links for exported ply/obj/npz artifacts.

    Accepts several shapes for ``provenance["exports"]``:
      * the raw ``export_teacher_artifacts()["artifacts"]`` mapping
        ``{name: {"status","path",...}}`` (only ``"written"`` entries linked);
      * a flat ``{label: path}`` mapping;
      * a sequence of path strings.
    """
    exports = prov.get("exports")
    items: list[tuple[str, Any, str | None]] = []
    if isinstance(exports, Mapping):
        for label, value in exports.items():
            if isinstance(value, Mapping):
                # export entry dict from export_teacher_artifacts.
                status = value.get("status")
                target = value.get("path")
                if target and status not in {"blocked", "skipped", None}:
                    items.append((str(label), target, str(status) if status else None))
                # also surface secondary paths (e.g. trajectory tum/json).
                for extra_key in ("tum_path", "json_path"):
                    extra = value.get(extra_key)
                    if extra and status not in {"blocked", None}:
                        items.append((f"{label}:{extra_key}", extra, None))
            else:
                items.append((str(label), value, None))
    elif isinstance(exports, (list, tuple)):
        items = [(Path(str(p)).suffix.lstrip(".") or "file", p, None) for p in exports]

    out: list[str] = []
    for label, target, status in items:
        link = _relative_link(target, base)
        if link:
            suffix = f" (`{status}`)" if status else ""
            out.append(f"- [`{Path(str(target)).name}`]({link}) - `{label}`{suffix}")
    return out


def _relative_link(target: Any, base: Path) -> str | None:
    if not target:
        return None
    tp = Path(str(target))
    try:
        rel = tp.relative_to(base)
        return rel.as_posix()
    except ValueError:
        # Not under base; fall back to a portable relative path or the name.
        try:
            import os
            return Path(os.path.relpath(tp, base)).as_posix()
        except (ValueError, OSError):
            return tp.name


def _file_record(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    return {"path": str(path), "bytes": int(size)}
