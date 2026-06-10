"""M9 teacher export -- turn the REAL fused map into inspectable geometry.

This module writes the geometry the teacher actually produced (no pretty
rendering, no completion, no hallucinated surfaces). Every artifact is derived
from real packets / voxel map / occupancy grid; a missing input is recorded as
an explicit blocked entry naming the absent object -- never a fake file.

Honesty rules honored here:
- Surface points are OBSERVED points only: voxel centers where ``surface_count
  > 0`` (re-placed in world using the same ``grid_min``/voxel convention the
  fuser used) and/or the packet lift ``X_world = (rays * radial_depth) @ R.T +
  t``. Nothing is interpolated to "fill in" unseen geometry.
- Only ``occupied_static`` (and ``movable_static`` if a static/dynamic state
  supplies it) points are written. ``dynamic`` points are EXCLUDED -- dynamic is
  never fused into static geometry.
- A mesh is written ONLY if it can be built faithfully from the observed points
  via open3d (ball-pivoting / Poisson on real points). If a faithful mesh is not
  feasible, the mesh is SKIPPED and recorded as ``mesh_skipped_point_cloud_only``
  -- a mesh is never faked.
- ``T_world_camera`` is camera-to-world; the camera centre in world is its
  translation column ``t``. The TUM line uses ``tx ty tz`` = ``t`` and the
  quaternion of ``R`` (the camera-to-world rotation).

``numpy`` and ``open3d``/``trimesh`` are imported lazily INSIDE functions so the
package import stays dependency-free.

Public entrypoint:
    export_teacher_artifacts(asset_id, packets, voxel_map, occupancy_grid,
                             static_dynamic_states, scale_posterior, out_dir,
                             *, map_report=None) -> dict
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    FrameRayPacket,
    OccupancyGrid2D,
    ScalePosterior,
    StaticDynamicState,
    VoxelMapState,
    VoxelOccupancyGrid3D,
)

# Per-channel RGB used when colouring the surface point cloud. Dynamic is never
# emitted, so it has no colour here.
_COLOR_OCCUPIED_STATIC = (0.90, 0.55, 0.10)  # warm orange
_COLOR_MOVABLE_STATIC = (0.20, 0.55, 0.95)  # blue
# Confidence threshold for treating a packet pixel as static when a
# StaticDynamicState supplies per-pixel probabilities.
_STATIC_PROB_THRESHOLD = 0.5


def export_teacher_artifacts(
    asset_id: str,
    packets: Sequence[FrameRayPacket],
    voxel_map: VoxelMapState | None,
    occupancy_grid: OccupancyGrid2D | None,
    static_dynamic_states: Sequence[StaticDynamicState] | None,
    scale_posterior: ScalePosterior | None,
    out_dir: str | Path,
    *,
    map_report: Mapping[str, Any] | None = None,
    voxel_occupancy_3d: VoxelOccupancyGrid3D | None = None,
    floor_align_rotation: Any = None,
) -> dict[str, Any]:
    """Write inspectable geometry artifacts for one teacher result.

    Parameters mirror the teacher's fused-map outputs. ``map_report`` is the
    optional ``fuse_static_map`` report; when present its ``grid_min`` /
    ``effective_voxel_size_m`` let the surface point cloud be recovered from the
    voxel map's surface voxels. Without it (or without ``voxel_map``) the surface
    cloud is recovered from the packet lift instead.

    Returns a dict of artifact paths + provenance + a ``channels`` summary. Each
    artifact that cannot be produced has an explicit ``blocked`` entry naming the
    missing object.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    provenance = _provenance(packets, scale_posterior, voxel_map, occupancy_grid)

    result: dict[str, Any] = {
        "module": "M9 - Teacher Geometry Export",
        "asset_id": str(asset_id),
        "out_dir": str(out_path),
        "provenance": provenance,
        "artifacts": {},
        "channels": {},
        "blocked": [],
        "blockers": [],
    }

    import numpy as np  # lazy

    # --- 1. surface point cloud (occupied_static, optional movable_static) -----
    pc_entry = _write_surface_point_cloud(
        asset_id, packets, voxel_map, static_dynamic_states,
        map_report, provenance, out_path, np, floor_align_rotation,
    )
    result["artifacts"]["surface_point_cloud"] = pc_entry
    if pc_entry.get("status") != "written":
        _record_block(result, "surface_point_cloud", pc_entry)

    surface_points = pc_entry.get("_points")  # in-memory array for mesh step
    surface_normals = pc_entry.get("_normals")

    # --- 2. camera trajectory (.ply + .tum.txt + .json) ----------------------
    traj_entry = _write_camera_trajectory(asset_id, packets, out_path, np, floor_align_rotation)
    result["artifacts"]["camera_trajectory"] = traj_entry
    if traj_entry.get("status") != "written":
        _record_block(result, "camera_trajectory", traj_entry)

    # --- 3. occupancy channels (.npz) ----------------------------------------
    occ_entry = _write_occupancy_channels(asset_id, occupancy_grid, out_path, np)
    result["artifacts"]["occupancy_channels"] = occ_entry
    if occ_entry.get("status") == "written":
        result["channels"] = occ_entry.get("channel_summary", {})
    else:
        _record_block(result, "occupancy_channels", occ_entry)

    # --- 4. mesh (honest TSDF/Poisson/ball-pivoting, else skip) ---------------
    mesh_entry = _write_mesh_if_feasible(
        asset_id, surface_points, surface_normals, voxel_map, out_path, np,
    )
    result["artifacts"]["mesh"] = mesh_entry
    if mesh_entry.get("status") not in {"written", "skipped"}:
        _record_block(result, "mesh", mesh_entry)

    # --- 5. primary output: 3D collision-band occupancy field (.npz) ----------
    occ3d_entry = _write_voxel_occupancy_3d(asset_id, voxel_occupancy_3d, out_path, np)
    result["artifacts"]["voxel_occupancy_3d"] = occ3d_entry
    if occ3d_entry.get("status") == "written":
        result["voxel_occupancy_3d_summary"] = occ3d_entry.get("field_summary", {})
    else:
        _record_block(result, "voxel_occupancy_3d", occ3d_entry)

    # Strip the in-memory point arrays out of the returned/serialized entry.
    for key in ("_points", "_normals", "_colors"):
        pc_entry.pop(key, None)

    result["blockers"] = _dedupe(result["blockers"])
    result["status"] = "exported" if not result["blocked"] else "exported_with_blocked_artifacts"
    return result


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _provenance(
    packets: Sequence[FrameRayPacket],
    scale_posterior: ScalePosterior | None,
    voxel_map: VoxelMapState | None,
    occupancy_grid: OccupancyGrid2D | None,
) -> dict[str, Any]:
    """Derive the per-result provenance label + scale/units/frame metadata.

    Provenance label per ARCHITECTURE: measured_reference | monocular_DA3 |
    learned_metric_prior | manual_anchor | unavailable. Derived from packet
    ``source`` + ``provenance`` (NOT a packet field).
    """
    label = "unavailable"
    backbone = None
    metric_evidence = None
    learned_prior = None
    sources = sorted({str(getattr(p, "source", "")) for p in packets}) if packets else []

    # A learned metric-depth prior is the authoritative soft anchor: it lives on
    # the ScalePosterior's evidence sources (measured=False). Detect it there so
    # the provenance label is not down-graded just because the packet provenance
    # does not echo the manifest flag.
    has_learned_prior = False
    if scale_posterior is not None:
        for ev in getattr(scale_posterior, "scale_sources", ()) or ():
            etype = getattr(getattr(ev, "evidence_type", None), "value", None)
            if etype == "learned_metric_depth_prior":
                has_learned_prior = True

    if packets:
        first = packets[0]
        src = str(getattr(first, "source", ""))
        prov = getattr(first, "provenance", {}) or {}
        metric_evidence = prov.get("metric_evidence")
        learned_prior = bool(has_learned_prior)
        backbone = prov.get("backbone_name")
        if src.startswith("measured_reference"):
            label = "measured_reference"
        elif src.startswith("external_artifact"):
            if has_learned_prior:
                label = "learned_metric_prior"
            else:
                label = "monocular_DA3"
        else:
            label = "unavailable"

    scale_status = None
    scale_mean = None
    scale_std = None
    relative_unc = None
    if scale_posterior is not None:
        scale_status = scale_posterior.metric_acceptance_status.value
        scale_mean = float(scale_posterior.scale_mean)
        scale_std = float(scale_posterior.scale_std)
        relative_unc = float(scale_posterior.relative_scale_uncertainty)

    # Units honesty: only a metric status is meters; otherwise unitless
    # similarity-frame reconstruction.
    if scale_status in {"measured_metric", "metric_pseudo_label"}:
        units = "meters"
    elif scale_status is None:
        units = "unknown_no_scale_posterior"
    else:
        units = "reconstruction_units_unitless_similarity"

    coordinate_frame = None
    if voxel_map is not None:
        coordinate_frame = voxel_map.coordinate_frame
    elif occupancy_grid is not None:
        coordinate_frame = occupancy_grid.grid_frame

    return {
        "provenance_label": label,
        "packet_sources": sources,
        "backbone_name": backbone,
        "metric_evidence": metric_evidence,
        "learned_metric_depth_prior": learned_prior,
        "scale_status": scale_status,
        "scale_mean": scale_mean,
        "scale_std": scale_std,
        "relative_scale_uncertainty": relative_unc,
        "units": units,
        "coordinate_frame": coordinate_frame,
    }


# ---------------------------------------------------------------------------
# surface point cloud
# ---------------------------------------------------------------------------


def _write_surface_point_cloud(
    asset_id: str,
    packets: Sequence[FrameRayPacket],
    voxel_map: VoxelMapState | None,
    static_dynamic_states: Sequence[StaticDynamicState] | None,
    map_report: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    out_path: Path,
    np: Any,
    floor_align_rotation: Any = None,
) -> dict[str, Any]:
    path = out_path / "surface_point_cloud.ply"

    # Prefer voxel-map surface voxels (the fused static surface) when we can
    # place them in world via the map_report grid_min/voxel. Otherwise fall back
    # to the honest packet lift of high-confidence static pixels.
    points = None
    colors = None
    method = None

    grid_min, eff_voxel = _grid_geometry(map_report)
    if voxel_map is not None and grid_min is not None and eff_voxel is not None:
        points, colors = _surface_points_from_voxels(voxel_map, grid_min, eff_voxel, np)
        if points is not None and points.shape[0] > 0:
            method = "voxel_surface_centers"

    if points is None or points.shape[0] == 0:
        points, colors, dropped = _surface_points_from_packets(
            packets, static_dynamic_states, np, floor_align_rotation
        )
        method = "packet_lift_static_pixels_floor_aligned"
        if points.shape[0] == 0:
            return {
                "status": "blocked",
                "missing_object": "surface_points",
                "reason": "no_static_surface_points_recovered",
                "path": str(path),
                "dropped_dynamic_or_lowconf_points": dropped,
            }

    # Estimate normals so an honest mesh step is possible later.
    normals = _estimate_normals(points, np)

    try:
        import open3d as o3d  # lazy
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
        ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        writer = "open3d"
        if not ok:
            raise RuntimeError("open3d write_point_cloud returned False")
    except Exception:  # noqa: BLE001 -- fall back to trimesh, then record honestly
        try:
            import trimesh  # lazy
            cloud = trimesh.PointCloud(
                np.asarray(points, dtype=np.float64),
                colors=(np.asarray(colors, dtype=np.float64) * 255).astype("uint8"),
            )
            cloud.export(str(path))
            writer = "trimesh"
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "blocked",
                "missing_object": "point_cloud_writer",
                "reason": f"point_cloud_write_failed:{type(exc).__name__}:{exc}",
                "path": str(path),
            }

    n_occ = int(np.count_nonzero(np.all(np.isclose(colors, _COLOR_OCCUPIED_STATIC), axis=1)))
    n_mov = int(np.count_nonzero(np.all(np.isclose(colors, _COLOR_MOVABLE_STATIC), axis=1)))
    return {
        "status": "written",
        "path": str(path),
        "writer": writer,
        "recovery_method": method,
        "point_count": int(points.shape[0]),
        "occupied_static_points": n_occ,
        "movable_static_points": n_mov,
        "dynamic_points_excluded": True,
        "size_bytes": _size(path),
        "_points": points,
        "_normals": normals,
        "_colors": colors,
    }


def _surface_points_from_voxels(voxel_map: VoxelMapState, grid_min, eff_voxel, np):
    surface_count = np.asarray(voxel_map.surface_count)
    surf = surface_count > 0.0
    idx = np.argwhere(surf)  # (M,3)
    if idx.shape[0] == 0:
        return None, None
    grid_min = np.asarray(grid_min, dtype=np.float64).reshape(3)
    centers = grid_min[None, :] + (idx.astype(np.float64) + 0.5) * float(eff_voxel)
    # The fused map has no static/dynamic input -> every surface voxel is
    # occupied_static (movable/dynamic are honestly zero). Colour accordingly.
    colors = np.tile(np.asarray(_COLOR_OCCUPIED_STATIC, dtype=np.float64), (centers.shape[0], 1))
    return centers, colors


def _surface_points_from_packets(
    packets: Sequence[FrameRayPacket],
    static_dynamic_states: Sequence[StaticDynamicState] | None,
    np: Any,
    floor_align_rotation: Any = None,
):
    """Lift static surface points from packets, EXCLUDING dynamic pixels.

    Uses the same lift as ``mapping._collect_world_points``:
        X_world = (rays * radial_depth[:,None]) @ R.T + t

    When a StaticDynamicState matches a packet AND its per-pixel arrays match the
    packet's ray count, pixels below the static threshold (or flagged dynamic)
    are dropped. Otherwise all of the packet's points are occupied_static (the
    map carries no dynamic inference, so withholding all points would discard
    real observed geometry).
    """
    state_by_frame: dict[int, StaticDynamicState] = {}
    for state in static_dynamic_states or ():
        state_by_frame[int(state.frame_id)] = state

    pts_list = []
    col_list = []
    dropped = 0
    for packet in packets:
        rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
        depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
        T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
        R = T[:3, :3]
        t = T[:3, 3]
        x_world = (rays * depth[:, None]) @ R.T + t[None, :]

        n = x_world.shape[0]
        keep = np.ones(n, dtype=bool)
        colors = np.tile(
            np.asarray(_COLOR_OCCUPIED_STATIC, dtype=np.float64), (n, 1)
        )

        state = state_by_frame.get(int(packet.frame_id))
        if state is not None:
            stat = np.asarray(state.static_probability, dtype=np.float64).reshape((-1,))
            dyn = np.asarray(state.dynamic_probability, dtype=np.float64).reshape((-1,))
            mov = _movable_from_state(state, stat.shape[0], np)
            if stat.shape[0] == n:
                # Dynamic-dominant pixels are excluded from static geometry.
                is_dynamic = (dyn >= stat) & (dyn > _STATIC_PROB_THRESHOLD)
                keep = ~is_dynamic & (stat >= _STATIC_PROB_THRESHOLD)
                dropped += int(np.count_nonzero(~keep))
                if mov is not None:
                    movable_pixels = (mov > _STATIC_PROB_THRESHOLD) & keep
                    colors[movable_pixels] = _COLOR_MOVABLE_STATIC

        if np.any(keep):
            pts_list.append(x_world[keep])
            col_list.append(colors[keep])

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), dropped
    pts = np.concatenate(pts_list, axis=0)
    if floor_align_rotation is not None:
        # Match the floor-aligned voxel frame the rest of the export uses.
        pts = pts @ np.asarray(floor_align_rotation, dtype=np.float64).T
    return pts, np.concatenate(col_list, axis=0), dropped


def _movable_from_state(state: StaticDynamicState, n: int, np: Any):
    # The contract's StaticDynamicState has no movable channel (static/dynamic/
    # unknown simplex). A movable channel may be carried in residual_summary as a
    # convenience; otherwise there is none.
    residual = getattr(state, "residual_summary", {}) or {}
    movable = residual.get("movable_probability")
    if movable is None:
        return None
    arr = np.asarray(movable, dtype=np.float64).reshape((-1,))
    if arr.shape[0] != n:
        return None
    return arr


def _estimate_normals(points, np: Any):
    if points.shape[0] < 8:
        return None
    try:
        import open3d as o3d  # lazy
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(10)
        return np.asarray(pcd.normals, dtype=np.float64)
    except Exception:  # noqa: BLE001 -- normals are optional
        return None


# ---------------------------------------------------------------------------
# camera trajectory
# ---------------------------------------------------------------------------


def _write_camera_trajectory(
    asset_id: str,
    packets: Sequence[FrameRayPacket],
    out_path: Path,
    np: Any,
    floor_align_rotation: Any = None,
) -> dict[str, Any]:
    ply_path = out_path / "camera_trajectory.ply"
    tum_path = out_path / "camera_trajectory.tum.txt"
    json_path = out_path / "camera_trajectory.json"

    if not packets:
        return {
            "status": "blocked",
            "missing_object": "frame_ray_packets",
            "reason": "no_packets_for_camera_trajectory",
            "paths": {"ply": str(ply_path), "tum": str(tum_path), "json": str(json_path)},
        }

    # Sort by frame_id for a coherent polyline.
    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    # Transform poses into the floor-aligned frame so the trajectory matches the
    # floor-aligned occupancy/point-cloud (rigid: R -> R_align R, t -> R_align t).
    r_align = None
    if floor_align_rotation is not None:
        r_align = np.asarray(floor_align_rotation, dtype=np.float64)
    centers = []
    tum_lines = []
    json_frames = []
    for packet in ordered:
        T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
        R = T[:3, :3]
        t = T[:3, 3]
        if r_align is not None:
            R = r_align @ R
            t = r_align @ t
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
        centers.append(t)
        qx, qy, qz, qw = _rotation_to_quaternion(R, np)
        ts = float(int(packet.frame_id))  # no real timestamps -> use frame index
        tum_lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )
        json_frames.append(
            {
                "frame_id": int(packet.frame_id),
                "camera_center_world": [float(t[0]), float(t[1]), float(t[2])],
                "quaternion_xyzw": [float(qx), float(qy), float(qz), float(qw)],
                "T_world_camera": [[float(v) for v in row] for row in T],
            }
        )

    centers_arr = np.asarray(centers, dtype=np.float64)

    # PLY: write camera centres as points plus a polyline (edges) so the path is
    # visible in a viewer.
    _write_polyline_ply(ply_path, centers_arr, np)

    tum_path.write_text(
        "# TUM trajectory: timestamp tx ty tz qx qy qz qw "
        "(timestamp = frame_id; T_world_camera camera-to-world)\n"
        + "\n".join(tum_lines)
        + "\n",
        encoding="utf-8",
    )

    import json as _json
    json_path.write_text(
        _json.dumps(
            {
                "asset_id": str(asset_id),
                "convention": "T_world_camera_camera_to_world",
                "timestamp_meaning": "frame_id_no_real_timestamps",
                "frame_count": len(json_frames),
                "frames": json_frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "status": "written",
        "paths": {
            "ply": str(ply_path),
            "tum": str(tum_path),
            "json": str(json_path),
        },
        "frame_count": len(json_frames),
        "sizes_bytes": {
            "ply": _size(ply_path),
            "tum": _size(tum_path),
            "json": _size(json_path),
        },
    }


def _write_polyline_ply(path: Path, centers, np: Any) -> None:
    """Write an ASCII PLY with vertices + edges (a polyline along the path)."""
    n = centers.shape[0]
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Atlas3R camera trajectory (camera centres in world)",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        f"element edge {max(0, n - 1)}",
        "property int vertex1",
        "property int vertex2",
        "end_header",
    ]
    for i in range(n):
        lines.append(f"{centers[i, 0]:.6f} {centers[i, 1]:.6f} {centers[i, 2]:.6f}")
    for i in range(n - 1):
        lines.append(f"{i} {i + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rotation_to_quaternion(R, np: Any):
    """Camera-to-world rotation -> (qx, qy, qz, qw). Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ---------------------------------------------------------------------------
# occupancy channels
# ---------------------------------------------------------------------------


def _write_occupancy_channels(
    asset_id: str,
    occupancy_grid: OccupancyGrid2D | None,
    out_path: Path,
    np: Any,
) -> dict[str, Any]:
    path = out_path / "occupancy_channels.npz"
    if occupancy_grid is None:
        return {
            "status": "blocked",
            "missing_object": "occupancy_grid_2d",
            "reason": "no_occupancy_grid_to_export",
            "path": str(path),
        }

    g = occupancy_grid
    arrays = {
        "P_free": np.asarray(g.P_free, dtype=np.float64),
        "P_occupied_static": np.asarray(g.P_occupied_static, dtype=np.float64),
        "P_movable_static": np.asarray(g.P_movable_static, dtype=np.float64),
        "P_dynamic": np.asarray(g.P_dynamic, dtype=np.float64),
        "P_unknown": np.asarray(g.P_unknown, dtype=np.float64),
        "height_min_m": np.asarray(g.height_min_m, dtype=np.float64),
        "height_max_m": np.asarray(g.height_max_m, dtype=np.float64),
        "origin_world": np.asarray(g.origin_world, dtype=np.float64),
        "resolution_m": np.asarray(float(g.resolution_m), dtype=np.float64),
        "scale_uncertainty": np.asarray(float(g.scale_uncertainty), dtype=np.float64),
        "map_confidence": np.asarray(float(g.map_confidence), dtype=np.float64),
    }
    # grid_frame is a string -> store as 0-d object/str array.
    np.savez_compressed(
        str(path),
        grid_frame=np.asarray(str(g.grid_frame)),
        **arrays,
    )

    def _frac(channel):
        a = arrays[channel]
        return float(np.count_nonzero(a > 0.0)) / float(a.size) if a.size else 0.0

    channel_summary = {
        "grid_dims": [int(d) for d in arrays["P_free"].shape],
        "grid_frame": str(g.grid_frame),
        "resolution_m": float(g.resolution_m),
        "origin_world": [float(v) for v in g.origin_world],
        "free_cell_fraction": _frac("P_free"),
        "occupied_static_cell_fraction": _frac("P_occupied_static"),
        "movable_static_cell_fraction": _frac("P_movable_static"),
        "dynamic_cell_fraction": _frac("P_dynamic"),
        "unknown_cell_fraction": _frac("P_unknown"),
        "scale_uncertainty": float(g.scale_uncertainty),
        "map_confidence": float(g.map_confidence),
    }
    return {
        "status": "written",
        "path": str(path),
        "size_bytes": _size(path),
        "channel_summary": channel_summary,
    }


# ---------------------------------------------------------------------------
# 3D collision-band occupancy field
# ---------------------------------------------------------------------------


def _write_voxel_occupancy_3d(
    asset_id: str,
    grid3d: VoxelOccupancyGrid3D | None,
    out_path: Path,
    np: Any,
) -> dict[str, Any]:
    """Write the primary robot output -- the band-bounded ``VoxelOccupancyGrid3D``
    -- as a compressed ``.npz`` of per-channel probability volumes + band metadata.

    Nothing is fabricated: a missing field is an explicit blocked entry naming the
    absent object.
    """
    path = out_path / "voxel_occupancy_3d.npz"
    if grid3d is None:
        return {
            "status": "blocked",
            "missing_object": "voxel_occupancy_3d",
            "reason": "no_3d_occupancy_field_to_export",
            "path": str(path),
        }

    g = grid3d
    arrays = {
        "P_free": np.asarray(g.P_free, dtype=np.float64),
        "P_occupied_static": np.asarray(g.P_occupied_static, dtype=np.float64),
        "P_movable_static": np.asarray(g.P_movable_static, dtype=np.float64),
        "P_dynamic": np.asarray(g.P_dynamic, dtype=np.float64),
        "P_unknown": np.asarray(g.P_unknown, dtype=np.float64),
        "map_confidence": np.asarray(g.map_confidence, dtype=np.float64),
        "origin_world": np.asarray(g.origin_world, dtype=np.float64),
        "voxel_size_m": np.asarray(float(g.voxel_size_m)),
        "floor_axis": np.asarray(int(g.floor_axis)),
        "band_min_m": np.asarray(float(g.band_min_m)),
        "band_max_m": np.asarray(float(g.band_max_m)),
        "scale_uncertainty": np.asarray(float(g.scale_uncertainty)),
    }
    np.savez_compressed(
        str(path),
        grid_frame=np.asarray(str(g.grid_frame)),
        acceptance_category=np.asarray(str(g.acceptance_category.value)),
        **arrays,
    )

    def _frac(channel: str) -> float:
        a = arrays[channel]
        return float(np.count_nonzero(a > 0.0)) / float(a.size) if a.size else 0.0

    dims = [int(d) for d in arrays["P_free"].shape]
    floor_axis = int(g.floor_axis)
    field_summary = {
        "grid_dims": dims,
        "grid_frame": str(g.grid_frame),
        "floor_axis": floor_axis,
        "band_slices": dims[floor_axis] if 0 <= floor_axis < len(dims) else None,
        "voxel_size_m": float(g.voxel_size_m),
        "band_min_m": float(g.band_min_m),
        "band_max_m": float(g.band_max_m),
        "origin_world": [float(v) for v in g.origin_world],
        "acceptance_category": g.acceptance_category.value,
        "scale_uncertainty": float(g.scale_uncertainty),
        "free_voxel_fraction": _frac("P_free"),
        "occupied_static_voxel_fraction": _frac("P_occupied_static"),
        "movable_static_voxel_fraction": _frac("P_movable_static"),
        "dynamic_voxel_fraction": _frac("P_dynamic"),
        "mean_map_confidence": (
            float(np.mean(arrays["map_confidence"])) if arrays["map_confidence"].size else 0.0
        ),
    }
    return {
        "status": "written",
        "path": str(path),
        "size_bytes": _size(path),
        "field_summary": field_summary,
    }


# ---------------------------------------------------------------------------
# mesh (only if faithful)
# ---------------------------------------------------------------------------


def _write_mesh_if_feasible(
    asset_id: str,
    surface_points,
    surface_normals,
    voxel_map: VoxelMapState | None,
    out_path: Path,
    np: Any,
) -> dict[str, Any]:
    path = out_path / "mesh.ply"

    if surface_points is None or surface_points.shape[0] < 100:
        return {
            "status": "skipped",
            "reason": "mesh_skipped_point_cloud_only",
            "detail": "too_few_observed_surface_points_for_faithful_mesh",
            "point_count": 0 if surface_points is None else int(surface_points.shape[0]),
        }

    try:
        import open3d as o3d  # lazy
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": "mesh_skipped_point_cloud_only",
            "detail": f"open3d_unavailable:{type(exc).__name__}",
        }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(surface_points, dtype=np.float64))
    if surface_normals is not None and surface_normals.shape[0] == surface_points.shape[0]:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(surface_normals, dtype=np.float64))
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(10)

    # Ball-pivoting reconstructs ONLY where there are observed points (no
    # hole-filling / extrapolation) -> faithful to the data. Radii scale from the
    # average nearest-neighbour spacing.
    try:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        if dists.size == 0:
            raise RuntimeError("no neighbour distances")
        avg = float(np.mean(dists))
        radii = [avg * f for f in (1.5, 3.0, 6.0)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": "mesh_skipped_point_cloud_only",
            "detail": f"ball_pivoting_failed:{type(exc).__name__}:{exc}",
        }

    n_v = len(mesh.vertices)
    n_f = len(mesh.triangles)
    if n_v == 0 or n_f == 0:
        return {
            "status": "skipped",
            "reason": "mesh_skipped_point_cloud_only",
            "detail": "ball_pivoting_produced_empty_mesh",
        }

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    ok = o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    if not ok:
        return {
            "status": "skipped",
            "reason": "mesh_skipped_point_cloud_only",
            "detail": "open3d_write_triangle_mesh_returned_false",
        }

    return {
        "status": "written",
        "path": str(path),
        "method": "ball_pivoting_on_observed_points",
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.triangles)),
        "size_bytes": _size(path),
        "honesty": "observed_points_only_no_hole_filling",
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid_geometry(map_report: Mapping[str, Any] | None):
    if not isinstance(map_report, Mapping):
        return None, None
    grid_min = map_report.get("grid_min")
    eff_voxel = map_report.get("effective_voxel_size_m")
    if grid_min is None or eff_voxel is None:
        return None, None
    try:
        gm = tuple(float(v) for v in grid_min)
        ev = float(eff_voxel)
    except (TypeError, ValueError):
        return None, None
    if len(gm) != 3 or ev <= 0.0:
        return None, None
    return gm, ev


def _record_block(result: dict[str, Any], artifact: str, entry: Mapping[str, Any]) -> None:
    missing = entry.get("missing_object", artifact)
    reason = entry.get("reason", "blocked")
    result["blocked"].append(
        {"artifact": artifact, "missing_object": missing, "reason": reason}
    )
    result["blockers"].append(f"{artifact}_blocked:{missing}")


def _size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


__all__ = ["export_teacher_artifacts"]
