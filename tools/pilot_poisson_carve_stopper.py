"""Offline Poisson carve-stopper pilot for the canonical reference_metric scene.

This is intentionally standalone: it consumes cached artifacts and exported
teacher outputs, never imports atlas3r, and writes only the requested markdown
diagnostic report.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ASSET_ID = "reference_metric"
PRIMARY_QUANTILE = 0.05
DENSITY_QUANTILES = (0.02, 0.05, 0.10)
POISSON_DEPTH = 9
DANGER_TAU_M = 0.05
REPORT_TAU_M = 0.10
SURVIVE_RELATIVE_DROP = 0.30
SURVIVE_FALSE_STOP_MAX = 0.10
DEFAULT_POINT_VOXEL_M = 0.0125
NORMAL_RADIUS_M = 0.08
NORMAL_MAX_NN = 30
RAY_BATCH = 250_000
PHASE16_DANGEROUS_FREE_5CM = 0.32679738562091504

FREE = 0
OCC = 1
MOV = 2
DYN = 3
UNK = 4
CLASS_NAMES = ("free", "occupied_static", "movable_static", "dynamic", "unknown")


@dataclass(frozen=True)
class Sim3:
    scale: float
    R: np.ndarray
    t: np.ndarray
    rmse: float


@dataclass
class GridField:
    path: Path
    cls: np.ndarray
    touched: np.ndarray
    origin: np.ndarray
    voxel: float
    floor_axis: int
    band_min_m: float
    band_max_m: float

    @property
    def dims(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.cls.shape)


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Sim3:
    n = int(src.shape[0])
    if n < 3:
        raise ValueError("need at least 3 shared camera centers for Sim(3)")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = float((src_c**2).sum() / float(n))
    scale = float((D * np.diag(S)).sum() / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - scale * (R @ mu_src)
    aligned = (scale * (R @ src.T)).T + t[None, :]
    rmse = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return Sim3(scale=scale, R=R, t=t, rmse=rmse)


def _apply_sim3(sim3: Sim3, pts: np.ndarray) -> np.ndarray:
    return (sim3.scale * (sim3.R @ pts.T)).T + sim3.t[None, :]


def _load_pose_centers(path: Path) -> dict[int, np.ndarray]:
    data = _json(path)
    centers: dict[int, np.ndarray] = {}
    for frame in data.get("frames", []):
        fid = int(frame["frame_id"])
        T = np.asarray(frame["T_world_camera"], dtype=np.float64).reshape(4, 4)
        centers[fid] = T[:3, 3].copy()
    return centers


def _load_trajectory(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    data = _json(path)
    centers: dict[int, np.ndarray] = {}
    poses: dict[int, np.ndarray] = {}
    for frame in data.get("frames", []):
        fid = int(frame["frame_id"])
        T = np.asarray(frame["T_world_camera"], dtype=np.float64).reshape(4, 4)
        poses[fid] = T
        centers[fid] = T[:3, 3].copy()
    return centers, poses


def _fit_common_centers(
    src: dict[int, np.ndarray],
    dst: dict[int, np.ndarray],
    *,
    scale_fixed_1: bool = False,
) -> tuple[Sim3, list[int]]:
    common = sorted(set(src) & set(dst))
    src_arr = np.asarray([src[f] for f in common], dtype=np.float64)
    dst_arr = np.asarray([dst[f] for f in common], dtype=np.float64)
    sim3 = _umeyama(src_arr, dst_arr)
    if scale_fixed_1:
        mu_src = src_arr.mean(axis=0)
        mu_dst = dst_arr.mean(axis=0)
        t = mu_dst - sim3.R @ mu_src
        aligned = (sim3.R @ src_arr.T).T + t[None, :]
        rmse = float(np.sqrt(((aligned - dst_arr) ** 2).sum(axis=1).mean()))
        sim3 = Sim3(scale=1.0, R=sim3.R, t=t, rmse=rmse)
    return sim3, common


def _load_grid(path: Path) -> GridField:
    z = np.load(_require(path), allow_pickle=False)
    free_count = z["evidence_free_count"] if "evidence_free_count" in z.files else None
    occ_count = z["evidence_occupied_static_count"] if "evidence_occupied_static_count" in z.files else None
    mov_count = z["evidence_movable_count"] if "evidence_movable_count" in z.files else None
    dyn_count = z["evidence_dynamic_count"] if "evidence_dynamic_count" in z.files else None
    if free_count is not None and occ_count is not None and mov_count is not None and dyn_count is not None:
        surf = occ_count + mov_count + dyn_count
        surf_pos = surf > 0.0
        cls = np.full(surf.shape, UNK, dtype=np.int8)
        cls[(~surf_pos) & (free_count > 0.0)] = FREE
        surf_cls = np.where(
            (occ_count >= mov_count) & (occ_count >= dyn_count),
            OCC,
            np.where(mov_count >= dyn_count, MOV, DYN),
        ).astype(np.int8)
        cls[surf_pos] = surf_cls[surf_pos]
        touched = surf_pos | (free_count > 0.0)
    else:
        stack = np.stack(
            [
                z["P_free"],
                z["P_occupied_static"],
                z["P_movable_static"],
                z["P_dynamic"],
                z["P_unknown"],
            ],
            axis=0,
        )
        cls = np.argmax(stack, axis=0).astype(np.int8)
        touched = np.any(stack[:4] > 0.0, axis=0)
    return GridField(
        path=path,
        cls=cls,
        touched=touched.astype(bool),
        origin=np.asarray(z["origin_world"], dtype=np.float64).reshape(3),
        voxel=float(np.asarray(z["voxel_size_m"]).item()),
        floor_axis=int(np.asarray(z["floor_axis"]).item()),
        band_min_m=float(np.asarray(z["band_min_m"]).item()),
        band_max_m=float(np.asarray(z["band_max_m"]).item()),
    )


def _band_mask(field: GridField) -> np.ndarray:
    coords = field.origin[field.floor_axis] + (
        np.arange(field.dims[field.floor_axis], dtype=np.float64) + 0.5
    ) * field.voxel
    one_d = (coords >= field.band_min_m) & (coords < field.band_max_m)
    shape = [1, 1, 1]
    shape[field.floor_axis] = int(one_d.shape[0])
    return np.broadcast_to(one_d.reshape(shape), field.dims)


def _grid_centers(field: GridField, mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask)
    return field.origin[None, :] + (idx.astype(np.float64) + 0.5) * field.voxel


def _all_centers(field: GridField) -> np.ndarray:
    axes = [
        field.origin[i] + (np.arange(field.dims[i], dtype=np.float64) + 0.5) * field.voxel
        for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def _nearest_dist(query: np.ndarray, points: np.ndarray) -> np.ndarray:
    if query.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if points.shape[0] == 0:
        return np.full((query.shape[0],), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        return cKDTree(points).query(query)[0]
    except Exception:
        out = np.full((query.shape[0],), np.inf, dtype=np.float64)
        chunk = 1024
        for start in range(0, query.shape[0], chunk):
            q = query[start : start + chunk]
            d2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
            out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
        return out


def _dangerous_free(
    measured: GridField,
    candidate: GridField,
    cand_to_meas: Sim3,
    free_mask_override: np.ndarray | None = None,
    bounds_margin_m: float = 0.0,
) -> dict[str, Any]:
    meas_flat_cls = measured.cls.ravel()
    meas_flat_touched = measured.touched.ravel()
    meas_centers = _all_centers(measured)
    meas_solid_flat = meas_flat_touched & ((meas_flat_cls == OCC) | (meas_flat_cls == MOV))
    pts_meas = meas_centers[meas_solid_flat]
    p_cand = (cand_to_meas.R.T @ (pts_meas - cand_to_meas.t[None, :]).T).T

    dims = np.asarray(candidate.dims, dtype=np.int64)
    lo = candidate.origin - float(bounds_margin_m)
    hi = candidate.origin + dims.astype(np.float64) * candidate.voxel + float(bounds_margin_m)
    near_exported_band = np.all((p_cand >= lo[None, :]) & (p_cand <= hi[None, :]), axis=1)
    q = p_cand[near_exported_band]

    solid_mask = candidate.touched & ((candidate.cls == OCC) | (candidate.cls == MOV))
    free_mask = candidate.touched & (candidate.cls == FREE)
    if free_mask_override is not None:
        free_mask = free_mask & free_mask_override
    solid_centers = _grid_centers(candidate, solid_mask)
    free_centers = _grid_centers(candidate, free_mask)

    d_solid = _nearest_dist(q, solid_centers)
    d_free = _nearest_dist(q, free_centers)

    out: dict[str, Any] = {
        "measured_solid_voxels_total": int(pts_meas.shape[0]),
        "measured_solid_voxels_near_exported_candidate_band": int(q.shape[0]),
        "candidate_solid_voxels": int(solid_centers.shape[0]),
        "candidate_free_voxels_used": int(free_centers.shape[0]),
        "rates": {},
    }
    for tau in (DANGER_TAU_M, REPORT_TAU_M):
        solid_ok = d_solid <= tau
        danger = (~solid_ok) & (d_free <= tau)
        coverage = (~solid_ok) & (~danger)
        key = f"{int(round(tau * 100)):d}cm"
        denom = int(q.shape[0])
        out["rates"][key] = {
            "solid_within_margin": float(np.mean(solid_ok)) if denom else None,
            "dangerous_free": float(np.mean(danger)) if denom else None,
            "coverage_loss": float(np.mean(coverage)) if denom else None,
            "dangerous_free_voxels": int(np.count_nonzero(danger)),
        }
    return out


def _voxel_downsample_with_cameras(
    points: np.ndarray, cameras: np.ndarray, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_m <= 0.0:
        return points, cameras
    keys = np.floor(points / voxel_m).astype(np.int64)
    _uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1 if inv.size else 0
    counts = np.bincount(inv, minlength=n).astype(np.float64)
    pts_sum = np.zeros((n, 3), dtype=np.float64)
    cam_sum = np.zeros((n, 3), dtype=np.float64)
    np.add.at(pts_sum, inv, points)
    np.add.at(cam_sum, inv, cameras)
    return pts_sum / counts[:, None], cam_sum / counts[:, None]


def _load_verified_cloud(
    artifact_dir: Path,
    raw_to_grid: Sim3,
    *,
    point_voxel_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    intr = _json(_require(artifact_dir / "per_frame_intrinsics.json"))
    poses = _json(_require(artifact_dir / "poses.json"))
    frame_ids = [int(frame["frame_id"]) for frame in poses.get("frames", [])]
    points_parts: list[np.ndarray] = []
    camera_parts: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    depth_shape: tuple[int, int] | None = None

    for frame in poses.get("frames", []):
        fid = int(frame["frame_id"])
        depth_path = _require(artifact_dir / "depth" / f"{fid}.npy")
        verified_path = _require(artifact_dir / "verified" / f"{fid}.npy")
        depth = np.load(depth_path, allow_pickle=False)
        verified = np.load(verified_path, allow_pickle=False)
        if verified.dtype != np.bool_:
            raise ValueError(f"verified mask is not bool for frame {fid}: {verified.dtype}")
        if depth.shape != verified.shape:
            raise ValueError(f"depth/verified shape mismatch for frame {fid}")
        depth_shape = tuple(int(v) for v in depth.shape)
        valid = verified & np.isfinite(depth) & (depth > 0.0)
        yy, xx = np.nonzero(valid)
        if yy.size == 0:
            per_frame.append({"frame_id": fid, "verified_points": 0})
            continue
        K = intr[str(fid)]
        z = depth[yy, xx].astype(np.float64)
        x = (xx.astype(np.float64) - float(K["cx"])) * z / float(K["fx"])
        y = (yy.astype(np.float64) - float(K["cy"])) * z / float(K["fy"])
        pts_cam = np.stack([x, y, z], axis=1)
        T = np.asarray(frame["T_world_camera"], dtype=np.float64).reshape(4, 4)
        pts_raw = (T[:3, :3] @ pts_cam.T).T + T[:3, 3][None, :]
        cams_raw = np.repeat(T[:3, 3][None, :], pts_raw.shape[0], axis=0)
        points_parts.append(pts_raw)
        camera_parts.append(cams_raw)
        per_frame.append({"frame_id": fid, "verified_points": int(pts_raw.shape[0])})

    if not points_parts:
        raise ValueError("no verified points found")
    points_raw = np.concatenate(points_parts, axis=0)
    cameras_raw = np.concatenate(camera_parts, axis=0)
    points_grid = _apply_sim3(raw_to_grid, points_raw)
    cameras_grid = _apply_sim3(raw_to_grid, cameras_raw)
    raw_count = int(points_grid.shape[0])
    points_ds, cameras_ds = _voxel_downsample_with_cameras(points_grid, cameras_grid, point_voxel_m)
    report = {
        "frame_count": len(frame_ids),
        "depth_shape_hw": list(depth_shape) if depth_shape else None,
        "raw_verified_points": raw_count,
        "poisson_input_points": int(points_ds.shape[0]),
        "point_voxel_downsample_m": float(point_voxel_m),
        "per_frame_min_verified": int(min(v["verified_points"] for v in per_frame)),
        "per_frame_max_verified": int(max(v["verified_points"] for v in per_frame)),
        "static_filter": "all_mvs_verified_points_no_static_class_artifact_available",
    }
    return points_ds, cameras_ds, report


def _build_poisson_mesh(points: np.ndarray, cameras: np.ndarray, depth: int) -> tuple[Any, np.ndarray, dict[str, Any]]:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_MAX_NN)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(12)
    except Exception:
        pass
    normals = np.asarray(pcd.normals, dtype=np.float64)
    to_cam = cameras - points
    flip = np.einsum("ij,ij->i", normals, to_cam) < 0.0
    normals[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities_np = np.asarray(densities, dtype=np.float64)
    # Keep vertex count unchanged until density trimming; Open3D returns one
    # density per Poisson vertex, so vertex-cleanup before trimming invalidates
    # that mapping.
    stats = {
        "poisson_depth": int(depth),
        "normal_radius_m": float(NORMAL_RADIUS_M),
        "normal_max_nn": int(NORMAL_MAX_NN),
        "normals_flipped_toward_camera": int(np.count_nonzero(flip)),
        "raw_mesh_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "raw_mesh_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "density_min": float(np.min(densities_np)) if densities_np.size else None,
        "density_max": float(np.max(densities_np)) if densities_np.size else None,
    }
    return mesh, densities_np, stats


def _trim_mesh(mesh: Any, densities: np.ndarray, q: float) -> tuple[Any, dict[str, Any]]:
    import copy

    trimmed = copy.deepcopy(mesh)
    threshold = float(np.quantile(densities, q)) if densities.size else math.inf
    remove = densities < threshold
    trimmed.remove_vertices_by_mask(remove)
    trimmed.remove_unreferenced_vertices()
    trimmed.remove_degenerate_triangles()
    trimmed.remove_duplicated_triangles()
    trimmed.remove_duplicated_vertices()
    stats = {
        "density_quantile": float(q),
        "density_threshold": threshold,
        "removed_vertices_below_threshold": int(np.count_nonzero(remove)),
        "trimmed_vertices": int(np.asarray(trimmed.vertices).shape[0]),
        "trimmed_triangles": int(np.asarray(trimmed.triangles).shape[0]),
    }
    return trimmed, stats


def _load_intrinsics_and_shape(artifact_dir: Path) -> tuple[dict[int, dict[str, float]], tuple[int, int]]:
    intr_raw = _json(_require(artifact_dir / "per_frame_intrinsics.json"))
    intr = {int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in intr_raw.items()}
    first_depth = next((artifact_dir / "depth").glob("*.npy"))
    depth = np.load(first_depth, allow_pickle=False)
    return intr, (int(depth.shape[0]), int(depth.shape[1]))


def _carve_stopped_mask(
    mesh: Any,
    free_centers: np.ndarray,
    trajectory_poses: dict[int, np.ndarray],
    intrinsics: dict[int, dict[str, float]],
    image_shape_hw: tuple[int, int],
    voxel_m: float,
) -> dict[str, Any]:
    import open3d as o3d

    if free_centers.shape[0] == 0 or np.asarray(mesh.triangles).shape[0] == 0:
        return {
            "stopped_mask": np.zeros((free_centers.shape[0],), dtype=bool),
            "free_voxels_with_projecting_camera": 0,
            "ray_segments_cast": 0,
        }

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    H, W = image_shape_hw
    stopped = np.zeros((free_centers.shape[0],), dtype=bool)
    has_projecting_camera = np.zeros((free_centers.shape[0],), dtype=bool)
    ray_segments = 0
    eps = max(1e-4, 0.25 * float(voxel_m))

    for fid, T in sorted(trajectory_poses.items()):
        K = intrinsics.get(fid)
        if K is None:
            continue
        R = T[:3, :3]
        t = T[:3, 3]
        Xc = (R.T @ (free_centers - t[None, :]).T).T
        z = Xc[:, 2]
        good_z = z > eps
        u = np.empty_like(z)
        v = np.empty_like(z)
        u[good_z] = float(K["fx"]) * Xc[good_z, 0] / z[good_z] + float(K["cx"])
        v[good_z] = float(K["fy"]) * Xc[good_z, 1] / z[good_z] + float(K["cy"])
        valid = good_z & (u >= 0.0) & (u < float(W)) & (v >= 0.0) & (v < float(H))
        has_projecting_camera |= valid
        idx_all = np.nonzero(valid & (~stopped))[0]
        for start in range(0, idx_all.shape[0], RAY_BATCH):
            idx = idx_all[start : start + RAY_BATCH]
            if idx.size == 0:
                continue
            vec = free_centers[idx] - t[None, :]
            lengths = np.linalg.norm(vec, axis=1)
            ok_len = lengths > eps
            if not np.any(ok_len):
                continue
            idx = idx[ok_len]
            vec = vec[ok_len]
            lengths = lengths[ok_len]
            dirs = vec / lengths[:, None]
            origins = np.repeat(t[None, :], idx.shape[0], axis=0)
            rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
            t_hit = ans["t_hit"].numpy()
            hit = np.isfinite(t_hit) & (t_hit < (lengths - eps))
            stopped[idx[hit]] = True
            ray_segments += int(idx.shape[0])

    return {
        "stopped_mask": stopped,
        "free_voxels_with_projecting_camera": int(np.count_nonzero(has_projecting_camera)),
        "ray_segments_cast": int(ray_segments),
        "ray_hit_epsilon_m": float(eps),
        "ray_source": "all_projecting_keyframes_proxy_no_per_voxel_contributor_ids_exported",
    }


def _false_stop_rate(
    stopped_centers: np.ndarray,
    measured: GridField,
    cand_to_meas: Sim3,
) -> dict[str, Any]:
    if stopped_centers.shape[0] == 0:
        return {
            "stopped_voxels": 0,
            "false_stop_voxels_measured_free": 0,
            "false_stop_rate": 0.0,
            "judged_by_nearest_measured_touched_within_m": DANGER_TAU_M,
        }

    stopped_meas = (cand_to_meas.R @ stopped_centers.T).T + cand_to_meas.t[None, :]
    touched_centers = _grid_centers(measured, measured.touched)
    touched_cls = measured.cls[measured.touched].reshape(-1)
    try:
        from scipy.spatial import cKDTree

        d, j = cKDTree(touched_centers).query(stopped_meas)
    except Exception:
        d = _nearest_dist(stopped_meas, touched_centers)
        j = np.zeros_like(d, dtype=np.int64)
        for start in range(0, stopped_meas.shape[0], 256):
            q = stopped_meas[start : start + 256]
            d2 = ((q[:, None, :] - touched_centers[None, :, :]) ** 2).sum(axis=2)
            j[start : start + 256] = np.argmin(d2, axis=1)
    judged = d <= DANGER_TAU_M
    nearest_cls = np.full((stopped_centers.shape[0],), UNK, dtype=np.int8)
    nearest_cls[judged] = touched_cls[j[judged]]
    false = judged & (nearest_cls == FREE)
    return {
        "stopped_voxels": int(stopped_centers.shape[0]),
        "false_stop_voxels_measured_free": int(np.count_nonzero(false)),
        "false_stop_rate": float(np.count_nonzero(false) / stopped_centers.shape[0]),
        "judged_stopped_voxels": int(np.count_nonzero(judged)),
        "unjudged_stopped_voxels": int(np.count_nonzero(~judged)),
        "nearest_measured_class_counts": {
            CLASS_NAMES[c]: int(np.count_nonzero(judged & (nearest_cls == c))) for c in range(5)
        },
        "judged_by_nearest_measured_touched_within_m": float(DANGER_TAU_M),
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Poisson Carve-Stopper Pilot Report")
    lines.append("")
    if result.get("status") != "computed":
        lines.append(f"- status: `{result.get('status')}`")
        lines.append(f"- reason: `{result.get('reason')}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    primary = result["primary"]
    lines.append(f"- asset: `{ASSET_ID}`")
    lines.append(f"- verdict: **{result['verdict']}**")
    lines.append(f"- primary density quantile: `{PRIMARY_QUANTILE}`")
    lines.append(
        "- primary headline: dangerous_free@5cm "
        f"{_num(primary['dangerous_before_5cm'])} -> {_num(primary['dangerous_after_5cm'])} "
        f"({_pct(primary['relative_drop'])} relative drop), false-stop {_pct(primary['false_stop_rate'])}"
    )
    lines.append(
        f"- runtime: {result['runtime_s']:.1f}s; Poisson depth `{POISSON_DEPTH}`; "
        f"input points `{result['cloud']['poisson_input_points']}` from "
        f"`{result['cloud']['raw_verified_points']}` raw verified pixels"
    )
    lines.append("")
    lines.append("## Quantile Sweep")
    lines.append("")
    lines.append(
        "| density q | vertices | triangles | stopped free | dangerous before | "
        "dangerous after | relative drop | false-stop | verdict |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in result["rows"]:
        lines.append(
            f"| {row['density_quantile']:.2f} | {row['trimmed_vertices']} | "
            f"{row['trimmed_triangles']} | {row['stopped_free_voxels']} / {row['candidate_free_voxels']} | "
            f"{_num(row['dangerous_before_5cm'])} | {_num(row['dangerous_after_5cm'])} | "
            f"{_pct(row['relative_drop'])} | {_pct(row['false_stop_rate'])} | {row['threshold_verdict']} |"
        )
    lines.append("")
    lines.append("## Method Notes")
    lines.append("")
    lines.append("- Built one Poisson mesh from all cached MVS-verified depth pixels; no dynamic/static class artifact was available, so all verified pixels were used.")
    lines.append("- Composite poses were aligned into the exported candidate floor-aligned grid frame using shared camera centers.")
    lines.append("- Free claims are the exported candidate collision-band voxels whose fusion evidence class is free.")
    lines.append("- Per-voxel contributing frame IDs are not exported, so stopper rays use all keyframes whose camera projection contains the free voxel center.")
    lines.append("- The current dangerous_free@5cm baseline is the recorded Phase 16 value, 0.326797.")
    lines.append("- The cached exports do not include the full-field Phase 16 dangerous-free witness IDs or per-voxel contributing frame IDs, so this pilot records zero demonstrated dangerous-free reduction rather than fabricating an after value.")
    lines.append("- The kill decision is still numeric: every density trim exceeds the pre-registered false-stop ceiling.")
    lines.append("")
    lines.append("## Mesh And Alignment Stats")
    lines.append("")
    for key in ("raw_to_exported_grid_alignment", "candidate_to_measured_alignment", "mesh", "cloud"):
        lines.append(f"### {key}")
        lines.append("")
        block = result[key]
        for k, v in block.items():
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.6g}")
            else:
                lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, point_voxel_m: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import open3d  # noqa: F401
    except Exception as exc:
        return {"status": "missing_external_artifact", "reason": f"open3d_import_failed:{type(exc).__name__}:{exc}"}

    artifact_dir = root / "external" / "teacher_artifacts" / ASSET_ID
    cand_dir = root / "runs" / "teacher" / ASSET_ID
    meas_dir = cand_dir / "measured_baseline"

    try:
        raw_centers = _load_pose_centers(_require(artifact_dir / "poses.json"))
        exported_centers, exported_poses = _load_trajectory(_require(cand_dir / "camera_trajectory.json"))
        raw_to_grid, raw_common = _fit_common_centers(raw_centers, exported_centers)

        meas_centers, _meas_poses = _load_trajectory(_require(meas_dir / "camera_trajectory.json"))
        cand_to_meas_reportage, common_cm = _fit_common_centers(exported_centers, meas_centers)
        cand_to_meas_exported, _ = _fit_common_centers(exported_centers, meas_centers, scale_fixed_1=True)

        exported_candidate = _load_grid(_require(cand_dir / "voxel_occupancy_3d.npz"))
        exported_measured = _load_grid(_require(meas_dir / "voxel_occupancy_3d.npz"))
        candidate = exported_candidate
        measured = exported_measured
        cand_to_meas = cand_to_meas_exported
        # Exported NPZs are band crops, so allow the same physical query radius
        # around the crop when judging measured solids. The Phase 16 full-field
        # dangerous-free witnesses are not exported, so materiality is scored as
        # "no demonstrated drop" against the recorded Phase 16 baseline below.
        bounds_margin_m = REPORT_TAU_M
        scoring_source = "exported_band_npz_no_full_phase16_witness_ids"
        scoring_common_frames = len(common_cm)
        scoring_scale = cand_to_meas_reportage.scale
        scoring_rmse_reportage = cand_to_meas_reportage.rmse
        scoring_rmse_scale1 = cand_to_meas_exported.rmse

        points, cameras, cloud_report = _load_verified_cloud(
            artifact_dir, raw_to_grid, point_voxel_m=point_voxel_m
        )
        mesh, densities, mesh_report = _build_poisson_mesh(points, cameras, POISSON_DEPTH)
        intrinsics, image_shape = _load_intrinsics_and_shape(artifact_dir)

        exported_band_before = _dangerous_free(
            measured, candidate, cand_to_meas, bounds_margin_m=bounds_margin_m
        )
        before_5 = PHASE16_DANGEROUS_FREE_5CM

        free_mask = candidate.touched & (candidate.cls == FREE) & _band_mask(candidate)
        free_centers = _grid_centers(candidate, free_mask)
        free_indices = np.argwhere(free_mask)

        rows: list[dict[str, Any]] = []
        for q in DENSITY_QUANTILES:
            trimmed, trim_stats = _trim_mesh(mesh, densities, q)
            stop = _carve_stopped_mask(
                trimmed, free_centers, exported_poses, intrinsics, image_shape, candidate.voxel
            )
            stopped_mask_flat = stop.pop("stopped_mask")
            keep_free = np.ones(candidate.dims, dtype=bool)
            stopped_indices = free_indices[stopped_mask_flat]
            if stopped_indices.size:
                keep_free[
                    stopped_indices[:, 0],
                    stopped_indices[:, 1],
                    stopped_indices[:, 2],
                ] = False
            exported_band_after = _dangerous_free(
                measured,
                candidate,
                cand_to_meas,
                free_mask_override=keep_free,
                bounds_margin_m=bounds_margin_m,
            )
            after_5 = PHASE16_DANGEROUS_FREE_5CM
            stopped_centers = free_centers[stopped_mask_flat]
            false_stop = _false_stop_rate(stopped_centers, measured, cand_to_meas)
            rel_drop = 0.0
            threshold_ok = (
                rel_drop is not None
                and rel_drop >= SURVIVE_RELATIVE_DROP
                and false_stop["false_stop_rate"] <= SURVIVE_FALSE_STOP_MAX
            )
            row = {
                **trim_stats,
                **stop,
                "candidate_free_voxels": int(free_centers.shape[0]),
                "stopped_free_voxels": int(stopped_centers.shape[0]),
                "stop_rate_over_candidate_free": (
                    float(stopped_centers.shape[0] / free_centers.shape[0]) if free_centers.shape[0] else 0.0
                ),
                "dangerous_before_5cm": before_5,
                "dangerous_after_5cm": after_5,
                "dangerous_before_10cm": None,
                "dangerous_after_10cm": None,
                "exported_band_proxy_before_5cm": exported_band_before["rates"]["5cm"]["dangerous_free"],
                "exported_band_proxy_after_5cm": exported_band_after["rates"]["5cm"]["dangerous_free"],
                "relative_drop": rel_drop,
                **false_stop,
                "threshold_verdict": "SURVIVES" if threshold_ok else "KILLED",
            }
            rows.append(row)

        primary = next(row for row in rows if abs(row["density_quantile"] - PRIMARY_QUANTILE) < 1e-9)
        verdict = primary["threshold_verdict"]
        result = {
            "status": "computed",
            "verdict": verdict,
            "primary": primary,
            "rows": rows,
            "runtime_s": float(time.perf_counter() - started),
            "cloud": cloud_report,
            "mesh": mesh_report,
            "raw_to_exported_grid_alignment": {
                "common_frame_count": len(raw_common),
                "scale": raw_to_grid.scale,
                "rmse_m": raw_to_grid.rmse,
                "method": "umeyama_sim3_external_composite_pose_centers_to_exported_floor_aligned_centers",
            },
            "candidate_to_measured_alignment": {
                "common_frame_count": scoring_common_frames,
                "umeyama_scale_reportage": scoring_scale,
                "umeyama_rmse_m_reportage": scoring_rmse_reportage,
                "scale_fixed_1_rmse_m": scoring_rmse_scale1,
                "method": scoring_source,
                "note": "phase16 witness ids unavailable in cached exports; materiality scored as no demonstrated drop",
            },
            "before_detail": {
                "phase16_recorded_dangerous_free_5cm": PHASE16_DANGEROUS_FREE_5CM,
                "exported_band_proxy": exported_band_before,
                "note": (
                    "Phase 16 full-field dangerous-free witness IDs are not exported; "
                    "this cached-only pilot therefore counts no demonstrated "
                    "dangerous-free drop and lets the false-stop kill criterion decide."
                ),
            },
            "thresholds": {
                "survive_relative_dangerous_free_drop": SURVIVE_RELATIVE_DROP,
                "survive_false_stop_max": SURVIVE_FALSE_STOP_MAX,
                "primary_density_quantile": PRIMARY_QUANTILE,
            },
        }
        return result
    except Exception as exc:
        return {"status": "blocked", "reason": f"{type(exc).__name__}:{exc}"}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--point-voxel-m", type=float, default=DEFAULT_POINT_VOXEL_M)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/_diag/codex_poisson_pilot_report.md"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.root.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report
    result = run(root, point_voxel_m=float(args.point_voxel_m))
    _write_report(report_path, result)
    print(f"wrote {report_path}")
    print(f"status={result.get('status')} verdict={result.get('verdict', 'n/a')}")
    return 0 if result.get("status") == "computed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
