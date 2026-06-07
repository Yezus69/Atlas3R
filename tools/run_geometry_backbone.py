#!/usr/bin/env python3
"""Standalone monocular geometry-backbone runner for the Atlas3R M3 adapter.

This is the *honest* monocular geometry source. It runs real incremental
Structure-from-Motion (SfM) with OpenCV over a set of selected keyframes and
emits external artifacts under a gitignored directory that the M3 geometry
adapter later ingests.

Honesty guarantees (NON-NEGOTIABLE):
  * Nothing is fabricated. Poses come from real essential-matrix / PnP
    recovery; depth comes from real triangulated points (and optionally a real
    monocular depth model). If a frame fails to register, it is OMITTED -- no
    invented pose is ever written.
  * If SfM cannot recover >= 2 camera poses (degenerate motion, insufficient
    parallax, or too few matches), NO poses are written and the blocker
    ``insufficient_parallax_or_matches_for_sfm`` is recorded. The downstream
    teacher classifies accordingly.
  * Monocular SfM has a free global scale (gauge freedom). Unless real metric
    intrinsics AND a metric depth model are supplied, ``pose_units`` and depth
    ``units`` are ``unitless_similarity`` and ``metric_evidence`` is ``false``.
    This runner NEVER claims measured metric ground truth.
  * When no intrinsics file is supplied, a pinhole *prior* focal is assumed
    (focal = 0.8 * max(W, H), principal point at image center). This is
    recorded as an assumed prior in provenance and forces non-metric units.
  * If the monocular depth model cannot be loaded, dense depth is NOT
    fabricated: the runner falls back to sparse triangulated depth and records
    the blocker ``dense_monocular_depth_model_unavailable``.

This file is a CLI tool. It is deliberately NOT importable by the
``atlas3r`` package and keeps every heavy dependency (numpy/cv2/torch) lazily
imported inside ``main`` / helper functions so importing this module stays
cheap and side-effect free.

Artifact layout (written under ``--out-dir``/<asset_id>/):
  backbone_manifest.json   run-level manifest + provenance
  intrinsics.json          pinhole intrinsics (real or assumed prior)
  poses.json               T_world_camera (camera-to-world) per registered frame
  depth_meta.json          depth convention/units/density/model
  depth/<frame_id>.npy     float32 [H, W] depth (sparse triangulated or dense)
  confidence/<frame_id>.npy  optional float32 [H, W] in [0, 1]
  run_report.json          honest run report with metrics and blockers
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Real monocular SfM geometry-backbone runner for Atlas3R M3.",
    )
    parser.add_argument("--asset-id", required=True, help="Canonical asset id, e.g. phone_room.")
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory holding decoded RGB frames for the asset.",
    )
    parser.add_argument(
        "--frame-glob",
        default="frame_*.jpg",
        help="Glob used to enumerate/sort frames for zero-based frame_id ordering.",
    )
    parser.add_argument(
        "--keyframes",
        default=None,
        help=(
            "Comma-separated frame ids, OR a path to runs/m1/<asset>_keyframes.json. "
            "If omitted, all frames matching --frame-glob are used (evenly capped)."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help=(
            "Optional path to a pinhole intrinsics JSON, or an inline JSON object "
            "with fx,fy,cx,cy,width_px,height_px. If absent, a prior focal is assumed "
            "(focal = 0.8 * max(W, H)) and units stay unitless_similarity."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="external/teacher_artifacts",
        help="Root output dir; artifacts are written under <out-dir>/<asset_id>/.",
    )
    parser.add_argument(
        "--use-depth-model",
        action="store_true",
        help="Attempt a real monocular depth model (MiDaS via torch.hub) for dense depth.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the depth model: auto|cpu|cuda (default auto).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=12,
        help="Cap on number of keyframes used for SfM (evenly subsampled if exceeded).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4096,
        help="Max SIFT features per frame.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Frame discovery / keyframe selection
# --------------------------------------------------------------------------- #
def _natural_key(path):
    import re

    name = path.name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def _enumerate_frames(frames_dir, frame_glob):
    """Return sorted list of frame paths. Index == zero-based frame_id (M1 convention)."""
    from pathlib import Path

    base = Path(frames_dir)
    if not base.is_dir():
        return []
    frames = sorted(base.glob(frame_glob), key=_natural_key)
    return frames


def _load_keyframe_spec(keyframes_arg, all_frames):
    """Resolve the --keyframes argument into (frame_id, frame_path) pairs.

    Supports three forms:
      * None              -> use every frame index.
      * "13,178,228"      -> explicit comma-separated zero-based ids.
      * a JSON path        -> runs/m1/<asset>_keyframes.json with
                             ``selected_frame_ids`` and an optional
                             ``metadata.selected_frame_sources`` id->filename map.
    Returns (pairs, source_description, warnings).
    """
    import json
    from pathlib import Path

    warnings = []
    n = len(all_frames)

    def _by_index(frame_id):
        if 0 <= frame_id < n:
            return all_frames[frame_id]
        return None

    if keyframes_arg is None:
        pairs = [(i, all_frames[i]) for i in range(n)]
        return pairs, "all_frames_in_glob", warnings

    candidate = Path(keyframes_arg)
    if candidate.exists() and candidate.is_file():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"keyframes_json_unreadable:{exc}")
            return [], f"unreadable_keyframes_json:{candidate}", warnings
        frame_ids = data.get("selected_frame_ids")
        if not isinstance(frame_ids, list):
            warnings.append("keyframes_json_missing_selected_frame_ids")
            return [], f"invalid_keyframes_json:{candidate}", warnings
        sources = {}
        meta = data.get("metadata")
        if isinstance(meta, dict):
            raw_sources = meta.get("selected_frame_sources")
            if isinstance(raw_sources, dict):
                sources = raw_sources
        pairs = []
        for raw_id in frame_ids:
            if not isinstance(raw_id, int) or raw_id < 0:
                continue
            # Prefer the explicit filename mapping from M1; it is authoritative
            # because frame filenames may be 1-based while frame_ids are 0-based.
            path = None
            src_name = sources.get(str(raw_id))
            if isinstance(src_name, str) and src_name.strip():
                explicit = all_frames[0].parent / src_name if all_frames else Path(src_name)
                if explicit.exists():
                    path = explicit
                else:
                    warnings.append(f"keyframe_source_missing:{src_name}")
            if path is None:
                path = _by_index(raw_id)
            if path is not None and path.exists():
                pairs.append((raw_id, path))
            else:
                warnings.append(f"keyframe_id_unresolved:{raw_id}")
        return pairs, f"m1_keyframes_json:{candidate.name}", warnings

    # Comma-separated id list.
    pairs = []
    for token in str(keyframes_arg).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            frame_id = int(token)
        except ValueError:
            warnings.append(f"keyframe_token_not_int:{token}")
            continue
        path = _by_index(frame_id)
        if path is not None and path.exists():
            pairs.append((frame_id, path))
        else:
            warnings.append(f"keyframe_id_out_of_range:{frame_id}")
    return pairs, "explicit_id_list", warnings


def _evenly_cap(pairs, max_frames):
    """Subsample pairs to at most ``max_frames`` keeping temporal spread."""
    if max_frames <= 0 or len(pairs) <= max_frames:
        return pairs
    import numpy as np

    idx = np.unique(np.rint(np.linspace(0, len(pairs) - 1, num=max_frames)).astype(int))
    return [pairs[i] for i in idx]


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def _resolve_intrinsics(intrinsics_arg, width_px, height_px):
    """Return (intrinsics_dict, intrinsics_source, assumed_prior_bool).

    intrinsics_dict: {model, fx, fy, cx, cy, width_px, height_px}
    If no usable intrinsics are supplied, a pinhole prior is assumed and the
    third return value is True (forces non-metric / unitless similarity).
    """
    import json
    from pathlib import Path

    def _assumed_prior():
        focal = 0.8 * float(max(width_px, height_px))
        return (
            {
                "model": "pinhole",
                "fx": focal,
                "fy": focal,
                "cx": width_px / 2.0,
                "cy": height_px / 2.0,
                "width_px": int(width_px),
                "height_px": int(height_px),
            },
            "assumed_prior_focal_0.8_max_wh",
            True,
        )

    if not intrinsics_arg:
        return _assumed_prior()

    data = None
    candidate = Path(intrinsics_arg)
    source_desc = None
    if candidate.exists() and candidate.is_file():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            source_desc = f"intrinsics_file:{candidate.name}"
        except (OSError, json.JSONDecodeError):
            data = None
    if data is None:
        try:
            data = json.loads(intrinsics_arg)
            source_desc = "intrinsics_inline_json"
        except (json.JSONDecodeError, TypeError):
            data = None

    if not isinstance(data, dict):
        return _assumed_prior()

    # Accept a nested {"intrinsics": {...}} too.
    if "fx" not in data and isinstance(data.get("intrinsics"), dict):
        data = data["intrinsics"]

    try:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        w = int(data.get("width_px", width_px))
        h = int(data.get("height_px", height_px))
    except (KeyError, TypeError, ValueError):
        return _assumed_prior()

    if not (fx > 0 and fy > 0 and w > 0 and h > 0):
        return _assumed_prior()

    return (
        {
            "model": "pinhole",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width_px": w,
            "height_px": h,
        },
        source_desc or "intrinsics_supplied",
        False,
    )


def _intrinsics_matrix(intr, np):
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Feature detection and matching
# --------------------------------------------------------------------------- #
def _load_gray(path, target_size, cv2, np):
    """Load an image as grayscale, resizing to (W, H) if it differs from intrinsics."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    target_w, target_h = target_size
    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img


def _detect_features(frames, target_size, max_features, cv2, np):
    """Detect SIFT features per frame. Returns list of (keypoints_xy, descriptors)."""
    sift = cv2.SIFT_create(nfeatures=int(max_features))
    out = []
    for frame_id, path in frames:
        gray = _load_gray(path, target_size, cv2, np)
        if gray is None:
            out.append((frame_id, path, None, None))
            continue
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < 8:
            out.append((frame_id, path, None, None))
            continue
        pts = np.array([kp.pt for kp in kps], dtype=np.float64)
        out.append((frame_id, path, pts, desc))
    return out


def _match_pair(desc_a, desc_b, cv2, np):
    """Lowe-ratio-tested mutual matches between two descriptor sets.

    Returns (idx_a, idx_b) integer arrays into the respective keypoint lists.
    """
    if desc_a is None or desc_b is None:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_a.astype(np.float32), desc_b.astype(np.float32), k=2)
    idx_a = []
    idx_b = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < 0.75 * n.distance:
            idx_a.append(m.queryIdx)
            idx_b.append(m.trainIdx)
    return np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _Rt_to_T_world_camera(R_cw, t_cw, np):
    """Convert OpenCV world->camera [R|t] into a 4x4 camera-to-world transform.

    OpenCV convention: x_cam = R_cw @ x_world + t_cw  (R_cw, t_cw are world->cam).
    Camera-to-world: R_wc = R_cw^T, c = -R_cw^T @ t_cw (camera center in world).
    T_world_camera = [[R_wc, c], [0,0,0,1]].
    """
    R_wc = R_cw.T
    c = -R_wc @ t_cw.reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = c
    return T


def _project_world_to_camera(X_world, R_cw, t_cw, K, np):
    """Project world points into a camera. Returns (uv [N,2], z [N], valid mask)."""
    X_cam = (R_cw @ X_world.T).T + t_cw.reshape(1, 3)
    z = X_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((X_world.shape[0], 2), np.nan, dtype=np.float64)
    safe = z.copy()
    safe[~valid] = 1.0
    proj = (K @ X_cam.T).T
    uv[:, 0] = proj[:, 0] / safe
    uv[:, 1] = proj[:, 1] / safe
    return uv, z, valid


# --------------------------------------------------------------------------- #
# Incremental SfM
# --------------------------------------------------------------------------- #
def _run_sfm(feature_table, K, cv2, np):
    """Real incremental SfM over the keyframes.

    Pipeline:
      1. Pairwise SIFT + ratio-test matches between consecutive keyframes.
      2. Seed pair = consecutive pair with the most inlier matches that yields a
         well-conditioned essential matrix (enough parallax). Initialize the two
         camera poses (first = identity world frame) and triangulate the seed.
      3. Register each remaining frame by 2D-3D PnP (RANSAC) against the growing
         point cloud, then triangulate new points with already-registered
         neighbours.

    Returns a dict with per-frame world->camera poses (only for registered
    frames), the 3D point cloud, per-point track observations, mean reprojection
    error, and diagnostic counters. Frames that cannot be registered are simply
    absent from ``poses`` -- never invented.
    """
    valid_frames = [(fid, path, pts, desc) for (fid, path, pts, desc) in feature_table if desc is not None]
    diagnostics = {
        "frames_with_features": len(valid_frames),
        "seed_pair": None,
        "seed_inliers": 0,
        "pnp_registrations": [],
        "pairwise_match_counts": [],
    }
    if len(valid_frames) < 2:
        return {
            "poses_cw": {},
            "points_world": np.empty((0, 3), dtype=np.float64),
            "point_tracks": [],
            "mean_reproj_error_px": None,
            "diagnostics": diagnostics,
            "registered_order": [],
        }

    order_index = {fid: i for i, (fid, *_rest) in enumerate(valid_frames)}
    feat_by_id = {fid: (pts, desc) for (fid, _p, pts, desc) in valid_frames}

    # 1) Pairwise matches between consecutive keyframes (temporal adjacency).
    pair_matches = {}
    for a in range(len(valid_frames) - 1):
        fid_a = valid_frames[a][0]
        fid_b = valid_frames[a + 1][0]
        pts_a, desc_a = feat_by_id[fid_a]
        pts_b, desc_b = feat_by_id[fid_b]
        ia, ib = _match_pair(desc_a, desc_b, cv2, np)
        pair_matches[(fid_a, fid_b)] = (ia, ib)
        diagnostics["pairwise_match_counts"].append(
            {"from": fid_a, "to": fid_b, "matches": int(len(ia))}
        )

    # 2) Choose the seed pair: best essential-matrix geometry with parallax.
    seed = _select_seed_pair(valid_frames, feat_by_id, pair_matches, K, cv2, np)
    if seed is None:
        return {
            "poses_cw": {},
            "points_world": np.empty((0, 3), dtype=np.float64),
            "point_tracks": [],
            "mean_reproj_error_px": None,
            "diagnostics": diagnostics,
            "registered_order": [],
        }

    fid0, fid1, R1, t1, inl_a, inl_b = seed
    diagnostics["seed_pair"] = [fid0, fid1]
    diagnostics["seed_inliers"] = int(len(inl_a))

    poses_cw = {
        fid0: (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
        fid1: (R1, t1.reshape(3)),
    }

    pts0, _ = feat_by_id[fid0]
    pts1, _ = feat_by_id[fid1]
    X_seed = _triangulate(
        pts0[inl_a], pts1[inl_b],
        poses_cw[fid0][0], poses_cw[fid0][1],
        poses_cw[fid1][0], poses_cw[fid1][1],
        K, np,
    )
    # point_tracks: list of dicts {X: idx into points_world, obs: {fid: kp_idx}}
    points_world = []
    point_tracks = []
    good = _finite_in_front(X_seed, poses_cw[fid0], poses_cw[fid1], np)
    for j in range(X_seed.shape[0]):
        if not good[j]:
            continue
        points_world.append(X_seed[j])
        point_tracks.append({"obs": {fid0: int(inl_a[j]), fid1: int(inl_b[j])}})

    # 3) Register remaining frames in temporal order via PnP, then grow points.
    registered = {fid0, fid1}
    ordered_ids = [fid for (fid, *_rest) in valid_frames]
    progress = True
    while progress:
        progress = False
        for fid in ordered_ids:
            if fid in registered:
                continue
            obj_pts, img_pts, used_track_idx = _gather_2d3d(
                fid, registered, ordered_ids, feat_by_id, pair_matches,
                points_world, point_tracks, np,
            )
            if obj_pts.shape[0] < 6:
                continue
            ok, R_cw, t_cw, inlier_mask = _solve_pnp(obj_pts, img_pts, K, cv2, np)
            if not ok:
                continue
            poses_cw[fid] = (R_cw, t_cw.reshape(3))
            registered.add(fid)
            progress = True
            n_inl = int(inlier_mask.sum()) if inlier_mask is not None else obj_pts.shape[0]
            diagnostics["pnp_registrations"].append(
                {"frame_id": fid, "object_points": int(obj_pts.shape[0]), "inliers": n_inl}
            )
            # Attach this frame's PnP-inlier observations to the matched tracks so
            # they participate in subsequent triangulation/reprojection bookkeeping.
            for k, (tidx, fid_kp) in enumerate(used_track_idx):
                if inlier_mask is None or inlier_mask[k]:
                    point_tracks[tidx]["obs"].setdefault(fid, fid_kp)
            _grow_points_for_frame(
                fid, registered, ordered_ids, feat_by_id, pair_matches,
                poses_cw, points_world, point_tracks, K, cv2, np,
            )

    if not points_world:
        points_world_arr = np.empty((0, 3), dtype=np.float64)
    else:
        points_world_arr = np.asarray(points_world, dtype=np.float64)

    mean_err, per_frame_err = _mean_reprojection_error(
        poses_cw, points_world_arr, point_tracks, feat_by_id, K, np
    )
    diagnostics["per_frame_reproj_error_px"] = per_frame_err

    return {
        "poses_cw": poses_cw,
        "points_world": points_world_arr,
        "point_tracks": point_tracks,
        "mean_reproj_error_px": mean_err,
        "diagnostics": diagnostics,
        "registered_order": [fid for fid in ordered_ids if fid in poses_cw],
    }


def _select_seed_pair(valid_frames, feat_by_id, pair_matches, K, cv2, np):
    """Pick the consecutive pair giving the most essential-matrix inliers with
    sufficient parallax. Returns (fid0, fid1, R, t, inlier_idx_a, inlier_idx_b)."""
    best = None
    best_inliers = 0
    for a in range(len(valid_frames) - 1):
        fid_a = valid_frames[a][0]
        fid_b = valid_frames[a + 1][0]
        ia, ib = pair_matches.get((fid_a, fid_b), (None, None))
        if ia is None or len(ia) < 15:
            continue
        pts_a, _ = feat_by_id[fid_a]
        pts_b, _ = feat_by_id[fid_b]
        p_a = pts_a[ia]
        p_b = pts_b[ib]
        E, mask = cv2.findEssentialMat(
            p_a, p_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None or E.shape != (3, 3):
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < 15:
            continue
        n_good, R, t, pose_mask = cv2.recoverPose(E, p_a[mask], p_b[mask], K)
        if n_good < 12:
            continue
        pose_mask = pose_mask.ravel().astype(bool)
        sel_a = ia[mask][pose_mask]
        sel_b = ib[mask][pose_mask]
        if len(sel_a) < 12:
            continue
        # Parallax check: median angle between bearing vectors must be non-trivial.
        if not _has_parallax(pts_a[sel_a], pts_b[sel_b], K, np):
            continue
        if len(sel_a) > best_inliers:
            best_inliers = len(sel_a)
            best = (fid_a, fid_b, R, t, sel_a, sel_b)
    return best


def _has_parallax(pa, pb, K, np, min_median_deg=0.5):
    """Median bearing-angle change between matched points; rejects pure rotation."""
    Kinv = np.linalg.inv(K)
    ha = np.hstack([pa, np.ones((pa.shape[0], 1))])
    hb = np.hstack([pb, np.ones((pb.shape[0], 1))])
    da = (Kinv @ ha.T).T
    db = (Kinv @ hb.T).T
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    cos = np.clip(np.sum(da * db, axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos))
    return float(np.median(angles_deg)) >= min_median_deg


def _triangulate(p_a, p_b, R_a, t_a, R_b, t_b, K, np):
    """Triangulate matched 2D points from two world->camera poses. Returns [N,3]."""
    import cv2

    P_a = K @ np.hstack([R_a, t_a.reshape(3, 1)])
    P_b = K @ np.hstack([R_b, t_b.reshape(3, 1)])
    X_h = cv2.triangulatePoints(P_a, P_b, p_a.T, p_b.T)
    X = (X_h[:3] / X_h[3]).T
    return X.astype(np.float64)


def _finite_in_front(X, pose_a, pose_b, np):
    """Mask of points that are finite and in front of both cameras."""
    R_a, t_a = pose_a
    R_b, t_b = pose_b
    za = (R_a @ X.T).T[:, 2] + t_a[2]
    zb = (R_b @ X.T).T[:, 2] + t_b[2]
    finite = np.all(np.isfinite(X), axis=1)
    return finite & (za > 1e-4) & (zb > 1e-4) & (np.linalg.norm(X, axis=1) < 1e4)


def _gather_2d3d(fid, registered, ordered_ids, feat_by_id, pair_matches,
                 points_world, point_tracks, np):
    """Collect 2D-3D correspondences for PnP of frame ``fid``.

    For every registered neighbour that has a match edge with ``fid``, find
    keypoints in ``fid`` matched to keypoints belonging to an existing 3D track.
    """
    pts_fid, _ = feat_by_id[fid]
    # Build keypoint-index -> track-index lookup per registered frame.
    kp_to_track = {}
    for tidx, track in enumerate(point_tracks):
        for obs_fid, kp_idx in track["obs"].items():
            kp_to_track.setdefault(obs_fid, {})[kp_idx] = tidx

    obj_pts = []
    img_pts = []
    track_refs = []
    seen_tracks = set()
    for other in registered:
        ia, ib, swap = _edge_matches(other, fid, pair_matches)
        if ia is None:
            continue
        other_kp_to_track = kp_to_track.get(other, {})
        for k in range(len(ia)):
            other_kp = ia[k]
            fid_kp = ib[k]
            tidx = other_kp_to_track.get(int(other_kp))
            if tidx is None or tidx in seen_tracks:
                continue
            seen_tracks.add(tidx)
            obj_pts.append(points_world[tidx])
            img_pts.append(pts_fid[int(fid_kp)])
            track_refs.append((tidx, int(fid_kp)))
    if not obj_pts:
        return (np.empty((0, 3)), np.empty((0, 2)), [])
    return (
        np.asarray(obj_pts, dtype=np.float64),
        np.asarray(img_pts, dtype=np.float64),
        track_refs,
    )


def _edge_matches(fid_from, fid_to, pair_matches):
    """Return (idx_from, idx_to, swapped) for the match edge between two frames."""
    if (fid_from, fid_to) in pair_matches:
        ia, ib = pair_matches[(fid_from, fid_to)]
        if len(ia) == 0:
            return None, None, False
        return ia, ib, False
    if (fid_to, fid_from) in pair_matches:
        ib, ia = pair_matches[(fid_to, fid_from)]
        if len(ia) == 0:
            return None, None, True
        return ia, ib, True
    return None, None, False


def _solve_pnp(obj_pts, img_pts, K, cv2, np):
    """RANSAC PnP -> (ok, R_cw, t_cw, inlier_mask)."""
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.reshape(-1, 1, 3),
            img_pts.reshape(-1, 1, 2),
            K,
            dist,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return False, None, None, None
    if not ok or inliers is None or len(inliers) < 6:
        return False, None, None, None
    R_cw, _ = cv2.Rodrigues(rvec)
    inlier_mask = np.zeros(obj_pts.shape[0], dtype=bool)
    inlier_mask[inliers.ravel()] = True
    return True, R_cw, tvec, inlier_mask


def _grow_points_for_frame(fid, registered, ordered_ids, feat_by_id, pair_matches,
                           poses_cw, points_world, point_tracks, K, cv2, np):
    """Triangulate new 3D points between newly-registered ``fid`` and a registered
    neighbour, appending tracks not yet represented in the cloud."""
    # Existing keypoint coverage for fid (avoid duplicating already-tracked kps).
    fid_tracked = set()
    for track in point_tracks:
        if fid in track["obs"]:
            fid_tracked.add(track["obs"][fid])

    pts_fid, _ = feat_by_id[fid]
    R_b, t_b = poses_cw[fid]
    for other in registered:
        if other == fid:
            continue
        ia, ib, _swap = _edge_matches(other, fid, pair_matches)
        if ia is None:
            continue
        pts_other, _ = feat_by_id[other]
        R_a, t_a = poses_cw[other]
        new_a = []
        new_b = []
        new_pairs = []
        for k in range(len(ia)):
            okp = int(ia[k])
            fkp = int(ib[k])
            if fkp in fid_tracked:
                continue
            new_a.append(pts_other[okp])
            new_b.append(pts_fid[fkp])
            new_pairs.append((okp, fkp))
        if len(new_a) < 1:
            continue
        X_new = _triangulate(
            np.asarray(new_a), np.asarray(new_b), R_a, t_a, R_b, t_b, K, np
        )
        good = _finite_in_front(X_new, (R_a, t_a), (R_b, t_b), np)
        for j in range(X_new.shape[0]):
            if not good[j]:
                continue
            okp, fkp = new_pairs[j]
            if fkp in fid_tracked:
                continue
            fid_tracked.add(fkp)
            points_world.append(X_new[j])
            point_tracks.append({"obs": {other: okp, fid: fkp}})


def _mean_reprojection_error(poses_cw, points_world, point_tracks, feat_by_id, K, np):
    """Mean reprojection error (px) across all track observations; also per-frame."""
    if points_world.shape[0] == 0 or not point_tracks:
        return None, {}
    errs = []
    per_frame = {}
    for tidx, track in enumerate(point_tracks):
        if tidx >= points_world.shape[0]:
            continue
        X = points_world[tidx]
        for fid, kp_idx in track["obs"].items():
            if fid not in poses_cw:
                continue
            R_cw, t_cw = poses_cw[fid]
            X_cam = R_cw @ X + t_cw
            if X_cam[2] <= 1e-6:
                continue
            u = K[0, 0] * X_cam[0] / X_cam[2] + K[0, 2]
            v = K[1, 1] * X_cam[1] / X_cam[2] + K[1, 2]
            pts, _ = feat_by_id[fid]
            obs = pts[kp_idx]
            e = float(np.hypot(u - obs[0], v - obs[1]))
            if np.isfinite(e):
                errs.append(e)
                per_frame.setdefault(fid, []).append(e)
    if not errs:
        return None, {}
    per_frame_mean = {int(fid): float(np.mean(vals)) for fid, vals in per_frame.items()}
    return float(np.mean(errs)), per_frame_mean


# --------------------------------------------------------------------------- #
# Depth construction
# --------------------------------------------------------------------------- #
def _sparse_radial_depth(fid, R_cw, t_cw, points_world, point_tracks, intr, np):
    """Build a SPARSE radial-range depth map for a frame from triangulated points.

    For each track observed in this frame, project the 3D point and write the
    radial range (Euclidean distance from camera center) at the observed pixel.
    Depth comes solely from real triangulated 3D points -- never fabricated.
    Returns (depth [H,W] float32, confidence [H,W] float32, valid_count).
    """
    h = int(intr["height_px"])
    w = int(intr["width_px"])
    depth = np.zeros((h, w), dtype=np.float32)
    conf = np.zeros((h, w), dtype=np.float32)
    count = 0
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    for tidx, track in enumerate(point_tracks):
        if fid not in track["obs"] or tidx >= points_world.shape[0]:
            continue
        X = points_world[tidx]
        X_cam = R_cw @ X + t_cw
        z = X_cam[2]
        if z <= 1e-6:
            continue
        u = fx * X_cam[0] / z + cx
        v = fy * X_cam[1] / z + cy
        iu = int(round(u))
        iv = int(round(v))
        if 0 <= iu < w and 0 <= iv < h:
            radial = float(np.linalg.norm(X_cam))
            if radial > 0 and np.isfinite(radial):
                depth[iv, iu] = radial
                conf[iv, iu] = 1.0
                count += 1
    return depth, conf, count


def _load_depth_model(device_arg):
    """Attempt to load MiDaS via torch.hub. Returns (model, transform, device, info)
    or (None, None, None, blocker_string)."""
    try:
        import torch
    except ImportError as exc:
        return None, None, None, f"torch_unavailable:{exc}"

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    last_error = "no_model_attempted"
    for model_name in ("DPT_Hybrid", "MiDaS_small"):
        try:
            model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_name == "MiDaS_small":
                transform = transforms.small_transform
            else:
                transform = transforms.dpt_transform
            model.to(device).eval()
            return model, transform, device, {"model_name": model_name, "device": device}
        except Exception as exc:  # noqa: BLE001 - torch.hub raises many exception types
            last_error = exc
            continue
    return None, None, None, f"dense_monocular_depth_model_unavailable:{type(last_error).__name__}:{last_error}"


def _dense_depth_for_frame(model, transform, device, frame_path, intr, cv2, np):
    """Run MiDaS to get a dense RELATIVE inverse-depth-ish prediction, resized to
    intrinsics image size. Returns a positive [H,W] float32 relative depth or None."""
    import torch

    img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sample = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = model(sample)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(int(intr["height_px"]), int(intr["width_px"])),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    rel = pred.detach().cpu().numpy().astype(np.float64)
    # MiDaS predicts affine-invariant inverse depth (disparity-like): larger =
    # nearer. Return it as a strictly-positive disparity map and align it to the
    # sparse cloud in DISPARITY space downstream (where MiDaS is affine-invariant).
    # This avoids the 1/(min-disparity) divergence a direct depth conversion has,
    # which previously blew the farthest pixel up to ~1e9 and exploded the extent.
    if not np.any(np.isfinite(rel)):
        return None
    lo = float(np.nanmin(np.where(np.isfinite(rel), rel, np.nan)))
    disparity = (rel - lo) + 1e-3  # strictly > 0; the affine shift is absorbed by alignment
    disparity = np.where(np.isfinite(disparity) & (disparity > 0.0), disparity, np.nan)
    return disparity.astype(np.float32)


def _align_dense_to_sparse(disparity, sparse_depth, sparse_conf, np):
    """Align MiDaS affine-invariant disparity to the sparse triangulated cloud in
    DISPARITY (inverse-depth) space, then convert to a bounded radial depth map.

    Fits ``sparse_disparity ~ a * disparity + b`` (robust IRLS) at sparse pixels,
    where ``sparse_disparity = 1 / sparse_radial_depth`` -- this is the space in
    which MiDaS is affine-invariant. Converts the dense map via
    ``depth = 1 / (a*disparity + b)`` and keeps only physically plausible positive
    depths bounded by a multiple of the observed sparse depth range. Pixels whose
    aligned disparity is non-positive, or whose depth exceeds the cap (unreliable
    far/sky extrapolation), are marked invalid (0) -- never fabricated as a
    surface. Units remain unitless_similarity (inherited from the sparse cloud's
    arbitrary global scale).

    Returns ``(aligned_depth [H,W] float32, a, b, n_anchors)`` or ``(None, ...)``.
    """
    ys, xs = np.where(sparse_conf > 0)
    if ys.size < 8:
        return None, None, None, int(ys.size)
    disp_vals = disparity[ys, xs].astype(np.float64)
    tgt_depth = sparse_depth[ys, xs].astype(np.float64)
    finite = (
        np.isfinite(disp_vals)
        & np.isfinite(tgt_depth)
        & (tgt_depth > 0.0)
        & (disp_vals > 0.0)
    )
    disp_vals = disp_vals[finite]
    tgt_depth = tgt_depth[finite]
    if disp_vals.size < 8:
        return None, None, None, int(disp_vals.size)
    tgt_disp = 1.0 / tgt_depth  # align in inverse-depth space

    weights = np.ones_like(disp_vals)
    a, b = 1.0, 0.0
    for _ in range(3):
        W = weights
        A = np.stack([disp_vals, np.ones_like(disp_vals)], axis=1)
        AtW = A.T * W
        try:
            sol = np.linalg.solve(AtW @ A, AtW @ tgt_disp)
        except np.linalg.LinAlgError:
            return None, None, None, int(disp_vals.size)
        a, b = float(sol[0]), float(sol[1])
        resid = np.abs((a * disp_vals + b) - tgt_disp)
        scale = np.median(resid) + 1e-9
        weights = 1.0 / (1.0 + (resid / (1.4826 * scale)) ** 2)

    # Bound depth by a multiple of the observed sparse depth range so a few
    # unreliable far/sky pixels cannot explode the reconstruction extent.
    depth_cap = float(np.percentile(tgt_depth, 99)) * 5.0
    if not np.isfinite(depth_cap) or depth_cap <= 0.0:
        depth_cap = float(np.max(tgt_depth)) * 5.0
    aligned_disp = a * disparity.astype(np.float64) + b
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(aligned_disp > 1e-6, 1.0 / aligned_disp, 0.0)
    depth = np.where(
        np.isfinite(depth) & (depth > 1e-6) & (depth <= depth_cap), depth, 0.0
    )
    return depth.astype(np.float32), a, b, int(disp_vals.size)


def _optical_z_from_radial(depth_radial, intr, np):
    """Convert a radial-range depth map to optical_z. depth_meta declares radial,
    so this is only a guard helper; kept for completeness/reuse."""
    h, w = depth_radial.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    us = (np.arange(w) - cx) / fx
    vs = (np.arange(h) - cy) / fy
    uu, vv = np.meshgrid(us, vs)
    ray_norm = np.sqrt(uu * uu + vv * vv + 1.0)
    return (depth_radial / ray_norm).astype(np.float32)


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #
def _write_json(path, payload):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv=None):
    args = _parse_args(argv)

    # Lazy heavy imports inside main only.
    import json  # noqa: F401 - used by writers via closure
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    try:
        import cv2
    except ImportError as exc:
        out_root = Path(args.out_dir) / args.asset_id
        _write_json(
            out_root / "run_report.json",
            {
                "asset_id": args.asset_id,
                "status": "blocked",
                "blockers": [f"opencv_unavailable:{exc}"],
                "poses_recovered": 0,
            },
        )
        print(f"BLOCKED: OpenCV unavailable: {exc}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir) / args.asset_id
    out_root.mkdir(parents=True, exist_ok=True)
    depth_dir = out_root / "depth"
    conf_dir = out_root / "confidence"

    run_started = datetime.now(timezone.utc).isoformat()
    blockers = []
    warnings = []

    # 1) Enumerate frames + resolve keyframes.
    all_frames = _enumerate_frames(args.frames_dir, args.frame_glob)
    if not all_frames:
        blockers.append("no_frames_found_for_glob")
        _write_run_report(
            out_root, args, run_started, blockers, warnings,
            frames_attempted=0, poses_recovered=0, points=0,
            mean_reproj=None, depth_source="none", extra={},
        )
        print("BLOCKED: no frames found.", file=sys.stderr)
        return 1

    pairs, kf_source, kf_warnings = _load_keyframe_spec(args.keyframes, all_frames)
    warnings.extend(kf_warnings)
    pairs = _evenly_cap(pairs, args.max_frames)
    if len(pairs) < 2:
        blockers.append("insufficient_keyframes_for_sfm")
        _write_run_report(
            out_root, args, run_started, blockers, warnings,
            frames_attempted=len(pairs), poses_recovered=0, points=0,
            mean_reproj=None, depth_source="none",
            extra={"keyframe_source": kf_source},
        )
        print("BLOCKED: fewer than 2 keyframes resolved.", file=sys.stderr)
        return 1

    # 2) Determine image size from the first keyframe (for intrinsics prior).
    first_img = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if first_img is None:
        blockers.append("first_keyframe_unreadable")
        _write_run_report(
            out_root, args, run_started, blockers, warnings,
            frames_attempted=len(pairs), poses_recovered=0, points=0,
            mean_reproj=None, depth_source="none",
            extra={"keyframe_source": kf_source},
        )
        print("BLOCKED: first keyframe unreadable.", file=sys.stderr)
        return 1
    height_px, width_px = first_img.shape[:2]

    intr, intr_source, assumed_prior = _resolve_intrinsics(args.intrinsics, width_px, height_px)
    K = _intrinsics_matrix(intr, np)

    # pose_units / metric_evidence policy: monocular SfM is gauge-free, and an
    # assumed-prior focal cannot anchor metric scale. Either case => non-metric.
    metric_evidence = False
    pose_units = "unitless_similarity"
    units_note = (
        "monocular SfM has free global scale; "
        + ("focal is an assumed prior" if assumed_prior else "intrinsics supplied but no metric anchor")
    )

    # 3) Feature detection + SfM.
    feature_table = _detect_features(pairs, (width_px, height_px), args.max_features, cv2, np)
    sfm = _run_sfm(feature_table, K, cv2, np)
    poses_cw = sfm["poses_cw"]
    points_world = sfm["points_world"]
    point_tracks = sfm["point_tracks"]
    mean_reproj = sfm["mean_reproj_error_px"]

    poses_recovered = len(poses_cw)
    if poses_recovered < 2:
        blockers.append("insufficient_parallax_or_matches_for_sfm")
        # Write manifest/intrinsics for provenance but NO poses, NO depth.
        _write_intrinsics(out_root, intr)
        _write_manifest(
            out_root, args, intr_source, pose_units, metric_evidence,
            assumed_prior, kf_source, pairs, units_note,
            depth_density="none", model_name=None,
        )
        _write_run_report(
            out_root, args, run_started, blockers, warnings,
            frames_attempted=len(pairs), poses_recovered=poses_recovered,
            points=int(points_world.shape[0]), mean_reproj=mean_reproj,
            depth_source="none",
            extra={
                "keyframe_source": kf_source,
                "intrinsics_source": intr_source,
                "assumed_prior_focal": assumed_prior,
                "sfm_diagnostics": sfm["diagnostics"],
            },
        )
        print(
            f"BLOCKED: only {poses_recovered} pose(s) recovered; "
            "insufficient parallax/matches for SfM.",
            file=sys.stderr,
        )
        return 1

    # 4) Optional dense monocular depth model.
    depth_model = depth_transform = depth_device = None
    depth_model_info = None
    if args.use_depth_model:
        depth_model, depth_transform, depth_device, info = _load_depth_model(args.device)
        if depth_model is None:
            # info is the blocker string.
            blockers.append("dense_monocular_depth_model_unavailable")
            warnings.append(str(info))
        else:
            depth_model_info = info

    use_dense = depth_model is not None

    # 5) Per-frame depth + confidence + poses.
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    path_by_fid = {fid: path for (fid, path) in pairs}

    pose_frames = []
    depth_frames_written = []
    dense_alignment = []
    any_dense = False
    for fid in sfm["registered_order"]:
        R_cw, t_cw = poses_cw[fid]
        T_wc = _Rt_to_T_world_camera(R_cw, t_cw, np)
        # Validate homogeneity before recording.
        if not np.allclose(T_wc[3, :], [0.0, 0.0, 0.0, 1.0]):
            warnings.append(f"non_homogeneous_pose_skipped:{fid}")
            continue
        pose_frames.append(
            {
                "frame_id": int(fid),
                "source_frame_path": str(path_by_fid[fid]),
                "T_world_camera": T_wc.tolist(),
            }
        )

        sparse_depth, sparse_conf, valid_count = _sparse_radial_depth(
            fid, R_cw, t_cw, points_world, point_tracks, intr, np
        )
        if valid_count == 0:
            warnings.append(f"no_triangulated_depth_for_frame:{fid}")

        frame_depth = sparse_depth
        frame_conf = sparse_conf
        frame_is_dense = False
        if use_dense and valid_count >= 8:
            disparity_map = _dense_depth_for_frame(
                depth_model, depth_transform, depth_device, path_by_fid[fid], intr, cv2, np
            )
            if disparity_map is not None:
                aligned, a, b, n_anchor = _align_dense_to_sparse(
                    disparity_map, sparse_depth, sparse_conf, np
                )
                if aligned is not None:
                    frame_depth = aligned
                    # Dense confidence: moderate, since scale is borrowed from sparse.
                    frame_conf = np.full_like(aligned, 0.5, dtype=np.float32)
                    frame_is_dense = True
                    any_dense = True
                    dense_alignment.append(
                        {
                            "frame_id": int(fid),
                            "scale_a": a,
                            "shift_b": b,
                            "anchors": n_anchor,
                            "alignment_space": "disparity_inverse_depth",
                        }
                    )
                else:
                    warnings.append(f"dense_alignment_failed_fellback_sparse:{fid}")

        # Only write depth if THIS frame produced real depth: either aligned
        # dense depth, or at least one triangulated sparse sample. An all-zero
        # sparse map is never written -- that would be fabricated emptiness.
        has_real_depth = frame_is_dense or valid_count > 0
        if not has_real_depth:
            continue

        np.save(depth_dir / f"{fid}.npy", frame_depth.astype(np.float32))
        np.save(conf_dir / f"{fid}.npy", frame_conf.astype(np.float32))
        depth_frames_written.append(int(fid))

    if not pose_frames:
        blockers.append("insufficient_parallax_or_matches_for_sfm")

    depth_density = "dense_monocular_model" if any_dense else "sparse_triangulated"
    model_name = depth_model_info["model_name"] if (any_dense and depth_model_info) else None
    depth_source = depth_density if depth_frames_written else "none"

    # 6) Write all artifacts.
    _write_intrinsics(out_root, intr)
    _write_poses(out_root, pose_units, pose_frames)
    _write_depth_meta(out_root, depth_density, model_name)
    _write_manifest(
        out_root, args, intr_source, pose_units, metric_evidence,
        assumed_prior, kf_source, pairs, units_note,
        depth_density=depth_density, model_name=model_name,
    )
    _write_run_report(
        out_root, args, run_started, blockers, warnings,
        frames_attempted=len(pairs), poses_recovered=len(pose_frames),
        points=int(points_world.shape[0]), mean_reproj=mean_reproj,
        depth_source=depth_source,
        extra={
            "keyframe_source": kf_source,
            "intrinsics_source": intr_source,
            "assumed_prior_focal": assumed_prior,
            "metric_evidence": metric_evidence,
            "depth_frames_written": depth_frames_written,
            "dense_alignment": dense_alignment,
            "depth_model_info": depth_model_info,
            "sfm_diagnostics": sfm["diagnostics"],
            "registered_order": [int(f) for f in sfm["registered_order"]],
        },
    )

    print(
        f"OK: asset={args.asset_id} poses={len(pose_frames)} "
        f"points={int(points_world.shape[0])} "
        f"mean_reproj_px={mean_reproj} depth={depth_source} "
        f"metric_evidence={metric_evidence} blockers={blockers}"
    )
    return 0


def _write_intrinsics(out_root, intr):
    _write_json(
        out_root / "intrinsics.json",
        {
            "model": "pinhole",
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "width_px": int(intr["width_px"]),
            "height_px": int(intr["height_px"]),
        },
    )


def _write_poses(out_root, pose_units, pose_frames):
    _write_json(
        out_root / "poses.json",
        {
            "convention": "T_world_camera",
            "translation_units": pose_units,
            "frames": pose_frames,
        },
    )


def _write_depth_meta(out_root, depth_density, model_name):
    _write_json(
        out_root / "depth_meta.json",
        {
            "depth_convention": "radial_range",
            "units": "unitless_similarity",
            "scale_to_meters_or_null": None,
            "density": depth_density,
            "model_name_or_null": model_name,
        },
    )


def _write_manifest(out_root, args, intr_source, pose_units, metric_evidence,
                    assumed_prior, kf_source, pairs, units_note,
                    depth_density, model_name):
    from datetime import datetime, timezone

    method = "incremental_sfm_opencv_sift_essential_pnp"
    if depth_density == "dense_monocular_model":
        method += "+midas_dense_depth_aligned_to_sparse"
    _write_json(
        out_root / "backbone_manifest.json",
        {
            "backbone_name": "atlas3r_opencv_monocular_sfm",
            "method": method,
            "asset_id": args.asset_id,
            "source_frames": [str(path) for (_fid, path) in pairs],
            "intrinsics_source": intr_source,
            "pose_units": pose_units,
            "metric_evidence": metric_evidence,
            "provenance": {
                "tool": "tools/run_geometry_backbone.py",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "frames_dir": str(args.frames_dir),
                "frame_glob": args.frame_glob,
                "keyframe_source": kf_source,
                "assumed_prior_focal": assumed_prior,
                "focal_prior_rule": "0.8*max(W,H)" if assumed_prior else None,
                "depth_density": depth_density,
                "depth_model_name": model_name,
                "units_note": units_note,
            },
            "notes": (
                "Real monocular SfM. Poses are camera-to-world (T_world_camera). "
                "Depth is radial range. Global scale is a free gauge "
                "(unitless_similarity). No measured metric ground truth is "
                "claimed. Frames that failed registration are omitted."
            ),
        },
    )


def _write_run_report(out_root, args, run_started, blockers, warnings,
                      frames_attempted, poses_recovered, points, mean_reproj,
                      depth_source, extra):
    from datetime import datetime, timezone

    payload = {
        "asset_id": args.asset_id,
        "tool": "tools/run_geometry_backbone.py",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if (blockers and poses_recovered < 2) else (
            "ok" if poses_recovered >= 2 else "blocked"
        ),
        "frames_attempted": int(frames_attempted),
        "poses_recovered": int(poses_recovered),
        "points_triangulated": int(points),
        "mean_reprojection_error_px": mean_reproj,
        "depth_source": depth_source,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings))[:64],
        **extra,
    }
    _write_json(out_root / "run_report.json", payload)


if __name__ == "__main__":
    raise SystemExit(main())
