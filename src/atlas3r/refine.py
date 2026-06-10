"""Global geometry refinement (keyframe-graph bundle-lite).

First real multi-view consistency layer for the offline teacher. Given a set of
monocular ``FrameRayPacket`` keyframes (typically the DA3 candidate), this
jointly refines, over a bounded keyframe graph:

- a per-frame SE(3) pose delta (small left-multiplied twist on
  ``T_world_camera``; frame 0 is FIXED to anchor the gauge), and
- a per-frame log-depth AFFINE correction ``d_i'(u) = exp(alpha_i log d_i(u) + beta_i)``,
- optionally a single global scale (only when ``fix_global_scale`` is False, i.e.
  a non-metric reconstruction whose gauge is free).

The objective is a robust (``soft_l1``) least-squares over:
1. cross-frame depth consistency: for sampled pixels in frame ``i``, lift to
   world, transform into a neighbour frame ``j``, reproject through ``j``'s
   camera model, and compare ``log d_j'(nearest sampled pixel)`` against
   ``log ||X_cam_j||`` (the predicted radial range). This is the SAME
   lift/reproject convention as :mod:`atlas3r.visibility`
   (``X_world = R_i d r_i + t_i``; ``X_cam_j = R_j^T (X_world - t_j)``), but the
   neighbour's depth is read at the nearest sampled pixel, not a median proxy.
2. a backbone depth prior pulling ``alpha -> 1`` / ``beta -> 0`` (trust the
   monocular depth unless overlap evidence justifies bending it),
3. temporal smoothness on consecutive ``alpha`` / ``beta`` / pose-delta deltas.

Honesty guards:
- The total robust cost is measured at the identity parameterization
  (``cost_before``) and at the solution (``cost_after``). The refined packets are
  returned ONLY if ``cost_after < cost_before``; otherwise the ORIGINAL packets
  are returned with status ``no_improvement_kept_initialization``. We never adopt
  a worse solution and never claim an improvement we did not measure.
- Nothing is fabricated: a refined depth that is not strictly positive skips that
  row; a packet that fails contract validation is a skipped frame with a reason.

``numpy`` and ``scipy.optimize.least_squares`` are imported lazily INSIDE
``refine_scene`` so importing :mod:`atlas3r` stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import (
    ContractValidationError,
    DepthConvention,
    FrameRayPacket,
)

# Prior / smoothness weights. These are intentionally modest: the cross-frame
# depth-consistency residual is the data term and must dominate, while the
# priors keep the affine correction near identity and the trajectory smooth so a
# weak-overlap frame cannot run away.
PRIOR_ALPHA_WEIGHT = 1.0      # pull alpha_i -> 1
PRIOR_BETA_WEIGHT = 1.0       # pull beta_i -> 0
PRIOR_POSE_WEIGHT = 2.0       # anchor each twist toward 0 (poses are tracked)
SMOOTH_AFFINE_WEIGHT = 0.25   # |alpha_{i+1}-alpha_i|, |beta_{i+1}-beta_i|
SMOOTH_POSE_WEIGHT = 0.25     # |xi_{i+1}-xi_i|
ROBUST_F_SCALE = 0.1          # soft_l1 transition (in log-depth units)

# A refinement is only ADOPTED if it reduces the robust cost by a meaningful
# margin -- both an absolute floor (so numerical dust like 1e-16 on an already
# perfectly consistent scene is never mistaken for an improvement) and a small
# relative fraction. Otherwise we honestly keep the initialization.
MIN_ABS_COST_REDUCTION = 1e-6
MIN_REL_COST_REDUCTION = 1e-4

DEFAULT_MAX_OVERLAP_PAIRS = 48
# Bounds keep the optimization honest and finite: the input poses are already
# tracked, so a pose delta is a *small* correction (a few degrees / centimetres),
# not a re-localization. The affine correction may not collapse or explode the
# depth. Combined with PRIOR_POSE_WEIGHT, this stops a single weak-overlap frame
# from swinging its pose to over-explain the sparse correspondences.
POSE_TWIST_BOUND = 0.15       # ~8.6 deg rotation / 0.15 m translation per coord
ALPHA_BOUND = (0.6, 1.6)
BETA_BOUND = (-1.0, 1.0)
GLOBAL_LOG_SCALE_BOUND = (-1.5, 1.5)


# Markers in packet provenance['method'] identifying bundle-adjustment-grade
# pose backends (the artifact manifest method is carried into every packet).
BA_GRADE_POSE_MARKERS = ("colmap_pose_backend",)


def refine_scene(
    packets: Sequence[FrameRayPacket],
    *,
    max_iterations: int = 60,
    sample_per_frame: int = 400,
    fix_global_scale: bool = True,
    freeze_poses: bool | None = None,
) -> tuple[list[FrameRayPacket], dict]:
    """Refine global geometry of ``packets`` over a keyframe graph.

    Returns ``(refined_packets, refine_report)``. On any blocking condition or
    when no improvement is found, the ORIGINAL packets are returned unchanged
    with an explicit status, never a fabricated substitute.

    ``freeze_poses``: when poses come from a bundle-adjustment-grade backend
    (provenance method contains a ``BA_GRADE_POSE_MARKERS`` entry), this heuristic
    cannot improve them -- measured: it DEGRADED COLMAP poses 3-10x by dragging
    them toward noisy monocular depth (desk 0.0213 -> 0.243 m). Default
    ``None`` auto-detects from packet provenance; only the per-frame log-depth
    affine (and global scale when unfixed) is optimized when frozen.
    """
    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    if freeze_poses is None:
        freeze_poses = any(
            marker in str((getattr(pk, "provenance", {}) or {}).get("method", ""))
            for pk in ordered
            for marker in BA_GRADE_POSE_MARKERS
        )
    base_report: dict[str, Any] = {
        "module": "refine - global geometry refinement",
        "method": (
            "keyframe_graph_logdepth_affine_only_poses_frozen_ba_grade"
            if freeze_poses
            else "keyframe_graph_se3_plus_logdepth_affine_soft_l1_least_squares"
        ),
        "frame_count": len(ordered),
        "poses_frozen_ba_grade": bool(freeze_poses),
    }

    if len(ordered) < 2:
        return list(ordered), {
            **base_report,
            "status": "insufficient_packets_for_refinement",
            "blockers": ("need_at_least_two_packets_to_refine_multi_view",),
            "provenance": _provenance_label(ordered),
        }

    import numpy as np  # type: ignore

    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        return list(ordered), {
            **base_report,
            "status": "missing_optimizer_dependency",
            "error": str(exc),
            "blockers": ("scipy_optimize_least_squares_unavailable",),
            "provenance": _provenance_label(ordered),
        }

    # ------------------------------------------------------------------
    # Prepare per-frame geometry (sampled rays/depths/pixels).
    # ------------------------------------------------------------------
    frames = []
    for packet in ordered:
        prepared = _prepare_frame(packet, sample_per_frame, np)
        if prepared is None:
            continue
        frames.append(prepared)

    if len(frames) < 2:
        return list(ordered), {
            **base_report,
            "status": "insufficient_observed_frames_after_sampling",
            "blockers": ("fewer_than_two_frames_with_usable_samples",),
            "provenance": _provenance_label(ordered),
        }

    n_frames = len(frames)

    # ------------------------------------------------------------------
    # Build keyframe graph: temporal neighbours + bounded overlap pairs.
    # ------------------------------------------------------------------
    edges = _build_keyframe_graph(frames, np)
    if not edges:
        return list(ordered), {
            **base_report,
            "status": "no_overlap_edges_no_constraints",
            "blockers": ("no_cross_frame_overlap_found_to_constrain_refinement",),
            "provenance": _provenance_label(ordered),
        }

    # KD-trees for nearest sampled pixel lookup in each frame (built once).
    pixel_trees = [_build_pixel_tree(frame["uv"], np) for frame in frames]

    # ------------------------------------------------------------------
    # Fixed correspondence set (built ONCE at the identity parameters).
    #
    # scipy.least_squares requires a residual vector of CONSTANT length across
    # evaluations. We therefore enumerate, once, every (source frame i, source
    # sample s, target frame j, target nearest-sample index n) that co-observes
    # at the identity geometry, and store it. Each correspondence ALWAYS emits
    # one residual thereafter; if a later iterate pushes a reprojection
    # out-of-bounds, that residual is held at its last in-bounds value's slot as
    # 0 contribution is wrong (it would reward leaving the frame). Instead we
    # keep evaluating the depth residual using the analytic radial range
    # ||X_cam_j|| (which is always defined for z>0); only when z<=0 (point
    # behind the target camera) do we fall back to a fixed bounded penalty so
    # the vector length is preserved without rewarding degeneracy.
    # ------------------------------------------------------------------
    correspondences = _build_correspondences(frames, edges, pixel_trees, np)
    if not correspondences["src_frame"].size:
        return list(ordered), {
            **base_report,
            "status": "no_cross_frame_correspondences",
            "edges_used": len(edges),
            "blockers": ("no_inbounds_cross_frame_reprojection_at_initialization",),
            "provenance": _provenance_label(ordered),
        }

    # ------------------------------------------------------------------
    # Parameter packing.
    #   x = [twist(6) for frames 1..n-1] (frame 0 fixed = anchor)
    #       + [alpha_i, beta_i for all frames]
    #       + [global_log_scale] (only if not fix_global_scale)
    # ------------------------------------------------------------------
    use_global_scale = not fix_global_scale
    n_pose = 0 if freeze_poses else 6 * (n_frames - 1)
    n_affine = 2 * n_frames
    n_params = n_pose + n_affine + (1 if use_global_scale else 0)

    x0 = np.zeros((n_params,), dtype=np.float64)
    # alpha_i defaults to 1, beta_i to 0.
    for i in range(n_frames):
        x0[n_pose + 2 * i] = 1.0  # alpha
        x0[n_pose + 2 * i + 1] = 0.0  # beta
    # global_log_scale defaults to 0 (scale = 1).

    lower = np.full((n_params,), -np.inf, dtype=np.float64)
    upper = np.full((n_params,), np.inf, dtype=np.float64)
    for k in range(n_pose):
        lower[k] = -POSE_TWIST_BOUND
        upper[k] = POSE_TWIST_BOUND
    for i in range(n_frames):
        lower[n_pose + 2 * i] = ALPHA_BOUND[0]
        upper[n_pose + 2 * i] = ALPHA_BOUND[1]
        lower[n_pose + 2 * i + 1] = BETA_BOUND[0]
        upper[n_pose + 2 * i + 1] = BETA_BOUND[1]
    if use_global_scale:
        lower[-1] = GLOBAL_LOG_SCALE_BOUND[0]
        upper[-1] = GLOBAL_LOG_SCALE_BOUND[1]

    ctx = {
        "frames": frames,
        "edges": edges,
        "correspondences": correspondences,
        "n_frames": n_frames,
        "n_pose": n_pose,
        "freeze_poses": bool(freeze_poses),
        "use_global_scale": use_global_scale,
        "np": np,
    }

    def residual_fn(x: Any) -> Any:
        return _residuals(x, ctx)

    # Honest baseline: robust cost at identity parameters.
    r0 = residual_fn(x0)
    cost_before = float(0.5 * np.sum(_soft_l1(r0, ROBUST_F_SCALE, np)))
    resid_before = _logdepth_residual_summary(x0, ctx)

    result = least_squares(
        residual_fn,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=ROBUST_F_SCALE,
        max_nfev=max(1, int(max_iterations)) * max(8, n_params),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    x_opt = np.asarray(result.x, dtype=np.float64)
    r1 = residual_fn(x_opt)
    cost_after = float(0.5 * np.sum(_soft_l1(r1, ROBUST_F_SCALE, np)))
    resid_after = _logdepth_residual_summary(x_opt, ctx)

    edges_used = len(edges)
    samples_used = int(sum(int(frame["uv"].shape[0]) for frame in frames))
    global_scale = float(np.exp(x_opt[-1])) if use_global_scale else 1.0

    cost_reduction_fraction = (
        (cost_before - cost_after) / cost_before if cost_before > 0.0 else 0.0
    )

    common_report = {
        **base_report,
        "frames_used": n_frames,
        "edges_used": edges_used,
        "samples_used": samples_used,
        "nfev": int(getattr(result, "nfev", 0)),
        "optimizer_status": int(getattr(result, "status", 0)),
        "optimizer_message": str(getattr(result, "message", "")),
        "cost_before": cost_before,
        "cost_after": cost_after,
        "cost_reduction_fraction": float(cost_reduction_fraction),
        "global_scale": global_scale,
        "fix_global_scale": bool(fix_global_scale),
        "residual_summary_before": resid_before,
        "residual_summary_after": resid_after,
        "robust_loss": "soft_l1",
        "f_scale": ROBUST_F_SCALE,
        "limitations": (
            "residuals_are_sparse_sampled_not_dense_per_pixel",
            "keyframe_graph_only_temporal_plus_bounded_overlap_pairs",
            "bounded_iterations_local_optimum_only",
            "pose_delta_is_small_twist_not_full_global_BA",
        ),
        "provenance": _provenance_label(ordered),
    }

    # ------------------------------------------------------------------
    # HONESTY GUARD: only adopt the solution if it reduces the robust cost by a
    # MEANINGFUL margin (absolute floor + relative fraction). A reduction of a
    # few ulps on an already-consistent scene is numerical dust, not an
    # improvement, and must not trigger packet churn or a ':refined' claim.
    # ------------------------------------------------------------------
    abs_reduction = cost_before - cost_after
    meaningful = (
        cost_after < cost_before
        and abs_reduction >= MIN_ABS_COST_REDUCTION
        and cost_reduction_fraction >= MIN_REL_COST_REDUCTION
    )
    if not meaningful:
        return list(ordered), {
            **common_report,
            "status": "no_improvement_kept_initialization",
            "iterations": 0,
            "per_frame_corrections": (),
            "blockers": (),
        }

    # ------------------------------------------------------------------
    # Rebuild refined packets. Frame 0 keeps its pose (delta=0 anchor) but may
    # still receive an affine depth correction.
    # ------------------------------------------------------------------
    refined: list[FrameRayPacket] = []
    per_frame_corrections: list[dict[str, Any]] = []
    skipped_frames: list[dict[str, Any]] = []

    for fi, frame in enumerate(frames):
        packet = frame["packet"]
        alpha = float(x_opt[n_pose + 2 * fi])
        beta = float(x_opt[n_pose + 2 * fi + 1])
        if fi == 0 or freeze_poses:
            delta = np.eye(4, dtype=np.float64)
        else:
            twist = x_opt[6 * (fi - 1):6 * fi]
            delta = _se3_exp(twist, np)

        rebuilt = _rebuild_packet(
            packet=packet,
            delta=delta,
            alpha=alpha,
            beta=beta,
            global_scale=global_scale,
            np=np,
        )
        if rebuilt is None or isinstance(rebuilt, dict):
            reason = rebuilt.get("reason") if isinstance(rebuilt, dict) else "rebuild_failed"
            skipped_frames.append({"frame_id": int(packet.frame_id), "reason": reason})
            # Keep the original packet so the scene is not silently truncated.
            refined.append(packet)
            per_frame_corrections.append(
                {
                    "frame_id": int(packet.frame_id),
                    "applied": False,
                    "reason": reason,
                }
            )
            continue

        refined.append(rebuilt)
        t_delta = float(np.linalg.norm(delta[:3, 3]))
        rot_deg = _rotation_angle_deg(delta[:3, :3], np)
        per_frame_corrections.append(
            {
                "frame_id": int(packet.frame_id),
                "applied": True,
                "alpha": alpha,
                "beta": beta,
                "pose_delta_translation_norm": t_delta,
                "pose_delta_rotation_deg": rot_deg,
            }
        )

    report = {
        **common_report,
        "status": "refined",
        "iterations": int(getattr(result, "nfev", 0)),
        "per_frame_corrections": tuple(per_frame_corrections),
        "skipped_frames": tuple(skipped_frames),
        "blockers": (),
    }
    return refined, report


# ---------------------------------------------------------------------------
# Frame preparation + keyframe graph
# ---------------------------------------------------------------------------


def _prepare_frame(packet: FrameRayPacket, sample_per_frame: int, np: Any) -> dict[str, Any] | None:
    rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
    depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
    conf = np.asarray(packet.confidence, dtype=np.float64).reshape((-1,))
    T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
    n = rays.shape[0]
    if n == 0:
        return None

    valid = np.isfinite(depth) & (depth > 0.0)
    valid &= np.all(np.isfinite(rays), axis=1)
    idx_all = np.argwhere(valid).reshape(-1)
    if idx_all.shape[0] == 0:
        return None

    # Evenly subsample for a bounded optimization.
    if idx_all.shape[0] > sample_per_frame:
        pick = np.unique(
            np.rint(np.linspace(0, idx_all.shape[0] - 1, num=sample_per_frame)).astype(np.int64)
        )
        idx = idx_all[pick]
    else:
        idx = idx_all

    rays_s = rays[idx]
    depth_s = depth[idx]
    conf_s = np.clip(conf[idx], 0.0, 1.0)
    log_depth_s = np.log(np.maximum(depth_s, 1e-9))

    # Recover the sampled pixel UV from the camera model so the neighbour
    # nearest-pixel lookup is exact (rays were built from these pixels).
    camera = packet.camera_model
    fx = float(getattr(camera, "fx"))
    fy = float(getattr(camera, "fy"))
    cx = float(getattr(camera, "cx"))
    cy = float(getattr(camera, "cy"))
    # ray = (x/norm, y/norm, 1/norm) with x=(u-cx)/fx, y=(v-cy)/fy.
    # So x = ray_x/ray_z, y = ray_y/ray_z, and u = fx*x + cx, v = fy*y + cy.
    rz = rays_s[:, 2]
    safe_rz = np.where(np.abs(rz) > 1e-12, rz, 1e-12)
    u = fx * (rays_s[:, 0] / safe_rz) + cx
    v = fy * (rays_s[:, 1] / safe_rz) + cy
    uv = np.stack((u, v), axis=1)

    R = T[:3, :3]
    t = T[:3, 3]
    return {
        "packet": packet,
        "frame_id": int(packet.frame_id),
        "rays": rays_s,
        "depth": depth_s,
        "log_depth": log_depth_s,
        "conf": conf_s,
        "uv": uv,
        "R0": R,
        "t0": t,
        "camera": camera,
    }


def _build_keyframe_graph(frames: Sequence[dict[str, Any]], np: Any) -> list[tuple[int, int]]:
    """Temporal neighbour edges plus a bounded set of overlap pairs.

    Overlap candidacy is decided by whether frame ``i``'s sampled surface
    centroid reprojects in-bounds into frame ``j`` (a real visibility test, not
    a fabricated link).
    """
    n = len(frames)
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    # Temporal: consecutive frames (both directions add information; we keep one
    # ordered edge i->i+1, residuals are evaluated source->target).
    for i in range(n - 1):
        edges.append((i, i + 1))
        seen.add((i, i + 1))

    # Overlap: non-adjacent pairs that actually co-observe.
    centroids_world = []
    for frame in frames:
        X_cam = frame["rays"] * frame["depth"][:, None]
        X_world = X_cam @ frame["R0"].T + frame["t0"][None, :]
        centroids_world.append(np.median(X_world, axis=0))

    candidates: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 2, n):
            if _co_observes(frames[i], frames[j], centroids_world[i], np):
                candidates.append((i, j))

    for pair in _evenly_sample(candidates, DEFAULT_MAX_OVERLAP_PAIRS):
        if pair not in seen:
            edges.append(pair)
            seen.add(pair)

    return edges


def _co_observes(frame_i: dict[str, Any], frame_j: dict[str, Any], centroid_i_world: Any, np: Any) -> bool:
    Rj = frame_j["R0"]
    tj = frame_j["t0"]
    X_cam_j = Rj.T @ (centroid_i_world - tj)
    res = frame_j["camera"].project((float(X_cam_j[0]), float(X_cam_j[1]), float(X_cam_j[2])))
    return bool(res.get("valid"))


def _build_pixel_tree(uv: Any, np: Any) -> Any:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return ("kdtree", cKDTree(uv))
    except Exception:
        return ("brute", uv)


def _nearest_pixel_index(tree: Any, query_uv: Any, np: Any) -> int | None:
    kind, payload = tree
    if kind == "kdtree":
        _, idx = payload.query(query_uv)
        return int(idx)
    # Brute fallback.
    uv = payload
    if uv.shape[0] == 0:
        return None
    d2 = np.sum((uv - query_uv[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


# ---------------------------------------------------------------------------
# Correspondences + residual model
# ---------------------------------------------------------------------------


def _build_correspondences(
    frames: Sequence[dict[str, Any]],
    edges: Sequence[tuple[int, int]],
    pixel_trees: Sequence[Any],
    np: Any,
) -> dict[str, Any]:
    """Enumerate a FIXED set of cross-frame depth correspondences once.

    At the identity geometry, for each edge ``(i, j)`` and each source sample we
    lift -> transform -> reproject into ``j``; if it lands in-bounds we record
    ``(i, source_sample, j, nearest_target_sample, weight)``. This list is then
    constant for the whole optimization so the residual vector length is fixed.
    The target nearest-sample index is pinned here (data association is fixed),
    which keeps the problem well-posed; the residual still moves with alpha/beta/
    pose/scale because depths and the predicted radial range both depend on the
    live parameters.
    """
    src_frame: list[int] = []
    src_idx: list[int] = []
    tgt_frame: list[int] = []
    tgt_idx: list[int] = []
    weight: list[float] = []

    for (i, j) in edges:
        fi = frames[i]
        fj = frames[j]
        Ri, ti = fi["R0"], fi["t0"]
        Rj, tj = fj["R0"], fj["t0"]
        d_i = fi["depth"]
        X_world = (fi["rays"] * d_i[:, None]) @ Ri.T + ti[None, :]
        X_cam_j = (X_world - tj[None, :]) @ Rj
        cam_j = fj["camera"]
        tree_j = pixel_trees[j]
        conf_j = fj["conf"]
        for k in range(X_cam_j.shape[0]):
            xj = X_cam_j[k]
            res = cam_j.project((float(xj[0]), float(xj[1]), float(xj[2])))
            if not res.get("valid"):
                continue
            d_pred = res.get("radial_depth_m")
            if d_pred is None or not (d_pred > 0.0):
                continue
            pixel_uv = res.get("pixel_uv")
            if pixel_uv is None:
                continue
            nn = _nearest_pixel_index(tree_j, np.asarray(pixel_uv, dtype=np.float64), np)
            if nn is None:
                continue
            src_frame.append(i)
            src_idx.append(int(k))
            tgt_frame.append(j)
            tgt_idx.append(int(nn))
            weight.append(float(np.sqrt(max(conf_j[nn], 1e-3))))

    return {
        "src_frame": np.asarray(src_frame, dtype=np.int64),
        "src_idx": np.asarray(src_idx, dtype=np.int64),
        "tgt_frame": np.asarray(tgt_frame, dtype=np.int64),
        "tgt_idx": np.asarray(tgt_idx, dtype=np.int64),
        "weight": np.asarray(weight, dtype=np.float64),
    }


def _unpack(x: Any, ctx: dict[str, Any]) -> tuple[list[Any], Any, Any, float]:
    np = ctx["np"]
    n_frames = ctx["n_frames"]
    n_pose = ctx["n_pose"]
    deltas: list[Any] = [np.eye(4, dtype=np.float64)]
    if ctx.get("freeze_poses"):
        deltas = [np.eye(4, dtype=np.float64) for _ in range(n_frames)]
    else:
        for fi in range(1, n_frames):
            twist = x[6 * (fi - 1):6 * fi]
            deltas.append(_se3_exp(twist, np))
    alphas = np.array([x[n_pose + 2 * fi] for fi in range(n_frames)], dtype=np.float64)
    betas = np.array([x[n_pose + 2 * fi + 1] for fi in range(n_frames)], dtype=np.float64)
    global_scale = float(np.exp(x[-1])) if ctx["use_global_scale"] else 1.0
    return deltas, alphas, betas, global_scale


def _corrected_pose(frame: dict[str, Any], delta: Any, np: Any) -> tuple[Any, Any]:
    T0 = np.eye(4, dtype=np.float64)
    T0[:3, :3] = frame["R0"]
    T0[:3, 3] = frame["t0"]
    T = delta @ T0
    return T[:3, :3], T[:3, 3]


def _cross_frame_log_residuals(x: Any, ctx: dict[str, Any]) -> Any:
    """Per-correspondence signed log-depth residual (fixed length).

    For correspondence (i, s) -> (j, n): with corrected source depth
    ``d_i' = exp(alpha_i log d_i + beta_i + log_scale)``, lift to world, transform
    into frame j, take the predicted radial range ``rho = ||X_cam_j||``, and
    compare against the neighbour's corrected observed depth
    ``log d_j'(n) = alpha_j log d_j(n) + beta_j + log_scale``:

        r = weight * (log d_j'(n) - log rho)

    If the point falls behind the target camera (z<=0) the predicted radial
    range is still ``||X_cam_j||`` but reprojection is meaningless; we keep the
    residual defined via the radial range magnitude so the length is fixed (the
    soft_l1 robustifier bounds its influence and the pose-delta bound prevents
    runaway). This is honest: no correspondence is silently created or dropped.
    """
    np = ctx["np"]
    frames = ctx["frames"]
    corr = ctx["correspondences"]
    n_frames = ctx["n_frames"]

    deltas, alphas, betas, global_scale = _unpack(x, ctx)
    log_scale = float(np.log(global_scale)) if ctx["use_global_scale"] else 0.0
    poses = [_corrected_pose(frames[fi], deltas[fi], np) for fi in range(n_frames)]

    sf = corr["src_frame"]
    si = corr["src_idx"]
    tf = corr["tgt_frame"]
    ti_idx = corr["tgt_idx"]
    w = corr["weight"]
    m = sf.shape[0]
    out = np.empty((m,), dtype=np.float64)

    # Group by edge for vectorized lift/transform.
    # Iterate unique (i, j) pairs present in the correspondence set.
    uniq: dict[tuple[int, int], list[int]] = {}
    for idx in range(m):
        uniq.setdefault((int(sf[idx]), int(tf[idx])), []).append(idx)

    for (i, j), rows in uniq.items():
        rows_arr = np.asarray(rows, dtype=np.int64)
        fi = frames[i]
        fj = frames[j]
        Ri, ti_w = poses[i]
        Rj, tj_w = poses[j]
        s_local = si[rows_arr]
        n_local = ti_idx[rows_arr]

        log_d_i = alphas[i] * fi["log_depth"][s_local] + betas[i] + log_scale
        d_i = np.exp(log_d_i)
        rays_s = fi["rays"][s_local]
        X_world = (rays_s * d_i[:, None]) @ Ri.T + ti_w[None, :]
        X_cam_j = (X_world - tj_w[None, :]) @ Rj
        rho = np.sqrt(np.sum(X_cam_j * X_cam_j, axis=1))
        log_rho = np.log(np.maximum(rho, 1e-9))

        log_d_obs = alphas[j] * fj["log_depth"][n_local] + betas[j] + log_scale
        out[rows_arr] = w[rows_arr] * (log_d_obs - log_rho)

    return out


def _residuals(x: Any, ctx: dict[str, Any]) -> Any:
    np = ctx["np"]
    n_frames = ctx["n_frames"]
    n_pose = ctx["n_pose"]
    # Affine params read directly (pose deltas are handled inside the data term
    # and the pose-magnitude prior below; no need to materialize SE(3) here).
    alphas = x[n_pose:n_pose + 2 * n_frames:2]
    betas = x[n_pose + 1:n_pose + 2 * n_frames:2]

    data_res = _cross_frame_log_residuals(x, ctx)

    # Backbone depth prior: alpha -> 1, beta -> 0.
    prior = np.empty((2 * n_frames,), dtype=np.float64)
    prior[0::2] = PRIOR_ALPHA_WEIGHT * (np.asarray(alphas, dtype=np.float64) - 1.0)
    prior[1::2] = PRIOR_BETA_WEIGHT * (np.asarray(betas, dtype=np.float64) - 0.0)

    # Pose-magnitude prior: each non-anchor twist (frames 1..n-1) is pulled
    # toward zero. The input trajectory is already tracked, so a large pose delta
    # is implausible; this prevents a single frame from over-rotating to explain
    # sparse correspondences. Frame 0 is the fixed gauge anchor (no twist param).
    pose_prior = PRIOR_POSE_WEIGHT * np.asarray(x[:n_pose], dtype=np.float64)

    # Temporal smoothness on affine + pose deltas (consecutive frames).
    # Pose-delta smoothness only exists when poses are free parameters.
    frozen = bool(ctx.get("freeze_poses"))
    smooth: list[float] = []
    for fi_idx in range(n_frames - 1):
        smooth.append(SMOOTH_AFFINE_WEIGHT * (alphas[fi_idx + 1] - alphas[fi_idx]))
        smooth.append(SMOOTH_AFFINE_WEIGHT * (betas[fi_idx + 1] - betas[fi_idx]))
        if frozen:
            continue
        xi_a = x[6 * (fi_idx - 1):6 * fi_idx] if fi_idx >= 1 else np.zeros(6)
        xi_b = x[6 * fi_idx:6 * (fi_idx + 1)]
        diff = np.asarray(xi_b, dtype=np.float64) - np.asarray(xi_a, dtype=np.float64)
        for comp in diff:
            smooth.append(SMOOTH_POSE_WEIGHT * float(comp))
    smooth_arr = np.asarray(smooth, dtype=np.float64) if smooth else np.zeros((0,), dtype=np.float64)

    return np.concatenate((data_res, prior, pose_prior, smooth_arr))


def _logdepth_residual_summary(x: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """Median / p90 of the RAW (unweighted) cross-frame log-depth residual."""
    np = ctx["np"]
    data_res = _cross_frame_log_residuals(x, ctx)
    corr = ctx["correspondences"]
    w = corr["weight"]
    # Undo the confidence weighting to report the raw |log-depth| residual.
    safe_w = np.where(w > 1e-9, w, 1.0)
    raw = np.abs(data_res / safe_w)
    if raw.size == 0:
        return {"count": 0, "median_log_depth_residual": 0.0, "p90_log_depth_residual": 0.0}
    return {
        "count": int(raw.shape[0]),
        "median_log_depth_residual": float(np.median(raw)),
        "p90_log_depth_residual": float(np.percentile(raw, 90.0)),
    }


# ---------------------------------------------------------------------------
# SE(3), robust loss, helpers
# ---------------------------------------------------------------------------


def _se3_exp(twist: Any, np: Any) -> Any:
    """Exponential map of a 6-vector twist (rho, phi) -> 4x4 SE(3).

    Convention: ``twist = [rho(3), phi(3)]`` with ``phi`` the rotation axis-angle
    and ``rho`` the translation part of the Lie algebra.
    """
    twist = np.asarray(twist, dtype=np.float64).reshape(-1)
    rho = twist[:3]
    phi = twist[3:6]
    theta = float(np.linalg.norm(phi))
    T = np.eye(4, dtype=np.float64)
    if theta < 1e-12:
        R = np.eye(3, dtype=np.float64)
        V = np.eye(3, dtype=np.float64)
    else:
        k = phi / theta
        K = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float64,
        )
        s = float(np.sin(theta))
        c = float(np.cos(theta))
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        V = (
            np.eye(3)
            + ((1.0 - c) / theta) * K
            + ((theta - s) / theta) * (K @ K)
        )
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def _rotation_angle_deg(R: Any, np: Any) -> float:
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _soft_l1(residuals: Any, f_scale: float, np: Any) -> Any:
    """Per-residual soft_l1 loss value: ``2*(sqrt(1+z) - 1)`` with ``z=(r/f)^2``.

    Matches scipy's ``soft_l1`` so ``0.5 * sum(rho)`` is the cost scipy reports.
    """
    z = (np.asarray(residuals, dtype=np.float64) / f_scale) ** 2
    return 2.0 * (np.sqrt(1.0 + z) - 1.0) * (f_scale ** 2)


def _evenly_sample(items: Sequence[Any], max_count: int) -> list[Any]:
    if max_count <= 0 or not items:
        return []
    if len(items) <= max_count:
        return list(items)
    step = len(items) / max_count
    return [items[int(i * step)] for i in range(max_count)]


def _provenance_label(packets: Sequence[FrameRayPacket]) -> str:
    """Architecture per-result provenance label derived from source/provenance."""
    if not packets:
        return "unavailable"
    p = packets[0]
    source = str(getattr(p, "source", ""))
    prov = getattr(p, "provenance", {}) or {}
    if source.startswith("measured_reference"):
        return "measured_reference"
    if source.startswith("external_artifact"):
        if prov.get("metric_evidence"):
            return "learned_metric_prior"
        return "monocular_DA3"
    return "unavailable"


# ---------------------------------------------------------------------------
# Refined packet reconstruction (contract-valid)
# ---------------------------------------------------------------------------


def _rebuild_packet(
    *,
    packet: FrameRayPacket,
    delta: Any,
    alpha: float,
    beta: float,
    global_scale: float,
    np: Any,
) -> FrameRayPacket | dict[str, Any] | None:
    rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
    depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
    conf = np.asarray(packet.confidence, dtype=np.float64).reshape((-1,))
    T0 = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))

    # Corrected pose: delta @ T_world_camera (still homogeneous c2w).
    T = delta @ T0
    # Enforce exact homogeneous last row.
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not bool(np.all(np.isfinite(T))):
        return {"reason": "non_finite_pose_after_refine"}

    # Corrected radial depth: exp(alpha log d + beta) * global_scale.
    log_d = alpha * np.log(np.maximum(depth, 1e-12)) + beta
    d_new = np.exp(log_d) * float(global_scale)

    keep = np.isfinite(d_new) & (d_new > 0.0) & np.all(np.isfinite(rays), axis=1)
    if not bool(np.any(keep)):
        return {"reason": "no_strictly_positive_refined_depth_rows"}

    rays_k = rays[keep]
    d_k = d_new[keep]
    conf_k = np.clip(conf[keep], 0.0, 1.0)

    # Re-normalize rays after the float round-trip (1e-5 tolerance).
    norms = np.linalg.norm(rays_k, axis=1)
    if not bool(np.all(norms > 1e-12)):
        return {"reason": "degenerate_ray_after_refine"}
    rays_unit = rays_k / norms[:, None]

    rays_packet = rays_unit.reshape((rays_unit.shape[0], 1, 3))
    depth_packet = d_k.reshape((d_k.shape[0], 1))
    conf_packet = conf_k.reshape((conf_k.shape[0], 1))

    # Carry forward provenance/uncertainty + add a refine record (never empty).
    new_uncertainty = dict(packet.uncertainty)
    new_uncertainty["refine"] = "keyframe_graph_se3_logdepth_affine"

    new_provenance = dict(packet.provenance)
    new_provenance["refine"] = {
        "method": "keyframe_graph_se3_plus_logdepth_affine_soft_l1",
        "alpha": float(alpha),
        "beta": float(beta),
        "global_scale": float(global_scale),
        "pose_delta_translation_norm": float(np.linalg.norm(delta[:3, 3])),
        "pose_delta_rotation_deg": _rotation_angle_deg(delta[:3, :3], np),
        "depth_model": "d_prime = exp(alpha*log d + beta) * global_scale",
        "rays_dropped": int(rays.shape[0] - rays_unit.shape[0]),
        "internal_depth_convention": "radial_range",
    }

    try:
        return FrameRayPacket(
            asset_id=packet.asset_id,
            frame_id=int(packet.frame_id),
            T_world_camera=[[float(v) for v in row] for row in T],
            rays_camera=rays_packet,
            radial_depth_m=depth_packet,
            confidence=conf_packet,
            camera_model=packet.camera_model,  # reuse same model object
            source=f"{packet.source}:refined",
            uncertainty=new_uncertainty,
            provenance=new_provenance,
            source_depth_convention=DepthConvention.RADIAL_RANGE,
            intrinsics=packet.intrinsics,
            rolling_shutter_model=packet.rolling_shutter_model,
            depth_residual_field=packet.depth_residual_field,
            camera_confidence=packet.camera_confidence,
        )
    except ContractValidationError as exc:
        return {"reason": f"frame_ray_packet_contract_rejected: {exc}"}
