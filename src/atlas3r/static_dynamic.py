"""M5 static / dynamic / movable / unknown evidence from geometry.

First REAL static-vs-dynamic evidence layer for the teacher. It consumes the
monocular ``FrameRayPacket`` candidates (sparse sampled rays, unit camera-frame
vectors, radial depth in meters) and produces, per frame, a per-sample
probability simplex over ``{static, dynamic, unknown}`` plus an explicit
``movable_static`` flag where geometry says a surface is real but repeatedly
contradicted as free by other views.

The verdict is GEOMETRIC. For each sampled surface point in frame *i* we lift it
to world space and reproject it into a set of neighbour frames *j* (reusing the
exact visibility lift/reproject convention: ``T_world_camera`` is camera-to-world
so ``X_cam_j = R_j^T (X_world - t_j)``; ``camera_j.project`` yields RADIAL range).
Against each neighbour we compute three independent geometric families:

  (a) cross-view depth INCONSISTENCY -- the reprojected pixel lands in-bounds and
      the neighbour also has a sampled observation nearby, but the predicted
      radial depth and the neighbour's observed radial depth disagree by a large
      robust log-depth residual that is NOT explained by occlusion -> dynamic
      candidate (the point moved / is inconsistent across views);

  (b) FREE-SPACE CONTRADICTION -- the point sits clearly IN FRONT of the
      neighbour's own surface along the same line of sight, i.e. the neighbour's
      ray passed THROUGH this location and terminated farther away, so the
      neighbour carved this location as free. A surface that is repeatedly carved
      free by others is movable / dynamic;

  (c) multi-view SUPPORT count -- how many neighbours reproject in-bounds AND
      agree on depth. Seen consistently by many views -> static. Seen by few /
      occluded everywhere -> unknown (NOT free, NOT static).

Optional SAM2 / Grounded-SAM2 masks (``external/teacher_artifacts/<asset>/masks/``
as a ``MaskTrackSet``) only SPREAD per-pixel geometric verdicts into coherent
object regions and aggregate them into per-track decisions. Masks never declare
dynamic on their own -- geometry decides, masks only group.

Honesty rules honoured here (ARCHITECTURE.md):
- Unknown is NEVER free and NEVER static -- it is genuine "not enough evidence".
- Dynamic is flagged so M7 fusion can EXCLUDE it from static occupancy.
- movable_static (static-but-free-contradicted) is kept DISTINCT from dynamic.
- Missing masks -> status ``mask_evidence_missing`` but geometry STILL runs.
- Single-frame / no-neighbour input -> everything is honestly ``unknown`` (no
  cross-view evidence to call anything static or dynamic), never fabricated.

``numpy`` is imported lazily inside functions; importing this module pulls no
heavy dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    ContractValidationError,
    FrameRayPacket,
    MaskTrack,
    MaskTrackDecision,
    MaskTrackSet,
    StaticDynamicLabel,
    StaticDynamicState,
)

# --- geometric thresholds (meters / log-ratio / pixels) --------------------
# A reprojected sample is "matched" to a neighbour observation only if the
# nearest neighbour sample is within this pixel radius. Sparse packets are not a
# dense grid, so matches outside this radius are treated as "no observation"
# (occlusion / not sampled), never as agreement or disagreement.
DEFAULT_MATCH_PIXEL_RADIUS = 8.0
# |log(d_pred) - log(d_obs)| below this => the two views agree on depth (support).
DEFAULT_DEPTH_AGREE_LOG = 0.10  # ~10.5% relative depth
# log-depth residual above this (and not pure occlusion) => inconsistency vote.
DEFAULT_DEPTH_INCONSISTENT_LOG = 0.25  # ~28% relative depth
# The point is "in front of" the neighbour surface (free-space contradiction)
# when log(d_obs) - log(d_pred) exceeds this: neighbour terminated FARTHER away,
# so its ray passed through this nearer location.
DEFAULT_FREE_CONTRADICTION_LOG = 0.25
# Minimum number of neighbour views that must AGREE on depth for a sample to be
# called confidently static. With independently-sampled sparse packets, two
# frames rarely co-observe the SAME pixel (mean observed-neighbour count ~1 on
# phone_room), so 1 cross-view depth agreement is the defensible floor: it is
# still REAL multi-view confirmation, never zero-evidence. Callers with denser
# co-sampled packets can raise this to demand stronger consensus.
DEFAULT_MIN_SUPPORT_FOR_STATIC = 1
# Fraction of in-bounds neighbours that must vote inconsistent for a dynamic call.
DEFAULT_DYNAMIC_VOTE_FRACTION = 0.5
# Fraction of in-bounds neighbours that must carve a sample free for movable.
DEFAULT_MOVABLE_VOTE_FRACTION = 0.34
# How many neighbour frames to test each frame against (evenly sampled across the
# trajectory so support is not purely temporally local).
DEFAULT_MAX_NEIGHBOURS = 8
# Cap on samples scored per frame to keep the O(samples * neighbours * match)
# cost bounded for dense monocular packets.
DEFAULT_MAX_SAMPLES_PER_FRAME = 1024

# Probability floor so the simplex never collapses to a hard 0/1 vertex; keeps
# the output an honest soft distribution.
_PROB_FLOOR = 0.02


def infer_static_dynamic(
    packets: Sequence[FrameRayPacket],
    *,
    masks_dir: str | Path | None = None,
    neighbor_pairs: Sequence[tuple[int, int]] | None = None,
    match_pixel_radius: float = DEFAULT_MATCH_PIXEL_RADIUS,
    depth_agree_log: float = DEFAULT_DEPTH_AGREE_LOG,
    depth_inconsistent_log: float = DEFAULT_DEPTH_INCONSISTENT_LOG,
    free_contradiction_log: float = DEFAULT_FREE_CONTRADICTION_LOG,
    min_support_for_static: int = DEFAULT_MIN_SUPPORT_FOR_STATIC,
    dynamic_vote_fraction: float = DEFAULT_DYNAMIC_VOTE_FRACTION,
    movable_vote_fraction: float = DEFAULT_MOVABLE_VOTE_FRACTION,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
    max_samples_per_frame: int = DEFAULT_MAX_SAMPLES_PER_FRAME,
) -> tuple[list[StaticDynamicState], dict[str, Any]]:
    """Return ``(static_dynamic_states, report)`` from geometric evidence.

    One :class:`StaticDynamicState` per frame whose ``static/dynamic/unknown``
    probability arrays sum to 1 elementwise (contract simplex). ``movable_static``
    samples are surfaces with real cross-view support but a repeated free-space
    contradiction; they are reported (counts/fractions and per-mask-track
    decisions) and are NOT folded into the dynamic channel.

    With fewer than two packets there is no cross-view evidence: every sample is
    honestly ``unknown`` and the status records why (never fabricated static).
    """
    thresholds = {
        "match_pixel_radius": float(match_pixel_radius),
        "depth_agree_log": float(depth_agree_log),
        "depth_inconsistent_log": float(depth_inconsistent_log),
        "free_contradiction_log": float(free_contradiction_log),
        "min_support_for_static": int(min_support_for_static),
        "dynamic_vote_fraction": float(dynamic_vote_fraction),
        "movable_vote_fraction": float(movable_vote_fraction),
        "max_neighbours": int(max_neighbours),
        "max_samples_per_frame": int(max_samples_per_frame),
        "probability_floor": _PROB_FLOOR,
    }

    base_report: dict[str, Any] = {
        "module": "M5 - Static/Dynamic/Movable/Unknown Evidence",
        "packet_count": len(packets),
        "thresholds": thresholds,
        "evidence_families_available": (
            "cross_view_depth_inconsistency",
            "free_space_contradiction",
            "multi_view_support_count",
        ),
    }

    if len(packets) < 1:
        return [], {
            **base_report,
            "status": "blocked_no_packets",
            "evidence_families_used": (),
            "frames": (),
            "blockers": ("no_packets_for_static_dynamic",),
        }

    import numpy as np  # type: ignore

    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    asset_id = str(ordered[0].asset_id)

    # --- optional mask evidence ------------------------------------------------
    mask_set, mask_status, mask_blockers = _load_mask_track_set(
        asset_id, masks_dir, ordered
    )
    mask_present = mask_set is not None

    # --- provenance derived from packet source (NOT a fabricated label) --------
    provenance = _derive_provenance(ordered)

    # --- prepare per-frame world geometry (reuse fusion/visibility convention) -
    prepared = [_prepare_frame(p, np, max_samples_per_frame) for p in ordered]
    frame_index = {pf["frame_id"]: i for i, pf in enumerate(prepared)}

    # Neighbour selection: explicit pairs if given, else evenly-spaced others.
    neighbour_map = _build_neighbour_map(
        prepared, neighbor_pairs, frame_index, max_neighbours
    )

    if len(ordered) < 2:
        # No cross-view evidence is possible. Be honest: all unknown.
        states, frame_reports = _all_unknown_states(prepared, np)
        states_or_empty, blockers = _finalize_states(states)
        return states_or_empty, {
            **base_report,
            "status": mask_status,
            "evidence_families_used": (),
            "mask_evidence": _mask_summary(mask_set),
            "provenance": provenance,
            "frames": tuple(frame_reports),
            "totals": _aggregate_totals(frame_reports),
            "blockers": tuple(blockers) + ("single_frame_no_cross_view_evidence",)
            + tuple(mask_blockers),
        }

    families_used: set[str] = set()
    states: list[StaticDynamicState] = []
    frame_reports: list[dict[str, Any]] = []

    for pf in prepared:
        neighbours = [prepared[j] for j in neighbour_map[pf["frame_id"]]]
        verdict = _score_frame_against_neighbours(
            pf, neighbours, np, thresholds, families_used
        )
        decisions = _mask_track_decisions(
            pf, verdict, mask_set, np, thresholds
        )
        state, fr = _build_state(pf, verdict, decisions, mask_present, np, thresholds)
        if state is not None:
            states.append(state)
        frame_reports.append(fr)

    states_or_empty, blockers = _finalize_states(states)
    totals = _aggregate_totals(frame_reports)

    report = {
        **base_report,
        "status": mask_status,
        "evidence_families_used": tuple(sorted(families_used)),
        "mask_evidence": _mask_summary(mask_set),
        "provenance": provenance,
        "frames": tuple(frame_reports),
        "totals": totals,
        "dynamic_excluded_from_static_fusion": True,
        "movable_distinct_from_dynamic": True,
        "blockers": tuple(blockers) + tuple(mask_blockers),
    }
    return states_or_empty, report


# ---------------------------------------------------------------------------
# per-frame geometry preparation
# ---------------------------------------------------------------------------


def _prepare_frame(packet: FrameRayPacket, np: Any, max_samples: int) -> dict[str, Any]:
    """Lift a packet's sampled rays to world space + recover sampled pixel UVs.

    Reuses the canonical lift ``X_world = R (d r) + t`` (``T_world_camera`` is
    camera-to-world). Pixel UVs are recovered from the unit rays through the
    SAME camera model used by the packet, so neighbour matching is done in the
    neighbour's own pixel frame -- no fabricated grid.
    """
    rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
    depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
    conf = np.asarray(packet.confidence, dtype=np.float64).reshape((-1,))
    T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
    R = T[:3, :3]
    t = T[:3, 3]

    n = rays.shape[0]
    if n > max_samples and max_samples > 0:
        idx = np.unique(np.rint(np.linspace(0, n - 1, num=max_samples)).astype(np.int64))
    else:
        idx = np.arange(n)
    rays = rays[idx]
    depth = depth[idx]
    conf = conf[idx]

    X_cam = rays * depth[:, None]
    X_world = X_cam @ R.T + t[None, :]

    camera = packet.camera_model
    # Recover this frame's own sampled pixel UVs from its unit rays (z>0 by
    # construction of the pinhole rays). Used as the neighbour observation grid.
    uv = _rays_to_pixels(rays, camera, np)

    return {
        "frame_id": int(packet.frame_id),
        "rays": rays,
        "depth": depth,
        "confidence": conf,
        "R": R,
        "t": t,
        "camera": camera,
        "X_world": X_world,
        "uv": uv,  # [N,2] sampled pixel coords in this frame
        "sample_count": int(rays.shape[0]),
    }


def _rays_to_pixels(rays: Any, camera: Any, np: Any) -> Any:
    """Project unit camera-frame rays to pixel coordinates via the camera model.

    ``u = fx * x/z + cx``; ``v = fy * y/z + cy`` (z>0 for valid pinhole rays).
    Returns ``[N,2]``; rays with non-positive z are pushed off-grid so they never
    match a neighbour sample.
    """
    z = rays[:, 2]
    safe_z = np.where(z > 1e-9, z, 1.0)
    u = float(camera.fx) * rays[:, 0] / safe_z + float(camera.cx)
    v = float(camera.fy) * rays[:, 1] / safe_z + float(camera.cy)
    bad = z <= 1e-9
    u = np.where(bad, -1e9, u)
    v = np.where(bad, -1e9, v)
    return np.stack((u, v), axis=1)


def _build_neighbour_map(
    prepared: Sequence[dict[str, Any]],
    neighbor_pairs: Sequence[tuple[int, int]] | None,
    frame_index: Mapping[int, int],
    max_neighbours: int,
) -> dict[int, list[int]]:
    """Map each frame_id -> list of neighbour prepared-list indices.

    Honours an explicit ``neighbor_pairs`` whitelist (directional source->target)
    when given; otherwise picks ``max_neighbours`` other frames evenly across the
    ordered trajectory (mixing temporally near and far views).
    """
    nmap: dict[int, list[int]] = {pf["frame_id"]: [] for pf in prepared}
    n = len(prepared)
    if neighbor_pairs:
        for src, tgt in neighbor_pairs:
            si = frame_index.get(int(src))
            ti = frame_index.get(int(tgt))
            if si is None or ti is None or si == ti:
                continue
            if ti not in nmap[prepared[si]["frame_id"]]:
                nmap[prepared[si]["frame_id"]].append(ti)
        return nmap

    for i in range(n):
        others = [j for j in range(n) if j != i]
        if len(others) <= max_neighbours:
            chosen = others
        else:
            step = len(others) / max_neighbours
            chosen = [others[int(k * step)] for k in range(max_neighbours)]
            chosen = sorted(set(chosen))
        nmap[prepared[i]["frame_id"]] = chosen
    return nmap


# ---------------------------------------------------------------------------
# core geometric scoring
# ---------------------------------------------------------------------------


def _score_frame_against_neighbours(
    pf: dict[str, Any],
    neighbours: Sequence[dict[str, Any]],
    np: Any,
    thresholds: Mapping[str, Any],
    families_used: set[str],
) -> dict[str, Any]:
    """Score every sample of ``pf`` against ``neighbours``.

    Returns per-sample vote counters:
      - ``support``: neighbours that reproject in-bounds AND agree on depth.
      - ``inconsistent``: neighbours in-bounds + observed nearby but log-depth
        residual exceeds the inconsistency threshold (and not pure occlusion).
      - ``free_contradiction``: neighbours whose own surface is clearly FARTHER
        along this line of sight (point is in front -> carved free).
      - ``inbounds_observed``: neighbours that both reproject in-bounds AND had a
        nearby observation (the valid denominator for vote fractions).
    """
    n_samples = pf["sample_count"]
    support = np.zeros(n_samples, dtype=np.int32)
    inconsistent = np.zeros(n_samples, dtype=np.int32)
    free_contra = np.zeros(n_samples, dtype=np.int32)
    inbounds_obs = np.zeros(n_samples, dtype=np.int32)
    inbounds_any = np.zeros(n_samples, dtype=np.int32)

    match_radius = float(thresholds["match_pixel_radius"])
    agree_log = float(thresholds["depth_agree_log"])
    inconsistent_log = float(thresholds["depth_inconsistent_log"])
    free_log = float(thresholds["free_contradiction_log"])

    X_world = pf["X_world"]

    for nb in neighbours:
        # Reproject pf's world points into the neighbour camera (camera-to-world
        # => X_cam_j = R_j^T (X_world - t_j)). Row-vector form matches mapping &
        # visibility: (X_world - t) @ R.
        X_cam_j = (X_world - nb["t"][None, :]) @ nb["R"]
        z_j = X_cam_j[:, 2]
        in_front = z_j > 1e-9

        cam = nb["camera"]
        safe_z = np.where(in_front, z_j, 1.0)
        u = float(cam.fx) * X_cam_j[:, 0] / safe_z + float(cam.cx)
        v = float(cam.fy) * X_cam_j[:, 1] / safe_z + float(cam.cy)
        in_bounds = (
            in_front
            & (u >= 0.0) & (u < float(cam.width_px))
            & (v >= 0.0) & (v < float(cam.height_px))
        )
        # Predicted radial range from neighbour camera to the point.
        d_pred = np.sqrt(np.sum(X_cam_j * X_cam_j, axis=1))

        inbounds_any += in_bounds.astype(np.int32)

        # Match each reprojected pixel to the neighbour's nearest sampled pixel.
        nb_uv = nb["uv"]
        nb_depth = nb["depth"]
        d_obs, matched = _nearest_observed_depth(
            u, v, in_bounds, nb_uv, nb_depth, match_radius, np
        )

        valid = matched & in_bounds & (d_obs > 0.0) & (d_pred > 0.0)
        inbounds_obs += valid.astype(np.int32)

        # Robust log-depth comparison. Positive resid = predicted farther than
        # observed (point behind neighbour surface => occlusion, NOT a vote).
        log_diff = np.where(valid, np.log(np.maximum(d_pred, 1e-9)) - np.log(np.maximum(d_obs, 1e-9)), 0.0)
        abs_diff = np.abs(log_diff)

        agree = valid & (abs_diff <= agree_log)
        support += agree.astype(np.int32)

        # Free-space contradiction: observed FARTHER than predicted by a margin
        # => the neighbour's ray passed THROUGH this nearer location (carved it
        # free). log_diff strongly negative.
        free_hit = valid & ((-log_diff) >= free_log)
        free_contra += free_hit.astype(np.int32)

        # Depth inconsistency that is NOT explained as the point being occluded
        # behind the neighbour surface. Occlusion = point clearly BEHIND (d_pred
        # >> d_obs, log_diff positive). We only count disagreement where the
        # point is NOT simply behind: either in front (free-ish) or within the
        # near band. This keeps honest occlusion from being mislabeled dynamic.
        not_pure_occlusion = log_diff <= (0.5 * inconsistent_log)
        inconsistent_hit = valid & (abs_diff >= inconsistent_log) & not_pure_occlusion
        inconsistent += inconsistent_hit.astype(np.int32)

    if np.any(support > 0):
        families_used.add("multi_view_support_count")
    if np.any(inconsistent > 0):
        families_used.add("cross_view_depth_inconsistency")
    if np.any(free_contra > 0):
        families_used.add("free_space_contradiction")

    return {
        "support": support,
        "inconsistent": inconsistent,
        "free_contradiction": free_contra,
        "inbounds_observed": inbounds_obs,
        "inbounds_any": inbounds_any,
        "neighbour_count": len(neighbours),
    }


def _nearest_observed_depth(
    u: Any,
    v: Any,
    in_bounds: Any,
    nb_uv: Any,
    nb_depth: Any,
    radius: float,
    np: Any,
) -> tuple[Any, Any]:
    """For each reprojected (u,v), find the neighbour's nearest sampled pixel.

    Returns ``(d_obs, matched)`` where ``matched`` is True only when the nearest
    neighbour sample lies within ``radius`` pixels. Sparse packets are not a
    dense grid: a reprojected pixel with no nearby sample is genuinely
    unobserved (occlusion / not sampled), not a disagreement.

    Bucketed nearest-neighbour: neighbour samples are hashed into a pixel grid of
    cell size ``radius`` so only the 3x3 surrounding cells are scanned per query.
    This keeps the match O(samples) instead of O(samples^2).
    """
    n = u.shape[0]
    d_obs = np.zeros(n, dtype=np.float64)
    matched = np.zeros(n, dtype=bool)
    m = nb_uv.shape[0]
    if m == 0:
        return d_obs, matched

    cell = max(radius, 1.0)
    nb_cx = np.floor(nb_uv[:, 0] / cell).astype(np.int64)
    nb_cy = np.floor(nb_uv[:, 1] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for k in range(m):
        buckets.setdefault((int(nb_cx[k]), int(nb_cy[k])), []).append(k)

    r2 = radius * radius
    q_cx = np.floor(u / cell).astype(np.int64)
    q_cy = np.floor(v / cell).astype(np.int64)
    for i in range(n):
        if not bool(in_bounds[i]):
            continue
        cx = int(q_cx[i])
        cy = int(q_cy[i])
        best = -1
        best_d2 = r2
        ui = float(u[i])
        vi = float(v[i])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand = buckets.get((cx + dx, cy + dy))
                if not cand:
                    continue
                for k in cand:
                    ddx = float(nb_uv[k, 0]) - ui
                    ddy = float(nb_uv[k, 1]) - vi
                    dd = ddx * ddx + ddy * ddy
                    if dd <= best_d2:
                        best_d2 = dd
                        best = k
        if best >= 0:
            matched[i] = True
            d_obs[i] = float(nb_depth[best])
    return d_obs, matched


# ---------------------------------------------------------------------------
# verdict -> probability simplex
# ---------------------------------------------------------------------------


def _build_state(
    pf: dict[str, Any],
    verdict: Mapping[str, Any],
    decisions: Sequence[MaskTrackDecision],
    mask_present: bool,
    np: Any,
    thresholds: Mapping[str, Any],
) -> tuple[StaticDynamicState | None, dict[str, Any]]:
    """Turn per-sample vote counters into a {static,dynamic,unknown} simplex.

    Decision logic per sample (geometry only):
      - dynamic: enough in-bounds-observed neighbours voted INCONSISTENT.
      - movable_static: surface has real support but a repeated FREE
        contradiction (kept distinct from dynamic; reported, not in dynamic chan).
      - static: support count >= min_support and not dynamic.
      - unknown: too little observation (never seen, or seen by too few).
    The three probability channels are a soft distribution summing to 1.
    """
    support = verdict["support"].astype(np.float64)
    inconsistent = verdict["inconsistent"].astype(np.float64)
    free_contra = verdict["free_contradiction"].astype(np.float64)
    obs = verdict["inbounds_observed"].astype(np.float64)
    n = support.shape[0]

    min_support = int(thresholds["min_support_for_static"])
    dyn_frac = float(thresholds["dynamic_vote_fraction"])
    mov_frac = float(thresholds["movable_vote_fraction"])

    obs_safe = np.maximum(obs, 1.0)
    dyn_ratio = np.where(obs > 0, inconsistent / obs_safe, 0.0)
    free_ratio = np.where(obs > 0, free_contra / obs_safe, 0.0)

    is_dynamic = (obs > 0) & (dyn_ratio >= dyn_frac) & (inconsistent >= 1)
    has_support = support >= float(min_support)
    is_movable = (~is_dynamic) & has_support & (free_ratio >= mov_frac) & (free_contra >= 1)
    is_static = (~is_dynamic) & (~is_movable) & has_support
    # everything else is unknown (seen by too few / never observed)
    is_unknown = (~is_dynamic) & (~is_static) & (~is_movable)

    # Build soft probabilities. We translate the discrete verdict into a
    # confidence-weighted simplex, keeping a probability floor so no channel is a
    # hard 0/1 vertex (honest soft evidence). movable_static is reported on the
    # side and lifts the STATIC channel (a movable surface is still static
    # geometry, just flagged movable), it does NOT add to dynamic.
    static_p = np.full(n, _PROB_FLOOR, dtype=np.float64)
    dynamic_p = np.full(n, _PROB_FLOOR, dtype=np.float64)
    unknown_p = np.full(n, _PROB_FLOOR, dtype=np.float64)

    # Dynamic confidence scales with the fraction of neighbours voting it dynamic.
    dyn_conf = np.clip(dyn_ratio, 0.0, 1.0)
    # Static confidence scales with normalized support (capped at neighbour cap).
    n_neighbours = max(int(verdict["neighbour_count"]), 1)
    support_norm = np.clip(support / float(n_neighbours), 0.0, 1.0)

    # Assign dominant mass per verdict, leaving floors on the other two.
    dynamic_p = np.where(is_dynamic, _PROB_FLOOR + (1.0 - 3.0 * _PROB_FLOOR) * np.maximum(dyn_conf, 0.5), dynamic_p)
    static_p = np.where(
        is_static | is_movable,
        _PROB_FLOOR + (1.0 - 3.0 * _PROB_FLOOR) * np.maximum(support_norm, 0.5),
        static_p,
    )
    unknown_p = np.where(is_unknown, _PROB_FLOOR + (1.0 - 3.0 * _PROB_FLOOR) * 1.0, unknown_p)

    # Normalize to a strict simplex (handles the residual floor mass on the two
    # non-dominant channels and guarantees sum==1 within contract tolerance).
    stacked = np.stack((static_p, dynamic_p, unknown_p), axis=1)
    stacked = np.clip(stacked, _PROB_FLOOR, None)
    row_sum = np.sum(stacked, axis=1, keepdims=True)
    stacked = stacked / row_sum
    static_prob = stacked[:, 0]
    dynamic_prob = stacked[:, 1]
    unknown_prob = stacked[:, 2]

    # Per-sample movable_static probability: the static-channel mass of samples
    # geometry flagged movable (a real surface, repeatedly carved free by other
    # views). It is kept DISTINCT from dynamic and is 0 where the sample is not
    # movable. The contract's StaticDynamicState simplex only has
    # static/dynamic/unknown, so this rides in ``residual_summary`` (the
    # export/fusion layers look for exactly this key) -- never folded into dynamic.
    movable_probability = np.where(is_movable, static_prob, 0.0)

    counts = {
        "static": int(np.count_nonzero(is_static)),
        "dynamic": int(np.count_nonzero(is_dynamic)),
        "movable_static": int(np.count_nonzero(is_movable)),
        "unknown": int(np.count_nonzero(is_unknown)),
    }
    total = max(n, 1)
    fractions = {k: v / total for k, v in counts.items()}

    residual_summary = {
        "evidence": "cross_view_geometry",
        "sample_count": int(n),
        "static_count": counts["static"],
        "dynamic_count": counts["dynamic"],
        "movable_static_count": counts["movable_static"],
        "unknown_count": counts["unknown"],
        "static_fraction": fractions["static"],
        "dynamic_fraction": fractions["dynamic"],
        "movable_static_fraction": fractions["movable_static"],
        "unknown_fraction": fractions["unknown"],
        "mean_support": float(np.mean(support)) if n else 0.0,
        "mean_inbounds_observed": float(np.mean(obs)) if n else 0.0,
        "mean_inconsistent_votes": float(np.mean(inconsistent)) if n else 0.0,
        "mean_free_contradiction_votes": float(np.mean(free_contra)) if n else 0.0,
        "movable_static_distinct_from_dynamic": True,
        "dynamic_excluded_from_static_fusion": True,
        "mask_grouping_applied": bool(mask_present),
        # Per-sample movable_static probability array (aligned 1:1 with the
        # static/dynamic/unknown simplex arrays). Painted into the movable channel
        # of the 3D field; 0 where not movable. Distinct from dynamic.
        "movable_probability": movable_probability,
    }

    # movable_static is a per-sample boolean carried in the report (so M7 can
    # exclude dynamic and treat movable distinctly). It is NOT a probability
    # channel on StaticDynamicState (the contract has only static/dynamic/unknown).
    frame_report = {
        "frame_id": pf["frame_id"],
        "sample_count": int(n),
        "counts": counts,
        "fractions": fractions,
        "mask_track_decision_count": len(decisions),
        "neighbour_count": int(verdict["neighbour_count"]),
    }

    try:
        state = StaticDynamicState(
            frame_id=pf["frame_id"],
            static_probability=static_prob,
            dynamic_probability=dynamic_prob,
            unknown_probability=unknown_prob,
            mask_track_decisions=tuple(decisions),
            residual_summary=residual_summary,
        )
    except ContractValidationError as exc:
        frame_report["status"] = "state_contract_rejected"
        frame_report["error"] = str(exc)
        return None, frame_report

    frame_report["status"] = "scored"
    return state, frame_report


# ---------------------------------------------------------------------------
# mask evidence (optional grouping; never decides dynamic by itself)
# ---------------------------------------------------------------------------


def _load_mask_track_set(
    asset_id: str,
    masks_dir: str | Path | None,
    prepared_or_packets: Sequence[Any],
) -> tuple[MaskTrackSet | None, str, tuple[str, ...]]:
    """Load a SAM2/Grounded-SAM2 MaskTrackSet if present.

    Looks for ``<masks_dir>/tracks.json`` (a list of mask tracks) under the
    asset's mask directory. Absent or unreadable -> ``mask_evidence_missing``
    with the geometry path still active. Returns ``(MaskTrackSet|None, status,
    blockers)``.
    """
    if masks_dir is None:
        candidate = (
            Path("external") / "teacher_artifacts" / asset_id / "masks"
        )
    else:
        candidate = Path(masks_dir)

    tracks_path = candidate / "tracks.json"
    if not tracks_path.is_file():
        return None, "mask_evidence_missing", (
            f"no_mask_track_set_at:{tracks_path.as_posix()}",
        )

    try:
        raw = json.loads(tracks_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, "mask_evidence_missing", (
            f"corrupt_mask_track_set:{exc}",
        )

    track_records = raw.get("tracks", raw) if isinstance(raw, Mapping) else raw
    tracks: list[MaskTrack] = []
    try:
        for rec in track_records:
            tracks.append(
                MaskTrack(
                    track_id=str(rec["track_id"]),
                    frame_ids=tuple(int(f) for f in rec["frame_ids"]),
                    mask_rle_or_bitmap=rec.get("mask_rle_or_bitmap", rec.get("mask", {"format": "rle", "ref": str(rec.get("track_id")) })),
                    mask_confidence=float(rec.get("mask_confidence", 0.5)),
                    prompt_or_source=str(rec.get("prompt_or_source", "sam2")),
                    semantic_label_optional=(
                        str(rec["semantic_label_optional"])
                        if rec.get("semantic_label_optional")
                        else None
                    ),
                )
            )
        mask_set = MaskTrackSet(tracks=tuple(tracks))
    except (KeyError, TypeError, ContractValidationError) as exc:
        return None, "mask_evidence_missing", (
            f"invalid_mask_track_set:{exc}",
        )

    return mask_set, "mask_evidence_present", ()


def _mask_track_decisions(
    pf: dict[str, Any],
    verdict: Mapping[str, Any],
    mask_set: MaskTrackSet | None,
    np: Any,
    thresholds: Mapping[str, Any],
) -> list[MaskTrackDecision]:
    """Spread per-sample GEOMETRIC verdicts into per-track decisions.

    For each mask track that includes this frame, the per-sample geometric votes
    on this frame are aggregated into a single label. The mask only GROUPS the
    samples into one object; the label is decided by GEOMETRY (majority of the
    frame's dynamic/static/unknown votes). Without per-pixel mask rasters we
    apply the frame-level geometric majority to every track touching the frame --
    honest grouping, never a mask-declared dynamic.
    """
    if mask_set is None:
        return []

    support = verdict["support"]
    inconsistent = verdict["inconsistent"]
    free_contra = verdict["free_contradiction"]
    obs = verdict["inbounds_observed"]
    n = support.shape[0]
    if n == 0:
        return []

    min_support = int(thresholds["min_support_for_static"])
    dyn_frac = float(thresholds["dynamic_vote_fraction"])
    mov_frac = float(thresholds["movable_vote_fraction"])

    obs_safe = np.maximum(obs.astype(np.float64), 1.0)
    dyn_ratio = np.where(obs > 0, inconsistent / obs_safe, 0.0)
    free_ratio = np.where(obs > 0, free_contra / obs_safe, 0.0)
    is_dynamic = (obs > 0) & (dyn_ratio >= dyn_frac) & (inconsistent >= 1)
    has_support = support >= float(min_support)
    is_movable = (~is_dynamic) & has_support & (free_ratio >= mov_frac) & (free_contra >= 1)
    is_static = (~is_dynamic) & (~is_movable) & has_support

    n_dyn = int(np.count_nonzero(is_dynamic))
    n_static = int(np.count_nonzero(is_static | is_movable))
    n_unknown = n - n_dyn - n_static

    # Frame-level geometric majority -> label + confidence.
    if n_dyn >= max(n_static, n_unknown) and n_dyn > 0:
        label = StaticDynamicLabel.DYNAMIC
        conf = n_dyn / n
        reason = "geometry_cross_view_inconsistency_majority"
    elif n_static >= max(n_dyn, n_unknown) and n_static > 0:
        label = StaticDynamicLabel.STATIC
        conf = n_static / n
        reason = "geometry_multi_view_support_majority"
    else:
        label = StaticDynamicLabel.UNKNOWN
        conf = max(n_unknown / n, _PROB_FLOOR)
        reason = "geometry_insufficient_cross_view_observation"

    conf = float(min(max(conf, 0.0), 1.0))
    frame_id = pf["frame_id"]
    decisions: list[MaskTrackDecision] = []
    for track in mask_set.tracks:
        if frame_id not in set(int(f) for f in track.frame_ids):
            continue
        try:
            decisions.append(
                MaskTrackDecision(
                    track_id=str(track.track_id),
                    label=label,
                    confidence=conf,
                    reason=reason,
                )
            )
        except ContractValidationError:
            continue
    return decisions


def _mask_summary(mask_set: MaskTrackSet | None) -> dict[str, Any]:
    if mask_set is None:
        return {"present": False, "track_count": 0}
    return {
        "present": True,
        "track_count": len(mask_set.tracks),
        "grouping_role": "spread_geometric_verdicts_into_objects",
        "decides_dynamic": False,
    }


# ---------------------------------------------------------------------------
# single-frame / fallbacks / aggregation
# ---------------------------------------------------------------------------


def _all_unknown_states(
    prepared: Sequence[dict[str, Any]], np: Any
) -> tuple[list[StaticDynamicState], list[dict[str, Any]]]:
    """No cross-view evidence: every sample is honestly unknown."""
    states: list[StaticDynamicState] = []
    frame_reports: list[dict[str, Any]] = []
    for pf in prepared:
        n = pf["sample_count"]
        static_prob = np.full(n, _PROB_FLOOR, dtype=np.float64)
        dynamic_prob = np.full(n, _PROB_FLOOR, dtype=np.float64)
        unknown_prob = np.full(n, 1.0 - 2.0 * _PROB_FLOOR, dtype=np.float64)
        residual_summary = {
            "evidence": "no_cross_view_evidence",
            "sample_count": int(n),
            "static_count": 0,
            "dynamic_count": 0,
            "movable_static_count": 0,
            "unknown_count": int(n),
            "reason": "single_frame_or_no_neighbours",
        }
        fr = {
            "frame_id": pf["frame_id"],
            "sample_count": int(n),
            "counts": {"static": 0, "dynamic": 0, "movable_static": 0, "unknown": int(n)},
            "fractions": {"static": 0.0, "dynamic": 0.0, "movable_static": 0.0, "unknown": 1.0 if n else 0.0},
            "mask_track_decision_count": 0,
            "neighbour_count": 0,
            "status": "unknown_no_cross_view",
        }
        try:
            states.append(
                StaticDynamicState(
                    frame_id=pf["frame_id"],
                    static_probability=static_prob,
                    dynamic_probability=dynamic_prob,
                    unknown_probability=unknown_prob,
                    mask_track_decisions=(),
                    residual_summary=residual_summary,
                )
            )
        except ContractValidationError as exc:
            fr["status"] = "state_contract_rejected"
            fr["error"] = str(exc)
        frame_reports.append(fr)
    return states, frame_reports


def _finalize_states(
    states: Sequence[StaticDynamicState],
) -> tuple[list[StaticDynamicState], list[str]]:
    blockers: list[str] = []
    if not states:
        blockers.append("no_static_dynamic_states_built")
    return list(states), blockers


def _aggregate_totals(frame_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_samples = 0
    agg = {"static": 0, "dynamic": 0, "movable_static": 0, "unknown": 0}
    for fr in frame_reports:
        counts = fr.get("counts", {})
        total_samples += int(fr.get("sample_count", 0))
        for key in agg:
            agg[key] += int(counts.get(key, 0))
    denom = max(total_samples, 1)
    return {
        "total_samples": total_samples,
        "counts": agg,
        "fractions": {k: agg[k] / denom for k in agg},
        "frames_scored": len(frame_reports),
    }


def _derive_provenance(packets: Sequence[FrameRayPacket]) -> dict[str, Any]:
    """Per-result provenance label derived from packet source + provenance.

    Mirrors the architecture's provenance taxonomy:
    measured_reference | monocular_DA3 | learned_metric_prior | manual_anchor |
    unavailable. Derived from ``packet.source`` and provenance flags -- never a
    fabricated claim.
    """
    sources = {str(p.source) for p in packets}
    first = packets[0]
    prov = dict(first.provenance) if isinstance(first.provenance, Mapping) else {}
    metric_evidence = bool(prov.get("metric_evidence", False))

    if any(s.startswith("measured_reference") for s in sources):
        label = "measured_reference"
    elif any(s.startswith("external_artifact") for s in sources):
        label = "learned_metric_prior" if metric_evidence else "monocular_DA3"
    else:
        label = "unavailable"

    return {
        "per_result_label": label,
        "packet_sources": tuple(sorted(sources)),
        "metric_evidence": metric_evidence,
        "note": "static/dynamic verdict is geometric; masks (if any) only group",
    }
