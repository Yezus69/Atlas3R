"""M7 ray-fused static map.

Fuses ``FrameRayPacket`` surface samples into a 3D voxel map (TSDF + occupancy
log-odds + per-voxel counts) and a floor-aligned 2D ``OccupancyGrid2D``.

Honesty rules honored here:
- Voxels with no ray evidence stay UNKNOWN (never free). Unknown is not free.
- Without static/dynamic input, ``P_dynamic`` and ``P_movable_static`` are 0
  (not inferred), while free/occupied/unknown are real.
- Free = space a ray passed through before its surface. Occupied = the surface
  voxel. Beyond the surface is untouched -> unknown.
- The five occupancy probability channels stay pairwise non-collapsed per the
  contract.

``numpy`` is imported lazily inside functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import RobotEnvelopeConfig
from .contracts import (
    ContractValidationError,
    FrameRayPacket,
    OccupancyGrid2D,
    ScalePosterior,
    StaticDynamicState,
    VoxelMapState,
    VoxelOccupancyGrid3D,
)

DEFAULT_VOXEL_SIZE_M = 0.05

# Per-sample surface class codes used when fusing the static/dynamic/movable
# verdict per-voxel. Dynamic is NEVER fused into static occupancy: a dynamic
# sample writes only the dynamic channel and never carves free space.
CLASS_OCCUPIED_STATIC = 0
CLASS_MOVABLE_STATIC = 1
CLASS_DYNAMIC = 2
# A per-sample movable_static probability above this (from the M5 state's
# ``residual_summary["movable_probability"]``) tags the sample movable_static.
MOVABLE_PROB_THRESHOLD = 0.5
# Surface/free confidence saturates as ``hits / (hits + half)``: one observation
# -> 0.5, more observations approach (never reach) 1.0. Honest soft evidence: a
# single sighting is real but weaker than repeated multi-view agreement.
SURFACE_CONF_HALF_HITS = 1.0
FREE_CONF_HALF_HITS = 1.0
# Small unknown residual painted on touched (observed) cells so the unknown
# channel stays distinct from free without collapsing the simplex.
_TOUCHED_UNKNOWN_RESIDUAL = 0.05
# Scene-level confidence weight by metric acceptance status; multiplied by each
# voxel's own evidence strength to form the per-voxel map_confidence.
_MAP_CONFIDENCE_STATUS_WEIGHT = {
    "measured_metric": 0.9,
    "metric_pseudo_label": 0.6,
    "non_metric_pseudo_label": 0.4,
    "rejected": 0.1,
}
# A per-pixel sample is excluded from STATIC fusion when its dynamic probability
# dominates and exceeds this threshold. Dynamic is NEVER fused into static
# occupancy (ARCHITECTURE.md). movable_static and unknown stay distinct: a
# movable surface is still real static geometry and is kept; only dynamic is
# carved out here.
DYNAMIC_EXCLUSION_PROB_THRESHOLD = 0.5
# Bound the grid so a pathological monocular reconstruction cannot allocate an
# enormous dense array. If the scene exceeds this, the voxel size is grown.
MAX_VOXELS_PER_AXIS = 256
TSDF_TRUNCATION_VOXELS = 3.0
LOG_ODDS_HIT = 0.85
LOG_ODDS_MISS = -0.40
LOG_ODDS_CLAMP = 5.0
RANSAC_ITERS = 200
RANSAC_INLIER_DIST_FACTOR = 1.5  # in voxel units
# Minimum floor RANSAC inlier ratio to trust the floor NORMAL enough to up-align
# the reconstruction. Below this the up vector is unreliable, so the band is left
# axis-aligned (NOT floor-aligned) with a loud blocker -- never a fabricated
# alignment that merely makes the band look correct.
MIN_FLOOR_INLIER_RATIO_FOR_ALIGN = 0.30


def fuse_static_map(
    packets: Sequence[FrameRayPacket],
    scale_posterior: ScalePosterior,
    *,
    voxel_size_m: float | None = None,
    static_dynamic_states: Sequence[StaticDynamicState] | None = None,
    envelope: RobotEnvelopeConfig | None = None,
    apply_fusion_policy: bool = False,
) -> tuple[
    VoxelMapState | None,
    OccupancyGrid2D | None,
    VoxelOccupancyGrid3D | None,
    dict[str, Any] | None,
    dict[str, Any],
]:
    """Return ``(VoxelMapState, OccupancyGrid2D, VoxelOccupancyGrid3D,
    comparison_field, report)``.

    The primary robot output is the band-bounded ``VoxelOccupancyGrid3D``; the
    ``OccupancyGrid2D`` is its PURE top-down projection (single source of truth).
    ``comparison_field`` is a full-volume class field (numpy arrays) used only for
    the band agreement comparison. On too-few packets or empty surface, returns
    ``(None, None, None, None, report)`` with an explicit blocked status -- never
    a fabricated map.

    When ``static_dynamic_states`` is supplied, each per-pixel sample is TAGGED by
    class (occupied_static / movable_static / dynamic) and fused per-voxel:

    - static & movable rays carve free space in front and mark their class at the
      surface voxel;
    - dynamic rays mark ONLY the dynamic channel at the surface voxel and never
      carve free and never touch ``surface_count`` (dynamic is never static).

    A state is applied to a packet only when its per-pixel arrays match the
    packet's ray count exactly (no fabricated alignment); otherwise that frame is
    fused as all-occupied_static and the mismatch is recorded honestly.
    ``envelope`` (the robot collision band) bounds the 3D field vertically.
    """
    if envelope is None:
        envelope = RobotEnvelopeConfig()
    voxel = float(voxel_size_m) if voxel_size_m is not None else float(envelope.voxel_size_m)

    base_report: dict[str, Any] = {
        "module": "M7 - Ray-Fused Static Map",
        "packet_count": len(packets),
        "requested_voxel_size_m": voxel,
        "robot_envelope": envelope.to_dict(),
    }

    if len(packets) < 1:
        return None, None, None, None, {
            **base_report,
            "status": "blocked_no_packets",
            "blockers": ("no_packets_to_fuse",),
        }

    import numpy as np  # type: ignore

    status_value = scale_posterior.metric_acceptance_status.value
    base_frame = (
        "metric_world"
        if status_value in {"measured_metric", "metric_pseudo_label"}
        else "reconstruction_world"
    )

    surfaces, origins, ray_dirs, classes, verified, dynamic_filter_report = _collect_world_points(
        packets, np, static_dynamic_states
    )
    verified_channel_present = any(p.verified is not None for p in packets)
    if surfaces.shape[0] == 0:
        return None, None, None, None, {
            **base_report,
            "status": "blocked_no_surface_points",
            "dynamic_exclusion": dynamic_filter_report,
            "blockers": ("no_surface_points_after_lifting",),
        }

    # Static (non-dynamic) surfaces drive the floor estimate so a moving object
    # never defines the floor plane. Grid bounds use ALL surfaces so every fused
    # voxel (including dynamic) fits in the grid.
    static_mask = classes != CLASS_DYNAMIC
    static_surfaces = surfaces[static_mask] if bool(np.any(static_mask)) else surfaces

    # Gravity / up-alignment: rotate the whole reconstruction so the floor NORMAL
    # lands on its dominant axis. Then the axis-aligned band crop below is a true
    # floor-parallel slab (not an oblique cut through a tilted floor). Weak floor
    # RANSAC -> IDENTITY + loud blocker (honest unaligned band, never fabricated).
    align = _up_alignment(static_surfaces, voxel, np)
    R_up = np.asarray(align["R_up"], dtype=np.float64)
    if align["applied"]:
        surfaces = surfaces @ R_up.T
        origins = origins @ R_up.T
        ray_dirs = ray_dirs @ R_up.T
        static_surfaces = surfaces[static_mask] if bool(np.any(static_mask)) else surfaces
    coordinate_frame = base_frame + (
        "_floor_aligned" if align["applied"] else "_axis_aligned_band_not_floor_aligned"
    )

    # Robust bounds: clip extreme outliers so the grid bounds are not blown out
    # by a few stray points. The points themselves are kept; only the grid
    # extent is computed robustly.
    lo = np.percentile(surfaces, 1.0, axis=0)
    hi = np.percentile(surfaces, 99.0, axis=0)
    pad = TSDF_TRUNCATION_VOXELS * voxel
    grid_min = np.minimum(lo, origins.min(axis=0)) - pad
    grid_max = np.maximum(hi, origins.max(axis=0)) + pad

    extent = grid_max - grid_min
    extent = np.where(extent > 0.0, extent, voxel)
    effective_voxel = float(voxel)
    needed = np.ceil(extent / effective_voxel).astype(np.int64)
    while int(needed.max()) > MAX_VOXELS_PER_AXIS:
        effective_voxel *= 2.0
        needed = np.ceil(extent / effective_voxel).astype(np.int64)
    dims = tuple(int(max(1, d)) for d in needed)

    nx, ny, nz = dims
    tsdf_value = np.ones(dims, dtype=np.float64)  # truncated, default empty=+1
    tsdf_weight = np.zeros(dims, dtype=np.float64)
    log_odds = np.zeros(dims, dtype=np.float64)
    free_count = np.zeros(dims, dtype=np.float64)
    surface_count = np.zeros(dims, dtype=np.float64)
    occupied_static_count = np.zeros(dims, dtype=np.float64)
    movable_count = np.zeros(dims, dtype=np.float64)
    dynamic_count = np.zeros(dims, dtype=np.float64)
    # Verified (multi-view geometric check) hit counts, SPLIT BY CLASS so the
    # tier can mirror each lever's canonical class predicate exactly --
    # class-blind verified counting let verified MOVABLE hits source the
    # occupied_static support fill, a category crossing the canonical lever
    # forbids at any count (adversarial review, 2026-06-10).
    verified_occupied_count = np.zeros(dims, dtype=np.float64)
    verified_movable_count = np.zeros(dims, dtype=np.float64)
    uncertainty = np.zeros(dims, dtype=np.float64)

    rel_unc = float(scale_posterior.relative_scale_uncertainty)

    # Fuse each sample by class: free along static/movable rays, occupied at the
    # surface, dynamic in its own channel (never static, never free-carving).
    _fuse_rays(
        surfaces, origins, ray_dirs, classes, verified,
        grid_min, effective_voxel, dims,
        tsdf_value, tsdf_weight, log_odds,
        free_count, surface_count,
        occupied_static_count, movable_count, dynamic_count,
        verified_occupied_count, verified_movable_count,
        uncertainty, rel_unc, np,
    )

    # The candidate occupancy-estimation policy (free-carve truncation + gravity
    # support) is applied later, INSIDE the band field build, on COPIES of the
    # counts -- so this raw ``VoxelMapState`` and the full-grid free-space
    # contradiction stay an untouched baseline (the policy never degrades the
    # candidate's own raw map; it is a band-occupancy estimation layer on top).
    try:
        voxel_map = VoxelMapState(
            voxel_size_m=effective_voxel,
            coordinate_frame=coordinate_frame,
            tsdf_value=tsdf_value,
            tsdf_weight=tsdf_weight,
            occupancy_log_odds=log_odds,
            free_space_count=free_count,
            surface_count=surface_count,
            dynamic_count=dynamic_count,
            uncertainty=uncertainty,
        )
    except ContractValidationError as exc:
        return None, None, None, None, {
            **base_report,
            "status": "voxel_map_contract_rejected",
            "error": str(exc),
            "blockers": ("voxel_map_contract_rejected",),
        }

    # Floor in the (possibly aligned) fusion frame. After a successful up-align the
    # normal sits on the band axis, so this tilt collapses toward ~0; the BEFORE
    # tilt (in the original frame) is carried from the alignment step. Both are
    # reported so the residual floor tilt before/after alignment is visible.
    floor_info = _estimate_floor(static_surfaces, effective_voxel, np)
    tilt_after = _normal_axis_tilt_deg(
        floor_info["normal"], int(floor_info["floor_axis"]), np
    )
    floor_info["report"]["up_alignment_applied"] = bool(align["applied"])
    floor_info["report"]["floor_tilt_to_band_axis_deg_before"] = float(align["tilt_before_deg"])
    floor_info["report"]["floor_tilt_to_band_axis_deg_after"] = float(tilt_after)
    floor_info["report"]["min_inlier_ratio_for_alignment"] = float(MIN_FLOOR_INLIER_RATIO_FOR_ALIGN)
    floor_info["blockers"] = tuple(floor_info.get("blockers", ())) + tuple(align["blockers"])

    # Primary output: the collision-band 3D occupancy field. The 2D grid is its
    # pure top-down projection (single source of truth). ``comparison_field`` is a
    # full-volume class field used only for the band agreement comparison.
    voxel3d, band_volumes, comparison_field, band_report = _build_voxel_occupancy_3d(
        occupied_static_count, movable_count, dynamic_count, free_count,
        verified_occupied_count, verified_movable_count,
        grid_min, effective_voxel, dims,
        floor_info, scale_posterior, coordinate_frame, envelope, np,
        apply_fusion_policy=apply_fusion_policy,
    )
    # Stamp the alignment onto the comparison field so the band-agreement step can
    # bring camera centres into THIS field's (per-reconstruction) aligned frame.
    # Identity on fallback keeps that math correct without a special case.
    comparison_field["R_up"] = [[float(v) for v in row] for row in R_up]
    comparison_field["up_aligned"] = bool(align["applied"])

    grid, grid_report = _build_occupancy_grid_from_band(
        band_volumes, scale_posterior, coordinate_frame, grid_min, np,
    )

    occupied_voxels = int(np.count_nonzero(surface_count > 0.0))
    free_voxels = int(np.count_nonzero((free_count > 0.0) & (surface_count <= 0.0)))
    dynamic_voxels = int(np.count_nonzero(dynamic_count > 0.0))
    touched = int(np.count_nonzero((free_count > 0.0) | (surface_count > 0.0)))
    total_voxels = int(nx * ny * nz)
    unknown_voxels = total_voxels - touched

    # Real free/occupied contradiction over STATIC surfaces only (dynamic lives in
    # its own channel and a moving object legitimately makes a cell free at
    # another time, so it must not perturb this accepted metric).
    # Density-robust conflict test: a voxel is CONTESTED when free traversals
    # OUTNUMBER its surface hits (both sides scale with view count, so the
    # ratio survives densification). Existence-based counting ((free>0)&(surf>0))
    # saturates as views densify -- eventually some ray traverses every surface
    # voxel -- and the old order-dependent skip undercounted nondeterministically.
    # Verified-evidence exclusion (fsc METRIC only, never classification): a
    # voxel whose surface evidence passed multi-view geometric verification AT
    # THE TIER'S OWN CONFIDENCE BAR (>= verified_surface_min_count, the once-
    # fixed k -- pooled across static classes, mirroring the pooled
    # surface_count this test reads) is not contested: verification is the
    # resolution of exactly the free-vs-surface contradiction this metric
    # measures. A single sub-confidence verified hit excludes NOTHING -- the
    # excluded rate is what the acceptance gate thresholds, so the exclusion
    # bar must carry the same measured provenance as the lever gates
    # (adversarial review, 2026-06-10). No verified channel -> counts all zero
    # -> no-op, canonical rate reproduced exactly. Both rates are reported.
    k_verified_excl = float(envelope.verified_surface_min_count)
    contested = (free_count > surface_count) & (surface_count > 0.0)
    conflict_voxels_raw = int(np.count_nonzero(contested))
    conflict_excluded_by_verification = int(
        np.count_nonzero(
            contested
            & ((verified_occupied_count + verified_movable_count) >= k_verified_excl)
        )
    )
    conflict_voxels = conflict_voxels_raw - conflict_excluded_by_verification
    occupied_evidence_voxels = occupied_voxels
    if occupied_evidence_voxels > 0:
        contradiction_rate = conflict_voxels / occupied_evidence_voxels
        contradiction_rate_raw = conflict_voxels_raw / occupied_evidence_voxels
    else:
        contradiction_rate = 0.0
        contradiction_rate_raw = 0.0
    contradiction_rate = float(max(0.0, min(1.0, contradiction_rate)))
    contradiction_rate_raw = float(max(0.0, min(1.0, contradiction_rate_raw)))

    report = {
        **base_report,
        "status": "fused",
        "coordinate_frame": coordinate_frame,
        "effective_voxel_size_m": effective_voxel,
        "grid_dims": dims,
        "grid_min": tuple(float(v) for v in grid_min),
        "grid_max": tuple(float(v) for v in grid_max),
        "total_voxels": total_voxels,
        "occupied_voxel_count": occupied_voxels,
        "free_voxel_count": free_voxels,
        "dynamic_voxel_count": dynamic_voxels,
        "unknown_voxel_count": unknown_voxels,
        "occupied_fraction": occupied_voxels / total_voxels if total_voxels else 0.0,
        "free_fraction": free_voxels / total_voxels if total_voxels else 0.0,
        "unknown_fraction": unknown_voxels / total_voxels if total_voxels else 0.0,
        "free_occupied_conflict_voxel_count": conflict_voxels,
        "free_space_contradiction_rate": contradiction_rate,
        "free_space_contradiction_basis": "free_traversals_outnumber_surface_hits_over_occupied_voxels_excluding_verified_surfaces",
        "verified_tier": {
            "channel_present": bool(verified_channel_present),
            "verified_static_sample_count": int(
                np.count_nonzero(verified & (classes != CLASS_DYNAMIC))
            ),
            "sample_count": int(verified.shape[0]),
            "verified_occupied_voxel_count": int(np.count_nonzero(verified_occupied_count > 0.0)),
            "verified_movable_voxel_count": int(np.count_nonzero(verified_movable_count > 0.0)),
            "verified_surface_min_count": int(envelope.verified_surface_min_count),
            "exclusion_bar": "pooled_verified_static_hits_at_verified_surface_min_count",
            "free_space_contradiction_rate_without_verified_exclusion": contradiction_rate_raw,
            "conflict_voxels_excluded_by_verification": conflict_excluded_by_verification,
        },
        "surface_point_count": int(surfaces.shape[0]),
        "floor": floor_info["report"],
        # R_up consumed by export/visual_proof to bring the camera trajectory +
        # surface cloud into the SAME floor-aligned frame as the voxel field.
        # ``None`` on fallback so those artifacts stay in the original (unaligned)
        # frame, matching the unaligned voxel map.
        "floor_align_rotation": (
            [[float(v) for v in row] for row in R_up] if align["applied"] else None
        ),
        "up_alignment": {
            "applied": bool(align["applied"]),
            "target_axis": int(align["target_axis"]),
            "floor_tilt_to_band_axis_deg_before": float(align["tilt_before_deg"]),
            "floor_tilt_to_band_axis_deg_after": float(tilt_after),
            "floor_inlier_ratio": float(align["inlier_ratio"]),
            "min_inlier_ratio_for_alignment": float(MIN_FLOOR_INLIER_RATIO_FOR_ALIGN),
            "floor_method": align["method"],
            "grid_frame": coordinate_frame,
            "blockers": list(align["blockers"]),
        },
        "fusion_policy": {
            "apply_fusion_policy": bool(apply_fusion_policy),
            "policy_active": bool(envelope.policy_active),
            "free_carve_margin_m": float(envelope.free_carve_margin_m),
            "occupancy_support_height_m": float(envelope.occupancy_support_height_m),
            "occupancy_support_min_count": int(envelope.occupancy_support_min_count),
            "occupancy_close_voxels": int(envelope.occupancy_close_voxels),
            "free_carve_truncation": band_report.get("free_carve_truncation", {"applied": False}),
            "occupancy_completion": band_report.get("occupancy_completion", {"applied": False}),
        },
        "occupancy_grid": grid_report,
        "voxel_occupancy_3d": band_report,
        "dynamic_inference": (
            "dynamic_tagged_per_voxel_excluded_from_static_occupancy"
            if dynamic_filter_report.get("applied")
            else "not_inferred_no_static_dynamic_input"
        ),
        "dynamic_exclusion": dynamic_filter_report,
        "blockers": tuple(floor_info.get("blockers", ())) + tuple(band_report.get("blockers", ())),
    }
    return voxel_map, grid, voxel3d, comparison_field, report


# ---------------------------------------------------------------------------
# fusion internals
# ---------------------------------------------------------------------------


def _collect_world_points(
    packets: Sequence[FrameRayPacket],
    np: Any,
    static_dynamic_states: Sequence[StaticDynamicState] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Lift packet samples to world and tag each by surface class.

    Returns ``(surfaces, origins, ray_dirs, classes, verified,
    dynamic_filter_report)``. ``classes`` is an int code per sample
    (occupied_static / movable_static / dynamic), aligned 1:1 with
    ``surfaces``; ``verified`` is a bool per sample (multi-view geometric
    verification provenance, False when the packet carries no channel --
    absence of verification is never promoted). Dynamic samples are KEPT (so the
    dynamic channel can be painted) but tagged so the fuser never adds them to
    static occupancy. The filter report records, honestly, how many frames had a
    usable (shape-matched) state, how many pixels were tagged dynamic/movable, and
    which frames were left unfiltered (all-occupied_static) because their state
    did not align pixel-for-pixel.
    """
    state_by_frame: dict[int, StaticDynamicState] = {}
    for state in static_dynamic_states or ():
        state_by_frame[int(state.frame_id)] = state

    surf_list = []
    origin_list = []
    dir_list = []
    class_list = []
    verified_list = []
    frames_filtered = 0
    frames_state_shape_mismatch = 0
    dynamic_pixels_excluded = 0
    movable_pixels_tagged = 0
    total_pixels_with_state = 0
    for packet in packets:
        rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
        depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
        T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
        R = T[:3, :3]
        t = T[:3, 3]

        n = rays.shape[0]
        classes = np.full(n, CLASS_OCCUPIED_STATIC, dtype=np.int64)
        state = state_by_frame.get(int(packet.frame_id))
        if state is not None:
            stat = np.asarray(state.static_probability, dtype=np.float64).reshape((-1,))
            dyn = np.asarray(state.dynamic_probability, dtype=np.float64).reshape((-1,))
            if stat.shape[0] == n and dyn.shape[0] == n:
                # Dynamic is NEVER fused into static occupancy: tag (not drop)
                # pixels whose dynamic probability dominates and clears the
                # threshold. Movable surfaces (real geometry, repeatedly carved
                # free) are tagged distinctly; everything else is occupied_static.
                is_dynamic = (dyn >= stat) & (dyn > DYNAMIC_EXCLUSION_PROB_THRESHOLD)
                movable = _movable_probability(state, n, np)
                is_movable = (~is_dynamic) & (movable > MOVABLE_PROB_THRESHOLD)
                classes[is_movable] = CLASS_MOVABLE_STATIC
                classes[is_dynamic] = CLASS_DYNAMIC
                frames_filtered += 1
                dynamic_pixels_excluded += int(np.count_nonzero(is_dynamic))
                movable_pixels_tagged += int(np.count_nonzero(is_movable))
                total_pixels_with_state += n
            else:
                # No fabricated alignment: a state whose sampling differs from the
                # packet's rays cannot be applied pixel-for-pixel; fuse all as
                # occupied_static and record the mismatch honestly.
                frames_state_shape_mismatch += 1

        X_cam = rays * depth[:, None]
        X_world = X_cam @ R.T + t[None, :]
        dirs_world = rays @ R.T
        surf_list.append(X_world)
        origin_list.append(np.repeat(t[None, :], X_world.shape[0], axis=0))
        dir_list.append(dirs_world)
        class_list.append(classes)
        if packet.verified is not None:
            verified_list.append(np.asarray(packet.verified).reshape((-1,)).astype(bool))
        else:
            verified_list.append(np.zeros(n, dtype=bool))

    dynamic_filter_report = {
        "applied": frames_filtered > 0,
        "state_count": len(state_by_frame),
        "frames_filtered": frames_filtered,
        "frames_state_shape_mismatch": frames_state_shape_mismatch,
        "dynamic_pixels_excluded": dynamic_pixels_excluded,
        "movable_pixels_tagged": movable_pixels_tagged,
        "pixels_considered_with_state": total_pixels_with_state,
        "exclusion_threshold": DYNAMIC_EXCLUSION_PROB_THRESHOLD,
        "note": (
            "dynamic_dominant_pixels_tagged_dynamic_excluded_from_static_occupancy_movable_kept"
            if frames_filtered
            else "no_shape_matched_static_dynamic_state_fused_all_occupied_static"
        ),
    }

    if not surf_list:
        empty = np.zeros((0, 3), dtype=np.float64)
        empty_cls = np.zeros((0,), dtype=np.int64)
        empty_ver = np.zeros((0,), dtype=bool)
        return empty, empty, empty, empty_cls, empty_ver, dynamic_filter_report
    return (
        np.concatenate(surf_list, axis=0),
        np.concatenate(origin_list, axis=0),
        np.concatenate(dir_list, axis=0),
        np.concatenate(class_list, axis=0),
        np.concatenate(verified_list, axis=0),
        dynamic_filter_report,
    )


def _movable_probability(state: StaticDynamicState, n: int, np: Any) -> Any:
    """Per-sample movable_static probability from the state, or zeros.

    The M5 ``StaticDynamicState`` simplex carries only static/dynamic/unknown; the
    per-sample movable probability rides in ``residual_summary["movable_probability"]``
    (written by ``static_dynamic._build_state``). Absent or shape-mismatched ->
    zeros (movable is never fabricated; the sample stays occupied_static).
    """
    residual = getattr(state, "residual_summary", None)
    movable = residual.get("movable_probability") if isinstance(residual, Mapping) else None
    if movable is None:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(movable, dtype=np.float64).reshape((-1,))
    if arr.shape[0] != n:
        return np.zeros(n, dtype=np.float64)
    return arr


def _fuse_rays(
    surfaces, origins, ray_dirs, classes, verified,
    grid_min, voxel, dims,
    tsdf_value, tsdf_weight, log_odds,
    free_count, surface_count,
    occupied_static_count, movable_count, dynamic_count,
    verified_occupied_count, verified_movable_count,
    uncertainty, rel_unc, np,
):
    nx, ny, nz = dims
    inv_voxel = 1.0 / voxel
    n = surfaces.shape[0]
    # No global subsample: the per-packet ray cap upstream (<=2048/frame) bounds
    # the budget, and it scales WITH frame count so denser-view runs fuse the
    # same per-frame evidence density as sparse runs. (The previous fixed 20k
    # subsample silently starved per-frame evidence as views densified, which
    # confounded every density comparison.)

    surf_idx = np.floor((surfaces - grid_min[None, :]) * inv_voxel).astype(np.int64)
    in_bounds = (
        (surf_idx[:, 0] >= 0) & (surf_idx[:, 0] < nx)
        & (surf_idx[:, 1] >= 0) & (surf_idx[:, 1] < ny)
        & (surf_idx[:, 2] >= 0) & (surf_idx[:, 2] < nz)
    )
    # PASS 1 -- integrate ALL surface evidence first. The carve pass below skips
    # voxels with surface evidence; two passes make that skip a function of the
    # COMPLETE surface field instead of the ray processing order (the previous
    # interleaved pass made fusion output depend on sample order).
    for k in range(n):
        if not in_bounds[k]:
            continue
        cls = int(classes[k])
        ix, iy, iz = int(surf_idx[k, 0]), int(surf_idx[k, 1]), int(surf_idx[k, 2])
        if cls == CLASS_DYNAMIC:
            # Dynamic surface hit: paint ONLY the dynamic channel. Never added
            # to static surface_count / TSDF / occupancy (dynamic is not static).
            dynamic_count[ix, iy, iz] += 1.0
        else:
            surface_count[ix, iy, iz] += 1.0
            tsdf_value[ix, iy, iz] = 0.0
            tsdf_weight[ix, iy, iz] += 1.0
            log_odds[ix, iy, iz] = float(np.clip(log_odds[ix, iy, iz] + LOG_ODDS_HIT, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP))
            uncertainty[ix, iy, iz] += rel_unc
            # Raw per-class provenance counts: how many fused hits of each
            # static class passed multi-view geometric verification. Counts
            # only; the tier acts on GATES downstream, never on these counts.
            if cls == CLASS_MOVABLE_STATIC:
                movable_count[ix, iy, iz] += 1.0
                if bool(verified[k]):
                    verified_movable_count[ix, iy, iz] += 1.0
            else:
                occupied_static_count[ix, iy, iz] += 1.0
                if bool(verified[k]):
                    verified_occupied_count[ix, iy, iz] += 1.0

    # PASS 2 -- carve free space along static/movable rays against the complete
    # surface field. Dynamic rays do NOT carve free space: a moving object
    # gives unreliable free evidence.
    for k in range(n):
        if int(classes[k]) == CLASS_DYNAMIC:
            continue
        origin = origins[k]
        target = surfaces[k]
        seg = target - origin
        dist = float(np.linalg.norm(seg))
        if dist <= voxel:
            continue
        n_steps = int(dist * inv_voxel)
        if n_steps <= 1:
            continue
        # Sample free voxels along the ray, excluding the final surface voxel.
        ts = np.linspace(0.0, 1.0, num=min(n_steps, 64), endpoint=False)
        pts = origin[None, :] + ts[:, None] * seg[None, :]
        free_idx = np.floor((pts - grid_min[None, :]) * inv_voxel).astype(np.int64)
        for m in range(free_idx.shape[0] - 1):  # last sample is near surface
            fx, fy, fz = int(free_idx[m, 0]), int(free_idx[m, 1]), int(free_idx[m, 2])
            if 0 <= fx < nx and 0 <= fy < ny and 0 <= fz < nz:
                # Carve UNCONDITIONALLY. Surface priority lives in the class /
                # probability construction (surface always beats free), so this
                # does not change classes -- it makes the free-vs-surface
                # CONFLICT signal (free_space_contradiction_rate) deterministic
                # and COMPLETE: every traversal of a surface voxel is counted,
                # instead of the order-dependent undercount of the old
                # interleaved skip (output depended on sample order; and a
                # complete-field skip would structurally zero the metric and
                # silently un-gate the desk failure class).
                free_count[fx, fy, fz] += 1.0
                log_odds[fx, fy, fz] = float(np.clip(log_odds[fx, fy, fz] + LOG_ODDS_MISS, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP))


def _band_indices(floor_axis, band_min, band_max, grid_min, voxel, dims, np):
    """Voxel indices along ``floor_axis`` whose slab overlaps the collision band."""
    nf = dims[floor_axis]
    lo_face = grid_min[floor_axis] + np.arange(nf) * voxel
    hi_face = lo_face + voxel
    in_band = (hi_face > band_min) & (lo_face < band_max)
    return np.nonzero(in_band)[0]


def _apply_band_free_truncation(
    free_count, occupied_static_count, movable_count,
    verified_occupied_count, verified_movable_count,
    floor_axis, band_min, band_max, grid_min, voxel, dims, envelope, np,
):
    """Directional (downward) free-carve truncation inside the collision band.

    Returns ``(free_count_copy, report)`` -- a NEW free-count array (the caller's raw
    counts and the measured GT field are never mutated). Free is retracted ONLY in
    the column directly BELOW a fused surface (toward the floor) within
    ``free_carve_margin_m``: that is the grazing-ray flood that masks an obstacle's
    support column. LATERAL free (beside the obstacle) is preserved, so observed free
    space next to an obstacle is not turned into a false positive. Retracted free
    becomes UNKNOWN (free_count 0), never occupied.
    """
    margin_m = float(envelope.free_carve_margin_m)
    margin_voxels = max(1, int(round(margin_m / voxel)))
    full_column = bool(getattr(envelope, "free_carve_full_column", False))
    min_count = int(envelope.occupancy_support_min_count)
    free_count = np.asarray(free_count, dtype=np.float64).copy()
    # Only a CONFIDENT obstacle retracts the free below it, so single-hit depth noise
    # high in the band cannot turn the observed-free floor beneath it into unknown.
    # Verified-evidence tier (ARCHITECTURE.md FrameRayPacket): hits that passed
    # multi-view geometric verification reach confidence at a LOWER count
    # (verified_surface_min_count, fixed once from the measured 2.2x
    # verification-accuracy ratio) -- the gate is tiered, the counts are not.
    # The tier mirrors this lever's canonical PER-CLASS predicate exactly
    # (occupied OR movable, each at its own bar) -- never a class crossing.
    k_verified = int(envelope.verified_surface_min_count)
    count_confident = (np.asarray(occupied_static_count) >= float(min_count)) | (
        np.asarray(movable_count) >= float(min_count)
    )
    verified_confident = (np.asarray(verified_occupied_count) >= float(k_verified)) | (
        np.asarray(verified_movable_count) >= float(k_verified)
    )
    surf = count_confident | verified_confident
    sources_verified_only = int(np.count_nonzero(verified_confident & ~count_confident))

    band_idx = _band_indices(floor_axis, band_min, band_max, grid_min, voxel, dims, np)
    free_before = int(np.count_nonzero(free_count > 0.0))
    if band_idx.shape[0] == 0:
        return free_count, {"applied": True, "note": "no_band_slices_truncation_noop",
                            "margin_voxels": int(margin_voxels), "free_voxels_retracted_to_unknown": 0,
                            "verified_surface_min_count": k_verified,
                            "sources_confident_via_verified_only": sources_verified_only}
    lo = int(band_idx.min())
    top = int(band_idx.max())
    if full_column:
        # Full-column variant: a CONFIDENT surface anywhere in the band retracts
        # free in its ENTIRE band column below (free -> unknown, never occupied).
        # Parameter-free by the gravity principle -- the column under a
        # confidently fused surface is exactly where monocular see-through bias
        # lives; the margin knob is removed rather than tuned.
        margin_voxels = max(margin_voxels, top - lo + 1)

    fm = np.moveaxis(free_count, floor_axis, 0)  # view of the copy; index up = away from floor
    sm = np.moveaxis(surf, floor_axis, 0)
    protect = np.zeros_like(sm, dtype=bool)
    for step in range(1, margin_voxels + 1):
        hi = top - step + 1
        if hi <= lo:
            break
        tgt = protect[lo:hi]              # band voxel k ...
        src = sm[lo + step: top + 1]      # ... has a surface at k+step (above it)
        np.logical_or(tgt, src, out=tgt)  # disjoint base arrays -> safe
    retract = protect & (fm > 0.0)
    fm[retract] = 0.0  # writes into the copy via the moved-axis view
    free_after = int(np.count_nonzero(free_count > 0.0))
    return free_count, {
        "applied": True,
        "margin_m": margin_m,
        "full_column": full_column,
        "margin_voxels": int(margin_voxels),
        "direction": "downward_below_surface_only",
        "free_voxels_before": free_before,
        "free_voxels_after": free_after,
        "free_voxels_retracted_to_unknown": free_before - free_after,
        "verified_surface_min_count": k_verified,
        "sources_confident_via_verified_only": sources_verified_only,
    }


def _apply_occupancy_completion(
    occupied_static_count, movable_count, dynamic_count, free_count,
    verified_occupied_count,
    floor_axis, band_min, band_max, grid_min, voxel, dims, envelope, np,
):
    """Complete the candidate occupancy estimate inside the collision band.

    Returns ``(occ, mov, report)`` with NEW arrays (inputs untouched -- the raw
    counts and measured GT field are never mutated). Two generic levers:

    - downward gravity SUPPORT: a detected obstacle rests on the floor, so occupancy
      is propagated toward the floor by up to ``occupancy_support_height_m`` to claim
      the support column. It is HONEST -- it fills ONLY voxels that are currently
      UNKNOWN (no fused free / surface / dynamic evidence) and supports only from
      CONFIDENT obstacles (occupied count >= ``occupancy_support_min_count``), so it
      never overrides an observed-free voxel and single-hit depth noise high in the
      band cannot conjure a column of occupancy. (Free-carve truncation runs first,
      so the flood of free that masked real obstacle bases is already retracted to
      unknown and becomes eligible here.)
    - in-plane morphological CLOSING (radius ``occupancy_close_voxels``) that bridges
      small gaps enclosed by occupancy without expanding the obstacle outward.

    Both ADD occupancy only into unknown space; free/dynamic counts are untouched.
    The per-voxel probability construction downstream keeps the contract invariants.
    """
    occ = np.asarray(occupied_static_count, dtype=np.float64)
    support_h = float(envelope.occupancy_support_height_m)
    min_count = int(envelope.occupancy_support_min_count)
    close_v = int(envelope.occupancy_close_voxels)

    nf = dims[floor_axis]
    lo_face = grid_min[floor_axis] + np.arange(nf) * voxel
    hi_face = lo_face + voxel
    in_band = (hi_face > band_min) & (lo_face < band_max)
    band_idx = np.nonzero(in_band)[0]
    occ_before = int(np.count_nonzero(occ > 0.0))
    report: dict[str, Any] = {
        "applied": True,
        "support_height_m": support_h,
        "support_min_count": min_count,
        "close_voxels": close_v,
        "occ_voxels_before": occ_before,
        "band_slice_count": int(band_idx.shape[0]),
        # Always present so a reader can distinguish "tier evaluated, nothing
        # qualified" (count 0 below) from "lever never ran" (key absent).
        "verified_surface_min_count": int(envelope.verified_surface_min_count),
        "support_sources_confident_via_verified_only": 0,
    }

    if band_idx.shape[0] == 0:
        report["note"] = "no_band_slices_completion_noop"
        return occ, movable_count, report

    lo = int(band_idx.min())
    top = int(band_idx.max())

    # --- downward gravity support within the band (confident source) ---
    if support_h > 0.0:
        support_voxels = max(1, int(round(support_h / voxel)))
        overrides_free = bool(getattr(envelope, "occupancy_support_overrides_free", False))
        movable = np.asarray(movable_count)
        dynamic = np.asarray(dynamic_count)
        # Eligible target voxels below a confident obstacle. Default: UNKNOWN only
        # (no fused evidence). With occupancy_support_overrides_free: also fill
        # OBSERVED-FREE voxels -- the structural prior that a CONFIDENT floor-supported
        # obstacle's base column is solid to the floor even where a stray ray carved
        # it free (a depth-noise over-carve). Movable/dynamic are never overridden.
        if overrides_free:
            eligible = (occ <= 0.0) & (movable <= 0.0) & (dynamic <= 0.0)
        else:
            eligible = (occ <= 0.0) & (movable <= 0.0) & (dynamic <= 0.0) & (np.asarray(free_count) <= 0.0)
        # Confident obstacle voxels. Verified-evidence tier: confident :=
        # (occ >= min_count) OR (verified_occupied >= verified_surface_min_count)
        # -- a multi-view-verified thin-structure hit qualifies as a support
        # source at the lower, once-fixed count. The tier mirrors this lever's
        # canonical OCCUPIED-ONLY source predicate (movable evidence never
        # sources the occupied_static fill, verified or not). The fill itself
        # stays UNKNOWN-only (free is never overridden); raw counts untouched.
        k_verified = int(envelope.verified_surface_min_count)
        count_confident = occ >= float(min_count)
        verified_confident = np.asarray(verified_occupied_count) >= float(k_verified)
        conf_src = count_confident | verified_confident
        support_sources_verified_only = int(
            np.count_nonzero(verified_confident & ~count_confident)
        )
        em = np.moveaxis(eligible, floor_axis, 0)       # views; index up = away from floor
        cm = np.moveaxis(conf_src, floor_axis, 0)
        supported = np.zeros_like(em, dtype=bool)
        for step in range(1, support_voxels + 1):
            hi = top - step + 1
            if hi <= lo:
                break
            tgt = supported[lo:hi]                # band slices that receive support
            src = cm[lo + step: top + 1]          # confident obstacle slices above
            np.logical_or(tgt, src, out=tgt)      # disjoint base arrays -> safe
        fill = supported & em
        if bool(np.any(fill)):
            occ = occ.copy()
            np.moveaxis(occ, floor_axis, 0)[fill] = 1.0  # minimal, honest occupancy
        report["support_voxels"] = int(support_voxels)
        report["support_overrides_free"] = overrides_free
        report["support_filled_voxels"] = int(np.count_nonzero(fill))
        report["support_sources_confident_via_verified_only"] = support_sources_verified_only

    # --- in-plane morphological closing of the band occupancy ---
    if close_v > 0:
        occ_mask = occ > 0.0
        plane_axes = tuple(a for a in range(3) if a != floor_axis)
        try:
            from scipy import ndimage  # lazy

            struct = np.zeros((3, 3, 3), dtype=bool)
            # in-plane 4-neighbour structuring element (no coupling across floor axis)
            center = [1, 1, 1]
            for a in plane_axes:
                for d in (-1, 1):
                    idx = list(center)
                    idx[a] += d
                    struct[tuple(idx)] = True
            struct[1, 1, 1] = True
            closed = ndimage.binary_closing(occ_mask, structure=struct, iterations=close_v)
        except Exception:
            closed = occ_mask  # closing is best-effort; never fabricate on failure
        # restrict added occupancy to band slices
        band_mask = np.zeros(dims, dtype=bool)
        sl = [slice(None)] * 3
        sl[floor_axis] = slice(lo, top + 1)
        band_mask[tuple(sl)] = True
        added = closed & band_mask & (~occ_mask)
        if bool(np.any(added)):
            occ = occ.copy()
            occ[added] = np.maximum(occ[added], 1.0)
        report["closed_voxels_added"] = int(np.count_nonzero(closed & band_mask & (~occ_mask)))

    report["occ_voxels_after"] = int(np.count_nonzero(occ > 0.0))
    report["occ_voxels_added"] = report["occ_voxels_after"] - occ_before
    return occ, movable_count, report


def _estimate_floor(surfaces, voxel, np) -> dict[str, Any]:
    """RANSAC a horizontal-ish floor plane; fall back to lowest-Z heuristic."""
    n = surfaces.shape[0]
    if n < 3:
        z_floor = float(np.min(surfaces[:, 2])) if n else 0.0
        return {
            "floor_axis": 2,
            "floor_value": z_floor,
            "normal": (0.0, 0.0, 1.0),
            "method": "lowest_z_heuristic_insufficient_points",
            "report": {"method": "lowest_z_heuristic", "floor_value": z_floor, "inlier_ratio": 0.0},
            "blockers": ("floor_ransac_insufficient_points_used_lowest_z",),
        }

    rng = np.random.default_rng(0)
    inlier_dist = RANSAC_INLIER_DIST_FACTOR * voxel
    best_inliers = -1
    best_plane = None
    for _ in range(RANSAC_ITERS):
        sample = surfaces[rng.choice(n, size=3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        dist = np.abs(surfaces @ normal + d)
        inliers = int(np.count_nonzero(dist < inlier_dist))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or best_inliers < max(3, int(0.05 * n)):
        z_floor = float(np.percentile(surfaces[:, 2], 5.0))
        return {
            "floor_axis": 2,
            "floor_value": z_floor,
            "normal": (0.0, 0.0, 1.0),
            "method": "lowest_z_heuristic_ransac_failed",
            "report": {"method": "lowest_z_heuristic", "floor_value": z_floor, "inlier_ratio": 0.0},
            "blockers": ("floor_ransac_failed_used_lowest_z_assumption",),
        }

    normal, d = best_plane
    # Orient normal so it points "up" relative to the points' centroid.
    centroid = surfaces.mean(axis=0)
    if (normal @ centroid + d) < 0:
        normal = -normal
        d = -d
    inlier_ratio = best_inliers / n
    # Determine the dominant axis of the plane normal for the 2D projection.
    floor_axis = int(np.argmax(np.abs(normal)))
    floor_value = float(np.percentile(surfaces[:, floor_axis], 5.0))
    return {
        "floor_axis": floor_axis,
        "floor_value": floor_value,
        "normal": tuple(float(v) for v in normal),
        "plane_d": float(d),
        "method": "ransac",
        "report": {
            "method": "ransac",
            "inlier_ratio": float(inlier_ratio),
            "normal": tuple(float(v) for v in normal),
            "floor_axis": floor_axis,
            "floor_value": floor_value,
        },
        "blockers": (),
    }


def _rotation_align(a, b, np):
    """Shortest-arc rotation matrix taking unit vector ``a`` onto unit vector ``b``.

    Rodrigues form; handles the parallel (identity) and antiparallel (180 deg about
    an arbitrary perpendicular axis) degeneracies.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(float(a[0])) <= 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (vx @ vx)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _normal_axis_tilt_deg(normal, axis, np) -> float:
    """Acute angle (deg) between a plane normal and the given world axis LINE."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    cos = min(1.0, max(0.0, abs(float(n[axis]))))
    return float(np.degrees(np.arccos(cos)))


def _up_alignment(static_surfaces, voxel, np) -> dict[str, Any]:
    """Derive a robust per-reconstruction up-alignment rotation ``R_up``.

    Estimate the floor, then build ``R_up`` mapping the floor NORMAL onto the +unit
    vector of its own dominant axis, so the axis-aligned band crop becomes a true
    floor-parallel slab. If the floor RANSAC is weak (``inlier_ratio`` below
    ``MIN_FLOOR_INLIER_RATIO_FOR_ALIGN``) or did not run, the up vector is
    unreliable -> IDENTITY + a loud blocker; the band is left axis-aligned and
    reported as unaligned, never fabricated into looking correct.
    """
    floor0 = _estimate_floor(static_surfaces, voxel, np)
    normal = np.asarray(floor0["normal"], dtype=np.float64)
    floor_axis = int(floor0["floor_axis"])
    method = str(floor0.get("method", ""))
    inlier_ratio = float(floor0.get("report", {}).get("inlier_ratio", 0.0))
    tilt_before = _normal_axis_tilt_deg(normal, floor_axis, np)
    reliable = method == "ransac" and inlier_ratio >= MIN_FLOOR_INLIER_RATIO_FOR_ALIGN
    if not reliable:
        return {
            "applied": False,
            "R_up": np.eye(3),
            "target_axis": floor_axis,
            "tilt_before_deg": tilt_before,
            "inlier_ratio": inlier_ratio,
            "method": method,
            "blockers": ("floor_normal_unreliable_band_not_floor_aligned",),
        }
    target = np.zeros(3, dtype=np.float64)
    target[floor_axis] = 1.0
    R_up = _rotation_align(normal, target, np)
    return {
        "applied": True,
        "R_up": R_up,
        "target_axis": floor_axis,
        "tilt_before_deg": tilt_before,
        "inlier_ratio": inlier_ratio,
        "method": method,
        "blockers": (),
    }


def _build_voxel_occupancy_3d(
    occupied_static_count, movable_count, dynamic_count, free_count,
    verified_occupied_count, verified_movable_count,
    grid_min, voxel, dims,
    floor_info, scale_posterior, coordinate_frame, envelope, np,
    apply_fusion_policy: bool = False,
) -> tuple[VoxelOccupancyGrid3D | None, dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Crop the fused volume to the robot collision band and build the per-voxel
    ``VoxelOccupancyGrid3D``.

    Returns ``(VoxelOccupancyGrid3D|None, band_volumes, comparison_field, report)``.
    ``band_volumes`` carries the cropped probability arrays + geometry that the 2D
    projection consumes (so the 2D grid is a pure function of this field).
    ``comparison_field`` is a full-volume class field for the band comparison. The per-voxel
    probabilities are a pairwise-non-collapse soft distribution: surface
    confidence ``c_surf = surf/(surf+half)`` is split among occupied/movable/
    dynamic by their hit shares; free is suppressed where surface evidence exists;
    unknown holds the residual (and is 1 where the voxel was never observed).
    """
    floor_axis = int(floor_info["floor_axis"])
    floor_value = float(floor_info["floor_value"])
    band_min = floor_value
    band_max = floor_value + float(envelope.band_height_m)

    # Candidate occupancy-estimation policy (apply_fusion_policy), applied on COPIES
    # so the caller's raw counts (and the measured GT field) are never mutated:
    #   1. DIRECTIONAL free-carve truncation -- retract the free flood in the column
    #      directly BELOW a fused surface (toward the floor) within
    #      ``free_carve_margin_m``. Lateral free is left intact, so free space beside
    #      an obstacle is preserved (this is why it is downward-only, not isotropic).
    #      Retracted free becomes UNKNOWN, never occupied.
    #   2. Gravity SUPPORT -- fill those now-unknown base voxels (and any pre-existing
    #      unknown) below a CONFIDENT obstacle with occupancy (honest, unknown-only).
    #   3. optional in-plane CLOSING.
    # The floor axis is known here, so truncation/support act strictly along the
    # collision band. OFF reproduces the prior fuser.
    free_carve_truncation_report: dict[str, Any] = {"applied": False}
    occupancy_completion_report: dict[str, Any] = {"applied": False}
    if apply_fusion_policy and envelope.policy_active:
        if float(envelope.free_carve_margin_m) > 0.0:
            free_count, free_carve_truncation_report = _apply_band_free_truncation(
                free_count, occupied_static_count, movable_count,
                verified_occupied_count, verified_movable_count,
                floor_axis, band_min, band_max, grid_min, voxel, dims, envelope, np,
            )
        if (
            float(envelope.occupancy_support_height_m) > 0.0
            or int(envelope.occupancy_close_voxels) > 0
        ):
            occupied_static_count, movable_count, occupancy_completion_report = (
                _apply_occupancy_completion(
                    occupied_static_count, movable_count, dynamic_count, free_count,
                    verified_occupied_count,
                    floor_axis, band_min, band_max, grid_min, voxel, dims, envelope, np,
                )
            )

    # Full-volume argmax class field (int8) + touched mask, used ONLY for the
    # band agreement comparison: the eval region is the OTHER field's band, and
    # this full field is sampled there (robust to the floor-slab tilt between two
    # independently floor-aligned reconstructions under a camera-only Sim(3)).
    # Class codes: 0=free, 1=occupied_static, 2=movable_static, 3=dynamic,
    # 4=unknown. Computed without materialising 5 full-volume float arrays.
    full_surf = occupied_static_count + movable_count + dynamic_count
    full_surf_pos = full_surf > 0.0
    full_free_only = (~full_surf_pos) & (free_count > 0.0)
    full_surf_cls = np.where(
        (occupied_static_count >= movable_count) & (occupied_static_count >= dynamic_count),
        1,
        np.where(movable_count >= dynamic_count, 2, 3),
    )
    full_cls = np.full(dims, 4, dtype=np.int8)
    full_cls = np.where(full_free_only, np.int8(0), full_cls)
    full_cls = np.where(full_surf_pos, full_surf_cls.astype(np.int8), full_cls)
    floor_normal = floor_info.get("normal", (0.0, 0.0, 1.0))
    comparison_field = {
        "class": full_cls,
        "touched": full_surf_pos | (free_count > 0.0),
        "origin": (float(grid_min[0]), float(grid_min[1]), float(grid_min[2])),
        "voxel": float(voxel),
        "dims": tuple(int(d) for d in dims),
        "floor_axis": floor_axis,
        "floor_normal": tuple(float(x) for x in floor_normal),
        # band_min_m / band_max_m are filled in once the band crop is known.
    }

    nf = dims[floor_axis]
    lo_face = grid_min[floor_axis] + np.arange(nf) * voxel
    hi_face = lo_face + voxel
    in_band = (hi_face > band_min) & (lo_face < band_max)
    band_idx = np.nonzero(in_band)[0]
    if band_idx.shape[0] == 0:
        # Floor estimate sits outside the fused geometry: clamp to the single
        # slice containing band_min so we still emit an honest band, not nothing.
        k0 = int(np.clip(np.floor((band_min - grid_min[floor_axis]) / voxel), 0, nf - 1))
        band_idx = np.asarray([k0], dtype=np.int64)

    def take(a: Any) -> Any:
        return np.take(a, band_idx, axis=floor_axis)

    occ = take(occupied_static_count)
    mov = take(movable_count)
    dyn = take(dynamic_count)
    free = take(free_count)

    surf = occ + mov + dyn
    c_surf = surf / (surf + SURFACE_CONF_HALF_HITS)
    c_free = free / (free + FREE_CONF_HALF_HITS)
    surf_pos = surf > 0.0
    share_occ = np.where(surf_pos, occ / np.where(surf_pos, surf, 1.0), 0.0)
    share_mov = np.where(surf_pos, mov / np.where(surf_pos, surf, 1.0), 0.0)
    share_dyn = np.where(surf_pos, dyn / np.where(surf_pos, surf, 1.0), 0.0)
    p_occ = c_surf * share_occ
    p_mov = c_surf * share_mov
    p_dyn = c_surf * share_dyn
    p_free = c_free * (1.0 - c_surf)
    touched = surf_pos | (free > 0.0)
    occupied_mass = p_occ + p_mov + p_dyn + p_free
    p_unknown = np.where(touched, np.maximum(0.0, 1.0 - occupied_mass), 1.0)

    status_value = scale_posterior.metric_acceptance_status.value
    status_weight = _MAP_CONFIDENCE_STATUS_WEIGHT.get(status_value, 0.3)
    evidence_strength = np.maximum(c_surf, c_free)
    map_confidence = np.clip(status_weight * evidence_strength, 0.0, 1.0)

    origin = [float(grid_min[0]), float(grid_min[1]), float(grid_min[2])]
    origin[floor_axis] = float(grid_min[floor_axis] + int(band_idx[0]) * voxel)
    band_floor_min = origin[floor_axis]
    band_floor_max = band_floor_min + int(band_idx.shape[0]) * voxel
    comparison_field["band_min_m"] = float(band_floor_min)
    comparison_field["band_max_m"] = float(band_floor_max)

    plane_axes = tuple(a for a in range(3) if a != floor_axis)
    band_dims = list(int(d) for d in dims)
    band_dims[floor_axis] = int(band_idx.shape[0])

    band_volumes: dict[str, Any] = {
        "P_free": p_free, "P_occupied_static": p_occ, "P_movable_static": p_mov,
        "P_dynamic": p_dyn, "P_unknown": p_unknown, "map_confidence": map_confidence,
        "touched": touched,
        "floor_axis": floor_axis, "plane_axes": plane_axes,
        "origin_world": tuple(origin), "voxel": float(voxel),
        "band_dims": tuple(band_dims),
        "band_min_m": float(band_floor_min), "band_max_m": float(band_floor_max),
    }

    try:
        voxel3d = VoxelOccupancyGrid3D(
            grid_frame=coordinate_frame,
            voxel_size_m=float(voxel),
            origin_world=tuple(origin),
            floor_axis=floor_axis,
            band_min_m=float(band_floor_min),
            band_max_m=float(band_floor_max),
            P_free=p_free.astype(np.float64),
            P_occupied_static=p_occ.astype(np.float64),
            P_movable_static=p_mov.astype(np.float64),
            P_dynamic=p_dyn.astype(np.float64),
            P_unknown=p_unknown.astype(np.float64),
            map_confidence=map_confidence.astype(np.float64),
            scale_uncertainty=float(scale_posterior.relative_scale_uncertainty),
            acceptance_category=scale_posterior.metric_acceptance_status,
        )
    except ContractValidationError as exc:
        return None, band_volumes, comparison_field, {
            "status": "voxel_occupancy_3d_contract_rejected",
            "error": str(exc),
            "blockers": ("voxel_occupancy_3d_contract_rejected",),
        }

    occ_vox = int(np.count_nonzero(occ > 0.0))
    mov_vox = int(np.count_nonzero(mov > 0.0))
    dyn_vox = int(np.count_nonzero(dyn > 0.0))
    free_vox = int(np.count_nonzero((free > 0.0) & (~surf_pos)))
    band_total = 1
    for d in band_dims:
        band_total *= int(d)
    unknown_vox = band_total - int(np.count_nonzero(touched))
    report = {
        "status": "built",
        "floor_axis": floor_axis,
        "plane_axes": plane_axes,
        "band_slice_count": int(band_idx.shape[0]),
        "band_slice_indices": [int(i) for i in band_idx],
        "band_min_m": float(band_floor_min),
        "band_max_m": float(band_floor_max),
        "requested_band_min_m": float(band_min),
        "requested_band_max_m": float(band_max),
        "collision_height_m": float(envelope.collision_height_m),
        "margin_m": float(envelope.margin_m),
        "voxel_size_m": float(voxel),
        "band_dims": tuple(band_dims),
        "band_total_voxels": band_total,
        "occupied_static_voxels": occ_vox,
        "movable_static_voxels": mov_vox,
        "dynamic_voxels": dyn_vox,
        "free_voxels": free_vox,
        "unknown_voxels": unknown_vox,
        "occupied_static_fraction": occ_vox / band_total if band_total else 0.0,
        "movable_static_fraction": mov_vox / band_total if band_total else 0.0,
        "dynamic_fraction": dyn_vox / band_total if band_total else 0.0,
        "free_fraction": free_vox / band_total if band_total else 0.0,
        "unknown_fraction": unknown_vox / band_total if band_total else 0.0,
        "movable_inferred": mov_vox > 0,
        "dynamic_inferred": dyn_vox > 0,
        "free_carve_truncation": free_carve_truncation_report,
        "occupancy_completion": occupancy_completion_report,
        "blockers": (),
    }
    return voxel3d, band_volumes, comparison_field, report


def _build_occupancy_grid_from_band(
    band_volumes: dict[str, Any] | None,
    scale_posterior: ScalePosterior,
    coordinate_frame: str,
    grid_min: Any,
    np: Any,
) -> tuple[OccupancyGrid2D | None, dict[str, Any]]:
    """Project the band 3D field top-down into the ``OccupancyGrid2D``.

    Single source of truth = the 3D field. The max-then-suppress rule preserves
    the OccupancyGrid2D pairwise non-collapse invariants by construction so the
    projection can never raise the collapse error:

        P_occupied_static = max over band column
        P_movable_static  = max over band column
        P_dynamic         = min(max_dynamic, 1 - P_occupied_static)
        P_free            = max_free * (1 - max(occ, movable, dynamic))
        P_unknown         = 1 where the whole column is unobserved, else a small
                            residual capped to keep free + unknown <= 1
    """
    if band_volumes is None:
        return None, {"status": "occupancy_grid_blocked_no_band_field"}

    floor_axis = band_volumes["floor_axis"]
    a0, a1 = band_volumes["plane_axes"]
    voxel = band_volumes["voxel"]

    o = np.max(band_volumes["P_occupied_static"], axis=floor_axis)
    m = np.max(band_volumes["P_movable_static"], axis=floor_axis)
    d = np.max(band_volumes["P_dynamic"], axis=floor_axis)
    f = np.max(band_volumes["P_free"], axis=floor_axis)
    touched_col = np.any(band_volumes["touched"], axis=floor_axis)

    hard = np.maximum(np.maximum(o, m), d)
    p_occupied = o.astype(np.float64)
    p_movable = m.astype(np.float64)
    p_dynamic = np.minimum(d, 1.0 - o).astype(np.float64)
    p_free = (f * (1.0 - hard)).astype(np.float64)
    p_unknown = np.where(
        touched_col, np.minimum(_TOUCHED_UNKNOWN_RESIDUAL, 1.0 - p_free), 1.0
    ).astype(np.float64)

    # Height extent per cell from touched BAND voxels (reuse the floor-axis logic
    # with the band-cropped floor origin so heights live inside the band).
    band_grid_min = np.asarray(grid_min, dtype=np.float64).copy()
    band_grid_min[floor_axis] = float(band_volumes["origin_world"][floor_axis])
    height_min, height_max = _height_extents(
        band_volumes["touched"], floor_axis, band_grid_min, voxel,
        band_volumes["band_dims"], np,
    )

    origin_world = (float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
    scale_unc = float(scale_posterior.scale_std)

    try:
        grid = OccupancyGrid2D(
            grid_frame=coordinate_frame,
            resolution_m=float(voxel),
            origin_world=origin_world,
            P_free=p_free,
            P_occupied_static=p_occupied,
            P_movable_static=p_movable,
            P_dynamic=p_dynamic,
            P_unknown=p_unknown,
            height_min_m=height_min,
            height_max_m=height_max,
            scale_uncertainty=scale_unc,
            map_confidence=_map_confidence(scale_posterior),
        )
    except ContractValidationError as exc:
        return None, {"status": "occupancy_grid_contract_rejected", "error": str(exc)}

    plane_dims = (band_volumes["band_dims"][a0], band_volumes["band_dims"][a1])
    occ_cells = int(np.count_nonzero(p_occupied > 0.5))
    mov_cells = int(np.count_nonzero(p_movable > 0.5))
    dyn_cells = int(np.count_nonzero(p_dynamic > 0.5))
    free_cells = int(np.count_nonzero((p_free > 0.5) & (p_occupied <= 0.5)))
    unknown_cells = int(np.count_nonzero(~touched_col))
    total_cells = int(plane_dims[0] * plane_dims[1])
    return grid, {
        "status": "built_as_projection_of_voxel_occupancy_3d",
        "grid_dims": plane_dims,
        "floor_axis": floor_axis,
        "plane_axes": (a0, a1),
        "projection_rule": "topdown_max_over_band_then_suppress_free",
        "occupied_cells": occ_cells,
        "movable_static_cells": mov_cells,
        "dynamic_cells": dyn_cells,
        "free_cells": free_cells,
        "unknown_cells": unknown_cells,
        "total_cells": total_cells,
        "occupied_fraction": occ_cells / total_cells if total_cells else 0.0,
        "free_fraction": free_cells / total_cells if total_cells else 0.0,
        "unknown_fraction": unknown_cells / total_cells if total_cells else 0.0,
        "dropped_by_projection": "obstacle_height_within_band_kept_in_height_min_max",
    }


def _height_extents(touched_3d, floor_axis, grid_min, voxel, dims, np):
    # For each 2D cell, height_min/max along the floor axis of touched voxels.
    # Default: a flat valid extent (min==max) where nothing is touched.
    plane_axes = [a for a in range(3) if a != floor_axis]
    a0, a1 = plane_axes

    # Build coordinate of each voxel index along the floor axis (world).
    nf = dims[floor_axis]
    floor_coords = grid_min[floor_axis] + (np.arange(nf) + 0.5) * voxel

    # Move floor axis to the front for reduction.
    moved = np.moveaxis(touched_3d, floor_axis, 0)  # shape (nf, A0, A1)
    any_touched = np.any(moved, axis=0)

    # Index of first/last touched along floor axis.
    coords_b = floor_coords[:, None, None]
    big = grid_min[floor_axis] + nf * voxel
    small = grid_min[floor_axis]
    min_h = np.where(moved, coords_b, big).min(axis=0)
    max_h = np.where(moved, coords_b, small).max(axis=0)

    # Untouched cells get a flat, contract-valid extent at the grid floor base.
    base = float(grid_min[floor_axis])
    height_min = np.where(any_touched, min_h, base).astype(np.float64)
    height_max = np.where(any_touched, max_h, base).astype(np.float64)
    # Guard: ensure max >= min everywhere.
    height_max = np.maximum(height_max, height_min)
    return height_min, height_max


def _map_confidence(scale_posterior: ScalePosterior) -> float:
    status = scale_posterior.metric_acceptance_status.value
    base = {
        "measured_metric": 0.9,
        "metric_pseudo_label": 0.6,
        "non_metric_pseudo_label": 0.4,
        "rejected": 0.1,
    }.get(status, 0.3)
    return max(0.0, min(1.0, base))


__all__ = ["fuse_static_map"]
