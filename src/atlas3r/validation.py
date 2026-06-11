"""M8 validation and acceptance.

Turns the scale posterior plus visibility/map evidence into a contract-valid
``ValidationReport`` and a final honest category.

Acceptance gate (mirrors the contract):
- ``accepted_for_metric_training=True`` requires a metric ScalePosterior status
  (``measured_metric`` or ``metric_pseudo_label``), validation passing, and no
  rejection reasons.
- ``non_metric_pseudo_label`` and ``rejected`` can never be accepted; they must
  carry explicit rejection reasons.

A ``phone_room`` unanchored reconstruction (status ``non_metric_pseudo_label``)
is therefore reported as not-accepted with reasons, never as measured GT.

``numpy`` is imported lazily inside functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ContractValidationError,
    FrameRayPacket,
    MetricAcceptanceStatus,
    ScalePosterior,
    ValidationReport,
)

# Thresholds for accepting a metric posterior.
MAX_CONTRADICTION_RATE_FOR_ACCEPT = 0.25
MAX_HELD_OUT_ERROR_FOR_ACCEPT = 0.50  # log-depth units
HIGH_HELD_OUT_ERROR_DEFAULT = 1.0  # used when not computable
# Fraction of inferred dynamic mass allowed to leak into the fused static surface
# before the metric gate rejects: dynamic is NEVER fused into static occupancy, so
# material residual leakage must block metric-training acceptance.
MAX_DYNAMIC_LEAKAGE_FOR_ACCEPT = 0.10

# Stage 0 evidence-mass thresholds (GT-free acceptance cascade, ARCHITECTURE.md).
# A consistency score over near-zero co-observation is vacuous: held-out error and
# free-space contradiction can read clean simply because the reconstruction barely
# overlaps itself. These thresholds sit inside a measured chasm on the canonical
# scenes (inbounds ratio 0.000/0.035 bad vs 0.605/0.648 good; depth-residual edge
# fraction 0.434/0.537 vs 1.000/1.000; mean confidence weight 0.116/0.187 vs
# 0.532/0.541). Values between the clusters are PROVISIONAL until detection-limit
# calibration assigns them measured authority.
MIN_MEDIAN_INBOUNDS_RATIO_FOR_ACCEPT = 0.30
MIN_DEPTH_RESIDUAL_EDGE_FRACTION_FOR_ACCEPT = 0.70
MIN_MEAN_CONFIDENCE_WEIGHT_FOR_ACCEPT = 0.30

# PROVENANCE-CONDITIONED Stage-2 bounds (calibration table:
# docs/gt_free_verification.md, 2026-06-11). Neither fsc nor the prerefine
# p90 residual ranks label quality ACROSS pose-provenance classes (fsc is
# clutter-confounded; p90 conflates depth difficulty with geometric error),
# but each separates cleanly WITHIN a class. Class membership is GT-free
# (the same BA-grade marker refine's freeze auto-detection uses).
# Class B (backbone/refined poses): fsc keeps its original calibrated 0.25;
# p90 <= 0.35 promotes the long-pending Stage-2b signal at its mid-chasm
# value (good 0.126 vs bad 0.553/0.833 within class) -- on the canonical
# population it only ADDS reasons to already-rejected configs.
# Class A (BA-grade frozen poses): good-cluster max x 1.2 provisional
# margin (fsc 0.461 -> 0.55, p90 0.548 -> 0.66), the vault EXCLUDED from
# calibration and serving as validation (vault_v3 reads fsc 0.607 / p90
# 0.785 -- above both bounds). Provisional until injection detection-limit
# calibration assigns measured authority (the Stage-0 precedent).
MAX_CONTRADICTION_RATE_BA_FROZEN = 0.55
MAX_PREREFINE_P90_BACKBONE = 0.35
MAX_PREREFINE_P90_BA_FROZEN = 0.66


def validate_and_accept(
    asset_id: str,
    packets: Sequence[FrameRayPacket],
    scale_posterior: ScalePosterior,
    visibility_residual_report: Mapping[str, Any],
    map_report: Mapping[str, Any],
    measured_reference: Sequence[FrameRayPacket] | None = None,
    static_dynamic_states: Sequence[Any] | None = None,
    band3d_agreement: Mapping[str, Any] | None = None,
    prerefine_p90_log_depth_residual: float | None = None,
) -> tuple[ValidationReport, str, dict[str, Any]]:
    """Return ``(ValidationReport, final_category, validation_report_dict)``.

    ``band3d_agreement`` is the optional per-voxel agreement of the monocular 3D
    field vs the measured 3D field inside the collision band. It is REPORTAGE: it
    is carried on the ``ValidationReport`` and surfaced in the dict, but it never
    gates ``accepted_for_metric_training`` (the category is driven by the scale
    posterior).
    """
    status = scale_posterior.metric_acceptance_status
    status_value = status.value

    base: dict[str, Any] = {
        "module": "M8 - Validation And Acceptance",
        "asset_id": asset_id,
        "packet_count": len(packets),
        "scale_status": status_value,
    }

    held_out_error, held_out_note = _held_out_render_error(packets, measured_reference)
    contradiction_rate = _free_space_contradiction_rate(map_report)
    floor_wall = _floor_wall_consistency(map_report)
    dynamic_leakage, dynamic_note = _dynamic_leakage_score(
        map_report, static_dynamic_states
    )

    # GT-free acceptance cascade (ARCHITECTURE.md). The cascade applies ONLY to
    # paths without measured evidence: a measured baseline's authority comes from
    # instruments, not internal consistency -- it is the yardstick, not the
    # examinee. Diagnostics are computed (and surfaced) for every path.
    cascade_applies = status is MetricAcceptanceStatus.METRIC_PSEUDO_LABEL
    stage0 = _evidence_mass_stage(visibility_residual_report, cascade_applies)
    stage1 = _gravity_alignment_stage(map_report, cascade_applies)

    # Provenance class (GT-free pipeline fact): BA-grade frozen poses carry
    # the same method marker refine's freeze auto-detection keys on.
    from .refine import BA_GRADE_POSE_MARKERS  # single source of the marker

    ba_grade = any(
        marker in str(p.provenance.get("method", ""))
        for p in packets
        for marker in BA_GRADE_POSE_MARKERS
    )
    fsc_bound = MAX_CONTRADICTION_RATE_BA_FROZEN if ba_grade else MAX_CONTRADICTION_RATE_FOR_ACCEPT
    p90_bound = MAX_PREREFINE_P90_BA_FROZEN if ba_grade else MAX_PREREFINE_P90_BACKBONE
    p90 = prerefine_p90_log_depth_residual
    # Missing p90 on a gated path is a missing signal, never a silent pass.
    stage2b_passes = p90 is not None and p90 <= p90_bound

    validation_passes = (
        held_out_error <= MAX_HELD_OUT_ERROR_FOR_ACCEPT
        and contradiction_rate <= fsc_bound
        and dynamic_leakage <= MAX_DYNAMIC_LEAKAGE_FOR_ACCEPT
        and (stage2b_passes or not cascade_applies)
    )

    rejection_reasons: list[str] = []
    accepted = False
    final_category = status_value

    metric_status = status in {
        MetricAcceptanceStatus.MEASURED_METRIC,
        MetricAcceptanceStatus.METRIC_PSEUDO_LABEL,
    }

    if status is MetricAcceptanceStatus.REJECTED:
        rejection_reasons.append("scale_posterior_rejected_reconstruction_degenerate")
        final_category = "rejected"
    elif status is MetricAcceptanceStatus.NON_METRIC_PSEUDO_LABEL:
        # Honest unanchored reconstruction: not accepted for metric training,
        # but NOT a hard rejection of the reconstruction itself.
        rejection_reasons.append(
            "non_metric_reconstruction_no_measured_scale_evidence_not_metric_training_grade"
        )
        final_category = "non_metric_pseudo_label"
    elif metric_status:
        cascade_passes = stage0["passes"] and stage1["passes"]
        if validation_passes and cascade_passes:
            accepted = True
            final_category = status_value
        else:
            # Metric posterior but validation failed -> downgrade honestly.
            # ALL failing stages contribute reasons (no early-exit): every
            # defect is reported, not just the first one found.
            rejection_reasons.extend(stage0["rejection_reasons"])
            rejection_reasons.extend(stage1["rejection_reasons"])
            if contradiction_rate > fsc_bound:
                rejection_reasons.append(
                    f"free_space_contradiction_rate_too_high:{contradiction_rate:.3f}"
                )
            if cascade_applies and not stage2b_passes:
                rejection_reasons.append(
                    "prerefine_residual_unavailable" if p90 is None else
                    f"prerefine_p90_log_depth_residual_too_high:{p90:.3f}"
                )
            if held_out_error > MAX_HELD_OUT_ERROR_FOR_ACCEPT:
                rejection_reasons.append(
                    f"held_out_render_error_too_high:{held_out_error:.3f}"
                )
            if dynamic_leakage > MAX_DYNAMIC_LEAKAGE_FOR_ACCEPT:
                rejection_reasons.append(
                    f"dynamic_leakage_into_static_too_high:{dynamic_leakage:.3f}"
                )
            accepted = False
            final_category = (
                "metric_pseudo_label"
                if status is MetricAcceptanceStatus.METRIC_PSEUDO_LABEL
                else "measured_metric"
            )

    # The contract forbids accepted reports carrying rejection reasons, and
    # forbids non-accepted reports with empty reasons. Reconcile:
    if accepted:
        rejection_reasons = []
    elif not rejection_reasons:
        rejection_reasons.append("not_accepted_for_metric_training_unspecified")

    try:
        report = ValidationReport(
            held_out_render_error=float(held_out_error),
            free_space_contradiction_rate=float(contradiction_rate),
            scale_posterior=scale_posterior,
            floor_wall_consistency=float(floor_wall),
            dynamic_leakage_score=float(dynamic_leakage),
            accepted_for_metric_training=bool(accepted),
            rejection_reasons=tuple(rejection_reasons),
            band3d_agreement=band3d_agreement,
        )
    except ContractValidationError as exc:
        # If the contract rejects our acceptance combination, fall back to a
        # safe not-accepted report carrying the contract error as a reason.
        report = ValidationReport(
            held_out_render_error=float(held_out_error),
            free_space_contradiction_rate=float(contradiction_rate),
            scale_posterior=scale_posterior,
            floor_wall_consistency=float(floor_wall),
            dynamic_leakage_score=float(dynamic_leakage),
            accepted_for_metric_training=False,
            rejection_reasons=(f"validation_report_contract_reconciled:{exc}",),
            band3d_agreement=band3d_agreement,
        )
        accepted = False

    validation_dict = {
        **base,
        "status": "validated",
        "final_category": final_category,
        "accepted_for_metric_training": bool(report.accepted_for_metric_training),
        "held_out_render_error": float(held_out_error),
        "held_out_render_error_note": held_out_note,
        "free_space_contradiction_rate": float(contradiction_rate),
        "floor_wall_consistency": float(floor_wall),
        "dynamic_leakage_score": float(dynamic_leakage),
        "dynamic_leakage_note": dynamic_note,
        "validation_passes_metric_gate": bool(validation_passes),
        "gate_cascade": {
            "stage0_evidence_mass": {k: v for k, v in stage0.items() if k != "rejection_reasons"},
            "stage1_gravity_alignment": {k: v for k, v in stage1.items() if k != "rejection_reasons"},
            "stage2b_prerefine_residual": {
                "applied": bool(cascade_applies),
                "passes": bool(stage2b_passes),
                "prerefine_p90_log_depth_residual": p90,
                "bound": float(p90_bound),
                "pose_provenance_class": "ba_grade_frozen" if ba_grade else "backbone_refined",
                "fsc_bound_for_class": float(fsc_bound),
                "threshold_authority": (
                    "provisional_good_cluster_margin_pending_injection_calibration"
                    if ba_grade else "mid_chasm_expanded_population_2026_06_11"
                ),
            },
            "applies_to_this_path": bool(cascade_applies),
            "scope_rule": (
                "cascade gates only paths without measured evidence; a measured "
                "baseline is the yardstick, not the examinee"
            ),
        },
        "rejection_reasons": tuple(report.rejection_reasons),
        "band3d_agreement": dict(band3d_agreement) if isinstance(band3d_agreement, Mapping) else band3d_agreement,
        "band3d_agreement_summary": _band3d_summary(band3d_agreement),
        "blockers": tuple(report.rejection_reasons) if not report.accepted_for_metric_training else (),
    }
    return report, final_category, validation_dict


def _band3d_summary(band3d_agreement: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compact headline of the per-voxel band agreement vs the measured 3D field.

    Reportage only -- never gates acceptance. ``None`` / non-computed states are
    surfaced verbatim, never fabricated into numbers.
    """
    if not isinstance(band3d_agreement, Mapping):
        return {"status": "absent"}
    status = band3d_agreement.get("status")
    if status != "computed":
        return {"status": status}
    keys = (
        "per_class_agreement",
        "occupied_static_iou",
        "occupied_recall_any_within_tolerance",
        "free_space_contradiction_rate",
        "dynamic_leakage_rate",
        "coverage_of_measured_band",
        "co_observed_band_voxels",
        "estimated_scale_monocular_to_measured",
    )
    return {"status": status, **{k: band3d_agreement.get(k) for k in keys}}


# ---------------------------------------------------------------------------
# GT-free acceptance cascade stages (ARCHITECTURE.md)
# ---------------------------------------------------------------------------


def _evidence_mass_stage(
    visibility_residual_report: Mapping[str, Any],
    applies: bool,
) -> dict[str, Any]:
    """Stage 0: does the reconstruction carry enough co-observation evidence for
    any consistency score to mean anything?

    Reads the visibility graph's overall statistics (already computed by M4,
    previously unread by the gate). Low evidence mass rejects regardless of how
    clean the consistency scores look -- a held-out error over near-zero overlap
    is vacuous, not reassuring. Missing statistics count as zero evidence (never
    silently pass), matching "unknown is not free" at the gate level.
    """
    overall = (
        visibility_residual_report.get("overall")
        if isinstance(visibility_residual_report, Mapping)
        else None
    )
    overall = overall if isinstance(overall, Mapping) else {}

    def _ratio(key: str) -> float:
        value = overall.get(key)
        if isinstance(value, (int, float)) and value == value:
            return float(value)
        return 0.0

    inbounds_ratio = _ratio("median_reprojection_inbounds_ratio")
    confidence_weight = _ratio("mean_confidence_weight")
    edge_count = _ratio("edge_count")
    edges_with_residual = _ratio("edges_with_depth_residual")
    edge_fraction = (edges_with_residual / edge_count) if edge_count > 0 else 0.0

    reasons: list[str] = []
    if inbounds_ratio < MIN_MEDIAN_INBOUNDS_RATIO_FOR_ACCEPT:
        reasons.append(
            f"evidence_mass_median_inbounds_ratio_too_low:{inbounds_ratio:.3f}"
        )
    if edge_fraction < MIN_DEPTH_RESIDUAL_EDGE_FRACTION_FOR_ACCEPT:
        reasons.append(
            f"evidence_mass_depth_residual_edge_fraction_too_low:{edge_fraction:.3f}"
        )
    if confidence_weight < MIN_MEAN_CONFIDENCE_WEIGHT_FOR_ACCEPT:
        reasons.append(
            f"evidence_mass_mean_confidence_weight_too_low:{confidence_weight:.3f}"
        )

    return {
        "passes": not reasons if applies else True,
        "applied": bool(applies),
        "median_reprojection_inbounds_ratio": inbounds_ratio,
        "depth_residual_edge_fraction": edge_fraction,
        "mean_confidence_weight": confidence_weight,
        "thresholds": {
            "min_median_inbounds_ratio": MIN_MEDIAN_INBOUNDS_RATIO_FOR_ACCEPT,
            "min_depth_residual_edge_fraction": MIN_DEPTH_RESIDUAL_EDGE_FRACTION_FOR_ACCEPT,
            "min_mean_confidence_weight": MIN_MEAN_CONFIDENCE_WEIGHT_FOR_ACCEPT,
        },
        "threshold_authority": (
            "provisional_mid_chasm_pending_detection_limit_calibration"
        ),
        "would_reject": bool(reasons),
        "rejection_reasons": reasons if applies else [],
    }


def _gravity_alignment_stage(
    map_report: Mapping[str, Any],
    applies: bool,
) -> dict[str, Any]:
    """Stage 1: is the collision band actually floor-aligned?

    Multiview geometry is gauge-blind to global tilt; only the gravity/floor
    prior covers it. The fuser already measures floor reliability and refuses to
    fabricate an alignment (``up_alignment_applied=False`` with a loud blocker)
    -- this stage makes the acceptance gate READ that verdict: a band that is not
    floor-parallel is a broken training label for a floor-band robot, whatever
    the consistency scores say.
    """
    floor = map_report.get("floor") if isinstance(map_report, Mapping) else None
    floor = floor if isinstance(floor, Mapping) else {}
    up_applied = bool(floor.get("up_alignment_applied", False))
    inlier = floor.get("inlier_ratio")
    inlier_ratio = float(inlier) if isinstance(inlier, (int, float)) and inlier == inlier else 0.0
    method = str(floor.get("method", "absent"))

    reasons: list[str] = []
    if not up_applied:
        reasons.append(
            f"gravity_alignment_unverified_band_not_floor_aligned_inlier:{inlier_ratio:.3f}"
        )

    return {
        "passes": not reasons if applies else True,
        "applied": bool(applies),
        "up_alignment_applied": up_applied,
        "floor_inlier_ratio": inlier_ratio,
        "floor_method": method,
        "would_reject": bool(reasons),
        "rejection_reasons": reasons if applies else [],
    }


# ---------------------------------------------------------------------------
# metric computations
# ---------------------------------------------------------------------------


def _held_out_render_error(
    packets: Sequence[FrameRayPacket],
    measured_reference: Sequence[FrameRayPacket] | None,
) -> tuple[float, str]:
    """Hold out one packet, reproject its surface into a neighbour, and compare
    the reprojected (predicted) depth against the neighbour's actually observed
    per-pixel depth at the nearest sampled pixel, in log space.

    The observed depth is the neighbour's own ``radial_depth_m`` at the sampled
    pixel closest to the reprojected pixel -- a real per-pixel observation, not a
    scene-wide median. When the reprojected pixel has no nearby neighbour
    observation (no real surface to compare against), that sample is skipped
    rather than compared against a fabricated target.

    Returns ``(error, note)``. When not computable, returns a high default with
    an explicit note (never a fabricated low error).
    """
    if len(packets) < 2:
        return HIGH_HELD_OUT_ERROR_DEFAULT, "not_computable_fewer_than_two_packets"

    import numpy as np  # type: ignore

    ordered = sorted(packets, key=lambda p: p.frame_id)
    held = ordered[len(ordered) // 2]
    # Compare held-out frame against its nearest neighbour by frame_id.
    others = [p for p in ordered if p.frame_id != held.frame_id]
    neighbour = min(others, key=lambda p: abs(p.frame_id - held.frame_id))

    held_rays = np.asarray(held.rays_camera, dtype=np.float64).reshape((-1, 3))
    held_depth = np.asarray(held.radial_depth_m, dtype=np.float64).reshape((-1,))
    Th = np.asarray(held.T_world_camera, dtype=np.float64).reshape((4, 4))
    Rh, th = Th[:3, :3], Th[:3, 3]

    Tn = np.asarray(neighbour.T_world_camera, dtype=np.float64).reshape((4, 4))
    Rn, tn = Tn[:3, :3], Tn[:3, 3]
    neighbour_rays = np.asarray(neighbour.rays_camera, dtype=np.float64).reshape((-1, 3))
    neighbour_depth = np.asarray(neighbour.radial_depth_m, dtype=np.float64).reshape((-1,))
    if neighbour_depth.size == 0:
        return HIGH_HELD_OUT_ERROR_DEFAULT, "not_computable_neighbour_has_no_depth"

    # Recover the neighbour's sampled pixel coordinates from its own rays so we
    # can look up the real observed depth at the reprojected pixel.
    obs_pixels = _project_rays_to_pixels(neighbour, neighbour_rays, np)
    if obs_pixels is None or obs_pixels.shape[0] == 0:
        return HIGH_HELD_OUT_ERROR_DEFAULT, "not_computable_neighbour_pixels_unrecoverable"
    # Match nearest observed pixel only within a tolerance so we compare against
    # a genuine local observation, not an arbitrarily distant one.
    width = float(getattr(neighbour.camera_model, "width_px", 0.0)) or 1.0
    match_tol_px = max(2.0, 0.02 * width)

    # Lift held-out surface to world, into neighbour camera, reproject.
    n = held_rays.shape[0]
    cap = min(n, 512)
    idx = np.unique(np.rint(np.linspace(0, n - 1, num=cap)).astype(np.int64))
    X_cam = held_rays[idx] * held_depth[idx][:, None]
    X_world = X_cam @ Rh.T + th[None, :]
    X_cam_n = (X_world - tn[None, :]) @ Rn

    residuals = []
    no_observation = 0
    for k in range(X_cam_n.shape[0]):
        result = neighbour.camera_model.project(
            (float(X_cam_n[k, 0]), float(X_cam_n[k, 1]), float(X_cam_n[k, 2]))
        )
        if not result.get("valid"):
            continue
        d_pred = result.get("radial_depth_m")
        if d_pred is None or not (d_pred > 0.0):
            continue
        pixel = result.get("pixel_uv")
        if pixel is None:
            continue
        # Nearest neighbour-observed sampled pixel to this reprojected pixel.
        du = obs_pixels[:, 0] - float(pixel[0])
        dv = obs_pixels[:, 1] - float(pixel[1])
        dist2 = du * du + dv * dv
        j = int(np.argmin(dist2))
        if float(dist2[j]) > match_tol_px * match_tol_px:
            no_observation += 1
            continue
        d_obs = float(neighbour_depth[j])
        if not (d_obs > 0.0):
            continue
        residuals.append(abs(_safe_log(d_pred) - _safe_log(d_obs)))

    if not residuals:
        return HIGH_HELD_OUT_ERROR_DEFAULT, (
            "not_computable_no_overlapping_reprojection_with_observed_depth"
            f"_skipped_{no_observation}_unobserved"
        )
    return (
        float(np.median(residuals)),
        f"log_depth_render_error_median_over_{len(residuals)}_pixels_vs_observed_depth"
        f"_skipped_{no_observation}_unobserved",
    )


def _project_rays_to_pixels(packet: FrameRayPacket, rays: Any, np: Any) -> Any | None:
    """Recover sampled pixel coordinates for each ray via the camera model.

    Each ray is a unit camera-frame direction; projecting a point along it (the
    ray itself) recovers the pixel it was sampled from. Returns an ``(N, 2)``
    array of ``(u, v)`` or ``None`` if no ray projects in front of the camera.
    """
    pixels = []
    for k in range(rays.shape[0]):
        result = packet.camera_model.project(
            (float(rays[k, 0]), float(rays[k, 1]), float(rays[k, 2]))
        )
        pixel = result.get("pixel_uv")
        if pixel is None:
            pixels.append((float("nan"), float("nan")))
        else:
            pixels.append((float(pixel[0]), float(pixel[1])))
    arr = np.asarray(pixels, dtype=np.float64)
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    return arr if arr.shape[0] > 0 else None


def _free_space_contradiction_rate(map_report: Mapping[str, Any]) -> float:
    """Fraction of occupied evidence that conflicts with observed free space.

    This reads the contradiction rate actually measured by the fuser from the
    ``VoxelMapState`` (voxels with both free-carve and surface evidence, over
    occupied voxels). It is NOT a hardcoded constant: a real cross-frame
    conflict, where one frame's surface voxel is another frame's carved-free
    voxel, is counted and surfaced as ``free_space_contradiction_rate`` in the
    map report. When no map was fused, contradiction is unknown -> reported as a
    conservative mid value so it cannot silently pass the gate.
    """
    if map_report.get("status") != "fused":
        return 0.5
    rate = map_report.get("free_space_contradiction_rate")
    if not isinstance(rate, (int, float)) or rate != rate:  # reject None/NaN
        # Map fused but no measured contradiction rate -> do not silently pass.
        return 0.5
    return float(max(0.0, min(1.0, float(rate))))


def _dynamic_leakage_score(
    map_report: Mapping[str, Any],
    static_dynamic_states: Sequence[Any] | None,
) -> tuple[float, str]:
    """Real measure of dynamic geometry leaking into the fused static surface.

    Dynamic is NEVER fused into static occupancy (ARCHITECTURE.md). M5 inference
    produces a per-pixel dynamic probability; M7 fusion EXCLUDES dynamic-dominant
    pixels via ``static_dynamic_states`` -- but only for frames whose state aligns
    pixel-for-pixel with the packet rays. Frames whose state shape does not match
    are fused UNFILTERED, so any dynamic pixels in those frames leak into static.

    The leakage score is the fraction of inferred dynamic-dominant samples that
    were NOT excluded from static fusion:

        leakage = dynamic_in_unfiltered_frames / total_dynamic_dominant_samples

    Sources, in order of fidelity:
    1. The fuser's own ``dynamic_exclusion`` block in ``map_report`` (the ground
       truth of what fusion actually carved out): if it reports any frames fused
       unfiltered while dynamic samples exist, that is real residual leakage.
    2. The M5 ``static_dynamic_states`` totals, used to size the dynamic mass and
       confirm dynamic inference ran.

    Returns ``(score, note)``. The note never claims "no dynamic inference" when
    inference actually ran; it reports what was excluded vs what leaked.
    """
    dyn_excl = map_report.get("dynamic_exclusion") if isinstance(map_report, Mapping) else None

    # Count inferred dynamic-dominant samples from the M5 states (if provided).
    total_dynamic_samples = 0
    total_samples = 0
    states_present = False
    for state in static_dynamic_states or ():
        states_present = True
        summary = getattr(state, "residual_summary", None)
        if isinstance(summary, Mapping):
            total_dynamic_samples += int(summary.get("dynamic_count", 0) or 0)
            total_samples += int(summary.get("sample_count", 0) or 0)

    if not isinstance(dyn_excl, Mapping):
        # Map was not fused with a dynamic-exclusion pass at all.
        if states_present and total_dynamic_samples > 0:
            # Dynamic WAS inferred but the fuser carried no exclusion record; we
            # cannot prove exclusion happened -> treat the inferred dynamic mass
            # as potentially leaked (conservative, never an optimistic zero).
            score = min(1.0, total_dynamic_samples / max(total_samples, 1))
            return score, (
                "dynamic_inferred_but_no_fusion_exclusion_record"
                f"_dynamic_samples_{total_dynamic_samples}_of_{total_samples}"
                "_treated_as_potential_leakage"
            )
        return 0.0, "no_dynamic_exclusion_record_and_no_dynamic_inference"

    applied = bool(dyn_excl.get("applied", False))
    frames_filtered = int(dyn_excl.get("frames_filtered", 0) or 0)
    frames_mismatch = int(dyn_excl.get("frames_state_shape_mismatch", 0) or 0)
    excluded = int(dyn_excl.get("dynamic_pixels_excluded", 0) or 0)

    if not applied and frames_mismatch == 0:
        # No static/dynamic state reached the fuser at all.
        if states_present and total_dynamic_samples > 0:
            score = min(1.0, total_dynamic_samples / max(total_samples, 1))
            return score, (
                "dynamic_inferred_but_no_state_applied_in_fusion"
                f"_dynamic_samples_{total_dynamic_samples}_of_{total_samples}"
                "_treated_as_potential_leakage"
            )
        return 0.0, "no_static_dynamic_state_reached_fusion_no_dynamic_inferred"

    # Dynamic inference ran and fusion saw the states. Excluded pixels are carved
    # out (no leakage). Residual leakage comes from frames fused UNFILTERED while
    # a state existed but did not align pixel-for-pixel: their dynamic-dominant
    # samples survived into the static surface.
    if frames_mismatch == 0:
        # Every state-bearing frame was filtered: dynamic-dominant pixels carved
        # out. No structural leakage path remains.
        return 0.0, (
            "dynamic_inferred_and_excluded_from_static_fusion"
            f"_excluded_{excluded}_pixels_over_{frames_filtered}_frames"
            "_zero_unfiltered_frames_no_leakage"
        )

    # Some frames were fused unfiltered. Estimate the leaked dynamic mass as the
    # mean per-frame dynamic rate (from inferred states) times the unfiltered
    # frame count, over the total inferred dynamic mass. When no per-state mass is
    # available, fall back to the unfiltered-frame fraction as a conservative
    # upper bound on leakage.
    total_state_frames = frames_filtered + frames_mismatch
    if total_dynamic_samples > 0 and total_samples > 0 and total_state_frames > 0:
        mean_dynamic_per_frame = total_dynamic_samples / total_state_frames
        leaked = mean_dynamic_per_frame * frames_mismatch
        score = min(1.0, leaked / total_dynamic_samples) if total_dynamic_samples else 0.0
    else:
        score = min(1.0, frames_mismatch / max(total_state_frames, 1))
    return float(max(0.0, min(1.0, score))), (
        "dynamic_inferred_but_"
        f"{frames_mismatch}_of_{total_state_frames}_frames_fused_unfiltered"
        f"_shape_mismatch_excluded_{excluded}_pixels_residual_dynamic_may_leak"
    )


def _floor_wall_consistency(map_report: Mapping[str, Any]) -> float:
    floor = map_report.get("floor")
    if not isinstance(floor, Mapping):
        return 0.0
    inlier = floor.get("inlier_ratio")
    if isinstance(inlier, (int, float)):
        return max(0.0, min(1.0, float(inlier)))
    return 0.0


def _safe_log(value: float) -> float:
    import math

    return math.log(max(value, 1e-9))


__all__ = ["validate_and_accept"]
