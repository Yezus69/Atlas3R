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


def validate_and_accept(
    asset_id: str,
    packets: Sequence[FrameRayPacket],
    scale_posterior: ScalePosterior,
    visibility_residual_report: Mapping[str, Any],
    map_report: Mapping[str, Any],
    measured_reference: Sequence[FrameRayPacket] | None = None,
) -> tuple[ValidationReport, str, dict[str, Any]]:
    """Return ``(ValidationReport, final_category, validation_report_dict)``."""
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
    dynamic_leakage = 0.0  # no dynamic inference -> no measured leakage
    dynamic_note = "no_dynamic_inference_dynamic_leakage_zero_by_construction"

    validation_passes = (
        held_out_error <= MAX_HELD_OUT_ERROR_FOR_ACCEPT
        and contradiction_rate <= MAX_CONTRADICTION_RATE_FOR_ACCEPT
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
        if validation_passes:
            accepted = True
            final_category = status_value
        else:
            # Metric posterior but validation failed -> downgrade honestly.
            if contradiction_rate > MAX_CONTRADICTION_RATE_FOR_ACCEPT:
                rejection_reasons.append(
                    f"free_space_contradiction_rate_too_high:{contradiction_rate:.3f}"
                )
            if held_out_error > MAX_HELD_OUT_ERROR_FOR_ACCEPT:
                rejection_reasons.append(
                    f"held_out_render_error_too_high:{held_out_error:.3f}"
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
        "rejection_reasons": tuple(report.rejection_reasons),
        "blockers": tuple(report.rejection_reasons) if not report.accepted_for_metric_training else (),
    }
    return report, final_category, validation_dict


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
