"""M5 scale posterior and metric gate.

Classifies a reconstruction's global scale into one of the four honest metric
categories using only the available scale evidence. It never invents a measured
metric anchor: ``measured=True`` evidence is required for ``measured_metric``,
soft/learned evidence backs at most ``metric_pseudo_label``, and a good but
unanchored monocular reconstruction is ``non_metric_pseudo_label`` (not
rejected merely for lacking scale).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    FrameRayPacket,
    MetricAcceptanceStatus,
    ScaleEvidence,
    ScalePosterior,
)

# Minimum packets for a non-degenerate reconstruction worth classifying.
MIN_PACKETS_FOR_RECONSTRUCTION = 2
# Unanchored monocular reconstruction carries a large scale band.
NON_METRIC_RELATIVE_UNCERTAINTY = 0.30
# Measured-metric reconstruction is already in meters; tiny residual band.
MEASURED_METRIC_SCALE_STD = 0.005
# Soft-evidence pseudo-label band.
METRIC_PSEUDO_RELATIVE_UNCERTAINTY = 0.12


def estimate_scale_posterior(
    packets: Sequence[FrameRayPacket],
    scale_evidence: Sequence[ScaleEvidence],
    validation_inputs: Mapping[str, Any] | None = None,
    has_measured: bool | None = None,
) -> tuple[ScalePosterior, dict[str, Any]]:
    """Return ``(ScalePosterior, scale_report)``.

    ``has_measured`` may be passed explicitly; otherwise it is derived from the
    presence of a ``measured=True`` ScaleEvidence source.
    """
    evidence = tuple(scale_evidence)
    measured_sources = tuple(e for e in evidence if getattr(e, "measured", False))
    measured_present = bool(measured_sources) if has_measured is None else bool(has_measured and measured_sources)

    packet_count = len(packets)
    base_report: dict[str, Any] = {
        "module": "M5 - Scale Posterior And Metric Gate",
        "packet_count": packet_count,
        "scale_evidence_count": len(evidence),
        "measured_evidence_count": len(measured_sources),
    }

    # Degenerate / too-few-packets => rejected (the only legitimate rejection
    # here is reconstruction weakness, NOT missing scale).
    if packet_count < MIN_PACKETS_FOR_RECONSTRUCTION:
        posterior = _build_posterior(
            scale_mean=1.0,
            scale_std=NON_METRIC_RELATIVE_UNCERTAINTY,
            sources=evidence,
            anchor_residuals=_anchor_residuals(evidence),
            status=MetricAcceptanceStatus.REJECTED,
        )
        return posterior, {
            **base_report,
            "status": "rejected",
            "decision": "too_few_packets_for_reconstruction",
            "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
            "blockers": ("reconstruction_degenerate_too_few_packets",),
        }

    if measured_present and _packets_are_measured(packets):
        scale_std = MEASURED_METRIC_SCALE_STD
        posterior = _build_posterior(
            scale_mean=1.0,
            scale_std=scale_std,
            sources=evidence,
            anchor_residuals=_anchor_residuals(evidence),
            status=MetricAcceptanceStatus.MEASURED_METRIC,
        )
        return posterior, {
            **base_report,
            "status": "measured_metric",
            "decision": "measured_evidence_and_measured_packets",
            "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
            "blockers": (),
        }

    if evidence:
        # Soft (or measured-but-non-measured-packet) evidence supports a metric
        # pseudo-label only. When the metric anchor council provides an explicit
        # scale estimate + uncertainty, use that distribution directly; older
        # learned-prior evidence still falls back to the confidence band.
        scale_mean, band, explicit_report = _soft_scale_distribution(evidence)
        residual_norm, scale_report_extra = _scale_borrow_residual_norm(
            evidence, packets
        )
        relative = float((band**2 + residual_norm**2) ** 0.5)
        posterior = _build_posterior(
            scale_mean=scale_mean,
            scale_std=scale_mean * relative,
            sources=evidence,
            anchor_residuals=_anchor_residuals(evidence),
            status=MetricAcceptanceStatus.METRIC_PSEUDO_LABEL,
        )
        return posterior, {
            **base_report,
            "status": "metric_pseudo_label",
            "decision": "soft_scale_evidence_present",
            "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
            "scale_std_construction": {
                "soft_scale_distribution": explicit_report,
                "evidence_relative_uncertainty": band,
                "scale_borrow_residual_norm": residual_norm,
                "rule": "sqrt(band^2 + residual_norm^2)",
                **scale_report_extra,
            },
            "blockers": (),
        }

    # No scale evidence at all: honest unanchored monocular reconstruction.
    posterior = _build_posterior(
        scale_mean=1.0,
        scale_std=NON_METRIC_RELATIVE_UNCERTAINTY,
        sources=(),
        anchor_residuals=(),
        status=MetricAcceptanceStatus.NON_METRIC_PSEUDO_LABEL,
    )
    return posterior, {
        **base_report,
        "status": "non_metric_pseudo_label",
        "decision": "no_scale_evidence_unanchored_reconstruction",
        "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
        "scale_units": "reconstruction_units_unitless_similarity",
        "blockers": (),
    }


def _build_posterior(
    *,
    scale_mean: float,
    scale_std: float,
    sources: Sequence[ScaleEvidence],
    anchor_residuals: Sequence[float],
    status: MetricAcceptanceStatus,
) -> ScalePosterior:
    relative = float(scale_std) / float(scale_mean)
    return ScalePosterior(
        scale_mean=float(scale_mean),
        scale_std=float(scale_std),
        relative_scale_uncertainty=relative,
        scale_sources=tuple(sources),
        anchor_residuals=tuple(anchor_residuals),
        metric_acceptance_status=status,
    )


def _packets_are_measured(packets: Sequence[FrameRayPacket]) -> bool:
    # A measured reconstruction has packets whose source declares measured
    # reference provenance. We do not infer "measured" from anything softer.
    for packet in packets:
        source = str(getattr(packet, "source", ""))
        if source.startswith("measured_reference"):
            return True
    return False


def _soft_relative_uncertainty(evidence: Sequence[ScaleEvidence]) -> float:
    confidences = [float(getattr(e, "confidence", 0.0)) for e in evidence]
    if not confidences:
        return METRIC_PSEUDO_RELATIVE_UNCERTAINTY
    mean_conf = sum(confidences) / len(confidences)
    # Higher confidence -> tighter band, floored to keep it a pseudo-label.
    band = METRIC_PSEUDO_RELATIVE_UNCERTAINTY * (1.0 + (1.0 - mean_conf))
    return max(0.02, band)


def _soft_scale_distribution(
    evidence: Sequence[ScaleEvidence],
) -> tuple[float, float, dict[str, Any]]:
    """Return ``(scale_mean, relative_uncertainty, report)`` for soft evidence.

    Council-produced evidence carries explicit ``scale_mean`` and ``scale_std``.
    That is the real anchor distribution and should not be flattened back into a
    fixed learned-prior confidence band. Non-council legacy evidence keeps the
    old conservative confidence-derived band around scale 1.
    """
    explicit = []
    for e in evidence:
        mean = getattr(e, "scale_mean", None)
        std = getattr(e, "scale_std", None)
        if mean is None or std is None:
            continue
        try:
            mean_f = float(mean)
            std_f = float(std)
        except (TypeError, ValueError):
            continue
        if mean_f > 0.0 and std_f > 0.0:
            explicit.append((mean_f, std_f, e))
    if not explicit:
        band = _soft_relative_uncertainty(evidence)
        return 1.0, band, {
            "basis": "legacy_soft_evidence_confidence_band",
            "scale_mean": 1.0,
            "relative_uncertainty": band,
        }

    import math

    logs = sorted(math.log(item[0]) for item in explicit)
    mid = len(logs) // 2
    if len(logs) % 2:
        scale_mean = math.exp(logs[mid])
    else:
        scale_mean = math.exp(0.5 * (logs[mid - 1] + logs[mid]))
    rels = [item[1] / item[0] for item in explicit]
    anchor_disagreement = max(abs(math.log(item[0] / scale_mean)) for item in explicit)
    relative = max(max(rels), anchor_disagreement, 0.02)
    return float(scale_mean), float(relative), {
        "basis": "explicit_scale_evidence_distribution",
        "scale_mean": float(scale_mean),
        "relative_uncertainty": float(relative),
        "anchor_disagreement_max_abs_log": float(anchor_disagreement),
        "evidence_ids": tuple(e.evidence_id for _, _, e in explicit),
    }


def _scale_borrow_residual_norm(
    evidence: Sequence[ScaleEvidence],
    packets: Sequence[FrameRayPacket],
) -> tuple[float, dict[str, Any]]:
    """Scene-relative scale-borrow misfit (Phase 19): the largest
    ``residual_after_optimization`` carried by the soft evidence (the Umeyama
    RMSE between BA camera centers and the backbone centers the borrowed
    scalar was fit on, backbone units) divided by the trajectory span
    (bounding-box diagonal of packet camera centers, same units). Both
    candidate-only; dimensionless. Returns ``(0.0, {...})`` when no evidence
    carries a residual (class-B backbone poses -- a prior, not a measurement).
    """
    residuals = _anchor_residuals(evidence)
    if not residuals or len(packets) < 2:
        return 0.0, {"basis": "no_scale_borrow_residual_recorded"}
    import numpy as np  # lazy

    centers = np.asarray(
        [
            np.asarray(p.T_world_camera, dtype=np.float64).reshape(4, 4)[:3, 3]
            for p in packets
        ]
    )
    span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    if span <= 1e-9:
        return 0.0, {"basis": "degenerate_zero_trajectory_span"}
    residual = max(residuals)
    return float(residual / span), {
        "basis": "max_evidence_residual_over_trajectory_bbox_diagonal",
        "residual_backbone_units": float(residual),
        "trajectory_span_backbone_units": span,
    }


def _anchor_residuals(evidence: Sequence[ScaleEvidence]) -> tuple[float, ...]:
    residuals: list[float] = []
    for e in evidence:
        value = getattr(e, "residual_after_optimization", None)
        if value is not None:
            try:
                residuals.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                continue
    return tuple(residuals)


__all__ = ["estimate_scale_posterior"]
