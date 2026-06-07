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
    ContractValidationError,
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
        # pseudo-label only. Uncertainty driven by evidence confidence.
        relative = _soft_relative_uncertainty(evidence)
        posterior = _build_posterior(
            scale_mean=1.0,
            scale_std=relative,
            sources=evidence,
            anchor_residuals=_anchor_residuals(evidence),
            status=MetricAcceptanceStatus.METRIC_PSEUDO_LABEL,
        )
        return posterior, {
            **base_report,
            "status": "metric_pseudo_label",
            "decision": "soft_scale_evidence_present",
            "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
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
