"""Camera, intrinsics, and scale ledgers for offline traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from atlas3r.contracts.frames import CameraModel
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.proposal_cache import DebugGeometryMode, ProposalCacheResult
from atlas3r.offline.run_manifest import write_json

ScaleMode = Literal["unanchored-soft-metric"]
_SOFT_METRIC_SOURCE = "depth_pro_vggt_soft_metric_prior"


@dataclass(frozen=True)
class CameraScaleLedgerResult:
    camera_ledger_path: str
    scale_ledger_path: str
    scale_hypotheses_path: str
    soft_metric_scale_ledger_path: str
    intrinsics_status: str
    scale_source: str
    selected_scale_mode: str
    scale_confidence: str
    physical_accuracy_allowed: bool


def write_camera_scale_ledgers(
    run_dir: str | Path,
    *,
    camera: CameraModel | None,
    frame_count: int,
    debug_geometry_mode: DebugGeometryMode,
    proposal_cache: ProposalCacheResult | None = None,
    disagreement: DisagreementResult | None = None,
    scale_mode: ScaleMode = "unanchored-soft-metric",
) -> CameraScaleLedgerResult:
    root = Path(run_dir)
    has_vggt = bool(proposal_cache is not None and proposal_cache.vggt_camera_records)
    has_depth_pro = bool(proposal_cache is not None and proposal_cache.depth_pro_camera_records)
    intrinsics_status = (
        "proposed" if has_vggt or has_depth_pro else "guessed" if camera is not None else "unknown"
    )
    scale_source = _scale_source(
        frame_count, debug_geometry_mode, has_vggt=has_vggt, has_depth_pro=has_depth_pro
    )
    physical_accuracy_allowed = False
    proposal_sources = []
    if has_vggt:
        proposal_sources.append("vggt")
    if has_depth_pro:
        proposal_sources.append("depth_pro")
    scale_proposal_sources = []
    if has_vggt:
        scale_proposal_sources.append("vggt_unanchored_metric_proposal")
    if has_depth_pro:
        scale_proposal_sources.append("depth_pro_metric_proposal_unanchored")
    soft_metric = _soft_metric_payload(
        frame_metadata_summary={},
        proposal_cache=proposal_cache,
        disagreement=disagreement,
        cross_view_metrics=None,
        scale_mode=scale_mode,
    )
    camera_payload = {
        "status": "partial" if camera is not None or has_vggt or has_depth_pro else "unavailable",
        "intrinsics_status": intrinsics_status,
        "camera": None if camera is None else camera.to_dict(),
        "proposal_source": proposal_sources[0] if len(proposal_sources) == 1 else None,
        "proposal_sources": proposal_sources,
        "vggt_intrinsics_proposals": [
            {
                "frame_id": record["frame_id"],
                "keyframe_index": record["keyframe_index"],
                "K": record["K"],
                "intrinsics_source": "vggt",
                "truth_boundary": record["truth_boundary"],
            }
            for record in (proposal_cache.vggt_camera_records if proposal_cache is not None else ())
        ],
        "depth_pro_intrinsics_proposals": [
            {
                "frame_id": record["frame_id"],
                "keyframe_index": record["keyframe_index"],
                "K": record.get("K"),
                "focal_px": record.get("focal_px"),
                "fx": record.get("fx"),
                "fy": record.get("fy"),
                "intrinsics_source": "depth_pro",
                "truth_boundary": record["truth_boundary"],
            }
            for record in (
                proposal_cache.depth_pro_camera_records if proposal_cache is not None else ()
            )
        ],
        "why": (
            "intrinsics proposed by teacher witnesses; not calibrated or measured"
            if has_vggt or has_depth_pro
            else "intrinsics guessed from image size; no calibration metadata"
            if camera is not None
            else "no frames decoded, so no intrinsics are known"
        ),
    }
    scale_payload = {
        "status": "partial",
        "scale_source": _SOFT_METRIC_SOURCE
        if scale_mode == "unanchored-soft-metric"
        else scale_source,
        "legacy_scale_source": scale_source,
        "anchoring_state": "unanchored" if scale_source != "synthetic_known" else "synthetic",
        "scale_status": "soft_metric_unanchored",
        "selected_scale_mode": "unanchored_soft_metric",
        "metric_scale_source": _SOFT_METRIC_SOURCE,
        "scale_confidence": soft_metric["scale_confidence"],
        "physical_accuracy_allowed": physical_accuracy_allowed,
        "claim_blocker": (
            "no measured scale anchor or named evaluation report is present"
            if not physical_accuracy_allowed
            else None
        ),
        "measured_geometry": False,
        "accuracy_report": False,
        "proposal_source": proposal_sources[0] if len(proposal_sources) == 1 else None,
        "proposal_sources": proposal_sources,
        "scale_proposal_sources": scale_proposal_sources,
        "allowed_sources": [
            "unknown",
            "unanchored_rgb_prior",
            "debug_flat_depth",
            "synthetic_known",
            "vggt_unanchored_metric_proposal",
            "depth_pro_metric_proposal_unanchored",
            "multiple_unanchored_teacher_proposals",
            "future_anchor",
            "measured",
        ],
    }
    write_json(root / "world" / "camera_ledger.json", camera_payload)
    write_json(root / "world" / "scale_ledger.json", scale_payload)
    write_json(root / "world" / "scale_hypotheses.json", soft_metric["scale_hypotheses"])
    write_json(root / "world" / "soft_metric_scale_ledger.json", soft_metric)
    return CameraScaleLedgerResult(
        camera_ledger_path="world/camera_ledger.json",
        scale_ledger_path="world/scale_ledger.json",
        scale_hypotheses_path="world/scale_hypotheses.json",
        soft_metric_scale_ledger_path="world/soft_metric_scale_ledger.json",
        intrinsics_status=intrinsics_status,
        scale_source=_SOFT_METRIC_SOURCE
        if scale_mode == "unanchored-soft-metric"
        else scale_source,
        selected_scale_mode="unanchored_soft_metric",
        scale_confidence=str(soft_metric["scale_confidence"]),
        physical_accuracy_allowed=physical_accuracy_allowed,
    )


def update_soft_metric_scale_ledgers(
    run_dir: str | Path,
    *,
    frame_metadata_summary: dict[str, object],
    proposal_cache: ProposalCacheResult | None,
    disagreement: DisagreementResult | None,
    cross_view_metrics: dict[str, object] | None,
    scale_mode: ScaleMode = "unanchored-soft-metric",
) -> dict[str, object]:
    root = Path(run_dir)
    payload = _soft_metric_payload(
        frame_metadata_summary=frame_metadata_summary,
        proposal_cache=proposal_cache,
        disagreement=disagreement,
        cross_view_metrics=cross_view_metrics,
        scale_mode=scale_mode,
    )
    write_json(root / "world" / "scale_hypotheses.json", payload["scale_hypotheses"])
    write_json(root / "world" / "soft_metric_scale_ledger.json", payload)
    scale_ledger_path = root / "world" / "scale_ledger.json"
    if scale_ledger_path.is_file():
        scale_payload = json.loads(scale_ledger_path.read_text(encoding="utf-8"))
        scale_payload.update(
            {
                "scale_source": _SOFT_METRIC_SOURCE,
                "metric_scale_source": _SOFT_METRIC_SOURCE,
                "scale_status": "soft_metric_unanchored",
                "selected_scale_mode": payload["selected_scale_mode"],
                "scale_confidence": payload["scale_confidence"],
                "physical_accuracy_allowed": False,
            }
        )
        write_json(scale_ledger_path, scale_payload)
    return payload


def _scale_source(
    frame_count: int,
    debug_geometry_mode: DebugGeometryMode,
    *,
    has_vggt: bool,
    has_depth_pro: bool,
) -> str:
    if has_vggt and has_depth_pro:
        return "multiple_unanchored_teacher_proposals"
    if has_vggt:
        return "vggt_unanchored_metric_proposal"
    if has_depth_pro:
        return "depth_pro_metric_proposal_unanchored"
    if debug_geometry_mode == "flat-depth":
        return "debug_flat_depth"
    if debug_geometry_mode == "synthetic-known":
        return "synthetic_known"
    if frame_count > 0:
        return "unanchored_rgb_prior"
    return "unknown"


def _soft_metric_payload(
    *,
    frame_metadata_summary: dict[str, object],
    proposal_cache: ProposalCacheResult | None,
    disagreement: DisagreementResult | None,
    cross_view_metrics: dict[str, object] | None,
    scale_mode: ScaleMode,
) -> dict[str, object]:
    depth_pro_available = bool(
        proposal_cache is not None and proposal_cache.depth_pro_depth_records
    )
    vggt_available = bool(proposal_cache is not None and proposal_cache.vggt_depth_records)
    exif_focal_available = bool(
        frame_metadata_summary.get("focal_mm_values")
        or frame_metadata_summary.get("focal_35mm_values")
    )
    depth_pro_focal_available = any(
        record.get("focal_px") is not None or record.get("K") is not None
        for record in (
            proposal_cache.depth_pro_camera_records if proposal_cache is not None else ()
        )
    )
    vggt_intrinsics_available = bool(
        proposal_cache is not None and proposal_cache.vggt_camera_records
    )
    teacher_agreement = _teacher_agreement_good(disagreement)
    cross_view_agreement = _cross_view_agreement_good(cross_view_metrics)
    signals = [
        depth_pro_available,
        vggt_available,
        teacher_agreement or cross_view_agreement,
        exif_focal_available or depth_pro_focal_available or vggt_intrinsics_available,
    ]
    positive = sum(1 for item in signals if item)
    confidence = (
        "high"
        if depth_pro_available and vggt_available and teacher_agreement and cross_view_agreement
        else "medium"
        if positive >= 2 and (teacher_agreement or cross_view_agreement)
        else "low"
    )
    reasons = _scale_reasons(
        depth_pro_available=depth_pro_available,
        vggt_available=vggt_available,
        exif_focal_available=exif_focal_available,
        depth_pro_focal_available=depth_pro_focal_available,
        vggt_intrinsics_available=vggt_intrinsics_available,
        teacher_agreement=teacher_agreement,
        cross_view_agreement=cross_view_agreement,
    )
    hypotheses = {
        "format_name": "atlas3r_scale_hypotheses",
        "format_version": 1,
        "selected_scale_mode": "unanchored_soft_metric",
        "scale_status": "soft_metric_unanchored",
        "metric_scale_source": _SOFT_METRIC_SOURCE,
        "depth_pro_metric_prior_available": depth_pro_available,
        "vggt_metric_prior_available": vggt_available,
        "exif_focal_available": exif_focal_available,
        "depth_pro_focal_available": depth_pro_focal_available,
        "vggt_intrinsics_available": vggt_intrinsics_available,
        "scale_confidence": confidence,
        "reasons": reasons,
        "scale_hypotheses": [
            {
                "name": "depth_pro_metric_prior",
                "available": depth_pro_available,
                "metric_scale_source": "depth_pro_metric_proposal_unanchored",
                "truth_status": "teacher_prior_not_physical_truth",
            },
            {
                "name": "vggt_multi_view_prior",
                "available": vggt_available,
                "metric_scale_source": "vggt_unanchored_metric_proposal",
                "truth_status": "teacher_prior_not_physical_truth",
            },
            {
                "name": "exif_focal_intrinsics_prior",
                "available": exif_focal_available,
                "truth_status": "intrinsics_proposal_not_calibration",
            },
        ],
        "physical_accuracy_claim": False,
    }
    return {
        "format_name": "atlas3r_soft_metric_scale_ledger",
        "format_version": 1,
        "selected_scale_mode": "unanchored_soft_metric"
        if scale_mode == "unanchored-soft-metric"
        else str(scale_mode),
        "scale_status": "soft_metric_unanchored",
        "metric_scale_source": _SOFT_METRIC_SOURCE,
        "depth_pro_metric_prior_available": depth_pro_available,
        "vggt_metric_prior_available": vggt_available,
        "exif_focal_available": exif_focal_available,
        "depth_pro_focal_available": depth_pro_focal_available,
        "vggt_intrinsics_available": vggt_intrinsics_available,
        "teacher_agreement_good": teacher_agreement,
        "cross_view_projection_agreement_good": cross_view_agreement,
        "scale_confidence": confidence,
        "reasons": reasons,
        "physical_accuracy_claim": False,
        "measured_geometry": False,
        "training_quality": False,
        "scale_hypotheses": hypotheses,
    }


def _teacher_agreement_good(disagreement: DisagreementResult | None) -> bool:
    if disagreement is None:
        return False
    rel_mean = _float_or_none(disagreement.summary.get("rel_depth_diff_mean"))
    rel_p95 = _float_or_none(disagreement.summary.get("rel_depth_diff_p95"))
    return bool(
        rel_mean is not None and rel_p95 is not None and rel_mean <= 0.10 and rel_p95 <= 0.25
    )


def _cross_view_agreement_good(metrics: dict[str, object] | None) -> bool:
    if metrics is None:
        return False
    count = _float_or_none(metrics.get("cross_view_projection_count"))
    mean = _float_or_none(metrics.get("cross_view_depth_residual_mean_m"))
    p95 = _float_or_none(metrics.get("cross_view_depth_residual_p95_m"))
    return bool(
        count is not None
        and count > 0
        and mean is not None
        and p95 is not None
        and mean <= 0.05
        and p95 <= 0.15
    )


def _scale_reasons(
    *,
    depth_pro_available: bool,
    vggt_available: bool,
    exif_focal_available: bool,
    depth_pro_focal_available: bool,
    vggt_intrinsics_available: bool,
    teacher_agreement: bool,
    cross_view_agreement: bool,
) -> list[str]:
    reasons = []
    reasons.append(
        "Depth Pro soft metric depth prior available"
        if depth_pro_available
        else "Depth Pro prior unavailable"
    )
    reasons.append(
        "VGGT multi-view geometry prior available" if vggt_available else "VGGT prior unavailable"
    )
    reasons.append(
        "EXIF focal metadata available as an intrinsics proposal"
        if exif_focal_available
        else "EXIF focal metadata unavailable"
    )
    if depth_pro_focal_available:
        reasons.append("Depth Pro focal proposal available")
    if vggt_intrinsics_available:
        reasons.append("VGGT intrinsics proposal available")
    reasons.append(
        "teacher depth agreement is good"
        if teacher_agreement
        else "teacher depth agreement is weak or unavailable"
    )
    reasons.append(
        "cross-view projection agreement is good"
        if cross_view_agreement
        else "cross-view projection agreement is weak or unavailable"
    )
    reasons.append("no anchor, calibration target, ruler, or measured distance is present")
    return reasons


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    try:
        return None if value is None else float(str(value))
    except (TypeError, ValueError):
        return None
