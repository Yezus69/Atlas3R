"""Camera, intrinsics, and scale ledgers for offline traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.contracts.frames import CameraModel
from atlas3r.offline.proposal_cache import DebugGeometryMode, ProposalCacheResult
from atlas3r.offline.run_manifest import write_json


@dataclass(frozen=True)
class CameraScaleLedgerResult:
    camera_ledger_path: str
    scale_ledger_path: str
    intrinsics_status: str
    scale_source: str
    physical_accuracy_allowed: bool


def write_camera_scale_ledgers(
    run_dir: str | Path,
    *,
    camera: CameraModel | None,
    frame_count: int,
    debug_geometry_mode: DebugGeometryMode,
    proposal_cache: ProposalCacheResult | None = None,
) -> CameraScaleLedgerResult:
    root = Path(run_dir)
    has_vggt = bool(proposal_cache is not None and proposal_cache.vggt_camera_records)
    intrinsics_status = "proposed" if has_vggt else "guessed" if camera is not None else "unknown"
    scale_source = _scale_source(frame_count, debug_geometry_mode, has_vggt=has_vggt)
    physical_accuracy_allowed = False
    camera_payload = {
        "status": "partial" if camera is not None or has_vggt else "unavailable",
        "intrinsics_status": intrinsics_status,
        "camera": None if camera is None else camera.to_dict(),
        "proposal_source": "vggt" if has_vggt else None,
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
        "why": (
            "intrinsics proposed by VGGT teacher witness; not calibrated or measured"
            if has_vggt
            else "intrinsics guessed from image size; no calibration metadata"
            if camera is not None
            else "no frames decoded, so no intrinsics are known"
        ),
    }
    scale_payload = {
        "status": "partial",
        "scale_source": scale_source,
        "anchoring_state": "unanchored" if scale_source != "synthetic_known" else "synthetic",
        "physical_accuracy_allowed": physical_accuracy_allowed,
        "claim_blocker": (
            "no measured scale anchor or named evaluation report is present"
            if not physical_accuracy_allowed
            else None
        ),
        "measured_geometry": False,
        "accuracy_report": False,
        "proposal_source": "vggt" if has_vggt else None,
        "allowed_sources": [
            "unknown",
            "unanchored_rgb_prior",
            "debug_flat_depth",
            "synthetic_known",
            "vggt_unanchored_metric_proposal",
            "future_anchor",
            "measured",
        ],
    }
    write_json(root / "world" / "camera_ledger.json", camera_payload)
    write_json(root / "world" / "scale_ledger.json", scale_payload)
    return CameraScaleLedgerResult(
        camera_ledger_path="world/camera_ledger.json",
        scale_ledger_path="world/scale_ledger.json",
        intrinsics_status=intrinsics_status,
        scale_source=scale_source,
        physical_accuracy_allowed=physical_accuracy_allowed,
    )


def _scale_source(
    frame_count: int, debug_geometry_mode: DebugGeometryMode, *, has_vggt: bool
) -> str:
    if has_vggt:
        return "vggt_unanchored_metric_proposal"
    if debug_geometry_mode == "flat-depth":
        return "debug_flat_depth"
    if debug_geometry_mode == "synthetic-known":
        return "synthetic_known"
    if frame_count > 0:
        return "unanchored_rgb_prior"
    return "unknown"
