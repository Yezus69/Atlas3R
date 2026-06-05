"""Camera, intrinsics, and scale ledgers for offline traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.contracts.frames import CameraModel
from atlas3r.offline.proposal_cache import DebugGeometryMode
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
) -> CameraScaleLedgerResult:
    root = Path(run_dir)
    intrinsics_status = "guessed" if camera is not None else "unknown"
    scale_source = _scale_source(frame_count, debug_geometry_mode)
    physical_accuracy_allowed = False
    camera_payload = {
        "status": "partial" if camera is not None else "unavailable",
        "intrinsics_status": intrinsics_status,
        "camera": None if camera is None else camera.to_dict(),
        "why": (
            "intrinsics guessed from image size; no calibration metadata"
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
        "allowed_sources": [
            "unknown",
            "unanchored_rgb_prior",
            "debug_flat_depth",
            "synthetic_known",
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


def _scale_source(frame_count: int, debug_geometry_mode: DebugGeometryMode) -> str:
    if debug_geometry_mode == "flat-depth":
        return "debug_flat_depth"
    if debug_geometry_mode == "synthetic-known":
        return "synthetic_known"
    if frame_count > 0:
        return "unanchored_rgb_prior"
    return "unknown"
