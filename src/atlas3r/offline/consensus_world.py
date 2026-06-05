"""Consensus world-state skeleton for the vertical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import CameraScaleLedgerResult
from atlas3r.offline.frame_cache import FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import write_json


@dataclass(frozen=True)
class ConsensusWorldResult:
    status: str
    world_state_path: str
    pose_status: str
    depth_status: str
    map_status: str


def write_consensus_world_state(
    run_dir: str | Path,
    *,
    frame_records: tuple[FrameRecord, ...],
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    ledgers: CameraScaleLedgerResult,
) -> ConsensusWorldResult:
    pose_status = "proposed" if proposal_cache.depth_proposal_available else "unknown"
    depth_status = "proposed" if proposal_cache.depth_proposal_available else "missing"
    map_status = "none"
    payload = {
        "status": "partial",
        "world_id": "offline_world_candidate",
        "frame_refs": [record.frame_path for record in frame_records],
        "keyframe_ids": [keyframe.frame_id for keyframe in keyframes],
        "proposal_manifest": proposal_cache.manifest_path,
        "camera_ledger": ledgers.camera_ledger_path,
        "scale_ledger": ledgers.scale_ledger_path,
        "pose_status": pose_status,
        "depth_status": depth_status,
        "map_status": map_status,
        "optimization_status": "not_run",
        "uncertainty_status": "heuristic_or_missing",
        "unresolved_fields": [
            "optimized_camera_poses",
            "teacher_disagreement",
            "render_mismatch",
            "object_tracks",
            "scale_anchor",
        ],
        "truth_boundary": {
            "label_type": "unanchored_mp4_pseudo",
            "metric_scale_source": ledgers.scale_source,
            "measured_geometry": False,
            "observed_only": True,
            "predicted_completion": False,
            "accuracy_report": False,
            "realtime_claim": False,
        },
    }
    write_json(Path(run_dir) / "world" / "world_state.json", payload)
    return ConsensusWorldResult(
        status="partial",
        world_state_path="world/world_state.json",
        pose_status=pose_status,
        depth_status=depth_status,
        map_status=map_status,
    )
