"""Consensus world-state skeleton for the vertical tracer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.camera_scale_ledger import CameraScaleLedgerResult
from atlas3r.offline.disagreement import DisagreementResult
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
    disagreement: DisagreementResult | None = None,
) -> ConsensusWorldResult:
    has_vggt = bool(proposal_cache.vggt_camera_records and proposal_cache.vggt_depth_records)
    has_depth_pro = bool(proposal_cache.depth_pro_depth_records)
    has_debug = bool(proposal_cache.debug_depth_records)
    has_global_pose = has_vggt or has_debug
    pose_status = (
        "proposed"
        if has_vggt
        else "debug"
        if has_debug
        else "missing_global_pose"
        if has_depth_pro
        else "unknown"
    )
    depth_status = "proposed" if has_vggt or has_depth_pro or has_debug else "missing"
    map_status = "preview" if has_global_pose else "none"
    proposal_source = proposal_cache.geometry_source if has_global_pose else None
    witnesses = []
    if has_vggt:
        witnesses.append("vggt")
    if has_depth_pro:
        witnesses.append("depth_pro")
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
        "proposal_source": proposal_source,
        "available_witnesses": witnesses,
        "proposal_statuses": {
            "vggt_camera_count": len(proposal_cache.vggt_camera_records),
            "vggt_depth_count": len(proposal_cache.vggt_depth_records),
            "depth_pro_camera_count": len(proposal_cache.depth_pro_camera_records),
            "depth_pro_depth_count": len(proposal_cache.depth_pro_depth_records),
            "debug_depth_count": len(proposal_cache.debug_depth_records),
        },
        "teacher_disagreement": None
        if disagreement is None
        else {
            "status": disagreement.status,
            "summary": disagreement.summary,
            "diagnostic_only": True,
            "optimized_consensus": False,
            "path": disagreement.json_path,
        },
        "optimization_status": "not_run",
        "uncertainty_status": "heuristic_or_missing",
        "unresolved_fields": [
            "optimized_camera_poses",
            "teacher_disagreement_optimizer",
            "render_mismatch",
            "object_tracks",
            "scale_anchor",
        ],
        "truth_boundary": {
            "label_type": "unanchored_teacher_consensus_map"
            if has_vggt or has_depth_pro
            else "unanchored_mp4_pseudo",
            "metric_scale_source": ledgers.scale_source,
            "scale_status": "soft_metric_unanchored",
            "measured_geometry": False,
            "observed_only": True,
            "predicted_completion": False,
            "hidden_geometry_measured": False,
            "accuracy_report": False,
            "realtime_claim": False,
            "usable_for_training": False,
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
