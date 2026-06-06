"""Training-cache manifest skeleton for offline traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.geometry_preview import GeometryPreviewResult
from atlas3r.offline.run_manifest import write_json


@dataclass(frozen=True)
class TrainingCacheResult:
    status: str
    manifest_path: str
    usable_for_training: bool


def write_training_cache_manifest(
    run_dir: str | Path,
    *,
    frame_index_path: str,
    keyframes_path: str,
    world_state_path: str,
    geometry: GeometryPreviewResult,
    disagreement: DisagreementResult | None = None,
) -> TrainingCacheResult:
    usable_for_training = False
    payload = {
        "status": "skeleton",
        "usable_for_training": usable_for_training,
        "why_not_training_quality": (
            "no real optimized labels, teacher consensus optimizer, scale anchor, or evaluation "
            "report exists"
        ),
        "proposal_source": geometry.source_teacher if geometry.point_count else None,
        "refs": {
            "frame_index": frame_index_path,
            "keyframes": keyframes_path,
            "world_state": world_state_path,
            "geometry_preview": geometry.geometry_npz_path,
            "geometry_ply": geometry.geometry_ply_path,
            "teacher_disagreement": None if disagreement is None else disagreement.json_path,
            "disagreement_maps": None if disagreement is None else disagreement.maps_npz_path,
            "consensus_preview": None if disagreement is None else disagreement.consensus_npz_path,
        },
        "truth_boundary": {
            "label_type": (
                "teacher_pseudo"
                if geometry.source_teacher == "vggt" and geometry.point_count
                else "debug_synthetic"
                if geometry.point_count
                else "unknown"
            ),
            "metric_scale_source": geometry.metric_scale_source,
            "measured_geometry": geometry.measured_geometry,
            "observed_only": geometry.observed_only,
            "predicted_completion": geometry.predicted_completion,
            "hidden_geometry_measured": False,
            "accuracy_report": False,
            "realtime_claim": False,
            "usable_for_training": usable_for_training,
        },
    }
    write_json(Path(run_dir) / "training_cache" / "training_cache_manifest.json", payload)
    return TrainingCacheResult(
        status="skeleton",
        manifest_path="training_cache/training_cache_manifest.json",
        usable_for_training=usable_for_training,
    )
