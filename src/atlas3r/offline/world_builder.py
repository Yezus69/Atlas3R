"""Minimal Offline World Builder report skeleton.

This module intentionally does not run teachers or optimize geometry yet.
"""

from __future__ import annotations

from atlas3r.input.video import VideoInspection
from atlas3r.teachers.base import AdapterStatus


def build_quality_report_skeleton(
    inspection: VideoInspection, teachers: tuple[AdapterStatus, ...]
) -> dict[str, object]:
    return {
        "format_name": "atlas3r_offline_world_builder_quality_report",
        "format_version": 1,
        "input": inspection.to_dict(),
        "teacher_witnesses": [teacher.to_dict() for teacher in teachers],
        "metrics": {
            "teacher_disagreement": None,
            "reprojection_error": None,
            "render_vs_frame_mismatch": None,
            "scale_source": "unanchored_rgb_prior",
            "surface_confidence": None,
            "object_track_consistency": None,
            "observed_vs_predicted_geometry": None,
            "map_completeness": None,
        },
        "truth_boundary": {
            "label_type": "unanchored_mp4_pseudo",
            "measured_geometry": False,
            "observed_only": True,
            "accuracy_report": False,
            "realtime_claim": False,
            "notes": "Skeleton report only; no teacher inference or optimization has run.",
        },
        "failure_modes": [
            "missing calibration",
            "unknown metric scale",
            "motion blur",
            "dynamic objects",
            "low parallax",
            "teacher disagreement",
        ],
    }
