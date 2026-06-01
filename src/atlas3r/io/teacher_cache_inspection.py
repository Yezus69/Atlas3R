"""Deterministic teacher cache inspection output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas3r.io.teacher_cache import load_teacher_prediction_cache


def teacher_cache_inspection_record(path: str | Path) -> dict[str, Any]:
    """Return deterministic inspection metadata for a validated teacher cache."""
    cache = load_teacher_prediction_cache(path)
    metadata = cache.metadata
    arrays = metadata["arrays"]
    return {
        "cache": {
            "format_name": metadata["format_name"],
            "format_version": metadata["format_version"],
            "validated": True,
            "accuracy_report": False,
            "accuracy_note": (
                "Teacher cache inspection validates cache metadata and payload state; "
                "it is not an accuracy report."
            ),
        },
        "adapter": {
            "name": cache.adapter_status.name,
            "display_name": cache.adapter_status.display_name,
            "status": cache.adapter_status.availability,
        },
        "frames": {
            "frame_count": metadata["frame_count"],
            "frame_ids": metadata["frame_ids"],
        },
        "coordinate_frame": metadata["coordinate_frame"],
        "scale_sources": metadata["scale_sources"],
        "arrays": {
            "stored": arrays["stored"],
            "directory": arrays["directory"],
            "optional_npz_keys": arrays["optional_npz_keys"],
            "frame_payloads": [
                {
                    "frame_id": summary["frame_id"],
                    "arrays_path": summary["arrays_path"],
                }
                for summary in cache.frame_summaries
            ],
        },
        "confidence_summaries": [
            {
                "frame_id": summary["frame_id"],
                "summary": summary["confidence_summary"],
            }
            for summary in cache.frame_summaries
        ],
        "uncertainty_summaries": [
            {
                "frame_id": summary["frame_id"],
                "summary": summary["uncertainty_summary"],
            }
            for summary in cache.frame_summaries
        ],
    }


def format_teacher_cache_inspection(path: str | Path) -> str:
    """Format deterministic teacher cache inspection JSON."""
    return json.dumps(teacher_cache_inspection_record(path), indent=2, sort_keys=True) + "\n"


__all__ = ["format_teacher_cache_inspection", "teacher_cache_inspection_record"]
