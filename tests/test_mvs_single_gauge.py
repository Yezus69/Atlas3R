"""Single-gauge composite scale decisions.

The MVS composite writer must not mix pose-gauge verified pixels with raw
backbone-gauge fill. The pure decision below is the pre-registered N>=500
per-frame rule plus global fallback; artifact IO is tested by the canonical
rebuild, not by synthetic depth files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_mvs_depth_backend.py"
_spec = importlib.util.spec_from_file_location("run_mvs_depth_backend", _TOOL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_single_gauge_uses_per_frame_scale_and_global_fallback():
    decision = _mod.single_gauge_v2_decisions(
        [
            {
                "frame_id": 10,
                "verified_count": 800,
                "raw_backbone_over_pose_mvs_median": 2.0,
            },
            {
                "frame_id": 20,
                "verified_count": 499,
                "raw_backbone_over_pose_mvs_median": 8.0,
            },
            {
                "frame_id": 30,
                "verified_count": 1000,
                "raw_backbone_over_pose_mvs_median": 6.0,
            },
        ],
        min_verified_pixels=500,
    )

    assert decision["depth_based_global_scale"] == pytest.approx(4.0)
    by_frame = {row["frame_id"]: row for row in decision["frames"]}
    assert by_frame[10]["frame_scale_source"] == "per_frame"
    assert by_frame[10]["fill_scale_multiplier"] == pytest.approx(0.5)
    assert by_frame[20]["frame_scale_source"] == "global_fallback"
    assert by_frame[20]["frame_scale_used"] == pytest.approx(4.0)
    assert by_frame[20]["fill_scale_multiplier"] == pytest.approx(0.25)
    assert by_frame[30]["frame_scale_source"] == "per_frame"
    assert by_frame[30]["fill_scale_multiplier"] == pytest.approx(1.0 / 6.0)


def test_single_gauge_requires_one_qualifying_frame():
    with pytest.raises(ValueError):
        _mod.single_gauge_v2_decisions(
            [
                {
                    "frame_id": 10,
                    "verified_count": 499,
                    "raw_backbone_over_pose_mvs_median": 2.0,
                }
            ],
            min_verified_pixels=500,
        )
