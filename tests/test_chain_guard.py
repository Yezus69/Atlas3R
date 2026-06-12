"""Chain guard: raw-vs-composite artifact consumption honesty marker.

Small invariant check (AGENTS.md: prevents a dangerous silent mistake): the
teacher once consumed a RAW backbone artifact (independently-posed views) from
``external/teacher_artifacts/<asset>`` while the intended COLMAP-pose+MVS
COMPOSITE sat unread in ``external/_composite_artifacts/<asset>`` -- producing
user-visible garbage with no marker. ``composite_shadow_blocker`` is the pure
decision; ``_composite_shadow_report`` is its filesystem wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

from atlas3r.geometry_adapter import (
    RAW_WHILE_COMPOSITE_BLOCKER,
    _composite_shadow_report,
    composite_shadow_blocker,
)

RAW_METHOD = "mapanything_feedforward_metric_geometry:facebook/map-anything-apache"
COMPOSITE_METHOD = RAW_METHOD + "+colmap_pose_backend+mvs_verified_depth"


def test_raw_consumed_while_composite_exists_is_flagged():
    assert (
        composite_shadow_blocker(RAW_METHOD, COMPOSITE_METHOD)
        == RAW_WHILE_COMPOSITE_BLOCKER
    )


def test_composite_consumed_is_clean():
    assert composite_shadow_blocker(COMPOSITE_METHOD, COMPOSITE_METHOD) is None
    # Hybrid (pose backend without MVS) already carries COLMAP poses -> clean.
    assert (
        composite_shadow_blocker(RAW_METHOD + "+colmap_pose_backend", COMPOSITE_METHOD)
        is None
    )


def test_no_composite_method_is_no_claim():
    assert composite_shadow_blocker(RAW_METHOD, None) is None
    assert composite_shadow_blocker(RAW_METHOD, "") is None


def test_composite_without_pose_backend_does_not_flag():
    # A second raw run parked under _composite_artifacts proves nothing.
    assert composite_shadow_blocker(RAW_METHOD, RAW_METHOD) is None


def test_missing_consumed_method_with_real_composite_is_flagged():
    # A manifest with no method string cannot claim COLMAP poses.
    assert (
        composite_shadow_blocker(None, COMPOSITE_METHOD)
        == RAW_WHILE_COMPOSITE_BLOCKER
    )


def _write_composite_manifest(root: Path, asset_id: str, payload) -> Path:
    composite_dir = root / "external" / "_composite_artifacts" / asset_id
    composite_dir.mkdir(parents=True)
    path = composite_dir / "backbone_manifest.json"
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return composite_dir


def test_wrapper_flags_and_names_the_composite(tmp_path):
    composite_dir = _write_composite_manifest(
        tmp_path, "phone_room_loop", {"method": COMPOSITE_METHOD}
    )
    report = _composite_shadow_report("phone_room_loop", tmp_path, RAW_METHOD)
    assert report is not None
    assert report["blocker"] == RAW_WHILE_COMPOSITE_BLOCKER
    assert report["composite_dir"] == str(composite_dir)
    assert report["composite_method"] == COMPOSITE_METHOD
    assert "--promote" in report["hint"]


def test_wrapper_clean_when_composite_consumed(tmp_path):
    _write_composite_manifest(tmp_path, "phone_room_loop", {"method": COMPOSITE_METHOD})
    assert (
        _composite_shadow_report("phone_room_loop", tmp_path, COMPOSITE_METHOD) is None
    )


def test_wrapper_no_composite_dir_is_no_claim(tmp_path):
    assert _composite_shadow_report("phone_room_loop", tmp_path, RAW_METHOD) is None


def test_wrapper_corrupt_composite_manifest_is_no_claim(tmp_path):
    _write_composite_manifest(tmp_path, "phone_room_loop", "{not json")
    assert _composite_shadow_report("phone_room_loop", tmp_path, RAW_METHOD) is None
