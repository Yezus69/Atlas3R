"""Promote step: atomic install of a composite into the teacher read path.

Windows reality check: ``os.replace`` cannot replace a non-empty directory, so
the implementation must only ever rename onto NON-existent destinations
(move-old-to-backup, then move-tmp-into-place) and must roll the backup back if
the second rename fails -- the teacher read path must never be left empty.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_mvs_depth_backend.py"
_spec = importlib.util.spec_from_file_location("run_mvs_depth_backend", _TOOL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
promote_composite = _mod.promote_composite


def _make_composite(root: Path, asset_id: str, payload: str) -> Path:
    out = root / "external" / "_composite_artifacts" / asset_id
    out.mkdir(parents=True)
    (out / "backbone_manifest.json").write_text(payload, encoding="utf-8")
    return out


def test_first_promote_installs_live_dir(tmp_path):
    out = _make_composite(tmp_path, "scene", '{"method": "m+colmap_pose_backend"}')
    result = promote_composite(out, "scene", root=tmp_path)
    live = tmp_path / "external" / "teacher_artifacts" / "scene"
    assert (live / "backbone_manifest.json").read_text(encoding="utf-8") == (
        '{"method": "m+colmap_pose_backend"}'
    )
    assert result["promoted_to"] == "external/teacher_artifacts/scene"
    assert result["superseded_backup"] is None
    # The composite source is left in place (it is the provenance record).
    assert out.exists()


def test_promote_backs_up_previous_live_dir(tmp_path):
    live = tmp_path / "external" / "teacher_artifacts" / "scene"
    live.mkdir(parents=True)
    (live / "backbone_manifest.json").write_text('{"method": "old_raw"}', encoding="utf-8")
    out = _make_composite(tmp_path, "scene", '{"method": "new_composite"}')

    result = promote_composite(out, "scene", root=tmp_path)
    assert (live / "backbone_manifest.json").read_text(encoding="utf-8") == (
        '{"method": "new_composite"}'
    )
    backup = tmp_path / "external" / "_superseded_artifacts" / "scene_1"
    assert (backup / "backbone_manifest.json").read_text(encoding="utf-8") == (
        '{"method": "old_raw"}'
    )
    assert result["superseded_backup"] == "external/_superseded_artifacts/scene_1"


def test_backup_names_never_collide(tmp_path):
    (tmp_path / "external" / "_superseded_artifacts" / "scene_1").mkdir(parents=True)
    live = tmp_path / "external" / "teacher_artifacts" / "scene"
    live.mkdir(parents=True)
    (live / "backbone_manifest.json").write_text("{}", encoding="utf-8")
    out = _make_composite(tmp_path, "scene", "{}")

    result = promote_composite(out, "scene", root=tmp_path)
    assert result["superseded_backup"] == "external/_superseded_artifacts/scene_2"


def test_failed_final_rename_rolls_back_live_dir(tmp_path, monkeypatch):
    """If the tmp->live rename fails AFTER the live dir moved to backup, the
    backup must be restored: the teacher read path is never left empty."""
    live = tmp_path / "external" / "teacher_artifacts" / "scene"
    live.mkdir(parents=True)
    (live / "backbone_manifest.json").write_text('{"method": "old_raw"}', encoding="utf-8")
    out = _make_composite(tmp_path, "scene", '{"method": "new_composite"}')

    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # first call = live->backup, second = tmp->live
            raise OSError("simulated lock on final rename")
        return real_replace(src, dst)

    monkeypatch.setattr(_mod.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated lock"):
        promote_composite(out, "scene", root=tmp_path)

    # Rolled back: previous artifact is live again, backup slot vacated.
    assert (live / "backbone_manifest.json").read_text(encoding="utf-8") == (
        '{"method": "old_raw"}'
    )
    assert not (tmp_path / "external" / "_superseded_artifacts" / "scene_1").exists()
    # The failed composite copy remains inspectable at the tmp sibling.
    tmp_sibling = tmp_path / "external" / "teacher_artifacts" / "scene__promote_tmp"
    assert (tmp_sibling / "backbone_manifest.json").read_text(encoding="utf-8") == (
        '{"method": "new_composite"}'
    )
