"""Pose-backend chain guard: refuse to re-pose an already-processed source.

The 2026-06-12 chain-guard change moved the backbone runners' default
``--out-dir`` to ``external/_raw_backbone_artifacts``. The pose backend's
``--source-artifacts`` default must follow (the printed
``run_sfm_pipeline.py`` next_command relies on it), and a source manifest
already carrying ``colmap_pose_backend``/``mvs_verified_depth`` (i.e. the
promoted live composite) must be refused, not silently re-processed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_colmap_pose_backend.py"
_spec = importlib.util.spec_from_file_location("run_colmap_pose_backend", _TOOL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _call(monkeypatch, capsys, tmp_path, extra_args=()):
    monkeypatch.setattr(_mod, "ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["run_colmap_pose_backend.py", "--asset-id", "scene",
         "--colmap-model", str(tmp_path / "model"), *extra_args],
    )
    rc = _mod.main()
    out = capsys.readouterr().out
    return rc, json.loads(out.splitlines()[-1])


def test_default_source_is_raw_staging_area(monkeypatch, capsys, tmp_path):
    rc, payload = _call(monkeypatch, capsys, tmp_path)
    assert rc == 1
    assert payload["status"] == "error_missing_source_artifact"
    assert "_raw_backbone_artifacts" in payload["missing"].replace("\\", "/")


def test_processed_source_is_refused(monkeypatch, capsys, tmp_path):
    src = tmp_path / "external" / "_raw_backbone_artifacts" / "scene"
    src.mkdir(parents=True)
    (src / "backbone_manifest.json").write_text(
        json.dumps({"method": "m+colmap_pose_backend+mvs_verified_depth"}),
        encoding="utf-8",
    )
    rc, payload = _call(monkeypatch, capsys, tmp_path)
    assert rc == 1
    assert payload["status"] == "error_source_already_pose_processed"
    assert payload["markers_found"] == ["colmap_pose_backend", "mvs_verified_depth"]


def test_raw_source_passes_the_guard(monkeypatch, capsys, tmp_path):
    """A genuinely raw source clears the guard and proceeds to the COLMAP
    model read (which fails here on the missing model -- past the guard)."""
    import pytest

    src = tmp_path / "external" / "_raw_backbone_artifacts" / "scene"
    src.mkdir(parents=True)
    (src / "backbone_manifest.json").write_text(
        json.dumps({"method": "mapanything_feedforward_metric_geometry"}),
        encoding="utf-8",
    )
    (src / "poses.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _call(monkeypatch, capsys, tmp_path)
