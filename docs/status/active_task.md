# Active Task - Real Room Build-World Map

## Goal

Make `offline build-world` produce a finite, non-collapsed, near-metric
`world_map_best/` plus a local viewer for the real `room_walk_001` capture,
without weakening truth-boundary or dependency-safety rules.

## Ordered Checkpoints

- [x] CP1: finite trajectory with non-collapsed room-scale geometry on the real capture.
- [x] CP2: export `world_map_best/viewer.html` that renders local map preview and camera path.
- [x] CP3: add an independent ground-plane/camera-height scale cue and record it in scale ledgers.
- [ ] CP4: object-size anchor deferred.
- [ ] CP5: dynamic masking deferred.

## Intended Edit Scope

- `src/atlas3r/offline/fused_world_map_artifacts.py`
- `src/atlas3r/offline/best_map_selection.py`
- `src/atlas3r/offline/camera_scale_ledger.py`
- `src/atlas3r/offline/ground_plane_scale.py`
- `src/atlas3r/cli.py`
- focused tests under `tests/`
- `docs/status/current_state.md`

## Verification Commands

- Passed: `python -m unittest tests.test_ground_plane_scale tests.test_best_map_selection tests.test_soft_metric_scale_ledger`
- Passed: `python -m pytest`: 89 tests.
- Passed: `python -m ruff format src tests`
- Passed: `python -m ruff check src tests`
- Passed: `python -m mypy src`
- Passed: `python -m compileall -q src tests`
- Passed real run:
  `python -m atlas3r offline build-world --input ...room_walk_001/frames --output runs/room_walk_001_build_world_near_metric_viewer_final ... --export-best-world-map --enable-ground-plane-scale --export-best-world-map-viewer`
- Passed Edge headless screenshot of
  `http://127.0.0.1:8765/viewer.html`.
