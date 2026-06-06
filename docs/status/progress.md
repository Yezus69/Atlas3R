# Progress Summary

This is a rolling current-state summary, not an append-only transcript.

## Current State

- Active branch: `codex/roomgraph-core-optimizer`.
- RoomGraph V1.2 adds a real track/depth/pose optimizer behind
  `offline build-world --optimize-roomgraph`.
- New track extraction writes `roomgraph/tracks.npz` and
  `roomgraph/track_summary.json`, with CoTracker used when available/requested
  and OpenCV LK available as an explicit or `auto` fallback.
- RoomGraph optimizer variants are `depth_only`, `pose_only`, and `joint`.
  Selection is metric-based and reports regressions as well as gains.
- `world_map_roomgraph/` exports fused points, PLY, sparse occupancy, observed
  voxel mesh, camera trajectory, `roomgraph_metrics.json`, and
  `roomgraph_report.md`.
- RoomGraph outputs remain unanchored teacher/optimizer geometry. Physical
  accuracy and training-quality claims remain false.

## Actual Room Evidence

- Input: `C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames`.
- Baseline run: `runs/room_walk_001_roomgraph_baseline`.
- RoomGraph run: `runs/room_walk_001_roomgraph`.
- Cached teacher proposals were replayed from the baseline run.
- A CoTracker-backed command was stopped because it ran too long on the actual
  room pass. The completed evidence run used `--roomgraph-track-source opencv-lk`.
- Selected RoomGraph variant: `pose_only`.
- Tracks: 58 tracks, 1,051 observations.
- Output artifacts exist and are nonzero under
  `runs/room_walk_001_roomgraph/world_map_roomgraph/`.

## RoomGraph Metrics

- Reprojection mean improved: `53.1085 px -> 34.1599 px` (`+35.7%`).
- Track inlier ratio improved: `0.1646 -> 0.3863`.
- Cross-view depth residual regressed:
  `0.02034 m -> 0.02364 m` (`-16.2%`).
- Mapped teacher disagreement was unchanged:
  `0.03292 -> 0.03292`.
- Camera collapse score regressed:
  `0.34697 m -> 0.29890 m` (`-13.9%`).
- Map bbox changed from about `1.72 x 0.75 x 0.96 m` to
  `1.33 x 0.74 x 1.12 m`.
- Point/voxel/mesh counts changed from
  `1,904,976 / 2,559 / 5,600` to `1,036,800 / 1,466 / 3,768`.

## Verification

- Passed: `python -m ruff format src tests`.
- Passed: `python -m ruff format --check src tests`.
- Passed: `python -m ruff check src tests`.
- Passed: `python -m mypy src`.
- Passed: `python -m unittest discover -s tests -p "test_*.py"`: 84 tests.
- Passed: `git diff --check` with line-ending warnings only.
- Passed: `python -m atlas3r offline build-world --help`.
- Passed: `python -m atlas3r smoke contracts`.
- Not run: `make lint`; `make` is not installed in this PowerShell shell. The
  equivalent commands from the Makefile passed.

## Known Gaps

- The completed RoomGraph output improves real track reprojection, but it does
  not improve the room-scale collapse metric.
- The OpenCV LK evidence path was used for the actual room run because the full
  CoTracker pass was too slow in this session.
- Scale remains unanchored soft-metric teacher scale.
- COLMAP/GLOMAP reconstruction remains unavailable on the actual room frames.
- No physical scale anchor, object permanence, final mesh reconstruction, or
  named evaluation report exists.
