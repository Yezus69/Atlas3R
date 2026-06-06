# Active Task - Offline V1.0 Room Walk Soft-Metric Map

## Goal

Run the no-anchor room-walk pipeline on
`C:/Users/Asav/source/repos/homebrain/data/inbox/room_walk_001/frames` and
export an inspectable `world_map_best/` without physical accuracy claims.

## Checklist

- [x] Confirm clean worktree and switch to V1.0 branch.
- [x] Run V0.9 baseline on the actual room frames before code edits.
- [x] Add dependency-safe JPG/EXIF metadata summary.
- [x] Add `--scale-mode unanchored-soft-metric` and soft-metric ledgers.
- [x] Add best-map selection, conservative cleanup, and top-down preview.
- [x] Add room-specific diagnostics and inspection instructions.
- [x] Add focused tests for metadata, scale ledger, best map, preview, CLI, and integration.
- [x] Run final room evidence command and optional longer run if feasible.
- [x] Run verification commands and update compact docs/status files.
