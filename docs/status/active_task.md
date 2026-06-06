# Active Task: Offline V0.9 Map Consistency Optimizer

Branch: `codex/offline-world-builder-v09-map-consistency-optimizer`

## Checklist

- [x] Confirm clean worktree and create the V0.9 branch from V0.8.
- [x] Read the requested architecture, truth-boundary, status, and code context.
- [x] Reflow compressed V0.8 status docs without deleting evidence numbers.
- [x] Add a NumPy map consistency optimizer called by `offline build-world`.
- [x] Write optimizer artifacts under `optimizer/` with conservative truth flags.
- [x] Export raw `world_map/` and optimized `world_map_optimized/` artifacts.
- [x] Add before/after projection diagnostics under `diagnostics/`.
- [x] Add CLI flags and keep `--optimize-map-consistency` tied to map export.
- [x] Add focused unit and integration tests for optimizer math, diagnostics,
  artifact export, CLI help, and training-cache references.
- [x] Run the requested format, lint, type, unit, CLI, smoke, and diff checks.
- [x] Run debug and real TUM evidence; run phone MP4 evidence only if found.
- [x] Update V0.9 report, current status, progress, decisions, contracts, and
  next-task docs with exact results and known gaps.
- [x] Commit as `feat(offline): optimize teacher map consistency`.
