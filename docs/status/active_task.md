# Active Task

Offline V0.8: export the first inspectable fused world map from existing VGGT
and Depth Pro teacher-pseudo proposals through `offline build-world`.

## Checklist

- [x] Confirm clean worktree and switch to the V0.8 branch.
- [x] Read required architecture, status, pipeline, witness, proposal, geometry,
  reporting, CLI, and test files.
- [x] Reflow touched status Markdown and keep docs compact.
- [x] Add fused world-map module for points, occupancy, observed voxel mesh,
  camera trajectory, manifest, and map-quality report.
- [x] Wire map export flags and artifacts through `offline build-world`.
- [x] Keep all map artifacts teacher-pseudo, observed-only, non-measured, and
  non-training-quality.
- [x] Add focused unit and integration tests.
- [x] Run format, lint, typecheck, unit, CLI, smoke, and evidence commands.
- [x] Update V0.8 report, contracts, architecture, progress, decisions, current
  state, and next task docs.
- [x] Commit `feat(offline): export fused world map artifacts`.
