# Active Task - Phase 6H Teacher Stitch Cache

Goal: stitch independent RGB teacher windows into coherent pseudo-world
submaps, export validated temporal teacher caches, and prove the path on a real
multi-window RGB-only run without weakening truth flags.

Branch: `codex/phase6h-teacher-stitch-cache`

Checklist:

- [x] Confirm clean worktree and create Phase 6H branch from Phase 6G.
- [x] Read required architecture, contract, evaluation, status, runtime,
  mapper, teacher, mesh, CLI, and test context.
- [x] Add Sim3 stitching math, window graph diagnostics, no-stitch mode, and
  explicit submap/rejection handling.
- [x] Integrate stitching defaults and diagnostics into `runtime map-rgb-teacher`
  while preserving Phase 6G behavior.
- [x] Export and inspect `atlas3r_teacher_temporal_cache` clips with conservative
  truth flags and pseudo target weights.
- [x] Add dependency-free unit, mapping, cache, and CLI tests.
- [x] Run format/lint/type/unit checks plus real no-stitch and stitched VGGT
  evidence runs.
- [x] Update concise contracts, docs/status, Phase 6H report, next task, and
  commit only source/docs/tests.
