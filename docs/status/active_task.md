# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2C.1 - clean mapper boundaries before Phase 2D by keeping
DepthObservation generic, moving synthetic conversion to the data side, and
removing teacher-cache replay imports of private TSDF helpers.
Relevant docs read: AGENTS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/mapping/observations.py,
src/atlas3r/mapping/cpu_tsdf.py,
src/atlas3r/mapping/teacher_cache_replay.py,
tests/unit/test_mapping_observations.py, tests/synthetic/test_cpu_tsdf.py,
tests/synthetic/test_teacher_cache_replay.py, and the Phase 2D handoff test.
Plan:
1. Move depth_observation_from_synthetic_frame out of mapping.observations and
   into a synthetic/data module exported by atlas3r.data.
2. Update CPU TSDF, tests, and package exports so DepthObservation remains
   public from atlas3r.mapping while synthetic conversion lives under data.
3. Add public TSDF grid helper functions and use them from CPU TSDF and
   teacher-cache replay instead of private cpu_tsdf helpers.
4. Add source guard tests for the boundary rules and update API contracts plus
   decisions if public module/helper names change.
5. Run the requested Ruff, mypy, unittest, and available make verification
   commands, then record results in docs/status/progress.md.
Checklist:
- [x] Read required docs and focused mapping/test files.
- [x] Rewrite active task for Phase 2C.1.
- [x] Move synthetic observation conversion to atlas3r.data.
- [x] Add public TSDF grid helper module and update imports.
- [x] Add/update focused architecture guard tests.
- [x] Update contracts, decisions, progress, and preserve Phase 2D next task.
- [x] Run verification commands and record results.
Known exclusions: Do not implement runtime scheduling, threads, CUDA, Metal,
neural models, video decoding, marching cubes, GLB/PLY export, web servers,
notebooks, or new heavy dependencies.
```
