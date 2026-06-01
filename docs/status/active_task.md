# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2C revised - replace synthetic-shaped private TSDF frame plumbing
with a public validated DepthObservation contract used by both synthetic TSDF
fusion and teacher-cache TSDF replay.
Relevant docs read: AGENTS.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/mapping/cpu_tsdf.py,
src/atlas3r/mapping/teacher_cache_replay.py,
src/atlas3r/data/synthetic_cube_room.py, tests/synthetic/test_cpu_tsdf.py,
tests/synthetic/test_teacher_cache_replay.py.
Plan:
1. Add src/atlas3r/mapping/observations.py with a validated DepthObservation
   dataclass and a synthetic-frame constructor.
2. Refactor CPU TSDF integration so the integration function consumes
   DepthObservation while preserving synthetic smoke artifacts and numerical
   behavior.
3. Refactor teacher-cache replay to convert replay frames into DepthObservation
   before TSDF fusion and remove the type ignore smell.
4. Export the new contract, document it in docs/08_API_CONTRACTS.md, and record
   the architecture decision.
5. Replace docs/status/next_task.md with the Phase 2D deterministic streaming
   runtime scheduler skeleton prompt and fix the stale PLANS.md Phase 0 wording.
6. Add focused tests for DepthObservation validation and the removed replay type
   ignore, then run formatting, lint, typecheck, unittest, and available make
   commands.
Checklist:
- [x] Read required docs and focused mapping/test files.
- [x] Rewrite active task for revised Phase 2C.
- [x] Add DepthObservation contract and constructors.
- [x] Refactor CPU TSDF synthetic integration path.
- [x] Refactor teacher-cache replay observation path.
- [x] Add/update focused tests.
- [x] Update contracts, decisions, progress, next task, and PLANS.md.
- [x] Run verification commands and record results.
Known exclusions: No neural inference, external model downloads, CUDA,
GLB/PLY export, marching cubes, object-aware mesh extraction, video decoding,
web servers, notebooks, or regression bundles in this revised Phase 2C.
```
