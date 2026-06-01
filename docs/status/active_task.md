# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2D - deterministic streaming runtime scheduler skeleton.
Relevant docs read: AGENTS.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/mapping/observations.py,
src/atlas3r/mapping/cpu_tsdf.py,
src/atlas3r/mapping/teacher_cache_replay.py,
src/atlas3r/models/adapters/fixture_teacher_adapter.py,
src/atlas3r/models/adapters/runner.py, src/atlas3r/cli.py,
Makefile, and focused unit/synthetic tests for adapters, teacher cache,
DepthObservation, CPU TSDF, teacher-cache replay, TSDF output inspection, and
CLI smoke wiring.
Plan:
1. Add runtime event dataclasses/enums with deterministic timestamp/latency
   placeholders, dropped-frame flags, bounded-memory counters, and path
   metadata.
2. Add a single-threaded runtime fixture scheduler that writes a tiny synthetic
   session, runs fixture-cube-room with store_arrays=True, replays the cache
   into CPU TSDF through DepthObservation, and writes event logs plus existing
   TSDF artifacts.
3. Wire atlas3r smoke runtime-fixture --output <folder> and update Makefile
   smoke/inspect without renaming existing artifact paths.
4. Add focused runtime tests for deterministic event logs, bounded-memory
   counters, CLI output, and path-named failures.
5. Update API contracts, decisions, progress, and replace next_task.md with
   the Phase 2E prompt after verification.
Checklist:
- [x] Read required docs and focused source/tests.
- [x] Rewrite active task for Phase 2D.
- [x] Add runtime event contracts.
- [x] Add deterministic fixture scheduler.
- [x] Wire runtime fixture CLI and Makefile.
- [x] Add/update focused runtime tests.
- [x] Update contracts, decisions, progress, and Phase 2E next task.
- [x] Run requested Ruff, mypy, unittest, and available make commands.
Known exclusions: Do not add threads, asyncio, GPU/CUDA/Metal, neural inference,
video decoding, external model dependencies, web servers, notebooks, GLB/PLY
export, marching cubes, object-aware mesh extraction, or accuracy claims.
```
