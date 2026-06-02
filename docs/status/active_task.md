# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 2E - dependency-free runtime fixture output inspection.
Relevant docs read: AGENTS.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md.
Relevant source/tests read: src/atlas3r/runtime/events.py,
src/atlas3r/runtime/scheduler.py,
src/atlas3r/mapping/tsdf_output_inspection.py,
tests/synthetic/test_runtime_fixture_scheduler.py,
tests/synthetic/test_tsdf_output_inspection.py,
tests/synthetic/test_teacher_cache_replay.py,
tests/synthetic/test_session_inspect.py, and
tests/synthetic/test_cpu_tsdf.py.
Plan:
1. Add a small runtime fixture inspection module that validates runtime_events.jsonl,
   runtime_summary.json, required runtime fixture artifacts, relative paths, timing
   placeholders, bounded-memory counters, frame IDs, and summary cross-checks.
2. Reuse the complete-mode TSDF output folder inspector for teacher_cache_tsdf.
3. Wire `atlas3r inspect runtime-fixture --input <folder>` with deterministic JSON
   output and path-named CLI errors without tracebacks.
4. Add focused tests for deterministic inspection, malformed/missing path errors,
   CLI behavior, and the Phase 3 handoff prompt.
5. Update API contracts, decisions if needed, progress, and next_task.md after verification.
Checklist:
- [x] Read required docs and focused source/tests.
- [x] Rewrite active task for Phase 2E.
- [x] Add runtime fixture inspection implementation.
- [x] Wire inspect CLI and Makefile if needed.
- [x] Add/update focused synthetic tests.
- [x] Update API contracts, decisions, and Phase 3 next task.
- [x] Run requested Ruff, mypy, unittest, and available make commands.
Known exclusions: Do not add threads, asyncio, GPU/CUDA, neural inference,
video decoding, external model dependencies, web servers, notebooks, GLB/PLY
export, marching cubes, object-aware mesh extraction, performance reports, or
accuracy claims.
```
