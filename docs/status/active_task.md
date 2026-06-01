# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 1A - add the first dependency-light teacher prediction cache and
runner skeleton without downloading weights, vendoring third-party code, cloud
APIs, neural inference, CUDA, or heavyweight visualization/export dependencies.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md, src/atlas3r/api/contracts.py,
src/atlas3r/api/validation.py, src/atlas3r/cli.py, src/atlas3r/io/session.py,
src/atlas3r/models/adapters/contracts.py,
src/atlas3r/models/adapters/registry.py,
src/atlas3r/models/adapters/vggt_adapter.py,
src/atlas3r/models/adapters/depth_pro_adapter.py, tests/unit/test_adapters.py,
tests/unit/test_cli.py, and tests/synthetic/test_session_inspect.py.
Plan:
1. Add `atlas3r.io.teacher_cache` with deterministic JSON/JSONL metadata and
   frame-summary read/write validation for `TeacherPrediction`.
2. Preserve adapter name, adapter status/capability metadata, frame IDs,
   coordinate frame, scale source, confidence summaries, and uncertainty
   summaries without serializing large tensors by default.
3. Add `atlas3r adapters run --adapter <name> --input <session.atlas3r>
   --output <cache_dir>` as a tiny runner surface that validates the input
   session and fails clearly for unavailable or stub-only adapters.
4. Document the public cache layout in docs/08_API_CONTRACTS.md and record a
   decision because this introduces a persistent format.
5. Add focused tests for deterministic cache output, reader validation, and CLI
   runner errors while preserving Phase 0A-0E tests.
6. Run requested verification commands, record results, and replace
   docs/status/next_task.md with the Phase 1B prompt.
Checklist:
- [x] Add teacher cache dataclasses/helpers.
- [x] Add cache writer deterministic output.
- [x] Add cache reader validation errors.
- [x] Add adapter runner skeleton and CLI command.
- [x] Update API docs and decisions.
- [x] Add/update tests.
- [x] Advance `docs/status/next_task.md` to Phase 1B.
- [x] Run verification commands and record results.
Known exclusions: No model downloads, vendored third-party code, cloud APIs,
neural inference, CUDA, heavyweight visualization/export dependencies, or model
weight assumptions in Phase 1A.
```
