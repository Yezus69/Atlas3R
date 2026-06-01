# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 0E - add dependency-safe teacher-adapter contracts, named stubs,
registry discovery, CLI listing, docs, and tests without downloading models or
adding heavyweight dependencies.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md,
docs/status/decisions.md, docs/status/next_task.md, src/atlas3r/api/contracts.py,
src/atlas3r/api/validation.py, src/atlas3r/cli.py,
src/atlas3r/models/adapters/__init__.py, tests/unit/test_contracts.py, and
tests/unit/test_cli.py.
Plan:
1. Add `atlas3r.models.adapters.contracts` with `FrameBatch`,
   `TeacherPrediction`, `GeometryTeacherAdapter`, adapter capabilities,
   availability status, and clear adapter runtime errors.
2. Add small `VGGTAdapter` and `DepthProAdapter` stubs that never import
   optional teacher packages at module import time and raise installation errors
   from construction when dependencies are missing.
3. Add a pure-Python adapter registry and `atlas3r adapters list`.
4. Document the public adapter contract in docs/08_API_CONTRACTS.md and add a
   decision entry because a new public interface is introduced.
5. Add focused tests for contract validation, stub import/dependency errors,
   registry status, and CLI listing while preserving Phase 0A-0D tests.
6. Run requested verification commands, record results, and replace
   docs/status/next_task.md with the Phase 1A prompt.
Checklist:
- [x] Add adapter contract dataclasses/protocol/errors.
- [x] Add VGGT and Depth Pro dependency-safe stubs.
- [x] Add registry and CLI listing.
- [x] Update API docs and decisions.
- [x] Add/update tests.
- [x] Advance `docs/status/next_task.md` to Phase 1A.
- [x] Run verification commands and record results.
Known exclusions: No model downloads, vendored third-party code, cloud APIs,
neural inference, CUDA, heavyweight visualization/export dependencies, or
model-weight assumptions in Phase 0E.
Verification results:
- `python -m ruff format src tests` formatted/checked files.
- `python -m ruff format --check src tests` passed.
- `python -m ruff check src tests` passed.
- `python -m mypy src` passed with no issues in 30 source files.
- `python -m unittest discover -s tests -p 'test_*.py'` ran 43 tests and passed.
- `python -m atlas3r adapters list` printed known adapter statuses.
- `make test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect`
  could not run because PowerShell reported: `make` is not recognized as the
  name of a cmdlet, function, script file, or operable program.
```
