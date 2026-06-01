# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 0A - implement stable NumPy data contracts and coordinate-frame math foundation only.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/00_FEASIBILITY_AND_TRUTH.md, docs/08_API_CONTRACTS.md, docs/10_CODEX_EXECUTION_GUIDE.md, docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md.
Plan:
1. Add NumPy as a base dependency while keeping existing dev tooling.
2. Implement public API contracts and validation helpers from docs/08_API_CONTRACTS.md.
3. Implement pure camera and transform math utilities.
4. Add deterministic unit tests for imports, contracts, enums, validation failures, and projection math.
5. Run requested formatting, lint, typecheck, unittest, and make commands where available.
6. Update progress and next-task status files.
Checklist:
- [x] Update packaging dependency.
- [x] Implement validation helpers.
- [x] Implement contract dataclasses and public exports.
- [x] Implement transform and pinhole math.
- [x] Add focused unit tests.
- [x] Run verification commands.
- [x] Update progress and next task prompt.
Known exclusions: No neural models, teacher adapters, TSDF fusion, mesh extraction, runtime video capture, or synthetic cube-room generator in Phase 0A.
Verification results:
- `python -m pip install -e ".[dev]"` succeeded.
- `python -m ruff format --check src tests` passed.
- `python -m ruff check src tests` passed.
- `python -m mypy src` passed.
- `python -m unittest discover -s tests -p test_*.py` ran 18 tests and passed.
- `make test`, `make lint`, and `make typecheck` could not run because `make` is not available on PATH in this environment.
```
