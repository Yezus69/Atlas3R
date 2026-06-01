# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 0B - implement deterministic synthetic cube-room generator and
`atlas3r smoke synthetic-cube-room` only.
Relevant docs read: README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md,
docs/status/next_task.md.
Plan:
1. Add a deterministic synthetic cube-room scene module using Phase 0A contracts.
2. Generate fixed intrinsics, camera poses, analytic depth, object masks, object
   records, and a ground-truth mesh chunk/world map.
3. Add a smoke session writer for a tiny `.atlas3r` folder.
4. Wire `atlas3r smoke synthetic-cube-room --output <folder>`.
5. Add tests for contract validation, projection/unprojection depth agreement,
   object-mask alignment, mesh/world validation, and smoke output files.
6. Run requested verification commands and update progress docs.
Checklist:
- [x] Implement synthetic scene contracts.
- [x] Implement smoke session writer.
- [x] Wire smoke CLI.
- [x] Add required tests.
- [x] Run verification commands.
- [x] Update progress and decision docs if needed.
Known exclusions: No neural models, teacher adapters, TSDF fusion, learned mesh
extraction, or runtime video capture in Phase 0B.
Verification results:
- `python -m ruff format src tests` formatted checked files.
- `python -m ruff format --check src tests` passed.
- `python -m ruff check src tests` passed.
- `python -m mypy src` passed with no issues in 20 source files.
- `python -m unittest discover -s tests -p test_*.py` ran 24 tests and passed.
- `make test`, `make lint`, `make typecheck`, and `make smoke` could not run
  because `make` is not available on PATH in this environment.
```
