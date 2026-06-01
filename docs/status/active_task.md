# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 0D - implement a deterministic pure-NumPy CPU TSDF reference
integrator for the Phase 0B synthetic cube-room session, with smoke CLI outputs
and conservative fixture metrics.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md,
docs/status/next_task.md, src/atlas3r/data/synthetic_cube_room.py,
src/atlas3r/cli.py, src/atlas3r/io/session.py,
tests/synthetic/test_synthetic_cube_room.py, and
tests/synthetic/test_session_inspect.py.
Plan:
1. Add `atlas3r.mapping.cpu_tsdf` with a small deterministic TSDF grid,
   synthetic-frame integration using `T_world_camera`, and point-surface
   extraction with confidence/uncertainty metadata.
2. Add conservative synthetic cube-room metrics against the Phase 0B
   ground-truth bounds/mesh and write deterministic JSON/NPZ smoke outputs.
3. Wire `atlas3r smoke tsdf-cube-room --output <folder>` and update `make smoke`
   to exercise the TSDF path.
4. Document the Phase 0D smoke output format and record a decision because a
   new smoke artifact schema is being introduced.
5. Add focused unit/synthetic tests for determinism, metadata, voxel-scale
   overlap, metrics, and CLI success while preserving existing Phase 0A-0C tests.
6. Run requested verification commands, record results, and replace
   docs/status/next_task.md with the Phase 0E prompt.
Checklist:
- [x] Implement CPU TSDF reference module.
- [x] Wire CLI smoke command and Makefile target.
- [x] Document output format and decision.
- [x] Add/update tests.
- [x] Advance `docs/status/next_task.md` to Phase 0E.
- [x] Run verification commands and record results.
Known exclusions: No CUDA, neural models, teacher adapters, nvblox, OpenGL, web
servers, notebooks, trimesh, or GLB export in Phase 0D.
Verification results:
- `python -m ruff format src tests` formatted/checked files.
- `python -m ruff format --check src tests` passed.
- `python -m ruff check src tests` passed.
- `python -m mypy src` passed with no issues in 25 source files.
- `python -m unittest discover -s tests -p 'test_*.py'` ran 37 tests and passed.
- `make test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect`
  could not run because PowerShell reported: `make` is not recognized as the
  name of a cmdlet, function, script file, or operable program.
```
