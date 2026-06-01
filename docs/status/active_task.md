# Active task

Codex should keep the current task checklist here so context compaction does not lose state.

```text
Goal: Phase 0C - implement a lightweight `.atlas3r` session reader/validator
and dependency-free SVG/HTML previews for the synthetic cube-room session.
Relevant docs read: AGENTS.md, README.md, PLANS.md, docs/08_API_CONTRACTS.md,
docs/status/active_task.md, docs/status/progress.md, docs/status/decisions.md,
src/atlas3r/data/synthetic_cube_room.py, src/atlas3r/cli.py, and
tests/synthetic/test_synthetic_cube_room.py.
Plan:
1. Add `atlas3r.io.session` to load and validate Phase 0B JSON/JSONL/NPZ
   sidecars back into existing contracts without eagerly loading depth arrays.
2. Add `atlas3r.visualization.session_preview` to write deterministic
   dependency-free HTML/SVG previews for top-down layout, first depth frame, and
   first object-id mask.
3. Wire `atlas3r inspect session --input <session.atlas3r> --output <preview>`.
4. Add `.gitignore` coverage for generated artifacts and update the Makefile
   with an `inspect` target.
5. Add focused tests for session IO, preview determinism, CLI inspect, and the
   Phase 0D handoff prompt.
6. Run requested verification commands and record results in progress docs.
Checklist:
- [x] Implement session IO module.
- [x] Implement preview writer.
- [x] Wire CLI and Makefile inspect target.
- [x] Add/update tests and `.gitignore`.
- [x] Advance `docs/status/next_task.md` to Phase 0D.
- [x] Run verification commands.
Known exclusions: No neural models, teacher adapters, TSDF fusion, OpenGL, web
servers, notebooks, matplotlib, Pillow, trimesh, or GLB export in Phase 0C.
Verification results:
- `python -m ruff format src tests` formatted/checked files.
- `python -m ruff format --check src tests` passed.
- `python -m ruff check src tests` passed.
- `python -m mypy src` passed with no issues in 24 source files.
- `python -m unittest discover -s tests -p 'test_*.py'` ran 33 tests and passed.
- `make test`, `make lint`, `make typecheck`, `make smoke`, and `make inspect`
  could not run because `make` is not recognized on PATH in this environment.
```
