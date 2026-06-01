# Codex Prompt - Atlas3R Phase 2E: Runtime Fixture Output Inspection

You are working in `Yezus69/Atlas3R` after Phase 2D.

Read only these files first unless a test failure requires more context:

- `AGENTS.md`
- `PLANS.md`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- `src/atlas3r/runtime/events.py`
- `src/atlas3r/runtime/scheduler.py`
- `src/atlas3r/mapping/tsdf_output_inspection.py`
- focused tests under `tests/synthetic/` that cover runtime fixture smoke,
  TSDF output inspection, teacher-cache replay, and CLI smoke wiring.

Do not paste large code blocks in chat. Modify files directly. Keep changes
small. Preserve all existing CLI commands and artifacts. Run tests before
declaring done.

## Task goal

Add a dependency-free inspection command for Phase 2D runtime fixture output
folders. This is validation and reporting only.

Do not add threads, asyncio, GPU/CUDA, neural inference, video decoding,
external model dependencies, web servers, notebooks, GLB/PLY export, marching
cubes, object-aware mesh extraction, or accuracy claims.

## Required implementation

Add:

```bash
atlas3r inspect runtime-fixture --input <folder>
```

The inspector should validate:

- `runtime_events.jsonl` event format, event index ordering, deterministic
  timestamp/latency placeholders, stage names, frame IDs, dropped-frame flags,
  bounded-memory counters, and relative paths;
- `runtime_summary.json` and its cross-checks against the event log;
- required generated session, teacher cache, full array payloads, and
  `teacher_cache_tsdf` artifacts;
- `teacher_cache_tsdf` by reusing the existing TSDF output folder inspector in
  complete mode.

The command should print deterministic JSON and report missing or malformed
paths without traceback. The output must state that runtime fixture inspection is
a diagnostic only, not a performance report and not an accuracy report.

## Documentation and status

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 2E
  checklist.
- Update `docs/08_API_CONTRACTS.md` with the runtime fixture inspection
  contract.
- Append to `docs/status/decisions.md` if the inspection JSON becomes a new
  public format.
- Append results to `docs/status/progress.md` after verification.
- Replace `docs/status/next_task.md` with the next Phase 3 prompt before
  declaring done.

## Verification commands

Run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p 'test_*.py'
```

If `make` is available, also run:

```bash
make test
make lint
make typecheck
make smoke
make inspect
```

Record unavailable commands and exact reasons in `docs/status/progress.md`.
