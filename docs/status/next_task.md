# Codex Prompt - Atlas3R Phase 2D: Deterministic Streaming Runtime Scheduler Skeleton

You are working in `Yezus69/Atlas3R` after Phase 2C revised.

Read only these files first unless a test failure requires more context:

- `AGENTS.md`
- `PLANS.md`
- `docs/08_API_CONTRACTS.md`
- `docs/status/active_task.md`
- `docs/status/progress.md`
- `docs/status/decisions.md`
- `src/atlas3r/mapping/observations.py`
- `src/atlas3r/mapping/cpu_tsdf.py`
- `src/atlas3r/mapping/teacher_cache_replay.py`
- `src/atlas3r/models/adapters/fixture_teacher_adapter.py`
- `src/atlas3r/models/adapters/runner.py`
- focused tests under `tests/unit/` and `tests/synthetic/` that cover the
  fixture adapter, teacher cache, TSDF replay, and CLI smoke wiring.

Do not paste large code blocks in chat. Modify files directly. Keep changes
small. Preserve all existing CLI commands and artifacts. Run tests before
declaring done.

## Task goal

Add a minimal deterministic streaming runtime scheduler skeleton that exercises
the existing synthetic cube-room fixture adapter cache path into TSDF mapping.
This is an event-log and bounded-memory runtime plumbing slice only.

Do not add threads, asyncio, GPU/CUDA, neural inference, video decoding,
external model dependencies, web servers, notebooks, GLB/PLY export, marching
cubes, object-aware mesh extraction, or accuracy claims.

## Required implementation

### 1. Add runtime event contracts

Create:

```text
src/atlas3r/runtime/events.py
```

Define small typed dataclasses or enums for deterministic per-stage event
records. Each event must include at least:

- event index;
- stage name;
- frame id when applicable;
- deterministic timestamp placeholder;
- deterministic latency placeholder;
- dropped-frame flag;
- bounded-memory counters;
- metadata for source/cache/output paths where applicable.

Keep timestamps deterministic placeholders for now, not wall-clock values.

### 2. Add scheduler skeleton

Create:

```text
src/atlas3r/runtime/scheduler.py
```

Implement a single-threaded deterministic fixture scheduler that:

- creates or consumes a tiny synthetic cube-room session;
- streams frames through the existing `fixture-cube-room` adapter cache path
  with `store_arrays=True`;
- replays the resulting full-array teacher cache into CPU TSDF mapping;
- writes deterministic event logs and the existing TSDF output artifacts;
- tracks bounded-memory counters and proves it does not accumulate all frame
  arrays beyond the tiny configured bound;
- uses the public `DepthObservation` path through existing mapper code.

This scheduler is not a real-time implementation yet. It should be simple,
deterministic, and dependency-free.

### 3. Wire CLI smoke command

Add:

```bash
atlas3r smoke runtime-fixture --output <folder>
```

The output folder should contain deterministic runtime event logs plus the
existing fixture cache/TSDF artifacts needed to inspect the smoke run. Do not
rename existing Phase 0D/1D/1E/2A/2B artifacts.

### 4. Update docs/status and contracts

- Rewrite `docs/status/active_task.md` before coding with a concise Phase 2D
  checklist.
- Update `docs/08_API_CONTRACTS.md` with the runtime fixture event-log contract
  and CLI output layout.
- Append to `docs/status/decisions.md` if the event log or scheduler output is
  a new public format.
- Append results to `docs/status/progress.md` after verification.
- Replace `docs/status/next_task.md` with the next Phase 2E prompt before
  declaring done.

### 5. Add or update tests

Add focused tests proving:

- runtime fixture event logs are deterministic across two output folders;
- event records include stage names, frame ids when applicable, dropped-frame
  flags, placeholder timestamps/latencies, and bounded-memory counters;
- the scheduler does not accumulate unbounded frame arrays on the tiny fixture;
- `atlas3r smoke runtime-fixture --output <folder>` writes expected artifacts
  and reports path-named failures without traceback;
- existing synthetic TSDF, teacher-cache replay, and inspection tests still
  pass.

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

## Done criteria

Done means:

- `src/atlas3r/runtime/events.py` and `src/atlas3r/runtime/scheduler.py` exist;
- the runtime fixture smoke command is wired and deterministic;
- event logs include deterministic per-stage records with bounded-memory
  counters;
- the synthetic fixture adapter cache path feeds TSDF mapping without new heavy
  dependencies;
- tests cover determinism and bounded-memory behavior;
- Ruff, mypy, and unittest pass in the available environment.
