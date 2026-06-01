# Codex Prompt - Atlas3R Phase 1A: Teacher Prediction Cache and Runner Skeleton

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 0E.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for teacher adapter discovery, prediction contracts, and file IO.

## Task goal

Create the first dependency-light Phase 1 teacher prediction cache and runner
surface. This task must not download model weights, vendor third-party
repositories, call cloud APIs, run neural inference, add CUDA, or add
heavyweight visualization/export dependencies.

## Required implementation

1. Add a serialized `TeacherPrediction` cache format.
   - Store metadata and per-frame contract summaries in a deterministic folder
     or JSON/JSONL layout.
   - Preserve adapter name, adapter availability/capability metadata, frame IDs,
     coordinate frame, scale source, confidence, and uncertainty summaries.
   - Do not silently drop uncertainty or confidence fields.

2. Add read/write validation helpers.
   - Validate cache metadata and frame prediction summaries with explicit errors.
   - Keep full tensor serialization narrow and dependency-light; if arrays are
     stored, prefer NumPy `.npz` with documented keys.

3. Add a tiny runner skeleton.
   - Add a CLI command such as `atlas3r adapters run --adapter <name> --input
     <session.atlas3r> --output <cache_dir>`.
   - The command may fail gracefully for unavailable/stub-only external adapters,
     but the error must include adapter name, status, and installation or
     implementation guidance.
   - Do not add real external model inference yet.

4. Add tests.
   - Cache writer output is deterministic.
   - Cache reader validates required metadata and summaries.
   - CLI runner reports clear unavailable/stub-only adapter errors.
   - Existing Phase 0A-0E tests keep passing.

5. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 1A
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` if a public cache format is introduced.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 1B prompt before declaring done.

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

Record any unavailable command with the exact environment reason in
`docs/status/progress.md`.
