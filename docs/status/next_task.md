# Codex Prompt - Atlas3R Phase 1B: Dependency-Free Fixture Teacher Adapter

You are working in the existing public repo `Yezus69/Atlas3R` after Phase 1A.
Read `AGENTS.md`, `README.md`, `PLANS.md`, `docs/08_API_CONTRACTS.md`, and
the current `docs/status/*` files before coding. Then read only the source and
tests needed for teacher adapter discovery, `.atlas3r` session IO, frame
contracts, and the Phase 1A teacher prediction cache.

## Task goal

Add the first cache-producing teacher runner path without downloading model
weights, vendoring third-party repositories, calling cloud APIs, running neural
inference, adding CUDA, or adding heavyweight visualization/export dependencies.
This should be a deterministic fixture teacher used only to exercise contracts,
cache writing, and runner plumbing.

## Required implementation

1. Add a dependency-free fixture teacher adapter.
   - It may consume the existing synthetic cube-room `.atlas3r` session outputs
     and reconstruct `TeacherPrediction` records from analytic depth/session
     sidecars.
   - It must be clearly named as a fixture/test adapter, not a real external
     model teacher.
   - Every output frame must carry confidence and uncertainty fields.
   - Do not claim measured geometry beyond the synthetic fixture.

2. Wire the runner to produce a Phase 1A cache for the fixture adapter.
   - `atlas3r adapters run --adapter <fixture-name> --input <session.atlas3r>
     --output <cache_dir>` should write a validated teacher cache.
   - External stubs such as VGGT and Depth Pro should continue to fail
     gracefully with adapter name, status, reason, and guidance.

3. Add validation and tests.
   - Fixture adapter status is `available`.
   - Fixture runner writes deterministic cache output.
   - Cache summaries preserve frame IDs, coordinate frame, scale source,
     confidence summaries, and uncertainty summaries.
   - Runner rejects non-synthetic or malformed sessions with explicit errors.
   - Existing Phase 0A-1A tests keep passing.

4. Update docs/status.
   - Rewrite `docs/status/active_task.md` before coding with a concise Phase 1B
     plan and checklist.
   - Update `docs/08_API_CONTRACTS.md` if the fixture adapter or runner cache
     behavior changes public contracts.
   - Append results to `docs/status/progress.md` after verification.
   - Append to `docs/status/decisions.md` only if an interface or format
     decision changed.
   - Replace this file with the Phase 1C prompt before declaring done.

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
