# Active task

Goal: Phase 3B.1 repo context-budget cleanup only.

Checklist:

- [x] Read only the allowed docs and `pyproject.toml`.
- [x] Add explicit context-budget rules to `AGENTS.md`.
- [x] Compact `docs/status/progress.md` into a current-state summary.
- [x] Compact `docs/status/decisions.md` into an ADR index.
- [x] Compact `docs/08_API_CONTRACTS.md` without changing public schemas.
- [x] Keep `docs/status/next_task.md` pointed at Phase 3C.
- [x] Add dependency-free line-budget guard tests.
- [x] Run Ruff, mypy, unittest discovery, `git diff --check`, and available make commands.

Exclusions: no feature code, model code, training, datasets, downloads, heavy
dependencies, video/image decoding, runtime mapping changes, TSDF changes,
inspection bundles, generated artifacts, or public API/CLI removals.
