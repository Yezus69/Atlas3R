# Phase 6E Repo Audit

Tracked-file snapshot: 221 files before Phase 6E additions.

## Essential Current Architecture/Source/Test Files

- Root project controls: `AGENTS.md`, `README.md`, `PLANS.md`, `pyproject.toml`,
  `Makefile`, `.gitignore`, `configs/`.
- Architecture and contract docs: `docs/00_*` through `docs/12_*`, especially
  `docs/01_SYSTEM_ARCHITECTURE.md`, `docs/08_API_CONTRACTS.md`, and
  `docs/09_EVALUATION.md`.
- Source packages under `src/atlas3r/`: `api`, `camera`, `data`, `eval`,
  `forge`, `io`, `mapping`, `models`, `pose`, `recording`, `runtime`,
  `teachers`, `training`, and visualization stubs.
- Phase 6E source additions are reachable from CLI/tests:
  `runtime/capture_adapters.py`, `runtime/live_replay.py`,
  `runtime/live_replay_outputs.py`, and `runtime/live_replay_types.py`.
- Tests under `tests/unit/` and `tests/synthetic/` remain essential. Phase 6E
  adds focused adapter and live replay scheduler tests.

## Compact Diagnostic Reports/Status Files

- Keep `docs/status/progress.md`, `active_task.md`, `decisions.md`, and
  `next_task.md` compact.
- Existing phase reports from Phase 4D through 6D are useful provenance and
  should remain unless replaced by a concise index in a later cleanup phase.
- Phase 6E adds `phase6e_live_replay_scheduler_report.md` and this audit.

## Generated Artifacts That Must Not Be Committed

- Ignored generated roots: `build/`, `data/`, `datasets/`, `runs/`,
  `checkpoints/`, `*.atlas3r/`, checkpoint files, tarballs, `.npz`, caches, and
  prediction previews.
- Phase 6E evidence runs belong under ignored `runs/`; no generated sparse TSDF
  state, PLY, JSONL, or report artifacts should be staged from `runs/`.

## Stale Or Duplicate Candidates

- No source, public API, CLI, test, or contract file is provably dead in this
  phase.
- Older status reports are candidates for future summarization only; they are
  not deleted because they document current architecture provenance.
- No third-party repos or model weights are vendored.
