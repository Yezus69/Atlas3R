# Active Task

Goal: Phase 5D teacher-weighted temporal student training and mapping loop.

Branch: `codex/phase5d-teacher-weighted-temporal-mapping-training`

Implementation commit: `670aac8`

Checklist:

- [x] Read Phase 5D goal, required docs/status files, teacher-signal validators,
  cache writer, map diagnostics, and existing temporal-v0 source.
- [x] Create branch from `codex/phase5c-external-teacher-runner-bootstrap`.
- [x] Add lazy teacher-signal temporal dataset aligned to source clip caches.
- [x] Add confidence/uncertainty-weighted temporal losses and diagnostics.
- [x] Add one small `TemporalMetricNetV1` model with per-frame
  depth/sigma/confidence and relative translation outputs.
- [x] Add `atlas3r train teacher-signals-temporal`.
- [x] Add `atlas3r teachers run-student-temporal`.
- [x] Add focused unit/smoke tests for dataset, losses, model shapes, CLI help,
  CPU 1-step train, export, inspect, and map diagnostics.
- [x] Run required format, lint, typecheck, unit, and diff checks.
- [x] Attempt real TUM measured teacher training/eval on local CUDA data.
- [x] Write Phase 5D report and update progress/decisions/next-task handoff.
- [x] Commit code, tests, and docs only; generated runs/caches remain ignored.

Verification:

- `python -m ruff format src tests`: passed, 134 files unchanged.
- `python -m ruff format --check src tests`: passed, 134 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, 98 source files checked.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 181 tests;
  existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git emitted CRLF conversion warnings.
- `where.exe make`: no `make` found in this Windows shell.

Real run:

- Measured-only CUDA run completed 20,000 steps on local TUM teacher caches.
- Student cache export, inspect-signals, and map-signals completed on 56 val
  clips.
- External Depth Pro was importable, but no explicit checkpoint URI/env var was
  configured; no external pseudo-label run was faked.
