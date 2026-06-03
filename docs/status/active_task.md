# Active Task

Goal: Phase 5B.1 repo slimming and teacher-signal hardening.

Branch: `codex/phase5b1-repo-slim-teacher-hardening`

Source inventory before cleanup:

- `src/atlas3r/teachers/*.py`: 1,365 total lines.
- Teacher files over 400 lines: none.
- `tests/unit/test_teacher_signals.py`: 271 lines.

Source inventory after cleanup:

- `src/atlas3r/teachers/*.py`: 1,359 total lines, net -6.
- Teacher files over 400 lines: none.
- `tests/unit/test_teacher_signals.py`: 436 lines, net +165.

Checklist:

- [x] Read requested Phase 5B.1 goal, compact docs/status files, teacher signal
  source files, CLI registration, and focused tests.
- [x] Create non-main cleanup branch from Phase 5B branch.
- [x] Remove teacher-signal inspection HTML/SVG preview outputs while keeping
  JSON and JSONL diagnostics.
- [x] Deduplicate `map-signals` observations by `frame_id`, reject conflicting
  duplicate payloads, and report before/after counts.
- [x] Make raw local NPZ ingest map `clip_<source_clip_id:06d>.npz` explicitly
  and reject bad, duplicate, out-of-range, or mismatched payloads.
- [x] Resolve relative source clip-cache manifest paths from the teacher cache
  root and reject invalid manifest entries with field paths.
- [x] Update focused tests and concise API/status docs; keep Phase 5C as the
  next handoff without executing it.
- [x] Run required format/lint/type/unit/diff checks plus `make` targets if
  available.
- [x] Record after-inventory, verification results, and commit
  `Slim and harden teacher signal bridge`.

Verification:

- `python -m ruff format src tests`: passed, 122 files unchanged.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, 88 files checked.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 170 tests;
  existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git emitted CRLF conversion warnings.
- `make test`, `make lint`, `make typecheck`: not run; `make` is not installed
  in this Windows shell.

Scope guard: no external teacher runners, model downloads, datasets, training,
new mapping algorithms, or new public feature surfaces beyond the cleanup
contracts above.
