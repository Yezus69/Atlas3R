# Active Task

Goal: Phase 5C external teacher runner bootstrap.

Branch: `codex/phase5c-external-teacher-runner-bootstrap`

Checklist:

- [x] Read Phase 5C goal, relevant contract/status docs, teacher-signal code,
  clip-cache IO, TUM data helpers, and adapter stubs.
- [x] Create branch from committed Phase 5B.1 cleanup branch.
- [x] Add dependency-safe external teacher runner contracts under
  `atlas3r.teachers.external`.
- [x] Add Depth Pro runner path with dependency-safe status, explicit external
  checkpoint configuration, fake-test execution, validated signal writing, and
  post-run inspect support.
- [x] Add VGGT local-output ingestion scaffold for validated local arrays.
- [x] Wire `atlas3r teachers run-depth-pro` and
  `atlas3r teachers ingest-vggt-local`.
- [x] Update external teacher runner API contract docs.
- [x] Add focused tests for optional dependency safety, unavailable status,
  fake runner cache writing, VGGT local ingest, inspect/map smoke, and CLI help.
- [x] Run required format/lint/type/unit/diff checks.
- [x] Record verification, known gaps, Phase 5D next task, and commit
  code/docs/tests only.

Verification:

- `python -m ruff format src tests`: passed, 128 files unchanged.
- `python -m ruff format --check src tests`: passed, 128 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, 93 source files checked.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 175 tests;
  existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git emitted CRLF conversion warnings.
- `where.exe make`: no `make` found in this Windows shell.

Real-run note: local TUM validation clip cache exists and `depth_pro` imports
from an external repo, but no external checkpoint URI is configured through
`--checkpoint-uri` or `ATLAS3R_DEPTH_PRO_CHECKPOINT`; the real Depth Pro run was
not attempted to avoid implicit weight loading.
