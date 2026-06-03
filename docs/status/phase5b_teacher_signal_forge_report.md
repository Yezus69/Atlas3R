# Phase 5B Teacher Signal Forge Report

## Changed Files

- `src/atlas3r/teachers/`: added teacher-signal cache validation, measured TUM
  forge, local-folder ingest, teacher-vs-clip inspection, and TSDF map bridge.
- `src/atlas3r/cli_teachers.py`, `src/atlas3r/cli.py`: added
  `atlas3r teachers forge-measured-tum|ingest-local|inspect-signals|map-signals`.
- `tests/unit/test_teacher_signals.py`: covers payload/manifest validation,
  traversal rejection, measured forge, local ingest, inspection, mapping, CLI
  help, and Phase 5C handoff.
- `docs/08_API_CONTRACTS.md`, `docs/status/*`: documented the stable
  teacher-signal cache and updated compact status/handoff state.

## Source-Line Audit

Preflight found existing over-450-line files in contracts, TUM data, mapping
sidecar/inspection, runtime inspection, checkpoint inference, and temporal
training. No obvious generated or duplicate Phase 5A source was found.

New source files stay at or below the threshold after splitting helpers:
`signals.py` 450, `measured_tum.py` 403, `map_eval.py` 439,
`_diagnostic_helpers.py` 126, `_map_bridge_helpers.py` 108, and
`cli_teachers.py` 153.

## Verification

- `python -m ruff format src tests`: passed, 122 files unchanged.
- `python -m ruff format --check src tests`: passed, 122 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, no issues in 88 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 164 tests.
  A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed with Git LF-to-CRLF working-tree warnings.
- `make test`, `make lint`, `make typecheck`: not run; `make` is not installed
  in this Windows shell.

## Real-Data Diagnostics

Local Phase 5A TUM clip caches existed and were used.

- Train measured teacher cache:
  `data/tum_rgbd/freiburg1_xyz_measured_teacher_train`, 330 signals.
- Validation measured teacher cache:
  `data/tum_rgbd/freiburg1_xyz_measured_teacher_val`, 56 signals.
- Validation `inspect-signals --max-clips 32`: overlap pixels 2,272,345,
  RMSE 0.0 m, MAE 0.0 m, AbsRel 0.0, 100% within 1 mm/5 mm/1 cm/5 cm/10 cm,
  mean valid overlap 73.9696%, mean confidence 0.7397, pose-center delta 0.0 m.
- Validation `map-signals --max-clips 16 --voxel-size-m 0.05`: TSDF surface
  points 2,661, unique source frame IDs 338-357, observed coverage estimate
  0.0373748454, uncertainty mean 0.0369071745 m, p95 0.0625409479 m.

Generated teacher caches and run outputs are ignored under `data/` and `runs/`.

## Limitations

- The measured TUM forge is measured visible RGB-D depth only; it does not mark
  hidden or completed geometry as measured.
- `ingest-local` validates/copies local payloads only and does not run external
  Depth Pro, VGGT, LingBot-Map, SAM, DINO, or other model code.
- `map-signals` is a CPU TSDF diagnostic bridge, not a mapping readiness,
  realtime, benchmark accuracy, or millimeter-accuracy report.
- Optional `--teacher-cache` temporal training was deferred; it should wait
  until Phase 5C validates external teacher-signal runner outputs.
