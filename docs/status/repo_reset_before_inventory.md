# Repo Reset Before Inventory

- Branch: `codex/offline-world-builder-repo-reset`
- Starting commit: `cfd6a80228e48b22ce3661babae1158b4450ba0f`
- Tracked files: 299
- Approximate text LOC: 61,000
- Approximate Python LOC: 54,680
- Top-level directories: `.git`, `.mypy_cache`, `.ruff_cache`, `build`,
  `configs`, `data`, `docs`, `runs`, `src`, `tests`

## Command Surface Found

- Root CLI: `python -m atlas3r --help`
- Smoke/inspect: synthetic cube room, TSDF cube room, teacher-cache TSDF,
  checkpoint TSDF, runtime fixture, session/cache/map inspection.
- Runtime: `fuse-recording`, `live-replay-recording`, `map-rgb-teacher`,
  `map-rgb-student`, `map-rgb-student-v2`, capture adapters, sparse stress.
- Training/eval/forge: `smgt-tiny`, `smgt-v2`, TUM RGB-D training/eval,
  synthetic overfit, teacher-signal temporal training, measured/pseudo temporal
  cache building, calibration gates, TUM clip forge.
- Teachers/adapters: Depth Pro, VGGT, measured TUM, teacher-signal cache, local
  ingestion, student temporal runner.
- Dataset/config surfaces: TUM RGB-D download/prepare plus realtime/export
  config YAMLs.

## Obvious Stale Areas

- `src/atlas3r/training/` student training, checkpoints, losses, temporal cache
  code.
- `src/atlas3r/runtime/` measured replay, live scheduler, RGB teacher/student
  mapping, student stream runtime.
- `src/atlas3r/models/smgt/` SMGT-tiny and SMGT-small-v2 model code.
- `src/atlas3r/eval/`, `src/atlas3r/forge/`, TUM dataset and teacher-signal
  pipelines tied to old student phases.
- Large old CLI fan-out in `cli_*.py` and `src/atlas3r/cli.py`.
- Old phase reports under `docs/status/`.
- Tracked config files for abandoned training/runtime/export commands.
- Local generated directories: `runs/`, `build/`, `.mypy_cache/`,
  `.ruff_cache/`, and `__pycache__/` trees.
