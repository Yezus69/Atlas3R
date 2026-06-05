# Repo Reset Report

## Inventory

| Metric | Before | After |
| --- | ---: | ---: |
| Tracked files | 299 | 49 |
| Approx text LOC | 61,000 | 2,896 |
| Approx Python LOC | 54,680 | 2,053 |

- Tracked file reduction: about 84%.
- Text LOC reduction: about 95%.
- Python LOC reduction: about 96%.

## Deleted Major Modules

- `src/atlas3r/training/`: SMGT, TUM, synthetic, temporal, checkpoint, loss,
  and cache training code.
- `src/atlas3r/runtime/`: measured replay, live scheduler, RGB teacher/student
  runtime mappers, student stream runtime, and sparse stress commands.
- `src/atlas3r/models/smgt/` and old `src/atlas3r/models/student/`: stale
  student models and checkpoint surfaces.
- `src/atlas3r/eval/`, `src/atlas3r/forge/`, old TUM data surfaces, old
  teacher signal pipelines, old session/TSDF smoke plumbing.
- Old `cli_*.py` command fan-out.
- Old phase reports under `docs/status/`.
- Training/runtime/export config YAMLs under `configs/`.
- Old synthetic/integration/unit tests for deleted features.

## Kept Modules

- `src/atlas3r/contracts/`: coordinate, frame, camera, pose, proposal,
  world-state, map-artifact, and truth-boundary contracts.
- `src/atlas3r/input/`: PPM sequence loading, video input inspection, recording
  manifest helpers, and camera metadata helpers.
- `src/atlas3r/teachers/`: dependency-safe witness registry with unavailable
  statuses and install hints.
- `src/atlas3r/offline/`: Offline V0 quality report skeleton only.
- `src/atlas3r/mapping/`: minimal NPZ/PLY artifact writers for inspection.
- `tests/unit/`: focused tests for the kept foundation.

## CLI Surface

Removed old commands include `runtime map-rgb-student`,
`runtime map-rgb-student-v2`, `runtime map-rgb-teacher`,
`runtime live-replay-recording`, `runtime fuse-recording`, `train smgt-tiny`,
`train smgt-v2`, `train build-measured-temporal-cache`,
`eval smgt-v2-calibrate-gate`, TUM train/eval/forge commands, old adapter run
commands, and old TSDF/session/runtime smoke commands.

Active CLI surface:

```bash
python -m atlas3r --help
python -m atlas3r offline --help
python -m atlas3r offline inspect-video --input <mp4-or-ppm-folder> --output <run>
python -m atlas3r teachers list
python -m atlas3r smoke contracts
```

## Generated Artifacts Removed

- Removed `runs/`, `build/`, `.mypy_cache/`, `.ruff_cache/`, and all
  `__pycache__/` trees found in the repo.
- No tracked checkpoints, model weights, teacher caches, generated runs,
  external repos, build products, or deployment artifacts remain.
- Ignored local `data/` was left untouched because it may contain user datasets
  rather than generated build output.

## Branch Cleanup

- Deleted 22 safe merged local stale branches.
- Kept `codex/offline-world-builder-repo-reset` and `main`.
- Kept `codex/phase4d-tum-eval-v2-training` because `git branch -d` refused
  deletion due to upstream merge state.
- Remote branches were not deleted. Suggestions are in
  `docs/status/branch_prune_suggestions.md`.

## Verification Results

Passed:

- `python -m ruff format src tests`
- `python -m ruff format --check src tests`
- `python -m ruff check src tests`
- `python -m mypy src`
- `python -m unittest discover -s tests -p "test_*.py"`: 16 tests
- `git diff --check`
- `python -m atlas3r --help`
- `python -m atlas3r smoke contracts`
- `python -m atlas3r teachers list`

Unavailable:

- `make test`, `make lint`, and `make typecheck` did not run because `make` is
  not installed on this Windows host (`make` is not recognized).

## Remaining Cleanup Debt

- `docs/status/active_task.md` still tracks this reset until the commit lands.
- Local ignored `data/` remains and should be reviewed manually before deletion.
- A local `codex/phase4d-tum-eval-v2-training` branch remains for upstream-state
  safety.
- Offline V0 still needs MP4 decoding, keyframe extraction, proposal cache
  schemas, and a real quality report implementation.
