# Phase 5A Real Multi-View Forge + Temporal TUM Report

Date: 2026-06-03

Branch: `codex/phase5a-real-multiview-forge-temporal`

## Scope

Phase 5A added the first reusable real multi-view clip-cache forge and a tiny
temporal geometry training path. It did not add external model repositories,
model weights, video decoders, TensorBoard/W&B, CUDA extensions, mesh export, or
mapping/realtime claims.

## Implemented

- `atlas3r forge tum-rgbd-clips`
- `atlas3r train tum-rgbd-temporal`
- `src/atlas3r/forge/clip_cache.py`
- `src/atlas3r/forge/tum_rgbd_clips.py`
- `src/atlas3r/training/tum_clip_dataset.py`
- `src/atlas3r/training/tiny_temporal_geometry_model.py`
- `src/atlas3r/training/temporal_losses.py`
- Focused clip-cache, CLI, optional Torch model/loss, and CPU train-smoke tests.

## Commands Run

Phase 5A real forge:

```bash
python -m atlas3r forge tum-rgbd-clips --manifest data/tum_rgbd/freiburg1_xyz_manifest_block.json --output data/tum_rgbd/freiburg1_xyz_clip_cache_train --split train --clip-length 5 --stride 1 --width 160 --height 120 --max-frame-gap-s 0.12 --write-pointmaps --write-normals
python -m atlas3r forge tum-rgbd-clips --manifest data/tum_rgbd/freiburg1_xyz_manifest_block.json --output data/tum_rgbd/freiburg1_xyz_clip_cache_val --split val --clip-length 5 --stride 1 --width 160 --height 120 --max-frame-gap-s 0.12 --write-pointmaps --write-normals
```

Phase 5A CUDA training:

```bash
python -m atlas3r train tum-rgbd-temporal --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_train/atlas3r_clip_cache_manifest.json --val-clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val/atlas3r_clip_cache_manifest.json --output runs/tum_rgbd_temporal_v0_5h --steps 12000 --batch-size 8 --device cuda --num-workers 4 --learning-rate 0.0003 --log-every 50 --val-every 500 --checkpoint-every 1000 --preview-every 1000 --seed 0 --amp --max-runtime-minutes 330
```

Verification:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
make --version
```

## Test Status

- `python -m ruff format src tests`: passed, 114 files unchanged.
- `python -m ruff format --check src tests`: passed, 114 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, no issues in 81 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 157 tests.
  A pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed with Git CRLF conversion warnings only.
- `make test`, `make lint`, and `make typecheck`: not run because `make` is not
  installed in this Windows shell.

## Clip Cache Stats

- Source manifest: `data/tum_rgbd/freiburg1_xyz_manifest_block.json`
- Source frames: 398 total, 338 train, 60 validation.
- Train cache: 330 clips, first clip ID 0, last clip ID 329.
- Validation cache: 56 clips, first clip ID 0, last clip ID 55.
- Clip length: 5 frames; stride: 1.
- Image size: `160x120`.
- Max frame gap: `0.12` seconds.
- Optional payloads written: camera/world pointmaps and approximate camera
  normals.
- Truth boundary: `diagnostic_only=true`, `accuracy_report=false`,
  `performance_report=false`, `teacher_source=tum_rgbd_sensor_depth_pose`.

## Temporal-V0 Run

Ignored output path: `runs/tum_rgbd_temporal_v0_5h/`

- Device: CUDA with AMP.
- Train clips: 330; validation clips: 56.
- Steps completed: 12,000.
- Stop reason: `completed_steps`.
- Best validation step: 11,500.
- Best validation center-depth RMSE: `0.1631360863` m.
- Best validation center-depth MAE: `0.1074985532` m.
- Best validation center-depth AbsRel: `0.0998849113`.
- Diagnostic center-depth within 10 cm: `63.1456734794` percent.
- Relative translation mean: `0.0180322000` m.
- Relative translation median: `0.0161373891` m.
- Relative translation p95: `0.0362428948` m.
- Relative rotation mean: `0.0` degrees because Phase 5A does not learn
  rotation.

## Phase 4D Comparison

Phase 4D v2 block eval remains the stronger depth diagnostic:

| Run | Depth RMSE m | Depth MAE m | AbsRel | Pose/translation diagnostic |
| --- | ---: | ---: | ---: | --- |
| Phase 4D v2 block eval | `0.0754485318` | `0.0422393417` | `0.0396844349` | absolute camera-center mean `0.0318729403` m |
| Phase 5A temporal-v0 best val | `0.1631360863` | `0.1074985532` | `0.0998849113` | relative translation mean `0.0180322000` m |

These are diagnostic real-data debug metrics, not benchmark accuracy reports.
The Phase 5A temporal model establishes the clip-cache/loss/training path, but
its depth quality is worse than the Phase 4D single-frame v2 checkpoint.

## Ignored Artifacts

The following generated artifacts were intentionally left ignored and must not
be committed:

- `data/tum_rgbd/freiburg1_xyz_clip_cache_train/`
- `data/tum_rgbd/freiburg1_xyz_clip_cache_val/`
- `runs/tum_rgbd_temporal_v0_5h/config.json`
- `runs/tum_rgbd_temporal_v0_5h/metrics.jsonl`
- `runs/tum_rgbd_temporal_v0_5h/validation_metrics.jsonl`
- `runs/tum_rgbd_temporal_v0_5h/summary.json`
- `runs/tum_rgbd_temporal_v0_5h/checkpoint_last.pt`
- `runs/tum_rgbd_temporal_v0_5h/checkpoint_best.pt`
- `runs/tum_rgbd_temporal_v0_5h/prediction_sample.npz`
- `runs/tum_rgbd_temporal_v0_5h/prediction_preview.html`
- `runs/tum_rgbd_temporal_v0_5h/prediction_preview.svg`

## Known Limitations

- Temporal-v0 is a tiny training-path model, not the final SMGT.
- It predicts center-frame depth/sigma/confidence and relative translation only.
- Rotation is not learned in Phase 5A.
- There is no object-aware mapping, mesh export, live runtime, external teacher
  integration, or benchmark report.
- TUM RGB-D sensor depth/pose are used as measured supervision for visible
  frames only; no hidden geometry is marked as measured.

## Next Phase Rationale

Relative translation diagnostics are low enough to justify keeping the temporal
loss path, but the depth comparison shows that better teacher signals are needed.
Phase 5B should add dependency-isolated external teacher output ingestion into
the same clip-cache/teacher-signal format, starting with local-folder contracts
and no vendored repos or weights.
