# Phase 4D TUM Eval And V2 Training Report

Status: completed as a diagnostic real-RGBD debug phase. This is not an
accuracy report, not a performance report, and not evidence of mapping or
realtime readiness.

## Scope

- Added and committed Phase 4D code in commit `35652e0`.
- Evaluated the Phase 4C `TinyDepthPoseNet` checkpoint on the block validation
  split with optional CPU TSDF diagnostics.
- Trained one `TinyMetricDepthNetV2` run with RGB plus intrinsics ray channels.
- Evaluated the v2 checkpoint on the same block validation split with optional
  CPU TSDF diagnostics.

## Commands Run

```bash
python -m pip install -e ".[dev,train]"
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
```

`make test`, `make lint`, and `make typecheck` were not run because `make` is
not installed in this Windows shell.

Real-data commands:

```bash
atlas3r eval tum-rgbd-checkpoint --checkpoint runs/tum_rgbd_freiburg1_xyz_overnight/checkpoint_best.pt --manifest data/tum_rgbd/freiburg1_xyz_manifest_block.json --output runs/phase4d_eval_baseline_block --split val --width 160 --height 120 --device cuda --write-tsdf

atlas3r train tum-rgbd-depth-pose --manifest data/tum_rgbd/freiburg1_xyz_manifest_block.json --output runs/tum_rgbd_freiburg1_xyz_v2_5h --model tiny-v2 --depth-loss log_l1 --steps 50000 --batch-size 16 --device cuda --num-workers 4 --learning-rate 0.0005 --width 160 --height 120 --log-every 50 --val-every 500 --checkpoint-every 1000 --preview-every 1000 --seed 1 --amp --max-runtime-minutes 330

atlas3r eval tum-rgbd-checkpoint --checkpoint runs/tum_rgbd_freiburg1_xyz_v2_5h/checkpoint_best.pt --manifest data/tum_rgbd/freiburg1_xyz_manifest_block.json --output runs/phase4d_eval_v2_block --split val --width 160 --height 120 --device cuda --write-tsdf
```

## Test Status

- `python -m ruff format src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed.
- `python -m unittest discover -s tests -p "test_*.py"`: 152 tests passed.
- `git diff --check`: passed.

## Data And Runs

- Manifest: `data/tum_rgbd/freiburg1_xyz_manifest_block.json`.
- Dataset: TUM RGB-D `freiburg1_xyz`.
- Associated selected frames: 398 total, 338 train and 60 validation.
- Split policy: block tail validation to reduce temporal-neighbor leakage.
- CUDA device: `NVIDIA GeForce RTX 4090`.
- Generated artifacts are ignored under `data/` and `runs/`.

## Metrics

Baseline Phase 4C checkpoint on the block validation split:

- Depth RMSE: `0.1463244370` m.
- Depth MAE: `0.0841197038` m.
- Depth AbsRel: `0.0746864870`.
- Camera-center mean/median/p95 error: `0.0225338522` /
  `0.0136320805` / `0.0523134239` m.
- CPU TSDF diagnostic chamfer-like mean: `0.1183717438` m.

TinyMetricDepthNetV2 training:

- Output: `runs/tum_rgbd_freiburg1_xyz_v2_5h`.
- Completed `19009` steps and stopped at `max_runtime_minutes`.
- Best validation step: `17000`.
- Best validation RMSE: `0.0737374956` m.
- Best validation MAE: `0.0425126282` m.
- Best validation AbsRel: `0.0403562384`.

TinyMetricDepthNetV2 checkpoint on the block validation split:

- Depth RMSE: `0.0754485318` m.
- Depth MAE: `0.0422393417` m.
- Depth AbsRel: `0.0396844349`.
- Camera-center mean/median/p95 error: `0.0318729403` /
  `0.0316584148` / `0.0478056282` m.
- CPU TSDF diagnostic chamfer-like mean: `0.0769099724` m.

Interpretation: v2 improved depth and TSDF diagnostic metrics on this debug
split, while camera-center mean/median error worsened relative to the Phase 4C
checkpoint. This supports moving to temporal multi-view data and relative-pose
loss plumbing rather than scaling the single-frame path.

## Artifact Paths

Ignored run outputs:

- `runs/phase4d_eval_baseline_block/summary.json`
- `runs/phase4d_eval_baseline_block/map_comparison.json`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/summary.json`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/checkpoint_best.pt`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/checkpoint_last.pt`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/prediction_sample.npz`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/prediction_preview.html`
- `runs/tum_rgbd_freiburg1_xyz_v2_5h/prediction_preview.svg`
- `runs/phase4d_eval_v2_block/summary.json`
- `runs/phase4d_eval_v2_block/map_comparison.json`

## Known Limitations

- TUM RGB-D sensor depth and pose are used as debug supervision only.
- The tiny models are not usable for mapping or realtime mapping.
- Evaluation measures depth and camera-center translation; it does not evaluate
  full pose accuracy.
- CPU TSDF point-set metrics are diagnostic and depend on voxel size, truncation,
  and sparse surface extraction.
- No external teacher model, object mapping, mesh export, or SMGT transformer was
  added in this phase.

## Next Phase

Start Phase 5A: build the reusable real multi-view clip cache and temporal
geometry training path from the TUM RGB-D manifest, without adding external
model repositories or committing generated data/checkpoints/previews.
