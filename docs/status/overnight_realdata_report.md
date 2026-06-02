# Phase 4C Real-Data Run Report

Branch: `codex/overnight-realdata-tum-rgbd`

Code commit: `0c7b31856e41c87a8da2a25fac4bd2c3c97d16e2`

Latest report commit before this update: `9780ecb`

## Commands Run

- `python -m pip install -e ".[dev,train]"`: passed.
- `python -m ruff format src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 146 tests.
- `python -m atlas3r train tum-rgbd-depth-pose --help`: passed.
- CUDA probe: Torch `2.1.0+cu121`, CUDA available, 3 devices, first device
  `NVIDIA GeForce RTX 4090`.
- `python -m atlas3r datasets tum-rgbd download --sequence freiburg1_xyz
  --output data/tum_rgbd`: first attempt failed with an HTTPS certificate
  verification error, later retry succeeded and extracted official files.
- `python -m atlas3r datasets tum-rgbd prepare --input
  data/tum_rgbd/rgbd_dataset_freiburg1_xyz --output
  data/tum_rgbd/freiburg1_xyz_manifest.json --stride 2 --max-frames 1200`:
  passed.
- CUDA smoke: `atlas3r train tum-rgbd-depth-pose` for 2 steps on the real
  manifest passed.
- Overnight run: `atlas3r train tum-rgbd-depth-pose` for 30,000 steps,
  `batch-size=16`, `160x120`, `device=cuda`, `num-workers=4`, `--amp`,
  `--max-runtime-minutes 480`: passed.
- Real checkpoint diagnostic: `atlas3r smoke checkpoint-tsdf --checkpoint
  runs/tum_rgbd_freiburg1_xyz_overnight/checkpoint_best.pt --input
  runs/tum_rgbd_freiburg1_xyz_overnight/checkpoint_tsdf_input_clip.npz --output
  runs/tum_rgbd_freiburg1_xyz_overnight/checkpoint_tsdf_real_debug --device
  cuda`: passed.

## Real-Data Result

- Download succeeded: yes.
- Manifest frame counts: 398 total, 358 train, 40 validation.
- Long CUDA training: completed 30,000 steps on `NVIDIA GeForce RTX 4090`.
- Best validation metrics: RMSE `0.2116048286` m, MAE `0.1216395150` m,
  AbsRel `0.0996375158`.
- Final train metrics: RMSE `0.2295434624` m, MAE `0.1348846406` m, AbsRel
  `0.1075940728`.
- Checkpoint and preview paths: under ignored
  `runs/tum_rgbd_freiburg1_xyz_overnight/`.
- Real checkpoint TSDF diagnostic: wrote 454 surface points and
  `metric_family=not_evaluated` because the NPZ clip has no target depth.
- Generated datasets, checkpoints, run folders, previews, and NPZ payloads
  committed: no; they are ignored by `.gitignore`.

## Required Artifacts

- `summary.json`: present.
- `metrics.jsonl`: present.
- `validation_metrics.jsonl`: present.
- `checkpoint_last.pt`: present.
- `checkpoint_best.pt`: present.
- `prediction_sample.npz`: present.
- `prediction_preview.html`: present.
- `prediction_preview.svg`: present.

## Blockers

- No active blocker for Phase 4C. The early certificate failure cleared on a
  later retry.

## Next Best Task

Run Phase 4D: load the TUM RGB-D debug checkpoint on held-out TUM RGB frames,
convert predictions to `DepthObservation`, feed CPU TSDF, and write
predicted-vs-ground-truth depth/pose/TSDF diagnostics without accuracy claims.
