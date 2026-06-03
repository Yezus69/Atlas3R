# Phase 5D Teacher-Weighted Temporal Training Report

Branch: `codex/phase5d-teacher-weighted-temporal-mapping-training`

Implementation commit: `670aac8`

## Scope

Phase 5D adds the first teacher-signal temporal student loop: validated
teacher-signal datasets, confidence/uncertainty-weighted temporal losses,
`TemporalMetricNetV1`, `atlas3r train teacher-signals-temporal`, and
`atlas3r teachers run-student-temporal`. The student export writes pseudo-label
teacher-signal caches only; it does not mark predicted geometry as measured.

## Environment

- Torch available: yes.
- CUDA available: yes.
- GPU used: NVIDIA GeForce RTX 4090.
- `depth_pro` importable: yes.
- `ATLAS3R_DEPTH_PRO_CHECKPOINT`: not set.
- External pseudo-label teacher caches found under `data/tum_rgbd/`: none.

## Teacher Caches

- Train: `data/tum_rgbd/freiburg1_xyz_measured_teacher_train`, 330 clips,
  measured TUM RGB-D sensor depth/pose teacher.
- Validation: `data/tum_rgbd/freiburg1_xyz_measured_teacher_val`, 56 clips,
  measured TUM RGB-D sensor depth/pose teacher.
- External pseudo-labels: skipped because no configured Depth Pro checkpoint URI
  or existing VGGT/Depth Pro cache was available. No external data was faked.

## Commands Run

```bash
python -m atlas3r train teacher-signals-temporal --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_train --val-teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val --output runs/phase5d_teacher_temporal_v1_tum --model temporal-v1 --steps 20000 --batch-size 8 --device cuda --amp --log-every 50 --val-every 500 --checkpoint-every 1000 --preview-every 1000 --max-runtime-minutes 330
python -m atlas3r teachers run-student-temporal --checkpoint runs/phase5d_teacher_temporal_v1_tum/checkpoint_best.pt --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val --output runs/phase5d_student_val_teacher_cache --device cuda --max-clips 56
python -m atlas3r teachers inspect-signals --clip-cache data/tum_rgbd/freiburg1_xyz_clip_cache_val --teacher-cache runs/phase5d_student_val_teacher_cache --output runs/phase5d_student_val_inspect --max-clips 56
python -m atlas3r teachers map-signals --teacher-cache runs/phase5d_student_val_teacher_cache --output runs/phase5d_student_val_map --max-clips 56 --voxel-size-m 0.05
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
where.exe make
```

## Training Result

Run folder: `runs/phase5d_teacher_temporal_v1_tum` (ignored/generated).

- Completed steps: 20,000.
- Stop reason: `completed_steps`.
- Best validation step: 19,000.
- Best validation RMSE: 0.087864619 m.
- Best validation MAE: 0.051004592 m.
- Best validation AbsRel: 0.046700478.
- Best validation within 5 cm: 68.832715%.
- Best validation within 10 cm: 89.795666%.
- Measured batch count: 8.0; pseudo batch count: 0.0.

These are diagnostic validation metrics on local TUM-derived caches, not
benchmark accuracy claims.

## Student Cache Diagnostics

Student cache: `runs/phase5d_student_val_teacher_cache` (ignored/generated).

Inspect-signals summary:

- Selected signals: 56.
- Overlap valid pixels: 4,002,592.
- RMSE: 0.089514971 m.
- MAE: 0.050030827 m.
- AbsRel: 0.046260278.
- Mean confidence: 0.761313686.
- Pose center delta mean: 0.0 m because export uses source clip poses.

Map-signals summary:

- Observations before dedupe: 280.
- Observations after dedupe: 60.
- Duplicate frame count: 220.
- Surface point count: 8,952.
- Voxel size: 0.05 m.
- Observed coverage estimate: 0.082271135.
- Mean uncertainty: 0.065754423 m.
- p95 uncertainty: 0.095919234 m.

## Comparisons

Prior local diagnostic summaries, where available:

- Phase 4D single-frame baseline `TinyDepthPoseNet`:
  RMSE 0.146324437 m, MAE 0.084119704 m, AbsRel 0.074686487.
- Phase 4D single-frame v2 `TinyMetricDepthNetV2`:
  RMSE 0.075448532 m, MAE 0.042239342 m, AbsRel 0.039684435.
- Phase 5A temporal-v0 center-depth model:
  center RMSE 0.163136086 m, center MAE 0.107498553 m,
  center AbsRel 0.099884911.
- Phase 5D temporal-v1 teacher-signal model:
  full-clip validation RMSE 0.087864619 m, MAE 0.051004592 m,
  AbsRel 0.046700478.

These runs use related but not identical evaluation surfaces, so they are useful
for local diagnostics only.

## Verification

- `python -m ruff format src tests`: passed, 134 files unchanged.
- `python -m ruff format --check src tests`: passed.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed, 98 source files checked.
- `python -m unittest discover -s tests -p "test_*.py"`: passed, 181 tests.
  Existing optional Torch/einops import warning appeared.
- `git diff --check`: passed with CRLF conversion warnings only.
- `where.exe make`: no `make` found in this Windows shell; make targets were not
  run.

## Issues And Skips

- First real training launch exposed an older generated-cache path convention:
  measured teacher manifests used repo-relative source clip-cache paths. The
  validator and dataset now prefer cache-relative paths but accept existing
  repo-relative source clip-cache paths when present.
- Initial student map diagnostic failed on overlapping clips because duplicate
  frame IDs had context-dependent predictions. Export now reuses the first
  predicted arrays for repeated `frame_id` values so existing map-signals dedupe
  works without changing the map command.
- No Depth Pro or VGGT real pseudo-labels were run because no checkpoint/output
  was configured.
- `TemporalMetricNetV1` is not final SMGT, not realtime, not mapping-ready, and
  not a benchmarked accuracy/performance model.
