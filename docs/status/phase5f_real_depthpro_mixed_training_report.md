# Phase 5F Real Depth Pro Mixed Training Report

Branch: `codex/phase5f-real-depthpro-mixed-training-runtime`

Base commit: `499deb3fbcacf636d1a59f4c15491e25bbe24a7b`

## Result

Phase 5F was not blocked: a real external Depth Pro install/checkpoint ran on
local TUM RGB-D clip-cache RGB frames, wrote pseudo-label teacher-signal caches,
passed the validation quality gate, trained a measured+pseudo temporal student,
exported the mixed checkpoint through the student teacher-signal path, and ran
the streaming map runtime.

It did not improve the Phase 5E measured-only runtime diagnostics. The mixed
checkpoint regressed slightly on measured validation depth/runtime metrics while
producing denser TSDF surfaces.

## Code Changes

- Added `--device auto|cuda|mps|cpu` to `atlas3r teachers run-depth-pro`.
- Split Depth Pro helpers into dependency-safe array/type/model modules.
- Moved real Depth Pro model and tensor inputs to the selected device when Torch
  is available.
- Passed focal length as a tensor for the installed Apple Depth Pro API.
- Added unique-frame prediction reuse by `frame_id` with duplicate RGB/K checks.
- Added deterministic output resize support for Depth Pro depth/confidence/sigma
  arrays back to clip-cache `H x W`.
- Fixed mixed measured+pseudo DataLoader collation when only some caches include
  `pointmap_camera_m`; pointmap loss now uses `pointmap_camera_valid`.
- Documented the Depth Pro runner contract in `docs/08_API_CONTRACTS.md`.

## External Teacher

`depth_pro` import: yes.

Installed package: `C:\Users\Asav\source\repos\slam\external\ml-depth-pro`

Checkpoint used: external `checkpoints/depth_pro.pt` from that local Depth Pro
checkout, passed with `--checkpoint-uri`. No third-party code, checkpoint, cache,
run folder, `.npz`, PLY, or preview artifact was added to git.

## Teacher Caches

Validation Depth Pro cache:

- path: `data/tum_rgbd/freiburg1_xyz_depth_pro_teacher_val`
- size: 23,238,023 bytes
- clips/signals: 56
- clip frame slots: 280
- unique real Depth Pro predictions: 60
- duplicate prediction reuses: 220
- resized prediction arrays: 0

Train Depth Pro cache:

- path: `data/tum_rgbd/freiburg1_xyz_depth_pro_teacher_train`
- size: 109,164,438 bytes
- clips/signals: 256
- clip frame slots: 1,280
- unique real Depth Pro predictions: 264
- duplicate prediction reuses: 1,016
- resized prediction arrays: 0

Depth Pro validation inspection against measured TUM validation cache:

- RMSE: 0.167579472 m
- MAE: 0.122303924 m
- AbsRel: 0.102098948
- valid overlap mean: 74.452976%
- confidence mean: 0.5
- within 1 mm / 5 mm / 1 cm / 5 cm / 10 cm: 0.240594% / 1.166694% / 2.289766% / 19.942103% / 55.304438%
- failure count: no failed clips reported by `inspect-signals`

The safety gate passed (`RMSE < 0.35 m`, `AbsRel < 0.25`).

Validation Depth Pro map diagnostic:

- `runs/phase5f_depthpro_val_map`
- first 16 clips, 80 observations before dedupe, 20 after dedupe
- 3,905 surface points, observed coverage estimate 0.075824

## Mixed Training

Command:

```bash
python -m atlas3r train teacher-signals-temporal --teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_train --teacher-cache data/tum_rgbd/freiburg1_xyz_depth_pro_teacher_train --val-teacher-cache data/tum_rgbd/freiburg1_xyz_measured_teacher_val --val-teacher-cache data/tum_rgbd/freiburg1_xyz_depth_pro_teacher_val --output runs/phase5f_teacher_temporal_v1_measured_depthpro --steps 20000 --batch-size 8 --device cuda --num-workers 0 --learning-rate 0.00025 --log-every 50 --val-every 500 --checkpoint-every 1000 --preview-every 1000 --amp --max-runtime-minutes 330 --hidden-channels 24 --bottleneck-channels 32 --measured-teacher-weight 1.0 --pseudo-teacher-weight 0.15
```

Run summary:

- success: true
- steps completed: 20,000
- checkpoint best step used by runtime: 19,000
- train records: 330 measured + 256 Depth Pro pseudo
- validation records: 56 measured + 56 Depth Pro pseudo
- best mixed validation RMSE: 0.132907202 m
- best mixed validation MAE: 0.092814473 m
- best mixed validation AbsRel: 0.087075680

For comparison, the Phase 5D measured-only training summary recorded best
validation RMSE 0.087864619 m, MAE 0.051004592 m, AbsRel 0.046700478 on the
measured-only validation cache.

## Student Export And Runtime

Student pseudo-label export:

- path: `data/tum_rgbd/freiburg1_xyz_student_teacher_phase5f_val`
- size: 55,540,652 bytes
- clips/signals: 56

Student export inspection against measured TUM validation cache:

- RMSE: 0.091704673 m
- MAE: 0.053628419 m
- AbsRel: 0.049205437
- valid overlap mean: 74.452976%
- confidence mean: 0.668774
- within 1 mm / 5 mm / 1 cm / 5 cm / 10 cm: 1.569558% / 7.823755% / 15.522791% / 65.800012% / 88.552568%

Streaming runtime comparison, 60 unique validation frames, checkpoint
`runs/phase5f_teacher_temporal_v1_measured_depthpro/checkpoint_best.pt`:

| Run | Pose mode | RMSE m | MAE m | AbsRel | Surface points | Coverage | Model inference ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5E measured-only | oracle | 0.090826432 | 0.050450069 | 0.046490420 | 8,622 | 0.084312 | 97.129 |
| Phase 5F mixed | oracle | 0.094295488 | 0.054796543 | 0.050120890 | 8,892 | 0.094852 | 98.387 |
| Phase 5E measured-only | student-relative | 0.090826432 | 0.050450069 | 0.046490420 | 8,732 | 0.085832 | 241.511 |
| Phase 5F mixed | student-relative | 0.094295488 | 0.054796543 | 0.050120890 | 9,329 | 0.097579 | 224.746 |

All runtime outputs remain diagnostic only: `accuracy_report=false`,
`performance_report=false`, `realtime_claim=false`, and `mapping_ready=false`.

## Verification

Passed:

- `python -m ruff format src tests`
- `python -m ruff format --check src tests`
- `python -m ruff check src tests`
- `python -m mypy src`
- `python -m unittest discover -s tests -p 'test_*.py'` (189 tests; pre-existing optional Torch/einops import warning)
- `git diff --check` (CRLF conversion warnings only)
- `python -m atlas3r teachers run-depth-pro --help`
- `python -m atlas3r train teacher-signals-temporal --help`
- `python -m atlas3r runtime stream-student-map --help`

`make` was not available in this Windows shell (`where.exe make` found no
matching executable), so `make test`, `make lint`, and `make typecheck` were not
run.

## Next Blocker

The next blocker is not environment availability. It is training quality:
single-frame Depth Pro pseudo-depth mixed at weight 0.15 did not improve measured
validation/runtime depth quality. Phase 5G should add a real pose-tracking
teacher and learned pose runtime path so Atlas3R can reduce source-pose
dependence instead of only adding single-frame depth pseudo-labels.
