# Phase 5G.1 Multi-Sequence TUM Generalization Report

Branch: `codex/phase5g1-multisequence-tum-generalization`

Base branch: `codex/phase5g-multisequence-pose-odometry`

## Result

Phase 5G.1 trained the existing `TemporalMetricNetV1` SE(3) temporal student on
three measured TUM RGB-D sequences. No new model was added. The only code change
was to remove the sequence-equality blocker in multi-cache teacher-signal
training and to record per-cache/per-sequence counts and validation metrics.

The current pipeline can train on multiple measured TUM teacher caches. It does
not yet generalize cleanly. `freiburg1_xyz` and `freiburg2_xyz` are usable as
diagnostics; `freiburg1_desk` remains bad for depth and much worse for
student-odometry rollout. Student odometry is still too drifty for mapping.

All metrics here are diagnostic run metrics against local TUM teacher caches.
They are not benchmark accuracy, realtime, mapping-ready, or millimeter-accuracy
claims.

## Data Attempts

| Sequence | Download/prep status | Notes |
| --- | --- | --- |
| `freiburg1_xyz` | reused local download; prepared new Phase 5G.1 block manifest and caches | Existing local archive/extract was present. |
| `freiburg1_desk` | downloaded and prepared successfully | Official CLI download wrote `rgbd_dataset_freiburg1_desk.tgz`, ground truth, and extracted folder. |
| `freiburg2_xyz` | downloaded and prepared successfully | Official CLI download wrote a 2.2 GB archive, ground truth, and extracted folder. |
| `freiburg3_long_office_household` | not attempted | Stopped after two additional successful sequences to keep the 30k-step run practical. No availability claim is made. |

Preparation policy used for all prepared sequences:

- manifest split: `--split-policy block --val-fraction 0.15`
- clip cache: `160x120`, clip length `5`, stride `2`, max frame gap `0.12s`
- measured teacher cache: TUM RGB-D sensor depth plus ground-truth `T_world_camera`

Clip and measured teacher-signal counts:

| Sequence | Train clips/signals | Val clips/signals |
| --- | ---: | ---: |
| `freiburg1_xyz` | 336 | 58 |
| `freiburg1_desk` | 251 | 43 |
| `freiburg2_xyz` | 1,556 | 273 |
| Total | 2,143 | 374 |

## Code Changes

- `TeacherSignalTemporalDataset` now allows mixed teacher caches across
  datasets/sequences when split, clip length, and image size match.
- Training summaries now record `cache_records`, `sequence_records`,
  `source_dataset_names`, and `source_sequence_names`.
- Validation JSONL now includes aggregate metrics plus `per_sequence` metrics.
- Added unit coverage for cross-sequence cache loading and validation-summary
  output.
- Updated `docs/08_API_CONTRACTS.md` for the multi-sequence cache contract.

## Training

Command:

```bash
python -m atlas3r train teacher-signals-temporal \
  --teacher-cache data/tum_rgbd/freiburg1_xyz_phase5g1_measured_teacher_train \
  --teacher-cache data/tum_rgbd/freiburg1_desk_phase5g1_measured_teacher_train \
  --teacher-cache data/tum_rgbd/freiburg2_xyz_phase5g1_measured_teacher_train \
  --val-teacher-cache data/tum_rgbd/freiburg1_xyz_phase5g1_measured_teacher_val \
  --val-teacher-cache data/tum_rgbd/freiburg1_desk_phase5g1_measured_teacher_val \
  --val-teacher-cache data/tum_rgbd/freiburg2_xyz_phase5g1_measured_teacher_val \
  --output runs/phase5g1_multisequence_tum_se3 \
  --steps 30000 --batch-size 8 --device cuda --num-workers 2 \
  --learning-rate 0.00025 --log-every 50 --val-every 500 \
  --checkpoint-every 1000 --preview-every 1000 --amp \
  --max-runtime-minutes 360 --hidden-channels 24 --bottleneck-channels 32 \
  --relative-translation-weight 1.0 --relative-rotation-weight 0.1 \
  --se3-pose-weight 1.0
```

Run summary:

| Field | Value |
| --- | ---: |
| Device / AMP | `cuda` / true |
| Steps completed | 30,000 |
| Stop reason | `completed_steps` |
| Best checkpoint step | 24,500 |
| Train records | 2,143 measured, 0 pseudo |
| Validation records | 374 measured, 0 pseudo |

Best aggregate validation:

| Metric | Value |
| --- | ---: |
| Depth RMSE / MAE / AbsRel | 0.265286 m / 0.180751 m / 0.187038 |
| Within 1mm / 5mm / 1cm / 5cm / 10cm | 0.512% / 2.565% / 5.134% / 24.840% / 45.679% |
| Relative translation mean / median / p95 | 0.005221 m / 0.004996 m / 0.009198 m |
| Relative rotation mean / median / p95 | 0.256970 deg / 0.233360 deg / 0.595935 deg |
| ATE-like center mean / median / p95 | 0.007248 m / 0.007308 m / 0.015018 m |
| RPE-like translation mean / median / p95 | 0.003630 m / 0.003372 m / 0.005693 m |
| RPE-like rotation mean / median / p95 | 0.250807 deg / 0.228455 deg / 0.461375 deg |

Best validation by sequence:

| Sequence | Depth RMSE | AbsRel | ATE mean | Relative trans mean | Relative rot mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `freiburg1_xyz` | 0.184351 m | 0.115135 | 0.012682 m | 0.008447 m | 0.373745 deg |
| `freiburg1_desk` | 0.521650 m | 0.723090 | 0.013189 m | 0.010018 m | 0.626150 deg |
| `freiburg2_xyz` | 0.240855 m | 0.117443 | 0.005181 m | 0.003799 m | 0.175429 deg |

## Runtime Evaluation

Student-odometry commands used the best checkpoint, each validation clip cache,
the matching measured teacher cache, `--max-frames 60`, and `--device cuda`.
Oracle pose was also run on `freiburg1_xyz`.

| Sequence / mode | Depth RMSE / MAE / AbsRel | Within 1mm / 5mm / 1cm / 5cm / 10cm |
| --- | ---: | ---: |
| `freiburg1_xyz` student-odometry | 0.186342 m / 0.113286 m / 0.086612 | 0.770% / 3.877% / 7.702% / 36.608% / 64.009% |
| `freiburg1_desk` student-odometry | 0.514743 m / 0.449059 m / 0.719807 | 0.021% / 0.096% / 0.193% / 1.081% / 2.633% |
| `freiburg2_xyz` student-odometry | 0.281236 m / 0.156362 m / 0.083914 | 0.476% / 2.348% / 4.685% / 24.270% / 48.917% |
| `freiburg1_xyz` oracle pose | 0.186342 m / 0.113286 m / 0.086612 | 0.770% / 3.877% / 7.702% / 36.608% / 64.009% |

Pose and map diagnostics:

| Sequence / mode | ATE mean / median / p95 | Rotation mean / median / p95 | RPE trans mean | RPE rot mean | Surface points | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `freiburg1_xyz` student-odometry | 0.137798 m / 0.132161 m / 0.243950 m | 2.568183 / 2.344120 / 6.112305 deg | 0.010458 m | 0.479756 deg | 14,576 | 0.084000 |
| `freiburg1_desk` student-odometry | 0.425468 m / 0.479950 m / 0.637138 m | 14.102981 / 12.800692 / 30.477988 deg | 0.012303 m | 0.750086 deg | 10,456 | 0.088657 |
| `freiburg2_xyz` student-odometry | 0.100351 m / 0.105344 m / 0.178756 m | 1.429469 / 1.370819 / 2.381208 deg | 0.003217 m | 0.279321 deg | 36,738 | 0.126058 |
| `freiburg1_xyz` oracle pose | 0.000000 m / 0.000000 m / 0.000000 m | 0.005460 / 0.003232 / 0.015867 deg | 0.000000 m | 0.005813 deg | 11,937 | 0.075324 |

Latency diagnostics, not performance claims. The first inference sample includes
warmup and skews the mean; p50/p95 are more useful:

| Sequence / mode | Model inference mean / p50 / p95 |
| --- | ---: |
| `freiburg1_xyz` student-odometry | 10.847 ms / 4.385 ms / 4.724 ms |
| `freiburg1_desk` student-odometry | 10.609 ms / 4.375 ms / 4.553 ms |
| `freiburg2_xyz` student-odometry | 12.481 ms / 4.767 ms / 5.955 ms |
| `freiburg1_xyz` oracle pose | 11.594 ms / 5.164 ms / 5.880 ms |

## Comparison To Phase 5G

Phase 5G trained only on the old `freiburg1_xyz` cache and reported best
validation depth RMSE `0.088052` m, runtime depth RMSE `0.090435` m, and
student-odometry ATE mean `0.160803` m / rotation mean `5.388368` deg over 60
validation frames.

Phase 5G.1 is not an apples-to-apples replacement because it uses new block
manifests, stride-2 caches, and multiple sequences. On `freiburg1_xyz`, runtime
depth got worse (`0.186342` m RMSE), but student-odometry pose improved
diagnostically (`0.137798` m ATE mean and `2.568183` deg rotation mean). On
`freiburg2_xyz`, pose was better than `freiburg1_xyz` by ATE mean, but absolute
rollout still drifted to `0.100351` m mean over 60 frames. On `freiburg1_desk`,
both depth and absolute pose were poor.

Answer to the goal questions:

1. Yes, the current SE(3) temporal student can train on multiple measured TUM
   sequences after removing the sequence-equality cache blocker.
2. No clear generalization win. Pose is better than the Phase 5G single-sequence
   report on `freiburg1_xyz` diagnostics, but depth is worse, and the held-out
   style `freiburg1_desk` result is bad.
3. Student odometry is still not remotely mapping-ready. Short-step RPE is
   smaller than absolute drift, but the rollout accumulates large position and
   rotation error, especially on `freiburg1_desk`.
4. The next blocker is external pose/pointmap teacher quality plus data/model
   robustness, not runtime mapping. More measured TUM data alone did not fix the
   drift or the desk depth failure.

## Verification

Commands run:

```bash
python -m ruff format src tests
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
where.exe make
```

Results:

- `python -m ruff format src tests`: passed; one test file reformatted.
- `python -m ruff format --check src tests`: passed; 147 files formatted.
- `python -m ruff check src tests`: passed.
- `python -m mypy src`: passed with no issues in 110 source files.
- `python -m unittest discover -s tests -p "test_*.py"`: passed 198 tests; the
  pre-existing optional Torch/einops import warning appeared.
- `git diff --check`: passed; Git warned changed LF files will convert to CRLF.
- `where.exe make`: no `make` found, so `make test`, `make lint`, and
  `make typecheck` were unavailable in this Windows shell.

## Next Blocker

Phase 5H should generate or ingest stronger external pose/pointmap teachers
with VGGT or LingBot-Map before mesh/object work. The current student-odometry
path is still diagnostic only and too drifty for reliable mapping evaluation.
