# 09 — Evaluation, Benchmarks, and Acceptance Gates

## Evaluation principle

No geometry or accuracy claim is allowed without an evaluation report. Atlas3R must make it easy to prove or disprove claims.

## Benchmark categories

### Pose

Metrics:

- ATE RMSE after SE(3) alignment for metric sequences;
- ATE RMSE after Sim(3) alignment for monocular/up-to-scale sequences;
- RPE translation/rotation over fixed intervals;
- tracking lost count;
- relocalization success;
- drift per meter.

Datasets:

- ScanNet/ScanNet++ sequences;
- ARKitScenes;
- TUM RGB-D if converted to RGB-only input for evaluation;
- ETH3D/Tanks and Temples for multi-view;
- synthetic long trajectories.

### Depth

Metrics:

- AbsRel;
- SqRel;
- RMSE;
- log RMSE;
- δ1/δ2/δ3;
- boundary F-score;
- depth temporal consistency;
- uncertainty calibration.

### Mesh/reconstruction

Metrics:

- Chamfer distance;
- precision/recall/F-score at thresholds: 1 mm, 5 mm, 1 cm, 5 cm;
- completeness;
- normal consistency;
- mesh watertightness for observed regions;
- chunk update stability;
- observed mesh chunk count and active/deleted chunk counts;
- total vertex/triangle count;
- chunk ID/version stability across updates;
- mesh chunk update latency p50/p95/max;
- NPZ/PLY loadability and payload byte counts;
- bbox validity and triangle index validity;
- observed-only truth flags with no hidden-geometry/RGB-only/realtime claim;
- object boundary preservation.

Important: 1 mm metrics only make sense when GT and camera resolution/calibration support it. For most room-scale RGB videos, 5 mm / 1 cm / 5 cm thresholds are more meaningful.

### Object instances

Metrics:

- 2D mask AP/IoU;
- 3D instance AP/IoU;
- object ID switches;
- track fragmentation;
- 3D bounding box error;
- object mesh Chamfer against known object meshes where available.

### Runtime

Metrics:

- end-to-end FPS;
- pose latency p50/p90/p99;
- map update latency p50/p90/p99;
- mesh update Hz;
- GPU memory;
- CPU memory;
- power estimate where available;
- long-run memory slope over 10k frames.

Scheduler diagnostics:

- target FPS versus measured processing latency;
- capture/map queue depths and configured bounds;
- dropped frame/keyframe counts with explicit reasons;
- keyframe selection counts and reasons;
- map update latency and sparse active block/voxel counters;
- memory counters showing no unbounded growth.

These diagnostics are not accuracy reports. They also are not realtime claims
unless the run is intentionally defined and recorded as a performance benchmark.

## Acceptance gates by milestone

### Gate 0: synthetic geometry

- projection/unprojection unit tests pass;
- known cube-room mesh recovered within <0.5 voxel mean surface error;
- object IDs correct on synthetic objects.

### Gate 1: teacher baseline

- a short real video produces poses, depth, and mesh;
- no crashes when optional teachers are missing;
- output metadata marks scale source honestly.

Phase 6G RGB teacher diagnostics satisfy only the teacher-baseline evidence
slice. A `runtime map-rgb-teacher` run may report:

- teacher-pseudo frame count, window count, depth valid pixel ratio, sparse TSDF
  active blocks/voxels, surface point count, mesh chunk/update counts, and
  vertex/triangle counts;
- teacher inference, map update, mesh update, and total pipeline latency
  percentiles as diagnostics, not realtime claims;
- if the RGB source is a measured recording, eval-only depth AbsRel/RMSE after
  median scale alignment and Sim3-aligned camera-center ATE. These metrics do
  not change the map and are not benchmark accuracy claims.

Required truth boundary: teacher-pseudo depth/pose are not measured geometry;
measured depth/pose are false for mapping, may be true only for eval sidecars,
and `metric_scale_source=teacher_scale_unverified` unless a named calibration
or benchmark report proves metric scale.

Phase 6H adds a teacher-baseline stitching/cache evidence slice, not a new
accuracy gate. A stitched `runtime map-rgb-teacher` run may compare against a
`--stitch-windows none` baseline using:

- accepted/rejected stitch edges and rejected windows;
- pseudo submap count;
- overlap camera-center RMSE and Sim3 scale distribution;
- boundary camera-center jump mean/p95/max at teacher window boundaries;
- mesh chunk and vertex/triangle counts;
- optional eval-only ATE/depth metrics against source recording measurements.

The expected Phase 6H quality target is explicit: stitching should improve at
least one tracked diagnostic against the no-stitch baseline, or the report must
mark the stitched path as worse and keep it diagnostic. The Freiburg Phase 6H
run improved eval-only Sim3 camera-center ATE RMSE but worsened boundary jump
metrics, so it remains evidence for cache/stitch plumbing only.

Validated teacher temporal caches are pseudo-label training data. Inspectors
must reject unsafe paths, invalid arrays, missing truth flags, measured-label
mislabeling, or model-weight payloads. Default pseudo target weights must remain
lower than measured target weights.

### Gate 2: student overfit

- model overfits synthetic sequence;
- all heads produce correct shapes;
- losses decrease predictably.

Core Phase A provides diagnostic Gate 2 evidence for a real pseudo-label cache,
not a benchmark accuracy claim. `train smgt-tiny` overfit on four Phase 6H
Freiburg teacher-cache clips for 1000 steps reduced first/final 100-step
`loss_total` means from `0.0429` to `-0.1272`. The weighted 3000-step run on
the same cache reduced first/final 100-step `loss_total` means from `0.1749`
to `-0.0659`. `runtime map-rgb-student` then mapped 64 RGB frames without VGGT
at student inference and produced nonzero observed mesh chunks. Eval-only
source measurements reported depth AbsRel/RMSE `0.0899 / 0.3042 m` after
median scale alignment, Sim3 camera-center ATE RMSE `0.0905 m`, and the student
beat constant-depth and no-motion baselines. These are diagnostics only:
pseudo training, unverified RGB prior scale, no object-aware fusion, no
realtime proof, and no final RGB-only readiness claim.

Core Phase A2 hardens this gate with strict temporal train/val/heldout splits,
stable metric windows, confidence/sigma/dynamic mapping gates, heldout RGB-only
mapping, and long-run diagnostics. The 5000-step Freiburg A2 run reached final
heldout teacher-cache depth AbsRel `0.1889`, but depth remained worse than the
constant-depth baseline ratio (`1.8837`) and pose was `22.5262x` the no-motion
baseline. The validation-selected checkpoint produced zero heldout mesh chunks;
the trained last checkpoint produced mesh but mapped `0.9993` of pixels under
the stricter gate. Gate 2 therefore marks current SMGT-tiny as diagnostic/toy,
not ready for object/dynamic fusion.

Core Phase A3 replaces the active diagnostic student with SMGT-small-v2 trained
from measured Freiburg train caches plus a low-weight pseudo cache. The 5000-step
run selected step 1500 by validation quality: depth AbsRel `0.0621` versus
constant baseline `0.1834`, pose center ratio `0.6968` versus no-motion, and
nonzero mesh potential. Calibrated heldout RGB-only mapping produced 32 mesh
chunks, mapped `0.3112` of pixels, and eval-only depth/pose both beat their
baselines. This is still local diagnostic evidence, not a benchmark accuracy,
realtime, final SMGT, or RGB-only-readiness claim.

### Gate 3: real validation

- student beats a naive DepthPro+PnP baseline on held-out validation;
- no unbounded memory growth over a long clip;
- uncertainty correlates with error.

### Gate 4: 30 FPS profile

On RTX 4090:

- pose update >= 30 FPS at target resolution;
- map update >= 10 Hz in balanced mode;
- p99 pose latency below one frame budget unless explicitly buffering;
- no memory leak over 10k frames.

### Gate 5: accuracy claims

- If claiming “mm-level” in a local mode, report F-score/Chamfer at 1 mm and 5 mm on a named calibrated dataset/capture.
- If only RGB prior scale was used, never label output as measured millimeter accurate.

## Evaluation command design

Create:

```bash
atlas3r eval pose --dataset scannetpp --split val --checkpoint path
atlas3r eval depth --dataset arkitscenes --split val --checkpoint path
atlas3r eval mesh --dataset synthetic_cube --checkpoint path
atlas3r eval runtime --video sample.mp4 --profile balanced
atlas3r eval report --run-dir runs/eval/<id>
```

Reports must include:

- model checkpoint hash;
- git commit;
- dataset version;
- command line;
- hardware;
- metrics table;
- warnings and failure cases.

## Failure-case test set

Maintain a small curated set:

- no parallax video;
- fast motion blur;
- rolling shutter pan;
- glass/mirror scene;
- white wall / low texture;
- moving person covering most frame;
- changing exposure;
- repeated pattern corridor;
- unknown cropped video with wrong focal metadata.

Expected behavior is not always high accuracy. Expected behavior is honest uncertainty and no catastrophic map poisoning.
