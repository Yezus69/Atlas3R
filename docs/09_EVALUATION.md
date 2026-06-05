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

### Gate 2: student overfit

- model overfits synthetic sequence;
- all heads produce correct shapes;
- losses decrease predictably.

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
