# PLANS.md — Atlas3R Implementation Plan

This is a living plan. Codex should update it only when implementation reality changes.

## Phase 0 — Repo skeleton and synthetic correctness

**Goal:** A tested empty-system skeleton with stable data contracts.

Deliverables:

- package layout under `src/atlas3r/`;
- `FramePacket`, `CameraModel`, `PoseEstimate`, `DepthPrediction`, `ObjectInstance`, `MeshChunk`, `WorldMap` contracts;
- coordinate convention tests;
- synthetic cube-room generator with known camera poses, depth maps, object masks, and mesh;
- `atlas3r smoke synthetic-cube-room` command.

Acceptance:

- `make test` passes;
- synthetic projection/unprojection round trip error < 1e-5 m in float64 tests;
- generated cube-room contract-valid mesh sidecar exports and reloads.

## Phase 1 — Teacher-baseline inference path

**Goal:** Produce pose/depth/pointmap/object-mask predictions using external open models without training Atlas3R yet.

Adapters:

- `LingBotMapAdapter`: streaming RGB pose/depth/pointmap baseline.
- `MapAnythingAdapter`: metric geometry teacher for images/calibration/depth variations.
- `VGGTAdapter`: camera/intrinsics/depth/point tracks teacher.
- `DepthProAdapter`: high-resolution metric depth and focal-length teacher.
- `SAM3Adapter`: image/video mask and object-track teacher.
- Optional: `MASt3RSLAMAdapter`, `SLAM3RAdapter`, `MonST3RAdapter` for benchmark comparison and pseudo-labels.

Acceptance:

- each adapter can be skipped if unavailable;
- at least one adapter can process a short local video and produce a `TeacherPrediction` file;
- adapter outputs validate against `docs/08_API_CONTRACTS.md`.

## Phase 2 — Mapping MVP

**Goal:** Convert depth+pose into a live mesh.

Deliverables:

- CPU TSDF reference integrator for correctness;
- optional CUDA/nvblox-backed TSDF integrator for speed;
- per-voxel uncertainty, color, and object-probability accumulation;
- incremental mesh extraction and chunk invalidation;
- GLB/PLY export.

Acceptance:

- synthetic room reconstruction Chamfer distance < 0.5 voxel;
- object IDs survive mesh extraction;
- mesh export loads in Blender or trimesh.

## Phase 3 — Streaming Metric Geometry Transformer MVP

**Goal:** Implement the compact neural student.

Model:

- DINOv3/ConvNeXt-like image encoder;
- temporal/geometry transformer with anchor context, local pose-reference window, and trajectory memory;
- heads for depth, normal, pointmap, pose, intrinsics, confidence, dynamic mask, object embedding, dense matches.

Acceptance:

- forward pass works on 8-frame clips;
- tensor shapes are tested;
- model can overfit the synthetic cube-room in < 1 hour;
- inference profile reports latency and memory.

## Phase 4 — Training pipeline

**Goal:** Train the student using metric GT and teacher distillation.

Deliverables:

- dataset adapters for ScanNet++, ARKitScenes, ScanNet, Replica/HM3D, Hypersim, TartanAir, CO3D/MegaDepth/RealEstate-style unposed data as available;
- teacher-prediction cache format;
- loss functions from `docs/05_LOSSES.md`;
- distributed training recipe for 2×4090.

Acceptance:

- single-batch train step passes;
- 1k-step overfit debug run decreases all supervised losses;
- validation metrics are written to JSON and TensorBoard/W&B-compatible logs.

## Phase 5 — Object-centric mapping

**Goal:** Split the fused scene into object-level meshes.

Deliverables:

- 2D mask tracking from SAM3 or Atlas3R object head;
- 3D object ID fusion and temporal association;
- per-object TSDF/surfel extraction for movable objects;
- static/dynamic filtering.

Acceptance:

- synthetic objects produce separate meshes;
- mask flicker does not create duplicate object IDs across short occlusions;
- dynamic objects do not poison the static map.

## Phase 6 - Real-time runtime

**Goal:** 30 FPS pose output with live mesh updates.

Runtime scheduler:

- tracker stream: every frame;
- mapper stream: selected keyframes;
- object stream: lower rate or keyframes;
- loop-closure stream: asynchronous;
- exporter stream: chunked mesh updates.

Acceptance on RTX 4090:

- 30 FPS pose for 512×384 input on sample videos;
- mesh update at >= 10 Hz in balanced mode;
- no unbounded memory growth over 10k frames.

Acceptance on Apple M-series:

- `lite` profile runs with Core ML/MPS path;
- explicit FPS and memory report; lower FPS is acceptable unless the specific chip/profile proves 30 FPS.

### Completed Phase 6A-6H product slices

- **6A:** stable measured `atlas3r_recording` boundary plus measured
  recording-to-CPU-TSDF diagnostic fusion.
- **6B:** sensor-folder recording importer and incremental CPU rebuild timing.
- **6C:** persistent dense CPU TSDF incremental backend.
- **6D:** sparse block CPU TSDF backend with online growth and memory stress
  diagnostics.
- **6E:** live capture/replay scheduler boundary: deterministic measured
  recording replay, bounded capture/map queues, explicit drop/keyframe reasons,
  `cpu-sparse` map updates, conservative live replay reports, and dependency-safe
  OpenCV/replay capture adapter status.
- **6F:** observed-only live mesh chunk updates from measured sparse replay:
  dirty sparse block tracking, versioned `block_<x>_<y>_<z>` chunk IDs, NPZ/PLY
  triangle payloads, manifest/update logs, loadability tests, and measured
  replay/profile evidence without RGB-only, hidden-geometry, realtime, or
  accuracy claims.
- **6G:** offline RGB teacher-assisted mapping bridge: RGB-only recording/image
  input through VGGT teacher-pseudo depth/pose/intrinsics, pseudo recording
  artifacts, existing sparse TSDF fusion, observed mesh chunks, and diagnostic
  eval on TUM source measurements without using measured truth for mapping.
- **6H:** multi-window RGB teacher stitching and cache export: adjacent VGGT
  windows are aligned with explicit Sim3 scale, inconsistent windows are
  rejected, stitched teacher pseudo observations drive observed mesh chunks, and
  validated temporal teacher caches are exported for SMGT student training.
- **Core A:** first learned SMGT-tiny diagnostic student mapping: SMGT-tiny
  trains from the Phase 6H pseudo teacher temporal cache, checkpoints carry
  conservative truth flags, and `runtime map-rgb-student` maps real RGB-only
  input through learned depth/pose/confidence predictions without VGGT at
  student inference.
- **Core A2:** SMGT-tiny generalization gauntlet: train/val/heldout temporal
  split manifests, confidence/sigma/dynamic mapping gates, stable training
  metric windows, heldout mapping diagnostics, and long-run map-growth evidence.
  Result: current SMGT-tiny is a diagnostic/toy baseline; do not start
  object/dynamic fusion from it.
- **Core A3:** SMGT-small-v2 measured/pseudo diagnostic student: measured
  temporal caches, mixed measured/pseudo training, RGB phase-correlation pose
  prior plus learned residual, validation-quality checkpoint selection,
  confidence/sigma calibration, and heldout RGB-only mapping. Result: local
  Freiburg heldout gates passed with a validation-selected checkpoint, but final
  SMGT, realtime, RGB-only readiness, object-aware fusion, and accuracy claims
  remain false.

### Next likely Core Phase A4

Broaden SMGT-small-v2 validation and profiling before object/dynamic fusion.
Acceptance should require at least one additional measured heldout sequence,
first-class memory/FPS instrumentation in runtime summaries, calibrated gates
that stay selective across sequences, and no regression of the A3 Freiburg
depth/pose/mesh gates.

## Phase 7 - Evaluation and release gates

**Goal:** Prevent misleading claims.

Reports:

- ATE/RPE pose;
- depth AbsRel/RMSE/δ metrics;
- mesh Chamfer, F-score at 1 mm / 5 mm / 1 cm / 5 cm thresholds;
- 3D object AP/IoU;
- runtime latency percentiles and memory;
- uncertainty calibration.

Acceptance:

- `atlas3r eval` produces a self-contained HTML/Markdown report;
- README claims match measured reports.
