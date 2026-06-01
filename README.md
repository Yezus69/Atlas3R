# Atlas3R: Real-Time RGB Neural Metric Mapping Architecture

**Purpose.** This repo blueprint tells Codex how to build a real-time RGB 3D environment mapping system from an empty repository. The target system ingests arbitrary RGB video or a live camera feed and continuously outputs:

1. camera pose in a metric world frame,
2. dense scene geometry,
3. object instances with 3D meshes,
4. incremental triangle/polygon mesh chunks that can be loaded by game engines.

The architecture is deliberately **mostly neural** but not purely neural. The high-probability design is a neural geometry front end plus a small geometry/map back end. A pure RGB-to-mesh transformer without geometric checks is too risky for the accuracy target.

## Non-negotiable truth boundary

The requirement “any RGB video” and “physically accurate to millimeter level” cannot be guaranteed simultaneously. Monocular RGB has intrinsic scale/depth ambiguity, surfaces can be hidden, rolling shutter and motion blur destroy measurements, and reflective/transparent/textureless surfaces are underconstrained. Atlas3R must therefore:

- output uncertainty for every pose, depth pixel, voxel, object, and mesh chunk;
- mark unobserved or hallucinated/completed surfaces as **predicted**, not measured;
- only claim millimeter-level accuracy for validated local regions with calibrated camera metadata, sufficient parallax, static visible surfaces, high resolution, low blur, and a benchmark report against metric ground truth;
- degrade gracefully when the input does not contain enough geometric information.

See `docs/00_FEASIBILITY_AND_TRUTH.md` before implementing anything.

## Architecture summary

```text
RGB frame/video
  -> frame IO + camera metadata + lens/rolling-shutter normalization
  -> Streaming Metric Geometry Transformer (SMGT)
       outputs depth, normals, confidence, pointmaps, intrinsics, camera pose,
       object embeddings, dynamic masks, dense matches
  -> lightweight pose graph + loop closure + map reprojection correction
  -> object-aware GPU TSDF / surfel fusion
  -> incremental marching-cubes mesh extraction + object mesh splitting
  -> glTF/GLB/USDZ/OBJ export + live API
```

Core idea: use open models as teachers and initialization, then train a compact streaming student that runs at 30 FPS. The runtime model should be smaller than the teachers. Teachers are used offline to generate pseudo-ground truth and consistency targets.

## Why this design has the highest chance of working

- **Streaming geometry priors.** LingBot-Map proves a bounded-memory streaming transformer can reconstruct long RGB videos using anchor context, local pose-reference windows, and trajectory memory.
- **Metric multi-view priors.** MapAnything and VGGT provide strong feed-forward camera/depth/pointmap supervision.
- **Fast dense SLAM precedent.** MASt3R-SLAM, SLAM3R, and VGGT-SLAM show that learned dense reconstruction can be tied into real-time SLAM-like systems.
- **Sharp metric depth.** Depth Pro provides high-resolution metric depth and focal-length supervision, especially useful for boundaries.
- **Object grouping.** SAM 3 / SAM 3.1, DINOv3, Mask3D, OpenMask3D, and Open3DIS provide the object and open-vocabulary supervision needed to split the mesh into objects.
- **Mesh reliability.** A TSDF/surfel map with incremental meshing is still the most reliable way to turn noisy per-frame depth into game-engine triangles.

## Repository layout Codex must create

```text
atlas3r/
  AGENTS.md                         # Codex operating rules
  README.md                         # project overview and quick start
  PLANS.md                          # living implementation plan
  pyproject.toml                    # Python package metadata
  Makefile                          # test/lint/profile commands
  configs/
    train_smgt.yaml
    infer_realtime.yaml
    datasets.yaml
    export_tensorrt.yaml
    export_coreml.yaml
  src/atlas3r/
    __init__.py
    api/                            # public Python API and server contracts
    camera/                         # intrinsics, distortion, rolling shutter
    data/                           # dataset adapters and unified schema
    models/                         # SMGT, heads, model adapters, exporters
    mapping/                        # TSDF/surfel fusion, meshing, chunks
    objects/                        # 2D/3D instance tracking and mesh splitting
    pose/                           # pose graph, loop closure, tracking filters
    runtime/                        # live camera, video, async scheduler
    training/                       # losses, distillation, trainers
    eval/                           # metrics and benchmark runners
    viz/                            # rerun/open3d viewer adapters
  tests/
    unit/
    integration/
    synthetic/
  docs/
    ...                             # architecture docs in this package
```

## Implementation order

1. **Skeleton and contracts.** Create typed data contracts, coordinate conventions, synthetic cube-room tests, and no-model baseline.
2. **Teacher baseline.** Wrap existing models as adapters: LingBot-Map / MapAnything / VGGT / Depth Pro / SAM3. Do not fork their code into this repo.
3. **Mapping MVP.** Feed teacher depth+pose into GPU/CPU TSDF, export GLB/PLY, pass synthetic geometry tests.
4. **Streaming student.** Implement SMGT and train it using metric datasets plus teacher distillation.
5. **Object-centric map.** Fuse SAM/DINO object tracks into 3D object IDs and split meshes.
6. **Optimization.** TensorRT for NVIDIA, Core ML / MPS path for Apple Silicon, plus quantized lite student.
7. **Benchmarks.** No accuracy claims unless `atlas3r eval` produces reports.

## Hardware assumptions

Training target: one workstation with 2× RTX 4090. Use teacher-output precomputation, mixed precision, gradient checkpointing, DDP/FSDP where needed, and small student variants. Do not attempt to train a 1B+ parameter model end-to-end on 2×4090 without precomputed teachers.

Runtime targets:

- **RTX 4090:** 512×384 or 640×384 input, 30 FPS camera pose, 10–30 Hz depth/map update depending on quality settings.
- **Apple M-series:** `lite` model, 384×288 or 448×336, Core ML/MPS path, object segmentation and loop closure scheduled below frame rate.

## Read this first

- `AGENTS.md`: rules for Codex so it does not flood context with useless code.
- `PLANS.md`: phased implementation plan.
- `docs/01_SYSTEM_ARCHITECTURE.md`: full dataflow and modules.
- `docs/05_LOSSES.md`: training objective.
- `docs/08_API_CONTRACTS.md`: exact outputs and coordinate conventions.
- `docs/09_EVALUATION.md`: acceptance tests and benchmark gates.

