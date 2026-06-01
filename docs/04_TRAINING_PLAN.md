# 04 — Training Plan

## Training philosophy

Train the deployable Atlas3R model as a compact student, not as a giant foundation model. Use a mixture of:

- metric ground truth;
- teacher distillation;
- self-supervised multi-view consistency;
- map-level consistency;
- uncertainty calibration.

The goal is not to win every offline benchmark. The goal is stable real-time streaming reconstruction with bounded memory and reliable confidence estimates.

## Hardware recipe for 2× RTX 4090

- PyTorch 2.x with CUDA, BF16/FP16 mixed precision.
- DDP first; FSDP/ZeRO only if model does not fit.
- Gradient checkpointing in transformer blocks.
- Precompute teacher outputs to NVMe.
- Use small image sizes early: 336×448 or 384×512.
- Progressive sequence length: 2→8→16→32→64→128.
- Keep batch small and accumulate gradients.
- Do not train large teachers inside the loop.

## Stage A — Synthetic sanity pretraining

**Goal:** prove code and losses are correct.

Data:

- synthetic cube rooms;
- simple object meshes;
- exact depth/pose/masks.

Train:

- depth head;
- pointmap head;
- pose head;
- object mask head;
- uncertainty head.

Acceptance:

- model overfits a single cube-room scene;
- pose error trends to near zero;
- mesh from predicted depth matches synthetic mesh within a small multiple of voxel size.

## Stage B — Single-frame and two-view metric geometry

**Goal:** learn high-quality metric depth, intrinsics, normals, and pairwise geometry.

Data:

- ScanNet++, ARKitScenes, Hypersim, TartanAir, Replica/HM3D rendered frames;
- teacher outputs from Depth Pro, MapAnything, VGGT.

Train:

- image encoder;
- depth/normal/intrinsic heads;
- two-view pointmap and relative pose heads;
- confidence calibration.

Important:

- Use metric losses only where true metric scale is known.
- For unmetric teacher outputs, use scale-aligned losses and mark scale source.

## Stage C — Short streaming clips

**Goal:** make SMGT causal and stable on 8–32 frame windows.

Train:

- anchor context;
- local pose-reference window;
- compact trajectory memory;
- dense matching head;
- pose graph pseudo-label distillation.

Losses:

- relative pose;
- global pose within a clip;
- pointmap consistency;
- depth temporal consistency;
- reprojection loss;
- memory consistency.

## Stage D — Long streaming and loop closure

**Goal:** prevent drift and bounded-memory collapse.

Data:

- long indoor/outdoor videos;
- subsampled keyframe trajectories;
- pseudo-labels from LingBot-Map, MASt3R-SLAM, VGGT-SLAM, COLMAP when available.

Train:

- trajectory memory;
- loop retrieval tokens;
- map render feedback;
- confidence-based keyframe selection.

Methods:

- progressive length up to 128–512 frames;
- truncated backprop through time;
- teacher forcing early, scheduled sampling later;
- detach old memory tokens but keep consistency losses against cached predictions.

## Stage E — Object-centric mapping

**Goal:** learn object masks, tracks, 3D object IDs, and dynamic filtering.

Data:

- SAM3 mask/track pseudo-labels;
- ScanNet/ARKit/Replica instance labels;
- OpenMask3D/Mask3D/Open3DIS pseudo-labels;
- synthetic scenes with exact object meshes;
- dynamic video datasets.

Train:

- object mask logits;
- object embeddings;
- 3D instance association;
- dynamic/static head;
- per-object motion head for rigid moving objects.

## Stage F — Map-level fine-tuning

**Goal:** make predictions fuse into stable meshes.

Use a differentiable or semi-differentiable map loop:

1. predict depth/pose/uncertainty for a clip;
2. integrate into a TSDF or surfel buffer;
3. render depth/color/object IDs back into training frames;
4. apply map reprojection, silhouette, normal, and TSDF consistency losses.

This stage is expensive. Run on short clips and selected datasets.

## Stage G — Compression and deployment

**Goal:** reach real-time.

Techniques:

- teacher-student distillation from `smgt_b` to `smgt_s` and `smgt_tiny`;
- structured pruning of attention heads/channels;
- quantization-aware training for INT8 / FP8 where supported;
- fixed-shape export profiles;
- operator replacement for Core ML compatibility;
- lower-rate object/loop modules.

## Training configs to create

```text
configs/train_synthetic.yaml
configs/train_depth_pointmap.yaml
configs/train_streaming_short.yaml
configs/train_streaming_long.yaml
configs/train_objects.yaml
configs/train_map_finetune.yaml
configs/distill_lite.yaml
```

Each config must define:

- datasets and weights;
- image resolution;
- sequence length;
- model variant;
- losses and weights;
- optimizer/scheduler;
- teacher cache paths;
- validation metrics;
- hardware settings.

