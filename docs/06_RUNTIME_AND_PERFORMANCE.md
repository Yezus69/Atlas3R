# 06 — Runtime, Performance, and Deployment

## Runtime principle

Do not make every module run every frame. For 30 FPS, separate **pose FPS** from **map update FPS** while keeping the output live.

- Pose stream: every frame, hard real-time target.
- Depth/geometry stream: every frame in performance mode or keyframes in balanced mode.
- Object segmentation stream: keyframes or lower rate unless GPU budget allows.
- Loop closure: asynchronous.
- Mesh extraction/export: chunked and asynchronous.

## Pipeline schedule

```text
Frame t arrives
  A preprocess frame t
  B run SMGT tracker path on frame t
  C output PoseEstimate[t]
  D if keyframe: run full depth/object/match heads
  E integrate depth into TSDF
  F update dirty mesh chunks
  G emit MapUpdate events
```

## Latency budget targets

### RTX 4090 balanced profile

Target input: 512×384 or 640×384.

| Component | Target average |
|---|---:|
| decode + preprocess | 1–3 ms |
| SMGT tracker path | 6–12 ms |
| full depth/point/object on keyframe | 10–25 ms async |
| local pose graph update | 1–4 ms |
| TSDF integration | 1–6 ms |
| dirty chunk meshing | 2–10 ms async |
| API emit | <1 ms |

Pose must remain 30 FPS even when full mapping runs lower than frame rate.

### Apple M-series lite profile

Target input: 384×288 or 448×336.

- Use `smgt_s` or `smgt_tiny`.
- Prefer ConvNeXt/DINOv3-small-style encoder over huge ViT.
- Use Core ML-friendly layers.
- Run object segmentation on keyframes.
- Run loop closure in idle windows.
- Report actual FPS by chip; do not claim 30 FPS on all M-series chips.

## Model export

### NVIDIA path

1. Train in PyTorch.
2. Freeze model variant and fixed input shape profiles.
3. Export to ONNX with dynamic batch disabled for production.
4. Build TensorRT engines for:
   - tracker-only path;
   - full mapping path;
   - object head path.
5. Use FP16/BF16 first; use INT8/QAT only after accuracy validation.

### Apple path

1. Keep a Core ML-friendly model variant.
2. Avoid custom ops in the production path.
3. Convert with coremltools.
4. Use MLProgram for larger models.
5. Split model functions if that improves memory reuse:
   - encoder;
   - tracker head;
   - depth/object heads.
6. Validate numerics against PyTorch outputs.

## Memory design

The system must run for >10k frames without unbounded memory growth.

Keep:

- anchor keyframes: fixed small count;
- local keyframes: sliding window;
- trajectory memory: compact tokens per keyframe;
- loop database: compressed global descriptors and selected keyframe thumbnails/tokens;
- map: hashed voxel blocks with eviction/export strategy for huge scenes.

Do not keep all dense image tokens forever.

## Keyframe policy

Default thresholds:

```yaml
keyframe:
  min_interval_frames: 3
  max_interval_frames: 30
  translation_baseline_ratio: 0.03   # relative to median depth
  rotation_deg: 8.0
  overlap_min: 0.65
  uncertainty_trigger: true
  new_object_trigger: true
```

The keyframe selector should be learned later, but start with deterministic thresholds for debugging.

## Quality profiles

```yaml
profiles:
  fast_pose:
    depth_every_n: 4
    object_every_n: 15
    voxel_size_m: 0.02
    mesh_hz: 5
  balanced:
    depth_every_n: 2
    object_every_n: 6
    voxel_size_m: 0.01
    mesh_hz: 10
  precision_local:
    depth_every_n: 1
    object_every_n: 3
    voxel_size_m: 0.002
    mesh_hz: 5
    bounded_volume_m: [3, 3, 3]
```

## Runtime failure handling

If tracking confidence drops:

1. continue emitting poses with `tracking_state=LOW_CONFIDENCE`;
2. increase keyframe rate;
3. run relocalization against loop database;
4. if relocalization fails, start a new submap;
5. when a loop/submap alignment is found, merge maps using Sim(3)/SE(3) depending on scale confidence.

If depth confidence drops:

- do not integrate low-confidence pixels;
- request more baseline if interactive UI exists;
- lower mesh confidence.

If dynamic objects dominate the frame:

- track camera against static background only;
- if static background insufficient, emit low-confidence pose and avoid map updates.

