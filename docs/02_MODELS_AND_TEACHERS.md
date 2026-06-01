# 02 — Open Models, Teachers, and What to Use Them For

Atlas3R should not train from scratch. Use existing open models as initialization, teachers, baselines, and evaluation comparators. Keep each dependency behind an adapter.

## Teacher/model table

| Model/project | Runtime use? | Training use | Why it matters | Risk/notes |
|---|---:|---:|---|---|
| LingBot-Map | Baseline + possible initialization | Teacher for streaming pose/depth/pointmaps | Strongest direct match to streaming RGB reconstruction. Uses anchor context, pose-reference window, trajectory memory, paged KV cache, long sequences. | Around 20 FPS at 518×378; target is 30 FPS, so distill/optimize. |
| MapAnything | Mostly teacher | Metric geometry teacher; optional geometric-input augmentation | Unified metric 3D transformer supporting images, calibration, poses, depth, reconstruction tasks. | Model weights may be non-commercial; verify license before product use. |
| VGGT | Teacher + baseline | Camera/intrinsics/depth/pointmap/track distillation | Very strong feed-forward multi-view geometry. | Heavy; use offline teacher or submap baseline. |
| VGGT-SLAM / VGGT-SLAM 2.0 | Baseline/comparator | Pseudo-labels for long videos and loop closure design | Incremental submap alignment, factor graph lessons, RGB real-time dense reconstruction. | Code/license availability can change; keep optional. |
| MASt3R-SLAM | Baseline/comparator | Pseudo-label poses and dense correspondences | Real-time monocular dense SLAM with 3D reconstruction priors; useful for geometric matching and back-end design. | Not purely NN; good reality check. |
| SLAM3R | Baseline/comparator | Pointmap/window alignment teacher | Feed-forward real-time dense monocular reconstruction with 20+ FPS claims. | May not output explicit camera in the same form; adapt carefully. |
| Depth Pro | Teacher, optional fallback | High-res metric depth, depth boundaries, focal length supervision | Sharp monocular metric depth and focal length estimation; useful for boundary-aware depth losses. | Not fast enough alone for 30 FPS at high resolution; distill to student. |
| SAM 3 / SAM 3.1 | Teacher; optional runtime in high-quality mode | 2D instance masks, video tracks, open-vocabulary segmentation | Best object segmentation/tracking teacher for object-centric meshing. | Heavy; runtime should use a distilled Atlas3R object head unless GPU budget allows. |
| DINOv3 | Encoder initialization / feature teacher | Dense visual features, object similarity, matching | Strong dense features; use small variants for runtime. | Large variants too heavy; use small/ConvNeXt variants for Apple. |
| MonST3R / Align3R | Teacher/comparator | Dynamic-scene geometry and pose pseudo-labels | Helps avoid poisoning static maps with moving objects. | Use for dynamic video training, not primary runtime. |
| Mask3D / OpenMask3D / Open3DIS | Mostly teacher/eval | 3D instance segmentation supervision | Teaches 3D object masks and open-vocabulary 3D grouping. | Usually offline and indoor-biased. |
| nvblox/Open3D/voxblox | Runtime mapping utility | Not a teacher | Reliable TSDF/mesh infrastructure. | Not neural, but this small geometry layer is necessary. |

## Recommended use by phase

### Phase 1 baseline

Use LingBot-Map first because it directly handles streaming RGB reconstruction. Feed its pose/depth/pointmap outputs into the TSDF mesher. Add Depth Pro as a depth-sharpness fallback and SAM3 for object masks.

### Phase 2 teacher cache generation

For each training clip, precompute:

- VGGT: camera/intrinsics/depth/point tracks on selected keyframe windows;
- MapAnything: metric depth/pose/pointmap with and without known calibration;
- Depth Pro: high-res depth + focal length;
- SAM3: masks/tracks;
- MASt3R-SLAM/VGGT-SLAM: long-video pose graph pseudo labels when GT is absent;
- MonST3R/Align3R: dynamic masks and dynamic pointmaps for moving-scene clips.

Store all teacher outputs in a versioned cache. Do not recompute teachers inside the main training loop.

### Phase 3 student initialization

- Initialize image encoder from DINOv3 small/base or ConvNeXt-DINOv3 small.
- Initialize geometry heads from Depth Pro-like DPT/FPN design where practical.
- Initialize temporal context design from LingBot-Map concepts, not necessarily weights unless license/architecture fits.
- Use VGGT/MapAnything outputs for pointmap/pose/intrinsic distillation.

## Adapter contract

Every adapter must return `TeacherPrediction`:

```python
@dataclass
class TeacherPrediction:
    model_name: str
    model_version: str
    frame_ids: list[int]
    K: Float32[Array, "T 3 3"] | None
    T_world_camera: Float32[Array, "T 4 4"] | None
    depth_m: Float32[Array, "T H W"] | None
    point_world: Float32[Array, "T H W 3"] | None
    normal_camera: Float32[Array, "T H W 3"] | None
    confidence: Float32[Array, "T H W"] | None
    masks: list[MaskPrediction] | None
    tracks: list[TrackPrediction] | None
    metadata: dict[str, Any]
```

Adapters must validate shapes and coordinate conventions before saving outputs.

## Licensing checklist

Before using any teacher in training data, Codex must add a license row to `docs/status/third_party_licenses.md`:

- repo URL;
- model checkpoint URL;
- code license;
- weights license;
- allowed use: research/commercial/unknown;
- attribution requirements;
- whether teacher outputs can be used for distillation.

If license is unclear, mark as `research-only` until resolved.

