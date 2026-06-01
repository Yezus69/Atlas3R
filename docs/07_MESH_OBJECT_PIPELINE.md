# 07 — Mesh and Object Pipeline

## Goal

Turn streaming neural predictions into stable triangle/polygon assets usable in game engines while preserving metric uncertainty and object identities.

## Geometry representations

Atlas3R uses multiple representations for different jobs:

1. **Per-frame depth/pointmap:** neural output, high frequency, noisy.
2. **Surfel map:** optional fast preview and object association.
3. **TSDF voxel map:** primary measured surface representation.
4. **Triangle mesh chunks:** game-engine output.
5. **Object meshes:** mesh subsets grouped by object ID.
6. **Completion mesh:** optional predicted hidden surfaces, always flagged as predicted.

## TSDF map design

### Spatial hashing

Use voxel blocks, e.g. `8×8×8` or `16×16×16` voxels per block. Key by integer block coordinate.

```text
BlockKey = floor(world_xyz / (voxel_size * block_size))
VoxelIndex = floor(world_xyz / voxel_size) mod block_size
```

### Voxel fields

```text
tsdf: float16/float32
weight: float16/float32
color: uint8 or float16 rgb mean
normal: packed float16 xyz
object_logits: compressed top-k object IDs + weights
semantic_embedding: optional compressed vector
uncertainty: float16/float32
last_update_frame: int32
flags: observed/single_view/completed/dynamic/low_confidence
```

### Integration

For each keyframe pixel:

- skip if invalid depth, sky, dynamic, reflective low-confidence, or outside truncation band;
- compute surface point in world;
- traverse voxels along ray near surface;
- update TSDF with confidence weight;
- update color from RGB;
- update object distribution from projected mask/object head;
- mark dirty blocks for meshing.

### De-integration and pose correction

When pose graph updates old keyframes, either:

1. keep per-keyframe integration records and de-integrate/re-integrate affected frames; or
2. store submaps and transform submaps after pose correction;
3. for MVP, rebuild local TSDF chunks after significant pose correction.

Start with submaps because they are simpler and robust.

## Meshing

### Incremental marching cubes

- Extract only dirty blocks and one-block boundaries.
- Deduplicate vertices on block boundaries.
- Compute normals from TSDF gradients.
- Assign vertex color and object ID by interpolated voxel attributes.
- Emit chunk version numbers for clients.

### Dual contouring option

Use dual contouring later for sharper edges and planar indoor geometry. It requires reliable normals and QEF solving; implement after marching cubes works.

### Mesh simplification

Game engines need manageable geometry.

Create multiple LODs:

- LOD0: near-original mesh.
- LOD1: quadric decimation target 50% faces.
- LOD2: collision/navmesh simplified.

Never simplify away object boundaries unless exporting a static background collision mesh.

## Object mesh splitting

For each mesh triangle:

1. collect object ID probabilities from vertices;
2. assign triangle to object if max probability > threshold;
3. otherwise assign to background/unknown;
4. run connected components per object;
5. merge components using 3D proximity and temporal track ID.

Each object instance stores:

```text
object_id
label candidates
T_world_object
oriented_bbox
mesh_chunk_ids
observed_surface_area
estimated_full_surface_area optional
observed_coverage_ratio
is_dynamic
confidence
uncertainty
```

## Object completion

Object completion is optional and must be separated from observed geometry.

Allowed completion methods:

- retrieve similar object prior from Objaverse/ShapeNet-like dataset;
- train an object SDF/mesh completion network;
- use symmetry/planarity priors for simple furniture.

Rules:

- completed triangles have `surface_source=completed_surface`;
- completed mesh can be exported as separate node/layer;
- physics/collision should default to observed mesh unless the user opts in.

## Textures and materials

### MVP

- vertex colors from fused RGB;
- per-object material placeholder;
- export GLB with vertex colors.

### Better texture pipeline

1. select keyframes with high view angle and sharpness;
2. unwrap mesh or use existing chunk UVs;
3. project RGB into atlas;
4. blend by view angle, distance, exposure, and confidence;
5. inpaint small unobserved texture holes only as predicted.

### Physical materials

Do not infer physical material properties as measured from RGB unless explicitly trained and evaluated. Use optional neural material head for rough labels: metal, glass, wood, fabric, etc., with confidence.

## Coordinate systems for game engines

Internal convention: meters, right-handed camera with +Z forward.

Export adapters:

- glTF: right-handed, meters; convert axes if needed.
- Unity: left-handed world by default; provide conversion matrix.
- Unreal: centimeters and different axes; export scale metadata and conversion.
- USDZ/RealityKit: meters; Apple-friendly axis conversion as needed.

Every export must include a metadata JSON:

```json
{
  "units": "meters",
  "coordinate_frame": "atlas3r_world_v1",
  "voxel_size_m": 0.01,
  "surface_sources": ["observed_surface", "single_view_prior"],
  "accuracy_report": "path/or/null",
  "scale_source": "rgb_prior|calibrated_rgb|known_anchor|external_pose"
}
```

