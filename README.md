# Atlas3R

Atlas3R is a Scale-Aware Monocular Reconstruction Teacher for RGB videos.

Core objective:

```text
large RGB video corpus
  -> reconstructability filtering
  -> scale-aware static reconstruction
  -> ray-fused TSDF + robot occupancy maps
  -> accepted metric pseudo-labels with uncertainty
```

Core pipeline:

```text
RGB video
  -> video quality and reconstructability gate
  -> keyframe selector
  -> ViPE/DA3 geometry backbone
  -> canonical ray/depth/pose representation
  -> visibility graph
  -> SAM2 mask grouping
  -> scale-aware robust optimizer
  -> static/dynamic inference loop
  -> ray-based TSDF + occupancy fusion
  -> floor-aligned robot occupancy grid
  -> held-out render + free-space validation
  -> metric acceptance / rejection
```

## Codex Context

Codex should load these files before changing the architecture or implementation:

```text
README.md        current state, roadmap, and next core milestone
ARCHITECTURE.md  architecture, math, contracts, and acceptance rules
AGENTS.md        Codex operating rules
```

This file is a compact state and roadmap file. Keep it current when the real
state of the repo changes.

## Current State

- The repository currently contains the three Codex context files.
- Runtime package, model adapters, CLI commands, tests, mapper runtime, and
  build targets have not been created yet.
- Third-party model repositories and weights are external dependencies behind
  adapters, not repo contents.

## Current Priority

Start with M0. Each implementation turn should move the earliest incomplete
milestone forward with a testable vertical slice tied to the architecture.

## Milestones

### M0 - Contract Scaffold

Create the smallest Python package scaffold for the architecture contracts.

Acceptance:

- no heavy model dependency imports at package import time;
- typed schemas exist for video input, reconstructability reports, keyframes,
  frame ray packets, scale posterior, visibility graph, mask tracks,
  static/dynamic state, optimized scene state, voxel map state, mesh metadata,
  occupancy grids, and validation reports;
- coordinate frames, units, tensor shapes, uncertainty fields, and acceptance
  statuses match `ARCHITECTURE.md`;
- focused tests verify shape validation, coordinate naming, metric-status
  gating, uncertainty presence, and unknown/free/occupied/dynamic separation.

### M1 - Video Gate And Keyframes

Implement dependency-safe video inspection, reconstructability scoring, and
keyframe selection.

Acceptance:

- videos with pure rotation, severe blur, zoom/stabilization artifacts, high
  dynamic foreground ratios, weak static structure, or poor overlap are rejected
  with explicit reasons;
- keyframes are selected for baseline, overlap, sharpness, and static content;
- all frames remain available for later validation and dense fusion;
- metric acceptance remains downstream of scale posterior and validation.

### M2 - Geometry Backbone Boundary

Add the first adapter boundary for ViPE/DA3 outputs.

Acceptance:

- ViPE integration uses an adapter with external dependency paths;
- missing dependency or missing artifact paths produce clear errors;
- adapter output normalizes to ray maps, radial depth, camera-to-world poses,
  confidence, and camera model metadata;
- MegaSaM is available as a selected fallback backend, not a parallel default.

### M3 - Mask Grouping Boundary

Add the SAM2 mask grouping interface without making semantics mandatory.

Acceptance:

- masks carry frame IDs, object IDs, confidence, and provenance;
- mask grouping is separate from static/dynamic classification;
- Grounded-SAM2 prompts are optional and used only for scale-anchor candidates.

### M4 - Visibility Graph And Optimizer Skeleton

Build the first robust graph over pose, depth correction, scale, camera/ray
correction, and static masks.

Acceptance:

- costs are robust and confidence-weighted;
- depth correction uses per-frame scale, log-depth bias, and smooth residual
  fields;
- global scale remains a variable with posterior uncertainty;
- backbone priors constrain geometry drift;
- metric acceptance is produced only by the metric gate.

### M5 - Ray-Based Mapping

Fuse optimized static rays into TSDF, occupancy, and floor-aligned grid outputs.

Acceptance:

- free space before surfaces is represented;
- unknown remains distinct from free;
- dynamic pixels are excluded before fusion;
- mesh chunks carry source frame IDs, observed coverage, voxel size, coordinate
  frame, metric scale source, and uncertainty summaries.

### M6 - Validation And Metric Gate

Add held-out render validation, free-space contradiction checks,
scale-posterior thresholds, and acceptance/rejection reports.

Acceptance:

- accepted outputs are separated into measured metric, metric pseudo-label, and
  non-metric pseudo-label categories;
- rejected videos record concrete rejection reasons;
- benchmark or calibration evaluation is the path for physical accuracy claims.

### M7 - Dataset Export

Export training caches only from accepted teacher outputs.

Acceptance:

- caches preserve confidence, uncertainty, scale status, dynamic/unknown/free
  channels, and provenance;
- third-party repositories and weights remain external;
- student training remains downstream of validated teacher labels.
