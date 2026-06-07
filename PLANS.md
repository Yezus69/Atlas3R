# PLANS.md - Pivot Plan

This is the active source of direction after the repository cleanse. It is not
an implementation log.

## Pivot 0 - Repository Cleanse

Goal: remove stale code and stale phase history so the repo can restart around
the scale-aware reconstruction teacher.

Acceptance:

- old `src/`, `tests/`, Python build config, and phase reports are removed;
- active local branch is renamed to `pivot/scale-aware-reconstruction-teacher`;
- local non-pivot branches are pruned;
- remote branches are left untouched unless explicitly deleted in a later task;
- README, AGENTS, contracts, progress, decisions, and next-task docs describe
  the pivot only.

Status: done in this cleanup session.

## Pivot V0 - Contract Scaffold

Create a minimal package and tests for the contracts only.

Acceptance:

- no heavy model dependency imports at package import time;
- frame rays, radial depth, camera-to-world transforms, scale posterior,
  static probability, mapper metadata, and acceptance reports have typed
  schemas;
- units, coordinate frames, tensor shapes, and uncertainty fields are documented
  in `docs/08_API_CONTRACTS.md`;
- focused unit tests pass.

## Pivot V1 - Video Gate And Keyframes

Implement dependency-safe video inspection, reconstructability scoring, and
keyframe selection.

Acceptance:

- videos with pure rotation, severe blur, zoom/stabilization artifacts, or high
  dynamic foreground ratios can be rejected with explicit reasons;
- keyframes are selected for baseline, overlap, sharpness, and static content;
- no video is accepted as metric at this stage.

## Pivot V2 - Geometry Backbone Boundary

Add the first adapter boundary for ViPE/DA3 outputs.

Acceptance:

- ViPE remains external to the repo;
- missing dependency or missing artifact paths produce clear errors;
- adapter output normalizes to ray maps, radial depth, pose, confidence, and
  camera model metadata;
- MegaSaM remains a planned fallback, not a parallel default path.

## Pivot V3 - Mask Grouping Boundary

Add the SAM2 mask grouping interface without making semantics mandatory.

Acceptance:

- masks carry frame IDs, object IDs, confidence, and provenance;
- mask grouping is separate from dynamic/static classification;
- Grounded-SAM2 prompts are optional and used only for scale-anchor candidates.

## Pivot V4 - Visibility Graph And Optimizer Skeleton

Build the first robust graph over pose, depth correction, scale, camera/ray
correction, and static masks.

Acceptance:

- costs are robust and confidence-weighted;
- depth correction is constrained by backbone priors and smooth residual fields;
- global scale remains a variable with posterior uncertainty;
- no output is called metric unless the metric gate passes.

## Pivot V5 - Ray-Based Mapping

Fuse optimized static rays into TSDF, occupancy, and floor-aligned grid outputs.

Acceptance:

- free space before surfaces is represented;
- unknown remains distinct from free;
- dynamic pixels are excluded before fusion;
- mesh chunks carry source frame IDs, observed coverage, voxel size, coordinate
  frame, metric scale source, and uncertainty summaries.

## Pivot V6 - Validation And Metric Gate

Add held-out render validation, free-space contradiction checks, scale-posterior
thresholds, and acceptance/rejection reports.

Acceptance:

- accepted outputs are separated into measured metric, metric pseudo-label, and
  non-metric pseudo-label categories;
- rejected videos record concrete rejection reasons;
- benchmark or calibration evaluation is required before any physical accuracy
  claim.

## Pivot V7 - Dataset Export

Export training caches only from accepted teacher outputs.

Acceptance:

- caches preserve confidence, uncertainty, scale status, dynamic/unknown/free
  channels, and provenance;
- third-party repos and weights are not vendored;
- student training remains downstream of validated teacher labels.
