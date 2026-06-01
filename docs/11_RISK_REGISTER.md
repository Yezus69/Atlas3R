# 11 — Risk Register

## Risk 1: Millimeter accuracy expectation is unrealistic

Impact: highest.

Mitigation:

- expose uncertainty everywhere;
- create explicit precision modes;
- require calibration/scale anchors for mm claims;
- benchmark before claiming.

## Risk 2: 30 FPS conflicts with heavy teachers

Impact: high.

Mitigation:

- teachers are offline only;
- deploy compact student;
- split tracker and mapper rates;
- quantize/export to TensorRT/Core ML;
- object segmentation runs below frame rate unless needed.

## Risk 3: Monocular scale drift

Impact: high.

Mitigation:

- predict intrinsics and scale uncertainty;
- anchor context and trajectory memory;
- loop closure and pose graph;
- optional known scale anchors;
- use MapAnything/DepthPro metric priors but do not over-trust them.

## Risk 4: Dynamic objects poison static map

Impact: high.

Mitigation:

- dynamic mask head;
- SAM3/MonST3R teacher labels;
- exclude dynamic pixels from static TSDF;
- per-object dynamic TSDFs.

## Risk 5: Unknown camera/lens/crop metadata

Impact: high.

Mitigation:

- camera model prediction head;
- record resize/crop transforms;
- train with camera augmentations;
- confidence lower when intrinsics uncertain.

## Risk 6: Teacher license or model availability

Impact: medium/high.

Mitigation:

- adapter isolation;
- license ledger;
- fallback teachers;
- no hard dependency on non-commercial weights for commercial use.

## Risk 7: Context compaction causes Codex to lose architecture

Impact: high for implementation quality.

Mitigation:

- keep `AGENTS.md`, `PLANS.md`, and `docs/status/*` updated;
- small tasks only;
- source-of-truth contracts and tests;
- no giant one-shot coding tasks.

## Risk 8: Mesh output looks good but is geometrically wrong

Impact: high.

Mitigation:

- mesh metrics on synthetic and real GT;
- uncertainty metadata;
- mark predicted/completed surfaces;
- no mesh simplification before geometry evaluation.

## Risk 9: Apple M deployment fails due to unsupported ops

Impact: medium.

Mitigation:

- maintain a Core ML-friendly `smgt_s` architecture;
- test export early;
- avoid exotic ops in production path;
- separate research and production models.

## Risk 10: Training on 2×4090 is underpowered for full model

Impact: medium/high.

Mitigation:

- precompute teacher cache;
- train compact student;
- progressive sequence length;
- gradient checkpointing;
- use synthetic/overfit tests first;
- avoid huge end-to-end teacher backprop.

