# Atlas3R

Atlas3R is the offline TEACHER half of a teacher→student robotics stack:

```text
internet-scale RGB video
-> scale-aware offline reconstruction (this repo)
-> GT-free verification gate: accept / downweight / reject
-> verified 3D occupancy training data
-> distilled real-time student (collision-band occupancy on embedded SoCs)
-> the indoor robots and lawn mowers we will build
```

We have no robot fleet. Tesla's auto-labeler moat is fleet-scale data; Matic's is an image-to-voxel net shipped on a 4GB Jetson. Ours is **verified yield**: a teacher honest enough that every accepted label carries measured evidence of its own trustworthiness — the acceptance gate is the fleet substitute. The teacher's job is not to make every video look reconstructed; it is to turn real video into geometry honest enough to train from, and to reject the rest. Rejecting bad video is part of the product.

Inside the teacher, one video becomes:

```text
RGB video
-> calibrated/metric reference evaluation when available
-> monocular geometry proposal
-> scale-aware static reconstruction
-> ray-fused TSDF and robot occupancy
-> accepted metric pseudo-labels only when evidence supports them
```

## Document Roles

Use the repository docs in this order:

1. `ARCHITECTURE.md` defines the system spec, math, data contracts, module responsibilities, and acceptance rules.
2. `README.md` states the current implementation state and the next priorities.
3. `AGENTS.md` defines how Codex/Claude should work without bloating the architecture or roadmap.

If these files conflict:

```text
Architecture contracts > README milestone wording > AGENTS operating preferences
```

The README must stay small. Replace stale state; do not append history. Measured evidence lives in `docs/`, not here.

## Current State

All milestones M0–M8 are complete; their binding specs are the ARCHITECTURE.md modules. The implemented state:

- The hardened teacher (`python -m atlas3r.teacher`) chains M1 availability → M2 measured reference → M3 geometry adapter → refinement → static/dynamic evidence → M4 visibility → M5 scale posterior → M7 ray-fused per-voxel map → export → visual proof → M8 validation, over every canonical track. The primary robot output is a floor-aligned, collision-band-bounded `VoxelOccupancyGrid3D` (contract in ARCHITECTURE.md); the `OccupancyGrid2D` is its pure top-down projection. Per-track reports land under `runs/teacher/`.
- M1 (`python -m atlas3r.m1`) registers/inspects assets and proposes keyframes; M2 (`python -m atlas3r.m2`) builds measured packets + measured `ScaleEvidence` from the TUM reference; M3 (`atlas3r.geometry_adapter`) normalizes external backbone artifacts under `external/teacher_artifacts/<asset>/`, returning `missing_external_artifact` with the exact regeneration command when absent.
- Default M3 backbone: **MapAnything** (`facebook/map-anything-apache`, Apache-2.0) via `tools/run_mapanything_backbone.py` in an isolated gitignored env; **Depth Anything 3** (`tools/run_da3_backbone.py`) is the supported fallback. Adopted on a measured band-scorecard win over DA3 on the gentle scene; the win does NOT generalize to realistic motion (poses degrade ~3×) — evidence: `docs/sota_backbone_research.md`, `docs/band_obstacle_recall_evidence.md` Phases 3–5.
- Refinement (`atlas3r.refine.refine_scene`) jointly optimizes per-frame SE(3) deltas + log-depth affine over a keyframe graph; refined packets are adopted only on a meaningful robust-cost drop. Measured data is never fed into the candidate's refinement.
- Static/dynamic (`atlas3r.static_dynamic`) decides `{static, dynamic, movable_static, unknown}` from cross-view geometry and fuses the verdict per voxel; dynamic never paints static and never carves free; unknown is never free.
- Honest categories: `measured_metric` is reachable ONLY via the measured baseline; the monocular candidate carries its own category, a camera-center Sim(3) error vs measured, and a per-voxel `band3d_agreement` (reportage, never gates). Contracts and rules: ARCHITECTURE.md.
- The monocular candidate applies a config-driven occupancy-estimation policy (`configs/robot_envelope.json`) — candidate-only, never the measured yardstick; the `occupancy_support_overrides_free` lever stays OFF (tried, rejected as fabrication). Spec: ARCHITECTURE.md Module 9; evidence: `docs/band_obstacle_recall_evidence.md`.
- The acceptance gate is a **GT-free verification cascade** (spec: ARCHITECTURE.md "GT-Free Acceptance Cascade"; evidence: `docs/gt_free_verification.md`): Stage 0 evidence mass, Stage 1 gravity alignment, Stage 2 consistency, plus an injected-corruption detection-limit harness (`tools/run_detection_limit.py`), an independent epipolar auditor (`atlas3r.epipolar_audit`, reportage), and a signal calibration table (`tools/run_signal_calibration.py`). Measured effect: `phone_room`'s false accept is fixed; every rejection names its real defects; `prerefine_p90_log_depth_residual` is a gate-promotion candidate (perfect rank correlation over 7 GT configs, pending leave-one-scene-out validation).
- Every result exposes a provenance label. Third-party models, weights, datasets, and generated artifacts remain external and gitignored; the repo adapts artifacts, never vendors them.
- The evaluation harness (`python -m atlas3r.evaluate`; `--no-run` consumes the latest run) aggregates the teacher's outputs into ONE deterministic, versioned scorecard under `runs/eval/` (content-hashed, diffed against the previous run). Missing/non-computed states surface verbatim, never as numbers.

## Canonical Evidence Loop

The acceptance gate is multi-scene. Canonical tracks (registered in `config/canonical_assets.json`, gated in `atlas3r.evaluate`):

```text
reference_metric       TUM freiburg1_xyz   gentle motion, measured RGB-D + GT trajectory
reference_metric_desk  TUM freiburg1_desk  realistic motion (harder)
reference_metric_room  TUM freiburg1_room  full room loop (hardest)
phone_room             user phone video    the real target input; no measured evidence
```

The three measured scenes prove whether monocular teacher output matches measured metric evidence across a difficulty spread; `phone_room` forces the system to handle the production input while refusing false metric claims. If an asset is absent, report `missing_asset` — never fabricate replacement data.

## Current Priority

Every change is gated against the deterministic scorecard (`runs/eval/scorecard.json`). The end goal: a teacher strong enough that `phone_room` legitimately PASSES the gate — by improving reconstruction, never by weakening the gate. State after the Phase 9/10 campaign (`docs/band_obstacle_recall_evidence.md`): pose, scale, and verified perception are solved at teacher level (COLMAP pose backend 4.5mm–3.6cm with `freeze_poses`; MVS verified depth 2.2×, composite recall 0.28→0.78; verified-evidence tier landed with a provably inert canonical path); the v2 recipe was single-shot evaluated and NOT adopted (mixed); phone_room under the full production recipe (SIMPLE_RADIAL self-calibration, distortion-remapped verified depth) is two named blockers from acceptance. Measured blockers, in order:

1. **Verified-surface placement accuracy** — the falsifier run (Phase 11) refuted the strong "the metric is the wall" read: under the new resolution-invariant `solid_distance_agreement` instrument, recovered thin-structure solids are real but placed ~5–10 cm off (MVS edge displacement), outside the robot's 5 cm collision margin (xyz F1@5cm: canonical 0.199 vs v2-composite 0.065). Levers (each needs pre-registration): multi-view consensus refinement of verified depth; per-pixel MVS uncertainty gating; AMB3R-class oracle to bound achievable placement. The voted metrics still understate quality ~6× at fine voxels (occ_iou 0.049 vs F1@10cm 0.310) — promotion of the invariant instruments to the law is a separate pre-registration. Throughput is solved: teacher 5m21s/4 tracks (32× vectorized fusion, equivalence-proven); MVS geometric verification ~2 s/keyframe marginal on a photometric substrate.
2. **phone_room floor evidence** — the camera-up prior landed (search-only, bar-preserving) and proved the floor plane genuinely holds only 8.2% of static points in the registered fragment (bar 0.30). Path: register the second video fragment / floor-rich coverage, or a pre-registered floor-support statistic that doesn't demand floor dominance.
3. **phone_room confidence-weight knife edge** — Stage 0 mean confidence weight 0.295 vs 0.30 (inbounds and edge-fraction now pass under the production recipe).
4. **Promote the GT-free severity signal** — leave-one-scene-out validation of `prerefine_p90_log_depth_residual`, then a gated Stage-2b threshold.
