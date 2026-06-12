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
- Canonical M3 artifacts are the v3 STABILITY COMPOSITES: COLMAP frozen-BA poses (`tools/run_colmap_pose_backend.py`) + 2-phase MVS verified depth with the perturbation-stability tier (`tools/run_mvs_depth_backend.py --stability-*`), built from raw video by `tools/run_sfm_pipeline.py` (loop-closed matching at linear cost). The learned backbone underneath is **MapAnything** (`facebook/map-anything-apache`, Apache-2.0, `tools/run_mapanything_backbone.py`); **Depth Anything 3** (`tools/run_da3_backbone.py`) is the supported fallback. Evidence: `docs/band_obstacle_recall_evidence.md` Phases 3–15.
- Refinement (`atlas3r.refine.refine_scene`) jointly optimizes per-frame SE(3) deltas + log-depth affine over a keyframe graph; refined packets are adopted only on a meaningful robust-cost drop. Measured data is never fed into the candidate's refinement.
- Static/dynamic (`atlas3r.static_dynamic`) decides `{static, dynamic, movable_static, unknown}` from cross-view geometry and fuses the verdict per voxel; dynamic never paints static and never carves free; unknown is never free.
- Honest categories: `measured_metric` is reachable ONLY via the measured baseline; the monocular candidate carries its own category, a camera-center Sim(3) error vs measured, and a per-voxel `band3d_agreement` (reportage, never gates). Contracts and rules: ARCHITECTURE.md.
- The monocular candidate applies a config-driven occupancy-estimation policy (`configs/robot_envelope.json`) — candidate-only, never the measured yardstick; the `occupancy_support_overrides_free` lever stays OFF (tried, rejected as fabrication). Spec: ARCHITECTURE.md Module 9; evidence: `docs/band_obstacle_recall_evidence.md`.
- The acceptance gate is a **GT-free verification cascade** (spec: ARCHITECTURE.md "GT-Free Acceptance Cascade"; evidence: `docs/gt_free_verification.md`): Stage 0 evidence mass, Stage 1 gravity alignment, Stage 2 consistency, plus an injected-corruption detection-limit harness (`tools/run_detection_limit.py`), an independent epipolar auditor (`atlas3r.epipolar_audit`, reportage), and a signal calibration table (`tools/run_signal_calibration.py`). Measured effect: `phone_room`'s false accept is fixed; every rejection names its real defects; Stage 2 is PROVENANCE-CONDITIONED (2026-06-11): `prerefine_p90_log_depth_residual` is PROMOTED (Stage 2b) with per-pose-provenance-class bounds, calibrated on the expanded 12-config population with the vault held out as validation — neither fsc nor p90 ranks label quality across classes (measured negative result in `docs/gt_free_verification.md`).
- A metric anchor council now compares the candidate against staged learned metric-depth artifacts before M5. Same-artifact primary anchors are self-audits only; independent anchors are soft `ScaleEvidence`, never measured evidence. Calibration on `reference_metric` and `reference_metric_desk` showed raw independent learned-anchor consensus under-covered true scale error, so uncalibrated learned anchor agreement is reportage-only with high posterior uncertainty. A residual-monotone risk certificate is implemented but promotion is disabled until enough measured calibration scenes exist; `reference_metric_room` and `phone_room` currently have no usable independent learned metric anchor. Evidence: `docs/metric_anchor_council.md`.
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

Every change is gated against the deterministic scorecard (`runs/eval/scorecard.json`). The end goal: a teacher strong enough that `phone_room` legitimately PASSES the gate — by improving reconstruction, never by weakening the gate. Full campaign evidence: `docs/band_obstacle_recall_evidence.md` Phases 1–15, `docs/gt_free_verification.md`, `docs/vault_runs.md` (sealed cross-camera Run 1: NO ALARMS).

1. **RECIPE v3 IS CANONICAL (2026-06-11)** — COLMAP frozen-BA poses + 2-phase MVS verified depth + perturbation-stability tier (τ=0.005, k=2) at the physics-derived 2.5 cm envelope; prior artifacts at `external/_pre_v3_artifacts_backup`. xyz is the accepted scene (honest F1@5cm 0.358, median solid placement 5.6 cm at claimed metric scale); the gate is provenance-conditioned (Stage-2b prerefine-p90 promoted; class bounds calibrated on the expanded population, vault-validated; every frozen verdict reproduced). Remaining blockers, measured: desk floor inlier 0.268 vs 0.30 (one floor improvement from a second accepted scene); room/phone evidence mass; phone fragment-B coverage. Loop closure at linear cost is the production matching policy (`tools/run_sfm_pipeline.py`).
2. **phone_room floor evidence** — the camera-up prior landed (search-only, bar-preserving) and proved the floor plane genuinely holds only 8.2% of static points in the registered fragment (bar 0.30). Path: register the second video fragment / floor-rich coverage, or a pre-registered floor-support statistic that doesn't demand floor dominance.
3. **phone_room confidence-weight knife edge** — Stage 0 mean confidence weight 0.295 vs 0.30 (inbounds and edge-fraction now pass under the production recipe).
4. **Promote the GT-free severity signal** — leave-one-scene-out validation of `prerefine_p90_log_depth_residual`, then a gated Stage-2b threshold.
