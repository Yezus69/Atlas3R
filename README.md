# Atlas3R

Atlas3R is a scale-aware monocular reconstruction teacher for RGB videos. Its job is not to make every video look reconstructed. Its job is to turn a small number of real videos into geometry that is honest enough to train from.

Core target:

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
2. `README.md` states the current implementation state and the next milestone to execute.
3. `AGENTS.md` defines how Codex/Claude should work without bloating the architecture or roadmap.

If these files conflict, resolve the conflict this way:

```text
Architecture contracts > README milestone wording > AGENTS operating preferences
```

The README must stay small. Replace stale state; do not append history.

## Current State

- The hardened teacher runs over both canonical tracks: `python -m atlas3r.teacher` chains M1 availability, M2 measured reference, M3 geometry adapter, **global geometry refinement**, **static/dynamic/movable/unknown evidence**, M4 visibility/residuals, M5 scale posterior, M7 ray-fused per-voxel map/occupancy, **geometry export**, **visual proof**, and M8 validation. The **primary robot output is a `VoxelOccupancyGrid3D`** — a per-voxel multichannel occupancy field (`P_free / P_occupied_static / P_movable_static / P_dynamic / P_unknown` + `map_confidence`) bounded to the robot's vertical collision envelope (config-driven via `configs/robot_envelope.json`: collision_height 0.20 m + 0.05 m margin → ~6 band slices at 0.05 m). The `OccupancyGrid2D` is now the **pure top-down projection** of that field (single source of truth = the 3D field), not a parallel fusion. It writes per-track `runs/teacher/<asset>_teacher_report.json`, `runs/teacher/teacher_summary.json`, and per-asset artifacts under `runs/teacher/<asset>/` (`voxel_occupancy_3d.npz`, occupancy `.npz`, point cloud + mesh `.ply`, camera trajectory `.ply/.tum/.json`, per-channel/top-down/band-slice `.png`, `index.md`).
- M1 (`python -m atlas3r.m1`) registers canonical tracks, inspects RGB assets, and proposes keyframes under `runs/m1/`. M2 (`python -m atlas3r.m2`) ingests the local TUM-style RGB-D `reference_metric` directory and emits measured `FrameRayPacket` sidecars plus measured `ScaleEvidence` under `runs/m2/`. M3 (`atlas3r.geometry_adapter`) normalizes external backbone artifacts under `external/teacher_artifacts/<asset>/` into canonical `FrameRayPacket` objects and rebuilds measured packets from the M2 sidecar; missing artifacts return `missing_external_artifact` with the exact regeneration command, never fabricated data.
- The default monocular geometry backbone is **MapAnything** (`facebook/map-anything-apache`, Apache-2.0 weights, Meta+CMU) run via `tools/run_mapanything_backbone.py` in an isolated, gitignored env (`external/mapanything_env`). ONE feed-forward pass emits metric depth + camera poses (cam2world) + intrinsics + confidence — collapsing the prior DA3 geometry+metric pair — wired as a soft `learned_metric_depth_prior` (`metric_evidence=false`, `measured=False`). It was adopted on a **measured band-scorecard win vs DA3** (`reference_metric` candidate `occupied_static_iou 0.078 → 0.128 (+65%)`, `coverage 0.72 → 0.99`, `per_class 0.868 → 0.926`, camera-center Sim(3) RMSE `0.105 → 0.078 m (−26%)`, free-space precision held, over-occupancy down; honest categories unchanged). The gain is map QUALITY (precision/coverage/poses) — band-fsc held ~flat, i.e. the obstacle-base miss is NOT closed (depth-accuracy-limited). **Generalization caveat (honest):** this win is measured on the *gentle* `reference_metric` = TUM freiburg1_xyz; a 2nd-scene check on the *harder* freiburg1_desk shows it **does NOT generalize** — both backbones produce ~3× worse camera poses (Sim(3) RMSE 0.08 → 0.25–0.33 m) under realistic motion and band agreement collapses (MapAnything occ_iou 0.13 → 0.0). The single-scene gate over-states real-world performance; pose accuracy under real motion is the next bottleneck. See `docs/band_obstacle_recall_evidence.md` Phase 4. **Depth Anything 3** (`tools/run_da3_backbone.py`, `external/da3_env`) remains a supported fallback. No weights/repos are vendored. Rationale + SOTA survey: `docs/sota_backbone_research.md`.
- **Refinement** (`atlas3r.refine.refine_scene`) jointly optimizes per-frame SE(3) pose deltas + log-depth affine corrections over a keyframe graph (cross-frame depth consistency, robust `soft_l1`); refined packets are adopted only when the robust cost drops by a meaningful margin, else the initialization is kept unchanged. Measured depth/pose is **never** fed into the monocular candidate's refinement.
- **Static/dynamic** (`atlas3r.static_dynamic.infer_static_dynamic`) decides `{static, dynamic, movable_static, unknown}` from cross-view geometry (depth inconsistency, free-space contradiction, multi-view support); optional SAM2 masks only group verdicts. The verdict is now **fused per-voxel** into the 3D field: each surface sample is tagged occupied_static / movable_static / dynamic; dynamic samples paint only the `P_dynamic` channel (never static, never free-carving), so the dynamic/movable channels are real (e.g. reference_metric candidate band: 2180 occupied, 28 movable, 229 dynamic voxels) — closing the prior gap where they were a fusion exclusion mask only. `movable_static` and `unknown` stay distinct (unknown is never free).
- Honest categories: `reference_metric` resolves to `measured_metric` **only via the measured baseline** (measured TUM depth+pose, accepted; built through the IDENTICAL fusion, giving a measured 3D field used as ground truth). Its monocular candidate (MapAnything backbone) is refined and validated **separately** with its own category, a camera-center error vs the measured baseline, **and a per-voxel band agreement** of its 3D field vs the measured 3D field inside the collision band (`band3d_agreement`: Sim(3)+floor-plane alignment, then per-voxel class agreement / occupied IoU / free-space-contradiction / dynamic-leakage / coverage). The `VoxelOccupancyGrid3D` is **floor-aligned per reconstruction**: the fuser derives an up-alignment rotation mapping the estimated floor normal onto the band axis and rotates the whole reconstruction by it, so the collision band is a true floor-parallel slab (candidate and measured each aligned to their OWN floor; measured never aligns the candidate). On the current TUM sequence this lifts the candidate's band agreement vs measured GT materially — residual floor tilt `35.3° → ~0°`, `occupied_static_iou 0.0096 → 0.074`, `per_class_agreement 0.04 → 0.87`, `coverage_of_measured_band 0.17 → 0.72`. The now-meaningful comparison also *honestly reveals* that the monocular candidate under-detects low obstacles: of 243 co-observed measured-solid band voxels (was 6 — the old comparison was degenerate), it calls 179 free, so `band3d_agreement.free_space_contradiction_rate` reads `0.74` (the old `0.0` was over 6 voxels, not safety). The candidate's OWN map free-space contradiction did **not** regress (`0.225 → 0.215`); closing the monocular obstacle gap is a depth-quality task, not a frame task. It stays `metric_pseudo_label` (the soft scale prior gates that, not the agreement, which is reportage only). When the floor RANSAC is too weak to trust the up vector (`phone_room`, inlier `0.22 < 0.30`), the band falls back to axis-aligned with a loud `floor_normal_unreliable_band_not_floor_aligned` blocker — never a fabricated alignment. `phone_room` resolves to `metric_pseudo_label` (MapAnything's learned soft metric prior); its `band3d_agreement` is honestly `missing_measured_3d_reference` (no measured GT — never fabricated).
- The monocular candidate applies a config-driven **occupancy-estimation policy** (`configs/robot_envelope.json` / `RobotEnvelopeConfig`) when fusing its collision-band field — applied ONLY to the candidate, never to the measured GT yardstick. Tuned for the MapAnything backbone's dense, high-coverage geometry, the policy combines **directional free-carve truncation** (`free_carve_margin_m=0.10`: retract the grazing-ray over-carve flood directly below a confident surface to UNKNOWN — honestly admits uncertainty, never claims occupied) with an honest **gravity/support prior** (`occupancy_support_height_m=0.15`, `occupancy_support_min_count=3`: fill only UNKNOWN band voxels below a CONFIDENT obstacle, never overriding an observed-free voxel). On the SPARSE DA3 backbone the truncation regressed per-class so it was OFF; on MapAnything it holds the guardrails. A `occupancy_support_overrides_free` lever (fill *observed-free* base voxels) is left **OFF on purpose**: it lowers band-fsc but FABRICATES occupancy over ray-traversal evidence (violating "free comes from ray traversal only"), so the repo rejects it — honest labels over metric scores. In-plane closing is another OFF-by-default lever. Evidence + sweeps: `docs/sota_backbone_research.md` and `docs/band_obstacle_recall_evidence.md`.
- Every result exposes a provenance label (`measured_reference | monocular_DA3 | learned_metric_prior | unavailable`). Third-party models, weights, datasets, and generated artifacts remain external and gitignored; the repo adapts their artifacts, never vendors them.
- The **evaluation harness** (`python -m atlas3r.evaluate`; `--no-run` consumes the latest run) is a read-and-aggregate pass over the teacher's existing honest outputs — it invents no metric, runs no model of its own, fabricates nothing. It emits ONE versioned scorecard under `runs/eval/` (`scorecard.json` + human `scorecard_summary.md` + `scorecard_diff.json`), stamped with git commit + commit timestamp, the `RobotEnvelopeConfig`, DA3/asset provenance, a SHA-256 of each source report, and a content SHA-256. For each track it pulls the scale posterior, camera-center Sim(3) error, `band3d_agreement`, map free/occupied/unknown fractions + free-space-contradiction rate, floor-estimate health, and final `acceptance_category` + reasons. Missing/non-computed states are surfaced verbatim, never as a number (`phone_room`: band3d `missing_measured_3d_reference`, camera-center comparison `not_computed` over a bare-null teacher field, `measured_baseline` `not_applicable`). The canonical `scorecard.json` is deterministic — two runs over the same teacher reports are byte-identical (wall-clock and working-tree dirtiness live only in the non-canonical `.md` header) — and each run diffs every metric against the previous scorecard.

## Canonical Evidence Loop

All meaningful implementation milestones should be evaluated against exactly two canonical video tracks:

### Track A: `reference_metric`

A public indoor sequence with RGB frames plus metric evidence such as camera calibration, measured depth, measured camera pose, RGB-D, LiDAR, benchmark ground truth, or laser-scan reconstruction.

Initial recommended source: a small TUM RGB-D indoor handheld sequence such as `freiburg1_room`, because it has RGB, depth, camera calibration, and ground-truth trajectory. ARKitScenes or ScanNet++ can replace or supplement it later when the project is ready for higher-fidelity phone-like indoor data.

Purpose:

```text
prove whether Atlas3R can compare its monocular teacher outputs against measured metric evidence
```

### Track B: `phone_room`

A user-captured handheld phone RGB video of a real room.

Initial status:

```text
unanchored or weakly anchored until measured evidence is supplied
```

Purpose:

```text
force the system to handle the real target input while refusing false metric claims
```

If either canonical asset is absent, implement the asset manifest, status reporting, and clear failure reason. Do not fabricate replacement data.

## Milestones

Each milestone names the architecture section it implements. The milestone is complete only when it produces a concrete artifact or report for the canonical video tracks, or a clear `missing_asset` / `missing_external_artifact` status if the required input is absent.

| Milestone | Objective | Architecture Spec | Done Means |
|---|---|---|---|
| M0 - Runtime Contract Foundation | Make architecture contracts executable in code. | API Contracts, Truth Boundary, Coordinate Rules | Importable package with strict contract/state validation. No fake reconstruction. |
| M1 - Canonical Asset Registry And Video Inspection | Register the two canonical video tracks, inspect real video metadata/frames, and propose keyframes without reconstructing. | Canonical Data Strategy, Module 1, Module 2 | Asset manifest, inspection reports, keyframe candidate reports, explicit missing/corrupt statuses. |
| M2 - Metric Reference Adapter | Load measured evidence for `reference_metric` and convert it into Atlas3R scale/depth/pose evidence. | Module 3, ScaleEvidence, Metric Categories | Reference track can expose measured calibration/depth/pose/scale evidence; phone track remains unanchored unless evidence is supplied. |
| M3 - ViPE/DA3 Artifact Adapter | Normalize external ViPE/DA3 outputs into ray/depth/pose packets. | Module 4, Camera Projection Contract, Depth Convention Conversion | External artifacts become canonical `FrameRayPacket` data with projection, radial depth, confidence, and provenance. Missing artifacts fail clearly. |
| M4 - Visibility And Residual Reports | Determine which keyframes observe the same static surfaces and compute residual evidence. | Module 6, Global Optimization Evidence | Reports contain overlap edges, depth residuals, reprojection summaries, free-space conflicts, and confidence weights. |
| M5 - Scale Posterior And Metric Gate | Estimate scale evidence quality and classify outputs without mapping yet. | Module 7, Validation And Acceptance | Reference track can be evaluated against measured evidence; phone track is metric pseudo-label only if evidence supports it, otherwise non-metric/rejected. |
| M6 - Static/Dynamic Mask Evidence | Ingest SAM2/Grounded-SAM2 mask artifacts and use geometry residuals to classify static/dynamic/unknown. | Module 5, Module 8 | Dynamic pixels are identified before fusion; mask grouping alone never declares metric truth. |
| M7 - Ray-Fused Static Map | Fuse accepted static rays into TSDF, log-odds occupancy, and a floor-aligned robot grid. | Module 9, Module 10 | Free, occupied_static, movable_static, dynamic, and unknown remain separate. Unknown is never exported as free. |
| M8 - End-To-End Teacher Acceptance | Produce accepted/rejected teacher outputs with validation reports and dataset-ready provenance. | Module 11, Final Acceptance Rules | Outputs are measured_metric, metric_pseudo_label, non_metric_pseudo_label, or rejected with concrete reasons. |

## Current Priority

The teacher now emits a per-voxel **`VoxelOccupancyGrid3D`** bounded to the robot collision band as the primary output, with the dynamic/movable channels genuinely painted, the 2D grid as a pure projection of it, the band a **true floor-parallel slab** (per-reconstruction gravity/up-alignment), a measured-GT band agreement validated with real numbers on `reference_metric`, and the **MapAnything geometry backbone adopted** on a measured scorecard win. The **measured baseline is captured in a deterministic scorecard** (`python -m atlas3r.evaluate`), so every change is gated as provably better or worse against `runs/eval/scorecard.json` (current highlights, MapAnything candidate: `reference_metric` `occupied_static_iou` ≈ 0.128, `coverage_of_measured_band` ≈ 0.99, `per_class_agreement` ≈ 0.93, camera-center Sim(3) RMSE ≈ 0.078 m, band `free_space_contradiction_rate` ≈ 0.73; `phone_room` measured comparisons honestly absent and its band left axis-aligned with a loud blocker). Next priorities: (1) **make the pipeline GENERALIZE to realistic camera motion** — the headline win is on the *gentle* xyz scene; on the harder freiburg1_desk both backbones' poses degrade ~3× (Sim(3) RMSE 0.08→0.25–0.33 m) and band agreement collapses (`docs/band_obstacle_recall_evidence.md` Phase 4). This is now the #1 blocker for the actual phone-video product: the acceptance gate must include harder scenes, and **pose accuracy under real motion** (better keyframe selection/overlap, pose refinement, or a more robust backbone) is the lever — not more band-fusion tuning; (2) **close the residual obstacle-base miss HONESTLY** — even with MapAnything the candidate still calls ~73% of co-observed collision-band solid voxels free (band-fsc flat while occ_iou jumped, because coverage rose to 0.99 — it now *sees* the band but still under-places the lowest obstacle bases). A fusion lever that fills those observed-free bases (`occupancy_support_overrides_free`) *would* lower band-fsc but FABRICATES occupancy over ray evidence, so it was tried and rejected (left OFF). The honest path is a more accurate backbone: run the non-commercial **AMB3R** oracle (7-Scenes 1.74 cm, under the 12 cm gap) to bound what is achievable + decide if a commercial-clean backbone suffices; (2) **verify MapAnything on a second public indoor scene** + `phone_room` qualitatively, and consider the CC-BY-NC `map-anything` (13-dataset) checkpoint for the *teacher/eval* role; (3) **stabilize `phone_room`'s floor estimate** so its band can be floor-aligned instead of falling back; (4) ingest real SAM2/Grounded-SAM2 masks to group the per-voxel verdict into object instances.
