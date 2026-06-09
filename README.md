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
- The monocular geometry backbone is **Depth Anything 3** (learned multi-view depth + poses) run via `tools/run_da3_backbone.py` in an isolated, gitignored env (`external/da3_env`, CUDA torch). A metric-depth model (`da3metric-large`) anchors scale as a `learned_metric_depth_prior` (soft, `measured=False`). No weights/repos are vendored.
- **Refinement** (`atlas3r.refine.refine_scene`) jointly optimizes per-frame SE(3) pose deltas + log-depth affine corrections over a keyframe graph (cross-frame depth consistency, robust `soft_l1`); refined packets are adopted only when the robust cost drops by a meaningful margin, else the initialization is kept unchanged. Measured depth/pose is **never** fed into the monocular candidate's refinement.
- **Static/dynamic** (`atlas3r.static_dynamic.infer_static_dynamic`) decides `{static, dynamic, movable_static, unknown}` from cross-view geometry (depth inconsistency, free-space contradiction, multi-view support); optional SAM2 masks only group verdicts. The verdict is now **fused per-voxel** into the 3D field: each surface sample is tagged occupied_static / movable_static / dynamic; dynamic samples paint only the `P_dynamic` channel (never static, never free-carving), so the dynamic/movable channels are real (e.g. reference_metric candidate band: 1272 occupied, 26 movable, 206 dynamic voxels) — closing the prior gap where they were a fusion exclusion mask only. `movable_static` and `unknown` stay distinct (unknown is never free).
- Honest categories: `reference_metric` resolves to `measured_metric` **only via the measured baseline** (measured TUM depth+pose, accepted; built through the IDENTICAL fusion, giving a measured 3D field used as ground truth). Its monocular DA3 candidate is refined and validated **separately** with its own category, a camera-center error vs the measured baseline, **and a per-voxel band agreement** of its 3D field vs the measured 3D field inside the collision band (`band3d_agreement`: Sim(3)+floor-plane alignment, then per-voxel class agreement / occupied IoU / free-space-contradiction / dynamic-leakage / coverage). On the current TUM sequence the refined candidate is honestly *over-occupied* (its ~30° floor tilt pollutes the axis-aligned band → low occupied IoU ≈ 0.01) but **safe** (`free_space_contradiction_rate = 0.0`, `dynamic_leakage_rate = 0.0`: it never calls a real obstacle free) — exactly why it stays `metric_pseudo_label` and never inherits `measured_metric`. `phone_room` resolves to `metric_pseudo_label` (DA3 scaled by the learned soft prior); its `band3d_agreement` is honestly `missing_measured_3d_reference` (no measured GT — never fabricated). The agreement is reportage only; it never gates acceptance.
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

The teacher now emits a per-voxel **`VoxelOccupancyGrid3D`** bounded to the robot collision band as the primary output, with the dynamic/movable channels genuinely painted, the 2D grid as a pure projection of it, and a measured-GT band agreement validated with real numbers on `reference_metric`. The **measured baseline is now captured in a deterministic scorecard** (`python -m atlas3r.evaluate`), so every future change is gated as provably better or worse against `runs/eval/scorecard.json` (current baseline highlights: `reference_metric` camera-center Sim(3) RMSE ≈ 0.105 m, candidate band `occupied_static_iou` ≈ 0.0096 with `free_space_contradiction_rate` 0.0; `phone_room` measured comparisons honestly absent). Next priorities: (1) make the band a true floor-parallel slab via a stable gravity alignment (the candidate's ~30° floor tilt currently pollutes the axis-aligned band; a robust per-reconstruction up-alignment would both clean the candidate field and raise occupied IoU vs measured); (2) improve the monocular candidate's metric geometry (the refined DA3 candidate is over-occupied vs measured) so its `band3d_agreement` rises; (3) ingest real SAM2/Grounded-SAM2 masks (`external/teacher_artifacts/<asset>/masks/tracks.json`) to group the per-voxel verdict into object instances; (4) tighten the M4 depth-residual and M8 held-out render-error from robust proxies toward dense per-pixel comparisons where overlapping observations exist.
