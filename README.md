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
3. `AGENTS.md` defines how Codex should work without bloating the architecture or roadmap.

If these files conflict, resolve the conflict this way:

```text
Architecture contracts > README milestone wording > AGENTS operating preferences
```

The README must stay small. Replace stale state; do not append history.

## Current State

- `atlas3r` has an M0 runtime contract foundation from the previous Codex pass.
- The previous milestone wording was too disconnected from the architecture and should be replaced by this data-grounded roadmap.
- Reconstruction, real model execution, optimization, mapping, validation, dataset export, and robot training are not production behavior yet.
- Third-party models and weights remain external. The repo should adapt their artifacts, not vendor them.
- The next valuable work must be grounded in two canonical videos, not toy scenes or synthetic fixtures.

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

The next priority is **M1 - Canonical Asset Registry And Video Inspection**. Build the smallest real-data vertical slice that registers the two canonical tracks, inspects actual video assets when present, proposes keyframes from decoded frames, and reports why a track is usable, missing, corrupt, or insufficient. Do not implement reconstruction or metric acceptance in M1.
