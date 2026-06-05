# Core SMGT Tiny A2 Generalization Report

Generated on branch `codex/core-smgt-tiny-generalization-gauntlet`.
Evidence was produced from pre-commit HEAD
`84f00db6cb68a86d9eb9fa4b6e29ff719a4a0a51`; the final commit is recorded in
the turn summary because a commit cannot contain its own final SHA.

## Scope

Core Phase A2 hardened and falsified the current `SMGTTiny` student before any
object or dynamic fusion work. No SAM/object/dynamic fusion was added.

Truth boundary: all student mapping here is diagnostic, RGB-only at inference,
observed-only, teacher-free during student inference, and uses measured
depth/pose only for eval sidecars. It is not final SMGT, realtime evidence,
metric accuracy, object-aware fusion, hidden-geometry completion, or RGB-only
production readiness.

## Data And Split

- Teacher cache:
  `runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache`
- Source recording:
  `runs/phase6a_recording_freiburg1_xyz_val/recording`
- Heldout RGB sidecar:
  `runs/core_smgt_tiny_a2_heldout_recording_freiburg1_xyz_val/recording`
- Split manifest:
  `runs/core_smgt_tiny_a2_weighted_split_freiburg1_xyz_val/smgt_tiny_split_manifest.json`

The split is deterministic temporal frame ranges with no frame-ID overlap.

| Split | Clips | Frames | Frame IDs |
| --- | ---: | ---: | --- |
| train | 17 | 72 | 676-747 |
| val | 5 | 24 | 748-771 |
| heldout | 5 | 24 | 772-795 |

Boundary clips `17` and `23` were omitted to prevent leakage.

## Training

Command:

```bash
python -m atlas3r train smgt-tiny --teacher-cache runs/phase6h_rgb_teacher_stitched_freiburg1_xyz_val/teacher_temporal_cache --output runs/core_smgt_tiny_a2_weighted_split_freiburg1_xyz_val --steps 5000 --batch-size 4 --device cuda:0 --amp --val-split 0.2 --heldout-split 0.2 --save-every 500
```

The run completed 5000 steps on `cuda:0` in 4833.94 s. The training window
summary passed because train-set depth/pose metrics improved, but that is not
the heldout acceptance gate. `loss_total` crossed zero, so no percent decrease is
used as proof.

| Metric | Train final | Val final | Heldout final |
| --- | ---: | ---: | ---: |
| depth AbsRel | 0.015704 | 0.162807 | 0.188903 |
| depth RMSE m | 0.025599 | 0.192591 | not primary |
| depth vs constant baseline ratio | 0.072762 | 1.474964 | 1.883660 |
| pose center vs no-motion ratio | 0.503362 | 6.215859 | 22.526175 |
| confidence Brier | 0.003495 | 0.011239 | not primary |

Heldout teacher-cache verdict: depth meets the absolute `<=0.20` target at the
final eval, but it is still worse than the constant-depth baseline ratio. Pose
misses the hard target badly: `22.526x` the no-motion baseline.

Checkpoint selection exposed another failure mode. `checkpoint_best.pt` and
`checkpoint_heldout_best.pt` both selected step `1` by loss. The trained
`checkpoint_last.pt` is step `5000`.

## Heldout Mapping

Exact validation-selected command:

```bash
python -m atlas3r runtime map-rgb-student --input runs/core_smgt_tiny_a2_heldout_recording_freiburg1_xyz_val/recording --output runs/core_smgt_tiny_a2_student_map_heldout --checkpoint runs/core_smgt_tiny_a2_weighted_split_freiburg1_xyz_val/checkpoint_best.pt --device cuda:0 --max-frames 64 --frame-stride 1 --clip-length 8 --clip-overlap 4 --image-size 120x160 --voxel-size-m 0.05 --truncation-voxels 3.0 --pixel-stride 12 --student-confidence-threshold 0.30 --student-max-sigma-m 1.0 --student-map-valid-policy confidence_sigma --export-mesh-chunks --mesh-format ply --rgb-only
```

Result: failed cleanly with `RGB student mapping produced no observed mesh
chunks`. Partial diagnostics wrote zero sparse blocks, zero active voxels, zero
surface points, and zero mesh chunks. The eval sidecar for this step-1 checkpoint
had depth AbsRel `0.204560` versus constant-depth baseline `0.146337`, so depth
did not beat the baseline.

Trained-checkpoint diagnostic command used the same heldout RGB frames and
settings, replacing only the checkpoint and output:
`checkpoint_last.pt` to `runs/core_smgt_tiny_a2_student_map_heldout_checkpoint_last`.

| Run | Checkpoint | Frames | Chunks | Vertices | Triangles | Mapped pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation-selected | best step 1 | 24 | 0 | 0 | 0 | no mesh |
| A2 trained | last step 5000 | 24 | 45 | 18,972 | 9,486 | 0.999262 |
| all-positive diagnostic | last step 5000 | 24 | 45 | 18,972 | 9,486 | 1.000000 |
| previous Core A | best step 1000 | 24 | 45 | 24,704 | 12,352 | 1.000000 |

For `checkpoint_last.pt`, eval-only measured sidecars reported depth AbsRel/RMSE
`0.056616 / 0.192684 m` and Sim3 camera-center ATE RMSE `0.003094 m`; both beat
the eval-only constant-depth and no-motion baselines. These measurements are not
mapping inputs and do not override the teacher-cache heldout failure.

## Confidence Gate

`confidence_sigma` gating on `checkpoint_last.pt` filtered only 340 of 460,800
heldout pixels.

| Policy | raw_valid | confidence_valid | sigma_valid | dynamic_rejected | mapped |
| --- | ---: | ---: | ---: | ---: | ---: |
| confidence_sigma | 1.000000 | 0.999262 | 0.999262 | 0.000000 | 0.999262 |
| all_positive | 1.000000 | 0.999262 | 0.999262 | 0.000000 | 1.000000 |

The stricter gate exists and reports correctly, but this checkpoint is not
calibrated enough for the gate to reduce map junk meaningfully.

## Long-Run Diagnostic

Command used the full available 120-frame recording because the source is
shorter than `--max-frames 300`, with `checkpoint_last.pt` and the same
confidence/sigma gate:
`runs/core_smgt_tiny_a2_student_map_longrun`.

| Metric | Result |
| --- | ---: |
| frames | 120 |
| mesh chunks | 113 |
| vertices / triangles | 75,580 / 37,790 |
| active blocks | 141 |
| active voxels | 24,519 |
| mapped pixel ratio | 0.999353 |
| pose translation norm range m | 0.000-1.348847 |
| map update p50 / p95 / max ms | 46.478 / 49.149 / 52.815 |
| mesh update p50 / p95 / max ms | 156.803 / 233.030 / 256.653 |
| student inference p50 / p95 / max ms | 58.365 / 97.329 / 542.435 |
| total pipeline ms | 68,556.374 |
| sampled peak process RSS MiB | 1,164.06 |
| sampled peak GPU0 memory MiB | 1,818 |

Long-run failure flags: mapped pixel ratio stayed near 1.0 despite confidence
gating. Active blocks and voxels grew from 38/2,997 to 141/24,519 over 120
frames; this is not an explosive crash, but the near-all-pixel mapping means the
map is not adequately filtered.

## Verdict

Mark the current SMGT-tiny as a diagnostic/toy baseline, not a validated
foundation for object/dynamic fusion. The implementation now has useful
confidence/sigma/dynamic gates, heldout split discipline, and clearer metrics,
but the model failed the teacher-cache heldout pose target, validation-selected
checkpoint mapping produced zero mesh, and the trained checkpoint maps almost
all pixels under the stricter gate.

Next work should replace or harden the student core and confidence calibration
before object/dynamic fusion.
