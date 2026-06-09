# Evidence: Closing the Monocular Collision-Band Obstacle Gap

Scope: turn the `reference_metric` monocular candidate's collision-band occupancy
from "honest but inaccurate" toward trustworthy GT. Acceptance scoreboard is
`python -m atlas3r.evaluate` / `runs/eval/scorecard.json` only.

## Baseline (commit 6f72607, robot envelope 0.20 m + 0.05 m, 0.05 m voxels)

`reference_metric` monocular candidate vs measured 3D GT, collision band:

| metric | baseline |
|---|---|
| `band3d_agreement.occupied_static_iou` | 0.0737 |
| `band3d_agreement.free_space_contradiction_rate` | 0.7366 |
| `band3d_agreement.per_class_agreement` | 0.8685 |
| `band3d_agreement.coverage_of_measured_band` | 0.7157 |
| candidate map `free_space_contradiction_rate` | 0.2149 |
| camera-center Sim(3) RMSE (m) | 0.1048 |

Determinism confirmed: re-running teacher+eval from cached DA3 depth reproduces
every band metric exactly (RANSAC seeded; fusion deterministic).

## Diagnosed dominant cause (measured, not guessed)

Confusion over the 6116 co-observed band voxels:

```
meas_free__cand_free:            5248
meas_free__cand_occupied_static:  625   (over-occupancy, concentrated HIGH in band)
meas_occupied_static__cand_free:  179   (the 0.74 contradiction)
meas_occupied_static__cand_occ:    64
```

**The contradiction is a sharp height gradient — obstacle BASES fail worst:**

| height above floor | meas-solid | cand FREE | contradiction |
|---|---|---|---|
| 0–5 cm | 36 | 36 | 1.00 |
| 5–10 cm | 40 | 39 | 0.975 |
| 10–15 cm | 41 | 36 | 0.878 |
| 15–20 cm | 39 | 29 | 0.744 |
| 20–26 cm | 38 | 19 | 0.500 |

**It is NOT sparsity / blindness.** The candidate has *more* surface voxels than
measured (3451 vs 3094). Of the 179 contradiction voxels:
- 115 / 179 (64%) have a candidate surface inside the 10 cm match box, but the box
  is flooded by free (mean **89 free vs 13 surface** voxels per box) so the
  majority vote reads FREE;
- 60% have a candidate surface within 15 cm, 98% within 25 cm; median nearest
  candidate surface = **11.9 cm**.

Over-occupancy (625) rises with height (18→185 across the band), i.e. noisy
candidate surfaces placed slightly too high.

**Conclusion — two compounding FUSION effects (not pose, not blindness):**
1. **Over-carving floods obstacle bases.** Free-carve has no truncation margin, so
   grazing rays from other frames carve the voxels just in front of / below real
   surfaces free; the base column reads free even though a candidate surface sits
   ~12 cm away.
2. **Detected surfaces don't claim their support column.** A real obstacle rests on
   the floor and occupies the whole column from its top down to the floor, but the
   raw fuser only marks the single hit voxel, leaving the base unoccupied.

Both are fixable post-fusion, reusing cached DA3 depth (no re-inference).

## Candidates tried (fast harness: scorecard-identical band numbers, ~60 configs)

Levers (all config-driven via `RobotEnvelopeConfig`, default OFF, candidate-only):
`free_carve_margin_m` (directional downward free-carve truncation below a confident
surface), `occupancy_support_height_m` + `occupancy_support_min_count` (unknown-only
gravity support below a confident obstacle), `occupancy_close_voxels` (in-plane
closing). Representative rows vs baseline (occ_iou 0.0737, band_fsc 0.7366,
per_class 0.8685, free_space_precision 0.967, map_fsc 0.2149):

| config | occ_iou | band_fsc | per_class | free_prec | over_occ | verdict |
|---|---|---|---|---|---|---|
| naive support s.25 (override free) | 0.053 | 0.556 | 0.682 | 0.968 | 1812 | REJECT: occupancy inflation (occ_iou DOWN, per_class collapses) |
| isotropic truncation margin .05 | 0.088 | 0.638 | 0.851 | 0.971 | 756 | per_class −0.018 (lateral free flips) |
| directional truncation margin .10 mc2 | 0.085 | 0.679 | 0.862 | 0.969 | 678 | per_class −0.006 (still trades) |
| margin .05 + support .10 mc3 | 0.082 | 0.696 | 0.864 | 0.969 | 665 | per_class −0.005 |
| **support .10 mc3 (no truncation)** | **0.0777** | **0.7202** | **0.8681** | **0.9677** | 632 | **WIN: clean Pareto** |

Key learning: any *free retraction* (truncation, isotropic or directional) lowers
band_fsc more but flips a few measured-free voxels in the 10 cm comparison box ->
per_class regresses 0.4–1.8% (a guarded metric). Naive gravity support that
*overrides* observed free inflates occupancy broadly (the rejected anti-pattern:
occ_iou DOWN, occ_precision halved). Only **unknown-only, confident-source** support
adds occupancy without ever touching a free observation.

## What won and why

**`occupancy_support_height_m = 0.10`, `occupancy_support_min_count = 3`** (truncation
and closing left OFF). It is the honest gravity prior: a voxel that is currently
UNKNOWN and sits within 10 cm below a CONFIDENT (>=3 fused hits) obstacle is filled
occupied. It never overrides an observed-free voxel and never fires on single-hit
depth noise, so it is a pure obstacle-recall gain.

Verified on the REAL scorecard (`python -m atlas3r.evaluate`), `reference_metric`
monocular candidate vs measured GT band:

| metric | baseline -> winner | guardrail |
|---|---|---|
| `occupied_static_iou` | 0.0737 -> **0.0777** (+5.4%) | primary up |
| band `free_space_contradiction_rate` | 0.7366 -> **0.7202** (-2.2%) | primary down |
| `per_class_agreement` | 0.8685 -> 0.8681 (-0.0004) | held (rounding floor) |
| free-space precision (cand-free that is meas-free) | 0.967 -> **0.9677** | up |
| occupancy precision | 0.0929 -> **0.0971** | up |
| candidate map `free_space_contradiction_rate` | 0.2149 -> 0.2149 | held exactly |
| camera-center Sim(3) RMSE | 0.1048 -> 0.1048 | held exactly (poses untouched) |
| `coverage_of_measured_band` / `dynamic_leakage` | 0.7157 / 0.0 -> 0.7157 / 0.0 | held |
| final_category (ref / phone) | measured_metric / metric_pseudo_label -> same | honest |

Both primary safety metrics improved while every guardrail held or improved.
Free-space precision and occupancy precision BOTH rise, so the occ_iou gain is real
recall, not occupancy inflation.

Determinism: the scorecard `tracks` block is byte-identical on a full teacher+eval
rerun; with the policy OFF the teacher reproduces the committed baseline exactly.

## Generalization

- **Within-scene robustness (data-grounded, no new DA3 inference):** the 14 DA3
  keyframes were split into disjoint halves and each rebuilt as an INDEPENDENT
  reconstruction. On the non-degenerate half B (7 frames, 5 common): occ_iou
  0.0309 -> **0.0406**, band_fsc 0.8704 -> **0.8293**, per_class 0.8829 -> 0.8822
  (flat) — same signature as the full set. Half A is degenerate (baseline occ_iou
  0, only 3 common frames, no >=3-hit obstacle) so the confidence gate makes the
  policy an exact NO-OP — it does not fabricate occupancy from sparse evidence.
- **Parameter robustness:** support heights 0.10–0.20 m at min_count 3–4 all move
  occ_iou up / band_fsc down with per_class flat (see sweep). Not a single-setting
  artifact.
- **phone_room (no measured GT):** the policy applies, band still builds,
  `final_category` stays `metric_pseudo_label`, blockers unchanged — no regression
  on the floor-unreliable track.
- **Cross-scene check — MISSING ASSET (not fabricated):** a truly independent second
  indoor scene is not present. To run it:
  ```
  # fetch a second TUM RGB-D sequence (e.g. freiburg1_desk)
  curl -L -o data/freiburg1_desk.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
  # register + inspect (M1), build measured packets (M2)
  python -m atlas3r.m1 && python -m atlas3r.m2
  # run the DA3 backbone for its keyframes (GPU env), then re-run teacher + evaluate
  python tools/run_da3_backbone.py --asset-id freiburg1_desk
  python -m atlas3r.evaluate
  ```
  status: `missing_asset:second_independent_indoor_scene` — the win is confirmed on
  `reference_metric` (full + independent subset) only; cross-scene confirmation is
  pending this asset.

## Adversarial cross-check (4 independent reviewers, each tried to REFUTE)

All four upheld the win on their own measurements (none could refute it):

- **inflation/precision** (upheld): reconstructed occ_iou and band_fsc exactly from the
  confusion in both teacher reports; free-space precision 0.9670→0.9677 (UP) and
  occupancy precision 0.0929→0.0971 (UP) — "mathematically incompatible with broad
  occupancy inflation". The IoU rise is the SAME 4 voxels as the FN drop; full-grid
  occupied fraction UNCHANGED. Audited the code: support can never override a free
  voxel (free_carve OFF; fill gated to UNKNOWN).
- **honesty/invariants** (upheld): measured GT field BYTE-IDENTICAL between policy-on
  and policy-off (untouched yardstick); candidate `P_free` and `P_dynamic` unchanged
  (no unknown→free, no dynamic painting); 200k-sample brute-force found 0 probability
  / pairwise-collapse violations; categories preserved.
- **determinism** (upheld): scorecard `tracks` byte-identical on rerun; policy-OFF via
  env-override reproduces the committed baseline full-dict-exactly. Flagged that the
  scorecard envelope provenance did not surface the occupancy policy — **fixed**: the
  scorecard now records `occupancy_support_height_m` / `occupancy_support_min_count` /
  `free_carve_margin_m` / `occupancy_close_voxels` / `env_override`.
- **generalization** (upheld): reproduced the subset numbers; independently verified the
  `missing_asset` fetch URL is live (HTTP 200, genuine TUM freiburg1_desk); phone_room
  not regressed. Honest caveats it raised (and which this report already states):
  cross-scene confirmation is genuinely absent (within-scene only), and the absolute
  gains are modest.

## Ruled out / failed checks

- Free-carve truncation (isotropic or directional, any margin/confidence): lowers
  band_fsc more but REGRESSES per_class (0.4–1.8%) by flipping measured-free voxels
  in the comparison box -> not landed (guarded metric). Left as an OFF config lever
  with a documented trade.
- In-plane closing: regresses per_class and occ_iou (expands into free) -> OFF.
- Naive gravity support overriding observed free: occupancy inflation, occ_iou DOWN
  -> rejected outright.
- Bigger gains (band_fsc to ~0.61) ARE reachable with truncation at a ~0.5–0.7%
  per_class cost; not landed because per_class is a hard no-regress guardrail. The
  bulk of the remaining obstacle gap is depth-inference-limited (candidate surfaces
  are ~12 cm displaced), a depth-quality task, not a fusion task.

## Phase 2 — exhaustive optimization / occupancy / stitching / depth search

After landing the gravity-support policy, the optimization, fusion, stitching, and
depth-inference layers were searched for further band-recall gains. An analysis
fan-out (4 independent analysts) ranked candidates; each was implemented
config-gated and measured on the candidate-quality harness (band numbers identical
to the scorecard). **Every lever was a negative result or a guardrail-regressing
trade** — firmly establishing that the residual gap is an intrinsic DA3 monocular
depth-accuracy limit, not a fusion / optimization / frame-count / resolution issue.

| lever | result | occ_iou | band_fsc | per_class | verdict |
|---|---|---|---|---|---|
| baseline (landed) | — | 0.0777 | 0.7202 | 0.8681 | reference |
| confidence-gated free-carve | reverted | 0.035 | 0.676 | 0.594 | NEGATIVE — over-occupancy explodes |
| spatial depth residual δ(u) G=2 | reverted | 0.009 | 0.966 | 0.828 | NEGATIVE — overfits gauge-ambiguous cost |
| DA3 dense 28 keyframes | reverted | 0.031 | 0.940 | 0.925 | NEGATIVE — shifts surfaces further |
| DA3 res 644 (14 frames) | not landed | 0.060 | 0.584 | 0.743 | TRADE — band_fsc↓ but occ_iou/per_class↓ |

1. **Confidence-aware fusion** — the 4/4-analyst top pick — is a NEGATIVE result.
   Gating free-carve by DA3 per-pixel confidence (low-conf = far, −0.88 correlated
   with depth) removes free broadly; the 10 cm majority-vote box then fills with
   nearby surfaces → over-occupancy explodes (632→1889→2456 voxels), occ_iou drops,
   per_class collapses. The free-carve is mostly *correct*; removing it destroys
   precision. (Lesson: strong analyst consensus is not correctness — measure.)

2. **Smooth spatial depth residual `δ_i(u)`** (architecture-blessed; per-frame G×G
   bilinear log-depth control grid, smoothness + bound + prior) is NEGATIVE for
   occupancy. It *lowers* the internal cross-frame cost AND camera RMSE
   (0.105→0.072) AND scale error (sim3 1.23→1.06), yet *wrecks* band occupancy
   (occ_iou →0.009, band_fsc →0.97). The cross-frame depth-consistency objective is
   gauge-ambiguous; extra DOF overfit it into a self-consistent-but-wrong shape.
   **It also exposed a real risk**: the refinement adopts on `cost_after < cost_before`,
   so the teacher would auto-adopt this occupancy-degrading solution — the internal
   cost is a poor proxy for occupancy. A correctly-supervised δ(u) needs a free-space
   or surface data term, not the consistency residual.

3. **Denser DA3 keyframes** (the flagged "next safety lever") is NEGATIVE. DA3 is
   deterministic (a 14-frame control run reproduced the baseline exactly) and fast
   (~0.5 s forward). Re-running with 28 keyframes (the 14 + midpoints, preserving the
   8 measured-common frames) made recall WORSE (occ_iou 0.078→0.031, band_fsc
   0.72→0.94): more evenly-spaced frames changed the joint geometry (sim3 1.23→1.32)
   and shifted surfaces further from the measured obstacles.

4. **DA3 resolution** is a TRADE, not a clean win. 14 frames at process_res 504/644
   (vs 420) lowers band_fsc substantially (0.72→0.66→0.58 — fewer missed obstacles)
   and camera RMSE (0.105→0.094→0.091), but regresses occ_iou (→0.060) and per_class
   (→0.743): higher-res depth is denser but still displaced. A safety-first
   deployment could choose higher resolution (fewer missed obstacles) accepting the
   IoU/per_class trade; it is not a clean Pareto win, so the default resolution is
   unchanged.

5. **Pose relaxation** (looser twist bound / weaker pose prior, no depth DOF) is
   NEGATIVE/trade. Looser twist (0.30) drops band_fsc to 0.49 but regresses
   per_class (0.87→0.73) AND camera RMSE (0.105→0.112) — the poses drift without a
   depth correction to anchor them. No clean stitching win; the camera-RMSE gain
   seen with the spatial grid came specifically from the grid absorbing depth error,
   not from pose freedom.

**Conclusion:** the residual ~72% band-obstacle miss is dominated by ~12 cm DA3
monocular depth displacement that no cheap lever (fusion, optimization, keyframe
count, resolution) cleanly fixes. Closing it materially needs a better depth source
— a stronger geometry backbone, or genuine measured-depth anchoring (which would no
longer be a pure monocular teacher). The landed gravity-support policy remains the
clean, guardrail-safe win.
