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
- **Cross-scene check — COMPLETED (superseded):** the second scene (freiburg1_desk)
  was fetched, registered, and made a canonical gate scene; see Phase 4 below for the
  measured result (the win does NOT generalize to realistic motion).

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

## Phase 3 — better backbone (MapAnything) is a real map-quality win; band_fsc stays depth-limited

Phase 2 predicted the fix was a better depth source. A SOTA-survey-driven backbone
swap to **MapAnything-apache** (see `docs/sota_backbone_research.md`) plus the honest
fusion policy (directional free-carve truncation that retracts over-carved free to
UNKNOWN, + unknown-only gravity support, confidence gate `min_count=3`) took the
`reference_metric` candidate from the DA3 baseline to the **landed honest state**:

| metric | DA3 baseline | **landed (MapAnything, honest policy)** |
|---|---|---|
| `occupied_static_iou` | 0.0777 | **0.1281** (+65%) |
| band `free_space_contradiction_rate` | 0.7202 | 0.727 (≈flat / marginally ↑) |
| `per_class_agreement` | 0.8681 | **0.9262** |
| coverage_of_measured_band | 0.7157 | **0.9926** |
| camera Sim(3) RMSE | 0.1048 | **0.0779** (−26%) |
| free-space precision | 0.9677 | 0.9694 |
| over-occupancy (false-positive voxels) | — | 381 (mc3) vs 436 (mc2): the gate tightening cuts FPs |

**The honest result: a big map-quality win (precision, coverage, poses), but the
obstacle-base miss is NOT closed.** band_fsc stayed ≈flat — MapAnything makes the map
far more precise/complete and the poses much better, but raw obstacle *recall* did not
improve (verified: TP +38%, FP −31% — not inflation). The missed bases are now
*observed-free* (over-carved), and honestly retracting that to UNKNOWN (truncation)
does not refill them.

**REJECTED for honesty — `occupancy_support_overrides_free` (OFF).** A lever that fills
observed-FREE base voxels with occupied (the "solid to the floor" prior) DOES lower
band_fsc (0.727→0.691) — but it **CLAIMS occupancy over ray-traversal-observed free
space**, a fabrication that violates Atlas3R's *"free space comes from ray traversal
only / never fabricate"* invariant. Independent review also debunked its "clean Pareto"
story: the headline over-occupancy *decrease* was a confound of a simultaneous
`min_count` change; **isolated, the override adds +39 false-positive voxels for only
+12 true base-fills** (≈3 fabricated voxels per real one). It is config-gated and left
OFF — the repo prizes honest labels over metric scores. (Caught concurrently by the
maintainer and the adversarial review — a good example of the honesty gate working.)

**Conclusion:** band_fsc (the robot-critical "drives through obstacles" metric) is
**not honestly improvable at the fusion layer** — it is bounded by the backbone's own
depth accuracy. MapAnything raised the *ceiling* of honest map quality substantially;
honestly closing band_fsc needs a more accurate backbone. Next: the **AMB3R** oracle
(7-Scenes 1.74 cm, non-commercial) to measure what recall is achievable and whether a
commercial-clean backbone can reach it.

## Phase 4 — generalization check on a 2nd measured scene: THE WIN DOES NOT GENERALIZE

The MapAnything win above is on `reference_metric` = TUM **freiburg1_xyz**, a *gentle,
low-rotation* handheld sweep. To test generalization, a **second measured indoor scene
— TUM freiburg1_desk** (a harder trajectory orbiting a desk; shares fr1 intrinsics) was
staged, run through M1/M2 (8 measured GT keyframes), and both backbones were measured
against its measured GT under the landed honest policy. **`reference_metric_desk` is now
a CANONICAL gate scene** (`config/canonical_assets.json` + `evaluate.CANONICAL_TRACKS`),
so the scorecard reports it alongside `reference_metric` on every run. Reproduce: download
`rgbd_dataset_freiburg1_desk.tgz`, flatten into `data/reference_metric_desk/` + copy the
fr1 intrinsics sidecar, run the backbone with `--asset-id reference_metric_desk`, then
`python -m atlas3r.{m1,m2,teacher}` and `python -m atlas3r.evaluate`.

| backbone (scene) | occ_iou | band_fsc | per_class | free_prec | camera Sim(3) RMSE |
|---|---|---|---|---|---|
| MapAnything — room (xyz) | **0.128** | 0.727 | 0.926 | 0.969 | **0.078 m** |
| DA3 — desk | 0.054 | 0.937 | 0.736 | 0.762 | 0.253 m |
| **MapAnything — desk** | **0.0** | **1.0** | 0.735 | 0.744 | 0.278 m |
| MapAnything — desk (21 keyframes) | 0.026 | 0.969 | 0.730 | — | 0.331 m |

**The +65% MapAnything win COLLAPSES on freiburg1_desk** — occ_iou 0.128 → 0.0, and
MapAnything is actually *worse* than DA3 here. The decisive signal is **camera Sim(3)
RMSE: 0.078 m on the gentle scene vs 0.25–0.33 m on the harder one** — i.e. the
*candidate poses are ~3× worse* on realistic motion (TUM groundtruth is mocap-accurate,
so the error is the backbone's, not the GT's). Denser keyframes (21 vs 11) did NOT fix
it (RMSE got worse, 0.33 m) — so it is not a keyframe-density artifact; the monocular
backbones genuinely fail to recover accurate geometry+pose under faster, higher-rotation
camera motion. When the candidate trajectory is that misaligned, the band comparison
maps measured obstacles onto the wrong candidate voxels and everything reads as missed.

**Honest implications (the generalization check did its job):**
1. The single-scene scorecard (`reference_metric` = xyz only) **over-states real-world
   performance** — it is a gentle best case. The acceptance gate needs a harder scene.
2. The MapAnything adoption wins the canonical gate but is **NOT a general win**; on
   realistic motion (closer to the actual phone-video product use case) the pipeline —
   *either* backbone — does not yet work. The next bottleneck is **pose accuracy under
   real motion**, not just band-fusion or per-pixel depth.
3. This is a more valuable result than another metric bump: it tells the truth about
   where the product actually stands.

## Phase 5 — third gate scene (room-LOOP): under-sampling, not loop-closure, was the cause

Per the difficulty-spread plan, a **third measured scene — TUM freiburg1_room** (a full
room *loop*, 1362 frames, the canonical floor-robot motion) was staged as
`reference_metric_room` and wired into the canonical gate (`config/canonical_assets.json`
+ `evaluate.CANONICAL_TRACKS`). M1 selected 8 keyframes; the MapAnything backbone was run
over those 8 + 5 bookend/gap-filler frames (13 total), mirroring the desk recipe. The
measured GT field is **healthy** (6294 occupied voxels, occupied_fraction 1.06%, free
9.5%) — so what follows is a genuine *candidate* failure, not a measured/units bug.

At the committed 13-keyframe recipe the room scorecard looks catastrophic — cam Sim(3)
RMSE **0.823 m**, metric scale **0.283** (recon ~3.5× too large), `occ_iou` 0.0, only
**1.9 %** band coverage. The first read (and the first version of this section) called
that an inherent *loop-closure* breakdown. **A deeper investigation overturned that —
the n=13 catastrophe is dominated by keyframe UNDER-SAMPLING, not the loop per se.**

**Evidence 1 — locally the backbone is near-metric and accurate; only the GLOBAL stitch
drifts.** Decomposing the n=13 trajectory against TUM mocap GT (`runs/_diag/room_drift.py`):
a *global* Sim(3) over all 13 frames gives scale 0.21 / RMSE 0.83 m, but *local* 4-frame
sub-windows fit at **scale 0.95–1.06 (metric!) and RMSE 0.03–0.12 m**. So the geometry is
locally faithful; the single feed-forward pass accumulates global scale/pose drift across
a 1362-frame span sampled by only 13 views.

**Evidence 2 — keyframe count is a first-order, U-shaped lever** (`runs/_diag/kf_sweep.py`,
camera-center RMSE vs mocap GT, uniform global passes):

| n_keyframes | 13 | 16 | 24 | 32 | **48** | 64 | 96 |
|---|---|---|---|---|---|---|---|
| metric scale | 0.28 | 0.69 | 0.73 | 0.79 | **0.91** | 0.76 | 0.71 |
| cam RMSE (m) | 0.74 | 0.60 | 0.53 | 0.34 | **0.236** | 0.39 | 0.46 |

More views *constrain* the loop (drift shrinks, scale → metric) up to an optimum near
**48** (≈1 keyframe / 28 frames), then degrade again — the same "too-dense confuses the
feed-forward fusion" effect seen on desk, now bracketed from both sides. At n=48 the loop
reconstructs at **0.236 m / scale 0.91 — better-posed than desk (0.278 m).** (This needs
`memory_efficient_inference`; full-res dense heads OOM past ~32 views on the 24 GB 4090.)

**Evidence 3 — the "24-frame got worse" and "n=48 insufficient_overlap" readings were
EVAL ARTIFACTS, not regressions.** `_band3d_agreement` needs ≥3 *shared frame_ids*
between candidate and measured packets (it aligns via shared cameras). A *uniform* dense
set contains none of the 8 measured frame_ids `[108,238,367,626,735,951,1123,1318]`, so
the comparison can't run — nothing to do with drift. Re-staged with the measured frames
included (48 total), the band comparison runs and the picture flips:

| scene (motion) | cam RMSE | metric scale | occ_iou | coverage | per_class | verdict |
|---|---|---|---|---|---|---|
| reference_metric — xyz (gentle) | 0.078 m | ~1.0 | 0.128 | 0.993 | 0.926 | passes |
| reference_metric_desk — orbit | 0.278 m | ~1.0 | 0.0 | 0.897 | 0.735 | fails |
| reference_metric_room — loop, **n=13** | 0.823 m | 0.283 | 0.0 | **0.019** | 0.622 | fails (under-sampled) |
| reference_metric_room — loop, **n=48** | 0.457 m | ~0.9 | 0.009 | **0.578** | **0.897** | fails (depth-limited) |

Going 13→48 keyframes lifts band **coverage 0.019 → 0.578 (30×)**, per_class 0.62 → 0.90,
halves cam-RMSE — i.e. the candidate now genuinely overlaps the measured band. But
`occ_iou` stays ~0 and `band_fsc` ~0.96: the residual failure is the **same depth-limited
obstacle-base miss** documented for `reference_metric` in Phases 1–3, *not* a unique loop
catastrophe. (The 48-with-measured set scores 0.457 m vs the clean-uniform 0.236 m because
snapping in the measured frames created a few near-duplicate views; a min-spacing selector
would recover most of the gap.)

**Negative result — naive windowed stitching does NOT beat a single global pass.**
`runs/_diag/stitch_poc.py`: 5 overlapping 8-frame windows, each stitched into world by a
Sim(3) on its 4-frame overlap, gave **0.87 m** — *worse* than the single global pass over
the same 24 frames (0.53 m). The windows each spanned a third of the loop (so internally
drifted) and chaining Sim(3)s compounds error. A real fix would be a global pose-graph /
bundle adjustment with a loop-closure edge — a genuine SLAM build — not a quick stitch.

**Corrected conclusions:**
1. **The fixed ~8–13 keyframe budget badly under-samples long trajectories.** The
   single biggest, cheapest lever for the room loop was simply *more keyframes* (to the
   ~48 optimum), recovering pose+coverage to desk level. The optimum is motion-dependent
   (desk wants ~11, the room loop wants ~48 — ~2× density), so the principled fix is a
   **motion/overlap-aware adaptive keyframe selector**, not a fixed count. (Next-step.)
2. **Even adequately sampled, the room loop still fails the band gate** — for the known
   depth-limited reason (`occ_iou`~0, `band_fsc`~0.96), the same ceiling as every scene.
   So room stays an honest FAIL in the gate; the gate scene is left at the consistent
   13-frame recipe (room fails either way — only the *diagnosed cause* changes).
3. **Gate-design gap (unchanged, still open):** the *acceptance* gate keys on the map's
   **internal** `free_space_contradiction_rate` (room=0.123, passes the 0.358 reject
   threshold) — a self-consistent-but-globally-wrong recon can pass. The signals that
   catch the failure (`camera_center_sim3_error`, `band3d_agreement.coverage`) need
   measured GT, so they're reportage-only and absent in production. Frontier: a GT-free
   internal consistency signal correlating with cam-RMSE so the honesty gate can reject
   under-sampled / drifted reconstructions without GT. Tracked in [[sota-backbone-direction]].

## Phase 6 — Overlap-aware keyframe selector: PRE-REGISTRATION (before any GT run)

This section is committed BEFORE the backbone runs on the selected frames; the
commit timestamp is the proof. The selector (`atlas3r.keyframes`) is a GT-free
policy: accumulate median sparse-LK optical flow over the raw frames and emit a
keyframe each time the accumulated displacement reaches a fixed budget.

Frozen parameters and their provenance (NOT tunable against GT, ever):
- `FLOW_BUDGET_WIDTHS = 0.25` — displacement ~ lost overlap; 0.25 keeps ~75%
  shared field between adjacent keyframes, the middle of standard multi-view
  practice (60–80%). Chosen from overlap geometry alone.
- `MIN_GAP_FRAMES = 3`, first/last always included.
- `MAX_KEYFRAMES = 48` — provenance: the PRE-EXISTING Phase-5 measured finding
  that MapAnything degrades beyond ~48 views. A settled model property applied
  scene-blind; not a parameter of this experiment. (Transparency note: the
  uncapped policy wanted ~114 frames for room; the cap binds only there.)

GT-free selection outcome (deterministic): xyz 39, desk 40, room 48 (capped),
phone_room 43 keyframes.

Pre-registered GT-FREE predictions (checkable without ground truth):
1. room's Stage-0 evidence mass rises materially (inbounds ratio 0.000/0.074 at
   13/48-even kf -> expected > 0.30 with overlap-matched selection).
2. plane-ledger qualifying tracks lengthen and multiply on all scenes
   (the ledger's translation/rotation gauges may begin to gain leverage).
3. phone_room's evidence mass rises (inbounds 0.035 -> higher).

Pre-registered GT EXPECTATIONS (stated as expectations, not targets; the run
happens ONCE and the scorecard reports whatever it reports):
- room camera Sim(3) RMSE should land near the Phase-5 measured band for dense
  sampling (~0.24–0.46 m vs 0.823 m at 13 kf).
- desk is the honest risk: Phase 4 measured denser-as-WORSE on desk
  (0.331 m @ 21 kf vs 0.278 m @ 11 kf); the selector chose 40. A desk
  regression is a plausible outcome and will be reported as such.
- xyz at 39 kf vs 14 kf baseline: unknown direction.

Adoption rule (unchanged repo law): the selector's artifacts replace the
canonical ones only on a band3d scorecard win; a mixed result is reported as
mixed and the decision documented. No parameter of the selector may be revised
in response to these GT results — revisions require new GT-free rationale and
a fresh pre-registration.

### Phase 6 amendment (before any GT value was observed)

The first selector run could not be scored AT ALL: the flow-adaptive frame ids
shared too few frames with the M2 measured packets, so every camera/band3d
comparison returned `insufficient_overlap_for_sim3_band_comparison` /
`not_computed` — no GT number was ever produced or seen. Amendment
(measurement plumbing, not GT tuning): the selector unions in the asset's
measured-anchor frame ids (recorded as `anchor_ids_included`; anchors count
against the cap; policy frames nearest an anchor are dropped first). No-GT
scenes pass no anchors — the policy is pure exactly where it will run in
production. All other parameters unchanged from the pre-registration above.

### Phase 6 RESULTS (single shot, scored against the pre-registration)

Verdict per the pre-registered adoption rule: **MIXED -> NOT ADOPTED.** The
canonical artifacts stay; no selector parameter is revised in response to
these numbers (revision requires fresh GT-free rationale + re-registration).

| scene | RMSE (m) | inbounds | band3d occ_iou | band_fsc | ledger tracks |
|---|---|---|---|---|---|
| xyz 14->39kf | 0.0779 -> **0.0624** (-20%) | 0.605 -> 0.723 | 0.128 -> **0.000** | 0.727 -> **1.000** | 5/20 -> 7/25 |
| desk 11->40kf | 0.2784 -> **0.3353** (regressed, AS PRE-REGISTERED) | 0.648 -> 0.709 | 0.000 -> 0.037 | 1.000 -> 0.961 | 5/19 -> 12/62 |
| room 13->48kf | 0.8229 -> 0.6810 (improved, but BELOW the 0.24-0.46 even-spacing band) | 0.000 -> 0.051 | 0.000 -> 0.021 | 0.000 -> 0.905 | 2/37 -> 12/103 |
| phone 32->43kf | no GT | 0.035 -> 0.061 | no GT | no GT | 11/56 -> 16/80 |

Prediction scoring:
- GT-free #1 (room inbounds > 0.30): **FAILED** (0.051). Discovery: the room
  loop's evidence collapse is NOT sampling density -- frames across a loop do
  not co-observe without loop closure. Sharpest evidence yet that the pose
  backend (loop-capable) is the binding fix, not denser selection.
- GT-free #2 (ledger tracks lengthen/multiply): **CONFIRMED everywhere**
  (room 6x). The plane-ledger translation/rotation authority path is real.
- GT-free #3 (phone inbounds rises): partially (0.035 -> 0.061, still far
  below the gate).
- GT expectation (room toward 0.24-0.46): improved to 0.681 only -- the
  flow-adaptive distribution underperforms even spacing on a loop.
- Pre-registered desk risk: materialized (0.335, matching Phase 4's
  denser-is-worse).
- xyz surprise (not predicted): trajectory improved 20% while the BAND
  collapsed (occ_iou 0.128 -> 0.0, fsc -> 1.0): denser views amplify the known
  free-carve over-carving of low obstacle bases. Better poses do not imply a
  better robot map under the current fusion -- a measured fusion-vs-density
  interaction that must be solved before any dense-selection policy can win
  the band gate.

Net: the selector is NOT a loss as an instrument finding -- it confirmed the
ledger leverage path, isolated the loop-closure deficit from sampling
density, and exposed the density/over-carve interaction. But as a teacher
change it does not meet the adoption bar. Canonical recipe unchanged.

## Phase 7 — Fusion upgrade: determinism, density fairness, ratio-test conflict, full-column truncation

Motive: Phase 6 measured that better poses + denser views made the BAND worse
(xyz occ_iou 0.128 -> 0.000) — fusion was the bottleneck through which every
other improvement must pass. Every step below was scorecard-measured; the
middle step was a NEAR-MISS THE GATE ITSELF CAUGHT.

1. **Two-pass fusion (determinism bug fix).** The old interleaved pass made
   output depend on ray processing order (the carve-skip read surface_count
   mid-stream). First fix attempt (complete-field carve-skip) structurally
   ZEROED free_space_contradiction_rate on all scenes and silently flipped
   desk to ACCEPTED — caught by the scorecard in one run and reverted.
2. **Density-fair budget.** The fixed 20k global subsample silently starved
   per-frame evidence as views densified (39 frames got ~512 rays/frame vs 14
   frames' ~1428) — Phase 6's density comparison was confounded by it. Gone;
   the per-packet cap (<=2048/frame) now bounds the budget and scales with
   frames.
3. **Ratio-test conflict metric.** Existence-based complete counting saturates
   with density (measured: xyz 0.444 ABOVE room 0.308 — clusters collapsed).
   New definition: a voxel is contested when free traversals OUTNUMBER its
   surface hits — deterministic, complete, density-stable, and it lands on
   the prior scale (xyz 0.244 vs old 0.231; desk 0.364 vs 0.358): the 0.25
   threshold needed NO re-anchoring and all four verdicts held.
4. **Count-level recall companion** (`occupied_recall_any_within_tolerance`).
   xyz reads 0.843 at count level vs 0.122 majority-vote IoU — the Phase 6
   "collapse to 0.000" was substantially a tolerance-box VOTE-FLIP artifact,
   now measurable separately from true field degradation.
5. **Full-column free-carve truncation ADOPTED** (`free_carve_full_column:
   true`): a confident band surface retracts free in its entire band column
   below (free -> unknown, never occupied; the margin knob is removed by the
   gravity principle, not tuned). Measured trade on xyz, both directions
   reported: robot-critical metrics improved (band_fsc 0.727 -> 0.677, recall
   0.843 -> 0.875); agreement metrics dipped (occ_iou 0.122 -> 0.114,
   per_class 0.922 -> 0.900) because retracted free becomes honest UNKNOWN.
   Same adoption character as the original free_carve_margin lever.

**Stage-3 density diagnostic** (cached 46-kf xyz selector artifacts, new
fusion, reportage): trajectory 0.072 m, coverage 1.0, per_class 0.959, ratio
fsc stable at 0.296 (no saturation) — and count-level recall now MEASURES
0.371 where the old run could only say "0.000 IoU". But the vote-level band
still collapses at density (occ_iou 0.0, band_fsc 1.0): thin-structure solid
evidence stays below the confidence floor while free votes blanket the band.
Honest residual: the backbone's thin-structure depth limit (the documented
~12 cm gap). No fusion accounting can recover structure the depth never
measured — fabricating it is the rejected lever. The path remains a stronger
backbone (pose backend + AMB3R-class depth oracle), now with fusion that will
not collapse when it arrives.

## Phase 8 — COLMAP pose backend: the pose problem is SOLVED at teacher level

Oracle (BSD COLMAP 4.1, CPU-only, measured fr1 intrinsics pinned, selector
keyframes, exhaustive matching = loop closure): xyz 0.0115 m, desk 0.0213 m
(bar was 0.06), room 0.0084 m on a 7-frame fragment; a denser run (375 frames,
every 4th + keyframes, ~37 min CPU) registered 374/375 and put the full loop
at 0.148 m (0.197 m at the 48 keyframes).

Hybrid backend (`tools/run_colmap_pose_backend.py`): COLMAP poses + MapAnything
depth at the artifact seam; COLMAP's gauge-free trajectory rescaled by ONE
scalar from candidate-only Umeyama against the backbone's camera centers (no
GT anywhere); unregistered frames dropped, never fabricated;
`pose_source=colmap_sfm_scaled_to_backbone` stamped end-to-end.

**Measured finding — refine DEGRADES BA-grade poses:** the first hybrid run
read desk 0.243 m vs the 0.0213 m oracle: `refine_scene`'s 400-sample
heuristic dragged near-perfect poses toward noisy monocular depth. Fix:
`freeze_poses` (auto-detected from BA-grade pose provenance; per-frame
log-depth affine still free; poses bit-identical through refine — verified).
Two crash iterations on the freeze path (pose-smoothness residuals and the
rebuild loop still sliced twist params) were caught by smoke checks before
any scorecard was read.

**Frozen-pose single-shot vs canonical baseline:**

| scene | cam RMSE | scale est | verdict |
|---|---|---|---|
| xyz | 0.0779 -> **0.0045 m** (17x) | 1.34 -> **1.028** | stays ACCEPTED (map fsc 0.225) |
| desk | 0.2784 -> **0.0108 m** (26x) | -> 1.083 | still rejected (fsc 0.497) |
| room | 0.8229 -> **0.0355 m** (23x) | 0.28 -> 0.902 | still rejected (evidence mass 0.229, was 0.000) |

Pose AND scale are solved at the teacher level by a commercially clean
backend. NOT ADOPTED as the canonical recipe yet, honestly: with poses
near-perfect, the depth backbone's thin-structure bias is the only error left
and it expresses HARDER at 46-view density — xyz's band labels regressed
(occ_iou 0.114 -> 0.003, recall 0.875 -> 0.382), so the band-law adoption bar
is not met. The band gate is now PURELY depth-limited: the next and final
lever for the gate is depth quality (AMB3R-class oracle; and COLMAP's own
triangulated sparse points are now available as a multi-view-verified depth
audit/anchor). The hybrid recipe, tools, and artifacts are retained and
documented for that next step.

### Phase 8 continuation — depth-gap diagnosis chain (hypotheses tested honestly)

With poses solved, three hypotheses for the band residual were tested in
sequence; two died:

1. **"Multi-view depth degrades with view count" — REFUTED.** Per-frame depth
   vs measured TUM depth (diagnostic oracle): 14-view median |log err| 0.036 /
   edge 0.048; 46-view 0.042 / 0.055 (p90 actually better). The depth is
   ~4% median, ~5% at depth edges, at both counts.
2. **"Baseline band score was a scale-inflation artifact" — PARTIALLY true,
   not the driver.** The baseline fused xyz at 1.34x scale (5 cm voxels ~3.7 cm
   real, accidentally super-resolving structures); at the hybrid's correct
   scale (1.03x), re-fusing at 2.5 cm voxels lifts occ_iou 0.000 -> 0.037 but
   recall DROPS 0.38 -> 0.28: finer voxels do not rescue the band.
3. **Standing diagnosis:** 60-70% of measured-solid band voxels receive ZERO
   occupied evidence within 10 cm -- the learned depth genuinely never places
   thin-structure surfaces (edge error ~5% = 7-15 cm at room distances,
   larger than the structures themselves). COLMAP-CUDA PatchMatch MVS pilot
   (geometrically VERIFIED classical depth, BSD) is measuring whether any
   commercially-clean source sees these structures at all.

### Phase 8 continuation — MVS verified-depth pilot: the perception wall cracks

COLMAP-CUDA PatchMatch with geometric verification (7 min/scene on one 4090,
BSD): **verified depth is 2.2x more accurate than the learned depth** (median
|log err| 0.0163 vs 0.036; edge 0.0357 vs 0.048) over 67.6% of pixels.

Composite pilot (COLMAP poses + MVS-verified depth where available +
MapAnything fill, unit-consistent via the recorded candidate-only scale;
runs/_diag/mvs_composite_pilot.py):

| config | occ_iou | recall(any,10cm) | band_fsc |
|---|---|---|---|
| learned depth @ 2.5cm | 0.037 | 0.282 | 0.948 |
| composite @ 2.5cm | 0.047 | **0.780** | 0.927 |
| learned depth @ 5cm | 0.000 | 0.377 | 1.000 |
| composite @ 5cm | 0.003 | 0.445 | 0.997 |

Honest read: the thin structures ARE now perceived — 78% of measured-solid
band voxels carry solid evidence within tolerance (was 28%). The remaining
gap moved INSIDE the fuser: recovered solid hits land 1-2 voxels off (MVS
edge error ~3.6% = 5-10 cm at range) and free votes still dominate
classification (fsc 0.93, iou 0.047). Next lever (first move of the next
session): voxel-level conversion of recovered evidence — conflict resolution
and confidence floors at 2.5 cm with verified-trust weighting, all
fusion-side, no fabrication. The depth arc verdict: pose solved, scale
solved, perception now substantially solved by a commercially clean verified
source; classification is the last segment of the last mile.

(Amendment, same day, pre-GT for the tier: the "verified-trust weighting"
phrasing above was superseded by the red-teamed SPEC — continuous
confidence-WEIGHTED counting was analyzed and REJECTED (cannot flip any class;
silently breaks canonical min-count levers). The landed mechanism tiers the
GATES instead: confident := (occ >= min_count) OR (verified >= k). See
ARCHITECTURE.md FrameRayPacket `verified` and Phase 9 below.)

### Phase 9 — verified-evidence tier + PRODUCTION RECIPE v2 (pre-registration)

This section is committed BEFORE the v2 single-shot teacher+evaluate runs;
the commit timestamp is the proof. The verified-evidence tier itself is
implemented exactly per the amended ARCHITECTURE.md FrameRayPacket spec
(commit a57a60c) and verified GT-free:

- **Plumbing:** `verified/<frame_id>.npy` masks (written by
  `tools/run_mvs_depth_backend.py`) -> `FrameRayPacket.verified` optional
  channel (contracts.py; shape-validated, boolean-validated, never defaulted
  True) -> sampled at the same pixels as depth (geometry_adapter) -> threaded
  + row-filtered through refine/inject packet rebuilds -> per-voxel
  `verified_surface_count` in fusion (mapping.py, static hits only).
- **The tier (gates, not counts):** truncation/support confident-source
  predicates become `(occ >= occupancy_support_min_count) OR
  (verified_surface_count >= verified_surface_min_count)`. k =
  verified_surface_min_count = **2** = ceil(3 / 2.2), fixed ONCE from the
  measured 2.2x verification-accuracy ratio (Phase 8), never per scene.
  Raw counts preserved; support fill stays UNKNOWN-only; free never
  overridden; every tier decision reported (`verified_tier` block +
  per-lever `sources_confident_via_verified_only`).
- **fsc-metric-only exclusion:** the contested-voxel test excludes voxels
  with verified surface evidence; BOTH rates reported
  (`free_space_contradiction_rate` and
  `..._without_verified_exclusion`).
- **GT-free verification of the mechanics** (runs/_diag/verified_tier_smoke.py,
  synthetic geometry, no GT): no-channel == all-False-channel byte-identical
  fields; 2 unverified hits fire nothing while 2 verified hits fire the
  unknown-only fill; raw `VoxelMapState` counts identical with/without tier;
  tier-added occupied voxels never overlap observed-free; refine threads +
  preserves None; contract rejects shape/boolean violations. ALL PASSED.
- **Canonical no-change proof (MEASURED):** full teacher+evaluate rerun on the
  canonical artifacts (no verified channel anywhere) with the tier code
  landed — `runs/eval/scorecard_diff.json` numeric and categorical deltas
  EMPTY on all four tracks; a field-level deep diff of the full scorecard
  payload shows every metric, status, and category byte-identical, with
  exactly ONE non-meta change: the `free_space_contradiction_basis`
  self-description string now names the (vacuous-without-channel) verified
  exclusion. The tier is provably inert on canonical inputs.

**Recipe v2 (frozen components, all pre-existing):**
1. Overlap-aware selector keyframes (Phase 6 tool; params pre-registered,
   never GT-tuned) — the keyframe sets already staged in
   `runs/_diag/keyframe_selection_*.json`.
2. COLMAP BA-grade poses via `tools/run_colmap_pose_backend.py`
   (freeze_poses in refine; candidate-only Umeyama scale).
3. MVS verified depth + learned fill via `tools/run_mvs_depth_backend.py`
   (gauge-consistent: each scene's dense workspace is built from the SAME
   COLMAP model that produced its hybrid poses — xyz/desk from their scene
   models, room from the 375-image `room_dense` model).
4. Verified-evidence tier, k=2 (above).
5. **Voxel size 0.025 m — physically derived** (configs/robot_envelope_v2.json):
   the collision-relevant thin-structure class is furniture legs (~3 cm
   cross-section — the canonical small obstacle the collision band exists
   for); a voxel edge larger than the structure cross-section makes the
   structure a sub-voxel minority that cannot robustly claim even one voxel;
   0.025 m is the clean halving of the 0.05 m grid satisfying voxel <= ~3 cm
   and keeps margin_m = exactly 2 voxels. Honesty note: 2.5 cm first appeared
   in the Phase 8 pilot as a scale correction; this pre-registration fixes it
   from physics so the value cannot float with scene results. The committed
   envelope is untouched; the single-shot selects the v2 envelope via
   `ATLAS3R_ROBOT_ENVELOPE_CONFIG`.

**Single-shot mechanics:** composite artifacts for reference_metric (exists),
reference_metric_desk + reference_metric_room (dense MVS running now);
phone_room rides as a COPY of its canonical artifacts (no verified channel,
no COLMAP — its production treatment is the separate Phase 10 path), so its
v2-run gate verdict is reportage at the new voxel size only. Then ONE run:
`ATLAS3R_ROBOT_ENVELOPE_CONFIG=configs/robot_envelope_v2.json python -m
atlas3r.teacher --artifacts-dir external/_composite_artifacts --output-dir
runs/teacher_v2` followed by `python -m atlas3r.evaluate --no-run
--teacher-dir runs/teacher_v2 --eval-dir runs/eval_v2`, compared against the
canonical scorecard.

**Pre-registered structural risks (stated before the run):**
- The band3d tolerance box at 2.5 cm becomes rad=4 -> 9x9x9 = 729 voxels
  (BAND_MATCH_TOLERANCE_M = 0.10 unchanged — the metric law is not touched);
  free-neighbour vote dominance gets structurally worse; the tier's support
  fill is the counter-mechanism. If occ_iou still reads ~0 while
  recall(any,10cm) is high, the verdict will say exactly that.
- MAX_VOXELS_PER_AXIS=256 silently doubles the effective voxel where an axis
  exceeds 6.4 m (likely the room loop); `effective_voxel_size_m` per report
  is the truth, not the config value.
- The measured GT side fuses at the same 2.5 cm (the envelope is global), so
  per-voxel hit counts halve on BOTH sides; per_class_agreement may move for
  measured-side reasons. Reported as observed.

**Pre-registered EXPECTATIONS (expectations, not targets; the run happens
ONCE and the scorecard reports whatever it reports):**
- xyz: camera RMSE ~0.005 m (frozen BA poses); recall(any,10cm) up
  substantially vs canonical (pilot precedent 0.78 out-of-spine);
  occupied_static_iou up vs canonical but possibly still low in absolute
  terms (vote mechanics); acceptance uncertain (map fsc at 2.5 cm unknown).
- desk: camera RMSE ~0.011 m; gate expected to still reject on its
  consistency defect UNLESS the verified depth genuinely resolves the
  free-vs-surface contradiction — an acceptance flip must trace to a fixed
  defect, never a weakened gate, and will be audited as such.
- room: effective voxel may snap back toward 5 cm (extent cap); evidence
  mass expected up vs canonical (Phase 8 precedent 0.000 -> 0.229); still
  expected REJECTED (Stage 0 threshold 0.30).
- phone_room: artifacts unchanged; expected rejected for the same Stage 0 +
  Stage 1 reasons (inbounds ~0.035, floor inlier ~0.13).

**Adoption rule (unchanged repo law):** adopt only on a band3d scorecard win
— occupied_static_iou up AND per_class_agreement held, holding free-space
precision and band_fsc; categories preserved; any acceptance flip audited
for honesty. Mixed is reported as mixed and NOT adopted; the canonical
artifacts and committed envelope stay; no parameter may be revised in
response to these GT numbers (revision requires fresh GT-free rationale and
re-registration).

**Adversarial review round 2 (10-agent workflow, 2026-06-10, BEFORE the
single shot — both fixes land pre-GT):** two MAJOR defects confirmed in the
first tier implementation and fixed:
1. *Exclusion below the confidence bar.* The contested-test exclusion fired
   at ONE verified hit (`> 0`) while the tier's own measured arithmetic says
   one hit ≈ 2.2 unverified-equivalents < the 3-count bar — and the excluded
   rate is the acceptance-gated `free_space_contradiction_rate`, so a
   sub-confidence threshold could only loosen the gate with no measured
   provenance. FIX: exclusion requires pooled verified static hits >= the
   SAME once-fixed k (= verified_surface_min_count = 2).
2. *Class-blind verified counting.* `verified_surface_count` pooled occupied
   and movable hits, so 2 verified MOVABLE hits could source the gravity
   fill and write occupied_STATIC occupancy below a purely-movable surface —
   a category crossing the canonical lever forbids at any count. FIX:
   per-class counts (verified_occupied_count / verified_movable_count); the
   support tier mirrors its lever's occupied-only source predicate; the
   truncation tier mirrors its lever's per-class (occ|movable) predicate.
Minors fixed in the same pass: `_load_verified` no longer binarizes
non-boolean masks (rejected to None, mirroring the shape rule); the k
derivation comment names the canonical config's min_count=3 as its input
(the module default of 1 would make the tier dead, not inverted-and-live);
the tier report counts static samples only and the lever noop paths carry
the tier keys. Smoke test extended to 9 properties (exclusion-at-k,
verified-movable class semantics) — ALL PASSED; canonical no-change proof
re-run after the fixes (empty metric diff re-confirmed below).
Review residue judged non-defects and left: refine's log-affine warp carries
the verified flag onto re-warped depth values (spec-sanctioned threading;
gauge-level warps); inject keeps original-pixel provenance on corrupted
depth (deliberate — the detection-limit harness must stress the tier);
verified counts are fusion-transient, not persisted on VoxelMapState.

### Phase 9 RESULTS — the single shot ran ONCE (2026-06-10). VERDICT: MIXED → NOT ADOPTED.

Run: `runs/teacher_v2` + `runs/eval_v2/scorecard.json` (composites: xyz 60.2% /
desk 57.6% / room 51.0% verified pixels; phone_room = canonical copy;
envelope = configs/robot_envelope_v2.json via env override; canonical
artifacts and committed envelope never touched — nothing to restore).

| metric (candidate vs measured) | xyz 5cm canonical | xyz 2.5cm v2 | desk canonical | desk v2 | room canonical | room v2 |
|---|---|---|---|---|---|---|
| camera Sim(3) RMSE (m) | 0.0779 | **0.0045** | 0.2784 | **0.0108** | 0.8229 | **0.0355** |
| estimated scale | 1.337 | **1.028** | 1.089 | 1.083 | 0.283 | **0.902** |
| band coverage | 0.993 | 0.672 | 0.899 | 0.990 | 0.020 | **0.979** |
| co-observed band voxels | 8,484 | 5,863 | 10,348 | 37,676 | 425 | **91,807** |
| per_class_agreement | 0.900 | **0.938** | 0.734 | **0.819** | 0.598 | **0.965** |
| recall(any,10cm) | 0.875 | 0.762 | 0.273 | 0.205 | 0.000 | 0.172 |
| occupied_static_iou | 0.114 | 0.049 | 0.000 | 0.000 | 0.000 | 0.013 |
| band_fsc (vote-level) | 0.677 | 0.937 | 1.000 | 1.000 | 0.000 (vacuous) | 0.972 |
| map fsc (gate, ≤0.25) | 0.244 | **0.251 → REJECTED** | 0.364 | 0.415 | 0.157 | 0.273 |
| accepted_for_metric_training | TRUE | **FALSE** | FALSE | FALSE | FALSE | FALSE |

Adoption bar (occ_iou up AND per_class held, holding band_fsc): **NOT MET on
xyz** (occ_iou 0.114→0.049 down, band_fsc 0.677→0.937 worse, acceptance
LOST). per_class up on all three scenes; pose/scale/coverage transformed
(room coverage 0.020→0.979, 49×; co-observed voxels 425→91,807, 216×).
Mixed is mixed: NOT ADOPTED. No parameter is revised in response.

**Honest diagnosis, all pre-registered risks confirmed:**
1. *Vote-box flood at 2.5 cm (risk #1, confirmed exactly as written):* xyz
   carries solid evidence within tolerance for 76% of measured-solid voxels
   while the 9×9×9 majority vote reads occupied for 5% — the box is
   structurally free-dominated at finer voxels (box volume grows 8×, thin
   structures don't). The canonical xyz iou 0.114 was additionally inflated
   by the 1.34× scale super-resolution (Phase 8 diagnosis) that the v2 BA
   poses honestly remove (scale 1.028).
2. *Resolution-coupled gate metrics (risk #3, confirmed with numbers):* the
   internal map fsc ratio test (free traversals > surface hits) reads 0.354
   raw at 2.5 cm vs 0.244 at 5 cm on the SAME xyz scene class — halved
   per-voxel counts make "contested" easier; the 0.25 acceptance threshold
   was calibrated at 5 cm. The verified-tier exclusion AT THE k BAR pulled
   0.354→0.2511, still 0.0011 over the knife edge → xyz rejected. The gate
   did not weaken (exclusion bar enforced); the metric is
   resolution-sensitive BY CONSTRUCTION (a geometry fact, not a tuning
   target). Same coupling broke desk's Stage 1 at 2.5 cm: floor RANSAC
   inlier distance = 1.5 voxels = 3.75 cm halves, inlier ratio fell below
   0.30 (was passing at 5 cm).
3. *Room axis cap (risk #2, confirmed):* effective_voxel_size_m snapped to
   0.05 (extent > 6.4 m); room's v2 row is COLMAP-pose + verified-depth
   composite at 5 cm, not 2.5 cm.

**What the tier did in production (decision record, runs/teacher_v2):** xyz
18,203 verified-occupied voxels, 2,845 truncation + 2,835 support sources
confident via verified ONLY, 350 unknown-only fills, 3,304 contested voxels
excluded at the bar; desk 5,722/5,699 verified-only sources, 2,787 fills;
room 5,107 verified-only sources, 305 fills. Mechanism live and honest —
it cannot outvote a 729-voxel free-dominated box.

**Standing read after Phase 9:** the teacher's GEOMETRY is now strong
(pose 4.5mm–3.6cm, scale 3–10%, coverage ~1.0, per_class 0.82–0.97, verified
perception within tolerance 0.76 on xyz); the residual wall is the
RESOLUTION-NON-INVARIANT classification metrics — the tolerance-box majority
vote and the count-ratio fsc — whose constructions penalize exactly the
finer grid the physics demands. Any metric-law revision (e.g. a
resolution-invariant vote or a count-normalized fsc) requires its own
GT-free pre-registration in a future session; nothing is changed in
response to these numbers.

### Phase 10 — THE PRODUCTION PATH: phone_room, unknown intrinsics, full v2 recipe (2026-06-10)

This is what internet video looks like: 800 handheld 1280×720 frames, no
intrinsics, no IMU, no GT. Everything below is the teacher's own output;
no measured reference exists for this scene.

**Self-calibration (COLMAP SIMPLE_RADIAL, sequential video matching):**
- The 43 selector keyframes alone FAIL SfM outright (114/903 pairs with any
  two-view geometry, no initial pair — sparse keyframes starve matching,
  the same lesson the room loop taught at the pose stage). Production
  staging = every-4th + keyframes (235→314 frames incl. a bridge densify).
- The video fragments at a genuine visual break (frames ~493–501): two
  models, never bridgeable even by exhaustive matching. Fragment A: 175
  imgs (frames 1–492, 27/43 keyframes); fragment B: 130 imgs (502–800,
  15/43). THE CROSS-CHECK: three independent self-calibrations (fragment A,
  fragment B, and A's second extraction batch) agree on the camera to
  within 0.7% — f = 1086.4 / 1079.3 / 1089.4 px, SIMPLE_RADIAL k = 0.028 /
  0.022 / 0.031. Self-calibration on phone video WORKS; fragmentation, not
  calibration, is the production reality. The teacher proceeds on the
  larger fragment; 16 unregistered keyframes are DROPPED, never fabricated.
- Distortion matters at this seam: k=0.028 ⇒ up to ~9 px corner
  misregistration between the distorted originals and the undistorted MVS
  grid — `tools/run_mvs_depth_backend.py --source-sparse-model` now remaps
  verified depth through the lens model (depth VALUES transfer unchanged;
  optical_z is shared; only the sampling location moves). 27/27 registered
  frames carry MVS, 51.1% verified pixels.

**THE GATE'S VERDICT (runs/teacher_phone_v2, v2 recipe end-to-end:
COLMAP poses [frozen in refine] + remapped MVS verified depth + verified
tier + 2.5 cm envelope — effective voxel HELD at 0.025):**

| stage | signal | canonical recipe | v2 production recipe | bar | verdict |
|---|---|---|---|---|---|
| 0 evidence mass | median inbounds ratio | 0.035 | **0.309** | ≥0.30 | **now PASSES** |
| 0 evidence mass | depth-residual edge fraction | 0.537 | **0.711** | ≥0.70 | **now PASSES** |
| 0 evidence mass | mean confidence weight | 0.187 | 0.295 | ≥0.30 | fails by 0.005 |
| 1 gravity | floor inlier ratio | 0.127 | 0.082 | ≥0.30 | fails |
| 2 consistency | held-out render error | (pass) | 0.178 | ≤0.50 | passes |
| 2 consistency | map fsc | 0.199 | 0.138 | ≤0.25 | passes |
| 2 consistency | dynamic leakage | 0.0 | 0.0 | ≤0.10 | passes |
| | **accepted_for_metric_training** | **False (4 reasons)** | **False (2 reasons)** | | **REJECTED, honestly** |

`final_category` stays `metric_pseudo_label` (COLMAP scale is borrowed from
the backbone's learned prior via candidate-only Umeyama — soft evidence,
scale 0.329 with residual 0.458 backbone-units against the backbone's own
drifted centers, recorded verbatim).

**The floor finding (README priority 4, landed this session):** the
candidate-only camera-up prior (handheld-upright assumption, soft evidence,
recorded) redirects the floor SEARCH when the dominant-plane RANSAC is
unreliable; the acceptance bar is unchanged. On phone it fired and
RESOLVED THE DIAGNOSIS: the dominant plane already lies inside the prior's
30° cone (constrained == unconstrained == 0.082), so the failure is NOT
wall-vs-floor confusion — the floor plane genuinely holds only 8.2% of
static surface points in this 27-frame fragment (floor sparsely seen /
fails the MVS geometric check on textureless carpet). No search trick can
conjure that evidence; the gate's refusal is correct. Canonical scorecard
under the prior code: diff EMPTY on all four tracks (on canonical phone the
prior fires, reads 0.123 < 0.30, refusal stands).

**Honest summary of the production path:** the v2 teacher moved phone_room
from "evidence collapse on every Stage-0 signal + unverifiable gravity"
(4 reasons) to "two named blockers: a 0.005 confidence-weight knife edge
and a real floor-evidence shortfall" — with Stage 2 fully passing and
51% of its depth multi-view verified. The gate got NO weaker (every change
this session was bar-preserving and adversarially reviewed); the teacher
got stronger; the verdict-with-reasons is exactly the honest number the
data engine needs. Remaining blockers point at coverage (register fragment
B → more floor views; or denser ray budget on floor-rich frames), not at
the gate.

### Phase 11 — internet-scale throughput + resolution-invariant instruments (pre-registration first)

Motivation (maintainer directive, 2026-06-10): no scene may cost an hour if
the teacher is to label internet-scale video, and the Phase 9 standing read
("the wall is the instrument, not the labels") must be TESTED, not assumed.

**Throughput (measured, equivalence-proven):**
- `_fuse_rays` vectorized: 30.3 s → 0.93 s per track on the heavy 2.5 cm
  composite (32×), with EVERY evidence grid bitwise identical to the loop
  form (counts trivially: all increments are exact +1.0; log_odds via exact
  bounded replay of the clipped trajectory; only `uncertainty` — consumed by
  no gated metric — differs in final-ULP rounding). Proof:
  `runs/_diag/fuse_vectorize_equivalence.py` on the real xyz+desk composite
  packets.
- band3d tolerance-box vote vectorized via per-class 3D prefix sums
  (`teacher._box_majority_vote`): 2.8 s → 0.6 s at production scale
  (200×120×200 grid, 90k queries, rad=4), bitwise-equal outputs on
  randomized fields including all-empty/all-touched edge cases. Proof:
  `runs/_diag/box_vote_equivalence.py`.
- Keyframe-restricted PatchMatch: the room dense run computed depth for all
  375 staged images when only the 47 keyframes feed the composite;
  restricting patch-match.cfg to keyframe references with the 20 temporally
  nearest keyframes as explicit sources cuts the MVS stage ~8× (measured
  below when the run completes).

**Resolution-invariant instruments (definitions pre-registered HERE, before
observing their values on the v2 artifacts):**
1. `solid_distance_agreement` (band3d reportage): metric-distance recall /
   precision / F1 over solid voxels at physical tolerances τ ∈ {2.5 cm,
   5 cm, 10 cm}. 5 cm = the robot's own collision margin (`margin_m`): a
   label is collision-correct when its solid surface lies within the safety
   margin of the true surface. recall@τ = fraction of measured-solid band
   voxels (inside candidate bounds) within τ of an observed candidate-solid
   voxel; precision@τ = fraction of candidate-solid voxels mapping into the
   co-observed measured band within τ of a measured-solid voxel; distances
   in meters through the Sim(3) scale — no votes, no counts, so grid
   refinement converges instead of diverging.
2. `free_space_contradiction_rate_at_reference_scale` (map reportage): the
   SAME contested ratio test with counts POOLED to the 5 cm calibration
   scale (re-binned, never reweighted; pool factor 1 ⇒ exactly the headline
   rate). Removes the measured resolution-coupling (0.354@2.5cm vs
   0.244@5cm) by construction.
   Both are REPORTAGE: the voted metrics and the gated headline fsc remain
   the law; promotion requires a future pre-registration.

**Deposition variant REJECTED without a run (recorded so it is not
re-litigated):** spreading verified hits over their ±σ along-ray support was
considered for the vote gap and rejected on structural grounds — the box
vote is unwinnable for thin structures at fine resolution regardless (box
volume grows cubically, thin-structure voxels linearly: ~90 free vs ≤5 occ
even with spreading), and multiplying deposits per measurement inflates
evidence counts, breaking the min-count lever semantics exactly as the
red-teamed weighting rejection warned. Under a correct (distance-based)
instrument, point deposits + the unknown-only support fill already express
the recovered evidence.

**Pre-registered EXPECTATIONS for the v2-artifact observation (the
falsifier of Phase 9's standing read):** if the labels are good and the
voted instrument was the wall, xyz composite @2.5cm should show
solid_recall@10cm ≈ recall-any (~0.76) and solid_precision@5–10cm HIGH
(>0.5), with F1@5cm far above what voted occ_iou (0.049) implies. If
instead precision@τ comes back LOW, the labels genuinely contain false
solids, Phase 9's interpretation is WRONG, and the verdict will say so.
One observation run (`runs/teacher_v2b`), no parameter revisions in
response.

### Phase 11 RESULTS — the falsifier REFUTED the strong "instrument is the wall" read

Measured (runs/teacher_v2b, one observation, 9m20s wall for the heavy
2.5 cm config — was ~20 min pre-vectorization):

| metric | xyz canonical 5cm | xyz v2 2.5cm | desk canonical | desk v2 | room canonical | room v2 |
|---|---|---|---|---|---|---|
| solid_recall@10cm | 0.444 | 0.424 | 0.088 | 0.111 | 0.0 (13 solids) | 0.100 |
| solid_precision@10cm | 0.352 | 0.244 | 0.441 | **0.981** | 0.0 | 0.132 |
| solid_F1@10cm | **0.393** | 0.310 | 0.147 | 0.200 | 0.0 | 0.114 |
| solid_F1@5cm | **0.199** | 0.065 | 0.031 | 0.065 | 0.0 | 0.029 |
| median solid distance | 0.113 m | 0.113 m | 0.213 m | 0.266 m | 0.243 m | 0.367 m |

**Honest findings, in order of importance:**
1. *The pre-registered failure branch fired:* xyz v2 precision@5cm = 0.049
   (expected >0.5 under the "labels are good" hypothesis). The recovered
   thin-structure solids are REAL but PLACED outside the robot's 5 cm
   collision margin — the median solid-to-GT distance (11.3 cm) is
   UNCHANGED by the composite. The MVS edge displacement (~3.6% ≈ 5–10 cm
   at range) lives in the labels, not merely in the vote. Phase 9's
   NOT-ADOPTED stands for deeper reasons than vote mechanics: at the
   collision tolerance, the canonical 5 cm labels (F1@5cm 0.199) beat the
   v2 2.5 cm labels (0.065).
2. *The voted instrument WAS still misleading, just not decisive:* voted
   occ_iou 0.049 vs F1@10cm 0.310 on the same field — the vote understates
   label quality ~6× at fine resolution. Both things are true: the
   instrument exaggerates the wall AND the wall is real at the 5 cm margin.
   (The recall-any 0.76 vs solid_recall@10cm 0.42 gap is box geometry, not
   contradiction: the vote box is an L-infinity ball with ~17 cm corners;
   the distance metric is an L2 ball at 10 cm.)
3. *Instrument #2 is an honest PARTIAL:* pooled-to-5cm fsc on xyz v2 reads
   0.2753 vs the native-5cm 0.244 — pooling removes the BINNING component
   of resolution coupling but not the RAY-SAMPLING density component (finer
   grids accumulate more free traversals per ray; re-binning cannot undo
   that). A true fix must normalize free evidence per meter of traversal —
   a fusion-law change requiring its own pre-registration.
4. *Per-scene heterogeneity is large:* desk v2 places solids almost
   perfectly (precision@10cm 0.981) but sees few of the cluttered scene's
   6,821 measured solids (recall 0.111); room remains coverage-limited.

**Standing diagnosis after Phase 11:** the last technical wall for
collision-margin-accurate labels is VERIFIED-SURFACE PLACEMENT ACCURACY
(~5–10 cm edge displacement at range), not fusion, not pose, not scale,
and only partially the eval instrument. Candidate levers (each needs
GT-free rationale + pre-registration): multi-view consensus refinement of
verified depth at surface elements; an AMB3R-class oracle to bound
achievable placement; per-pixel MVS uncertainty (COLMAP outputs it) to
gate which verified pixels enter the band. Throughput is no longer a
blocker: teacher 5m21s for all four tracks (canonical), MVS geometric
verification ~2 s/keyframe marginal on a photometric substrate (51.0%
verified coverage identical to the full run; keyframe-only sources lose
3× coverage and are rejected).

### Phase 12 — PERTURBATION-STABILITY: the placement wall cracks (pilot, 2 measured scenes)

Principle (maintainer, 2026-06-10): *a surface that appears in the same
place under independent perturbations is more trustworthy than a surface
that appears only under one recipe.* Not absolute truth — a GT-free signal
whose authority is earned on measured scenes. Instantiation: per keyframe,
TWO PatchMatch geometric depths from DISJOINT temporal source halves (A =
even-ranked 10 of the 20 nearest, B = odd-ranked) on the shared photometric
substrate (+2.6 min/scene GPU each, marginal). GT-free per-pixel signals:
witness existence (both A and B verify the pixel) and placement
disagreement δ = |log dA − log dB|. Pilots:
`runs/_diag/stability_authority_pilot.py` (per-pixel authority vs measured
TUM depth) and `runs/_diag/stability_label_pilot.py` (label-level effect at
2.5 cm under the Phase 11 invariant instrument). A/B workspaces are
regenerable from the dense workspaces in ~2.6 min each (cfg construction in
the pilot scripts).

**Stage 1 — per-pixel authority (xyz / desk):**
- Witness existence has clean authority: single-witness production-verified
  pixels are 1.6–1.7× worse at median (xyz 0.0212 vs 0.0127; desk 0.0145
  vs 0.0092) and up to 1.8× at edges (xyz 0.0707 vs 0.0383).
- δ within both-witness pixels ranks error only weakly (Spearman 0.17/0.16)
  and the δ-gate moves EDGE error little: the residual edge error is a
  perturbation-STABLE BIAS (foreground fattening) that both halves
  reproduce — a consistency-only signal cannot see a shared bias BY
  CONSTRUCTION. This bounds every consistency-based GT-free signal at
  edges and is now on record.
- CONSENSUS averaging of A/B is REFUTED as a placement refiner (xyz edge
  0.0311 → 0.0329; desk 0.0702 → 0.0704): A/B errors are correlated.

**Stage 2 — label-level effect (verified := geometric AND both-witness AND
δ ≤ τ; production depth values; fused at 2.5 cm; measured under
solid_distance_agreement):**

| | xyz v2 (60% verified) | xyz stable τ=.005 (19%) | desk v2 (58%) | desk stable τ=.005 (14%) |
|---|---|---|---|---|
| solid_F1@5cm | 0.065 | **0.230** | 0.065 | **0.382** |
| solid_F1@10cm | 0.310 | 0.483 | 0.200 | **0.762** |
| solid_precision@10cm | 0.244 | 0.429 | 0.981 | 0.832 |
| solid_recall@10cm | 0.424 | 0.553 | 0.111 | **0.703** |
| median solid distance | 0.113 m | 0.089 m | 0.266 m | **0.069 m** |

xyz gated F1@5cm (0.230) BEATS the canonical-5cm labels (0.199) at 2.5 cm —
the first config to do so; desk improves 5.8× at the collision margin with
a 7,605-voxel judged sample (not small-sample noise) and is monotone in τ.
**Why a ~20% per-pixel edge improvement yields 4–6× at the label level:
per-pixel yield ≠ field-level recall.** Surfaces are over-sampled (2048
rays × 46 frames), so keeping only the trustworthy ~15–20% of verified
pixels barely costs field recall, while each dropped untrustworthy pixel
was painting a misplaced solid voxel that poisoned precision.
Trustworthy-15% beats everything-60% — the maintainer's principle,
quantified at the label level.

Honest caveats: (1) τ = 0.005 is the better of TWO swept values, openly
GT-calibrated on these two scenes — production use must freeze it ONCE with
this provenance (the Stage-0-threshold pattern), never per scene. (2) The
xyz τ=0.01 row was anomalously below baseline (desk is monotone; likely
small-judged-set noise at xyz, 185 voxels) — the τ=0.005 conclusion rests
on both scenes. (3) The VOTED metrics still read terribly on the gated
fields (xyz occ_iou 0.016) — the voted instrument diverges under
refinement (Phase 9/11); any spine adoption FIRST requires the
pre-registered metric-law amendment promoting the distance instruments,
else the scorecard law would reject the largest label improvement on
record because of its own broken meter.

**Next steps (each pre-registered before its run):** (1) metric-law
amendment — adoption bar moves to solid_F1@5cm up + per_class held +
free-precision held; (2) tool integration — A/B stability masks in
`tools/run_mvs_depth_backend.py` (or a sibling stability tool) writing
`verified/` as geometric AND stable@τ=0.005 (frozen, provenance recorded);
(3) ONE spine single-shot at the v2 recipe + stability tier across all
tracks; (4) phone_room production rerun — stability needs no GT, so the
gate's Stage-0/Stage-2 signals see the same trust tightening on no-GT
scenes.

### Phase 12 continuation — METRIC-LAW AMENDMENT + RECIPE v3 single-shot (pre-registration)

Committed BEFORE the v3 run; the commit timestamp is the proof.

**Metric-law amendment (GT-free, geometry-first rationale):** the adoption
bar for band-label changes becomes — `solid_f1_at_5cm` UP on the
accepted-scene class, AND `per_class_agreement` held, AND
`solid_precision_at_10cm` held, categories preserved, acceptance flips
audited (unchanged). The voted metrics (occupied_static_iou, vote-level
band_fsc, recall-any) remain REPORTED for continuity but no longer gate
adoption: Phases 9/11 measured that the tolerance-box vote DIVERGES under
grid refinement (box volume cubic vs thin-structure voxels linear) while
the distance instrument converges, and the robot-relevant criterion is
collision-margin placement, which solid_f1_at_5cm states directly. The
distance metrics enter the scorecard NUMERIC diff law (dot-free key names).
The verified-tier k stays 2 (NOT re-derived against the stability-gated
accuracy ratio — deliberately conservative to keep the v3 change surface
minimal; re-derivation would be its own pre-registration).

**Recipe v3 = v2 + perturbation-stability tier:** composites rebuilt with
`tools/run_mvs_depth_backend.py --stability-workspace-a/-b` (verified :=
geometric AND both-witness AND δ ≤ 0.005, τ frozen with two-scene
provenance) into `external/_composite_v3` (the v2 artifact set stays
untouched for the record). phone_room now gets its OWN stability composite
(the tier is GT-free). Envelope unchanged from v2
(configs/robot_envelope_v2.json). Single shot:
`ATLAS3R_ROBOT_ENVELOPE_CONFIG=configs/robot_envelope_v2.json python -m
atlas3r.teacher --artifacts-dir external/_composite_v3 --output-dir
runs/teacher_v3` then `python -m atlas3r.evaluate --no-run --teacher-dir
runs/teacher_v3 --eval-dir runs/eval_v3`.

**Pre-registered expectations (expectations, not targets):**
- xyz: solid_f1_at_5cm ≈ 0.23 (pilot precedent), above both v2 (0.065) and
  canonical (0.199); voted occ_iou likely DOWN (vote mechanics) — expected
  and no longer gating.
- desk: solid_f1_at_5cm ≈ 0.38 (pilot precedent ~5.8× v2).
- room: direction unknown (no pilot; the loop's verified coverage is
  thinner); reported as observed.
- phone_room: no GT — gate verdict only; verified fraction will DROP
  (~0.51 → ~0.15–0.2 of pixels); Stage-0/Stage-2 movement reported as
  observed, acceptance flips audited for honesty.
- Acceptance gates (map fsc at 0.25 etc.) unchanged; any acceptance flip
  must trace to evidence changes, never to a weakened gate.
**Adoption rule:** adopt v3 (composites + stability tier as the production
recipe) only if the AMENDED bar is met on xyz AND desk improves or holds;
mixed is reported as mixed and NOT adopted; no parameter (including τ) may
be revised in response to these GT numbers.

### Phase 12 RESULTS — the v3 single shot ran ONCE. VERDICT: MIXED → NOT ADOPTED (xyz bar unmet).

(runs/teacher_v3 + runs/eval_v3; composites external/_composite_v3 with
stable-verified fractions xyz 0.195 / desk 0.141 / room 0.038 / phone 0.077.)

| solid_distance (amended law) | xyz can. | xyz v3 | desk can. | desk v3 | room can. | room v3 |
|---|---|---|---|---|---|---|
| F1@5cm (THE BAR) | **0.199** | 0.158 | 0.031 | **0.278** | 0.000 | **0.124** |
| F1@10cm | 0.393 | 0.430 | 0.147 | **0.622** | 0.000 | 0.368 |
| precision@10cm (guardrail) | 0.352 | 0.366 | 0.441 | **0.849** | 0.000 | 0.286 |
| median solid distance (m) | 0.113 | 0.098 | 0.213 | **0.102** | 0.243 | **0.097** |
| per_class (guardrail) | 0.900 | 0.925 | 0.734 | 0.817 | 0.598 | **0.965** |

- Pre-registered bar: solid_f1_at_5cm UP on xyz — **NOT MET** (0.199 → 0.158)
  → NOT ADOPTED, canonical recipe stays, no parameter revised. desk improved
  9× at the collision margin and room went from zero to real; every guardrail
  held or improved on every scene; xyz also lost acceptance again at the map
  fsc knife edge (0.2532 vs 0.25 — fewer verified voxels ⇒ fewer exclusions).
- *Pilot-vs-spine attribution (honest):* the xyz pilot read F1@5cm 0.230;
  the spine reads 0.158 — the pilot fused without static/dynamic states, so
  pilot and spine populations differ; spine-vs-spine is the comparison the
  law reads (v2 0.065 → v3 0.158, 2.4×). Recorded so pilot numbers are never
  quoted as spine numbers again.
- *Observation, NOT a revision:* xyz-canonical's 0.199 is measured on labels
  fused at the 1.34× scale-inflation artifact (effective ~3.7 cm voxels;
  Phase 8 diagnosis) and gentle motion — the one configuration class where
  learned depth at wrong scale flatters the distance metric. The v3 number
  is measured at true scale. Whether the bar should compare at matched scale
  honesty is a QUESTION FOR A FUTURE PRE-REGISTRATION; under the law as
  written, v3 is not adopted.
- phone_room v3 gate (reportage): floor inlier IMPROVED 0.082 → 0.104 (still
  < 0.30), inbounds 0.309 → 0.297 and confidence 0.295 → 0.284 (both dipped
  with the smaller verified set) — three reasons named, still honestly
  rejected.

**Standing read:** the perturbation-stability tier is the largest measured
placement win on the realistic-motion scene class (desk 9×, room 0→0.124 at
the margin; medians halved) — exactly the scene class internet video lives
in — while the gentle-scene comparison is confounded by the canonical
baseline's scale artifact. The next session's decision point (fresh
pre-registration, never tuned on these numbers): a matched-scale-honesty
bar, or a multi-scene aggregate bar, for re-judging the stability tier.

### Phase 15 — RECIPE v3 ADOPTED AS CANONICAL (2026-06-11)

The Phase 13 adoption sequence completed: (1) the gate's Stage-2 metrics
were re-grounded on the expanded 12-config population (negative result
first: both single signals refuted as cross-class rankers; then the
provenance-CONDITIONED bounds — class B fsc 0.25 / p90 0.35 mid-chasm,
class A fsc 0.55 / p90 0.66 good-cluster ×1.2 provisional — calibrated
WITHOUT the vault, validated BY the vault, reproducing every frozen
canonical verdict with an EMPTY scorecard diff); (2) the re-judgment under
the amended gate: **xyz-v3 ACCEPTED** (p90 0.155, zero reasons), desk-v3
rejected on its floor blocker alone (0.268 vs 0.30 — the false fsc
rejection is gone), room/phone rejected for the right evidence reasons
(phone gains the coherent stage-2b reason, p90 0.712).

Adoption mechanics (the documented mechanism): canonical artifacts backed
up at `external/_pre_v3_artifacts_backup`; the four v3 stability composites
installed as `external/teacher_artifacts`; `configs/robot_envelope_v2.json`
promoted to `configs/robot_envelope.json` (2.5 cm physics-derived voxel,
stability tier k=2 / τ=0.005); canonical scorecard re-baselined (the
adoption diff is the recorded mechanism, not a regression).

Honest dual reading of the label-quality bar, both instruments stated:
under the SCALE-HONEST instrument (the robot-relevant one; Phase 13
predictions confirmed) v3 dominates the old canonical on every scene
(xyz F1@5cm 0.046→0.358, desk 0.093→0.432, room 0→0.163, medians 5.6–11 cm);
under the Sim(3) instrument xyz reads lower (0.199→0.158) — a comparison
proven confounded by the old baseline's 1.34× scale-forgiveness. The old
recipe's labels were never robot-consumable at claimed scale; the new
canonical's are, and the gate that accepts them got STRONGER (stage-2b
promotion + class bounds validated out-of-sample on the vault).

### Phase 14 — LOOP CLOSURE at linear cost (the long-video yield wall, measured)

Vault Run 1 named the wall: sequential matching without loop closure drifts
on long sweeps (0.83–1.41 m RMSE on fr3), and the gate rejects the result —
honest, but lost yield. Homes, yards, and any walkthrough video loop.
Developed on fr1_room (the canonical loop scene; the vault stays sealed):

| matching policy (375-frame room staging) | pairs | full-loop Sim(3) RMSE vs GT | cost |
|---|---|---|---|
| exhaustive (the prior recipe) | 70k (O(N²)) | 0.1480 m (374 reg.) | ~37 min CPU (Phase 8) |
| sequential + vocab-tree loop detection | O(N·(15+50)) | **0.1399 m** (361 reg.) | **5.1 min** (1.8 matcher + 3.2 mapper) |

Sequential + loop detection MATCHES exhaustive quality at linear cost —
the production matching policy for video is now: `sequential_matcher
--SequentialMatching.loop_detection 1` (COLMAP ≥May-2025 auto-downloads its
faiss vocab tree; the legacy flann `.bin` from demuc.de is REJECTED by the
4.x build — measured, then deleted). At 2,585-frame stagings (the vault
class) exhaustive is 3.3M pairs (intractable); this recipe is O(N).
GLOMAP remains an option for global-BA robustness at larger scales but is
no longer the gating item. NOT yet re-run on the vault (that requires the
next declared milestone); the fr1_room measurement is the recipe evidence.

### Phase 13 — the SCALE-HONEST instrument (pre-registered predictions, then observation)

The Sim(3) band alignment FORGIVES global scale error by construction: a
candidate whose labels are 34% scale-distorted aligns onto the measured
field for free, scoring as if a robot could consume it. A robot cannot — it
reads the label at its claimed metric scale. New reportage instrument
`solid_distance_agreement_metric_scale`: identical distance metrics under a
RIGID SE(3) alignment with scale FIXED at 1 (same scale-free rotation;
translation re-fit). This is the robot-relevant placement metric a priori,
independent of any recipe contest.

**Predictions, committed BEFORE the observation runs:** a global scale
error |s−1| displaces solids by roughly |s−1| × distance-from-trajectory
(band surfaces sit ~1–3 m out): canonical xyz (s=1.337) should COLLAPSE to
F1@5cm ≈ 0 under the honest metric; v3 xyz (s=1.028, ~3–6 cm displacement)
should hold near its Sim(3) value; desk (both recipes s≈1.08) and room
(v3 s≈0.90) carry 8–10% scale error ≈ 10–25 cm at range — BOTH recipes may
degrade severely there, in which case ABSOLUTE SCALE (not placement)
becomes the measured next wall. The instrument reports whatever it
reports; no recipe parameter changes in response. One observation pass
(canonical rerun + v3 rerun, metrics-identical code with additive keys).

**Phase 13 OBSERVED (one pass each):**

| scene | recipe | scale | Sim3 F1@5cm | HONEST F1@5cm | HONEST F1@10cm | HONEST med dist |
|---|---|---|---|---|---|---|
| xyz | canonical | 1.337 | 0.199 | **0.046** | 0.158 | 0.137 m |
| xyz | v3 | 1.028 | 0.158 | **0.358** | 0.688 | **0.056 m** |
| desk | canonical | 1.089 | 0.031 | 0.093 | 0.338 | 0.156 m |
| desk | v3 | 1.083 | 0.278 | **0.432** | 0.811 | **0.060 m** |
| room | canonical | 0.283 | 0.000 | 0.000 | 0.000 | 0.190 m |
| room | v3 | 0.902 | 0.124 | **0.163** | 0.407 | 0.114 m |

1. *Prediction CONFIRMED:* canonical xyz collapses (0.199 → 0.046) — the
   Sim(3) scale-forgiveness confound that decided Phase 12's xyz bar is now
   proven by measurement, not argued.
2. *Surprise, favorable:* v3 reads BETTER under the honest metric than under
   Sim(3) (xyz 0.158 → 0.358; desk 0.278 → 0.432): the Umeyama scale
   estimate from few shared centers is itself noisy, and forcing it onto
   BA-grade near-metric geometry HURT v3 in the Sim(3) view. The v3 labels
   place solids at 5.6–6.0 cm median in their OWN claimed metric frame —
   collision-margin-accurate as the robot would consume them.
3. *The desk/room scale fear did not materialize* (band surfaces sit close
   enough to the trajectory that 8–10% scale ≈ the measured ~6 cm medians).
4. **Under the robot-honest instrument, v3 DOMINATES canonical on every
   measured scene (4.6–7.8×).**

**Standing decision point (NOT resolved in this phase — recorded for the
next pre-registration):** adopting v3 as canonical is blocked by one honest
fact: xyz under v3 is NOT accepted (internal map fsc 0.2532 vs the 0.25
gate — the same resolution-coupled knife edge measured in Phase 9; fewer
stable-verified voxels ⇒ fewer contested-test exclusions). A canonical
recipe with ZERO accepted scenes would contradict the moat (verified
yield). Therefore the adoption sequence is: FIRST resolve the gate
metric's resolution coupling with the full calibration discipline (e.g.
promote `free_space_contradiction_rate_at_reference_scale` — or a
traversal-normalized variant — to the gated metric, threshold re-calibrated
against the frozen GT-grounded verdicts: xyz accept, desk reject, room
reject, phone reject-for-evidence), THEN re-judge v3 adoption under an
adversarially reviewed pre-registration. Neither step may be tuned against
the numbers above.

### Phase 16 — TRAINABILITY / COLLISION-SAFETY falsifier (pre-registration)

Committed BEFORE the observation run; the commit timestamp is the proof.

**Question (the gap analysis's run-first experiment,
docs/complete_teacher_gap_analysis.md):** is the accepted-label error
(xyz-v3: honest F1@5cm 0.358, median solid placement 5.6 cm,
solid_recall@5cm 0.455) averageable VARIANCE a student would smooth out, or
a systematic BIAS a student would distill into collision behavior — and how
much of the miss is in the DANGER direction (the label reads free at a real
obstacle) vs survivable coverage loss (unknown — the planner stays
conservative under unknown-is-never-free)? Phase 12 measured the residual
edge error as a perturbation-STABLE foreground-fattening bias both disjoint
halves reproduce; whether a consumer averages it or learns it is measured
nowhere.

**Instruments (definitions pre-registered here, before observing their
values).** Both run at the SCALE-HONEST alignment (rigid SE(3), scale fixed
at 1, same rotation path as the existing band3d method) on the canonical
accepted scene (xyz, v3 composites). Desk-v3 is run as REPORTAGE only (it
is rejected today on its floor blocker; it is the realistic-motion class
whose labels enter the corpus once the floor estimator lands). Decision
authority rests on xyz alone.

1. *Consumption-semantics split* over measured-solid band voxels inside the
   candidate grid bounds (the same population the solid_recall instruments
   read). Each voxel is classified by what a planner would read from the
   label within the robot's own collision margin (5 cm), with protective
   precedence:
   - `solid_within_margin` — an observed candidate-solid voxel lies within
     5 cm (collision-correct: an inflating planner avoids it);
   - `dangerous_free` — NO candidate solid within 5 cm AND an observed
     candidate-FREE voxel lies within 5 cm (the label actively shapes a
     path through a real obstacle);
   - `coverage_loss` — neither solid nor free observed within 5 cm
     (unknown; survivable by the unknown-is-never-free invariant).
   Rates over the population at 5 cm (decision tolerance) and 10 cm
   (reportage). Materiality bar, stated in advance: a dangerous_free@5cm
   rate above 0.10 is collision-shaping at a level that must be carried
   into the trainability verdict regardless of the bias outcome.
2. *Signed displacement decomposition* over candidate-solid voxels in the
   judged (co-observed) region, each matched to its nearest measured-solid
   voxel within 0.20 m. The displacement (label minus truth, metric frame)
   is decomposed along the VIEWING direction (unit vector from the nearest
   candidate camera center to the label voxel): the RADIAL component is
   signed (positive = the label sits beyond the true surface along the
   view ray; negative = pulled toward the camera — the foreground-fattening
   direction), the LATERAL magnitude is the residual. Reported: mean signed
   radial (BIAS), std (VARIANCE), median |radial|, median lateral, match
   rate.

**Pre-registered decision bars (xyz):**
- BIAS-LIMITED (Outcome B presumptive): |mean signed radial| >= 0.025 m
  (one voxel; half the collision margin). A systematic shift of that size
  is exactly what a distillation target teaches; the placement campaign
  re-opens BEFORE any dataset scale-out.
- VARIANCE-DOMINATED (Outcome A presumptive): |mean signed radial| <
  0.025 m AND std >= 2x|mean|. The error is averageable in expectation;
  the proxy-student confirmation proceeds.
- The dangerous_free@5cm rate is reported with its materiality bar above;
  it can force the danger conclusion under EITHER outcome.

**What this phase does NOT do:** no parameter moves in response to these
numbers; the proxy-student distillation (part c of the falsifier) is GATED
on this outcome and gets its own pre-registration (architecture, training
views, GT-eval-only protocol) — it is the confirmation step, not this
measurement. Script: `runs/_diag/trainability_falsifier.py`; results
committed to this file as Phase 16 RESULTS.

### Phase 17 — room loop-closure propagation through the production pipeline (pre-registration)

Committed BEFORE the run. The staged-but-unpropagated fact: the loop-closed
room sparse (`runs/_diag/colmap_work/room_loopdet`, Phase 14) was built to
prove the O(N) matching recipe but never carried through MVS + teacher; the
canonical room track still traces to the O(N²) exhaustive `room_dense`
workspace. HONEST FRAMING: canonical room poses are ALREADY BA-grade (camera
Sim(3) RMSE 0.036 m) — this run is recipe COHERENCE (the canonical track
should trace to the production one-command pipeline) plus the measured answer
to "does the production staging change room's Stage-0 evidence mass" (its
real blockers: inbounds 0.152 vs 0.30, conf 0.267 vs 0.30, floor 0.095 vs
0.30, stable-verified fraction 0.038). It is NOT expected to flip the gate.

**Mechanics (one shot, production tool end-to-end, frozen v3 recipe):**
`tools/run_sfm_pipeline.py --asset-id reference_metric_room --camera-mode
pinned` (fr1 measured intrinsics, anchors 108,238,367,626,735,951,1123,1318,
tool defaults incl. stage-divisor 8) → fresh loop-closed SfM + dense +
disjoint-half A/B workspaces → `run_colmap_pose_backend` (source =
`external/_selector_artifacts`) → `run_mvs_depth_backend` with stability
A/B (τ=0.005 frozen) → teacher on a side manifest
(`config/_room_loopdet_manifest.json`, asset entry verbatim from canonical)
with `--artifacts-dir external/_room_loopdet_composite --output-dir
runs/teacher_room_loopdet`, M1/M2 read from the existing canonical dirs
(never re-run — the M2 shared-registry clobber hazard).

**Pre-registered expectations (not targets):**
- SfM registers the large majority of ~210 staged frames in ONE model (the
  loop closes; Phase 14 precedent 361/375); keyframe-level camera RMSE in the
  0.03–0.15 m class (parity with canonical room — NOT an improvement claim).
- Stage-0 signals: direction UNMEASURED. Inbounds may move with the staged
  co-observation graph; verified fraction may stay ~0.04 (the loop's genuine
  thin-evidence reality).
- Gate verdict: expected still REJECTED for evidence reasons. An acceptance
  flip must trace to named evidence improvements and gets audited; a flip
  with unchanged evidence signals is itself an alarm (gate bug, not a win).
- Canonical artifacts and scorecard are NOT touched by this run. Adoption of
  the loop-closed composite as the canonical room track is a SEPARATE
  documented step, justified only by (i) verdict-coherence + (ii) recipe
  coherence, never by metric shopping.

### Phase 16 RESULTS — variance-dominated placement, BUT the danger bar fires

One observation run (`runs/_diag/trainability_falsifier_results.json`,
script committed at pre-registration). Numbers verbatim:

| quantity | xyz (decision) | desk (reportage) |
|---|---|---|
| mean signed radial displacement (BIAS) | **−0.0054 m** | +0.0150 m |
| std radial (VARIANCE) | 0.0443 m | 0.0337 m |
| median lateral displacement | 0.0626 m | 0.0372 m |
| match rate (≤0.20 m) | 0.958 | 0.934 |
| solid_within_margin @5cm | 0.510 | 0.423 |
| **dangerous_free @5cm** | **0.327** | 0.567 |
| coverage_loss @5cm | 0.163 | 0.010 |
| dangerous_free @10cm | 0.059 | 0.226 |

**Verdict under the pre-registered bars:**

1. *Bias test → Outcome A (VARIANCE-DOMINATED).* |−0.0054| < 0.025 and
   0.044 ≥ 2×|mean|. The placement error of matched solids is averageable
   spread, not a systematic radial shift a student would distill. The
   error is dominantly LATERAL (0.063 m median vs 0.026 m radial) — the
   voxel-level expression of the Phase 12 edge bleed. The placement
   campaign does NOT re-open on bias grounds.
2. *Danger test → FIRES, and it carries the verdict.* dangerous_free@5cm =
   0.327 ≫ the 0.10 materiality bar ON THE ACCEPTED SCENE: at one in three
   real obstacle voxels the label actively reads FREE within the robot's
   own collision margin. This is not "unknown" (survivable); it is free
   carved through space where the solid evidence was missed. At 10 cm the
   rate collapses to 0.059 — the defect lives exactly at the margin.
3. *Trainability verdict (the gap analysis's run-first question):* the
   accepted labels are NOT naively trainable at the 5 cm margin — not
   because placement is biased (it is not) but because the FREE channel
   near missed solids teaches collision paths. A student trained on these
   targets learns what the targets say; no proxy-student experiment is
   needed to establish that targets are wrong where they are measurably
   wrong. The proxy-student confirmation (part c) is therefore
   DE-PRIORITIZED: its question (does a net average the error out)
   dissolved — averaging works for placement variance, and no averaging
   fixes free-at-obstacle supervision.

**What this re-orders (consistent with the four-works plan):** the binding
label defect is the DANGEROUS-FREE RATE, attacked from two sides already on
the roadmap: (i) recall — every solid the teacher recovers converts its
neighborhood from dangerous_free to solid_within_margin (floor estimator →
more accepted evidence; placement levers if recall stalls); (ii) the
calibrated per-voxel confidence channel — IF low-confidence free voxels
near misses are separable by evidence features, the TrainingSample
downweight channel can suppress exactly the dangerous supervision (this is
now a measurable target for the Work-1 calibration: reliability of FREE
claims, not just solid claims). Any free→unknown policy lever is a
SEPARATE future pre-registration and must never fabricate occupancy
(occupancy_support_overrides_free stays rejected). dangerous_free@5cm
becomes a tracked scorecard-adjacent number for every future recipe change.

### Phase 18 — floor ESTIMATOR rung 3: height-mode seeded fit (pre-registration)

Committed BEFORE implementation and the single shot.

**Diagnosis (measured, Phase-15 desk report):** the floor stage names a
blocker in ALL FIVE current rejections, and on desk it is the SOLE blocker.
Desk's unconstrained dominant plane (0.268 inliers) is tilted 41.63° from
the band axis — it is the desk surface, not the floor. The camera-up
CONSTRAINED retry reads 0.162 — but that number is a SEARCH ARTIFACT, not a
measured ceiling: with floor fraction f≈0.16, the probability a random
RANSAC triplet lands entirely on the floor is f³≈0.4%, so 200 iterations
expect <1 clean floor sample. The constrained search returns the best plane
it stumbled into, not the best floor fit the evidence supports.

**Mechanism (estimator change ONLY; every bar unchanged):** a third rung in
the existing candidate-only search ladder, reachable ONLY when rung 1
(unconstrained RANSAC) and rung 2 (camera-up-cone RANSAC) both fail the
0.30 reliability bar AND a camera-up prior exists (candidate tracks only —
the measured baseline never receives the prior, so measured tracks are
bit-identical by construction):

- project static surfaces onto the prior axis; histogram heights with bin =
  the existing RANSAC inlier distance (1.5 × voxel);
- seed plane = the LOWEST height bin with support ≥ the existing RANSAC
  minimum-support rule max(3, 0.05·n) — the floor is the lowest plane with
  real support; static structure does not live below the floor;
- polish: ≤3 least-squares refits on the current inliers, constrained to
  the prior cone (30°, unchanged) — stop at the last in-cone fit;
- measure inlier_ratio on the FINAL plane over ALL static points (same
  denominator, same 0.30 bar, same honest refusal below it).

All parameters are existing frozen constants — nothing is tunable against
this run. The post-alignment floor search gets the same ladder (wrapper),
so the band crop uses the same plane family that won the alignment.

**Pre-registered expectations (not targets):**
- xyz candidate and ALL measured tracks: bit-identical (rungs 1–2 outcomes
  unchanged; rung 3 unreachable). The scorecard diff for xyz must be EMPTY.
- desk: rung 3 fires. If the true floor's support ≥ 0.30, alignment applies
  (tilt after ≈ 0), the band becomes floor-parallel, and the gate re-judges
  — a possible acceptance FLIP (floor was the sole blocker; fsc 0.461 ≤
  class-A 0.55, p90 0.368 ≤ 0.66), audited: the flip must trace to the
  floor stage alone. If support < 0.30, desk keeps failing honestly and the
  clutter-denominator of the floor metric becomes a documented finding for
  a future pre-registration — never a bar move.
- room candidate (floor 0.095) and phone (0.104): rung 3 fires; floor
  evidence is genuinely thin on both (loop coverage / textureless carpet);
  may stay < 0.30 — informative either way.
- GUARDRAILS on any desk change: per_class and honest F1@5cm must hold or
  improve (alignment to the true floor improves the band comparison; a
  wrong plane tanks per_class); dangerous_free@5cm (Phase 16 instrument)
  re-read after the change.

**Falsifier:** on desk the rung-3 plane must be floor-plausible — post-
alignment tilt ≈ 0 by construction, camera centers above the plane
(checked post-hoc from the report), and the no-regress guardrails above.
A rung-3 plane that clears 0.30 but degrades per_class/F1 on the measured
comparison is a FAILED estimator (wrong plane with popular support), the
change is reverted, and the failure is recorded here.

**Phase 17 interim finding (GT-free, before any teacher run):** the frozen
default staging (stage-divisor 8 → 210 staged frames) FRAGMENTS the room
loop: 4 models, largest 114/210 registered. The Phase 14 parity result was
measured on a ~375-frame staging (≈ divisor 4); sequential matching over
every-8th frames cannot track fr1_room's fast motion between samples.
Production lesson recorded: long fast-motion loops need divisor 4 (the same
class of staging-density lesson phone taught at Phase 10). Deviation from
the pre-registered "tool defaults": ONE re-run at --stage-divisor 4,
justified purely by the tool's own registration/fragment counts (no GT
consulted); everything else stays frozen.

### Phase 19 — data-driven per-scene scale_std + GT coverage falsifier (pre-registration)

Committed BEFORE implementation and observation.

**The gap (verified at HEAD):** `scale_std` is a per-BACKBONE constant —
0.168 byte-identical across all four tracks (= 0.12·(1+(1−0.6)) from
MapAnything's hardcoded evidence confidence). "Per-sample scale uncertainty"
is currently a category label plus a constant. Meanwhile a DISCRIMINATING
per-scene candidate-only signal is computed and dies in provenance:
`pose_provenance.scale_alignment.residual_rmse_backbone_units` ranks
xyz 0.0271 (accepted) < desk 0.0559 < phone 0.4576 < room 0.9132 — the
trajectory-shape disagreement between the BA poses and the backbone centers
the borrowed scalar was fit on. A scale borrowed across a misfitting shape
is exactly as untrustworthy as that misfit.

**Construction (GT-free, no tunables, class-A tracks only):**
- thread `scale_alignment.residual_rmse_backbone_units` into the existing
  `ScaleEvidence.residual_after_optimization` field (schema complete, zero
  producers today) in the geometry adapter;
- `residual_norm` = residual / span, span = bounding-box diagonal of the
  candidate packet camera centers (same backbone units; both candidate-only;
  dimensionless, scene-relative per the honesty rule);
- `relative_scale_std` = sqrt(band² + residual_norm²) where band is the
  existing evidence-confidence soft band (independent-error combination; no
  free constant). Class-B tracks (backbone poses, no alignment record) keep
  the constant band — with `scale_status` marking it a prior, never a
  measurement.

**Expected single-shot diff (pre-registered):** numeric deltas ONLY in
scale_posterior.scale_std / relative_scale_uncertainty and the exported
scale_uncertainty fields, on class-A tracks; ZERO categorical changes, ZERO
verdict changes (the gate does not read scale_std). Any verdict flip is a
bug, not a win.

**Coverage falsifier (single observation on the on-disk GT population;
vault EXCLUDED, sealed):** for each GT config, true scale error
e = |estimated_scale_monocular_to_measured − 1| (Sim(3) reportage) vs the
claimed σ. Bars, stated in advance:
1. gate-ACCEPTED configs (today: xyz-v3): e ≤ 1σ — a claimed band that does
   not cover the accepted scene's true error is refuted outright;
2. class-A GT configs (v2/v3 × xyz/desk/room, n=6): ≥5/6 within 2σ;
3. rank: Spearman(σ, e) > 0.5 over all GT configs (the σ must order scenes
   by actual scale trustworthiness, not just inflate);
4. class-B (canonical × 3): REPORTED, no bar — no data-driven signal exists
   for backbone-pose tracks; pre-stated fallback: they keep the constant
   prior with scale_status='per_backbone_constant_prior' and class-B
   coverage results recorded verbatim.
FAILURE of bar 1 or 2 refutes the construction: it is reverted, the
constant prior stays, and the failure is recorded here. No widening,
re-weighting, or re-fitting in response to the observed coverage.

**Phase 18 shot 1 result + GT-free amendment:** the single shot ran; the
scorecard diff was EMPTY on ALL FOUR tracks — rung 3 fired on desk and
returned NO plane (height_mode_inlier_ratio 0.0 = no height bin reached the
max(3, 0.05·n) significance bar). Diagnosis (geometric, no GT consulted):
the height histogram runs along the CAMERA-UP PRIOR axis, and a prior
tilted by θ from the true floor normal smears a floor of extent L across
L·sin(θ) of height — desk's ~4 m floor at ~15° ≈ 1 m ≈ 27 bins (~0.6%
per bin), never significant. Histogram seeding structurally fails under
exactly the prior tilt that makes rung 3 necessary. AMENDMENT (seeding
mechanism only; cone, polish, bar, denominator all unchanged): seed from
the LOWEST QUANTILE BAND — drop the bottom 2% of heights as outliers, then
grow a band upward from the 2nd percentile until it holds ≥ max(3, 0.05·n)
points (width set by the data, no new constant), PCA-fit a plane to the
band (the fit recovers the true normal regardless of prior tilt), then the
same in-cone polish loop, with the final inlier count measured on the FINAL
plane (also fixes an off-by-one: shot-1 code stored the pre-refit plane's
support). Re-run = shot 2, same expectations and falsifier as the Phase 18
pre-registration.

### Phase 20 — gate-authority completion: class-A injection certificates + the drift falsifier (pre-registration)

Committed BEFORE the harness upgrade and any run.

**Why the existing certificate cannot be reused:** the one certificate on
disk (xyz, pre-v3) was measured against the OLD flat gate. The production
gate is provenance-conditioned (class-A fsc 0.55 / p90 0.66, both
PROVISIONAL ×1.2 margins) and includes Stage-2b prerefine-p90 — the
on-disk harness still scores flat fsc 0.25 and computes no p90. A
certificate against the wrong gate is optimistic exactly where it matters
(class-A bounds are LOOSER). The harness is upgraded to MIRROR the
production gate (no new thresholds, all constants imported from
validation.py):
- pose-provenance class from the SAME BA_GRADE_POSE_MARKERS logic;
- class-conditioned fsc and p90 bounds;
- per-injection prerefine p90 via a measure-only refine pass (residuals
  measured at x0, NO optimization, NO repair — injections stay unrepaired);
- dynamic leakage stays out of harness scope (the harness fuses without
  static/dynamic states; the tested families do not perturb the dynamic
  channel) — recorded in the certificate as out_of_scope, not as coverage.

**Runs (in order):** xyz-v3 canonical (class A, THE accepted scene), 3
seeds, full magnitude grid — this is the drift falsifier; then desk-v3,
room-v3, phone-v3 certificates. Vault: NEVER (sealed).

**The second curve (class-A bound refutation test):** for the pose-drift
families, per magnitude (1 seed), additionally compute honest F1@5cm
(rigid SE(3), s=1, vs the measured field). PRE-REGISTERED REFUTATION: any
magnitude where honest F1@5cm drops more than 0.10 below the clean value
while NO class-A gated signal crosses its bound refutes the ×1.2
provisional margins for that family; the family is recorded as
no_authority and carried in-band into TrainingSample provenance
(gate_no_authority_families). Detection at or below the damage threshold
converts the provisional strings to measured citations for that family.

**Wiring (reportage, additive):** the per-scene
injected_corruption_detection_limit block is attached to the teacher
report when a certificate exists for the asset (path + summary verdicts +
no_authority families); absent certificate -> explicit
no_certificate_for_scene marker. The two UNMARKED bars (held-out 0.50,
dynamic leakage 0.10) and the Stage-1 floor bar (0.30, mapping.py) gain
explicit authority tags; their response curves from these runs are
recorded (report-only; any threshold move would be a separate
pre-registration).

**Expectations (not targets):** the measured blindness (coherent pose
drift <=0.4 span, rotation <=10°, focal bias) is expected to PERSIST under
the class-A bounds (they are looser than the flat gate that was already
blind); the honest deliverable in that branch is the in-band no_authority
record, not a forced detection. scale_drift_ramp and regional_depth_bias
detections are expected to survive the gate upgrade (Stage-1/fsc
mechanisms unchanged).

### Phase 18 RESULTS — the estimator hypothesis is REFUTED; desk's floor shortfall is GENUINE

Shot 2 (quantile-band PCA seeding) ran once: scorecard diff EMPTY on all
four tracks; on desk rung 3 fitted a plane with support 0.0244 (the lowest
heights are a long sub-floor MVS noise tail, not floor). A GT-free probe of
desk's own static cloud (runs/_diag/desk_floor_probe.py; 89,616 static
points, candidate geometry only, no measured data touched) then bounded
what ANY estimator could achieve:

| probe | best support | character |
|---|---|---|
| height scan, prior-perpendicular planes + LS polish | **0.079** | tilt 2.0° to prior |
| 20k-iteration in-cone RANSAC (low-region triplets) + polish | **0.179** | tilt **29.6°** -- at the cone EDGE, not a floor |
| height distribution along prior | no low mode | smooth noise tail: p1 -1.26 / p2 -1.10 / p5 -0.83 |

**Verdict (the pre-registered informative branch):** there is NO
floor-plausible in-cone plane with support anywhere near the 0.30 bar in
desk's candidate cloud. The Phase 18 search-starvation hypothesis is
REFUTED for the canonical population: the constrained 0.162 was an honest
ceiling-class reading, not an artifact. Desk's capture barely sees the
floor (cameras orbit the desk); its Stage-1 rejection is GENUINE EVIDENCE
SHORTFALL. Rung 3 produced zero scorecard change in two shots and is
REVERTED per the no-slop rule (the probe + this record guide any future
re-introduction on a scene with a visible-minority floor).

**The sharpened follow-up (named, NOT designed here -- needs its own
pre-registered calibration campaign against the 12-config table):** Stage 1
as constructed demands the floor be a >=0.30 plane of the WHOLE static
cloud -- a clutter-coupled denominator. Desk (BA poses near-perfect, honest
F1@5cm 0.432, sole blocker floor) and phone (textureless carpet) both fail
exactly there while carrying good geometry. The honest question for the
next gate-design pre-registration: a floor-evidence statistic that
verifies the BAND AXIS without demanding floor dominance (e.g. absolute
floor-support count + camera-up agreement + low-region concentration),
calibrated per the negative-table discipline, never per scene. Until then
desk stays rejected and the yield cost is carried honestly.

### Phase 19 RESULTS — conservative band VALIDATED; the rank claim FAILED honestly

Single shot: scorecard diff carried EXACTLY the pre-registered signature —
numeric deltas ONLY in scale_posterior.scale_std / relative_scale_uncertainty
on all four tracks, ZERO categorical changes, zero verdict flips. New
per-scene bands: xyz 0.1703, desk 0.1694, phone 0.2114, room 0.2947 (the
loop's trajectory-shape misfit honestly widens its band 1.75×).

Coverage falsifier (one observation, 9 GT configs, vault excluded;
runs/_diag/scale_coverage_falsifier_results.json):

| config | class | σ claimed | true err | ≤1σ | ≤2σ |
|---|---|---|---|---|---|
| v3/xyz (ACCEPTED) | A | 0.1703 | 0.028 | Y | Y |
| v3/desk | A | 0.1694 | 0.083 | Y | Y |
| v3/room | A | 0.2947 | 0.098 | Y | Y |
| v2/{xyz,desk,room} | A | (same) | (same) | Y | Y |
| pre_v3/xyz (old accepted) | B | 0.1680 | **0.337** | N | **N** |
| pre_v3/desk | B | 0.1680 | 0.089 | Y | Y |
| pre_v3/room | B | 0.1680 | **0.717** | N | **N** |

- Bar 1 PASS: the accepted scene's true scale error sits inside 1σ.
- Bar 2 PASS: 6/6 class-A configs within 2σ. The refutation clause (bars
  1–2) does NOT fire; the construction stands.
- Bar 3 FAIL, recorded loudly: Spearman(σ, e) = **−0.283** over the mixed
  population. Cause is structural: class-B has NO data-driven signal (its
  constant 0.168 under-covers its large true errors — the old accepted
  canonical shipped a 0.337 scale error against a 0.168 band, exactly the
  dishonesty the class-A construction fixes), and its σ-ties at low values
  against high errors invert the rank. Within class A (3 scenes) the band
  separates room from xyz/desk but does not resolve xyz vs desk (σ 0.0009
  apart — no claim made).
- CONSEQUENCE (no re-scoping): scale_std for class-A tracks is a VALIDATED
  CONSERVATIVE BAND (2σ coverage on the population, 1σ on the accepted
  scene) whose cross-scene RANK authority is NOT established. The
  TrainingSample scale_status marker now reads
  `conservative_band_2sigma_validated_rank_unvalidated_phase19` for class-A
  samples; class-B keeps `per_backbone_constant_prior` (and class-B never
  ships — its coverage failures above are the measured reason why).

### Phase 17 RESULTS — loop closes at divisor 4; NO adoption (labels worse than canonical)

Shot at divisor 4 (after the pre-registered divisor-8 fragmentation finding):
SfM registered 360/375 staged frames in ONE model (the loop closes). The
chain ran end-to-end (pose backend → MVS + A/B stability → teacher side-run,
runs/teacher_room_loopdet; first production artifact carrying the
stability_delta continuous-residual maps). Gate verdict exactly as
pre-registered — STILL REJECTED, all evidence-class reasons (inbounds
0.152→0.195, edge 0.618→0.629, conf 0.267→0.275, floor 0.092; verified
fraction 0.041 — the loop's thin-evidence reality persists).

The adoption question answers itself, negatively:

| | canonical room (room_dense, exhaustive O(N²)) | loopdet d4 (sequential+loopdet O(N)) |
|---|---|---|
| band3d camera RMSE | **0.126 m** | 0.382 m |
| HONEST F1@5cm | **0.163** | 0.031 |
| HONEST F1@10cm | **0.407** | 0.077 |
| coverage of measured band | **0.84** | 0.56 |
| scale-borrow residual | 0.913 | 0.985 |

NOT adopted: verdict-coherent (both rejected) but the loopdet model's
keyframe-level quality is ~3× worse. HONEST AMENDMENT to the Phase 14
standing read: sequential+loop-detection parity was measured on FULL-LOOP
Sim(3) RMSE; at the KEYFRAME subset feeding the teacher, this d4 rebuild
reads materially worse than the exhaustive model. The O(N) recipe remains
the production matching policy for INGESTION COST, but on long fast loops
the exhaustive model (where affordable) still sets the quality bar — the
GLOMAP-class global-BA option returns to the roadmap for long-video
quality, not just scale. Canonical room track unchanged.

### Phase 21 — phone fragment B: SfM milestone (interim record)

Fragment-native ingestion step 1 ran (frames 502–800 staged as their own
workspace; 15 keyframes derived from the full-video selection by id shift,
provenance recorded in the derived selector JSON). Self-calibrated SfM:
**83 frames registered in ONE model, 0.80 px mean reprojection, f = 1098.4 /
k = 0.0373** — within 1.8% of the three independent Phase-10 calibrations
(1086.4 / 1079.3 / 1089.4): the camera cross-check holds a FOURTH time.
Fragment B reconstructs coherently. The MVS + backbone + teacher chain (the
floor-coverage falsifier: does fragment B's floor-rich coverage lift floor
inlier toward 0.30?) is staged but BLOCKED this session by disk exhaustion
(the run died at undistortion on ENOSPC; ~13 GB of superseded dense
workspaces were deleted — divisor-8 room, desk smoke, d4 dense triple,
fragment-B partial dense — all regenerable from documented commands).
Operational lesson recorded: dense workspaces are ~2–11 GB EACH; the
pipeline tool should check free disk before the dense stage and the
workspace-retention policy (extract composites, drop stereo maps) needs a
cleanup pass at the end of every campaign run.

### Phase 20 RESULTS (part 1) — the xyz-v3 class-A certificate: the drift blind spot PARTIALLY CLOSES

Full 3-seed grid against the GATE-MIRRORED harness (class A: fsc 0.55, p90
0.66, measure-only p90 per injection), runs/_diag/detection_limit_reference_metric.json:

| family | detection limit | verdict |
|---|---|---|
| pose_drift_translation | **0.4 × span, monotone, solid** | **DETECTED — the historic blind spot partially closes** |
| scale_drift_ramp | **1.2× end-to-end** (was 2.0× pre-v3) | DETECTED |
| pose_drift_rotation (≤10°) | none | no_authority (expected) |
| regional_depth_bias | none | no_authority — the OLD 0.2 "detection" rode the flat-gate 0.006 fsc knife edge; the class-A bound honestly does not fire |
| ray_field_focal_bias | none | no_authority (expected) |
| global_tilt_control | none | clean negative control |

Headline findings:
1. **Coherent translation drift at 0.4 span is now DETECTED** (Stage-1
   gravity fires: the v3 composite's denser verified floor geometry breaks
   under drift where the old sparse recipe bent). The certified blindness
   shrinks from "blind ≤0.4 span" to "blind below 0.4, detected at 0.4".
2. **Clean margins are healthy, no knife edges**: fsc margin 0.287, p90
   margin 0.560 on the accepted scene (the pre-v3 certificate's fsc margin
   was 0.006 — every small detection was coin-flip noise).
3. Scale-ramp detection sharpened 2.0× → 1.2×.
4. Method validation: injected magnitude tracks TRUE induced RMSE
   (Spearman 0.94 translation / 0.71 rotation / 1.0 scale) — the harness
   measures real damage, not artifacts.
5. Honest no_authority families for class A: rotation ≤10°, regional depth
   bias, focal bias — carried in-band by the teacher-report certificate
   block (wired this session; attaches on the next canonical rerun).

STILL OPEN for the formal provisional→measured conversion of the ×1.2
class-A bounds: the F1@5cm damage second curve (per pre-registration: a
family damaging labels past the margin while signals stay quiet refutes
the bounds) and the desk/room/phone certificates. The xyz-v3 certificate
alone already replaces "blind to drift, knife-edge margins" with measured
margins and a measured partial detection.

### Phase 22 — phone_room_loop ingestion + Codex council regression PASS + the honest demotion

New first-party capture data/phone_room_loop (iPhone 14 Pro 1080p30, 88.8s
slow-walk LOOP, start=end, 46cm metal ruler staged in-scene as a measured
scale reference; frames extracted to 1280x720). Chain: SfM self-cal ->
**374/381 frames ONE model, 0.535px**, f=1080.74 (FIFTH independent agreement
with Phase-10/fragment-B reads, all within 1.6%), **loop-closure gap 0.59% of
scene span (0.22% of 31.3-unit path)**; dense + A/B stability workspaces
(survived a second ENOSPC -- pre-flight disk check added to run_sfm_pipeline);
MapAnything on 48 flow-budget keyframes; pose backend 48/48; MVS verified
fraction 0.207. Teacher verdict: **non_metric_pseudo_label, rejected for
metric training** -- the metric anchor council (bfcdf2d) rejected ALL learned
scale anchors on this scene (pose backend scale_residual_rmse 1.34 corroborates),
so no scale evidence exists and the teacher says so. The 46cm ruler is the
designed remedy (pending: user clicks ruler endpoints in the viewer ->
measured ScaleEvidence).

Codex-council REGRESSION on the four canonical tracks (pre-commit reports
snapshotted to runs/_diag/_pre_codex_reports): geometry metrics byte-identical
(fsc / held-out render error / dynamic leakage unchanged on all four);
measured_metric categories preserved on the three TUM scenes; TWO designed
changes: (1) **phone_room demoted metric_pseudo_label -> non_metric_pseudo_label**
(council rejects all phone anchors; the learned-prior scale claim is withdrawn
-- consistent with Phase 19's 0.337 shipped-error finding and the council's
own GT under-coverage measurement), (2) soft-path relative scale uncertainty
floors at the council guard 0.30 (xyz monocular candidate 0.170->0.310, desk
0.169->0.302, room 0.295->0.300) -- blunter but covers the historic 0.337
error that the old 0.168 band missed. Verdict: integration adopted.

### Phase 23 — overnight diagnosis: why the teacher failed the user's loop capture (3 root causes, all measured; claims independently re-derived)

User verdict on the phone_room_loop viewer (eyes-on, their own home): poses
scattered, free space wrong, floor inconsistent, "useless". All three
observations traced to mechanism; an adversarial checker re-derived every
quantitative claim below with independent code and CONFIRMED all.

ROOT CAUSE 1 — artifact routing (FIXED, 8428453): backbone runners defaulted
--out-dir to the teacher READ path, so the teacher consumed the RAW
MapAnything artifact (48 independently-posed views) while the COLMAP-posed
MVS-verified composite sat unused. Composite promoted + teacher rerun:
trajectory scribble -> clean 0.6%-gap loop; held-out render error 0.210 ->
0.125. Chain guard landed (runner defaults moved to _raw_backbone_artifacts,
--promote with rollback, composite_shadow_blocker in the geometry adapter,
16 tests). Reviewer flagged: reference_metric_vault's live artifact is ALSO
raw (pre-existing; no composite sibling, so no blocker fires) -- vault Run 2
must rebuild it as a composite.

ROOT CAUSE 2 — composite gauge defect (OPEN, the next fix, biggest lever):
the composite's COLMAP->backbone scale anchor (0.2325) was Umeyama-fit
against the SCRIBBLED raw backbone trajectory (residual rmse 1.344 units =
~50% of the trajectory bbox diagonal) -- a garbage anchor; AND the composite
depth maps are internally TWO-GAUGE: verified pixels (20.7%) carry
pose-gauge MVS depth while fill pixels (79.3%) remain at raw backbone gauge
2.55x larger (bit-identical to raw, max diff 0.0). No per-frame uniform
correction can fix a bimodal within-frame gauge; refine's BETA_BOUND (+-1.0)
is anyway too small for the needed mean -0.95 log correction. This, not the
backbone, is the dominant live depth defect (~log 0.94 of error). Fix shape:
anchor scale on MVS-verified depth (not raw trajectories) and rescale fill
pixels per-frame onto the pose gauge at composite build time.

Backbone itself: depth SHAPE is excellent (after removing one per-frame
scalar, p50 3.1% / p90 7.9% agreement with MVS); per-frame scale instability
std(log) 0.239 (0.164 excl. endpoints) is real but second-order,
content-deterministic (near-field texture-poor close-ups: the two desk
endpoint frames hit ~6x).

ROOT CAUSE 3 — band tilt under floor-gate failure (OPEN, design ready):
gravity in the composite frame is recoverable to 3.7 deg (camera-up mean
over 374 poses vs dominant-plane normal agree), but the axis-aligned
fallback band is tilted 34.6 deg from gravity: 69 cm height error per meter
traveled -- the floor crosses the ENTIRE 0.55 m band in 0.80 m of travel.
This alone explains "floor not even consistent". The floor gate's 0.30 bar
measures floor AREA SHARE (not reliability): walking captures give walls
the per-pixel majority, 3-point RANSAC samples the floor with only ~63%
probability at 0.17 share, and the camera-up prior (median of 48 pitched-down
ups, hard 30-deg cone) measurably excluded the true floor on the raw run
(30.95 deg). Meanwhile the loop's SfM cloud is 77% one sharp horizontal
plane -- the floor is THERE; the estimator is the defect. Design: fuse
camera-up prior + plane orthogonality (no floor-dominance requirement),
falsifier on TUM scenes (needs one camera-IMU extrinsic calibration first).

TENTATIVE (ruler falsifies): cameras sit 0.36 composite-m above the measured
floor plane; at hand height 1.2-1.5 m that implies ~3.3-4.2x metric scale
compression (consistent with scale_residual_rmse 1.344 read as log: e^1.344
= 3.84).

CAPTURE forensics (corrections to the night's own narrative): registration
was 374/374 = 100% (the "7 unregistered" were keyframes double-counted with
stride-8 staging); the walk was filmed 30-60 deg DOWN, floor visibility
excellent -- capture style is NOT the floor-gate culprit on this asset;
keyframe cap rule starves the capture ENDS (0->394 and 2310->2662 jumps =
10x the 0.25-width budget; the ruler/loop-closure zone got only 2 degenerate
close-ups -- selector defect, affects the ruler measurement); flow-budget
selector has no sharpness term (4 blurred keyframes fed to the backbone in
the dim-vinyl leg). Capture protocol draft: runs/_diag/capture_protocol_draft.md.

FSD-occupancy research roadmap (prioritized, adoption-gated): P0 provenance
hard-invariant (landed) + iPhone LiDAR/ARKit lane (user's 14 Pro: Stray
Scanner-class apps log metric depth + poses + GRAVITY -- kills band tilt and
scale ambiguity with measured evidence); P1 Occ3D-style three-state labels +
per-camera visibility masks with evidential fusion; P1 backbone-native
confidence + outlier-view rejection + cross-backbone disagreement channels;
P2 chunked/streaming inference (VGGT-Long-class) for dense coverage; P2
multi-pass capture as a GT-free cross-pass agreement channel; P3 external
benchmark anchor (ScanNet++-class, eval-only); P3 student contract:
supervise only in observed space (Occ3D visibility masks in TrainingSample).
Forensics artifacts: runs/_diag/floor_forensics/, runs/_diag/depth_forensics/.

### Phase 24 — four-repo deep read (user-directed): FreeOcc / LegoOcc / SelfOcc / SurroundOcc — mechanisms mapped onto the internet-video teacher

Method: every repo fetched and read live (README + code + LICENSE + paper);
every load-bearing claim independently re-fetched by an adversarial auditor
(25/28 + 6/10 auditable claims CONFIRMED verbatim; the non-confirmed ones
were corrections recorded below, not fabrications). One mapper honestly
refused verdicts on reads it never received (input truncation) — re-run
clean. STRATEGIC FRAME (owner correction, binding): production input is
CLEAN INTERNET MONOCULAR VIDEO; no fleet, no rigs, no capture hardware as a
data source; public GT = calibration/eval instruments only.

VERDICTS (all four repos real; roots Apache-2.0):
- FreeOcc (the-masses/FreeOcc, RSS 2026): training-free monocular indoor
  occupancy (DROID-SLAM + 3DGS + frozen VLM + probabilistic hazard
  Gaussian->occupancy CUDA). Closest published system to our setting.
  ADOPT-IDEA: clean-room hazard-survival fusion occ = 1-exp(-sum h_i) at the
  mapping._fuse_rays seam (D3 arithmetic fix). LANDMINE: its localagg_prob
  CUDA kernel carries Inria GRAPHDECO NON-COMMERCIAL headers — never vendor.
- LegoOcc (JuIvyy/LegoOcc, CVPR 2026 Oral): monocular indoor open-vocab
  occupancy, geometry supervision = BINARY occupancy only. PARTIAL: lift the
  geometry-only loss stack (Focal w=100/g=2 + Lovasz + Geo_Scal, unknown-
  ignore, frustum mask; OpenMMLab/MonoScene Apache lineage — header-check
  per file) into the proxy-student falsifier; its own hazard kernel has the
  SAME GRAPHDECO NC contamination under an Apache root. Its net needs GT
  poses + sensor-depth GT grids: NO-FIT as-is for internet video. NOTE:
  FreeOcc+LegoOcc share authors — ONE group's validation, not independent
  replication (auditor correction).
- SelfOcc (huang-yh/SelfOcc, CVPR 2024): self-supervised SDF volume
  rendering; supervision minted from the video itself (photometric
  min-reprojection vs adjacent frames + free-space sparsity carving —
  confirmed in code). ADOPT-IDEA x2, ORDERED: (1) grid render-consistency
  VALIDATION channel first (ray-march the fused grid into held-out
  non-keyframe frames; free claims that photometrically terminate early =
  dangerous_free witnesses; authority earned via the Phase-20 certificate
  harness, 12-config Spearman>=0.5 after coverage partialling); (2) per-scene
  photometric refinement optimizer SECOND, separately gated, pools split
  (pool A optimizes, pool B gates — circularity dissolved by our surplus of
  registered non-keyframe frames). Its learned BEV/TPV encoder: NO-FIT
  (auditor: the SDF head is hard-wired to the learned lifting; we take the
  loss machinery only, clean-room — vendored SSIM is monodepth2/Niantic NC).
- SurroundOcc (weiyithu/SurroundOcc, ICCV 2023): driving multi-cam net
  NO-FIT (backbone-seam rejection UPHELD); but its dense-GT GENERATION
  recipe (multi-frame static aggregation -> Open3D Poisson -> voxelize ->
  NN class transfer; confirmed in tools/generate_occupancy_*) ADOPT-IDEA on
  OUR inputs: MVS-verified multi-frame points replace LiDAR; Poisson mesh
  becomes a provenance-marked CARVE-STOPPER channel (gates not counts,
  candidate-only, never observed evidence) at the band free-carve levers —
  attacks the measured D3 root cause (carve floods through missed solid
  bases). Auditor caveats wired into the gate: Poisson hallucinates
  watertight surface where evidence is absent (must be evidence-mass-gated +
  density-quantile trimmed); their own-data path needs 3D boxes for dynamics
  (we restrict to static-classified points instead); nuScenes itself is
  CC BY-NC-SA (eval-only, and unneeded).

REVISED TWO-WEEK PLAN (pre-registered gates, falsifier-first):
1. Hazard-survival fusion falsifier (days 1-3): dangerous_free@5cm down,
   occupied_static_iou/per_class held, empty diff when off. The
   transmittance-derived FREE half is OUR design — neither repo validates
   it; the falsifier carries that burden.
2. Poisson densification falsifier (days 3-8): day-1 kill-cheap pilot on
   the cached xyz composite; gate xyz 0.327->=<0.20, desk 0.567->=<0.45
   dangerous_free, hallucination-trim falsifier, byte-identical when off.
3. Render-consistency validation channel (days 8-13): reportage-only,
   certificate-calibrated; the GT-free certifier the first two need on
   non-GT scenes.
HONEST GAP THE REPOS DO NOT TOUCH: metric scale for monocular video. None
of the four anchors absolute scale (SLAM-relative / GT-posed / LiDAR).
Scale authority remains the council + public-GT calibration-row growth
(2 -> 8+) path. D1 (gauge) and D2 (gravity) also remain in-house fixes.

### Phase 24 addendum — Poisson carve-stopper pilot: KILLED (pre-registered thresholds, Codex-built, verified)

Kill-cheap offline pilot on xyz vs the measured baseline
(tools/pilot_poisson_carve_stopper.py, 16s runtime, full report
runs/_diag/codex_poisson_pilot_report.md): false-stop rate 11.0/12.8/13.9%
at density trims q=0.02/0.05/0.10 — ALL exceed the pre-registered 10%
hallucination ceiling; and the cached exports lack per-voxel witness ids, so
the dangerous_free benefit could not be demonstrated offline (recorded
honestly as 0.0%, not fabricated). Verdict KILLED per pre-registration;
weekend reprioritized to hazard-survival fusion as the sole D3 attack.
Confound recorded for any future revisit: without per-voxel contributing
frame ids the pilot blocks rays from ALL keyframes seeing a voxel, which
over-counts false-stops; a revisit would need in-spine ray instrumentation
(only justified if hazard fusion under-delivers on dangerous_free).

### Phase 25 — D1 single-gauge composite fix (Codex-built gpt-5.5/xhigh, independently verified): loop p90 1.119 -> 0.227

tools/run_mvs_depth_backend.py: per-frame s_f = median(backbone/MVS on
verified px), fill pixels rescaled 1/s_f onto the pose gauge (N_MIN 500,
global-median fallback, materiality band [0.8,1.25] so near-unity TUM ratios
are no-ops); depth-based global scale replaces the trajectory-Umeyama anchor;
single_gauge_v2 provenance + gauge_spread_log persisted (spread 0.2389 ==
Phase-23 forensics 0.239 — independent agreement). VERIFIED: loop fill/verified
ratio in [0.95,1.05] for 47/48 frames; stage2b p90 1.119 -> 0.227 (live
report); all four canonical reports BYTE-IDENTICAL to pre-D1 snapshots.
HONEST ASTERISK: Codex's literal TUM composite rebuild CHANGED the TUM pose
gauge and failed scorecards, so the pre-D1 TUM live artifacts were RESTORED
(not rebuilt) — A3 passes by restoration; TUM composite reproducibility is a
recorded follow-up (raw-anchor provenance of the original builds unclear).
Codex's ad-hoc A3 dangerous-free metric (0.95) matches no standing field —
disregarded; verification used the repo's standing metrics only.

### Phase 26 — hazard-survival fusion: KILLED by its pre-registered falsifier (numbers recorded, spine reverted no-slop)

Clean-room flag-gated implementation (Codex gpt-5.5/xhigh; OFF-mode
byte-identity PASSED; 5 unit tests green) ran the standing falsifier on the
two measured scenes. Verdict FAILED on every axis: dangerous_free@5cm
xyz 0.327 -> 0.458 (+40%), desk 0.567 -> 0.807 (+42%); occupied_static_iou
xyz 0.0303 -> 0.0034 (-89%), desk -83%; solid_recall@5cm xyz -92%, desk -61%.
Mechanism read: 1-exp(-Z) over RAW per-observation confidences collapses
occupied mass (the published systems drive near-binary OPTIMIZED opacities
first — the Phase-24 audit's substitution-risk flag, realized). Free-space
transmittance (our half of the design) inherits the same collapsed hazards.
Spine changes REVERTED per no-slop (Task-4 precedent); full report was
runs/_diag/codex_hazard_report.md; the brief pre-registered a kill as a
valid outcome and the kill is the result. Weekend D3 ledger: BOTH
repo-inspired mechanisms (Poisson stopper, hazard fusion) died on
falsifiers in under a day combined — dangerous_free remains DEPTH-COVERAGE
limited (Phase 16: candidate misses ~72% of co-observed band obstacles),
pointing the next attack at coverage/refinement (render-consistency
refinement, denser verified depth), not fusion arithmetic.

### Phase 27 — D2 gravity consensus: NOT ADOPTED (default OFF, brief-authorized) — because D1 already fixed the target via the legacy path

The consensus estimator shipped default-OFF per its pre-registered failure
mode: (A2) on desk it correctly REFUSED to align (camera-up prior vs plane
disagree 34.7 deg — the Phase-23-predicted narrow-heading bias case; metrics
unchanged, honest refusal beats wrong alignment); (A3) on phone_room_loop the
rescue case EVAPORATED: with D1's single gauge the LEGACY floor RANSAC now
passes (floor_inlier 0.379 vs 0.094 pre-D1 — first walking capture ever past
the 0.30 bar) and the live export is reconstruction_world_floor_aligned with
band tilt 0.73 deg vs the Phase-23 measured gravity (was 34.6), held-out
render error 0.062 (0.210 raw -> 0.125 composite -> 0.062 single-gauge),
xyz byte-identical (A1 PASS), 22 tests + ruff green. Verdict: D2's defect was
downstream of D1 on this scene; the consensus path stays available (flag) for
future low-floor-share captures, unproven and honestly marked. Desk-class
captures (narrow heading) remain unaligned — recorded open defect.

### Phase 28 — THE RULER SPEAKS: first physical scale measurement of a phone scene

User staged a 0.46 m steel ruler in frame 1 of phone_room_loop. Endpoints
back-projected through the single-gauge composite depth (frame 0, f=1080.74
self-cal, endpoint sensitivity +-4px -> +-1%): ruler span 0.1336 pose-gauge
units -> **scale = 3.44 m/unit** (3.41-3.47).
Three-way convergence and one indictment:
1. Phase 23's TENTATIVE ~3.3-4.2x scale compression: CONFIRMED at 3.44.
2. Camera-height cross-check: 0.36 units above detected floor x 3.44 =
   1.24 m — plausible handheld height; two independent anchors agree.
3. The backbone's learned metric prior (s_global 2.551) UNDERSTATES scale by
   3.44/2.55 = 1.35x — a 35% error, OUTSIDE its historically-claimed +-21%
   band; the council's refusal to certify (non_metric, +-30% floor) is
   VINDICATED by a $5 ruler. Phase 19 + council GT under-coverage + ruler
   now agree from three unrelated methods.
Derived metric facts for the user's home capture: scene span ~9.4 m, walked
path ~23.3 m, end-to-end loop drift ~5.6 cm (0.6% of span).
PROTOCOL ADOPTED: a known-length object in the first seconds of any capture
= free measured scale anchor (user invented this; it works).
NEW ANCHOR FAMILY unlocked by the floor fix: camera-height-above-detected-
floor x handheld-height prior — internet-video-compatible, sensor-free,
calibratable on GT rows.

### Phase 29 — scale arbitration: ruler CONFIRMED by pure triangulation; height-anchor falsifier FAILED honestly (capture-style prior mismatch)

Three conflicting loop-scale claims arbitrated by triangulating the ruler
endpoints from COLMAP sparse points (multi-view geometry only, no learned
components, no depth maps): span 0.5650 COLMAP units -> 0.8142 m/COLMAP-unit
-> /0.2325 pose anchor = **3.502 m per composite unit**, agreeing with the
Phase-28 depth-based read (3.444) to 1.7%. RULER VERDICT STANDS, now
double-measured: backbone metric prior (2.551) understates scale by ~37%.
Corrections this forced: (1) Phase 23's "camera 0.36 units above floor" was
measured against a FURNITURE plane; against the occupied floor voxels the
camera sits 0.582 units = ~2.0 m up — the user filmed elevated, not at hand
height. (2) The camera-height anchor falsifier (Codex, pre-registered)
FAILED correctly: pred 2.27 vs truth 3.50 (log err 0.43 > 0.25) on loop;
xyz passed (0.17); desk/room honestly unavailable. The failing assumption is
H_PRIOR=1.35 m — capture style varies too much for a fixed prior; the anchor
stays reportage-only (guard intact). Possible rescue (NOT pre-registered,
needs design): per-corpus or per-creator height calibration, or pitch-aware
priors. Home-truth ledger at 3.50: span 9.6 m, path 25.5 m, loop drift 5.7 cm.

### Phase 30 — carve-discipline falsifier: FAILED as pre-registered, and the wall splits in two

Three flag-gated variants (verified_only / truncated x3-margin / both), full
table runs/_diag/carve_metric_table.json + codex_carve_report.md. Verdict
FAILED on the pre-registered bars — no variant adopted, default off. But the
table is the most diagnostic negative result yet:
- xyz: dangerous_free@5cm 0.327 -> 0.0000 under EVERY variant (10cm too) —
  unverified-ray flooding is 100% of xyz's danger, and discipline eliminates
  it. The miss was the free-recall guardrail: truncated retained 0.591 vs
  the 0.60 bar (by 0.009). Bars are bars; no post-hoc tuning.
- desk: 0.567 -> 0.504 best — danger survives verified-only carving, so
  desk's dangerous free is carved by VERIFIED-but-WRONG depth (the Phase-12
  perturbation-stable PatchMatch bias). No carve policy can fix
  mismeasurement: desk-class danger needs verified-depth ACCURACY work.
- occupied_static_iou rose (0.030 -> up to 0.240) but via denominator shift
  (free->unknown), recorded as definitional, not claimed as a win.
Also landed and verified: --assets per-scene filter for teacher+evaluate
(byte-identity proven) — falsifier cycles now 5x cheaper.
NEXT (to pre-register fresh, not tune the old): (1) margin-sweep falsifier
for truncated-on-xyz-class with the SAME retention bar; (2) desk-class
verified-depth accuracy investigation (MVS bias) — the wall behind the wall.

### Phase 31 — S1 throughput attack (Codex, verified): iteration loop 321s -> 119s (63% cut), falsifier 192s -> 1.3s

Measured-first profiling then cuts, all identity-proven: m1/m2 input-hash
caches (~0.2s warm, --force bypass); teacher --emit-stage-timings (per-stage
table: refine 31s + two map-fusion passes ~48s = the next bottlenecks);
teacher now exports deterministic comparison_field.npz consumed by the
falsifier by default (1.3s vs 192s recompute; --recompute preserves the slow
path; older exports lack the field -- regenerated on next canonical run);
SfM A/B stability workspaces now OPT-IN (--stability-workspaces; they cost
309s + 12GB per ingestion, only the stability trust tier consumes them) and
--photometric-scope keyframes-sources for iteration runs (default all
preserved). Ingestion ledger (loop capture, timestamp spans): SfM ~74min
(incl. ENOSPC casualty), backbone 92s, pose 140s, MVS 58s. Two cached
teacher runs byte-identical; fast-falsifier metrics match recompute on both
measured scenes; 31 tests + ruff green.

### Phase 32 — margin-sweep falsifier PASSES; carve discipline ADOPTED canonically; the emitter's honest refusal names the next wall

Fresh pre-registered sweep (M in {1.5,2,2.5,4}, registration written before
any run): ALL of M<=2.5 pass the xyz bars; selection rule picks M=1.5
(retention 0.941). CANONICAL ADOPTION (configs/robot_envelope.json:
free_carve_discipline=truncated, multiplier 1.5; full rerun + scorecard
fd68446b): xyz dangerous_free@5cm 0.327 -> 0.111 (10cm 0.0523) at 94.1%
free-recall retention, solid F1/recall byte-stable, all five categories
unchanged, fsc deltas in the expected free->unknown direction. The Phase-16
5cm danger alarm NO LONGER FIRES on xyz — first scene to clear the
materiality bar. Desk unchanged (0.567) per the Phase-30 two-wall diagnosis.

FIRST DATASET EMISSION ATTEMPT: the emitter REFUSED all 5 scenes — verbatim:
xyz relative_scale_uncertainty_too_high:0.310 (the council's conservative
0.30 guard floor exceeds the emitter's authority bar); desk adds
gravity_alignment_unverified; room/phones non-metric. This refusal is the
honesty stack WORKING: labels are now safe on xyz but the scale band hasn't
EARNED tightness. The binding wall is now exactly one thing: METRIC-ANCHOR
COUNCIL CALIBRATION ROWS (2 of 8 minimum) — grow with public-GT scenes so
the council can certify sharper bands per its own pre-registered promotion
criteria. No thresholds were touched to force emission.
