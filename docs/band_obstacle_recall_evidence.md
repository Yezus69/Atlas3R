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

## Phase 5 — third gate scene: a full room-LOOP breaks the backbone even harder

Per the difficulty-spread plan, a **third measured scene — TUM freiburg1_room** (a full
room *loop*, 1362 frames, the canonical floor-robot motion) was staged as
`reference_metric_room` and wired into the canonical gate (`config/canonical_assets.json`
+ `evaluate.CANONICAL_TRACKS`). M1 selected 8 keyframes; the MapAnything backbone was run
over those 8 + 5 bookend/gap-filler frames (13 total), mirroring the desk recipe. The
measured GT field is **healthy** (6294 occupied voxels, occupied_fraction 1.06%, free
9.5%) — so what follows is a genuine *candidate* failure, not a measured/units bug.

| scene (motion) | cam Sim(3) RMSE | est. metric scale | occ_iou | coverage_of_measured_band | verdict |
|---|---|---|---|---|---|
| reference_metric — xyz (gentle jitter) | **0.078 m** | ~1.0 | **0.128** | 0.993 | passes |
| reference_metric_desk — desk-orbit (hard) | 0.278 m | ~1.0 | 0.0 | 0.897 | fails |
| **reference_metric_room — full loop (hardest)** | **0.823 m** | **0.283** | 0.0 | **0.019** | fails harder |

**On the room loop the feed-forward backbone blows up on *two* axes at once:** camera
Sim(3) RMSE **0.82 m** (median 0.72, max 1.39) **and metric scale 0.283** — i.e. the
candidate reconstruction is ~3.5× too large *and* badly mis-posed, so after Sim(3)
alignment it overlaps only **1.9 %** of the measured band. (Note the *band* `band_fsc`
reads 0.0 here only because there is almost no overlap left to contradict — read it
together with `coverage_of_measured_band: 0.019`, never alone.) Pushing to a denser,
cadence-matched 24-frame set made it **worse**, not better — the comparison degraded to
`insufficient_overlap_for_sim3_band_comparison` (can't even fit a Sim(3)). So this is the
*loop*, not under-sampling: the feed-forward backbone has no loop-closure, so a trajectory
that returns near its start accumulates unbounded pose+scale drift.

**Why keep a scene the pipeline fails (twice now)?** Because the gate's job is to tell the
truth, and the truth has a clear gradient: **gentle jitter works → desk-orbit fails →
room-loop fails harder.** Two distinct realistic failure modes (orbit + loop) prove the
collapse is not desk-specific, and loop-closure is *exactly* the motion a floor-cleaning
robot produces. The candidate is correctly emitted only as `metric_pseudo_label` with
`accepted_for_metric_training: False`, so nothing garbage is shipped as `measured_metric`.

**Gate-design gap this exposes (next-step, not yet fixed):** the *acceptance* gate keys on
the candidate map's **internal** free/occupied conflict (`map.free_space_contradiction_rate`
= 0.123 for room — comfortably below the 0.358 reject threshold), which a self-consistent
*but globally wrong* reconstruction can pass. The signals that actually catch the room
failure — `camera_center_sim3_error` (0.82 m) and `band3d_agreement.coverage` (0.019) —
are computed only on gate scenes (they need measured GT) and are currently **reportage,
not blocking**. In production (phone video, no GT) we cannot compute them, so the pipeline
*cannot self-detect this failure yet*. That is the real frontier: either (a) a backbone
with loop-closure / global consistency (the pose axis), or (b) a GT-free internal
consistency signal that correlates with cam-RMSE so the honesty gate can reject loop
blow-ups without measured GT. Tracked in [[sota-backbone-direction]].
