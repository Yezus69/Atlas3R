# GT-Free Verification Stack — Design and Measured Evidence

Status: Stage 0 + Stage 1 LIVE in the acceptance gate (commit `a16985a`);
instruments (injection detection-limit harness, epipolar auditor) landed in
`d55e9e8`; everything below is measured on the canonical scenes, never
fabricated. Generated artifacts live under gitignored `runs/_diag/`
(`signal_calibration_table.json`, `detection_limit_reference_metric.json`,
`epipolar_audit_*.json`); regenerate with the commands in each section.

## Why this exists

The teacher's purpose is to pseudo-label internet-scale video, and internet
video has no ground truth. Tesla verifies its auto-labels with fleet-scale
recurrence; Matic iterates on-device with its own robots. Atlas3R has neither
— this verification stack is what stands in for the fleet, which is why its
honesty is the moat rather than a tax. The acceptance test for any GT-free quality signal
is fixed: **it must reproduce the GT-grounded verdicts on the measured scenes
where we can check.** The measured failure that forced this work: `phone_room`
was `accepted_for_metric_training=true` while carrying a 36.9° floor tilt, a
0.035 reprojection-inbounds ratio, and zero GT — the gate was blind to defects
the pipeline had already measured.

Three principles (ARCHITECTURE.md, GT-Free Acceptance Cascade):

1. **Independence** — auditors must use different evidence/algorithms than the
   builder optimized (the optimizer's own residuals are its homework).
2. **Evidence mass** — a consistency score over near-zero co-observation is
   vacuous, not reassuring; low evidence mass rejects regardless of scores.
3. **Measured authority** — a signal only counts where it has been shown it
   WOULD have detected an error (injection detection limits, noise floors).

## The cascade (live)

```text
Stage 0  evidence mass   (visibility stats, previously computed but UNREAD)
Stage 1  gravity         (floor up-alignment verdict, previously UNREAD)
Stage 2  consistency     (held-out render error, fsc, dynamic leakage)
META     detection limit (injected-corruption calibration, per scene)
```

Scope rule: the cascade gates ONLY paths without measured evidence. The
measured baseline is the yardstick, not the examinee. All failing stages
contribute reasons — no early-exit, every defect is named.

### Measured gate verdicts after wiring (teacher re-run, commit `a16985a`)

| scene | before | after | why |
|---|---|---|---|
| xyz (0.078 m RMSE) | accepted | **accepted** | passes both stages (inbounds 0.605, floor 0.622) |
| desk (0.278 m) | rejected (fsc 0.358) | **rejected** (fsc; stages pass) | its defect is consistency, not evidence — correct |
| room@13kf (0.823 m) | rejected by a held-out catch over **41 pixels** (luck) | **rejected for the real reasons**: inbounds 0.000, edge fraction 0.434, confidence 0.116, floor 0.252, held-out 0.573 | evidence collapse named |
| phone_room (36.9° tilt, no GT) | **falsely accepted** | **rejected**: inbounds 0.035, edge fraction 0.537, confidence 0.187, gravity inlier 0.127 | the gate now reads the blocker the fuser already raised |

All measured baselines unchanged (`measured_metric`, accepted).

## Calibration table (falsifier part 1)

`python tools/run_signal_calibration.py` — full candidate pipeline over every
cached backbone-artifact config; identical recipe per row; GT labels only
where a measured M2 reference exists.

| config | kf | backbone | inbounds | pre_p90 | held_out | fsc | floor | RMSE (m) |
|---|---|---|---|---|---|---|---|---|
| xyz_ma_14kf | 14 | mapanything | 0.605 | **0.126** | 0.063 | 0.231 | 0.622 | 0.0779 |
| xyz_da3 | 14 | da3 | 0.590 | **0.143** | 0.027 | 0.215 | 0.630 | 0.1048 |
| desk_da3 | 11 | da3 | 0.664 | **0.489** | 0.044 | 0.440 | 0.512 | 0.2534 |
| desk_ma_11kf | 11 | mapanything | 0.648 | **0.552** | 0.037 | 0.358 | 0.472 | 0.2784 |
| desk_ma_dense | 21 | mapanything | 0.633 | **0.591** | 0.098 | 0.359 | 0.392 | 0.3314 |
| room_ma_48kf_me | 48 | mapanything | 0.074 | **0.686** | 0.954 | 0.165 | 0.197 | 0.4571 |
| room_ma_13kf | 13 | mapanything | 0.000 | **0.833** | 0.573 | 0.123 | 0.252 | 0.8229 |
| phone_ma (no GT) | 32 | mapanything | 0.035 | 0.556 | 0.027 | 0.078 | 0.127 | — |
| phone_da3 (no GT) | 32 | da3 | 0.000 | 0.392 | 0.037 | 0.098 | 0.220 | — |

(`room_ma_24kf` / `room_ma_48kf` rows carry signals but their GT comparison
did not compute — recorded verbatim in the table JSON, never fabricated.)

### Findings (n=7 GT rows: a screen, not a calibration)

1. **`prerefine_p90_log_depth_residual` rank-orders all 7 GT configs
   perfectly** — Spearman **1.0** vs RMSE, and **1.0 after partialling out
   coverage** (the red-team's kill criterion). It generalizes across scenes
   AND backbones, and three of the seven configs (xyz_da3, desk_da3,
   desk_ma_dense) did not exist when the signal was first noticed, which
   answers much of the post-hoc-selection concern. The good/bad chasm is wide:
   all RMSE < 0.11 score ≤ 0.143; all RMSE > 0.25 score ≥ 0.489. Status:
   **candidate pending leave-one-scene-out validation — it gates nothing yet**
   and is surfaced in the scorecard as reportage.
2. **The confound prediction is confirmed**: after partialling out coverage,
   fsc (partial ρ = 0.08) and inbounds ratio (0.02) carry no accuracy
   information. They are an inconsistency catcher and an evidence-mass gate
   respectively — the cascade's division of labor, now measured.
3. Also strong after the coverage partial: floor_inlier_ratio (−0.92),
   mean_confidence_weight (−0.87), postrefine_p90 (0.87).
4. Stage 0 already rejects the danger-band row (room@48kf_me, 0.457 m:
   inbounds 0.074 < 0.30) — on the current grid the cascade has no false
   accepts and no false rejects.

## Injection detection limits (falsifier part 2; measured authority)

`python tools/run_detection_limit.py --asset reference_metric --seeds 3` —
57 injections over 6 corruption families applied to the FINISHED
reconstruction, NO refinement repair pass, magnitudes in scene-relative units
(gauge freedom grants no cm ruler without an anchor). Detection requires
response MONOTONICITY and a SOLID threshold crossing (beyond across-seed
spread) — both guards were added after the naive min-rule produced knife-edge
artifacts.

Certificate for xyz (`injected_corruption_detection_limit`):

| family | verdict |
|---|---|
| regional_depth_bias (out of refine span) | **detected, DL = 0.2 peak log-depth** — solid monotone fsc response 0.25→0.29→0.33 |
| scale_drift_ramp | **detected, DL = 2.0× end-to-end** — deterministic Stage-1 floor loss (the ramp warps the floor plane below RANSAC reliability) |
| pose_drift_translation | **no reliable detection at any tested magnitude (≤ 0.4 × trajectory span)** |
| pose_drift_rotation | **no reliable detection (≤ 10° end-to-end)** |
| ray_field_focal_bias (out of refine span) | no reliable detection (knife-edge noise only) |
| global_tilt_control (negative control) | correctly NOT detected by internal signals |

### Findings

1. **The internal-consistency suite is measurably blind to coherent pose
   drift.** Under a 0.4-span drift: inbounds stays ≥ 0.52, held-out unchanged
   (0.06), fsc fluctuates randomly around its threshold. Every apparent
   detection was the fsc knife edge (clean 0.244 vs gate 0.25 — margin
   0.006). A clean reading on the pose families is now reported as
   `no_authority`, never as evidence of correctness. This measured gap is the
   quantified case for the pose-accuracy roadmap (epipolar auditing,
   split-and-merge, SLAM-grade pose backend).
2. **The headline GT metric partially absorbs coherent drift.** True Sim(3)
   camera-center RMSE stayed ≈ 0.07 m under a 0.4-span injected drift — the
   similarity alignment soaks up trajectory-axis-aligned ramps. RMSE numbers
   under-state coherent drift; this affects how every past and future RMSE
   claim should be read.
3. **The pipeline is tilt-equivariant when the floor is reliable.** 15–37°
   rigid tilts: floor RANSAC re-found the tilted floor and re-aligned the band
   in 9/9 runs, signals bit-identical, zero solid internal responses. Tilt
   risk therefore concentrates exactly where the floor is unreliable — which
   is precisely what Stage 1 rejects. The negative control certifies, per
   scene, that tilt coverage rests on the gravity stage alone.
4. The certificate covers ONLY the tested corruption families; errors shaped
   unlike every tested family are explicitly out of scope, and detection
   limits transfer to no other scene.

## Independent epipolar auditor

`python -m atlas3r.epipolar_audit --asset <id>` — ORB + ratio test + 5-point
RANSAC + cheirality: a different algorithm class than the DUSt3R-lineage
backbone; rotation comparisons are scale-free. Intrinsics from MEASURED
calibration where it exists (TUM fr1 file; `intrinsics_source:
measured_calibration`); the backbone-intrinsics fallback is honestly demoted.
Wired into the teacher report as `epipolar_audit_status` — **reportage only,
it gates nothing** until detection-limit calibration assigns it authority.

Final teacher-run results (claimed-overlap pair selection, v2):

| scene | result |
|---|---|
| xyz | audited: 23 valid pairs, rotation deviation median 6.8° / p90 16.2° vs auditor cycle-residual noise floor median 8.7° / p90 19.2° → **below_auditor_noise_floor_no_authority** (honest: cannot certify the fine scale) |
| desk | abstained (6 valid pairs under BOTH blind and claimed-overlap selection — motion blur genuinely starves ORB exactly where the backbone struggles) |
| room@13kf | abstained (6 valid pairs) |
| room@48kf | abstained (1 valid pair blind → 9 under claimed-overlap selection, still below the 15-pair floor; the danger-band verdict correctly rests on Stage 0, which rejects room@48 at inbounds 0.074) |
| phone_room | **audited and DEVIATING: 19 valid pairs, rotation deviation median 15.0° / p90 21.2° vs noise floor 2.7° → above_auditor_noise_floor** — the independent auditor corroborates the gate's rejection of the production-class input from a completely different evidence family (caveats recorded in-band: noise floor from 1 triangle; backbone intrinsics → authority demoted) |

Findings: the auditor never false-vouches; abstention is recorded as
**authority loss, never a pass** (low-texture/blur correlates with backbone
failure). Its own cycle residual is its self-reported noise floor — ORB-class
two-view geometry on 640×480 frames cannot certify fine-grained accuracy, only
gross rotation error. Pair selection audits the visibility graph's CLAIMED
overlap edges (test the reconstruction's own assertions) after blind
wide-baseline selection starved on loop trajectories (phone_room: 2 → 19
valid pairs under the v2 selection, which is what enabled the catch above).

## Plane ledger (rigid-world drift audit)

`python -m atlas3r.plane_ledger --asset <id>` — first-principles instrument
for the measured blind spot above: pairwise consistency cannot see coherent
drift (drift moves each camera and its attached depth together), but drift
cannot preserve the WORLD-FRAME CONSTANCY of revisited rigid structure. Per
frame, RANSAC planes from the backbone depth (no photometrics — immune to the
blur that starves ORB); transform through the poses under audit; chain into
plane TRACKS; audit each track for in-track trends. Offset trend → translation
drift; normal rotation → rotation drift; |offset| ramp → scale drift. The
ledger reports its own noise floor, its evidence mass, and its DIRECTION
authority (conditioning of the tracked-normal span — a track is blind to
translation perpendicular to its normal; the report names the blind axis).

Measured verdicts (xyz injections, 3 seeds, monotone + beyond-seed-noise
required; GT calibration table as PURE blind test — thresholds touched no GT):

| gauge | injection response (xyz) | blind GT test (7 configs, 3 scenes × 2 backbones) | verdict |
|---|---|---|---|
| scale ramp (`ledger_scale_ramp_p90_abs_log_ratio`) | monotone solid: rate gauge 0.0445 → 0.0744 (@1.5×) → 0.132 (@2.0×); responds BELOW the gate's prior 2.0× detection limit | **Spearman 0.93 vs RMSE, 0.92 after partialling out coverage** (good scenes ~0.10; room@13kf 1.575 — its true failure was 3.5× scale wander) | **authority earned — gate-promotion candidate** |
| translation drift (offset trend/rate) | non-monotone, swamped by per-track noise (clean p90 0.231 span; ~14 keyframes × 5-frame tracks give no leverage); dominant-vs-blind-axis probes show no separation | negative partial correlation | **no_authority — honestly recorded** |
| rotation drift (normal p90) | flat under ≤10° end-to-end (within-track share below the ~8° normal noise) | partial 0.83 (suggestive, uncalibrated) | no_authority from injections; cross-scene signal noted, not claimed |
| tilt negative control | **bit-identical under 15–37° rigid tilt** — planes rotate with the world; gauge-invariant exactly as theory requires | — | clean negative control |

Diagnosis for the failed gauges: track leverage, not concept — 14 keyframes
with 4–8-frame tracks cannot resolve a ramp against plane-fit noise. The fix
is denser, overlap-aware keyframing (roadmap #1), which lengthens tracks; the
direction-resolved authority machinery is already in place for that day.
Honesty caveat stated in-band everywhere: the ledger shares the backbone's
depth; its audit axis is temporal coherence through the poses, and its
authority is the measured response above, never assumed.

Prior-art check (2-agent sweep, 2026-06-10): plane-SLAM uses planes as
optimization constraints; double-wall detection and GT-free map posteriors
exist as global scores. The ledger-as-instrument combination — plane tracks
audited post-hoc with per-DOF gauges, evidence-mass abstention, and
injection-calibrated, direction-resolved authority — was not found published.

## What this stack can NEVER certify

1. **Absolute metric scale.** Monocular gauge freedom: every signal verifies
   shape and trajectory up to a similarity. Detection limits are reported in
   scene-relative units on purpose; a "10 cm" claim on a reconstruction whose
   scale is 3.5× off (room, measured) would fabricate a ruler.
2. **Global tilt from geometry alone** — covered solely by the gravity stage
   (measured, see negative control), which means: no reliable floor, no
   accepted band.
3. **Error shapes outside the tested corruption families.**
4. **Transfer of any threshold or limit to other scenes/domains** — Stage 0
   thresholds sit mid-chasm on the measured clusters and carry
   `provisional_mid_chasm_pending_detection_limit_calibration` authority;
   per-scene certificates exist precisely because transfer is not assumed.

## Next steps (in evidence order)

1. Leave-one-scene-out validation of `prerefine_p90_log_depth_residual` →
   promotion to a gated Stage-2b severity threshold (provisional mid-chasm
   value ≈ 0.35 sits in the measured 0.143–0.489 gap).
2. Run the detection-limit harness on desk/room/phone (per-scene certificates;
   xyz only so far) and add the visibility `median_log_depth_residual` to the
   harness suite to test whether residual-class signals see coherent drift.
3. Close the measured coherent-drift blindness: the epipolar auditor with
   claimed-overlap selection + a drift-sensitive comparison (loop-composition
   over the visibility graph), and/or the pose-backend roadmap (COLMAP/GLOMAP
   oracle first, DROID-SLAM after license verification).
4. Wire the per-scene `injected_corruption_detection_limit` block into the
   teacher report/scorecard once per-scene runs exist (today it lives in
   `runs/_diag/`).

Production data point (2026-06-10): the gate's verdict on `phone_room` under
the full v2 production recipe — self-calibrated COLMAP poses + multi-view
verified depth — moved from 4 rejection reasons to 2 (Stage-0 inbounds and
edge-fraction now PASS; confidence weight misses by 0.005; Stage-1 floor
support genuinely 0.082, prior-assisted search included). Full record with
every stage's numbers: `docs/band_obstacle_recall_evidence.md` Phase 10.
The Stage-1 path gained a candidate-only camera-up PRIOR this session: it
redirects the floor RANSAC search when the dominant-plane search is
unreliable, under the recorded handheld-upright assumption — the acceptance
bar (0.30 inlier) is unchanged, so it cannot weaken the gate (measured:
canonical scorecard diff empty; on canonical phone the prior fires, reads
0.123 < 0.30, and the refusal stands).

## Vault protocol (anti-overfitting safeguard)

The gravest failure mode for a fleet-less teacher is silent overfitting to the
scenes it is developed against: every threshold and policy in this repo is
exercised on the fr1 gate scenes, and a teacher accidentally specialized to
them would be wrong on no-GT internet video with no warning. The structural
safeguard is a SEALED VAULT SCENE: `reference_metric_vault` = TUM
freiburg3_long_office_household, a DIFFERENT camera (fr3 calibration) than
every gate scene, registered only in `config/vault_assets.json` (never in the
canonical manifest or `CANONICAL_TRACKS`). Rules, binding: vault runs happen
only at declared milestones via `--manifest config/vault_assets.json`; every
run is recorded with its commit hash; nothing is ever tuned in response to a
vault result -- a vault regression is an overfitting ALARM about the
development process, not a bug to fix on the vault scene; a vault scene used
for tuning is burned and must be replaced.
