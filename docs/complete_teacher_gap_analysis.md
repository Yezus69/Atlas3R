# The Complete-Teacher Gap Analysis (2026-06-11, at v3-canonical, HEAD e71fa02)

Question: what stands between today's teacher and one we trust enough to train the
student on its accepted dataset and put that student on a real robot?

Method: multi-agent evidence sweep (6 code-level audits + 4 independent candidate
lenses + merge + adversarial attack), every load-bearing number re-verified against
the repo at HEAD. End-state requirements judged against: (1) calibrated per-voxel
confidence, (2) honest per-sample scale uncertainty, (3) indoor AND outdoor
generalization, (4) gate thresholds with measured (not provisional) authority.

The falsifiers below are SKETCHES. Each campaign pre-registers its exact bars,
single-shot mechanics, and terminal conditions at execution time, per repo law.
Nothing here moves a threshold or re-litigates a documented rejection.

## Corrections to the working state (verified this analysis; supersede prior wording)

- **"desk is one floor improvement away (0.268 vs 0.30)" UNDERSTATES.** The
  0.268-inlier plane is tilted 41.63° from the band axis — not floor-plausible
  (likely the desk surface); the camera-up-constrained floor search reads 0.162
  (`runs/teacher/reference_metric_desk_teacher_report.json` up_alignment
  constrained_inlier_ratio). The gap is estimator-shaped (~1.85×), not a nudge.
- **A discriminating per-scene scale signal already exists and is dead.**
  `pose_provenance.scale_alignment.residual_rmse_backbone_units` in
  `external/_composite_v3/*/poses.json` ranks xyz 0.0271 (accepted) < desk 0.0559
  < phone_room 0.4576 < room 0.9132 (rejected); nothing in src/ consumes it.
- **scale_std 0.168 is a per-BACKBONE constant**, byte-identical across all four
  tracks (0.12·(1+(1−0.6)) from MapAnything's hardcoded 0.6 confidence), not
  per-scene uncertainty. scale.py hardwires scale_mean=1.0 in all four branches.
- **Two live gate bars carry NO authority marker at all** (a third class beyond
  provisional/measured): MAX_HELD_OUT_ERROR_FOR_ACCEPT=0.50 (validation.py:34),
  MAX_DYNAMIC_LEAKAGE_FOR_ACCEPT=0.10 (validation.py:39). The Stage-1 numeric bar
  (0.30) lives at mapping.py:80, outside validation.py.
- **The class-A p90 bound (0.66 = 0.548×1.2) is anchored on room's own p90 — a
  Stage-0-REJECTED scene sits in the "good cluster" that set the bar.**
- **room's loop-closed sparse model exists unpropagated**
  (`runs/_diag/colmap_work/room_loopdet`, 2026-06-11) — never carried through
  MVS + teacher; the canonical room track still traces to room_dense. Caveat:
  room's MVS verified-pixel fraction is 0.038, so Stage-0 may fail even with
  loop-closed poses (unmeasured either way; the re-run is informative regardless).
- **The export gap is narrower than "does not exist":** voxel_occupancy_3d.npz
  already carries 6 of the 9 TrainingSample spec fields; missing are the per-frame
  side (RGB path-join, intrinsics serialization, packaged T_camera_to_grid),
  license/URI provenance (zero license fields anywhere in code), and the
  cross-scene manifest. Packaging, not new geometry.
- **2D and 3D grids ship DIFFERENT semantics under the same names**: map_confidence
  (per-voxel array vs scene scalar) and scale_uncertainty (relative_scale_uncertainty
  vs scale_std — equal today only because scale_mean==1.0). Fix before either field
  carries a real measurement.
- **Per-pixel evidence is computed then discarded**: packet.confidence is contract-
  validated and never read by fusion (every sample fuses at +1.0); the continuous
  stability residual |log dA − log dB| is binarized at tools/run_mvs_depth_backend.py:264;
  per-voxel class-split hit/free/verified counts are dropped at export. No
  calibration can run from exported artifacts today.
- **phone_room is provenance-classed class-A (ba_grade_frozen) despite being the
  fragmented capture** — the class marker keys on the pose-method string, not on
  registration completeness.

## THE FOUR MAJOR WORKS

### 1. Calibrated trust channels — per-voxel confidence + per-sample scale honesty

**What.** Replace the two numbers a student would weight its loss by — both
heuristic placeholders today — with measured quantities, under the repo's existing
fit-on-designated-scenes / freeze-once / apply-scene-blind pattern.
(a) Per-voxel confidence: persist the discarded evidence (class-split hit/free/
verified counts, packet.confidence, continuous stability residual) through fusion
and export; bin voxels by evidence via the existing GT-join (`_band3d_agreement`);
fit a reliability-measured channel meaning P(label correct at the 5 cm margin);
freeze; validate out-of-sample. (b) Scale: wire the dead residual_rmse signal
(+ refine residuals, parallax conditioning) into a data-driven per-scene scale_std;
build the first anchor producer — camera-height-above-floor on Stage-1-passing
scenes (floor plane + camera centers already exist; one dot product + height
prior; candidate-only, provenance recorded). Anchorless web samples stay honestly
non_metric_pseudo_label. Fix both 2D/3D field-semantics divergences first.

**Evidence it is load-bearing.** map_confidence = hardcoded status-step ×
one-hit-saturating evidence (mapping.py:55, :1279–1281) — an accepted
metric_pseudo_label scene can only express ~0.3–0.6; zero reliability validation
anywhere; the channel is blind to the verified tier (measured 2.2× accuracy) and
to stability (measured 1.6–1.7× single-vs-both-witness separation; keeping the
stable ~15–20% moved F1@5cm 0.065→0.230/0.382). scale_std is a per-backbone
constant; a 1.34× scale error collapsed honest F1@5cm 0.199→0.046 (Phase 13) —
scale error is fatal at the margin and the current field cannot see it.

**Done.** Exported per-voxel confidence with monotone reliability and ECE/Brier
under a pre-registered bound on fit scenes AND a scene the fit never saw; verified
+ stability evidence demonstrably feeding the channel; per-scene scale_std
data-driven and ranking xyz<desk<phone<room (the dead residuals already do);
exported scale_uncertainty interval covering GT-measured scale at a pre-registered
rate; anchor ScaleEvidence with recorded provenance; both channels installed as the
TrainingSample downweight by absolute calibrated threshold (never per-clip quantile).

**Falsifier.** (a) LOSO reliability: if top-decile-confidence voxels are not
measurably more GT-correct than bottom-decile on the held-out scene, the continuous
channel is refuted — honest fallback is ordinal trust tiers (stable-verified /
verified / unverified) whose per-tier ratios are already measured. (b) Scale
coverage test, runnable TODAY on the existing 12-config population: claimed ±σ
failing to contain true Umeyama-vs-GT scale at the stated rate refutes the channel
and demotes those classes to non_metric. (c) Injected 1.5–2.0× scale ramp must
widen claimed uncertainty or trip the gate (plane-ledger scale-ramp gauge has
earned authority, Spearman 0.93 blind); a tight σ surviving the ramp is refuted.

**Constraint.** With n=1 accepted scene, any fit has single-scene authority — the
persistence half lands NOW (before mass scene processing), the fit half is gated on
Work 3 delivering ≥3 accepted scenes.

### 2. Gate-authority completion — certificates beyond xyz, drift blindness, full enumeration

**What.** (a) Run the detection-limit harness on desk/room/phone AND the class-A
xyz-v3 asset with frozen-BA-shaped corruption families; wire the
injected_corruption_detection_limit block into the teacher report so every accepted
label names the families its gate provably catches and the families where it has
no_authority; retire both provisional strings. (b) Aim instruments at the certified
blind spot (coherent pose drift ≤0.4 span, rotation ≤10°, focal bias — all DL=null):
loop-composition epipolar audit over the visibility graph; ledger pose gauges given
track leverage via denser keyframing (the measured failure was leverage, not
concept). (c) Sweep the complete threshold enumeration into the calibration
program: the two UNMARKED bars (held-out 0.50, leakage 0.10) and the Stage-1 bar
living outside validation.py. At web scale, authority transfers per provenance
CLASS (thresholds) + per-scene certificates where affordable — never assumed.

**Evidence.** Exactly one certificate on disk (xyz), unwired (grep detection_limit
in src/ hits only the authority string at validation.py:354). Class-A ×1.2 bounds
provisional (validation.py:63–70, :249), anchored partly on a Stage-0-rejected
scene's p90. The blindness is measured and production-relevant: the vault's actual
failure mode WAS drift (0.83–1.41 m), caught only because gross; phone_room was
once falsely accepted with a 36.9° tilt — that is what a poisoned student sample
looks like, and there is no fleet to catch the next one. The class-B
regional-depth-bias DL=0.2 rests on a 0.006 fsc knife edge.

**Done.** Per-scene certificates for all gate scenes + the accepted class-A asset,
in-band in every teacher report; both provisional strings replaced by measured
citations; a measured DL for coherent drift well below the vault failure magnitude
OR the blindness honestly carried in-band on every exported sample; all bars
calibrated or explicitly authority-tagged; empty categorical scorecard diff on
clean scenes (any flip = authority bought by moving a bar).

**Falsifier (center of gravity, runnable today).** Inject 0.2–0.4-span coherent
drift into the frozen accepted xyz-v3 reconstruction; re-run the upgraded gate.
Monotone beyond-seed detection ⇒ blindness closed, ×1.2 margin dies. Still blind ⇒
the verdict stands MEASURED and accepted samples carry the no_authority family in
TrainingSample provenance — an honest fail, not papered over. Second curve: the
magnitude where class-A signals fire vs where honest F1@5cm degrades past the
margin — a family that damages labels while both signals stay under threshold
REFUTES the ×1.2 bounds and re-derives or demotes class-A acceptance.

### 3. Accepted-corpus yield — floor estimator, loop-closure propagation, fragment-native ingestion

**What.** Convert measured rejection walls into accepted scenes by ADDING EVIDENCE,
zero threshold moves. (a) Floor ESTIMATOR: find the true floor under the camera-up
prior (desk needs 0.162→0.30 on the actual floor, not a dominant-plane nudge) —
the floor stage names a blocker in ALL FIVE current rejections and also gates the
camera-height scale anchor (Work 1) and outdoor gravity authority (Work 4).
(b) Propagate the staged loop-closure recipe: rebuild room from room_loopdet
through MVS + teacher; re-run the sealed vault ONCE at its declared milestone.
(c) Fragment-native ingestion: per-fragment SfM workspaces, teacher runs, and gate
verdicts — starting with phone_room fragment B (130 images, the floor-rich coverage
phone's own rejection names; today dropped entirely).

**Evidence.** Yield today: 1 accepted track of 4 canonical + 0 of 2 vault (~17–25%)
— cannot train a generalizing student, and gives Works 1–2 single-scene authority.
Floor blocker values: desk 0.268-tilted/0.162-true (SOLE blocker), room 0.095,
phone 0.104, vault 0.156/0.149, all vs 0.30. Loop closure measured at exhaustive
parity (0.1399 vs 0.1480 m, 5.1 vs ~37 min). phone inbounds 0.297 is 0.003 under
its bar — the rejection is really floor + p90.

**Done.** ≥2 of 4 canonical scenes accepted via evidence improvements with an empty
categorical diff on already-accepted scenes and zero threshold moves; vault Run 2
executed once at a declared milestone with no alarm fired; run_sfm_pipeline.py
emits N gated fragment workspaces per video with per-fragment yield accounting;
rejection-reason histogram carries no pose-drift class, only evidence-class reasons.

**Falsifier.** Three pre-registered single-shots: improved floor estimator must
find a floor-plausible (low-tilt) plane ≥0.30 on desk AND the scene must pass the
untouched remaining gates; room-from-loopdet — if Stage-0 still fails with
loop-closed poses (verified fraction 0.038 says it may), room's rejection is
evidence-genuine and the long-loop yield estimate is revised down (informative, not
failure); phone fragment B end-to-end — floor still ~0.10 with full coverage means
phone's rejection is evidence-genuine. Vault: drift >0.5 m means loop closure does
not generalize off fr1_room; an accept with worse measured placement than
fr1-rejected configs fires the documented gate-overfit alarm.

### 4. Outdoor/mower domain entry — honesty marking first, lawn calibration scenes, then representation

**What.** Sequenced by hazard, not ambition. (1) Frame-export step (smaller than
billed: m1 already decodes video; no imwrite exists in the repo) feeding
run_sfm_pipeline.py; the EXTERNAL_CANDIDATE contract slot already exists.
(2) Explicit domain/no_authority marking in the gate path BEFORE the first lawn
scene is scored — load-bearing because the dangerous outdoor failure is SILENT
MIS-ACCEPTANCE: a smooth uniform slope can pass the gravity gate slope-aligned
(RANSAC accepts any dominant plane ≥0.30 inside the 30° camera-up cone; nothing
distinguishes slope-aligned from gravity-aligned; the injection harness measured
15–37° rigid tilts as bit-identical to every internal signal), and every threshold
an outdoor scene would meet was calibrated exclusively indoors with no domain flag
in code. (3) ≥2 self-captured lawn calibration scenes (flat + independently
measured slope) through the FROZEN v3 pipeline → outdoor gravity authority gets a
number. (4) Only then the genuinely large work: mower envelope semantics — the band
is one planar slab above a single global floor height; terrain-following
(height-above-local-ground) is a mapping.py representation change, plus a mower
envelope file with its own pre-registered physics derivation (lawn obstacle class,
not chair legs). Backbone outdoor quality is an open empirical question the clips
answer, NOT a swap decision.

**Evidence.** True zero in code (grep outdoor|lawn|mower|grass over src/, tools/,
configs/ → no matches); zero outdoor scenes; the backbone survey contains zero
outdoor evidence for any candidate; mowers are half the business. Envelope override
mechanism already exists (a mower config FILE costs nothing); what no field can
express is terrain-following and traversable vegetation.

**Done.** ≥2 lawn clips flow the full v3 pipeline as external_candidate with honest
gate verdicts; gravity verdicts match measured ground truth at recorded inlier
ratios — or no_authority recorded with the measured reason; domain marker live
before any outdoor scene is scored; outdoor verified-pixel fraction and rejection
reasons reported, not assumed.

**Falsifier.** Two-sided slope test on a lawn clip with a measured reference
(inclinometer/RTK). Gate ACCEPTS a slope-aligned band while Stage 1 reports a
confident "floor" ⇒ the indoor gravity design is REFUTED for outdoor and is
redesigned before any lawn label trains the student. Everything rejects at
Stage 0/1 with no calibration path ⇒ the outdoor yield wall is measured and the
IMU-gravity-anchor becomes the named blocker. Expected Stage-1 rejections on
undulating lawns count as success.

## Two items deliberately NOT on the majors list

**Run FIRST — the trainability/collision-safety verdict (cheap, decision-bearing).**
Is honest F1@5cm 0.358 / 5.6 cm median / solid_recall@5cm 0.455 (54.5% of
measured solids have no solid label within the robot's own margin) trainable signal
or a bias the student will learn? Phase 12 measured the residual edge error as a
perturbation-STABLE foreground-fattening bias both disjoint halves reproduce;
whether a net averages it out or distills it is measured nowhere. Pre-registered
experiment: signed surface-displacement decomposition (bias vs variance,
edge vs interior) + consumption-semantics split (labeled-free over measured-solid =
collision-shaping vs unknown = survivable coverage loss — the danger-direction rate
is currently unmeasured at the gated level) + a small proxy occupancy net distilled
from accepted labels only (GT eval-only) with planner replay counting virtual
collisions. Outcome A (variance, proxy ≥0.358 with shrunken signed edge error):
trainable at the margin today — provisional on n=1 scene, strengthened as Work 3
raises yield. Outcome B (proxy reproduces the signed fattening): 0.358 is
bias-limited, and a FIFTH campaign — placement (backbone-class; AMB3R oracle bound
is the documented lever) — re-opens BEFORE any dataset scale-out. This experiment
orders everything else; it is not itself a campaign.

**Land EARLY — TrainingSample/DatasetManifest contract + emitter (enabler, not major).**
Spec-first in ARCHITECTURE.md API Contracts; emitter is mostly a JOIN of artifacts
that exist (npz already carries 6/9 fields; RGB is a path-join; intrinsics are in
memory; T_camera_to_grid is a two-file join). Genuinely new: license/URI provenance
(zero license fields in code; one NC source poisons the sellable dataset) and the
cross-scene aggregator (vacuous over today's 1-scene corpus — defer until Work 3
yields). MANDATORY honesty markers: calibration-status fields in-band
(confidence_calibration: 'uncalibrated_heuristic', scale_status mirroring the
constant) so early exports cannot masquerade as carrying the channels Work 1 has
not delivered. Its mutation falsifiers (train-from-manifest-alone, hash catches
corruption, unknown→free rejected, rejected-scene refusal, NC injection rejected)
fold into the trainability rig as its delivery precondition.

## Sequencing (dependency-driven, not preference)

1. Trainability falsifier (days) — its outcome can add the fifth campaign.
2. Export contract + evidence-persistence half of Work 1 (days) — BEFORE mass scene
   processing, so the corpus ships exportable and calibration-ready.
3. Work 3 floor estimator — the single highest-leverage item in the repo (blocker
   in all five rejections; gates Work 1's anchor and Work 4's gravity authority).
4. Work 2's xyz-v3 drift falsifier + F1-vs-signal curve — runnable today, in
   parallel; per-scene certificates scale out as Work 3 raises yield.
5. Work 4 cheap layers (frame export, domain marker, first clips) — immediately, in
   parallel; representation work only after first lawn evidence.
6. Work 1 fit half — gated on ≥3 accepted scenes; vault read-only at declared
   milestones only.
