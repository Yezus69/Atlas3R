# Atlas3R Backbone Decision Report — Closing the ~12 cm Indoor Monocular Surface-Displacement Gap

**Audience:** Atlas3R maintainer. **Question answered:** which SOTA (2024–2026) model(s) do we plug in to cut the measured ~12 cm indoor monocular surface displacement and raise collision-band obstacle accuracy — without reinventing the wheel or vendoring?

**Verification status:** Web access worked. The four highest-stakes claims and the in-code honesty seam (`src/atlas3r/geometry_adapter.py:309-343`) were independently re-checked. Two survey numbers were wrong and are corrected below (MapAnything Apache AbsRel; GLORIE-SLAM ScanNet depth-L1). One survey claim (MASt3R-SLAM "2.25 cm") was misattributed and is corrected to ~8.7 cm.

> Provenance: produced by a multi-agent research workflow (4 domain researchers over live web → 12 per-model fit assessments → synthesis), anchored on the measured bottleneck in `docs/band_obstacle_recall_evidence.md`. Every "metric" claim below is a SOFT prior under Atlas3R's invariants — only TUM RGB-D GT is `measured_metric`, and it stays eval-only.

---

## 0. Refresh addendum — 2026-06-09 (3-agent currency re-verification, live web)

The conclusions below were re-verified against primary sources today. **The ranking did
not change** (MapAnything-apache #1 commercial-clean; Pi3/MoGe-2 unchanged; driving-occupancy
rejected; indoor SSC = second-phase). Material updates:

- **Two corrections to Section 5 (verified from the installed MapAnything source):**
  1. **Install is an editable SOURCE install, NOT `pip install mapanything`** (no PyPI
     package): `git clone https://github.com/facebookresearch/map-anything.git && cd
     map-anything && pip install -e .` (Python 3.12 conda env recommended; MapAnything assumes
     a CUDA/Linux env — on Windows run it under WSL/Linux).
  2. **MapAnything poses are OpenCV cam2world (= `T_world_camera`) → the adapter must NOT
     invert** (DA3 *did* invert w2c→c2w; do not copy that step). `depth_z` IS optical_z
     (matches `_source_depth_to_radial`). Verified `infer` signature + output keys are in
     `tools/run_mapanything_backbone.py` (built this session). MapAnything is now v1.1.2
     (2026-05-30); Apache checkpoint (4.91 GB) live, identical API.
- **NEW model — AMB3R (CVPR 2026, arXiv 2511.20343):** the strongest indoor metric geometry
  found, and the first with numbers **below the 12 cm gap** — ScanNet multi-view AbsRel **1.9**
  (vs MapAnything 4.0, VGGT 2.3), 7-Scenes accuracy **1.74 cm**, TUM SLAM ATE **2.7 cm**. BUT
  it is **NOT commercial-clean** (no repo LICENSE + freezes the non-commercial VGGT front-end;
  Google-Drive weights) and emits **pointmaps** (needs a Pi3-style adapter, not the depth→ray
  drop-in). → **teacher/eval-only ORACLE.** Strategic use: run AMB3R as a non-commercial
  upper-bound to answer *"is the 12 cm band gap even closable by a better backbone, and is the
  commercial-clean MapAnything-apache good enough — or is an AMB3R-class backbone required,
  which would be a commercial blocker to escalate?"* This is now the most valuable second
  experiment after the MapAnything A/B.
- **Occupancy line moved but verdict holds (second-phase recall only):** new **SGR-OCC**
  (arXiv 2603.14076, Mar 2026) is the indoor SSC SOTA (Occ-ScanNet 58.55 IoU / 49.89 mIoU) but
  its repo is still a **placeholder (LICENSE+README only, no code/weights)**. **EmbodiedOcc++**
  code IS released (license unstated → treat NC). New entries **AdaSFormer** + **LegoOcc**
  (Apache-2.0, open-vocab). **Every one is frozen on Depth-Anything-V2 and reports only
  IoU/mIoU — none reports cm/m surface error, so none escapes the 12 cm ceiling.** No
  Depth-Anything-4, no MoGe-3.
- **Honesty constant:** benchmark AbsRel does NOT predict band recall (DA3 scores well on
  ScanNet AbsRel yet misses 72% of band obstacles) — the `band3d_agreement` scorecard A/B is
  the only honest arbiter, and the Apache checkpoint's isolated indoor number is unpublished.

---

## 0b. PILOT RESULT — MapAnything-apache RUN and MEASURED (2026-06-09)

The pilot is no longer hypothetical: `tools/run_mapanything_backbone.py` was built,
MapAnything-apache (v1.1.2, Apache-2.0) was installed in an isolated env
(`external/mapanything_env`, source install) and **ran end-to-end on Windows/CUDA**
over the SAME 14 `reference_metric` keyframes DA3 used (metric depth 0.8–4.1 m;
1 forward pass for depth+pose+intrinsics+conf). Measured head-to-head on the
`band3d_agreement` scorecard (same fusion + measured-GT, gravity-support policy
re-tuned for the denser geometry: `free_carve_margin_m=0.10, occupancy_support_height_m=0.15,
occupancy_support_min_count=2`):

| metric | DA3 baseline | MapAnything-apache | Δ |
|---|---|---|---|
| `occupied_static_iou` | 0.0777 | **0.1216** | **+57%** |
| `coverage_of_measured_band` | 0.7157 | **0.9939** | +39% |
| `per_class_agreement` | 0.8681 | **0.9201** | +6% |
| free-space precision | 0.9677 | **0.9695** | + |
| camera-center Sim(3) RMSE | 0.1048 m | **0.0779 m** | −26% |
| band `free_space_contradiction_rate` | 0.7202 | 0.7211 | flat |

**This is a measured win on the adopt criterion (occ_iou up, per_class held/up) and on
camera RMSE, holding free-space precision and band_fsc.** Notes: (1) the directional
free-carve truncation lever that *regressed* per_class on DA3's sparse geometry now
*helps* on MapAnything's dense geometry — backbone swap + policy levers compose;
(2) band_fsc is flat, not down — MapAnything covers 99% of the band (vs 72%) so it
co-observes MORE solid voxels (337 vs 243); better placement raises occ_iou but the
*fraction* still missed is similar — the residual obstacle-base miss is the next axis;
(3) honesty unchanged: MapAnything stays `metric_pseudo_label` (learned soft prior),
measured TUM eval-only. **Caveats before full adoption:** the Apache checkpoint's
isolated indoor accuracy is unpublished and this is ONE sequence — verify on
`phone_room` (runs, no band GT) and a second indoor scene; then wire MapAnything as the
default M3 backbone (regenerate `external/teacher_artifacts/` via the new runner) and
confirm the full `python -m atlas3r.evaluate` scorecard + determinism.

## 1. TL;DR — what to try FIRST, in order

The diagnosed root cause is an **intrinsic monocular depth-accuracy limit of the DA3 backbone** (surfaces ~12 cm displaced, worst at obstacle bases near the floor; 72% of co-observed band obstacles missed). Every fusion/optimization/keyframe lever is already exhausted. The honest lever is **a more accurate and/or genuinely metric geometry source at the M3 backbone seam** — *not* a new occupancy module, *not* VoxFormer. All three picks slot into the existing **M3 geometry adapter** (`load_geometry_artifacts` in `geometry_adapter.py`), keep metric scale a soft prior, and require no vendoring.

| # | Model | Slots into | Mechanism vs the 12 cm gap | License for teacher/eval | Commercial path |
|---|-------|-----------|----------------------------|--------------------------|-----------------|
| **1** | **MapAnything (`facebook/map-anything-apache`)** | M3 backbone (drop-in: depth+pose+intrinsics in ONE model, replaces the DA3 geometry+da3metric pair) | SOTA-class feed-forward metric geometry; explicit metric-scale head should tighten obstacle-base placement vs DA3's soft anchor | Apache-2.0 code **and** Apache checkpoint | **Yes** (Apache weights) |
| **2** | **Pi3 (π³)** | M3 backbone (relative-geometry source + external anchor) | **Highest raw indoor depth accuracy** (NYU AbsRel 0.054, Bonn 0.049) → most likely to shrink per-pixel displacement *if* anchored | Code BSD-3; **weights CC-BY-NC** | No (research/teacher only) |
| **3** | **MoGe-2** | M3 depth-only prior (per-keyframe; poses from #1/SfM) | **Sharpest near-surface detail** (NYUv2 point-inliers 93.6%) — directly targets the obstacle-base edge smear | MIT (verify HF revision) | Yes |

**Decisive framing:** run **MapAnything-Apache as the first A/B against `da3metric-large`** — it is the only pick that is metric, commercially clean, drop-in for the full factored artifact set, *and* indoor-trained (ScanNet++ v2 + TartanAirV2-WB are in its Apache training set). If MapAnything underdelivers on the band, **Pi3** is the highest-accuracy relative backbone for the teacher/eval role, and **MoGe-2** is the cheapest edge-sharpening complement. Do not adopt any of them blindly — adopt only on a **measured win on the band3d_agreement scorecard** (Section 5). Every honest fit verdict landed at **"pilot," not "adopt."**

---

## 2. VoxFormer & the occupancy-prediction family — explicit verdict

**VoxFormer (NVIDIA, CVPR 2023): DO NOT ADOPT.** All four independent investigations agree, and the reasoning is structural, not incidental:

- **Wrong input:** VoxFormer's stage-1 depends on **stereo / pseudo-LiDAR depth** (MobileStereoNet); Atlas3R is monocular RGB only. ([NVlabs/VoxFormer](https://github.com/NVlabs/VoxFormer), [arXiv 2302.12251](https://arxiv.org/abs/2302.12251))
- **Wrong domain:** verified **only on SemanticKITTI / KITTI-360 / nuScenes** — outdoor driving. **No NYUv2/ScanNet/TUM/7-Scenes results exist.** Its label space (cars, roads) and 0.2 m driving voxels at tens-of-meters range are useless for a ~0.25 m floor-level collision band where failures live at 0–15 cm obstacle bases.
- **Wrong output:** it **outputs occupancy** — which is Atlas3R's own job — not the depth/pose/pointmap artifacts the M3 adapter consumes. It is not a geometry backbone.
- **It would inherit, not fix, the gap:** its stage-1 still lifts an off-the-shelf depth prior, so it carries the same monocular depth error.
- **License:** NVIDIA Source Code License-NC + CC-BY-NC-SA weights — non-commercial — and the driving weights would need a full indoor retrain Atlas3R explicitly cannot afford.

The **driving occupancy family as a whole** (SurroundOcc, OccFormer, TPVFormer, FB-OCC, PanoOcc, SparseOcc, GaussianFormer, OccMamba, and 2024–26 successors) is the **wrong branch** for the same reasons: surround multi-camera rigs, driving label space, no indoor weights.

**The correct sibling reference IS the indoor monocular SSC line** — ISO (ECCV 2024), EmbodiedOcc/++ (CVPR/ACM-MM 2025), GA-MonoSSC (ICCV 2025), SGR-OCC (2026). **Verdict on that line: not the first move, and only "partly" addresses the gap.** The decisive limitation: **every viable indoor occupancy model derives its metric scale from the SAME frozen Depth-Anything / DA-V2 class of prior that produces Atlas3R's 12 cm displacement** — so they do **not** escape the depth-accuracy ceiling. The only honest expected gain is **learned voxel completion** (snapping obstacle bases to the floor where ray-marched depth-fusion structurally cannot) — a plausible **recall** lever, not a proven **displacement** fix. If you ever pursue this branch:
- **EmbodiedOcc** (CC-BY-NC-SA-4.0; posed-monocular-video → 0.08 m metric occupancy; RTX 4090-trained) is the closest-shape candidate, but emits **occupancy artifacts, not depth/pose** — it needs a *new* occupancy-ingestion adapter (a sibling to `geometry_adapter.py`), not the existing depth→ray path. Its `EmbodiedOcc++` plane/normal regularization is the one mechanism aimed at obstacle-base placement, but **no paper in this line reports any cm/m surface error — only IoU/mIoU**, so the 12 cm claim is entirely unverified for it.
- **ISO** (Apache-2.0, cleanest license) is a per-keyframe occupancy *proposer* but wraps Depth-Anything-v1 as a frozen prior → inherits the depth ceiling; its verdict was the bluntest: **"no"** on the 12 cm displacement axis (single-image, no multi-view consistency, only voxel IoU reported).
- **SGR-OCC** (2026) is the accuracy frontier and its ray-constrained refinement targets exactly the failure mode, **but the repo is a placeholder (no code/weights as of mid-2026)** and license is murky (Apache stub over CC-BY-NC-SA + ScanNet lineage). **Track, do not adopt.**

**Bottom line:** the leverage is in the depth/geometry backbone (Section 1), not in adopting an occupancy predictor. Keep the indoor SSC line as a *second-phase recall experiment* only after a stronger backbone is measured.

---

## 3. Ranked shortlist

For each: what it gives · indoor evidence · license · 24 GB feasibility · integration sketch · does it address the 12 cm gap.

### Tier 1 — pilot now (M3 backbone swap)

**1. MapAnything — `facebook/map-anything-apache` (Meta + CMU, arXiv 2509.13414)** — *PRIMARY A/B vs da3metric-large.*
- **Gives:** one model emitting factored depth_z + camera poses (4×4) + intrinsics (3×3) + pts3d + per-pixel conf + masks from 1..N monocular RGB views — a 1:1 match for Atlas3R's M3 artifact contract, collapsing the DA3 geometry + da3metric pair into one.
- **Indoor evidence:** SOTA on ScanNet++ v2 / ETH3D / TartanAirV2-WB. **CORRECTION (verified):** the headline ScanNet++ v2 depth AbsRel ~0.033 belongs to the benchmarked (full/NC-class) model; the **README publishes NO isolated ScanNet++ AbsRel for the Apache checkpoint** — the "~0.06 Apache" figure is an **unconfirmed estimate**. The Apache checkpoint is documented only as "competitive with VGGT" and is the **weaker** of the two. ([github.com/facebookresearch/map-anything](https://github.com/facebookresearch/map-anything), [huggingface.co/facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache))
- **License:** Apache-2.0 **code AND `-apache` checkpoint** (commercial-clean; the two-checkpoint split is real: `map-anything` (CC-BY-NC) vs `map-anything-apache` (Apache-2.0)). Removes DA3's CC-BY-NC ambiguity.
- **24 GB:** Yes — ~14 GB peak for 16 views; a handful of offline keyframes is comfortable.
- **Integration:** isolated `external/mapanything_env`, `pip install mapanything`, `tools/run_mapanything_backbone.py` mirroring `tools/run_da3_backbone.py`; map output dict → existing schema; set `metric_evidence=false`, `learned_metric_depth_prior=true`, `units='meters'`. **Pin pose (w2c vs c2w) and depth (z vs radial) conventions from the installed source** — the adapter's `_source_depth_to_radial` handles `optical_z`, and `run_da3_backbone` already does the w2c→c2w step to copy.
- **12 cm gap:** **PARTLY.** Right layer, metric head, indoor-trained → real reason to expect tighter surfaces. But (a) the legally-usable Apache checkpoint is weaker and its indoor number unpublished; (b) AbsRel is global relative depth — **DA3 also scores well on ScanNet++ AbsRel yet still misses 72% of band obstacles**, so the benchmark does not predict band recall. Must be measured on the scorecard.

**2. Pi3 (π³, arXiv 2507.13347, ICLR 2026)** — *highest-accuracy relative backbone; teacher/eval role.*
- **Gives:** feed-forward poses (4×4 OpenCV cam-to-world) + per-view local pointmaps + conf from unordered/unposed RGB; permutation-equivariant. Depth = z of pointmap.
- **Indoor evidence:** **best in the set** — NYU-v2 AbsRel 0.054 / δ1 0.956 (beats VGGT 0.056); Bonn 0.049 / δ1 0.975. *Caveat:* these are **up-to-scale / affine-invariant**, reported after per-frame GT scale alignment — relative-geometry quality, not metric placement.
- **License:** code BSD-3 (commercial OK); **weights CC-BY-NC 4.0 (non-commercial)** → fine for the offline teacher and research labels, **blocker if labels ship in a commercial product.**
- **24 GB:** Yes — 959M params, faster than VGGT.
- **Integration:** `external/pi3_env`; convert local_points → radial/optical depth raster; **no intrinsics head** in base Pi3 → reuse the assumed-focal prior (`0.8*max(W,H)`) recorded as `assumed_prior`, or use Pi3X conditioned on known intrinsics. Set `metric_evidence=false` and **do NOT set `learned_metric_depth_prior`** (it is up-to-scale) → M5 treats it as `non_metric_pseudo_label` until an external anchor is supplied.
- **12 cm gap:** **PARTLY.** Best raw accuracy → most likely to shrink displacement *if* a reliable metric anchor is supplied; but up-to-scale AbsRel may not drop the metric floor-band error proportionally.

**3. MoGe-2 (Microsoft, arXiv 2507.02546, NeurIPS 2025)** — *complementary per-keyframe sharp-detail prior.*
- **Gives:** single-image metric point map + depth + normals + intrinsics + mask. **No poses** → poses come from MapAnything/Pi3/SfM.
- **Indoor evidence:** NYUv2 point-inliers δ1ᵖ **93.6%**, Relᵖ 8.19% (beats UniDepth-V2 91.9%, Depth Pro 81.9%); iBims-1 ~5.63, ETH3D ~7.19. *Caveat:* all numbers are **post optimal-scale/affine alignment** — relative-geometry sharpness, not raw metric scale; the regressed global scalar has **no published raw AbsRel**.
- **License:** **MIT** (code + HF weights; DINOv2 subtree Apache-2.0). Most permissive metric option. Pin the exact HF revision (issue #98 was a README omission, resolved as MIT on the model card).
- **24 GB:** Yes, easily (~326M params, ~60 ms/image).
- **Integration:** `external/moge2_env`, `tools/run_moge2_backbone.py` loading `Ruicheng/moge-2-vitl-normal`; emit depth + mask per keyframe; poses from the multi-view track. **Safer variant:** keep MoGe-2's affine-invariant geometry and **re-anchor its scale to the SfM cloud**, demoting its global scalar (the least-validated part) to a prior.
- **12 cm gap:** **PARTLY.** Its entire pitch — sharp fine-grained geometry — targets the obstacle-base edge smear exactly. But the metric scalar is a soft prior, and per-frame independence can inject inter-keyframe scale inconsistency that must be re-anchored.

### Tier 2 — strong evidence / inspiration, heavier or license-capped

**4. MASt3R-SLAM (CVPR 2025, arXiv 2412.12392)** — *evidence that joint pose+geometry optimization beats a bigger net.*
- **Indoor evidence (CORRECTED & verified from the paper's Table 3):** 7-Scenes reconstruction **Accuracy 0.089 m, Completion 0.085 m, Chamfer 0.087 m** (≈**8.7 cm**, ICP-aligned whole-scene RMSE) — **NOT the "2.25 cm" one survey claimed** (misattributed). TUM RGB-D ATE 0.030 m calibrated / 0.060 m uncalibrated; 7-Scenes ATE 0.047 m. The ~8.7 cm is still below DA3's ~12 cm, but it is a global cloud RMSE, **not** a near-floor/collision-band number.
- **License:** CC-BY-NC-SA-4.0 + NAVER MASt3R checkpoint restrictions → non-commercial.
- **Honesty:** authors **explicitly discard MASt3R's native metric scale** ("a large source of inconsistency") and define poses in **Sim(3)** → provides **zero honest metric anchor**; must land as `non_metric_pseudo_label` (`units=unitless_similarity`, `scale_to_meters=null`), anchored only by Atlas3R's measured-TUM fitting.
- **Verdict:** **medium priority.** Needs ordered video and is a full real-time SLAM system to wrap — heavier than a single forward pass. Strong *inspiration* that the gap is closable by joint optimization; weaker as a near-term drop-in.

### Tier 3 — usable but unlikely to move the needle / capped

- **VGGT / VGGT-SLAM** (CVPR 2025 Best Paper): closest paradigm-match to DA3, mature ecosystem, **a commercial checkpoint exists** (`facebook/VGGT-1B-Commercial`). Indoor depth ScanNet AbsRel 0.049 / NYUv2 0.035. **But DA3 already reports beating VGGT (+35.7% pose, +23.6% geometry)**, so a DA3→VGGT swap likely *widens* the gap; scale is a learned prior (VGGT-SLAM resolves only up to a 15-DoF SL(4) **projective** ambiguity uncalibrated — weaker than similarity). The 1B weights are non-commercial; the commercial checkpoint's Acceptable-Use Policy **prohibits heavy-machinery / bodily-harm-risk operation**, which a floor-cleaning robot deployment could plausibly trip — **needs legal review.** Sanity baseline at best.
- **Metric3D v2 / UniDepthV2 / Apple Depth Pro** (metric-depth-scale domain): all single-image, **no poses**, so they only swap the depth source (poses stay from SfM/DA3). Metric3D v2 (BSD-2 text but **author states non-commercial**, ambiguous; NYUv2 AbsRel 0.063 ~ decimeter error at indoor range, marginal over DA3). UniDepthV2 (CC-BY-NC; best intrinsics-free zero-shot, **TUM-RGBD δ1 90.5** — the domain Atlas3R validates on — but TUM 3D F-score only 62.9 and ARel ~22%). Depth Pro (Apple Sample-Code license; **best boundary sharpness, recall 0.173 vs DA-v2 0.107** → targets obstacle-base smear, but weak/unproven metric scale, no poses). Treat as **edge-sharpness refiners after a metric model sets scale**, not primaries.

---

## 4. What NOT to do / dead ends

| Don't | Why |
|-------|-----|
| **Adopt VoxFormer or any driving occupancy net** | Stereo/pseudo-LiDAR input, outdoor driving domain, occupancy output (Atlas3R's own job), NC weights, 0.2 m driving voxels. Wrong on every axis. |
| **Swap DA3 → VGGT for accuracy** | DA3 already reports beating VGGT; likely *widens* the gap. Use only as a sanity baseline. |
| **Adopt DUSt3R / MASt3R / Spann3R / Fast3R / SLAM3R as backbones** | DUSt3R-class accuracy, superseded indoors by Pi3/DA3/MapAnything; heaviest NC licenses; Spann3R/Fast3R optimize scale/speed (irrelevant — a few keyframes is the regime), not sub-12 cm accuracy. |
| **Adopt GLORIE-SLAM / Splat-SLAM to fix depth** | **CORRECTION (verified): GLORIE's 3.24 cm depth-L1 is Replica (synthetic) only; ScanNet (real indoor) reports ATE ~7.5 cm and NO per-frame depth-L1** — real-world depth accuracy unproven. Both optimize scale/shift of a *relative* mono-depth prior under BA — exactly the fusion/optimization lever Atlas3R already exhausted. Splat-SLAM's ScanNet depth-L1 ~11.4 cm is in the same ~12 cm class. |
| **Adopt CUT3R** | "Metric" is unreliable indoors — 7-Scenes **unaligned** AbsRel ~1.0 (off by ~2×). Would likely worsen base placement. |
| **Adopt ZoeDepth / DA-V2-metric-indoor head as the fix** | Same DA lineage / superseded on indoor zero-shot by Metric3D v2 / UniDepthV2; an intrinsic-accuracy ceiling won't clear 12 cm. |
| **Use Marigold / GeoWizard / Lotus as a primary** | Relative/affine-invariant — **not metric**, cannot anchor scale; only an edge-sharpness refiner after a metric model. |
| **Adopt an indoor occupancy net (ISO/EmbodiedOcc/SGR-OCC) as the first move** | They sit on the same Depth-Anything prior → inherit the 12 cm ceiling; report only IoU/mIoU, never cm-level surface error; need a new occupancy adapter, not the depth→ray path. SGR-OCC has no released code. Second-phase *recall* experiment at most. |
| **Feed measured TUM depth into any candidate** | Measured depth is **eval/ground-truth ONLY**, never a candidate input. |

---

## 5. Concrete first experiment — the cheapest high-signal test

**Pick:** MapAnything-Apache vs DA3, framed as a **swap of the M3 artifact** (no vendoring, scale stays a soft prior).

1. **Isolated env (no vendoring):** create `external/mapanything_env`, then **source-install** (NOT `pip install mapanything` — there is no PyPI package): `git clone https://github.com/facebookresearch/map-anything.git external/map-anything-src && <env>/python -m pip install -e external/map-anything-src` (after installing a CUDA torch). Weights download to the HF cache, never into git. **(Built this session: `tools/run_mapanything_backbone.py` with the verified API/conventions.)**
2. **Runner:** add `tools/run_mapanything_backbone.py` (standalone, not imported by the `atlas3r` package, heavy imports lazy), mirroring `tools/run_da3_backbone.py`. Inputs: `--asset-id / --frames-dir / --num-frames / --device`. Run `MapAnything.from_pretrained('facebook/map-anything-apache').infer(views, use_amp=True, amp_dtype='bf16')` over **the SAME M1 keyframe set DA3 used** (prior experiments showed changing the joint geometry, e.g. denser keyframes, can shift surfaces and *regress* band recall — hold the keyframe set fixed).
3. **Adapt OUTPUT only** into `external/teacher_artifacts/<asset_id>/`, matching the frozen schema in `geometry_adapter.py`:
   - `depth_z` → `depth/<frame>.npy`, `depth_meta.json` with `depth_convention='optical_z'` (the adapter converts to radial via `_source_depth_to_radial`).
   - `camera_poses` → `poses.json` as `T_world_camera` (invert if MapAnything emits world-to-camera — **verify convention from the installed source**, exactly as `run_da3_backbone` does its w2c→c2w step).
   - `intrinsics` → `intrinsics.json` (fx,fy,cx,cy,width_px,height_px). `conf` → `confidence/<frame>.npy` in [0,1].
4. **Honesty wiring** in `backbone_manifest.json`: `metric_evidence=false`, `learned_metric_depth_prior=true`, `units='meters'`, `scale_to_meters_or_null=1.0`, `model_name='map-anything-apache'`, `backbone_name='atlas3r_mapanything'`. **Verified in code:** `geometry_adapter.py:319` emits a `LEARNED_METRIC_DEPTH_PRIOR` ScaleEvidence with `measured=False` only when `learned_metric_depth_prior==true AND units=='meters'` → M5 caps it at `metric_pseudo_label`, **never `measured_metric`**. Identical contract to DA3 — **no honesty rules change.**
5. **Same acceptance harness:** `python -m atlas3r.teacher` then `python -m atlas3r.evaluate`. Compare `runs/eval/scorecard.json` band metrics **head-to-head vs the DA3 baseline**: `occupied_static_iou`, band `free_space_contradiction_rate`, `per_class_agreement`, the **0–26 cm base-height contradiction profile** (the decisive metric — it isolates the obstacle-base failure mode the 12 cm gap names), and camera-center Sim(3) RMSE. Record in `docs/band_obstacle_recall_evidence.md`.
6. **Adopt/reject gate:** keep DA3 as default until MapAnything **wins `occupied_static_iou` AND holds the `per_class` guardrail** on the real scorecard. **The benchmark AbsRel does NOT decide this — the band scorecard does.** Do **not** feed measured TUM depth as input.

**If MapAnything-Apache underdelivers:** the same harness + adapter pattern A/Bs **Pi3** (set `non_metric_pseudo_label`, supply external anchor) next, then **MoGe-2** as a depth-only re-anchored prior. Build the runner once; the adapter seam is shared.

---

## 6. Honesty caveats — what the web could NOT confirm / verify before trusting

1. **MapAnything Apache indoor accuracy is UNPUBLISHED (independently re-verified).** The strong ScanNet++ v2 AbsRel (~0.033) is the benchmarked full/NC-class model; the README gives **no isolated ScanNet++ AbsRel for `map-anything-apache`** (documented only as "competitive with VGGT"). The "~0.06 Apache" figure is a **plausible-but-unconfirmed estimate** — its band benefit must be *measured*, not assumed.
2. **AbsRel ≠ band recall (the central honesty point).** Benchmark AbsRel is a global relative-depth mean dominated by mid-range pixels; **DA3 scores well on ScanNet++ AbsRel yet still misses 72% of 0.25 m-band obstacles.** No candidate's leaderboard number predicts the floor-band, obstacle-base, absolute-placement metric Atlas3R actually fails on. The scorecard A/B is the only honest arbiter.
3. **Two survey numbers were wrong and are corrected here:** (a) MASt3R-SLAM's 7-Scenes reconstruction is **~8.7 cm (Table 3), not 2.25 cm**; (b) GLORIE-SLAM's **3.24 cm depth-L1 is synthetic Replica only — ScanNet real-world depth-L1 is not reported** (only ATE ~7.5 cm). Treat any single survey number as a hypothesis until re-checked.
4. **All "metric" claims are SOFT priors, never measurement.** MapAnything, MoGe-2, Metric3D v2, UniDepthV2, Depth Pro all regress scale from learned priors; CUT3R's indoor "metric" is unreliable (unaligned 7-Scenes AbsRel ~1.0). Under Atlas3R's invariants these are at most `metric_pseudo_label`. Only TUM RGB-D ground truth is `measured_metric`, and it stays eval-only.
5. **License risks to clear before any commercial shipping of labels/models:**
   - Pi3, UniDepthV2, UniK3D, MASt3R(-SLAM), EmbodiedOcc: **non-commercial weights** — fine for the offline research teacher, blocker if labels train a shipped product.
   - Metric3D v2: BSD-2 text **conflicts** with the authors' "non-commercial usage" note — get written clarification before product use.
   - VGGT commercial checkpoint: Acceptable-Use Policy may prohibit robot/heavy-machinery deployment — **legal review.**
   - MapAnything-Apache (Apache-2.0) and MoGe-2 (MIT, pin HF revision) are the **commercially clean** options — a deliberate reason MapAnything is pick #1.
   - SGR-OCC: license murky (Apache stub over CC-BY-NC-SA + ScanNet lineage) **and no code released** — track only.
6. **Convention & platform risks:** pin each model's pose (w2c vs c2w) and depth (z vs along-ray) convention from the *installed source* before trusting `poses.json` — never guess. UniDepth/some repos target Linux/CUDA; Atlas3R is on Windows → run in WSL/Linux or the ONNX path, skip optional CUDA-op compilation.
7. **Residual depth floor is intrinsic.** Even the best learned monocular backbone carries an irreducible per-pixel error; a stronger backbone *plausibly* shrinks the 12 cm gap but is not *proven* to clear the band tolerance until the scorecard says so. The honest expectation everywhere is "pilot and measure," not "adopt."

**Relevant code/files:**
- `src/atlas3r/geometry_adapter.py` — M3 ingestion + honesty seam (`load_geometry_artifacts`; `LEARNED_METRIC_DEPTH_PRIOR` emission ~lines 309–343; `_source_depth_to_radial` depth-convention conversion).
- `src/atlas3r/scale.py` — M5 scale ladder that caps learned priors at `metric_pseudo_label`.
- `src/atlas3r/contracts.py` — `VoxelOccupancyGrid3D`, `ScaleEvidence`, `ScaleEvidenceType`.
- `tools/run_da3_backbone.py` — the exact runner pattern to mirror for `run_mapanything_backbone.py` (no vendoring).
- `src/atlas3r/teacher.py`, `evaluate.py`, `validation.py` — the acceptance harness for the Section 5 A/B.
