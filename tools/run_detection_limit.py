"""Per-scene injected-corruption detection-limit calibration (S8).

Implements the "measured authority" principle (ARCHITECTURE.md Module 11):
inject known corruptions into the FINISHED candidate reconstruction (no
refinement repair pass), re-score the GT-free signal suite on every injected
variant, and report -- per scene, per corruption family -- the smallest
injected magnitude the CURRENT acceptance gate would have rejected.

The output certificate is named `injected_corruption_detection_limit` on
purpose: it certifies detection of THE TESTED FAMILIES ONLY, in scene-relative
units. It never claims metric accuracy (gauge freedom: without an anchor there
is no cm ruler) and never claims coverage of error shapes outside the tested
families.

On scenes WITH a measured M2 reference, the true Sim(3) RMSE delta per
injection is also computed -- GT validates the METHOD (does the gate's
response track real induced error?), never the per-scene verdict.

Usage:
  python tools/run_detection_limit.py [--asset reference_metric]
      [--artifacts-dir external/teacher_artifacts] [--seeds 3] [--quick]

Output: runs/_diag/detection_limit_<asset>.json + console summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas3r.config import load_robot_envelope  # noqa: E402
from atlas3r.geometry_adapter import (  # noqa: E402
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from atlas3r.inject import (  # noqa: E402
    CORRUPTION_FAMILIES,
    DEFAULT_MAGNITUDES,
    IN_REFINE_SPAN,
    MAGNITUDE_UNITS,
    inject_corruption,
    trajectory_span_m,
)
from atlas3r.mapping import fuse_static_map  # noqa: E402
from atlas3r.refine import refine_scene  # noqa: E402
from atlas3r.scale import estimate_scale_posterior  # noqa: E402
from atlas3r.validation import (  # noqa: E402
    MAX_CONTRADICTION_RATE_FOR_ACCEPT,
    MAX_HELD_OUT_ERROR_FOR_ACCEPT,
    _evidence_mass_stage,
    _gravity_alignment_stage,
    _held_out_render_error,
)
from atlas3r.visibility import build_visibility_graph  # noqa: E402
from atlas3r import teacher as T  # noqa: E402

MEASURED_SCENES = {"reference_metric", "reference_metric_desk", "reference_metric_room"}

# Suite recipe: visibility + held-out + fusion WITHOUT static/dynamic inference
# (near-static canonical scenes; identical recipe for clean baseline and every
# injection, so responses are attributable to the corruption alone).
SUITE_RECIPE = {
    "refine": "once_on_clean_artifacts_only -- injections get NO repair pass",
    "static_dynamic": "omitted (states=None) -- identical for baseline and injections",
    "fusion": "apply_fusion_policy=True (candidate policy)",
}


def score_suite(packets, soft_evidence, envelope) -> dict:
    """The GT-free signal suite + current-gate verdict for one packet set."""
    _, vis_report = build_visibility_graph(packets)
    post, _ = estimate_scale_posterior(packets, soft_evidence)
    _, _, _, cand_field, map_report = fuse_static_map(
        packets, post, static_dynamic_states=None,
        envelope=envelope, apply_fusion_policy=True,
    )
    held_out, _note = _held_out_render_error(packets, None)
    stage0 = _evidence_mass_stage(vis_report, True)
    stage1 = _gravity_alignment_stage(map_report, True)
    fsc = map_report.get("free_space_contradiction_rate")
    fsc = float(fsc) if isinstance(fsc, (int, float)) else 0.5

    reject_reasons = []
    reject_reasons += stage0["rejection_reasons"]
    reject_reasons += stage1["rejection_reasons"]
    if fsc > MAX_CONTRADICTION_RATE_FOR_ACCEPT:
        reject_reasons.append(f"free_space_contradiction_rate_too_high:{fsc:.3f}")
    if held_out > MAX_HELD_OUT_ERROR_FOR_ACCEPT:
        reject_reasons.append(f"held_out_render_error_too_high:{held_out:.3f}")

    floor = map_report.get("floor", {}) or {}
    return {
        "signals": {
            "median_reprojection_inbounds_ratio": stage0["median_reprojection_inbounds_ratio"],
            "depth_residual_edge_fraction": stage0["depth_residual_edge_fraction"],
            "mean_confidence_weight": stage0["mean_confidence_weight"],
            "held_out_render_error": held_out,
            "free_space_contradiction_rate": fsc,
            "unknown_fraction": map_report.get("unknown_fraction"),
            "floor_inlier_ratio": floor.get("inlier_ratio"),
            "floor_up_alignment_applied": floor.get("up_alignment_applied"),
        },
        "internal_signal_reject": bool(
            stage0["would_reject"]
            or fsc > MAX_CONTRADICTION_RATE_FOR_ACCEPT
            or held_out > MAX_HELD_OUT_ERROR_FOR_ACCEPT
        ),
        "gravity_stage_reject": bool(stage1["would_reject"]),
        "gate_would_reject": bool(reject_reasons),
        "reject_reasons": reject_reasons,
        "_cand_field": cand_field,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="reference_metric")
    parser.add_argument("--artifacts-dir", default="external/teacher_artifacts")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--quick", action="store_true",
        help="1 seed, outermost magnitudes only (smoke run)",
    )
    args = parser.parse_args()

    envelope, _ = load_robot_envelope(root=ROOT)
    mono, grep = load_geometry_artifacts(args.asset, ROOT, artifacts_dir=args.artifacts_dir)
    soft = grep.get("_scale_evidence", [])
    refined, _refine_report = refine_scene(mono, fix_global_scale=bool(soft))
    span = trajectory_span_m(refined)

    measured = None
    if args.asset in MEASURED_SCENES:
        measured, _ = load_measured_packets_from_m2(args.asset, ROOT)

    def true_rmse(packets):
        if measured is None:
            return None
        cam = T._compare_candidate_to_measured(packets, measured)
        return cam.get("trajectory_rmse_m_after_alignment")

    print(f"[detection_limit] {args.asset}: {len(refined)} keyframes, "
          f"trajectory span {span:.3f} (reconstruction units)", flush=True)

    t0 = time.time()
    clean = score_suite(refined, soft, envelope)
    clean.pop("_cand_field", None)
    clean_rmse = true_rmse(refined)
    print(f"[detection_limit] clean baseline: gate_would_reject="
          f"{clean['gate_would_reject']} reasons={clean['reject_reasons']} "
          f"true_rmse={clean_rmse} ({time.time()-t0:.0f}s)", flush=True)

    seeds = 1 if args.quick else max(1, args.seeds)
    runs = []
    for family in CORRUPTION_FAMILIES:
        mags = DEFAULT_MAGNITUDES[family]
        if args.quick:
            mags = (mags[0], mags[-1])
        for magnitude in mags:
            for seed in range(seeds):
                t1 = time.time()
                corrupted, record = inject_corruption(refined, family, magnitude, seed)
                suite = score_suite(corrupted, soft, envelope)
                suite.pop("_cand_field", None)
                row = {
                    "family": family,
                    "magnitude": magnitude,
                    "magnitude_units": MAGNITUDE_UNITS[family],
                    "seed": seed,
                    "in_refine_parametric_span": IN_REFINE_SPAN[family],
                    "suite": suite,
                    "true_sim3_rmse_m": true_rmse(corrupted),
                    "runtime_s": round(time.time() - t1, 1),
                }
                runs.append(row)
                print(
                    f"[detection_limit] {family}@{magnitude} seed={seed}: "
                    f"reject={suite['gate_would_reject']} "
                    f"internal={suite['internal_signal_reject']} "
                    f"gravity={suite['gravity_stage_reject']} "
                    f"rmse={row['true_sim3_rmse_m']} ({row['runtime_s']}s)",
                    flush=True,
                )

    # Detection limit per family: smallest magnitude rejected in >= 2/3 of seeds
    # (majority; with --quick's single seed it is 1/1 and labeled accordingly).
    majority = (seeds // 2) + 1
    detection_limits = {}
    for family in CORRUPTION_FAMILIES:
        fam_rows = [r for r in runs if r["family"] == family]
        mags = sorted({r["magnitude"] for r in fam_rows})
        dl = None
        per_mag = {}
        for m in mags:
            rejected = sum(
                1 for r in fam_rows
                if r["magnitude"] == m and r["suite"]["gate_would_reject"]
            )
            per_mag[str(m)] = f"{rejected}/{seeds}"
            if dl is None and rejected >= majority:
                dl = m
        detection_limits[family] = {
            "detection_limit": dl,
            "units": MAGNITUDE_UNITS[family],
            "rejected_seeds_per_magnitude": per_mag,
            "no_detection_at_any_tested_magnitude": dl is None,
        }

    # Negative-control analysis: did INTERNAL signals respond to rigid tilt?
    tilt_rows = [r for r in runs if r["family"] == "global_tilt_control"]
    tilt_internal_responses = sum(
        1 for r in tilt_rows if r["suite"]["internal_signal_reject"]
    )
    tilt_note = (
        "internal signals did not respond to rigid world tilt (gauge-blind, as "
        "expected); tilt coverage rests on the gravity/floor stage alone"
        if tilt_internal_responses == 0
        else f"UNEXPECTED: internal signals responded to rigid tilt in "
             f"{tilt_internal_responses}/{len(tilt_rows)} runs -- investigate "
             "before trusting the tilt-coverage claim"
    )

    # Method validation vs GT (only where measured evidence exists): response
    # monotonicity of true RMSE in injected magnitude, per geometric family.
    method_validation = None
    if measured is not None and clean_rmse is not None:
        def spearman(x, y):
            import numpy as np
            x, y = np.asarray(x, float), np.asarray(y, float)
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            rx -= rx.mean()
            ry -= ry.mean()
            d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
            return float((rx * ry).sum() / d) if d > 0 else float("nan")

        method_validation = {"note": (
            "GT validates the METHOD (injection magnitude must track real "
            "induced trajectory error), never a per-scene verdict"
        )}
        for family in ("pose_drift_translation", "pose_drift_rotation", "scale_drift_ramp"):
            fam = [r for r in runs if r["family"] == family
                   and isinstance(r["true_sim3_rmse_m"], (int, float))]
            if len(fam) >= 3:
                method_validation[family] = {
                    "spearman_magnitude_vs_true_rmse": round(
                        spearman([r["magnitude"] for r in fam],
                                 [r["true_sim3_rmse_m"] for r in fam]), 3),
                    "n": len(fam),
                }

    certificate = {
        "field_name_rationale": (
            "named injected_corruption_detection_limit, NOT verified accuracy: "
            "it certifies detection of the tested families only"
        ),
        "asset_id": args.asset,
        "artifacts_dir": args.artifacts_dir,
        "n_keyframes": len(refined),
        "trajectory_span_reconstruction_units": span,
        "gauge": "up_to_similarity -- magnitudes are scene-relative, never cm",
        "clean_baseline": {
            "gate_would_reject": clean["gate_would_reject"],
            "reject_reasons": clean["reject_reasons"],
            "signals": clean["signals"],
            "true_sim3_rmse_m": clean_rmse,
        },
        "detection_limits": detection_limits,
        "negative_control_tilt": {
            "internal_signal_responses": tilt_internal_responses,
            "n_tilt_runs": len(tilt_rows),
            "note": tilt_note,
        },
        "method_validation_vs_gt": method_validation,
        "suite_recipe": SUITE_RECIPE,
        "seeds": seeds,
        "coverage_disclaimer": (
            "errors shaped unlike every tested corruption family are NOT covered "
            "by this certificate; detection limits transfer to no other scene"
        ),
        "runs": runs,
    }

    out = ROOT / "runs/_diag" / f"detection_limit_{args.asset}.json"
    out.write_text(json.dumps(certificate, indent=2, default=str), encoding="utf-8")
    print(f"\n[detection_limit] wrote {out}")
    print(json.dumps({k: v for k, v in certificate.items() if k != "runs"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
