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
from typing import Any

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


LEDGER_SIGNALS = (
    "ledger_offset_drift_p90_span_fraction",
    "ledger_offset_rate_p90_span_fraction_per_frame",
    "ledger_normal_drift_p90_deg",
    "ledger_scale_ramp_p90_abs_log_ratio",
)


def score_suite(packets, soft_evidence, envelope) -> dict:
    """The GT-free signal suite + current-gate verdict for one packet set."""
    from atlas3r.plane_ledger import ledger_for_packets

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

    ledger = ledger_for_packets(packets)
    ledger_signals = ledger.get("signals", {}) if ledger.get("status") == "audited" else {}

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
            **{k: ledger_signals.get(k) for k in LEDGER_SIGNALS},
        },
        "ledger_status": ledger.get("status"),
        "ledger_direction_authority": ledger.get("direction_authority"),
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


# Thresholded signals the gate consults, with crossing direction.
GATED_SIGNALS = {
    "free_space_contradiction_rate": ("above", MAX_CONTRADICTION_RATE_FOR_ACCEPT),
    "held_out_render_error": ("above", MAX_HELD_OUT_ERROR_FOR_ACCEPT),
    "median_reprojection_inbounds_ratio": ("below", 0.30),
    "depth_residual_edge_fraction": ("below", 0.70),
    "mean_confidence_weight": ("below", 0.30),
}


def compute_detection_limits(runs, clean, seeds) -> dict:
    """Honest detection limits with two guards the naive min-rule lacks:

    1. MONOTONICITY: a family whose rejected-count is non-monotone in magnitude
       has no reliable detection limit -- rejections that appear at small
       magnitudes and vanish at large ones are threshold noise, not detection.
    2. SOLID CROSSING: a rejection only counts as detection if some gated
       signal's across-seed MEDIAN crosses its threshold by more than the
       across-seed half-range at that magnitude. A crossing inside the seed
       noise band is a knife-edge artifact (measured on this very harness:
       clean fsc 0.244 vs gate 0.25 -- a 0.006 margin that coin-flips).

    Also records, per gated signal, the clean baseline's margin to threshold --
    tiny margins mean small-magnitude 'detections' carry no authority.
    """
    import numpy as np

    majority = (seeds // 2) + 1
    clean_signals = clean["signals"]
    clean_margins = {}
    for sig, (direction, theta) in GATED_SIGNALS.items():
        v = clean_signals.get(sig)
        if isinstance(v, (int, float)):
            clean_margins[sig] = round(theta - v if direction == "above" else v - theta, 4)

    out: dict[str, Any] = {"_clean_margin_to_threshold": clean_margins}
    for family in CORRUPTION_FAMILIES:
        fam_rows = [r for r in runs if r["family"] == family]
        if not fam_rows:
            continue
        mags = sorted({r["magnitude"] for r in fam_rows})
        per_mag, rejected_counts, solid_by_mag = {}, [], {}
        for m in mags:
            rows = [r for r in fam_rows if r["magnitude"] == m]
            rejected = sum(1 for r in rows if r["suite"]["gate_would_reject"])
            rejected_counts.append(rejected)
            solid = False
            crossings = {}
            for sig, (direction, theta) in GATED_SIGNALS.items():
                vals = [r["suite"]["signals"].get(sig) for r in rows]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if not vals:
                    continue
                med = float(np.median(vals))
                halfrange = (max(vals) - min(vals)) / 2.0
                excess = (med - theta) if direction == "above" else (theta - med)
                if excess > 0:
                    crossings[sig] = {
                        "median_excess_over_threshold": round(excess, 4),
                        "across_seed_halfrange": round(halfrange, 4),
                        "solid": excess > halfrange,
                    }
                    solid = solid or excess > halfrange
            # Stage 1 is a boolean: floor alignment lost consistently across
            # seeds is a deterministic solid detection (no noise band to clear).
            floor_lost = sum(
                1 for r in rows
                if r["suite"]["signals"].get("floor_up_alignment_applied") is False
            )
            if floor_lost >= majority:
                crossings["stage1_floor_alignment_lost"] = {
                    "seeds_lost": f"{floor_lost}/{len(rows)}",
                    "solid": True,
                }
                solid = True
            solid_by_mag[str(m)] = solid
            per_mag[str(m)] = {
                "rejected_seeds": f"{rejected}/{seeds}",
                "solid_crossing": solid,
                "crossings": crossings,
            }
        # Monotone means: once detection starts, it persists at larger magnitudes.
        started = False
        monotone = True
        for c in rejected_counts:
            if c >= majority:
                started = True
            elif started:
                monotone = False
                break
        dl = None
        if monotone:
            for m in mags:
                entry = per_mag[str(m)]
                rej = int(entry["rejected_seeds"].split("/")[0])
                if rej >= majority and entry["solid_crossing"]:
                    dl = m
                    break
        out[family] = {
            "detection_limit": dl,
            "units": MAGNITUDE_UNITS[family],
            "response_monotone": monotone,
            "per_magnitude": per_mag,
            "verdict": (
                "no_reliable_detection_limit_response_non_monotone" if not monotone
                else ("no_solid_detection_at_any_tested_magnitude" if dl is None
                      else "detected")
            ),
        }
    return out


def compute_ungated_responses(runs, clean, signal_names) -> dict:
    """Response curves for signals WITHOUT gate thresholds (e.g. the plane
    ledger). Language discipline: an ungated signal RESPONDS (median moves
    beyond the across-seed half-range AND beyond the clean baseline); only a
    gated signal can DETECT. These curves are the pre-registration source for
    any future threshold -- labeled GT configs stay pure test."""
    import numpy as np

    clean_signals = clean.get("signals", {})
    out: dict[str, Any] = {}
    families = sorted({r["family"] for r in runs})
    for sig in signal_names:
        base_val = clean_signals.get(sig)
        per_family: dict[str, Any] = {"clean_baseline": base_val}
        for family in families:
            fam_rows = [r for r in runs if r["family"] == family]
            curve = {}
            for m in sorted({r["magnitude"] for r in fam_rows}):
                vals = [r["suite"]["signals"].get(sig)
                        for r in fam_rows if r["magnitude"] == m]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if not vals:
                    curve[str(m)] = {"status": "no_values"}
                    continue
                med = float(np.median(vals))
                halfrange = (max(vals) - min(vals)) / 2.0
                responds = (
                    isinstance(base_val, (int, float))
                    and abs(med - base_val) > max(halfrange, 1e-9)
                )
                curve[str(m)] = {
                    "median": round(med, 4),
                    "across_seed_halfrange": round(halfrange, 4),
                    "responds_beyond_seed_noise": bool(responds),
                }
            per_family[family] = curve
        out[sig] = per_family
    return out


def tilt_negative_control(runs, clean, seeds) -> dict:
    """Rigid world tilt is gauge-invisible to multiview consistency; internal
    signals must not GENUINELY respond. Knife-edge threshold crossings inside
    the across-seed noise band are counted separately from solid responses."""
    import numpy as np

    tilt_rows = [r for r in runs if r["family"] == "global_tilt_control"]
    raw_responses = sum(1 for r in tilt_rows if r["suite"]["internal_signal_reject"])
    solid_responses = 0
    for m in sorted({r["magnitude"] for r in tilt_rows}):
        rows = [r for r in tilt_rows if r["magnitude"] == m]
        for sig, (direction, theta) in GATED_SIGNALS.items():
            vals = [r["suite"]["signals"].get(sig) for r in rows]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if not vals:
                continue
            med = float(np.median(vals))
            halfrange = (max(vals) - min(vals)) / 2.0
            excess = (med - theta) if direction == "above" else (theta - med)
            if excess > 0 and excess > halfrange:
                solid_responses += 1
    floor_realigned = sum(
        1 for r in tilt_rows
        if r["suite"]["signals"].get("floor_up_alignment_applied")
    )
    return {
        "n_tilt_runs": len(tilt_rows),
        "raw_internal_rejections": raw_responses,
        "solid_internal_responses": solid_responses,
        "floor_realigned_runs": floor_realigned,
        "note": (
            "no SOLID internal response to rigid tilt (raw rejections are "
            "knife-edge threshold noise); the floor RANSAC re-found the tilted "
            "floor and re-aligned the band in "
            f"{floor_realigned}/{len(tilt_rows)} runs -- the pipeline is "
            "tilt-EQUIVARIANT when the floor is reliable, so tilt risk "
            "concentrates exactly where floor RANSAC is weak, which Stage 1 "
            "rejects. Tilt coverage rests on the gravity stage alone."
            if solid_responses == 0
            else f"UNEXPECTED: {solid_responses} SOLID internal responses to "
                 "rigid tilt -- investigate before trusting the tilt-coverage claim"
        ),
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
    parser.add_argument(
        "--reanalyze", default=None, metavar="JSON",
        help="recompute detection limits from an existing run JSON (no re-run)",
    )
    args = parser.parse_args()

    if args.reanalyze:
        path = Path(args.reanalyze)
        cert = json.loads(path.read_text(encoding="utf-8"))
        runs = cert["runs"]
        clean = cert["clean_baseline"]
        seeds = int(cert.get("seeds", 3))
        cert["detection_limits"] = compute_detection_limits(runs, clean, seeds)
        cert["negative_control_tilt"] = tilt_negative_control(runs, clean, seeds)
        path.write_text(json.dumps(cert, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in cert.items() if k != "runs"},
                         indent=2, default=str))
        print(f"[detection_limit] reanalyzed -> {path}")
        return 0

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

    detection_limits = compute_detection_limits(runs, clean, seeds)
    ledger_response = compute_ungated_responses(runs, clean, LEDGER_SIGNALS)

    # Direction-resolved authority probe (red-team requirement): a plane track
    # is blind to translation drift perpendicular to its normal, so random-axis
    # injections would fabricate direction-averaged authority. Probe the
    # ledger's own measured DOMINANT and BLIND axes deterministically.
    direction_probe = None
    auth = clean.get("ledger_direction_authority")
    if isinstance(auth, dict):
        probe_rows = []
        for axis_name in ("translation_dominant_axis_world", "translation_blind_axis_world"):
            axis = auth.get(axis_name)
            if not axis:
                continue
            for magnitude in (0.10, 0.20):
                corrupted, record = inject_corruption(
                    refined, "pose_drift_translation", magnitude, 0, direction=axis,
                )
                suite = score_suite(corrupted, soft, envelope)
                suite.pop("_cand_field", None)
                probe_rows.append({
                    "axis": axis_name,
                    "magnitude": magnitude,
                    "direction": record["direction_axis"],
                    "ledger_signals": {k: suite["signals"].get(k) for k in LEDGER_SIGNALS},
                    "gate_would_reject": suite["gate_would_reject"],
                    "true_sim3_rmse_m": true_rmse(corrupted),
                })
                print(f"[detection_limit] direction probe {axis_name}@{magnitude}: "
                      f"{probe_rows[-1]['ledger_signals']}", flush=True)
        direction_probe = {
            "note": (
                "deterministic drift along the ledger's measured dominant vs "
                "blind axis; authority along the blind axis is NOT claimed -- "
                "this records exactly where the ledger can and cannot see"
            ),
            "rows": probe_rows,
        }

    tilt_control = tilt_negative_control(runs, clean, seeds)

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
        "ledger_response_curves": ledger_response,
        "ledger_direction_probe": direction_probe,
        "negative_control_tilt": tilt_control,
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
