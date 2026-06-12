"""Run the metric anchor council on canonical cached artifacts.

This is a read/evaluate diagnostic. It does not run external models and does not
use measured GT to construct candidate outputs. For measured scenes, GT is read
only after the council estimate is produced, to check scale-error coverage.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas3r import teacher as T  # noqa: E402
from atlas3r.geometry_adapter import (  # noqa: E402
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from atlas3r.metric_council import run_metric_anchor_council  # noqa: E402
from atlas3r.refine import refine_scene  # noqa: E402
from atlas3r.scale import estimate_scale_posterior  # noqa: E402

SCENES = (
    ("reference_metric", "calibration"),
    ("reference_metric_desk", "calibration"),
    ("reference_metric_room", "held_out"),
    ("phone_room", "target_no_gt"),
)
MEASURED = {"reference_metric", "reference_metric_desk", "reference_metric_room"}
MIN_CALIBRATION_SCENES_FOR_PROMOTION = 8


def _consensus(report: dict) -> dict:
    c = report.get("consensus")
    return c if isinstance(c, dict) else {}


def _scale_error(asset: str, refined) -> dict | None:
    if asset not in MEASURED:
        return None
    measured, _ = load_measured_packets_from_m2(asset, ROOT)
    if not measured:
        return {"status": "missing_measured_packets"}
    comp = T._compare_candidate_to_measured(refined, measured)
    true_scale = comp.get("estimated_scale_monocular_to_measured")
    if not isinstance(true_scale, (int, float)) or true_scale <= 0.0:
        return {"status": "true_scale_not_computable", "camera_comparison": comp}
    return {
        "status": "computed",
        "true_scale_from_camera_sim3": float(true_scale),
        "camera_comparison": comp,
    }


def _row(asset: str, split: str) -> dict:
    t0 = time.time()
    mono, adapter = load_geometry_artifacts(asset, ROOT, artifacts_dir="external/teacher_artifacts")
    if not mono:
        return {
            "asset": asset,
            "split": split,
            "status": "missing_candidate_artifact",
            "adapter_status": {k: v for k, v in adapter.items() if not str(k).startswith("_")},
        }
    soft = adapter.get("_scale_evidence", [])
    refined, refine_report = refine_scene(mono, fix_global_scale=bool(soft))
    evidence, council = run_metric_anchor_council(
        asset,
        refined,
        ROOT,
        primary_artifacts_dir="external/teacher_artifacts",
    )
    posterior, posterior_report = estimate_scale_posterior(refined, evidence)
    scale_check = _scale_error(asset, refined)
    cons = _consensus(council)
    scale_mean = cons.get("scale_mean")
    rel_unc = cons.get("relative_scale_uncertainty")
    if scale_check and scale_check.get("status") == "computed" and isinstance(scale_mean, (int, float)):
        import math

        true_scale = float(scale_check["true_scale_from_camera_sim3"])
        rel_err = abs(math.log(float(scale_mean) / true_scale))
        scale_check["relative_log_scale_error"] = rel_err
        scale_check["covered_by_predicted_uncertainty"] = (
            bool(isinstance(rel_unc, (int, float)) and rel_err <= float(rel_unc))
        )
        raw = cons.get("raw_uncalibrated_consensus") if isinstance(cons, dict) else None
        if isinstance(raw, dict) and isinstance(raw.get("scale_mean"), (int, float)):
            raw_rel = raw.get("relative_scale_uncertainty")
            raw_err = abs(math.log(float(raw["scale_mean"]) / true_scale))
            scale_check["raw_uncalibrated_consensus"] = {
                "scale_mean": raw.get("scale_mean"),
                "relative_scale_uncertainty": raw_rel,
                "relative_log_scale_error": raw_err,
                "covered_by_predicted_uncertainty": (
                    bool(isinstance(raw_rel, (int, float)) and raw_err <= float(raw_rel))
                ),
            }
    return {
        "asset": asset,
        "split": split,
        "status": "computed",
        "candidate_packets": len(mono),
        "refined_packets": len(refined),
        "refine_status": refine_report.get("status"),
        "council": council,
        "scale_evidence": [T._jsonable(e) if hasattr(T, "_jsonable") else str(e) for e in evidence],
        "posterior": {
            "status": posterior.metric_acceptance_status.value,
            "scale_mean": posterior.scale_mean,
            "scale_std": posterior.scale_std,
            "relative_scale_uncertainty": posterior.relative_scale_uncertainty,
            "report": posterior_report,
        },
        "gt_scale_check": scale_check,
        "runtime_s": round(time.time() - t0, 1),
    }


def _analysis(rows: list[dict]) -> dict:
    gt = [
        r for r in rows
        if isinstance(r.get("gt_scale_check"), dict)
        and r["gt_scale_check"].get("status") == "computed"
    ]
    covered = [
        r["gt_scale_check"].get("covered_by_predicted_uncertainty")
        for r in gt
    ]
    return {
        "gt_scene_count": len(gt),
        "coverage_count": int(sum(1 for v in covered if v is True)),
        "coverage_fraction": (sum(1 for v in covered if v is True) / len(covered)) if covered else None,
        "held_out_scenes": [r["asset"] for r in gt if r.get("split") == "held_out"],
        "honesty_note": (
            "small-n diagnostic, not a statistically strong calibration; "
            "threshold authority requires more held-out measured scenes"
        ),
    }


def _raw_consensus(consensus: dict) -> dict:
    raw = consensus.get("raw_uncalibrated_consensus")
    if isinstance(raw, dict):
        return raw
    return consensus if isinstance(consensus, dict) else {}


def _row_risk_signature(row: dict) -> dict:
    council = row.get("council")
    if not isinstance(council, dict):
        return {"status": "missing_council"}
    anchors = council.get("anchor_reports")
    if not isinstance(anchors, (list, tuple)):
        return {"status": "missing_anchor_reports"}
    usable = [
        a for a in anchors
        if isinstance(a, dict)
        and a.get("independent_metric_anchor") is True
        and a.get("verdict") in {"accepted", "weak"}
        and isinstance((a.get("residual_risk_signature") or {}).get("risk_score"), (int, float))
    ]
    if not usable:
        return {
            "status": "no_usable_independent_anchor",
            "independent_usable_anchor_count": 0,
            "self_audit_anchor_ids": [
                a.get("anchor_id") for a in anchors
                if isinstance(a, dict) and a.get("independent_metric_anchor") is False
            ],
        }
    risks = [float(a["residual_risk_signature"]["risk_score"]) for a in usable]
    consensus = _consensus(council)
    raw = _raw_consensus(consensus)
    return {
        "status": "computed",
        "independent_usable_anchor_count": len(usable),
        "risk_score": max(risks),
        "best_anchor_risk_score": min(risks),
        "anchor_ids": [str(a.get("anchor_id")) for a in usable],
        "anchor_families": sorted({str(a.get("anchor_family")) for a in usable}),
        "raw_consensus_status": raw.get("status"),
        "raw_scale_mean": raw.get("scale_mean"),
        "raw_relative_scale_uncertainty": raw.get("relative_scale_uncertainty"),
        "basis": (
            "max residual-risk score among independent usable anchors; lower is "
            "easier, but only measured calibration can convert it to scale authority"
        ),
    }


def _observed_raw_scale_error(row: dict) -> float | None:
    check = row.get("gt_scale_check")
    if not isinstance(check, dict) or check.get("status") != "computed":
        return None
    raw = check.get("raw_uncalibrated_consensus")
    if isinstance(raw, dict) and isinstance(raw.get("relative_log_scale_error"), (int, float)):
        return float(raw["relative_log_scale_error"])
    if isinstance(check.get("relative_log_scale_error"), (int, float)):
        return float(check["relative_log_scale_error"])
    return None


def _risk_certificate(rows: list[dict]) -> dict:
    signatures = {str(row.get("asset")): _row_risk_signature(row) for row in rows}
    calibration = []
    for row in rows:
        if row.get("split") != "calibration":
            continue
        sig = signatures.get(str(row.get("asset")), {})
        err = _observed_raw_scale_error(row)
        if sig.get("status") != "computed" or err is None:
            continue
        calibration.append({
            "asset": row.get("asset"),
            "risk_score": sig.get("risk_score"),
            "raw_scale_mean": sig.get("raw_scale_mean"),
            "raw_relative_scale_uncertainty": sig.get("raw_relative_scale_uncertainty"),
            "observed_log_scale_error": err,
        })

    row_certificates = []
    for row in rows:
        asset = str(row.get("asset"))
        sig = signatures.get(asset, {"status": "missing_signature"})
        if sig.get("status") != "computed":
            row_certificates.append({
                "asset": asset,
                "split": row.get("split"),
                "status": "no_transfer_authority",
                "reason": sig.get("status"),
                "risk_signature": sig,
            })
            continue
        risk = float(sig["risk_score"])
        supporting = [
            c for c in calibration
            if isinstance(c.get("risk_score"), (int, float))
            and float(c["risk_score"]) >= risk
        ]
        if not supporting:
            row_certificates.append({
                "asset": asset,
                "split": row.get("split"),
                "status": "outside_calibrated_residual_support",
                "risk_signature": sig,
                "calibration_support_count": 0,
            })
            continue
        envelope = max(float(c["observed_log_scale_error"]) for c in supporting)
        raw_unc = sig.get("raw_relative_scale_uncertainty")
        calibrated_unc = max(
            envelope,
            float(raw_unc) if isinstance(raw_unc, (int, float)) else 0.0,
        )
        measured_error = _observed_raw_scale_error(row)
        row_certificates.append({
            "asset": asset,
            "split": row.get("split"),
            "status": (
                "reportage_only_calibration_underpowered"
                if len(calibration) < MIN_CALIBRATION_SCENES_FOR_PROMOTION
                else "calibrated_uncertainty_candidate"
            ),
            "risk_signature": sig,
            "calibration_support_count": len(supporting),
            "supporting_calibration_assets": [c["asset"] for c in supporting],
            "monotone_error_envelope": calibrated_unc,
            "measured_raw_log_scale_error": measured_error,
            "covered_if_envelope_used": (
                bool(measured_error <= calibrated_unc) if measured_error is not None else None
            ),
        })

    return {
        "module": "residual_monotone_scale_risk_certificate",
        "mathematical_claim": (
            "Global metric scale is unobservable from monocular video alone. "
            "A learned anchor may vote only inside a measured residual-support "
            "envelope; the envelope is monotone in residual risk and otherwise "
            "returns no transfer authority."
        ),
        "calibration_scene_count": len(calibration),
        "minimum_calibration_scenes_for_promotion": MIN_CALIBRATION_SCENES_FOR_PROMOTION,
        "promotion_status": (
            "disabled_calibration_underpowered"
            if len(calibration) < MIN_CALIBRATION_SCENES_FOR_PROMOTION
            else "candidate_requires_held_out_coverage_audit"
        ),
        "calibration_rows": calibration,
        "row_certificates": row_certificates,
    }


def main() -> int:
    rows = []
    for asset, split in SCENES:
        print(f"[metric_anchor_council] {asset} ({split}) ...", flush=True)
        try:
            row = _row(asset, split)
        except Exception as exc:  # keep blockers explicit per scene
            row = {
                "asset": asset,
                "split": split,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-5:],
            }
        rows.append(row)
        summary = {
            "asset": row.get("asset"),
            "status": row.get("status"),
            "consensus": (row.get("council") or {}).get("consensus") if isinstance(row.get("council"), dict) else None,
            "gt_scale_check": row.get("gt_scale_check"),
        }
        print(json.dumps(summary, default=str), flush=True)

    report = {
        "module": "metric_anchor_council_diagnostic",
        "recipe": {
            "candidate_artifacts_dir": "external/teacher_artifacts",
            "candidate_refine": "refine_scene(fix_global_scale=bool(adapter_soft_evidence))",
            "gt_use": "post-council evaluation only, never candidate construction",
        },
        "rows": rows,
        "analysis": _analysis(rows),
        "scale_risk_certificate": _risk_certificate(rows),
    }
    out = ROOT / "runs/_diag/metric_anchor_council_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[metric_anchor_council] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
