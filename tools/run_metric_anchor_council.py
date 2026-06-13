"""Run the metric anchor council on canonical cached artifacts.

This is a read/evaluate diagnostic. It does not run external models and does not
use measured GT to construct candidate outputs. For measured scenes, GT is read
only after the council estimate is produced, to check scale-error coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas3r import teacher as T  # noqa: E402
from atlas3r.geometry_adapter import (  # noqa: E402
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from atlas3r.m1 import load_canonical_assets  # noqa: E402
from atlas3r.metric_council import run_metric_anchor_council  # noqa: E402
from atlas3r.metric_council import (  # noqa: E402
    DEFAULT_EXTRA_ANCHORS,
    _camera_height_floor_prior_anchor,
    load_registered_pose_frames,
)
from atlas3r.refine import refine_scene  # noqa: E402
from atlas3r.scale import estimate_scale_posterior  # noqa: E402

SCENES = (
    ("reference_metric", "calibration"),
    ("reference_metric_desk", "calibration"),
    ("reference_metric_room", "held_out"),
    ("phone_room", "target_no_gt"),
    ("phone_room_loop", "target_ruler_check"),
)
HEIGHT_FALSIFIER_SCENES = (
    ("reference_metric", "calibration"),
    ("reference_metric_room", "held_out"),
    ("reference_metric_desk", "floor_unavailable_control"),
    ("phone_room_loop", "target_ruler_check"),
)
MEASURED = {"reference_metric", "reference_metric_desk", "reference_metric_room"}
MIN_CALIBRATION_SCENES_FOR_PROMOTION = 8
PHONE_ROOM_LOOP_RULER_SCALE = 3.44


def _consensus(report: dict) -> dict:
    c = report.get("consensus")
    return c if isinstance(c, dict) else {}


def _scale_error(
    asset: str,
    refined,
    *,
    measured_assets: set[str],
    m2_dirs: Sequence[str | Path],
) -> dict | None:
    if asset not in measured_assets:
        return None
    last_report: dict | None = None
    measured = []
    loaded_dir: str | Path | None = None
    for m2_dir in m2_dirs:
        measured, loader_report = load_measured_packets_from_m2(asset, ROOT, m2_dir=m2_dir)
        last_report = loader_report
        if measured:
            loaded_dir = m2_dir
            break
    if not measured:
        return {"status": "missing_measured_packets", "loader_report": last_report}
    comp = T._compare_candidate_to_measured(refined, measured)
    true_scale = comp.get("estimated_scale_monocular_to_measured")
    if not isinstance(true_scale, (int, float)) or true_scale <= 0.0:
        return {"status": "true_scale_not_computable", "camera_comparison": comp}
    return {
        "status": "computed",
        "m2_dir": str(loaded_dir),
        "true_scale_from_camera_sim3": float(true_scale),
        "camera_comparison": comp,
    }


def _teacher_map_inputs(
    asset: str,
    *,
    teacher_report_dir: str | Path = "runs/teacher",
) -> tuple[dict | None, object | None, dict]:
    path = ROOT / teacher_report_dir / f"{asset}_teacher_report.json"
    if not path.exists():
        return None, None, {
            "status": "missing_teacher_report",
            "path": str(path),
            "blockers": (f"missing_teacher_report:{asset}",),
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, {
            "status": "invalid_teacher_report",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "blockers": (f"invalid_teacher_report:{asset}",),
        }
    map_report = report.get("map_occupancy_status")
    if not isinstance(map_report, dict):
        return None, None, {
            "status": "missing_map_occupancy_status",
            "path": str(path),
            "blockers": (f"missing_map_occupancy_status:{asset}",),
        }
    floor = map_report.get("floor")
    if not isinstance(floor, dict):
        return None, map_report.get("floor_align_rotation"), {
            "status": "missing_floor_report",
            "path": str(path),
            "blockers": (f"missing_floor_report:{asset}",),
        }
    floor_payload = dict(floor)
    up = map_report.get("up_alignment")
    if isinstance(up, dict):
        floor_payload["up_alignment"] = up
        export = up.get("export_alignment")
        if isinstance(export, dict) and "export_alignment" not in floor_payload:
            floor_payload["export_alignment"] = export
    return floor_payload, map_report.get("floor_align_rotation"), {
        "status": "loaded",
        "path": str(path),
        "floor_status": {
            "up_alignment_applied": floor_payload.get("up_alignment_applied"),
            "method": floor_payload.get("method"),
            "inlier_ratio": floor_payload.get("inlier_ratio"),
        },
        "blockers": (),
    }


def _row(
    asset: str,
    split: str,
    *,
    candidate_artifacts_dir: str | Path,
    extra_anchors: Sequence[Mapping[str, object]] | None,
    measured_assets: set[str],
    m2_dirs: Sequence[str | Path],
    teacher_report_dir: str | Path,
) -> dict:
    t0 = time.time()
    mono, adapter = load_geometry_artifacts(
        asset, ROOT, artifacts_dir=candidate_artifacts_dir
    )
    if not mono:
        return {
            "asset": asset,
            "split": split,
            "status": "missing_candidate_artifact",
            "adapter_status": {k: v for k, v in adapter.items() if not str(k).startswith("_")},
        }
    soft = adapter.get("_scale_evidence", [])
    refined, refine_report = refine_scene(mono, fix_global_scale=bool(soft))
    floor_report, floor_rotation, floor_status = _teacher_map_inputs(
        asset,
        teacher_report_dir=teacher_report_dir,
    )
    pose_frames, pose_status = load_registered_pose_frames(
        asset, ROOT, artifacts_dir=candidate_artifacts_dir
    )
    evidence, council = run_metric_anchor_council(
        asset,
        refined,
        ROOT,
        primary_artifacts_dir=candidate_artifacts_dir,
        extra_anchors=extra_anchors,
        floor_report=floor_report,
        registered_pose_frames=pose_frames if pose_status.get("status") == "loaded" else None,
        floor_align_rotation=floor_rotation,
    )
    posterior, posterior_report = estimate_scale_posterior(refined, evidence)
    scale_check = _scale_error(
        asset,
        refined,
        measured_assets=measured_assets,
        m2_dirs=m2_dirs,
    )
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
        "height_anchor_inputs": {
            "floor_report": floor_status,
            "registered_poses": pose_status,
        },
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


def _cached_true_scale(asset: str) -> tuple[float | None, dict]:
    if asset == "phone_room_loop":
        return PHONE_ROOM_LOOP_RULER_SCALE, {
            "status": "ruler_measured_phone_room_loop",
            "true_scale": PHONE_ROOM_LOOP_RULER_SCALE,
            "provenance": "docs/band_obstacle_recall_evidence.md Phase 28 ruler scale",
        }
    path = ROOT / "runs/_diag/metric_anchor_council_report.json"
    if not path.exists():
        return None, {
            "status": "missing_metric_anchor_council_report",
            "path": str(path),
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "status": "invalid_metric_anchor_council_report",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    rows = report.get("rows")
    if not isinstance(rows, list):
        return None, {"status": "metric_anchor_council_rows_missing", "path": str(path)}
    for row in rows:
        if not isinstance(row, dict) or row.get("asset") != asset:
            continue
        check = row.get("gt_scale_check")
        if not isinstance(check, dict):
            continue
        true_scale = check.get("true_scale_from_camera_sim3")
        if isinstance(true_scale, (int, float)) and true_scale > 0.0:
            return float(true_scale), {
                "status": "cached_gt_camera_sim3_scale",
                "true_scale": float(true_scale),
                "path": str(path),
            }
    return None, {"status": "true_scale_not_found", "path": str(path)}


def _height_anchor_row(
    asset: str,
    split: str,
    *,
    candidate_artifacts_dir: str | Path = "external/teacher_artifacts",
    teacher_report_dir: str | Path = "runs/teacher",
) -> dict:
    floor_report, floor_rotation, floor_status = _teacher_map_inputs(
        asset,
        teacher_report_dir=teacher_report_dir,
    )
    pose_frames, pose_status = load_registered_pose_frames(
        asset, ROOT, artifacts_dir=candidate_artifacts_dir
    )
    anchor = _camera_height_floor_prior_anchor(
        asset,
        floor_report,
        pose_frames if pose_status.get("status") == "loaded" else None,
        floor_align_rotation=floor_rotation,
    )
    true_scale, truth = _cached_true_scale(asset)
    scale_pred = anchor.get("estimated_metric_scale")
    rel_unc = anchor.get("relative_scale_uncertainty")
    available = anchor.get("status") in {"accepted", "weak"}
    log_error = None
    error_pass = None
    coverage_pass = None
    if available and isinstance(scale_pred, (int, float)) and true_scale:
        log_error = abs(math.log(float(scale_pred) / float(true_scale)))
        error_pass = bool(log_error <= 0.25)
        coverage_pass = bool(isinstance(rel_unc, (int, float)) and log_error <= float(rel_unc))
    expected_unavailable = asset == "reference_metric_desk"
    return {
        "asset": asset,
        "split": split,
        "status": anchor.get("status"),
        "available": bool(available),
        "expected_unavailable": expected_unavailable,
        "availability_pass": (not available) if expected_unavailable else None,
        "anchor": anchor,
        "floor_report_input": floor_status,
        "registered_pose_input": pose_status,
        "true_scale": true_scale,
        "truth": truth,
        "scale_pred": scale_pred if isinstance(scale_pred, (int, float)) else None,
        "relative_uncertainty": rel_unc if isinstance(rel_unc, (int, float)) else None,
        "abs_log_scale_error": log_error,
        "error_pass": error_pass,
        "coverage_pass": coverage_pass,
    }


def _height_anchor_falsifier(
    *,
    candidate_artifacts_dir: str | Path = "external/teacher_artifacts",
    teacher_report_dir: str | Path = "runs/teacher",
) -> dict:
    rows = [
        _height_anchor_row(
            asset,
            split,
            candidate_artifacts_dir=candidate_artifacts_dir,
            teacher_report_dir=teacher_report_dir,
        )
        for asset, split in HEIGHT_FALSIFIER_SCENES
    ]
    available = [
        row for row in rows
        if row.get("available")
        and isinstance(row.get("abs_log_scale_error"), (int, float))
    ]
    error_pass = bool(available) and all(row.get("error_pass") is True for row in available)
    coverage_pass = bool(available) and all(row.get("coverage_pass") is True for row in available)
    enough_rows = len(available) >= 2
    desk_rows = [row for row in rows if row.get("asset") == "reference_metric_desk"]
    desk_unavailable_pass = bool(desk_rows and desk_rows[0].get("availability_pass") is True)
    passed = bool(error_pass and coverage_pass and enough_rows and desk_unavailable_pass)
    return {
        "module": "camera_height_floor_prior_falsifier",
        "pre_registered_rule": (
            "PASS requires abs(log(scale_pred/scale_true)) <= 0.25 on every "
            "available row, at least two available rows, and 0.20 relative "
            "uncertainty covering every observed error. Unavailable floor rows "
            "must stay unavailable."
        ),
        "rows": rows,
        "available_row_count": len(available),
        "minimum_available_rows": 2,
        "error_threshold_abs_log": 0.25,
        "coverage_relative_uncertainty": 0.20,
        "desk_unavailable_pass": desk_unavailable_pass,
        "verdict": "PASSED" if passed else "FAILED",
        "promotion_status": (
            "promotion_to_soft_evidence_candidate_allowed"
            if passed
            else "reportage_only_guard_remains"
        ),
    }


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _write_height_anchor_markdown(report: dict) -> Path:
    out = ROOT / "runs/_diag/codex_height_anchor_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera Height Floor Prior Falsifier",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        report["pre_registered_rule"],
        "",
        "| asset | status | h_med units | IQR | scale_pred | scale_true | abs log err | <=0.25 | covered by 0.20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["rows"]:
        anchor = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
        stat = anchor.get("height_statistic") if isinstance(anchor.get("height_statistic"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("asset")),
                    str(row.get("status")),
                    _fmt(stat.get("height_median_reconstruction_units")),
                    _fmt(stat.get("height_iqr_reconstruction_units")),
                    _fmt(row.get("scale_pred")),
                    _fmt(row.get("true_scale")),
                    _fmt(row.get("abs_log_scale_error")),
                    _fmt(row.get("error_pass")),
                    _fmt(row.get("coverage_pass")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Available rows: {report['available_row_count']} / {report['minimum_available_rows']} required.",
            f"Desk unavailable control: {_fmt(report['desk_unavailable_pass'])}.",
            f"Promotion status: `{report['promotion_status']}`.",
            "",
            "The anchor remains guarded by `_calibration_guard`; uncalibrated height agreement is reportage only.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _manifest_scenes(path: str | Path) -> tuple[tuple[str, str], ...]:
    assets = load_canonical_assets(path, repo_root=ROOT)
    return tuple((asset.asset_id, "calibration") for asset in assets)


def _manifest_measured_assets(path: str | Path) -> set[str]:
    assets = load_canonical_assets(path, repo_root=ROOT)
    return {
        asset.asset_id
        for asset in assets
        if getattr(asset.track_type, "value", str(asset.track_type)) == "reference_metric"
    }


def _dedupe_scenes(scenes: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for asset, split in scenes:
        if asset in seen:
            continue
        seen.add(asset)
        out.append((asset, split))
    return tuple(out)


def _parse_extra_anchor(raw: str) -> dict[str, object]:
    try:
        anchor_id, rest = raw.split("=", 1)
    except ValueError as exc:
        raise SystemExit(
            "--extra-anchor must be anchor_id=artifacts_dir[:anchor_family[:license_status]]"
        ) from exc
    parts = rest.split(":")
    artifacts_dir = parts[0]
    if not anchor_id.strip() or not artifacts_dir.strip():
        raise SystemExit("--extra-anchor requires non-empty anchor_id and artifacts_dir")
    return {
        "anchor_id": anchor_id.strip(),
        "artifacts_dir": artifacts_dir.strip(),
        "anchor_family": parts[1].strip() if len(parts) > 1 and parts[1].strip() else "learned_metric_depth",
        "license_status": parts[2].strip() if len(parts) > 2 and parts[2].strip() else "not_certified",
    }


def _extra_anchor_specs(raw_values: Sequence[str]) -> tuple[Mapping[str, object], ...] | None:
    if not raw_values:
        return None
    return tuple([*DEFAULT_EXTRA_ANCHORS, *(_parse_extra_anchor(v) for v in raw_values)])


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="use only --calibration-manifest scenes instead of appending to defaults",
    )
    parser.add_argument("--candidate-artifacts-dir", default="external/teacher_artifacts")
    parser.add_argument("--m2-dir", default="runs/m2")
    parser.add_argument(
        "--m2-fallback-dir",
        action="append",
        default=[],
        help="additional M2 output dir to search when measured packets are absent in --m2-dir",
    )
    parser.add_argument("--teacher-report-dir", default="runs/teacher")
    parser.add_argument(
        "--extra-anchor",
        action="append",
        default=[],
        help="anchor_id=artifacts_dir[:anchor_family[:license_status]]; may be repeated",
    )
    parser.add_argument("--report-json", default="runs/_diag/metric_anchor_council_report.json")
    parser.add_argument("--height-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    extra_anchors = _extra_anchor_specs(args.extra_anchor)
    m2_dirs = (args.m2_dir, *tuple(args.m2_fallback_dir))
    measured_assets = set(MEASURED)
    scenes: list[tuple[str, str]] = [] if args.manifest_only else list(SCENES)
    if args.calibration_manifest:
        scenes.extend(_manifest_scenes(args.calibration_manifest))
        measured_assets.update(_manifest_measured_assets(args.calibration_manifest))
    scenes_tuple = _dedupe_scenes(scenes)

    if args.height_only:
        height_report = _height_anchor_falsifier(
            candidate_artifacts_dir=args.candidate_artifacts_dir,
            teacher_report_dir=args.teacher_report_dir,
        )
        out = _write_height_anchor_markdown(height_report)
        print(f"[metric_anchor_council] wrote {out}")
        print(json.dumps({
            "verdict": height_report["verdict"],
            "promotion_status": height_report["promotion_status"],
            "available_row_count": height_report["available_row_count"],
        }, default=str), flush=True)
        return 0

    rows = []
    for asset, split in scenes_tuple:
        print(f"[metric_anchor_council] {asset} ({split}) ...", flush=True)
        try:
            row = _row(
                asset,
                split,
                candidate_artifacts_dir=args.candidate_artifacts_dir,
                extra_anchors=extra_anchors,
                measured_assets=measured_assets,
                m2_dirs=m2_dirs,
                teacher_report_dir=args.teacher_report_dir,
            )
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
            "candidate_artifacts_dir": args.candidate_artifacts_dir,
            "m2_dirs": m2_dirs,
            "calibration_manifest": args.calibration_manifest,
            "candidate_refine": "refine_scene(fix_global_scale=bool(adapter_soft_evidence))",
            "gt_use": "post-council evaluation only, never candidate construction",
            "extra_anchors": extra_anchors,
        },
        "rows": rows,
        "analysis": _analysis(rows),
        "scale_risk_certificate": _risk_certificate(rows),
        "height_anchor_falsifier": _height_anchor_falsifier(
            candidate_artifacts_dir=args.candidate_artifacts_dir,
            teacher_report_dir=args.teacher_report_dir,
        ),
    }
    out = ROOT / args.report_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    height_out = _write_height_anchor_markdown(report["height_anchor_falsifier"])
    print(f"[metric_anchor_council] wrote {out}")
    print(f"[metric_anchor_council] wrote {height_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
