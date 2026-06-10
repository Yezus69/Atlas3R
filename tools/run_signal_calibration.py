"""GT-free signal calibration table (falsifier part 1).

Tracked twin of the original runs/_diag/sweep_signals.py diagnostic.

For every cached backbone-artifact configuration with a measured M2 reference,
run the candidate pipeline (refine -> visibility -> static/dynamic -> posterior
-> fuse) and record:
  - the full GT-free signal vector (Stage 0 evidence mass, Stage 1 gravity,
    Stage 2 consistency, pre-refine residuals), and
  - the GT labels (camera-center Sim(3) RMSE vs measured; band3d occ_iou).

phone_room rows carry signals only (no GT -- never fabricated).

Then test the red-team's kill criterion: does ANY Stage-2 signal predict RMSE
after partialling out coverage (evidence mass)? With n~9 GT rows spanning
3 scenes x 2 backbones this is an honest screen, not a proof -- the output
states its own sample size.

Output: runs/_diag/signal_calibration_table.json + a printed markdown table.
Pure read of cached artifacts; writes nothing under runs/teacher.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas3r.config import load_robot_envelope  # noqa: E402
from atlas3r.geometry_adapter import (  # noqa: E402
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from atlas3r.mapping import fuse_static_map  # noqa: E402
from atlas3r.refine import refine_scene  # noqa: E402
from atlas3r.scale import estimate_scale_posterior  # noqa: E402
from atlas3r.static_dynamic import infer_static_dynamic  # noqa: E402
from atlas3r.validation import (  # noqa: E402
    _evidence_mass_stage,
    _gravity_alignment_stage,
    _held_out_render_error,
)
from atlas3r.visibility import build_visibility_graph  # noqa: E402
from atlas3r import teacher as T  # noqa: E402

SPF = 4096  # static/dynamic samples per frame; same recipe for every row

# (asset, artifacts_dir, label, backbone) -- duplicates of teacher_artifacts
# (_room_ma, _desk_ma) are excluded; phone rows carry no GT.
CONFIGS = [
    ("reference_metric", "external/teacher_artifacts", "xyz_ma_14kf", "mapanything"),
    ("reference_metric", "external/_da3_artifacts_backup", "xyz_da3", "da3"),
    ("reference_metric_desk", "external/teacher_artifacts", "desk_ma_11kf", "mapanything"),
    ("reference_metric_desk", "external/_desk_ma_dense", "desk_ma_dense", "mapanything"),
    ("reference_metric_desk", "external/_desk_da3", "desk_da3", "da3"),
    ("reference_metric_room", "external/teacher_artifacts", "room_ma_13kf", "mapanything"),
    ("reference_metric_room", "external/_room_ma24", "room_ma_24kf", "mapanything"),
    ("reference_metric_room", "external/_room_ma48", "room_ma_48kf", "mapanything"),
    ("reference_metric_room", "external/_room_ma48m", "room_ma_48kf_me", "mapanything"),
    ("phone_room", "external/teacher_artifacts", "phone_ma", "mapanything"),
    ("phone_room", "external/_da3_artifacts_backup", "phone_da3", "da3"),
]

MEASURED_SCENES = {"reference_metric", "reference_metric_desk", "reference_metric_room"}


def measured_field_for(asset: str, envelope, cache: dict):
    """Fuse the measured M2 packets once per scene (apply_fusion_policy=False:
    the GT yardstick never gets the candidate-only policy)."""
    if asset in cache:
        return cache[asset]
    measured, _ = load_measured_packets_from_m2(asset, ROOT)
    m2_report = ROOT / "runs/m2" / f"{asset}_measured_reference_report.json"
    base_ev = T._read_m2_scale_evidence({"report_path": str(m2_report)})
    meas_post, _ = estimate_scale_posterior(measured, base_ev)
    _, _, _, meas_field, _ = fuse_static_map(
        measured, meas_post, static_dynamic_states=None,
        envelope=envelope, apply_fusion_policy=False,
    )
    cache[asset] = (measured, meas_field)
    return cache[asset]


def eval_config(asset: str, adir: str, label: str, backbone: str, envelope, meas_cache):
    t0 = time.time()
    mono, grep = load_geometry_artifacts(asset, ROOT, artifacts_dir=adir)
    soft = grep.get("_scale_evidence", [])
    refined, refine_report = refine_scene(mono, fix_global_scale=bool(soft))

    _, vis_report = build_visibility_graph(refined)
    overall = vis_report.get("overall", {}) if isinstance(vis_report, dict) else {}

    masks_dir = ROOT / adir / asset / "masks"
    states, _ = infer_static_dynamic(
        refined, masks_dir=masks_dir, max_samples_per_frame=SPF
    )
    post, _ = estimate_scale_posterior(refined, soft)
    _, _, _, cand_field, map_report = fuse_static_map(
        refined, post, static_dynamic_states=states,
        envelope=envelope, apply_fusion_policy=True,
    )
    held_out, held_out_note = _held_out_render_error(refined, None)

    stage0 = _evidence_mass_stage(vis_report, True)
    stage1 = _gravity_alignment_stage(map_report, True)
    rb = refine_report.get("residual_summary_before", {}) or {}
    ra = refine_report.get("residual_summary_after", {}) or {}
    floor = map_report.get("floor", {}) or {}

    row = {
        "label": label,
        "asset": asset,
        "artifacts_dir": adir,
        "backbone": backbone,
        "n_keyframes": len(mono),
        # --- GT-free signal vector ---
        "signals": {
            "median_reprojection_inbounds_ratio": overall.get("median_reprojection_inbounds_ratio"),
            "depth_residual_edge_fraction": stage0["depth_residual_edge_fraction"],
            "mean_confidence_weight": overall.get("mean_confidence_weight"),
            "median_log_depth_residual_visibility": overall.get("median_log_depth_residual"),
            "prerefine_median_log_depth_residual": rb.get("median_log_depth_residual"),
            "prerefine_p90_log_depth_residual": rb.get("p90_log_depth_residual"),
            "postrefine_median_log_depth_residual": ra.get("median_log_depth_residual"),
            "postrefine_p90_log_depth_residual": ra.get("p90_log_depth_residual"),
            "refine_cost_reduction_fraction": refine_report.get("cost_reduction_fraction"),
            "held_out_render_error": held_out,
            "held_out_note": held_out_note,
            "map_free_space_contradiction_rate": map_report.get("free_space_contradiction_rate"),
            "unknown_fraction": map_report.get("unknown_fraction"),
            "free_fraction": map_report.get("free_fraction"),
            "occupied_fraction": map_report.get("occupied_fraction"),
            "mean_map_confidence": map_report.get("mean_map_confidence"),
            "floor_inlier_ratio": floor.get("inlier_ratio"),
            "floor_up_alignment_applied": floor.get("up_alignment_applied"),
            "scale_std": getattr(post, "scale_std", None),
        },
        "gate_cascade": {
            "stage0_would_reject": stage0["would_reject"],
            "stage1_would_reject": stage1["would_reject"],
        },
        # --- GT labels (null when no measured reference; never fabricated) ---
        "gt": None,
        "runtime_s": None,
    }

    if asset in MEASURED_SCENES:
        measured, meas_field = measured_field_for(asset, envelope, meas_cache)
        cam = T._compare_candidate_to_measured(refined, measured)
        ba = T._band3d_agreement(cand_field, meas_field, refined, measured)
        row["gt"] = {
            "cam_sim3_rmse_m": cam.get("trajectory_rmse_m_after_alignment"),
            "cam_sim3_median_m": cam.get("trajectory_median_m_after_alignment"),
            "cam_sim3_max_m": cam.get("trajectory_max_m_after_alignment"),
            "estimated_scale_monocular_to_measured": cam.get("estimated_scale_monocular_to_measured"),
            "band3d_status": ba.get("status"),
            "band3d_occ_iou": ba.get("occupied_static_iou"),
            "band3d_fsc": ba.get("free_space_contradiction_rate"),
            "band3d_coverage": ba.get("coverage_of_measured_band"),
        }
    row["runtime_s"] = round(time.time() - t0, 1)
    return row


def spearman(x, y):
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def partial_spearman(x, y, z):
    """Rank-based partial correlation of x,y controlling z (the coverage
    confound). Residualize ranks of x and y on ranks of z, then correlate."""
    import numpy as np

    def ranks(v):
        v = np.asarray(v, float)
        return np.argsort(np.argsort(v)).astype(float)

    rx, ry, rz = ranks(x), ranks(y), ranks(z)

    def resid(a, b):
        b1 = np.stack([b, np.ones_like(b)], axis=1)
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    denom = float(np.sqrt((ex ** 2).sum() * (ey ** 2).sum()))
    return float((ex * ey).sum() / denom) if denom > 0 else float("nan")


def analyze(rows):
    gt_rows = [r for r in rows if r["gt"] and isinstance(r["gt"].get("cam_sim3_rmse_m"), (int, float))]
    n = len(gt_rows)
    rmse = [r["gt"]["cam_sim3_rmse_m"] for r in gt_rows]
    coverage_proxy = [r["signals"]["median_reprojection_inbounds_ratio"] or 0.0 for r in gt_rows]

    signal_keys = [
        "prerefine_p90_log_depth_residual",
        "prerefine_median_log_depth_residual",
        "postrefine_p90_log_depth_residual",
        "held_out_render_error",
        "map_free_space_contradiction_rate",
        "unknown_fraction",
        "mean_confidence_weight",
        "mean_map_confidence",
        "floor_inlier_ratio",
        "refine_cost_reduction_fraction",
        "depth_residual_edge_fraction",
        "median_reprojection_inbounds_ratio",
    ]
    out = {
        "n_gt_rows": n,
        "honesty_note": (
            f"n={n} labeled configurations spanning 3 scenes x 2 backbones; rank "
            "correlations at this sample size are a SCREEN, not a calibration. "
            "No threshold derives authority from this table alone."
        ),
        "per_signal": {},
    }
    for key in signal_keys:
        vals = [r["signals"].get(key) for r in gt_rows]
        if any(not isinstance(v, (int, float)) for v in vals):
            out["per_signal"][key] = {"status": "missing_values"}
            continue
        out["per_signal"][key] = {
            "spearman_vs_rmse": round(spearman(vals, rmse), 3),
            "partial_spearman_vs_rmse_controlling_inbounds": (
                round(partial_spearman(vals, rmse, coverage_proxy), 3)
            ),
        }
    return out


def main():
    envelope, _ = load_robot_envelope(root=ROOT)
    meas_cache: dict = {}
    rows = []
    for asset, adir, label, backbone in CONFIGS:
        print(f"[sweep_signals] {label} ({asset} @ {adir}) ...", flush=True)
        try:
            row = eval_config(asset, adir, label, backbone, envelope, meas_cache)
        except Exception as exc:  # record the failure honestly, keep going
            row = {
                "label": label, "asset": asset, "artifacts_dir": adir,
                "backbone": backbone, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-3:],
            }
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("label", "runtime_s", "gt")}, default=str), flush=True)

    analysis = analyze(rows)
    table = {
        "recipe": {
            "refine": "refine_scene(fix_global_scale=bool(soft_evidence))",
            "static_dynamic_samples_per_frame": SPF,
            "fusion": "apply_fusion_policy=True (candidate policy); measured GT fused with policy OFF",
            "identical_for_every_row": True,
        },
        "rows": rows,
        "analysis": analysis,
    }
    out_path = ROOT / "runs/_diag/signal_calibration_table.json"
    out_path.write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(f"\n[sweep_signals] wrote {out_path}")

    # Markdown summary
    print("\n| config | kf | backbone | inbounds | pre_p90 | held_out | fsc | floor | RMSE_m | occ_iou |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") == "failed":
            print(f"| {r['label']} | - | {r['backbone']} | FAILED: {r['error'][:60]} |")
            continue
        s, g = r["signals"], r.get("gt") or {}

        def fmt(v, p=3):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "--"

        print(
            f"| {r['label']} | {r['n_keyframes']} | {r['backbone']} "
            f"| {fmt(s['median_reprojection_inbounds_ratio'])} | {fmt(s['prerefine_p90_log_depth_residual'])} "
            f"| {fmt(s['held_out_render_error'])} | {fmt(s['map_free_space_contradiction_rate'])} "
            f"| {fmt(s['floor_inlier_ratio'])} | {fmt(g.get('cam_sim3_rmse_m'), 4)} | {fmt(g.get('band3d_occ_iou'))} |"
        )
    print("\nAnalysis (screen, not calibration):")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
