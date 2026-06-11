"""Phase 16 trainability/collision-safety falsifier (pre-registered 27e416c).

Promoted from runs/_diag to tools/: dangerous_free@5cm is a tracked number
for every future recipe change (Phase 16 RESULTS verdict).

Measures, at the SCALE-HONEST alignment (rigid SE(3), s=1, same rotation path
as teacher._band3d_agreement):

1. consumption-semantics split of measured-solid band voxels in candidate
   bounds: solid_within_margin / dangerous_free / coverage_loss, protective
   precedence, at 5 cm (decision) and 10 cm (reportage);
2. signed displacement decomposition of judged candidate solids vs nearest
   measured solid (<= 0.20 m): radial (signed, along view dir from nearest
   candidate camera center; + = beyond true surface, - = pulled toward the
   camera) and lateral magnitude.

Decision scene: xyz (reference_metric, the accepted scene). desk = reportage.
Pure measurement: GT never touches the candidate path; nothing is tuned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atlas3r import teacher as T  # noqa: E402
from atlas3r.config import load_robot_envelope  # noqa: E402
from atlas3r.geometry_adapter import (  # noqa: E402
    load_geometry_artifacts,
    load_measured_packets_from_m2,
)
from atlas3r.mapping import fuse_static_map  # noqa: E402
from atlas3r.refine import refine_scene  # noqa: E402
from atlas3r.scale import estimate_scale_posterior  # noqa: E402
from atlas3r.static_dynamic import infer_static_dynamic  # noqa: E402

SPF = 4096
FREE, OCC, MOV = 0, 1, 2
MATCH_MAX_M = 0.20
TAUS = (0.05, 0.10)

SCENES = [
    ("reference_metric", "decision"),
    ("reference_metric_desk", "reportage"),
]


def fields_for(asset: str, envelope):
    mono, grep = load_geometry_artifacts(asset, ROOT, artifacts_dir="external/teacher_artifacts")
    soft = grep.get("_scale_evidence", [])
    refined, _ = refine_scene(mono, fix_global_scale=bool(soft))
    states, _ = infer_static_dynamic(
        refined, masks_dir=ROOT / "external/teacher_artifacts" / asset / "masks",
        max_samples_per_frame=SPF,
    )
    post, _ = estimate_scale_posterior(refined, soft)
    _, _, _, cand_field, _ = fuse_static_map(
        refined, post, static_dynamic_states=states,
        envelope=envelope, apply_fusion_policy=True,
    )
    measured, _ = load_measured_packets_from_m2(asset, ROOT)
    m2_report = ROOT / "runs/m2" / f"{asset}_measured_reference_report.json"
    base_ev = T._read_m2_scale_evidence({"report_path": str(m2_report)})
    meas_post, _ = estimate_scale_posterior(measured, base_ev)
    _, _, _, meas_field, _ = fuse_static_map(
        measured, meas_post, static_dynamic_states=None,
        envelope=envelope, apply_fusion_policy=False,
    )
    return refined, cand_field, measured, meas_field


def analyze(asset: str, role: str, envelope) -> dict:
    refined, cf, measured, mf = fields_for(asset, envelope)

    # --- alignment, identical math to _band3d_agreement (s=1 variant) ---
    R_up_c = np.asarray(cf.get("R_up", np.eye(3)), dtype=np.float64)
    R_up_m = np.asarray(mf.get("R_up", np.eye(3)), dtype=np.float64)

    def center(p):
        Tm = np.asarray(p.T_world_camera, dtype=np.float64).reshape((4, 4))
        return Tm[:3, 3]

    cand_by = {int(p.frame_id): R_up_c @ center(p) for p in refined}
    meas_by = {int(p.frame_id): R_up_m @ center(p) for p in measured}
    common = sorted(set(cand_by) & set(meas_by))
    if len(common) < 3:
        return {"asset": asset, "status": "insufficient_overlap"}
    src = np.asarray([cand_by[f] for f in common])
    dst = np.asarray([meas_by[f] for f in common])
    s, r_sim, _t, _a, _rm = T._umeyama_align(src, dst, np)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    n_c = np.asarray(cf.get("floor_normal", (0, 0, 1)), dtype=np.float64)
    n_m = np.asarray(mf.get("floor_normal", (0, 0, 1)), dtype=np.float64)
    nc_mapped = r_sim @ (n_c / (np.linalg.norm(n_c) + 1e-12))
    n_m_u = n_m / (np.linalg.norm(n_m) + 1e-12)
    if float(np.dot(nc_mapped, n_m_u)) < 0:
        n_m_u = -n_m_u
    R = T._rotation_align(nc_mapped, n_m_u, np) @ r_sim
    t1 = mu_dst - R @ mu_src  # rigid SE(3), scale fixed at 1 (scale-honest)

    # --- measured band population (same construction as _band3d_agreement) ---
    m_cls = np.asarray(mf["class"]).ravel()
    m_touch = np.asarray(mf["touched"]).ravel()
    m_org = np.asarray(mf["origin"], dtype=np.float64)
    m_v, m_dims, m_fa = float(mf["voxel"]), mf["dims"], int(mf["floor_axis"])
    bmin, bmax = float(mf["band_min_m"]), float(mf["band_max_m"])
    ax = [m_org[i] + (np.arange(m_dims[i]) + 0.5) * m_v for i in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    in_band = (centers[:, m_fa] >= bmin) & (centers[:, m_fa] < bmax)
    sel = in_band & m_touch
    pts_meas = centers[sel]
    meas_sel = m_cls[sel]

    # measured -> candidate frame at s=1
    p_c = (R.T @ (pts_meas - t1[None, :]).T).T
    cf_org = np.asarray(cf["origin"], dtype=np.float64)
    cf_v = float(cf["voxel"])
    cd = [int(d) for d in cf["dims"]]
    ci = np.floor((p_c - cf_org[None, :]) / cf_v).astype(np.int64)
    in_b = ((ci >= 0) & (ci < np.asarray(cd)[None, :])).all(axis=1)

    cf_cls = np.asarray(cf["class"])
    cf_touch = np.asarray(cf["touched"])
    meas_solid = ((meas_sel == OCC) | (meas_sel == MOV)) & in_b
    n_pop = int(np.count_nonzero(meas_solid))

    from scipy.spatial import cKDTree

    cand_solid_idx = np.argwhere(cf_touch & ((cf_cls == OCC) | (cf_cls == MOV)))
    cand_free_idx = np.argwhere(cf_touch & (cf_cls == FREE))
    solid_centers = cf_org[None, :] + (cand_solid_idx + 0.5) * cf_v
    free_centers = cf_org[None, :] + (cand_free_idx + 0.5) * cf_v
    q = p_c[meas_solid]
    d_solid = (
        cKDTree(solid_centers).query(q)[0] if solid_centers.shape[0] else np.full(q.shape[0], np.inf)
    )
    d_free = (
        cKDTree(free_centers).query(q)[0] if free_centers.shape[0] else np.full(q.shape[0], np.inf)
    )

    out = {
        "asset": asset, "role": role, "status": "computed",
        "alignment": "rigid_se3_scale_fixed_1",
        "umeyama_scale_reportage": float(s),
        "measured_solid_in_bounds": n_pop,
        "candidate_solid_observed": int(cand_solid_idx.shape[0]),
        "consumption_semantics": {},
    }
    for tau in TAUS:
        solid_ok = d_solid <= tau
        danger = (~solid_ok) & (d_free <= tau)
        cov_loss = (~solid_ok) & (~danger)
        key = f"{tau*100:g}cm"
        out["consumption_semantics"][key] = {
            "solid_within_margin": float(np.mean(solid_ok)) if n_pop else None,
            "dangerous_free": float(np.mean(danger)) if n_pop else None,
            "coverage_loss": float(np.mean(cov_loss)) if n_pop else None,
        }

    # --- signed displacement decomposition (judged candidate solids) ---
    q_meas = (R @ solid_centers.T).T + t1[None, :]
    in_slab = (q_meas[:, m_fa] >= bmin) & (q_meas[:, m_fa] < bmax)
    d_obs = (
        cKDTree(pts_meas).query(q_meas)[0] if pts_meas.shape[0] else np.full(q_meas.shape[0], np.inf)
    )
    judged = in_slab & (d_obs <= T.BAND_MATCH_TOLERANCE_M)
    meas_solid_pts = pts_meas[(meas_sel == OCC) | (meas_sel == MOV)]
    lab = q_meas[judged]
    if meas_solid_pts.shape[0] and lab.shape[0]:
        d_match, j_match = cKDTree(meas_solid_pts).query(lab)
        ok = d_match <= MATCH_MAX_M
        delta = lab[ok] - meas_solid_pts[j_match[ok]]  # label - truth (metric frame)
        cam_meas = (R @ np.asarray([cand_by[f] for f in common]).T).T + t1[None, :]
        nearest_cam = cam_meas[cKDTree(cam_meas).query(lab[ok])[1]]
        view = lab[ok] - nearest_cam
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        radial = np.einsum("ij,ij->i", delta, view)
        lateral = np.linalg.norm(delta - radial[:, None] * view, axis=1)
        out["signed_displacement"] = {
            "judged_candidate_solids": int(lab.shape[0]),
            "matched_within_0p20m": int(np.count_nonzero(ok)),
            "match_rate": float(np.mean(ok)),
            "mean_signed_radial_m_BIAS": float(np.mean(radial)),
            "std_radial_m_VARIANCE": float(np.std(radial)),
            "median_abs_radial_m": float(np.median(np.abs(radial))),
            "median_lateral_m": float(np.median(lateral)),
            "radial_sign_convention": "positive_beyond_true_surface_negative_toward_camera",
        }
    else:
        out["signed_displacement"] = {"status": "no_matchable_solids"}
    return out


def main():
    envelope, _prov = load_robot_envelope(root=ROOT)
    results = [analyze(a, role, envelope) for a, role in SCENES]
    out_path = ROOT / "runs/_diag/trainability_falsifier_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
