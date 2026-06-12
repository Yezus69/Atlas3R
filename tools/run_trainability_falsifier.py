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

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atlas3r import teacher as T  # noqa: E402

SPF = 4096
FREE, OCC, MOV = 0, 1, 2
MATCH_MAX_M = 0.20
TAUS = (0.05, 0.10)

SCENES = [
    ("reference_metric", "decision"),
    ("reference_metric_desk", "reportage"),
]


def fields_for(asset: str, envelope):
    from atlas3r.geometry_adapter import load_geometry_artifacts, load_measured_packets_from_m2
    from atlas3r.mapping import fuse_static_map
    from atlas3r.refine import refine_scene
    from atlas3r.scale import estimate_scale_posterior
    from atlas3r.static_dynamic import infer_static_dynamic

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


def centers_from_packets(packets, field: dict) -> dict[int, np.ndarray]:
    R_up = np.asarray(field.get("R_up", np.eye(3)), dtype=np.float64)

    def center(packet):
        Tm = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
        return Tm[:3, 3]

    return {int(packet.frame_id): R_up @ center(packet) for packet in packets}


def load_trajectory_centers(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for frame in data.get("frames", []):
        center = frame.get("camera_center_world")
        if center is None:
            Tm = np.asarray(frame["T_world_camera"], dtype=np.float64).reshape((4, 4))
            center = Tm[:3, 3]
        out[int(frame["frame_id"])] = np.asarray(center, dtype=np.float64)
    return out


def load_comparison_field(run_dir: Path) -> dict:
    comparison = run_dir / "comparison_field.npz"
    if comparison.exists():
        with np.load(comparison, allow_pickle=False) as z:
            return {
                "class": np.asarray(z["class"], dtype=np.int8),
                "touched": np.asarray(z["touched"], dtype=bool),
                "origin": np.asarray(z["origin"], dtype=np.float64),
                "voxel": float(np.asarray(z["voxel"]).item()),
                "dims": tuple(int(v) for v in np.asarray(z["dims"]).tolist()),
                "floor_axis": int(np.asarray(z["floor_axis"]).item()),
                "band_min_m": float(np.asarray(z["band_min_m"]).item()),
                "band_max_m": float(np.asarray(z["band_max_m"]).item()),
                "floor_normal": tuple(float(v) for v in np.asarray(z["floor_normal"]).tolist()),
                "R_up": np.asarray(z["R_up"], dtype=np.float64),
                "up_aligned": bool(np.asarray(z["up_aligned"]).item()),
                "alignment_path": str(np.asarray(z["alignment_path"]).item()),
            }

    exported = run_dir / "voxel_occupancy_3d.npz"
    if not exported.exists():
        raise FileNotFoundError(f"missing comparison field export under {run_dir}")
    with np.load(exported, allow_pickle=False) as z:
        p_free = np.asarray(z["P_free"], dtype=np.float64)
        p_occ = np.asarray(z["P_occupied_static"], dtype=np.float64)
        p_mov = np.asarray(z["P_movable_static"], dtype=np.float64)
        p_dyn = np.asarray(z["P_dynamic"], dtype=np.float64)
        surf = p_occ + p_mov + p_dyn
        surf_pos = surf > 0.0
        free_only = (~surf_pos) & (p_free > 0.0)
        surf_cls = np.where(
            (p_occ >= p_mov) & (p_occ >= p_dyn),
            OCC,
            np.where(p_mov >= p_dyn, MOV, 3),
        )
        cls = np.full(p_free.shape, 4, dtype=np.int8)
        cls = np.where(free_only, np.int8(FREE), cls)
        cls = np.where(surf_pos, surf_cls.astype(np.int8), cls)
        return {
            "class": cls,
            "touched": surf_pos | (p_free > 0.0),
            "origin": np.asarray(z["origin_world"], dtype=np.float64),
            "voxel": float(np.asarray(z["voxel_size_m"]).item()),
            "dims": tuple(int(v) for v in p_free.shape),
            "floor_axis": int(np.asarray(z["floor_axis"]).item()),
            "band_min_m": float(np.asarray(z["band_min_m"]).item()),
            "band_max_m": float(np.asarray(z["band_max_m"]).item()),
            "floor_normal": (0.0, 0.0, 1.0),
            "R_up": np.eye(3),
            "up_aligned": False,
            "alignment_path": "band_export_fallback",
        }


def find_asset_run_dir(asset: str, teacher_dirs: list[Path]) -> Path:
    for teacher_dir in teacher_dirs:
        candidate = teacher_dir / asset
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"missing teacher export for {asset} in "
        + ", ".join(str(path) for path in teacher_dirs)
    )


def analyze_from_exports(asset: str, role: str, teacher_dirs: list[Path]) -> dict:
    asset_dir = find_asset_run_dir(asset, teacher_dirs)
    measured_dir = asset_dir / "measured_baseline"
    cf = load_comparison_field(asset_dir)
    mf = load_comparison_field(measured_dir)
    cand_by = load_trajectory_centers(asset_dir / "camera_trajectory.json")
    meas_by = load_trajectory_centers(measured_dir / "camera_trajectory.json")
    out = analyze_fields(asset, role, cf, mf, cand_by, meas_by)
    out["input_source"] = "teacher_exports"
    out["teacher_export_dir"] = str(asset_dir)
    return out


def analyze_recomputed(asset: str, role: str, envelope) -> dict:
    refined, cf, measured, mf = fields_for(asset, envelope)
    cand_by = centers_from_packets(refined, cf)
    meas_by = centers_from_packets(measured, mf)
    out = analyze_fields(asset, role, cf, mf, cand_by, meas_by)
    out["input_source"] = "recomputed"
    return out


def analyze_fields(
    asset: str,
    role: str,
    cf: dict,
    mf: dict,
    cand_by: dict[int, np.ndarray],
    meas_by: dict[int, np.ndarray],
) -> dict:

    # --- alignment, identical math to _band3d_agreement (s=1 variant) ---
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


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", default="runs/teacher")
    parser.add_argument(
        "--fallback-teacher-dir",
        action="append",
        default=[],
        help="additional teacher export root to search for scenes missing from --teacher-dir",
    )
    parser.add_argument(
        "--out-path",
        default="runs/_diag/trainability_falsifier_results.json",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--recompute-missing",
        action="store_true",
        help="fall back to the slow reconstruction path if teacher exports are missing",
    )
    args = parser.parse_args(argv)

    teacher_dirs = [ROOT / args.teacher_dir] + [ROOT / p for p in args.fallback_teacher_dir]
    results = []
    envelope = None
    for asset, role in SCENES:
        if args.recompute:
            if envelope is None:
                from atlas3r.config import load_robot_envelope

                envelope, _prov = load_robot_envelope(root=ROOT)
            results.append(analyze_recomputed(asset, role, envelope))
            continue
        try:
            results.append(analyze_from_exports(asset, role, teacher_dirs))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if not args.recompute_missing:
                raise
            if envelope is None:
                from atlas3r.config import load_robot_envelope

                envelope, _prov = load_robot_envelope(root=ROOT)
            out = analyze_recomputed(asset, role, envelope)
            out["export_fallback_reason"] = str(exc)
            results.append(out)

    out_path = ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
