"""Independent epipolar pose auditor (the "independence" principle of the
GT-free acceptance cascade, ARCHITECTURE.md Module 11).

Audits the reconstruction's relative camera poses with a DIFFERENT algorithm
class than the geometry backbone: classic sparse-feature two-view geometry
(ORB + ratio-test matching + 5-point RANSAC essential matrix + cheirality
pose recovery). The audited quantities are gauge/scale-free:

- relative ROTATION deviation (degrees) between the audited two-view rotation
  and the reconstruction's relative rotation, per wide-baseline pair;
- relative TRANSLATION-DIRECTION deviation (degrees) -- direction only, scale
  is unobservable to a two-view auditor and is not claimed;
- auditor CYCLE residuals over frame triangles (composition of audited
  rotations around a closed loop must be identity) -- the auditor's own noise
  floor, computed without touching the reconstruction at all.

Independence bookkeeping (red-team requirement): the essential-matrix
decomposition needs camera intrinsics. When a MEASURED calibration exists
(e.g. the TUM `reference_metric_intrinsics_known.json`, instrument-derived,
independent of the backbone under audit) it is used and the report says
`intrinsics_source: measured_calibration`. When only backbone-derived
intrinsics exist, the auditor still runs but the report carries
`intrinsics_source: backbone_under_audit` -- a DEMOTED independence claim,
recorded, never hidden.

Abstention rules (evidence-mass principle applied to the auditor itself):
- a pair abstains below ``min_pair_inliers`` RANSAC inliers;
- the scene-level audit abstains below ``min_valid_pairs`` valid pairs, and an
  abstained audit is AUTHORITY LOSS, never a pass -- low-texture/blur scenes
  are exactly where the backbone fails too, so silence must not count as
  evidence of correctness.

``numpy``/``cv2`` are imported lazily inside functions (repo convention).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_MAX_PAIRS = 40
DEFAULT_MIN_PAIR_INLIERS = 30
DEFAULT_MIN_VALID_PAIRS = 15
ORB_FEATURES = 2000
RATIO_TEST = 0.75
RANSAC_THRESHOLD_PX = 1.0
RANSAC_CONFIDENCE = 0.999

KNOWN_INTRINSICS = {
    # The three freiburg1 scenes share the TUM fr1 measured calibration.
    "reference_metric": "external/teacher_artifacts/reference_metric_intrinsics_known.json",
    "reference_metric_desk": "external/teacher_artifacts/reference_metric_intrinsics_known.json",
    "reference_metric_room": "external/teacher_artifacts/reference_metric_intrinsics_known.json",
}


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _frame_files(asset_id: str, root: Path) -> list[Path]:
    """Natural-sorted frame file list; frame_id indexes into it (the same
    convention the backbone runners use)."""
    candidates = [root / "data" / asset_id / "rgb", root / "data" / asset_id]
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(
            [p for p in directory.iterdir()
             if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
            key=_natural_key,
        )
        if files:
            return files
    return []


def _intrinsics_for(asset_id: str, root: Path, packets: Sequence[Any], image_shape) -> tuple[Any, str]:
    """Camera matrix K at the audited image resolution + provenance label."""
    import numpy as np

    h_img, w_img = image_shape[:2]
    known_rel = KNOWN_INTRINSICS.get(asset_id)
    if known_rel:
        known_path = root / known_rel
        if known_path.is_file():
            data = json.loads(known_path.read_text(encoding="utf-8"))
            sx = w_img / float(data["width_px"])
            sy = h_img / float(data["height_px"])
            K = np.array([
                [data["fx"] * sx, 0.0, data["cx"] * sx],
                [0.0, data["fy"] * sy, data["cy"] * sy],
                [0.0, 0.0, 1.0],
            ])
            return K, "measured_calibration"

    # Fallback: backbone-derived camera model -- honesty-demoted independence.
    cam = packets[0].camera_model
    fx = float(getattr(cam, "fx", 0.0))
    fy = float(getattr(cam, "fy", 0.0))
    cx = float(getattr(cam, "cx", 0.0))
    cy = float(getattr(cam, "cy", 0.0))
    w_model = float(getattr(cam, "width_px", w_img)) or w_img
    h_model = float(getattr(cam, "height_px", h_img)) or h_img
    sx, sy = w_img / w_model, h_img / h_model
    K = np.array([
        [fx * sx, 0.0, cx * sx],
        [0.0, fy * sy, cy * sy],
        [0.0, 0.0, 1.0],
    ])
    return K, "backbone_under_audit"


def _rotation_angle_deg(R: Any, np: Any) -> float:
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _relative_pose_from_reconstruction(packet_i: Any, packet_j: Any, np: Any):
    """R_ji, t_ji such that X_cj = R_ji X_ci + t_ji (cam i -> cam j)."""
    Ti = np.asarray(packet_i.T_world_camera, dtype=np.float64).reshape(4, 4)
    Tj = np.asarray(packet_j.T_world_camera, dtype=np.float64).reshape(4, 4)
    Ri, ti = Ti[:3, :3], Ti[:3, 3]
    Rj, tj = Tj[:3, :3], Tj[:3, 3]
    R_ji = Rj.T @ Ri
    t_ji = Rj.T @ (ti - tj)
    return R_ji, t_ji


def _select_pairs(
    frame_ids: list[int],
    max_pairs: int,
    packets: Sequence[Any] | None = None,
) -> tuple[list[tuple[int, int]], str]:
    """Pair selection, in order of preference:

    1. CLAIMED-OVERLAP edges from the visibility graph: audit exactly the
       co-observations the reconstruction asserts. If the claim is true the
       images share matchable content; matched-but-deviating relative pose is
       caught drift; unmatched claimed overlap is itself suspicious. (Blind
       wide-baseline pairs starve the auditor on loop trajectories -- measured:
       room@48kf got 1/40 valid pairs under blind selection.)
    2. Fallback (no graph available): all (i<j) pairs sorted by frame
       separation descending with a uniform stride -- wide baselines
       prioritized, mid baselines represented.
    """
    if packets is not None:
        try:
            from .visibility import build_visibility_graph

            graph, _report = build_visibility_graph(packets)
            edges = list(getattr(graph, "temporal_edges", ()) or ()) + list(
                getattr(graph, "overlap_edges", ()) or ()
            )
            claimed = sorted(
                {
                    (min(int(e.source_frame_id), int(e.target_frame_id)),
                     max(int(e.source_frame_id), int(e.target_frame_id)))
                    for e in edges
                    if int(e.source_frame_id) != int(e.target_frame_id)
                },
                key=lambda p: (-(p[1] - p[0]), p[0]),
            )
            if claimed:
                if len(claimed) > max_pairs:
                    stride = len(claimed) / max_pairs
                    claimed = [claimed[int(k * stride)] for k in range(max_pairs)]
                return claimed, "visibility_graph_claimed_overlap_edges"
        except Exception:
            pass  # fall through to blind selection, recorded as such
    pairs = [(a, b) for ai, a in enumerate(frame_ids) for b in frame_ids[ai + 1:]]
    pairs.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
    if len(pairs) > max_pairs:
        stride = len(pairs) / max_pairs
        pairs = [pairs[int(k * stride)] for k in range(max_pairs)]
    return pairs, "blind_wide_baseline_stride"


def audit_scene(
    packets: Sequence[Any],
    asset_id: str,
    root: Path,
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    min_pair_inliers: int = DEFAULT_MIN_PAIR_INLIERS,
    min_valid_pairs: int = DEFAULT_MIN_VALID_PAIRS,
) -> dict[str, Any]:
    """Audit ``packets`` (the refined candidate) against two-view geometry
    recomputed from the RAW FRAMES. Returns a report; never raises on missing
    data -- missing inputs become explicit abstention statuses."""
    import cv2  # type: ignore
    import numpy as np

    base: dict[str, Any] = {
        "module": "epipolar_audit - independent two-view pose auditor",
        "asset_id": asset_id,
        "algorithm": "ORB + ratio-test + 5-point RANSAC essential + cheirality recoverPose",
        "audited_quantities": "relative rotation (deg), translation direction (deg); scale-free",
    }

    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    if len(ordered) < 3:
        return {**base, "status": "abstained_insufficient_packets",
                "authority": "none", "n_packets": len(ordered)}

    files = _frame_files(asset_id, root)
    if not files:
        return {**base, "status": "abstained_missing_frame_files",
                "authority": "none",
                "note": f"no rgb frames found under data/{asset_id}"}

    by_id = {int(p.frame_id): p for p in ordered}
    frame_ids = sorted(by_id)
    missing = [f for f in frame_ids if f >= len(files)]
    if missing:
        return {**base, "status": "abstained_frame_ids_out_of_range",
                "authority": "none", "missing_frame_ids": missing,
                "n_frame_files": len(files)}

    # Detect features once per frame.
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    features: dict[int, tuple[Any, Any]] = {}
    image_shape = None
    for fid in frame_ids:
        img = cv2.imread(str(files[fid]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        image_shape = img.shape
        kp, desc = orb.detectAndCompute(img, None)
        if desc is not None and len(kp) >= min_pair_inliers:
            features[fid] = (kp, desc)
    if len(features) < 3 or image_shape is None:
        return {**base, "status": "abstained_too_few_featureful_frames",
                "authority": "none", "featureful_frames": len(features)}

    K, intrinsics_source = _intrinsics_for(asset_id, root, ordered, image_shape)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pair_rows: list[dict[str, Any]] = []
    audited_rotations: dict[tuple[int, int], Any] = {}
    abstained = {"too_few_matches": 0, "too_few_inliers": 0, "recover_failed": 0,
                 "missing_features": 0}

    selected_pairs, pair_source = _select_pairs(sorted(features), max_pairs, ordered)
    base["pair_selection"] = pair_source

    for fid_i, fid_j in selected_pairs:
        if fid_i not in features or fid_j not in features:
            abstained["missing_features"] += 1
            continue
        kp1, d1 = features[fid_i]
        kp2, d2 = features[fid_j]
        knn = matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in (mn for mn in knn if len(mn) == 2)
                if m.distance < RATIO_TEST * n.distance]
        if len(good) < min_pair_inliers:
            abstained["too_few_matches"] += 1
            continue
        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC,
            prob=RANSAC_CONFIDENCE, threshold=RANSAC_THRESHOLD_PX,
        )
        if E is None or E.shape != (3, 3):
            abstained["recover_failed"] += 1
            continue
        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < min_pair_inliers:
            abstained["too_few_inliers"] += 1
            continue
        retval, R_audit, t_audit, _pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        if retval < min_pair_inliers:
            abstained["too_few_inliers"] += 1
            continue

        R_rec, t_rec = _relative_pose_from_reconstruction(by_id[fid_i], by_id[fid_j], np)
        rot_dev = _rotation_angle_deg(R_audit @ R_rec.T, np)
        t_rec_norm = float(np.linalg.norm(t_rec))
        if t_rec_norm > 1e-9:
            cosang = float(np.clip(
                np.dot(t_audit.reshape(3) / np.linalg.norm(t_audit),
                       t_rec / t_rec_norm), -1.0, 1.0))
            trans_dir_dev = float(np.degrees(np.arccos(cosang)))
        else:
            trans_dir_dev = None  # near-zero baseline: direction undefined, never fabricated
        audited_rotations[(fid_i, fid_j)] = R_audit
        pair_rows.append({
            "frame_i": fid_i, "frame_j": fid_j,
            "frame_separation": fid_j - fid_i,
            "ransac_inliers": n_inliers,
            "cheirality_inliers": int(retval),
            "rotation_deviation_deg": rot_dev,
            "translation_direction_deviation_deg": trans_dir_dev,
        })

    # Auditor noise floor: rotation composition around audited triangles.
    cycle_residuals = []
    audited_pairs = set(audited_rotations)
    fids = sorted(features)
    for ai, a in enumerate(fids):
        for b in fids[ai + 1:]:
            if (a, b) not in audited_pairs:
                continue
            for c in fids:
                if c <= b:
                    continue
                if (b, c) in audited_pairs and (a, c) in audited_pairs:
                    R_ab = audited_rotations[(a, b)]
                    R_bc = audited_rotations[(b, c)]
                    R_ac = audited_rotations[(a, c)]
                    cycle_residuals.append(
                        _rotation_angle_deg(R_bc @ R_ab @ R_ac.T, np))

    n_valid = len(pair_rows)
    if n_valid < min_valid_pairs:
        return {
            **base,
            "status": "abstained_insufficient_valid_pairs",
            "authority": "none",
            "authority_note": (
                "abstention is AUTHORITY LOSS, never a pass: low-texture/blur "
                "conditions correlate with backbone failure"
            ),
            "intrinsics_source": intrinsics_source,
            "n_valid_pairs": n_valid,
            "min_valid_pairs": min_valid_pairs,
            "abstained_pairs": abstained,
            "pairs": pair_rows,
        }

    rot_devs = [r["rotation_deviation_deg"] for r in pair_rows]
    trans_devs = [r["translation_direction_deviation_deg"] for r in pair_rows
                  if isinstance(r["translation_direction_deviation_deg"], (int, float))]
    rot_p90 = float(np.percentile(rot_devs, 90.0))
    cycle_p90 = float(np.percentile(cycle_residuals, 90.0)) if cycle_residuals else None
    # Measured-authority self-check: a deviation reading below the auditor's own
    # cycle-residual noise floor has no authority -- the auditor cannot
    # distinguish it from its own noise, and silence below the floor is never
    # evidence of correctness.
    if cycle_p90 is None:
        deviation_authority = "unknown_no_closed_triangles"
    elif rot_p90 > cycle_p90:
        deviation_authority = "above_auditor_noise_floor"
    else:
        deviation_authority = "below_auditor_noise_floor_no_authority"
    return {
        **base,
        "status": "audited",
        "authority": (
            "full" if intrinsics_source == "measured_calibration"
            else "demoted_backbone_intrinsics"
        ),
        "intrinsics_source": intrinsics_source,
        "n_valid_pairs": n_valid,
        "n_audited_frames": len(features),
        "abstained_pairs": abstained,
        "rotation_deviation_deg": {
            "median": float(np.median(rot_devs)),
            "p90": rot_p90,
            "max": float(np.max(rot_devs)),
        },
        "deviation_authority": deviation_authority,
        "translation_direction_deviation_deg": (
            {
                "median": float(np.median(trans_devs)),
                "p90": float(np.percentile(trans_devs, 90.0)),
                "n": len(trans_devs),
            } if trans_devs else {"status": "no_pairs_with_defined_direction"}
        ),
        "auditor_cycle_residual_deg": (
            {
                "median": float(np.median(cycle_residuals)),
                "p90": float(np.percentile(cycle_residuals, 90.0)),
                "n_triangles": len(cycle_residuals),
                "note": "auditor self-consistency (no reconstruction involved): its noise floor",
            } if cycle_residuals else {"status": "no_closed_triangles_audited"}
        ),
        "pairs": pair_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="reference_metric")
    parser.add_argument("--artifacts-dir", default="external/teacher_artifacts")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    args = parser.parse_args(argv)

    root = Path.cwd()
    import sys as _sys
    _sys.path.insert(0, str(root / "src"))
    from atlas3r.geometry_adapter import load_geometry_artifacts
    from atlas3r.refine import refine_scene

    mono, grep = load_geometry_artifacts(args.asset, root, artifacts_dir=args.artifacts_dir)
    soft = grep.get("_scale_evidence", [])
    refined, _ = refine_scene(mono, fix_global_scale=bool(soft))
    report = audit_scene(refined, args.asset, root, max_pairs=args.max_pairs)

    out = root / "runs/_diag" / f"epipolar_audit_{args.asset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "pairs"}
    print(json.dumps(summary, indent=2, default=str))
    print(f"[epipolar_audit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
