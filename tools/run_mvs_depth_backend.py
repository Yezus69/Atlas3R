"""Composite depth backend: MVS-verified depth where available, learned fill.

Proven by pilot (docs/band_obstacle_recall_evidence.md Phase 8 continuation):
COLMAP-CUDA PatchMatch geometric verification is 2.2x more accurate than the
learned depth (median |log err| 0.0163 vs 0.036, edge 0.0357 vs 0.048) over
~68% of pixels, and the composite lifted xyz count-level band recall
0.282 -> 0.780 at 2.5 cm voxels.

Takes a HYBRID artifact dir (COLMAP poses + learned depth, from
tools/run_colmap_pose_backend.py) plus the COLMAP dense workspace, and writes
a composite artifact dir where each frame's depth is MVS-verified depth where
the geometric check passed, learned depth elsewhere. Scale consistency: MVS
depth stays in the existing pose gauge, while learned fill pixels are rescaled
per frame onto that pose gauge when enough MVS-verified pixels exist, with a
pre-registered global fallback otherwise. Near-unity global ratios are recorded
but left as no-ops so the fix only fires on material two-gauge composites. No
ground truth is used.

Also writes per-frame ``verified/<frame_id>.npy`` boolean masks: the
multi-view-verification provenance of every pixel. These feed the
verified-evidence tier specified in ARCHITECTURE.md (implementation pending):
a voxel whose surface evidence is geometrically verified is not "contested"
by see-through free traversals -- verification is the resolution of exactly
that contradiction. Until that lands, the masks are provenance reportage.

Usage:
  python tools/run_mvs_depth_backend.py --asset-id reference_metric \
      --hybrid-artifacts external/_hybrid_artifacts \
      --dense-workspace runs/_diag/colmap_work/reference_metric/dense \
      [--out-dir external/_composite_artifacts] [--promote]

CHAIN GUARD: the composite under external/_composite_artifacts is NOT the
teacher read path. ``--promote`` atomically installs it as
external/teacher_artifacts/<asset> (the old dir is backed up to
external/_superseded_artifacts/<asset>_<n>). Default OFF; when off the exact
promote command is printed so the step is explicit, never silent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_MIN_SINGLE_GAUGE_PIXELS = 500
SINGLE_GAUGE_APPLY_RATIO_BAND = (0.8, 1.25)


def single_gauge_v2_decisions(
    frame_observations: list[dict],
    min_verified_pixels: int = N_MIN_SINGLE_GAUGE_PIXELS,
) -> dict:
    """Choose the depth-derived global gauge and per-frame fill rescale.

    ``raw_backbone_over_pose_mvs_median`` is the per-frame fill/MVS gauge ratio
    after applying the pose backend's COLMAP scale. Qualifying frames use their
    own estimate to undo backbone fill drift; low-support frames use the global
    median instead.
    """
    qualifying: list[float] = []
    for obs in frame_observations:
        scale = obs.get("raw_backbone_over_pose_mvs_median")
        if (
            int(obs.get("verified_count", 0)) >= min_verified_pixels
            and scale is not None
            and np.isfinite(float(scale))
            and float(scale) > 0.0
        ):
            qualifying.append(float(scale))
    if not qualifying:
        raise ValueError(
            "single_gauge_v2 needs at least one frame with enough verified pixels"
        )

    global_scale = float(np.median(np.asarray(qualifying, dtype=np.float64)))
    gauge_spread_log = (
        float(np.std(np.log(np.asarray(qualifying, dtype=np.float64))))
        if len(qualifying) > 1
        else 0.0
    )

    frames = []
    for obs in frame_observations:
        scale = obs.get("raw_backbone_over_pose_mvs_median")
        per_frame = (
            int(obs.get("verified_count", 0)) >= min_verified_pixels
            and scale is not None
            and np.isfinite(float(scale))
            and float(scale) > 0.0
        )
        frame_scale = float(scale) if per_frame else global_scale
        frames.append(
            {
                "frame_id": int(obs["frame_id"]),
                "verified_count": int(obs.get("verified_count", 0)),
                "raw_backbone_over_pose_mvs_median": (
                    None if scale is None else float(scale)
                ),
                "frame_scale_source": "per_frame" if per_frame else "global_fallback",
                "frame_scale_used": frame_scale,
                "fill_scale_multiplier": float(1.0 / frame_scale),
            }
        )
    return {
        "min_verified_pixels": int(min_verified_pixels),
        "depth_based_global_scale": global_scale,
        "qualifying_frame_count": len(qualifying),
        "gauge_spread_log": gauge_spread_log,
        "frames": frames,
    }


def read_colmap_depth(path: Path) -> np.ndarray:
    """COLMAP .bin depth map: ASCII 'w&h&c&' header then float32 data."""
    data = path.read_bytes()
    pos, vals = 0, []
    for _ in range(3):
        amp = data.index(b"&", pos)
        vals.append(int(data[pos:amp]))
        pos = amp + 1
    w, h, c = vals
    arr = np.frombuffer(data, dtype=np.float32, count=w * h * c, offset=pos)
    return arr.reshape(h, w, c).squeeze().astype(np.float64)


# COLMAP camera models supported by the remap. Params order:
# SIMPLE_PINHOLE f,cx,cy; PINHOLE fx,fy,cx,cy; SIMPLE_RADIAL f,cx,cy,k.
_SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL"}


def read_colmap_cameras(model_dir: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """camera_id -> (model_name, width, height, params) from cameras.txt.

    TXT only, on purpose: the 4.x BIN layout is rig-versioned and a wrong
    hand parse would silently mis-calibrate the remap. Missing TXT -> the
    exact conversion command, never a guess."""
    txt = model_dir / "cameras.txt"
    if not txt.exists():
        raise FileNotFoundError(
            f"{txt} missing -- convert first: external/colmap/bin/colmap.exe "
            f"model_converter --input_path {model_dir} --output_path {model_dir} "
            f"--output_type TXT"
        )
    cams: dict[int, tuple[str, int, int, list[float]]] = {}
    for row in txt.read_text().splitlines():
        if not row.strip() or row.startswith("#"):
            continue
        cam_id, model, w, h, *params = row.split()
        if model not in _SUPPORTED_CAMERA_MODELS:
            raise ValueError(f"unsupported camera model {model} in {txt}")
        cams[int(cam_id)] = (model, int(w), int(h), [float(p) for p in params])
    return cams


def read_image_camera_ids(model_dir: Path) -> dict[str, int]:
    """image name -> camera_id from images.txt (same TXT-only rule)."""
    txt = model_dir / "images.txt"
    if not txt.exists():
        raise FileNotFoundError(
            f"{txt} missing -- convert first: external/colmap/bin/colmap.exe "
            f"model_converter --input_path {model_dir} --output_path {model_dir} "
            f"--output_type TXT"
        )
    out: dict[str, int] = {}
    for line in txt.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        v = s.split()
        if len(v) >= 10 and v[0].isdigit():  # image row, not the points2D row
            out[v[9]] = int(v[8])
    return out


def distortion_aware_sample_map(
    learned_hw: tuple[int, int],
    orig_cam: tuple[str, int, int, list[float]],
    undist_cam: tuple[str, int, int, list[float]],
    depth_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row_idx, col_idx, valid) mapping each LEARNED-grid pixel to the MVS
    depth-map grid through the lens model: learned pixel -> original distorted
    pixel -> normalized ray (iterative SIMPLE_RADIAL inversion) -> undistorted
    pinhole pixel -> depth-map index. Depth VALUES need no correction --
    optical_z is shared between the original and undistorted cameras (same
    center, same axis); only the SAMPLING LOCATION moves with distortion."""
    ph, pw = learned_hw
    o_model, ow, oh, op = orig_cam
    u_model, uw, uh, up = undist_cam
    if o_model == "SIMPLE_PINHOLE":
        fx, fy, cx, cy, k = op[0], op[0], op[1], op[2], 0.0
    elif o_model == "PINHOLE":
        fx, fy, cx, cy = op[:4]
        k = 0.0
    else:  # SIMPLE_RADIAL
        fx, fy, cx, cy, k = op[0], op[0], op[1], op[2], op[3]
    if u_model == "SIMPLE_PINHOLE":
        ufx, ufy, ucx, ucy = up[0], up[0], up[1], up[2]
    else:  # PINHOLE (image_undistorter always writes PINHOLE)
        ufx, ufy, ucx, ucy = up[:4]

    jj, ii = np.meshgrid(np.arange(pw), np.arange(ph))
    u0 = (jj + 0.5) * ow / pw - 0.5
    v0 = (ii + 0.5) * oh / ph - 0.5
    xd = (u0 - cx) / fx
    yd = (v0 - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(6):  # fixed-point inversion of x_d = x (1 + k r^2)
        r2 = x * x + y * y
        denom = 1.0 + k * r2
        x = xd / denom
        y = yd / denom
    u1 = ufx * x + ucx
    v1 = ufy * y + ucy
    mh, mw = depth_hw
    col = np.rint((u1 + 0.5) * mw / uw - 0.5).astype(int)
    row = np.rint((v1 + 0.5) * mh / uh - 0.5).astype(int)
    valid = (row >= 0) & (row < mh) & (col >= 0) & (col < mw)
    return np.clip(row, 0, mh - 1), np.clip(col, 0, mw - 1), valid


def promote_composite(out: Path, asset_id: str, root: Path = ROOT) -> dict:
    """Atomically install the composite as the teacher read path.

    Two renames, never an in-place copy into the live dir, so the teacher can
    never observe a half-written artifact: (1) the composite is copied to a
    temp sibling under external/teacher_artifacts; (2) the existing live dir
    (if any) is renamed to external/_superseded_artifacts/<asset>_<n>; (3) the
    temp sibling is renamed into place. Each rename targets a non-existent
    destination (os.replace cannot replace a non-empty dir on Windows). If
    rename (3) fails after (2) succeeded, the backup is rolled back into the
    live path so the teacher read path is never left empty; the tmp sibling is
    left for inspection and is reclaimed by the next promote. Do NOT promote
    while a teacher run is reading the asset (Windows refuses to rename a dir
    with open handles).
    """
    live_root = root / "external" / "teacher_artifacts"
    superseded_root = root / "external" / "_superseded_artifacts"
    live = live_root / asset_id
    live_root.mkdir(parents=True, exist_ok=True)
    superseded_root.mkdir(parents=True, exist_ok=True)

    tmp = live_root / f"{asset_id}__promote_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(out, tmp)

    backup = None
    if live.exists():
        n = 1
        while (superseded_root / f"{asset_id}_{n}").exists():
            n += 1
        backup = superseded_root / f"{asset_id}_{n}"
        os.replace(live, backup)
    try:
        os.replace(tmp, live)
    except OSError:
        if backup is not None:
            # Rollback: restore the previous live artifact rather than leaving
            # the teacher read path empty. The failed composite stays at tmp.
            os.replace(backup, live)
        raise
    return {
        "promoted_to": live.relative_to(root).as_posix(),
        "superseded_backup": backup.relative_to(root).as_posix() if backup else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--hybrid-artifacts", default="external/_hybrid_artifacts")
    parser.add_argument("--dense-workspace", required=True)
    parser.add_argument("--out-dir", default="external/_composite_artifacts")
    parser.add_argument(
        "--source-sparse-model", default=None,
        help="COLMAP sparse model dir of the ORIGINAL (possibly distorted) "
             "camera. When given, MVS depth (which lives in the UNDISTORTED "
             "image geometry) is sampled through the lens model instead of by "
             "raw index ratio -- required for self-calibrated distorted video "
             "(e.g. SIMPLE_RADIAL phone footage); without it a distorted "
             "camera would misregister verified depth by up to ~k*r^3*f px.")
    parser.add_argument(
        "--stability-workspace-a", default=None,
        help="Dense workspace whose geometric depth came from one DISJOINT "
             "source half (perturbation A). With --stability-workspace-b, the "
             "verified mask is TIGHTENED to perturbation-STABLE pixels: "
             "verified := geometric AND witnessed by BOTH halves AND "
             "|log dA - log dB| <= stability-tau. Measured basis: "
             "docs/band_obstacle_recall_evidence.md Phase 12 (single-witness "
             "pixels are 1.6-1.8x worse; gating lifts solid F1@5cm 3.5-5.8x).")
    parser.add_argument("--stability-workspace-b", default=None)
    parser.add_argument(
        "--promote", action="store_true",
        help="after writing the composite, atomically replace "
             "external/teacher_artifacts/<asset> with it (old dir backed up to "
             "external/_superseded_artifacts/<asset>_<n>). Default OFF: the "
             "composite stays out of the teacher read path until promoted, and "
             "the exact promote command is printed.")
    parser.add_argument(
        "--stability-tau", type=float, default=0.005,
        help="Stability bar on |log dA - log dB| (scale-free). FROZEN at 0.005 "
             "from the Phase 12 two-scene calibration (xyz+desk, the stricter "
             "of the swept bars, monotone on desk) -- never tuned per scene.")
    args = parser.parse_args()
    if bool(args.stability_workspace_a) != bool(args.stability_workspace_b):
        print(json.dumps({"status": "error_stability_needs_both_workspaces"}))
        return 1

    src = ROOT / args.hybrid_artifacts / args.asset_id
    out = ROOT / args.out_dir / args.asset_id
    dense = ROOT / args.dense_workspace
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    (out / "verified").mkdir(exist_ok=True)

    poses = json.loads((out / "poses.json").read_text(encoding="utf-8"))
    prov = poses.get("pose_provenance", {})
    scale = float(prov.get("scale_alignment", {}).get("scale", 0.0))
    if scale <= 0:
        print(json.dumps({"status": "error_no_pose_scale_in_hybrid",
                          "note": "run tools/run_colmap_pose_backend.py first"}))
        return 1

    orig_cams = undist_cams = orig_cam_of = undist_cam_of = None
    resample_mode = "index_nn"
    if args.source_sparse_model:
        orig_cams = read_colmap_cameras(ROOT / args.source_sparse_model)
        orig_cam_of = read_image_camera_ids(ROOT / args.source_sparse_model)
        undist_cams = read_colmap_cameras(dense / "sparse")
        undist_cam_of = read_image_camera_ids(dense / "sparse")
        resample_mode = "distortion_aware_remap"

    stab_a = ROOT / args.stability_workspace_a if args.stability_workspace_a else None
    stab_b = ROOT / args.stability_workspace_b if args.stability_workspace_b else None
    tau = float(args.stability_tau)

    replaced_px, total_px, frames_done, frames_missing_mvs = 0, 0, 0, []
    frames_missing_camera: list[int] = []
    frames_missing_stability: list[int] = []
    sample_maps: dict = {}  # (cam_id pair, shapes) -> precomputed map
    frame_payloads: list[dict] = []

    def resample(arr: np.ndarray, name: str, ph: int, pw: int) -> np.ndarray:
        ah, aw = arr.shape
        if orig_cams is not None:
            key = (orig_cam_of[name], undist_cam_of[name], ph, pw, ah, aw)
            if key not in sample_maps:
                sample_maps[key] = distortion_aware_sample_map(
                    (ph, pw), orig_cams[key[0]], undist_cams[key[1]], (ah, aw)
                )
            row, col, valid = sample_maps[key]
            return np.where(valid, arr[row, col], 0.0)
        yi = (np.arange(ph) * ah / ph).astype(int)
        xi = (np.arange(pw) * aw / pw).astype(int)
        return arr[yi][:, xi]

    for fr in poses["frames"]:
        fid = int(fr["frame_id"])
        name = Path(str(fr["source_frame_path"])).name
        geo = dense / "stereo/depth_maps" / f"{name}.geometric.bin"
        npy = out / "depth" / f"{fid}.npy"
        if not npy.exists():
            continue
        if not geo.exists():
            frames_missing_mvs.append(fid)  # learned-only frame, recorded
            continue
        learned = np.load(npy).astype(np.float64)
        ph, pw = learned.shape[:2]
        if orig_cams is not None and (name not in orig_cam_of or name not in undist_cam_of):
            frames_missing_mvs.append(fid)
            frames_missing_camera.append(fid)  # no calibration claim -> learned-only
            continue
        mvs_raw = resample(read_colmap_depth(geo), name, ph, pw)
        mvs_pose = mvs_raw * scale
        verified = mvs_pose > 1e-6
        if stab_a is not None:
            geo_a = stab_a / "stereo/depth_maps" / f"{name}.geometric.bin"
            geo_b = stab_b / "stereo/depth_maps" / f"{name}.geometric.bin"
            if not geo_a.exists() or not geo_b.exists():
                # No independent witnesses -> no stability claim -> the frame
                # contributes learned depth only (verified stays all-False).
                frames_missing_stability.append(fid)
                verified = np.zeros_like(verified)
            else:
                a_r = resample(read_colmap_depth(geo_a), name, ph, pw)
                b_r = resample(read_colmap_depth(geo_b), name, ph, pw)
                with np.errstate(divide="ignore", invalid="ignore"):
                    delta = np.abs(np.log(a_r) - np.log(b_r))
                both_witness = (a_r > 1e-9) & (b_r > 1e-9)
                verified = verified & both_witness & (delta <= tau)
                # Persist the CONTINUOUS stability residual (NaN where either
                # witness is absent), not just the tau-thresholded boolean --
                # the best-provenanced per-pixel uncertainty signal in the
                # stack was previously computed and discarded here. float16:
                # spacing ~4e-6 near tau=0.005, ample for calibration binning.
                (out / "stability_delta").mkdir(parents=True, exist_ok=True)
                np.save(
                    out / "stability_delta" / f"{fid}.npy",
                    np.where(both_witness, delta, np.nan).astype(np.float16),
                )
        scale_mask = (
            verified
            & np.isfinite(learned)
            & np.isfinite(mvs_pose)
            & (learned > 1e-9)
            & (mvs_pose > 1e-9)
        )
        verified_count = int(scale_mask.sum())
        if verified_count > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                pose_ratio = learned[scale_mask] / mvs_pose[scale_mask]
            pose_median = float(np.median(pose_ratio))
        else:
            pose_median = None
        frame_payloads.append(
            {
                "frame_id": fid,
                "npy": npy,
                "learned": learned,
                "mvs_pose": mvs_pose,
                "verified": verified,
                "verified_count": verified_count,
                "raw_backbone_over_pose_mvs_median": pose_median,
                "before_backbone_over_pose_mvs_median": pose_median,
            }
        )
        replaced_px += int(verified.sum())
        total_px += verified.size
        frames_done += 1

    try:
        gauge = single_gauge_v2_decisions(frame_payloads)
    except ValueError as exc:
        print(json.dumps({"status": "error_single_gauge_v2_no_anchor", "error": str(exc)}))
        return 1

    frame_decisions = {row["frame_id"]: row for row in gauge["frames"]}
    depth_based_scale = float(gauge["depth_based_global_scale"])
    lo, hi = SINGLE_GAUGE_APPLY_RATIO_BAND
    fill_rescale_applied = not (lo <= depth_based_scale <= hi)
    for item in frame_payloads:
        decision = frame_decisions[item["frame_id"]]
        fill_multiplier = (
            float(decision["fill_scale_multiplier"]) if fill_rescale_applied else 1.0
        )
        scale_mask = (
            item["verified"]
            & np.isfinite(item["learned"])
            & np.isfinite(item["mvs_pose"])
            & (item["learned"] > 1e-9)
            & (item["mvs_pose"] > 1e-9)
        )
        if int(scale_mask.sum()) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                after_ratio = (
                    item["learned"][scale_mask]
                    * fill_multiplier
                    / item["mvs_pose"][scale_mask]
                )
            decision["after_fill_over_verified_median"] = float(np.median(after_ratio))
        else:
            decision["after_fill_over_verified_median"] = None
        decision["before_backbone_over_pose_mvs_median"] = item[
            "before_backbone_over_pose_mvs_median"
        ]
        decision["applied_fill_scale_multiplier"] = fill_multiplier
        out_depth = np.where(
            item["verified"],
            item["mvs_pose"],
            item["learned"] * fill_multiplier,
        )
        np.save(item["npy"], out_depth.astype(np.float32))
        np.save(out / "verified" / f"{item['frame_id']}.npy", item["verified"])

    pose_rescale = 1.0
    old_alignment = dict(prov.get("scale_alignment", {}))
    prov["scale_alignment"] = {
        "method": "median_backbone_over_pose_gauge_mvs_depth_single_gauge_v2",
        "scale": depth_based_scale,
        "scale_field_units": "raw_backbone_depth_per_pose_gauge_mvs_depth",
        "min_verified_pixels": int(gauge["min_verified_pixels"]),
        "qualifying_frames": int(gauge["qualifying_frame_count"]),
        "gauge_spread_log": float(gauge["gauge_spread_log"]),
        "apply_ratio_band": [float(lo), float(hi)],
        "fill_rescale_applied": bool(fill_rescale_applied),
        "previous_trajectory_umeyama_scale_alignment": old_alignment,
        "pose_translations_preserved_from_previous_alignment": True,
        "residual_rmse_backbone_units": old_alignment.get("residual_rmse_backbone_units"),
        "common_frames": old_alignment.get("common_frames"),
        "honesty": (
            "candidate-only fill-to-MVS gauge alignment; no measured data; "
            "verified MVS pixels and COLMAP poses preserve the existing pose gauge"
        ),
    }
    poses["pose_provenance"] = prov
    (out / "poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")

    manifest_path = out / "backbone_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method"] = manifest.get("method", "") + "+mvs_verified_depth+single_gauge_v2"
    manifest["depth_source"] = {
        "primary": "colmap_patchmatch_geometric_verified",
        "fill": "learned_backbone_depth_rescaled_single_gauge_v2",
        "verified_pixel_fraction": round(replaced_px / max(total_px, 1), 4),
        "frames_with_mvs": frames_done,
        "frames_learned_only": frames_missing_mvs,
        "scale_applied_from_pose_provenance": scale,
        "previous_trajectory_umeyama_scale": scale,
        "single_gauge_v2": {
            "min_verified_pixels": int(gauge["min_verified_pixels"]),
            "depth_based_global_scale": depth_based_scale,
            "scale_field_units": "raw_backbone_depth_per_pose_gauge_mvs_depth",
            "mvs_pose_scale_applied_from_pose_provenance": scale,
            "previous_trajectory_umeyama_scale": scale,
            "pose_translation_rescale": pose_rescale,
            "qualifying_frame_count": int(gauge["qualifying_frame_count"]),
            "gauge_spread_log": float(gauge["gauge_spread_log"]),
            "apply_ratio_band": [float(lo), float(hi)],
            "fill_rescale_applied": bool(fill_rescale_applied),
        },
        "verified_masks": "verified/<frame_id>.npy (bool, depth-map resolution)",
        "mvs_resample": resample_mode,
        "original_cameras": (
            {str(cid): {"model": c[0], "width": c[1], "height": c[2], "params": c[3]}
             for cid, c in orig_cams.items()} if orig_cams else None
        ),
        "frames_missing_camera": frames_missing_camera,
        "stability": (
            {
                "enabled": True,
                "tau": tau,
                "rule": "verified := geometric AND both disjoint-half witnesses AND |log dA - log dB| <= tau",
                "tau_provenance": "frozen_once_from_phase12_two_scene_calibration_xyz_desk",
                "workspace_a": str(args.stability_workspace_a),
                "workspace_b": str(args.stability_workspace_b),
                "frames_missing_stability": frames_missing_stability,
                "stability_delta_maps": (
                    "stability_delta/<frame_id>.npy (float16 |log dA - log dB|,"
                    " NaN where either witness absent; continuous residual"
                    " behind the tau-thresholded verified mask)"
                ),
            }
            if stab_a is not None else {"enabled": False}
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    depth_meta_path = out / "depth_meta.json"
    depth_meta = json.loads(depth_meta_path.read_text(encoding="utf-8"))
    depth_meta["single_gauge_v2"] = {
        "min_verified_pixels": int(gauge["min_verified_pixels"]),
        "depth_based_global_scale": depth_based_scale,
        "scale_field_units": "raw_backbone_depth_per_pose_gauge_mvs_depth",
        "mvs_pose_scale_applied_from_pose_provenance": scale,
        "previous_trajectory_umeyama_scale": scale,
        "pose_translation_rescale": pose_rescale,
        "qualifying_frame_count": int(gauge["qualifying_frame_count"]),
        "gauge_spread_log": float(gauge["gauge_spread_log"]),
        "apply_ratio_band": [float(lo), float(hi)],
        "fill_rescale_applied": bool(fill_rescale_applied),
        "frame_scales": sorted(gauge["frames"], key=lambda row: row["frame_id"]),
    }
    depth_meta_path.write_text(json.dumps(depth_meta, indent=2), encoding="utf-8")

    if args.promote:
        promote = promote_composite(out, args.asset_id)
    else:
        promote = {
            "enabled": False,
            "note": "composite is NOT the teacher read path until promoted",
            "promote_command": "python tools/run_mvs_depth_backend.py "
                               + subprocess.list2cmdline(sys.argv[1:])
                               + " --promote",
        }

    print(json.dumps({
        "status": "written",
        "out_dir": str(out.relative_to(ROOT).as_posix()),
        "frames_with_mvs": frames_done,
        "frames_learned_only": len(frames_missing_mvs),
        "verified_pixel_fraction": round(replaced_px / max(total_px, 1), 4),
        "promote": promote,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
