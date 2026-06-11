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
depth (COLMAP units) is multiplied by the SAME candidate-only scalar recorded
in the hybrid's pose provenance -- no ground truth anywhere.

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
      [--out-dir external/_composite_artifacts]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
        mvs = read_colmap_depth(geo)
        ph, pw = learned.shape[:2]
        mh, mw = mvs.shape

        def resample(arr, name=name, ph=ph, pw=pw):
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

        if orig_cams is not None and (name not in orig_cam_of or name not in undist_cam_of):
            frames_missing_mvs.append(fid)
            frames_missing_camera.append(fid)  # no calibration claim -> learned-only
            continue
        mvs_r = resample(mvs) * scale
        verified = mvs_r > 1e-6
        if stab_a is not None:
            geo_a = stab_a / "stereo/depth_maps" / f"{name}.geometric.bin"
            geo_b = stab_b / "stereo/depth_maps" / f"{name}.geometric.bin"
            if not geo_a.exists() or not geo_b.exists():
                # No independent witnesses -> no stability claim -> the frame
                # contributes learned depth only (verified stays all-False).
                frames_missing_stability.append(fid)
                verified = np.zeros_like(verified)
            else:
                a_r = resample(read_colmap_depth(geo_a))
                b_r = resample(read_colmap_depth(geo_b))
                with np.errstate(divide="ignore", invalid="ignore"):
                    delta = np.abs(np.log(a_r) - np.log(b_r))
                verified = verified & (a_r > 1e-9) & (b_r > 1e-9) & (delta <= tau)
        np.save(npy, np.where(verified, mvs_r, learned).astype(np.float32))
        np.save(out / "verified" / f"{fid}.npy", verified)
        replaced_px += int(verified.sum())
        total_px += verified.size
        frames_done += 1

    manifest_path = out / "backbone_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method"] = manifest.get("method", "") + "+mvs_verified_depth"
    manifest["depth_source"] = {
        "primary": "colmap_patchmatch_geometric_verified",
        "fill": "learned_backbone_depth",
        "verified_pixel_fraction": round(replaced_px / max(total_px, 1), 4),
        "frames_with_mvs": frames_done,
        "frames_learned_only": frames_missing_mvs,
        "scale_applied_from_pose_provenance": scale,
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
            }
            if stab_a is not None else {"enabled": False}
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "out_dir": str(out.relative_to(ROOT).as_posix()),
        "frames_with_mvs": frames_done,
        "frames_learned_only": len(frames_missing_mvs),
        "verified_pixel_fraction": round(replaced_px / max(total_px, 1), 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
