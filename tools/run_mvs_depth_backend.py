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
    args = parser.parse_args()

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
    resample = "index_nn"
    if args.source_sparse_model:
        orig_cams = read_colmap_cameras(ROOT / args.source_sparse_model)
        orig_cam_of = read_image_camera_ids(ROOT / args.source_sparse_model)
        undist_cams = read_colmap_cameras(dense / "sparse")
        undist_cam_of = read_image_camera_ids(dense / "sparse")
        resample = "distortion_aware_remap"

    replaced_px, total_px, frames_done, frames_missing_mvs = 0, 0, 0, []
    frames_missing_camera: list[int] = []
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
        if orig_cams is not None:
            if name not in orig_cam_of or name not in undist_cam_of:
                frames_missing_mvs.append(fid)
                frames_missing_camera.append(fid)  # no calibration claim -> learned-only
                continue
            key = (orig_cam_of[name], undist_cam_of[name], ph, pw, mh, mw)
            if key not in sample_maps:
                sample_maps[key] = distortion_aware_sample_map(
                    (ph, pw), orig_cams[key[0]], undist_cams[key[1]], (mh, mw)
                )
            row, col, valid = sample_maps[key]
            mvs_r = np.where(valid, mvs[row, col], 0.0) * scale
        else:
            yi = (np.arange(ph) * mh / ph).astype(int)
            xi = (np.arange(pw) * mw / pw).astype(int)
            mvs_r = mvs[yi][:, xi] * scale
        verified = mvs_r > 1e-6
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
        "mvs_resample": resample,
        "original_cameras": (
            {str(cid): {"model": c[0], "width": c[1], "height": c[2], "params": c[3]}
             for cid, c in orig_cams.items()} if orig_cams else None
        ),
        "frames_missing_camera": frames_missing_camera,
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
