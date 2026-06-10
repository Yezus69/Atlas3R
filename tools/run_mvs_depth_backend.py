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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--hybrid-artifacts", default="external/_hybrid_artifacts")
    parser.add_argument("--dense-workspace", required=True)
    parser.add_argument("--out-dir", default="external/_composite_artifacts")
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

    replaced_px, total_px, frames_done, frames_missing_mvs = 0, 0, 0, []
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
