"""Hybrid pose backend: COLMAP SfM poses + MapAnything depth (M3 artifacts).

Measured motivation (runs/_diag/colmap_oracle_results.json, 2026-06-10): on the
selector keyframes, COLMAP (BSD, CPU, fixed measured intrinsics) reaches
camera-center Sim(3) RMSE 0.0116 m on xyz and 0.0213 m on freiburg1_desk --
6.7x / 13x better than the feed-forward backbone's poses, crushing the 0.06 m
oracle bar on the realistic-motion scene. The pose problem is a pipeline gap,
not a model gap; this tool wires the fix at the existing artifact seam.

What it does: copy an existing MapAnything artifact dir, replace the pose of
every COLMAP-registered frame, drop unregistered frames from the manifest
(honest: a frame whose pose the SfM could not establish is not fabricated).

Scale handling (honesty-critical): COLMAP monocular SfM has gauge-free scale.
Depth stays attached to its camera, so pose/depth unit consistency is required
for fusion. The COLMAP trajectory is rescaled by ONE scalar obtained by
Umeyama-aligning COLMAP camera centers to the SOURCE BACKBONE's camera centers
over common frames -- candidate-only evidence, no ground truth anywhere. The
rotations and trajectory SHAPE stay COLMAP's; only the unit is borrowed from
the backbone's learned metric prior, so the hybrid inherits the same honest
scale category (metric_pseudo_label at most, never measured).

Provenance: pose_source=colmap_sfm_scaled_to_backbone is stamped in the
manifest and poses.json; the scale-alignment record (scalar, common-frame
count, residual) is preserved verbatim.

Usage:
  python tools/run_colmap_pose_backend.py --asset-id reference_metric \
      --colmap-model runs/_diag/colmap_work/reference_metric/sparse/0 \
      [--source-artifacts external/teacher_artifacts] \
      [--out-dir external/_hybrid_artifacts]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def parse_colmap_images(images_txt: Path) -> dict[str, np.ndarray]:
    """image name -> T_world_camera (4x4). COLMAP stores world-to-camera
    (X_cam = R X_world + t); we emit camera-to-world per the repo contract."""
    out: dict[str, np.ndarray] = {}
    for line in images_txt.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        v = s.split()
        if len(v) < 10 or not v[0].isdigit():
            continue
        qw, qx, qy, qz = map(float, v[1:5])
        t = np.array([float(v[5]), float(v[6]), float(v[7])])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        out[v[9]] = T
    return out


def umeyama_scale(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """Similarity scale aligning src->dst camera centers + residual RMSE."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, Sig, Vt = np.linalg.svd((D.T @ S) / len(src))
    W = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = float(np.trace(np.diag(Sig) @ W) / ((S ** 2).sum() / len(src)))
    t = mu_d - s * R @ mu_s
    resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    return s, float(np.sqrt((resid ** 2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--colmap-model", required=True,
                        help="COLMAP sparse model dir containing images.txt")
    parser.add_argument("--source-artifacts", default="external/teacher_artifacts")
    parser.add_argument("--out-dir", default="external/_hybrid_artifacts")
    args = parser.parse_args()

    src_dir = ROOT / args.source_artifacts / args.asset_id
    out_dir = ROOT / args.out_dir / args.asset_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    poses_path = out_dir / "poses.json"
    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    frames = payload["frames"]

    colmap_T = parse_colmap_images(Path(args.colmap_model) / "images.txt")
    # Frame entries carry the source frame path; match by file name.
    by_name = {}
    for fr in frames:
        name = Path(str(fr.get("source_frame_path", ""))).name
        if name:
            by_name[name] = fr

    common = [n for n in colmap_T if n in by_name]
    if len(common) < 5:
        print(json.dumps({"status": "error_too_few_common_frames", "common": len(common)}))
        return 1

    src_centers = np.array([colmap_T[n][:3, 3] for n in common])
    backbone_centers = np.array([
        np.asarray(by_name[n]["T_world_camera"], dtype=float)[:3, 3] for n in common
    ])
    scale, resid = umeyama_scale(src_centers, backbone_centers)

    new_frames = []
    replaced, dropped = [], []
    for fr in frames:
        name = Path(str(fr.get("source_frame_path", ""))).name
        if name in colmap_T:
            T = colmap_T[name].copy()
            T[:3, 3] *= scale  # unit borrowed from the backbone; shape/rotation COLMAP's
            fr = dict(fr)
            fr["T_world_camera"] = [[float(x) for x in row] for row in T]
            fr["pose_source"] = "colmap_sfm_scaled_to_backbone"
            new_frames.append(fr)
            replaced.append(int(fr["frame_id"]))
        else:
            dropped.append(int(fr["frame_id"]))  # unregistered: dropped, never fabricated

    payload["frames"] = new_frames
    payload["source"] = f"{payload.get('source', 'backbone')}+colmap_pose_backend"
    payload["pose_provenance"] = {
        "pose_source": "colmap_sfm_scaled_to_backbone",
        "colmap_model": str(Path(args.colmap_model).as_posix()),
        "registered_frames": len(replaced),
        "dropped_unregistered_frame_ids": dropped,
        "scale_alignment": {
            "method": "umeyama_similarity_scale_only_vs_backbone_camera_centers",
            "scale": scale,
            "common_frames": len(common),
            "residual_rmse_backbone_units": resid,
            "honesty": "candidate-only alignment; no measured data; rotations and trajectory shape are COLMAP's",
        },
    }
    poses_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest_path = out_dir / "backbone_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_ids"] = sorted(replaced)
    manifest["method"] = f"{manifest.get('method', '')}+colmap_pose_backend"
    manifest["pose_source"] = "colmap_sfm_scaled_to_backbone"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "written", "asset": args.asset_id,
        "out_dir": str(out_dir.relative_to(ROOT).as_posix()),
        "registered": len(replaced), "dropped": len(dropped),
        "scale": round(scale, 6), "scale_residual_rmse": round(resid, 4),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
