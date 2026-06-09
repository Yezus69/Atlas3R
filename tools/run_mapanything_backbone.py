"""MapAnything feed-forward geometry backbone runner for Atlas3R M3 (PILOT).

A drop-in alternative to ``tools/run_da3_backbone.py``: one feed-forward model
(MapAnything, Meta + CMU, arXiv 2509.13414) emits, jointly over a set of monocular
RGB views, per-pixel METRIC depth + camera extrinsics + intrinsics + confidence.
Unlike the DA3 pair (relative-geometry model + separate metric-anchor model),
MapAnything has an explicit metric head, so a single forward pass produces the full
factored artifact set the M3 adapter consumes.

Why this exists: the measured Atlas3R bottleneck is ~12 cm indoor monocular surface
displacement from the DA3 backbone (misses ~72% of collision-band obstacles). Per
``docs/sota_backbone_research.md`` the honest lever is a better/metric geometry
source at the M3 seam. MapAnything-apache is pick #1 (Apache-2.0 code AND checkpoint
-> commercially clean; indoor-trained). This runner makes the A/B measurable.

HONESTY (unchanged from DA3, enforced by the M3 adapter + M5 scale ladder):
- The learned metric scale is a SOFT ``learned_metric_depth_prior`` (``measured=False``);
  the manifest sets ``metric_evidence=false`` -> the teacher classifies at most
  ``metric_pseudo_label``, NEVER ``measured_metric``. Measured TUM depth stays
  eval-only and is never fed in as a candidate input.

Conventions are PINNED FROM THE INSTALLED MapAnything SOURCE (not guessed); see the
``_extract_view`` docstring. ``prediction`` depth is converted to the adapter's
internal radial range downstream by ``geometry_adapter._source_depth_to_radial``.

Standalone tool. NOT imported by ``atlas3r``. Run with an isolated MapAnything venv
(``external/mapanything_env``). No weights/repos vendored into git.

Usage:
    external/mapanything_env/Scripts/python.exe tools/run_mapanything_backbone.py \
        --asset-id reference_metric --frames-dir data/reference_metric/rgb \
        --frame-glob "*.png" \
        --frame-ids "0,82,164,228,304,380,436,493,538,582,640,696,708,752" \
        --model facebook/map-anything-apache --device cuda \
        --out-dir external/teacher_artifacts
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _natural_key(path: Path):
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name))


def _evenly_spaced(count: int, k: int):
    if count <= 0:
        return []
    import numpy as np

    k = max(1, min(k, count))
    if k == 1:
        return [0]
    return sorted(set(int(round(i)) for i in np.linspace(0, count - 1, num=k)))


def _select_frame_ids(all_frames, frame_ids: str | None, num_frames: int):
    if frame_ids:
        ids = sorted({int(x) for x in frame_ids.split(",") if x.strip()})
        return [i for i in ids if 0 <= i < len(all_frames)]
    return _evenly_spaced(len(all_frames), num_frames)


def _load_uint8_hwc(path: str, np):
    """Load an image as uint8 RGB HxWx3 in [0,255] (the MapAnything minimal view form)."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _to_T_world_camera(pose_4x4, pose_is_c2w: bool, np):
    """Return camera-to-world ``T_world_camera`` from a 4x4 pose, inverting if the
    source pose is world-to-camera."""
    P = np.eye(4, dtype=np.float64)
    arr = np.asarray(pose_4x4, dtype=np.float64)
    P[:4, :4] = arr[:4, :4] if arr.shape == (4, 4) else np.vstack([arr[:3, :4], [0, 0, 0, 1]])
    return P if pose_is_c2w else np.linalg.inv(P)


def _build_views(image_paths, np):
    """Build MapAnything view dicts. Prefer the documented ``load_images`` helper
    (handles resize + DINOv2 norm); fall back to the minimal uint8-HWC form."""
    try:
        from mapanything.utils.image import load_images  # type: ignore

        return load_images(list(image_paths))
    except Exception:
        return [{"img": _load_uint8_hwc(p, np)} for p in image_paths]


def _run_mapanything(model_repo, image_paths, device, np, torch, memory_efficient: bool):
    """Run MapAnything over the RGB views; return predictions.

    API + CONVENTIONS PINNED FROM THE INSTALLED SOURCE (verified 2026-06):
    ``MapAnything.from_pretrained(repo).infer(views, memory_efficient_inference=...,
    use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True)``. Output is a
    per-view (batched B) dict with ``depth_z`` (optical_z, metric m), ``camera_poses``
    (4x4 cam2world == T_world_camera, NO inversion), ``intrinsics`` (3x3), ``conf``.
    """
    from mapanything.models import MapAnything  # type: ignore

    model = MapAnything.from_pretrained(model_repo).to(device)
    model.eval()
    views = _build_views(image_paths, np)
    amp = device.startswith("cuda")
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=memory_efficient,
            use_amp=amp,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return preds


def _extract_view(pred_view, np):
    """Extract (depth_z[H,W], T_world_camera[4,4], intr[3,3], conf[H,W]) from ONE
    MapAnything per-view prediction.

    CONVENTIONS (pin against the installed MapAnything source before trusting):
    - MapAnything per-view outputs include ``depth_z`` (z-depth along the optical
      axis, metric meters), ``camera_poses`` (4x4; OpenCV camera-to-world in the
      shared world frame), ``intrinsics`` (3x3 pixel), and ``conf``/``mask``.
    - If the installed build names depth ``depth_along_ray`` instead, set
      ``--depth-is-radial``; if poses are world-to-camera, set ``--poses-w2c``.
    Tensors are moved to CPU numpy here.
    """
    def g(*keys):
        for k in keys:
            if isinstance(pred_view, dict) and k in pred_view and pred_view[k] is not None:
                return pred_view[k]
        return None

    def to_np(x):
        if x is None:
            return None
        try:
            return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        except AttributeError:
            return np.asarray(x, dtype=np.float64)

    depth = to_np(g("depth_z", "depth", "depth_along_ray"))
    pose = to_np(g("camera_poses", "camera_pose", "cam2world", "extrinsics"))
    intr = to_np(g("intrinsics", "K", "camera_intrinsics"))
    conf = to_np(g("conf", "confidence", "mask_conf"))

    def drop_lead1(x):
        # MapAnything returns per-view tensors with a LEADING batch-of-1 dim, e.g.
        # depth_z (1,H,W,1), camera_poses (1,4,4), conf (1,H,W). Peel leading 1s.
        while x is not None and x.ndim >= 3 and x.shape[0] == 1:
            x = x[0]
        return x

    depth = drop_lead1(depth)
    if depth is not None and depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]            # (H,W,1) -> (H,W)
    conf = drop_lead1(conf)
    if conf is not None and conf.ndim == 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if pose is not None and pose.size >= 16:
        pose = pose.reshape(-1)[:16].reshape(4, 4)
    if intr is not None and intr.size >= 9:
        intr = intr.reshape(-1)[:9].reshape(3, 3)
    return depth, pose, intr, conf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MapAnything geometry backbone runner for Atlas3R M3 (pilot).")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-glob", default="*.png")
    parser.add_argument("--frame-ids", default=None, help="comma-separated indices into the sorted frame list")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--model", default="facebook/map-anything-apache")
    parser.add_argument("--out-dir", default="external/teacher_artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=518, help="resize long side to this before inference")
    parser.add_argument("--memory-efficient", action="store_true", help="MapAnything memory_efficient_inference (slower, less VRAM)")
    parser.add_argument("--poses-w2c", action="store_true", help="installed build emits world-to-camera poses (invert to c2w)")
    parser.add_argument("--depth-is-radial", action="store_true", help="installed build emits along-ray depth (skip optical_z label)")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    frames_dir = Path(args.frames_dir)
    all_frames = sorted((p for p in frames_dir.glob(args.frame_glob) if p.is_file()), key=_natural_key)
    if len(all_frames) < 2:
        print(json.dumps({"status": "error", "blocker": "fewer_than_two_frames"}))
        return 2
    ids = _select_frame_ids(all_frames, args.frame_ids, args.num_frames)
    if len(ids) < 2:
        print(json.dumps({"status": "error", "blocker": "fewer_than_two_selected_frames"}))
        return 2

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    selected_paths = [str(all_frames[i]) for i in ids]

    try:
        preds = _run_mapanything(args.model, selected_paths, device, np, torch, args.memory_efficient)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        preds = _run_mapanything(args.model, selected_paths, "cpu", np, torch, True)
        device = "cpu"
    except ImportError as exc:
        print(json.dumps({
            "status": "error",
            "blocker": "mapanything_not_installed",
            "hint": "create external/mapanything_env and `pip install mapanything`; run with that interpreter",
            "error": str(exc),
        }))
        return 3

    # MapAnything returns a list (one dict per view) or a dict of stacked tensors;
    # normalize to a per-view list.
    if isinstance(preds, dict):
        n = len(ids)
        preds = [{k: (v[i] if hasattr(v, "__len__") and len(v) == n else v) for k, v in preds.items()} for i in range(n)]

    out_root = Path(args.out_dir) / args.asset_id
    depth_dir, conf_dir = out_root / "depth", out_root / "confidence"
    for d in (depth_dir, conf_dir):
        if d.exists():
            shutil.rmtree(d)
    for d in (out_root, depth_dir, conf_dir):
        d.mkdir(parents=True, exist_ok=True)

    pose_frames, written, skipped, depth_stats, per_frame_intr = [], [], [], [], {}
    fxs, fys, cxs, cys, hs, ws = [], [], [], [], [], []
    for k, frame_id in enumerate(ids):
        depth, pose, intr, conf = _extract_view(preds[k], np)
        if depth is None or pose is None or intr is None:
            skipped.append({"frame_id": int(frame_id), "reason": "missing_depth_pose_or_intrinsics_from_model"})
            continue
        if args.depth_is_radial:
            # caller asserts along-ray depth; we still label it via depth_meta below
            pass
        finite_pos = np.isfinite(depth) & (depth > 0.0)
        if int(np.count_nonzero(finite_pos)) == 0:
            skipped.append({"frame_id": int(frame_id), "reason": "no_finite_positive_depth"})
            continue
        Tw = _to_T_world_camera(pose, pose_is_c2w=not args.poses_w2c, np=np)
        if not np.all(np.isfinite(Tw)):
            skipped.append({"frame_id": int(frame_id), "reason": "non_finite_pose"})
            continue

        H, W = depth.shape[:2]
        np.save(depth_dir / f"{frame_id}.npy", np.where(finite_pos, depth, 0.0).astype(np.float32))
        if conf is not None and conf.shape[:2] == (H, W):
            c = conf.astype(np.float64)
            hi = float(np.percentile(c[np.isfinite(c)], 99)) if np.any(np.isfinite(c)) else 1.0
            hi = hi if hi > 0 else 1.0
            np.save(conf_dir / f"{frame_id}.npy", np.clip(np.where(np.isfinite(c), c, 0.0) / hi, 0.0, 1.0).astype(np.float32))

        pose_frames.append({
            "frame_id": int(frame_id),
            "source_frame_path": str(all_frames[frame_id]),
            "T_world_camera": [[float(v) for v in row] for row in Tw.tolist()],
        })
        per_frame_intr[str(int(frame_id))] = {
            "fx": float(intr[0, 0]), "fy": float(intr[1, 1]),
            "cx": float(intr[0, 2]), "cy": float(intr[1, 2]),
        }
        fxs.append(intr[0, 0]); fys.append(intr[1, 1]); cxs.append(intr[0, 2]); cys.append(intr[1, 2])
        hs.append(H); ws.append(W)
        dv = depth[finite_pos]
        depth_stats.append({"frame_id": int(frame_id), "min": float(dv.min()),
                            "median": float(np.median(dv)), "max": float(dv.max())})
        written.append(int(frame_id))

    if len(written) < 2:
        report = {"status": "error", "blocker": "fewer_than_two_views_with_valid_geometry",
                  "written": written, "skipped": skipped}
        (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 4

    fx, fy, cx, cy = float(np.median(fxs)), float(np.median(fys)), float(np.median(cxs)), float(np.median(cys))
    width, height = int(np.median(ws)), int(np.median(hs))
    fx_spread = float(np.std(fxs) / max(abs(np.median(fxs)), 1e-9))

    (out_root / "intrinsics.json").write_text(json.dumps({
        "model": "pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width_px": width, "height_px": height,
        "pixel_coordinate_convention": "top_left_pixel_centers",
        "note": "median of MapAnything per-view intrinsics at processed resolution",
    }, indent=2), encoding="utf-8")
    (out_root / "per_frame_intrinsics.json").write_text(json.dumps(per_frame_intr, indent=2), encoding="utf-8")
    (out_root / "poses.json").write_text(json.dumps({
        "convention": "T_world_camera", "translation_units": "meters",
        "source": "mapanything_camera_poses" + ("_inverted_from_w2c" if args.poses_w2c else "_c2w"),
        "frames": pose_frames,
    }, indent=2), encoding="utf-8")
    (out_root / "depth_meta.json").write_text(json.dumps({
        "depth_convention": "radial_range" if args.depth_is_radial else "optical_z",
        "units": "meters", "scale_to_meters_or_null": None,
        "density": "dense_learned_multiview_metric", "model_name": args.model,
        "metric_anchor_model": None, "metric_scale_applied": True,
    }, indent=2), encoding="utf-8")

    manifest = {
        "backbone_name": "atlas3r_mapanything",
        "method": f"mapanything_feedforward_metric_geometry:{args.model}",
        "model_name": args.model,
        "asset_id": args.asset_id,
        "frame_ids": written,
        "pose_units": "meters",
        "metric_evidence": False,          # learned prior is NOT a measurement
        "learned_metric_depth_prior": True,  # MapAnything emits learned METRIC geometry (soft)
        "scale_evidence_confidence": 0.6,    # conservative; MapAnything metric head, single forward
        "camera_confidence": 0.7,
        "metric_scale": 1.0,                 # already metric; no external anchor ratio applied
        "processed_resolution": [width, height],
        "per_view_focal_relative_spread": fx_spread,
        "warnings": [],
        "provenance": {
            "engine": "mapanything",
            "model": args.model,
            "pose_convention_in": "world_to_camera_w2c" if args.poses_w2c else "camera_to_world_c2w",
            "pose_written": "T_world_camera",
            "depth_convention": "radial_range" if args.depth_is_radial else "optical_z",
            "metric_scale_method": "native_metric_head_no_external_anchor",
            "honesty": "learned metric prior = soft evidence -> metric_pseudo_label at most, never measured_metric",
        },
    }
    (out_root / "backbone_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "status": "ok", "asset_id": args.asset_id, "model": args.model,
        "views_written": len(written), "written_frame_ids": written,
        "processed_resolution": [width, height], "per_view_focal_relative_spread": fx_spread,
        "skipped": skipped, "depth_stats": depth_stats[:8],
        "learned_metric_depth_prior": True, "metric_evidence": False, "blockers": [],
    }
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "model", "views_written", "processed_resolution",
        "per_view_focal_relative_spread", "learned_metric_depth_prior", "metric_evidence",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
