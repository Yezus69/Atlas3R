"""Real learned geometry backbone runner for Atlas3R M3 (Depth Anything 3).

Replaces classical SfM with learned multi-view geometry. Two real DA3 models are
combined:

- a GEOMETRY model (e.g. ``da3-large`` / ``da3nested-giant-large-1.1``) predicts,
  jointly over a set of views, per-pixel depth + camera extrinsics + intrinsics +
  confidence. Its depth is RELATIVE (similarity gauge).
- a METRIC model (``da3metric-large``) predicts monocular METRIC depth (meters).

The runner recovers a single global metric scale ``s`` by robustly matching the
metric model's depth to the geometry model's relative depth at the same pixels
(both run at the same processed resolution, so they are pixel-aligned), then
scales the whole reconstruction (depths and camera-center translations) by ``s``.
This is a learned metric PRIOR -- honest soft evidence
(``learned_metric_depth_prior``), never a measurement: the manifest sets
``metric_evidence=false`` so the teacher classifies at most
``metric_pseudo_label`` (never ``measured_metric``).

Conventions pinned from the installed DA3 source (not guessed):
- ``prediction.extrinsics`` is world-to-camera (w2c); we invert to
  ``T_world_camera`` (camera-to-world).
- ``prediction.depth`` is z-depth along the optical axis (optical_z); the M3
  adapter converts it to radial range.

Standalone tool. NOT imported by ``atlas3r``. Run with the isolated DA3 venv
interpreter (``external/da3_env``). No weights/repos vendored into git.

CHAIN GUARD: the default ``--out-dir`` is ``external/_raw_backbone_artifacts``,
which is NOT the teacher read path (``external/teacher_artifacts``). A raw
backbone artifact has independently-posed views; the teacher is meant to consume
the COLMAP-pose + MVS-verified COMPOSITE (tools/run_colmap_pose_backend.py ->
tools/run_mvs_depth_backend.py --promote). Pass ``--out-dir
external/teacher_artifacts`` only when you explicitly want the teacher to eat
the raw artifact.

Usage:
    external/da3_env/Scripts/python.exe tools/run_da3_backbone.py \
        --asset-id phone_room --frames-dir data/phone_room \
        --frame-glob "frame_*.jpg" --num-frames 32 \
        --geometry-model depth-anything/DA3-LARGE \
        --metric-model depth-anything/DA3METRIC-LARGE \
        --device cuda --out-dir external/_raw_backbone_artifacts
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

GEOMETRY_FALLBACK = "depth-anything/DA3-LARGE"


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


def _w2c_to_T_world_camera(extr_w2c, np):
    """Build a 4x4 T_world_camera (camera-to-world) from a DA3 [3,4] or [4,4]
    world-to-camera extrinsic by inverting it."""
    w2c = np.eye(4, dtype=np.float64)
    arr = np.asarray(extr_w2c, dtype=np.float64)
    w2c[:3, :4] = arr[:3, :4]
    return np.linalg.inv(w2c)


def _run_geometry(repo, image_paths, process_res, device, np, torch):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(repo).to(device)
    model.device = torch.device(device)
    model.eval()
    pred = model.inference(image_paths, process_res=process_res, export_dir=None)
    depth = np.asarray(pred.depth, dtype=np.float64)
    extr = None if pred.extrinsics is None else np.asarray(pred.extrinsics, dtype=np.float64)
    intr = None if pred.intrinsics is None else np.asarray(pred.intrinsics, dtype=np.float64)
    conf = None if pred.conf is None else np.asarray(pred.conf, dtype=np.float64)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return depth, extr, intr, conf


def _run_metric_depth(repo, image_paths, process_res, device, np, torch):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(repo).to(device)
    model.device = torch.device(device)
    model.eval()
    pred = model.inference(image_paths, process_res=process_res, export_dir=None)
    depth = np.asarray(pred.depth, dtype=np.float64)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return depth


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DA3 learned geometry backbone runner for Atlas3R M3.")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-glob", default="*.jpg")
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--geometry-model", default="depth-anything/DA3-LARGE")
    parser.add_argument("--metric-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--no-metric-scale", action="store_true", help="Skip metric anchoring (stay non-metric).")
    parser.add_argument(
        "--out-dir", default="external/_raw_backbone_artifacts",
        help="artifact root; default is the RAW staging area, NOT the teacher read "
             "path (external/teacher_artifacts). The teacher should consume the "
             "pose-corrected composite (run_colmap_pose_backend.py -> "
             "run_mvs_depth_backend.py --promote); pass external/teacher_artifacts "
             "explicitly to opt the raw artifact into the teacher read path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--conf-quantile", type=float, default=0.5, help="Keep pixels above this conf quantile for scale match.")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    frames_dir = Path(args.frames_dir)
    all_frames = sorted((p for p in frames_dir.glob(args.frame_glob) if p.is_file()), key=_natural_key)
    if len(all_frames) < 2:
        print(json.dumps({"status": "error", "blocker": "fewer_than_two_frames"}))
        return 2
    if args.frame_ids:
        ids = sorted({int(x) for x in args.frame_ids.split(",") if x.strip()})
        ids = [i for i in ids if 0 <= i < len(all_frames)]
    else:
        ids = _evenly_spaced(len(all_frames), args.num_frames)
    if len(ids) < 2:
        print(json.dumps({"status": "error", "blocker": "fewer_than_two_selected_frames"}))
        return 2
    image_paths = [str(all_frames[i]) for i in ids]

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # --- geometry model (poses + relative depth + intrinsics + conf) ---
    geometry_model_used = args.geometry_model
    warnings = []
    try:
        depth_rel, extr, intr, conf = _run_geometry(args.geometry_model, image_paths, args.process_res, device, np, torch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        warnings.append(f"geometry_model_oom_fellback:{args.geometry_model}->{GEOMETRY_FALLBACK}")
        geometry_model_used = GEOMETRY_FALLBACK
        depth_rel, extr, intr, conf = _run_geometry(GEOMETRY_FALLBACK, image_paths, args.process_res, device, np, torch)

    if extr is None or intr is None:
        print(json.dumps({
            "status": "error",
            "blocker": "geometry_model_returned_no_camera_poses",
            "hint": "use a multi-view geometry model (e.g. DA3-LARGE / DA3NESTED-GIANT-LARGE-1.1), not a depth-only model",
            "geometry_model": geometry_model_used,
        }))
        return 5

    n, height, width = depth_rel.shape

    # --- metric anchoring via a learned metric-depth model (soft prior) ---
    metric_scale = None
    metric_scale_spread = None
    metric_model_used = None
    per_frame_scale = []
    if not args.no_metric_scale:
        try:
            depth_metric = _run_metric_depth(args.metric_model, image_paths, args.process_res, device, np, torch)
            metric_model_used = args.metric_model
            if depth_metric.shape == depth_rel.shape:
                if conf is not None:
                    conf_thr = float(np.quantile(conf[np.isfinite(conf)], args.conf_quantile)) if np.any(np.isfinite(conf)) else -np.inf
                else:
                    conf_thr = -np.inf
                for k in range(n):
                    dr, dm = depth_rel[k], depth_metric[k]
                    mask = np.isfinite(dr) & (dr > 1e-6) & np.isfinite(dm) & (dm > 1e-6)
                    if conf is not None:
                        mask = mask & (conf[k] >= conf_thr)
                    if int(np.count_nonzero(mask)) >= 50:
                        per_frame_scale.append(float(np.median(dm[mask] / dr[mask])))
                if len(per_frame_scale) >= max(2, n // 3):
                    sarr = np.asarray(per_frame_scale, dtype=np.float64)
                    metric_scale = float(np.median(sarr))
                    metric_scale_spread = float(np.std(sarr) / max(abs(np.median(sarr)), 1e-9))
                else:
                    warnings.append("metric_scale_too_few_consistent_frames_staying_non_metric")
            else:
                warnings.append(f"metric_depth_shape_mismatch_{depth_metric.shape}_vs_{depth_rel.shape}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.append("metric_model_oom_staying_non_metric")
        except Exception as exc:  # robust: metric anchoring is best-effort
            warnings.append(f"metric_anchor_failed:{type(exc).__name__}:{exc}")

    apply_metric = metric_scale is not None and metric_scale > 0
    s = metric_scale if apply_metric else 1.0
    units = "meters" if apply_metric else "unitless_similarity"

    # --- write artifacts ---
    out_root = Path(args.out_dir) / args.asset_id
    depth_dir, conf_dir = out_root / "depth", out_root / "confidence"
    for d in (depth_dir, conf_dir):
        if d.exists():
            shutil.rmtree(d)
    for d in (out_root, depth_dir, conf_dir):
        d.mkdir(parents=True, exist_ok=True)

    fxs, fys = intr[:, 0, 0], intr[:, 1, 1]
    cxs, cys = intr[:, 0, 2], intr[:, 1, 2]
    fx, fy, cx, cy = float(np.median(fxs)), float(np.median(fys)), float(np.median(cxs)), float(np.median(cys))
    fx_spread = float(np.std(fxs) / max(abs(np.median(fxs)), 1e-9))

    pose_frames, written, skipped, depth_stats, per_frame_intr = [], [], [], [], {}
    for k, frame_id in enumerate(ids):
        d = depth_rel[k] * s
        finite_pos = np.isfinite(d) & (d > 0.0)
        if int(np.count_nonzero(finite_pos)) == 0:
            skipped.append({"frame_id": int(frame_id), "reason": "no_finite_positive_depth"})
            continue
        Tw = _w2c_to_T_world_camera(extr[k], np)
        if not np.all(np.isfinite(Tw)):
            skipped.append({"frame_id": int(frame_id), "reason": "non_finite_pose"})
            continue
        Tw[:3, 3] = Tw[:3, 3] * s  # scale camera-center translation into metric gauge

        np.save(depth_dir / f"{frame_id}.npy", np.where(finite_pos, d, 0.0).astype(np.float32))
        if conf is not None:
            c = conf[k].astype(np.float64)
            hi = float(np.percentile(c[np.isfinite(c)], 99)) if np.any(np.isfinite(c)) else 1.0
            hi = hi if hi > 0 else 1.0
            np.save(conf_dir / f"{frame_id}.npy", np.clip(np.where(np.isfinite(c), c, 0.0) / hi, 0.0, 1.0).astype(np.float32))

        pose_frames.append({
            "frame_id": int(frame_id),
            "source_frame_path": image_paths[k],
            "T_world_camera": [[float(v) for v in row] for row in Tw.tolist()],
        })
        per_frame_intr[str(int(frame_id))] = {
            "fx": float(intr[k, 0, 0]), "fy": float(intr[k, 1, 1]),
            "cx": float(intr[k, 0, 2]), "cy": float(intr[k, 1, 2]),
        }
        dv = d[finite_pos]
        depth_stats.append({"frame_id": int(frame_id), "min": float(dv.min()),
                            "median": float(np.median(dv)), "max": float(dv.max())})
        written.append(int(frame_id))

    if len(written) < 2:
        report = {"status": "error", "blocker": "fewer_than_two_views_with_valid_geometry",
                  "written": written, "skipped": skipped, "warnings": warnings}
        (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 4

    intrinsics_payload = {
        "model": "pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width_px": int(width), "height_px": int(height),
        "pixel_coordinate_convention": "top_left_pixel_centers",
        "note": "median of DA3 per-view intrinsics at processed resolution",
    }
    (out_root / "intrinsics.json").write_text(json.dumps(intrinsics_payload, indent=2), encoding="utf-8")
    (out_root / "per_frame_intrinsics.json").write_text(json.dumps(per_frame_intr, indent=2), encoding="utf-8")
    (out_root / "poses.json").write_text(json.dumps({
        "convention": "T_world_camera", "translation_units": units,
        "source": "depth_anything_3_inverse_of_w2c_extrinsics_metric_scaled" if apply_metric
        else "depth_anything_3_inverse_of_w2c_extrinsics",
        "frames": pose_frames,
    }, indent=2), encoding="utf-8")
    (out_root / "depth_meta.json").write_text(json.dumps({
        "depth_convention": "optical_z", "units": units, "scale_to_meters_or_null": None,
        "density": "dense_learned_multiview", "model_name": geometry_model_used,
        "metric_anchor_model": metric_model_used, "metric_scale_applied": apply_metric,
    }, indent=2), encoding="utf-8")

    # Confidence for the learned metric prior: tighter cross-frame scale agreement
    # -> higher confidence. Conservative; this is a soft prior, not a measurement.
    if apply_metric:
        spread = metric_scale_spread if metric_scale_spread is not None else 0.5
        scale_conf = max(0.2, min(0.85, 1.0 - 2.0 * spread))
    else:
        scale_conf = 0.0

    manifest = {
        "backbone_name": "depth_anything_3",
        "method": f"da3_multiview_geometry:{geometry_model_used}"
        + (f"+metric_anchor:{metric_model_used}" if apply_metric else ""),
        "model_name": geometry_model_used,
        "asset_id": args.asset_id,
        "frame_ids": written,
        "pose_units": units,
        "metric_evidence": False,  # learned prior is NOT a measurement
        "learned_metric_depth_prior": bool(apply_metric),
        "scale_evidence_confidence": float(scale_conf),
        "camera_confidence": 0.7,
        "metric_scale": None if not apply_metric else float(metric_scale),
        "metric_scale_relative_spread": metric_scale_spread,
        "per_frame_metric_scale": per_frame_scale,
        "processed_resolution": [int(width), int(height)],
        "per_view_focal_relative_spread": fx_spread,
        "warnings": warnings,
        "provenance": {
            "engine": "depth_anything_3",
            "geometry_model": geometry_model_used,
            "metric_anchor_model": metric_model_used,
            "extrinsics_convention_in": "world_to_camera_w2c",
            "pose_written": "T_world_camera_inverse_of_w2c",
            "depth_convention": "optical_z",
            "metric_scale_method": "robust_median_ratio_metric_over_relative_depth_per_pixel",
            "honesty": "learned metric prior = soft evidence -> metric_pseudo_label at most, never measured_metric",
        },
    }
    (out_root / "backbone_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "status": "ok", "asset_id": args.asset_id,
        "geometry_model": geometry_model_used, "metric_anchor_model": metric_model_used,
        "metric_scale_applied": apply_metric, "metric_scale": metric_scale,
        "metric_scale_relative_spread": metric_scale_spread,
        "depth_units": units, "views_written": len(written), "written_frame_ids": written,
        "processed_resolution": [int(width), int(height)],
        "per_view_focal_relative_spread": fx_spread, "skipped": skipped, "warnings": warnings,
        "learned_metric_depth_prior": bool(apply_metric), "scale_evidence_confidence": scale_conf,
        "depth_stats": depth_stats[:8], "blockers": [],
    }
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "geometry_model", "metric_anchor_model", "metric_scale_applied", "metric_scale",
        "metric_scale_relative_spread", "depth_units", "views_written", "processed_resolution",
        "per_view_focal_relative_spread", "learned_metric_depth_prior", "scale_evidence_confidence",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
