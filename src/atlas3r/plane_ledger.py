"""Plane-ledger drift verifier (GT-free, ARCHITECTURE.md Module 11 instrument).

First principles: the static world is RIGID -- a physical plane (floor, wall,
table top) is the same plane every time it is observed. Pairwise consistency
signals are blind to COHERENT pose drift because drift moves each camera and
its attached depth together (measured: the injection harness showed a
0.4x-trajectory-span drift invisible to every internal gate signal). What
drift cannot preserve is the world-frame constancy of revisited structure.

The ledger:
1. Per frame, extract dominant planes from the backbone's sampled depth points
   in CAMERA coordinates (sequential RANSAC; no photometrics -- works under
   the motion blur that starves feature-based auditors).
2. Transform each plane through the pose under audit:
   ``n_w = R n_c``, ``d_w = d_c + n_w . t``  (plane convention ``n . X = d``).
3. Chain frame-to-frame into PLANE TRACKS (adjacent-frame drift is tiny, so
   association needs no global matching).
4. Audit each track: in a correct reconstruction the world plane is CONSTANT.
   Systematic in-track trends read out exactly the corruption families the
   gate is measurably blind to:
   - offset linear trend            -> translation-drift gauge
   - normal rotation along track    -> rotation-drift gauge (degrees)
   - |offset| multiplicative ramp   -> scale-drift gauge

Honesty rules:
- This instrument shares the backbone's DEPTH (it is not depth-independent).
  Its audit axis is the TEMPORAL COHERENCE of rigid structure through the
  poses; per-frame depth bias that is viewpoint-stable cancels inside a track.
  Authority is assigned ONLY by the injection detection-limit harness
  (measured response curves), never assumed.
- Evidence mass: gauges aggregate only tracks with sufficient lifetime and
  inlier support; a scene with no qualifying tracks yields an explicit
  ``abstained_no_persistent_planes`` status -- abstention is authority loss,
  never a pass.
- Offsets are reconstruction units; scene-relative gauges are normalized by
  the trajectory span. No metric claim without an anchor (gauge freedom).
- REPORTAGE ONLY: nothing here gates acceptance until calibrated authority is
  measured and promoted under the repo's validation discipline.

``numpy`` is imported lazily inside functions (repo convention).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Per-frame RANSAC
MAX_PLANES_PER_FRAME = 4
RANSAC_ITERATIONS = 200
MIN_PLANE_INLIERS = 80
# Inlier threshold: fraction of the frame's median depth (scene-relative).
INLIER_DEPTH_FRACTION = 0.015
MIN_POINT_CONFIDENCE = 0.2
RANSAC_SEED = 1450  # deterministic; one fixed seed, never tuned per scene

# Track association (adjacent frames; one-frame gap bridging allowed)
ASSOC_MAX_NORMAL_ANGLE_DEG = 10.0
ASSOC_MAX_OFFSET_FRACTION_OF_SPAN = 0.15
MAX_FRAME_GAP = 2

# Evidence-mass floors for a track to count toward scene gauges
MIN_TRACK_LIFETIME_FRAMES = 4
MIN_TRACKS_FOR_AUTHORITY = 2


def extract_frame_planes(packet: Any, np: Any, rng: Any) -> list[dict[str, Any]]:
    """Sequential-RANSAC plane extraction from one packet's sampled points.

    Returns up to ``MAX_PLANES_PER_FRAME`` planes in CAMERA coordinates as
    ``{n_cam, d_cam, inliers, rms}`` with ``n . X = d`` and ``d >= 0``.
    """
    rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape(-1)
    conf = np.asarray(packet.confidence, dtype=np.float64).reshape(-1)
    keep = np.isfinite(depth) & (depth > 0) & (conf >= MIN_POINT_CONFIDENCE)
    points = rays[keep] * depth[keep][:, None]
    if points.shape[0] < MIN_PLANE_INLIERS:
        return []
    eps = INLIER_DEPTH_FRACTION * float(np.median(depth[keep]))

    planes: list[dict[str, Any]] = []
    remaining = points
    for _ in range(MAX_PLANES_PER_FRAME):
        n_pts = remaining.shape[0]
        if n_pts < MIN_PLANE_INLIERS:
            break
        best_inliers = None
        best_count = 0
        for _ in range(RANSAC_ITERATIONS):
            idx = rng.choice(n_pts, size=3, replace=False)
            p0, p1, p2 = remaining[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            dist = np.abs((remaining - p0) @ normal)
            inliers = dist < eps
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
        if best_inliers is None or best_count < MIN_PLANE_INLIERS:
            break
        # Least-squares refinement on inliers (SVD of centered points).
        pts = remaining[best_inliers]
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        n = vt[-1]
        n = n / np.linalg.norm(n)
        d = float(n @ centroid)
        if d < 0:
            n, d = -n, -d
        rms = float(np.sqrt(((pts @ n - d) ** 2).mean()))
        planes.append({
            "n_cam": [float(v) for v in n],
            "d_cam": d,
            "inliers": int(best_count),
            "rms": rms,
        })
        remaining = remaining[~best_inliers]
    return planes


def _to_world(plane: dict[str, Any], T: Any, np: Any) -> tuple[Any, float]:
    R, t = T[:3, :3], T[:3, 3]
    n_w = R @ np.asarray(plane["n_cam"])
    d_w = float(plane["d_cam"] + n_w @ t)
    if d_w < 0:
        n_w, d_w = -n_w, -d_w
    return n_w, d_w


def build_ledger(
    packets: Sequence[Any],
    *,
    trajectory_span: float,
) -> dict[str, Any]:
    """Extract planes, chain tracks, audit constancy. Pure function of the
    packets (geometry + poses under audit); no photometrics, no measured data.
    """
    import numpy as np

    rng = np.random.default_rng(RANSAC_SEED)
    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    span = max(trajectory_span, 1e-9)

    # Per-frame world planes.
    frames: list[dict[str, Any]] = []
    for packet in ordered:
        T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape(4, 4)
        cam_planes = extract_frame_planes(packet, np, rng)
        world_planes = []
        for plane in cam_planes:
            n_w, d_w = _to_world(plane, T, np)
            world_planes.append({
                "n": n_w, "d": d_w,
                "inliers": plane["inliers"], "rms": plane["rms"],
            })
        frames.append({"frame_id": int(packet.frame_id), "planes": world_planes})

    # Greedy frame-to-frame chaining (gap bridging up to MAX_FRAME_GAP).
    max_angle = np.deg2rad(ASSOC_MAX_NORMAL_ANGLE_DEG)
    max_offset = ASSOC_MAX_OFFSET_FRACTION_OF_SPAN * span
    tracks: list[dict[str, Any]] = []
    open_tracks: list[dict[str, Any]] = []
    for f_idx, frame in enumerate(frames):
        unmatched = list(range(len(frame["planes"])))
        for track in open_tracks:
            if f_idx - track["last_frame_idx"] > MAX_FRAME_GAP:
                continue
            last = track["observations"][-1]
            best_j, best_angle = None, max_angle
            for j in unmatched:
                cand = frame["planes"][j]
                cosang = float(np.clip(np.dot(last["n"], cand["n"]), -1.0, 1.0))
                angle = float(np.arccos(abs(cosang)))  # sign-agnostic direction
                if angle < best_angle and abs(cand["d"] - last["d"]) < max_offset:
                    best_angle, best_j = angle, j
            if best_j is not None:
                obs = frame["planes"][best_j]
                # Resolve sign so the track's normals stay on one side.
                if float(np.dot(last["n"], obs["n"])) < 0:
                    obs = {**obs, "n": -obs["n"], "d": -obs["d"]}
                track["observations"].append({**obs, "frame_idx": f_idx})
                track["last_frame_idx"] = f_idx
                unmatched.remove(best_j)
        for j in unmatched:
            obs = frame["planes"][j]
            open_tracks.append({
                "observations": [{**obs, "frame_idx": f_idx}],
                "last_frame_idx": f_idx,
            })
        # Retire stale tracks.
        still_open = []
        for track in open_tracks:
            if f_idx - track["last_frame_idx"] > MAX_FRAME_GAP:
                tracks.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open
    tracks.extend(open_tracks)

    # Audit each qualifying track.
    audited = []
    for track in tracks:
        obs = track["observations"]
        if len(obs) < MIN_TRACK_LIFETIME_FRAMES:
            continue
        f = np.asarray([o["frame_idx"] for o in obs], dtype=np.float64)
        d = np.asarray([o["d"] for o in obs], dtype=np.float64)
        normals = np.asarray([o["n"] for o in obs], dtype=np.float64)
        weights = np.asarray([o["inliers"] for o in obs], dtype=np.float64)

        # Offset linear trend over the track's life (coherent translation/scale
        # drift signature); residual std is the fit noise floor.
        A = np.stack([f - f.mean(), np.ones_like(f)], axis=1)
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        slope = float(coef[0])
        trend = abs(slope) * (f.max() - f.min())
        resid_std = float(np.std(d - A @ coef))

        # Rotation drift: mean normal of the first vs last third of the track.
        third = max(1, len(obs) // 3)
        n_a = normals[:third].mean(axis=0)
        n_b = normals[-third:].mean(axis=0)
        n_a /= np.linalg.norm(n_a)
        n_b /= np.linalg.norm(n_b)
        normal_drift_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(n_a, n_b)), -1.0, 1.0))))

        # Scale-drift signature: |offset| ratio late/early (multiplicative).
        d_a = float(np.median(np.abs(d[:third])))
        d_b = float(np.median(np.abs(d[-third:])))
        offset_ratio = (d_b / d_a) if d_a > 1e-9 else None

        lifetime = float(f.max() - f.min()) or 1.0
        audited.append({
            "lifetime_frames": len(obs),
            "mean_inliers": float(weights.mean()),
            "offset_trend": trend,
            "offset_trend_span_fraction": trend / span,
            "offset_rate_span_fraction_per_frame": (trend / span) / lifetime,
            "offset_resid_std_span_fraction": resid_std / span,
            "normal_drift_deg": normal_drift_deg,
            "offset_ratio_late_over_early": offset_ratio,
        })

    base = {
        "module": "plane_ledger - rigid-world drift verifier",
        "n_frames": len(frames),
        "n_frames_with_planes": sum(1 for fr in frames if fr["planes"]),
        "n_tracks_total": len(tracks),
        "n_tracks_qualifying": len(audited),
        "trajectory_span_reconstruction_units": span,
        "params": {
            "min_track_lifetime_frames": MIN_TRACK_LIFETIME_FRAMES,
            "min_plane_inliers": MIN_PLANE_INLIERS,
            "inlier_depth_fraction": INLIER_DEPTH_FRACTION,
            "ransac_seed": RANSAC_SEED,
        },
        "authority_note": (
            "reportage only; authority comes from injected-corruption response "
            "curves, never assumed. Shares backbone depth; audits temporal "
            "coherence of rigid structure through the poses under audit."
        ),
    }
    if len(audited) < MIN_TRACKS_FOR_AUTHORITY:
        return {
            **base,
            "status": "abstained_no_persistent_planes",
            "authority": "none",
            "note": (
                "fewer than the minimum qualifying plane tracks -- abstention "
                "is authority loss, never a pass"
            ),
        }

    def weighted_p90(values: Any) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), 90.0))

    trend_fracs = [a["offset_trend_span_fraction"] for a in audited]
    rate_fracs = [a["offset_rate_span_fraction_per_frame"] for a in audited]
    normal_drifts = [a["normal_drift_deg"] for a in audited]
    noise_floors = [a["offset_resid_std_span_fraction"] for a in audited]
    ratios = [a["offset_ratio_late_over_early"] for a in audited
              if isinstance(a["offset_ratio_late_over_early"], float)]
    ratio_dev = [abs(np.log(r)) for r in ratios if r and r > 0]

    # Direction-resolved authority (red-team requirement): a plane track is
    # blind to translation drift PERPENDICULAR to its normal. Report the
    # conditioning of the inlier-weighted normal span; the smallest principal
    # direction is the ledger's translation BLIND axis on this scene. A scene
    # tracking only floor+one-wall has a near-degenerate span -- the report
    # says so instead of letting random-axis injections fabricate
    # direction-averaged authority.
    all_normals, all_weights = [], []
    for track in tracks:
        for o in track["observations"]:
            all_normals.append(o["n"])
            all_weights.append(o["inliers"])
    N = np.asarray(all_normals, dtype=np.float64)
    w = np.sqrt(np.asarray(all_weights, dtype=np.float64))
    _, svals, vt = np.linalg.svd(N * w[:, None], full_matrices=False)
    svals = svals / max(float(svals[0]), 1e-12)
    blind_axis = [float(v) for v in vt[-1]]
    dominant_axis = [float(v) for v in vt[0]]

    return {
        **base,
        "status": "audited",
        "authority": "uncalibrated_reportage",
        "signals": {
            # Scene-level gauges (p90 across qualifying tracks). Span-relative
            # AND per-frame rate forms: a track witnesses only its lifetime's
            # fraction of an end-to-end ramp, so the rate is the cross-scene
            # comparable quantity; the trend is the witnessed total.
            "ledger_offset_drift_p90_span_fraction": weighted_p90(trend_fracs),
            "ledger_offset_rate_p90_span_fraction_per_frame": weighted_p90(rate_fracs),
            "ledger_normal_drift_p90_deg": weighted_p90(normal_drifts),
            "ledger_scale_ramp_p90_abs_log_ratio": (
                weighted_p90(ratio_dev) if ratio_dev else None
            ),
            # The ledger's own noise floor (residual around per-track trends):
            # a drift gauge below this floor has no authority.
            "ledger_noise_floor_p90_span_fraction": weighted_p90(noise_floors),
        },
        "direction_authority": {
            "normal_span_singular_values_relative": [float(s) for s in svals],
            "translation_blind_axis_world": blind_axis,
            "translation_dominant_axis_world": dominant_axis,
            "note": (
                "translation-drift authority exists only along directions the "
                "tracked normals span; drift along translation_blind_axis_world "
                "is invisible to this ledger on this scene"
            ),
        },
        "tracks": audited,
    }


def ledger_for_packets(packets: Sequence[Any]) -> dict[str, Any]:
    """Convenience wrapper: computes the trajectory span internally."""
    from .inject import trajectory_span_m

    return build_ledger(packets, trajectory_span=trajectory_span_m(packets))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys as _sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="reference_metric")
    parser.add_argument("--artifacts-dir", default="external/teacher_artifacts")
    args = parser.parse_args(argv)

    root = Path.cwd()
    _sys.path.insert(0, str(root / "src"))
    from atlas3r.geometry_adapter import load_geometry_artifacts
    from atlas3r.refine import refine_scene

    mono, grep = load_geometry_artifacts(args.asset, root, artifacts_dir=args.artifacts_dir)
    soft = grep.get("_scale_evidence", [])
    refined, _ = refine_scene(mono, fix_global_scale=bool(soft))
    report = ledger_for_packets(refined)

    out = root / "runs/_diag" / f"plane_ledger_{args.asset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tracks"}, indent=2, default=str))
    print(f"[plane_ledger] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
