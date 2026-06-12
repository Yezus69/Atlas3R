"""Metric anchor council for learned metric priors.

The council compares a candidate reconstruction against independent metric
anchor artifacts already staged on disk. It estimates scale from real packet
geometry: same-frame depth agreement plus cross-view projection residuals after
a fixed-scale camera alignment. Learned anchors remain soft evidence; they
never become measured evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import FrameRayPacket, ScaleEvidence, ScaleEvidenceType
from .geometry_adapter import load_geometry_artifacts
from .m1 import _resolve_path

DEFAULT_EXTRA_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "anchor_id": "depth_anything_3_metric_backup",
        "artifacts_dir": "external/_da3_artifacts_backup",
        "anchor_family": "depth_anything_3_metric_depth",
        "license_status": "soft_prior_license_not_certified_by_local_manifest",
    },
    {
        "anchor_id": "depth_anything_3_metric_desk",
        "artifacts_dir": "external/_desk_da3",
        "anchor_family": "depth_anything_3_metric_depth",
        "license_status": "soft_prior_license_not_certified_by_local_manifest",
    },
)

MIN_COMMON_FRAMES = 2
MIN_SINGLE_FRAME_SAMPLES = 96
MIN_CROSS_VIEW_MATCHES = 24
FRAME_STABILITY_ACCEPT = 0.15
FRAME_STABILITY_WEAK = 0.25
OUTLIER_ACCEPT = 0.35
CROSS_VIEW_ACCEPT = 0.50
CROSS_VIEW_WEAK = 0.80
INBOUNDS_ACCEPT = 0.05
PIXEL_MATCH_NORM = 0.018
SINGLE_FRAME_CAP = 768
CROSS_VIEW_CAP = 384
CALIBRATED_LEARNED_SCALE_ENABLED = False


@dataclass(frozen=True)
class AnchorSpec:
    anchor_id: str
    artifacts_dir: str | Path
    anchor_family: str
    license_status: str


def run_metric_anchor_council(
    asset_id: str,
    candidate_packets: Sequence[FrameRayPacket],
    root: str | Path,
    *,
    primary_artifacts_dir: str | Path,
    extra_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[ScaleEvidence], dict[str, Any]]:
    """Return council-derived soft scale evidence and a JSONable report."""
    root_path = Path(root)
    specs = _anchor_specs(primary_artifacts_dir, extra_anchors)
    base: dict[str, Any] = {
        "module": "Metric Anchor Council",
        "asset_id": asset_id,
        "candidate_packet_count": len(candidate_packets),
        "anchor_specs": tuple(
            {
                "anchor_id": s.anchor_id,
                "artifacts_dir": str(s.artifacts_dir),
                "anchor_family": s.anchor_family,
                "license_status": s.license_status,
            }
            for s in specs
        ),
        "truth_boundary": (
            "learned metric anchors are soft ScaleEvidence only; measured_metric "
            "still requires measured evidence"
        ),
    }
    if not candidate_packets:
        return [], {
            **base,
            "status": "blocked_no_candidate_packets",
            "blockers": ("no_candidate_packets_for_metric_anchor_council",),
        }

    primary_resolved = _resolve_path(Path(primary_artifacts_dir), root_path)
    anchor_reports: list[dict[str, Any]] = []
    for spec in specs:
        resolved_dir = _resolve_path(Path(spec.artifacts_dir), root_path)
        self_anchor = resolved_dir == primary_resolved
        packets, adapter = load_geometry_artifacts(
            asset_id,
            root_path,
            artifacts_dir=spec.artifacts_dir,
            max_rays_per_packet=SINGLE_FRAME_CAP,
        )
        adapter_public = {k: v for k, v in adapter.items() if not str(k).startswith("_")}
        if not packets:
            anchor_reports.append(
                {
                    "anchor_id": spec.anchor_id,
                    "anchor_family": spec.anchor_family,
                    "artifacts_dir": str(resolved_dir),
                    "status": "absent",
                    "independent_metric_anchor": not self_anchor,
                    "consensus_role": "self_audit_only" if self_anchor else "independent_metric_anchor",
                    "adapter_status": adapter_public,
                    "blockers": tuple(adapter_public.get("blockers", ()))
                    or (f"missing_anchor_artifact:{spec.anchor_id}",),
                }
            )
            continue
        anchor_reports.append(
            _evaluate_anchor(
                asset_id, candidate_packets, packets, spec, adapter_public,
                self_anchor=self_anchor,
            )
        )

    consensus = _consensus(anchor_reports)
    evidence: list[ScaleEvidence] = []
    if consensus.get("status") != "no_usable_anchors":
        frame_ids = tuple(int(p.frame_id) for p in candidate_packets)
        scale_mean = float(consensus["scale_mean"])
        scale_std = float(consensus["scale_std"])
        confidence = float(consensus["confidence"])
        evidence.append(
            ScaleEvidence(
                evidence_id=f"{asset_id}_metric_anchor_council_consensus",
                evidence_type=ScaleEvidenceType.LEARNED_METRIC_DEPTH_PRIOR,
                measured=False,
                source="metric_anchor_council:learned_metric_priors",
                frame_ids=frame_ids,
                confidence=confidence,
                scale_mean=scale_mean,
                scale_std=scale_std,
                residual_after_optimization=consensus.get("residual_after_optimization"),
                provenance={
                    "council_status": consensus["status"],
                    "accepted_anchor_ids": tuple(consensus.get("accepted_anchor_ids", ())),
                    "weak_anchor_ids": tuple(consensus.get("weak_anchor_ids", ())),
                    "outlier_anchor_ids": tuple(consensus.get("outlier_anchor_ids", ())),
                    "anchor_count": len(anchor_reports),
                    "scale_mean_semantics": (
                        "multiplicative correction from candidate packet units "
                        "to the council's learned metric gauge"
                    ),
                    "uncertainty_basis": consensus.get("uncertainty_basis"),
                    "calibration_status": (
                        "uncalibrated_anchor_agreement_until_measured_split "
                        "coverage is reported"
                    ),
                },
            )
        )

    return evidence, {
        **base,
        "status": "computed",
        "anchor_reports": tuple(anchor_reports),
        "consensus": consensus,
        "scale_evidence_count": len(evidence),
        "scale_evidence_records": tuple(_evidence_summary(ev) for ev in evidence),
        "blockers": tuple(
            b
            for ar in anchor_reports
            if ar.get("status") == "absent"
            for b in ar.get("blockers", ())
        ),
    }


def _anchor_specs(
    primary_artifacts_dir: str | Path,
    extra_anchors: Sequence[Mapping[str, Any]] | None,
) -> tuple[AnchorSpec, ...]:
    specs = [
        AnchorSpec(
            anchor_id="primary_metric_backbone",
            artifacts_dir=primary_artifacts_dir,
            anchor_family="primary_backbone_native_metric_depth",
            license_status="read_from_artifact_manifest_when_available",
        )
    ]
    extra_source = extra_anchors if extra_anchors is not None else DEFAULT_EXTRA_ANCHORS
    for raw in extra_source:
        specs.append(
            AnchorSpec(
                anchor_id=str(raw.get("anchor_id", "unnamed_anchor")),
                artifacts_dir=str(raw.get("artifacts_dir", "")),
                anchor_family=str(raw.get("anchor_family", "learned_metric_depth")),
                license_status=str(raw.get("license_status", "not_certified")),
            )
        )
    # Deduplicate by (id, dir) so passing primary as an extra is harmless.
    seen: set[tuple[str, str]] = set()
    out: list[AnchorSpec] = []
    for spec in specs:
        key = (spec.anchor_id, str(spec.artifacts_dir))
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return tuple(out)


def _evaluate_anchor(
    asset_id: str,
    candidate_packets: Sequence[FrameRayPacket],
    anchor_packets: Sequence[FrameRayPacket],
    spec: AnchorSpec,
    adapter_report: Mapping[str, Any],
    *,
    self_anchor: bool,
) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        return {
            "anchor_id": spec.anchor_id,
            "anchor_family": spec.anchor_family,
            "status": "absent",
            "independent_metric_anchor": not self_anchor,
            "consensus_role": "self_audit_only" if self_anchor else "independent_metric_anchor",
            "blockers": (f"missing_numpy_for_metric_anchor_council:{exc.name}",),
        }

    learned_metric = bool(adapter_report.get("learned_metric_depth_prior", False))
    units = str(adapter_report.get("depth_units") or "")
    if not learned_metric or units != "meters":
        return {
            "anchor_id": spec.anchor_id,
            "anchor_family": spec.anchor_family,
            "status": "rejected",
            "verdict": "rejected_non_metric_anchor",
            "independent_metric_anchor": not self_anchor,
            "consensus_role": "self_audit_only" if self_anchor else "independent_metric_anchor",
            "depth_units": units,
            "learned_metric_depth_prior": learned_metric,
            "blockers": ("anchor_does_not_provide_learned_metric_depth_prior",),
        }

    cand_by = {int(p.frame_id): p for p in candidate_packets}
    anch_by = {int(p.frame_id): p for p in anchor_packets}
    common = sorted(set(cand_by) & set(anch_by))
    if len(common) < MIN_COMMON_FRAMES:
        return {
            "anchor_id": spec.anchor_id,
            "anchor_family": spec.anchor_family,
            "status": "rejected",
            "verdict": "rejected_insufficient_frame_overlap",
            "independent_metric_anchor": not self_anchor,
            "consensus_role": "self_audit_only" if self_anchor else "independent_metric_anchor",
            "common_frame_ids": tuple(common),
            "common_frame_count": len(common),
            "blockers": (f"metric_anchor_common_frames_too_low:{len(common)}",),
        }

    single = _single_frame_scale(cand_by, anch_by, common, np)
    if single["sample_count"] < MIN_SINGLE_FRAME_SAMPLES:
        return {
            "anchor_id": spec.anchor_id,
            "anchor_family": spec.anchor_family,
            "status": "rejected",
            "verdict": "rejected_insufficient_depth_overlap",
            "independent_metric_anchor": not self_anchor,
            "consensus_role": "self_audit_only" if self_anchor else "independent_metric_anchor",
            "common_frame_ids": tuple(common),
            "single_frame_metric_agreement": single,
            "blockers": (f"metric_anchor_depth_samples_too_low:{single['sample_count']}",),
        }

    scale = float(single["scale_mean"])
    cross = _cross_view_residual(cand_by, anch_by, common, scale, np)
    verdict = _anchor_verdict(single, cross)
    rel_unc = _anchor_relative_uncertainty(single, cross, verdict)
    confidence = _anchor_confidence(single, cross, verdict)
    blockers = _anchor_blockers(verdict, single, cross)
    status = verdict
    consensus_role = "independent_metric_anchor"
    if self_anchor:
        status = f"self_audit_{verdict}"
        consensus_role = "self_audit_only"
        blockers = ["self_anchor_same_artifact_family_not_metric_evidence", *blockers]

    return {
        "anchor_id": spec.anchor_id,
        "anchor_family": spec.anchor_family,
        "artifacts_dir": str(spec.artifacts_dir),
        "model_name": adapter_report.get("backbone_name"),
        "method": adapter_report.get("method"),
        "license_status": spec.license_status,
        "status": status,
        "verdict": verdict,
        "independent_metric_anchor": not self_anchor,
        "consensus_role": consensus_role,
        "estimated_metric_scale": scale,
        "scale_std": float(scale * rel_unc),
        "relative_scale_uncertainty": rel_unc,
        "confidence": confidence,
        "common_frame_ids": tuple(common),
        "common_frame_count": len(common),
        "single_frame_metric_agreement": single,
        "multi_view_metric_consistency": cross,
        "residual_risk_signature": _anchor_risk_signature(single, cross),
        "threshold_authority": (
            "pre_registered_conservative_anchor_weighting; acceptance gate still "
            "requires validation and measured calibration"
        ),
        "blockers": tuple(blockers),
    }


def _single_frame_scale(cand_by: Mapping[int, Any], anch_by: Mapping[int, Any], common: Sequence[int], np: Any) -> dict[str, Any]:
    ratios_all = []
    residuals_all = []
    per_frame = []
    for frame_id in common:
        c = _packet_samples(cand_by[frame_id], np, SINGLE_FRAME_CAP)
        a = _packet_samples(anch_by[frame_id], np, SINGLE_FRAME_CAP)
        pairs = _match_uv(c["uv"], a["uv"], np, PIXEL_MATCH_NORM)
        if pairs.shape[0] == 0:
            per_frame.append({"frame_id": frame_id, "sample_count": 0, "status": "no_pixel_overlap"})
            continue
        cr = c["depth"][pairs[:, 0]]
        ar = a["depth"][pairs[:, 1]]
        valid = np.isfinite(cr) & np.isfinite(ar) & (cr > 0.0) & (ar > 0.0)
        if not bool(np.any(valid)):
            per_frame.append({"frame_id": frame_id, "sample_count": 0, "status": "no_positive_depth_pairs"})
            continue
        ratios = ar[valid] / cr[valid]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        if ratios.size == 0:
            per_frame.append({"frame_id": frame_id, "sample_count": 0, "status": "no_finite_ratios"})
            continue
        frame_scale = float(np.median(ratios))
        frame_resid = np.abs(np.log(ratios) - np.log(frame_scale))
        ratios_all.append(ratios)
        residuals_all.append(frame_resid)
        per_frame.append(
            {
                "frame_id": int(frame_id),
                "status": "computed",
                "sample_count": int(ratios.size),
                "scale_median": frame_scale,
                "scale_p10": float(np.percentile(ratios, 10.0)),
                "scale_p90": float(np.percentile(ratios, 90.0)),
                "median_abs_log_residual": float(np.median(frame_resid)),
                "outlier_rate_log_gt_0p25": float(np.mean(frame_resid > 0.25)),
            }
        )
    if not ratios_all:
        return {
            "status": "not_computed",
            "sample_count": 0,
            "per_frame": tuple(per_frame),
        }
    ratios_cat = np.concatenate(ratios_all)
    residuals_cat = np.concatenate(residuals_all)
    frame_scales = np.asarray(
        [row["scale_median"] for row in per_frame if row.get("status") == "computed"],
        dtype=np.float64,
    )
    scale = float(np.median(ratios_cat))
    log_frame = np.log(frame_scales / scale) if frame_scales.size else np.zeros((0,), dtype=np.float64)
    return {
        "status": "computed",
        "scale_mean": scale,
        "sample_count": int(ratios_cat.size),
        "frame_count_used": int(frame_scales.size),
        "per_frame": tuple(per_frame),
        "frame_to_frame_log_scale_std": float(np.std(log_frame)) if log_frame.size else 0.0,
        "frame_to_frame_scale_relative_iqr": _relative_iqr(frame_scales, np),
        "median_abs_log_depth_residual": float(np.median(np.abs(np.log(ratios_cat) - np.log(scale)))),
        "outlier_rate_log_gt_0p25": float(np.mean(residuals_cat > 0.25)),
    }


def _cross_view_residual(
    cand_by: Mapping[int, Any],
    anch_by: Mapping[int, Any],
    common: Sequence[int],
    scale: float,
    np: Any,
) -> dict[str, Any]:
    if len(common) < 3:
        return {
            "status": "not_computed_insufficient_pose_overlap",
            "sample_count": 0,
            "inbounds_ratio": 0.0,
            "matched_ratio": 0.0,
        }
    R_align, t_align, pose_rmse = _fixed_scale_camera_alignment(cand_by, anch_by, common, scale, np)
    residuals = []
    attempted = 0
    inbounds = 0
    matched = 0
    pairs = [(common[i], common[i + 1]) for i in range(len(common) - 1)]
    pairs += [(b, a) for a, b in pairs]
    pair_rows = []
    for src_id, dst_id in pairs:
        cp = cand_by[src_id]
        ap = anch_by[dst_id]
        cs = _packet_samples(cp, np, CROSS_VIEW_CAP)
        ad = _packet_samples(ap, np, SINGLE_FRAME_CAP)
        Tci = np.asarray(cp.T_world_camera, dtype=np.float64).reshape((4, 4))
        Rai = np.asarray(ap.T_world_camera, dtype=np.float64).reshape((4, 4))
        Rc, tc = Tci[:3, :3], Tci[:3, 3]
        Ra, ta = Rai[:3, :3], Rai[:3, 3]
        rays = cs["rays"]
        depths = cs["depth"]
        n = min(rays.shape[0], depths.shape[0])
        if n == 0:
            continue
        attempted += int(n)
        Xc = rays[:n] * depths[:n, None]
        Xcw = Xc @ Rc.T + tc[None, :]
        Xaw = (R_align @ (scale * Xcw).T).T + t_align[None, :]
        Xac = (Xaw - ta[None, :]) @ Ra
        uv_pred = []
        d_pred = []
        for x in Xac:
            proj = ap.camera_model.project((float(x[0]), float(x[1]), float(x[2])))
            if not proj.get("valid"):
                uv_pred.append((float("nan"), float("nan")))
                d_pred.append(float("nan"))
                continue
            pix = proj.get("pixel_uv")
            if pix is None:
                uv_pred.append((float("nan"), float("nan")))
                d_pred.append(float("nan"))
                continue
            uv_pred.append(_norm_pixel(pix, ap))
            d_pred.append(float(proj.get("radial_depth_m", float("nan"))))
        uv_pred_arr = np.asarray(uv_pred, dtype=np.float64)
        d_pred_arr = np.asarray(d_pred, dtype=np.float64)
        finite = np.all(np.isfinite(uv_pred_arr), axis=1) & np.isfinite(d_pred_arr) & (d_pred_arr > 0.0)
        inbounds += int(np.count_nonzero(finite))
        if not bool(np.any(finite)):
            pair_rows.append({"source_frame_id": int(src_id), "target_frame_id": int(dst_id), "matched": 0})
            continue
        src_idx = np.nonzero(finite)[0]
        match = _match_uv(uv_pred_arr[finite], ad["uv"], np, PIXEL_MATCH_NORM)
        if match.shape[0] == 0:
            pair_rows.append({"source_frame_id": int(src_id), "target_frame_id": int(dst_id), "matched": 0})
            continue
        pred_i = src_idx[match[:, 0]]
        obs_j = match[:, 1]
        d_obs = ad["depth"][obs_j]
        valid = np.isfinite(d_obs) & (d_obs > 0.0)
        if bool(np.any(valid)):
            r = np.abs(np.log(d_pred_arr[pred_i][valid]) - np.log(d_obs[valid]))
            residuals.append(r)
            matched += int(r.size)
            pair_rows.append(
                {
                    "source_frame_id": int(src_id),
                    "target_frame_id": int(dst_id),
                    "matched": int(r.size),
                    "median_log_depth_residual": float(np.median(r)),
                }
            )
    if not residuals:
        return {
            "status": "not_computed_no_cross_view_matches",
            "sample_count": 0,
            "attempted_samples": attempted,
            "inbounds_ratio": float(inbounds / attempted) if attempted else 0.0,
            "matched_ratio": 0.0,
            "camera_alignment_rmse_after_fixed_scale": pose_rmse,
            "pairs": tuple(pair_rows),
        }
    res = np.concatenate(residuals)
    return {
        "status": "computed",
        "sample_count": int(res.size),
        "attempted_samples": attempted,
        "inbounds_ratio": float(inbounds / attempted) if attempted else 0.0,
        "matched_ratio": float(matched / attempted) if attempted else 0.0,
        "median_log_depth_residual": float(np.median(res)),
        "p90_log_depth_residual": float(np.percentile(res, 90.0)),
        "outlier_rate_log_gt_0p25": float(np.mean(res > 0.25)),
        "camera_alignment_rmse_after_fixed_scale": pose_rmse,
        "pairs": tuple(pair_rows),
    }


def _packet_samples(packet: FrameRayPacket, np: Any, cap: int) -> dict[str, Any]:
    rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
    depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
    n = min(rays.shape[0], depth.shape[0])
    rays = rays[:n]
    depth = depth[:n]
    if n > cap:
        idx = np.unique(np.rint(np.linspace(0, n - 1, cap)).astype(np.int64))
        rays = rays[idx]
        depth = depth[idx]
    uv = []
    for ray in rays:
        try:
            proj = packet.camera_model.project((float(ray[0]), float(ray[1]), float(ray[2])))
            pix = proj.get("pixel_uv") if isinstance(proj, Mapping) else None
            uv.append(_norm_pixel(pix, packet) if pix is not None else (float("nan"), float("nan")))
        except Exception:
            uv.append((float("nan"), float("nan")))
    uv_arr = np.asarray(uv, dtype=np.float64)
    finite = np.all(np.isfinite(uv_arr), axis=1) & np.isfinite(depth) & (depth > 0.0)
    return {"rays": rays[finite], "depth": depth[finite], "uv": uv_arr[finite]}


def _norm_pixel(pixel_uv: Any, packet: Any) -> tuple[float, float]:
    width = float(getattr(packet.camera_model, "width_px", 0.0) or 0.0)
    height = float(getattr(packet.camera_model, "height_px", 0.0) or 0.0)
    intr = getattr(packet, "intrinsics", None)
    if (width <= 0.0 or height <= 0.0) and isinstance(intr, Mapping):
        width = float(intr.get("width_px", width) or width)
        height = float(intr.get("height_px", height) or height)
    width = width if width > 0.0 else 1.0
    height = height if height > 0.0 else 1.0
    return float(pixel_uv[0]) / width, float(pixel_uv[1]) / height


def _match_uv(src_uv: Any, dst_uv: Any, np: Any, threshold: float) -> Any:
    if src_uv.shape[0] == 0 or dst_uv.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = []
    th2 = threshold * threshold
    for start in range(0, src_uv.shape[0], 128):
        chunk = src_uv[start:start + 128]
        d = ((chunk[:, None, :] - dst_uv[None, :, :]) ** 2).sum(axis=2)
        j = np.argmin(d, axis=1)
        best = d[np.arange(chunk.shape[0]), j]
        ok = best <= th2
        if bool(np.any(ok)):
            ii = np.nonzero(ok)[0] + start
            pairs.extend((int(i), int(k)) for i, k in zip(ii, j[ok]))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _fixed_scale_camera_alignment(cand_by: Mapping[int, Any], anch_by: Mapping[int, Any], common: Sequence[int], scale: float, np: Any):
    src = np.asarray([_center(cand_by[f], np) * scale for f in common], dtype=np.float64)
    dst = np.asarray([_center(anch_by[f], np) for f in common], dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    H = s0.T @ d0
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    aligned = (R @ src.T).T + t[None, :]
    rmse = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return R, t, rmse


def _center(packet: Any, np: Any) -> Any:
    T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
    return T[:3, 3]


def _anchor_verdict(single: Mapping[str, Any], cross: Mapping[str, Any]) -> str:
    stability = float(single.get("frame_to_frame_log_scale_std", 1.0))
    outlier = float(single.get("outlier_rate_log_gt_0p25", 1.0))
    cross_status = str(cross.get("status"))
    cross_resid = float(cross.get("median_log_depth_residual", 9.0)) if cross_status == "computed" else 9.0
    inbounds = float(cross.get("inbounds_ratio", 0.0))
    cross_samples = int(cross.get("sample_count", 0) or 0)
    if (
        stability <= FRAME_STABILITY_ACCEPT
        and outlier <= OUTLIER_ACCEPT
        and cross_resid <= CROSS_VIEW_ACCEPT
        and inbounds >= INBOUNDS_ACCEPT
        and cross_samples >= MIN_CROSS_VIEW_MATCHES
    ):
        return "accepted"
    if (
        stability <= FRAME_STABILITY_WEAK
        and outlier <= 0.55
        and (cross_status != "computed" or cross_resid <= CROSS_VIEW_WEAK)
    ):
        return "weak"
    return "rejected"


def _anchor_relative_uncertainty(single: Mapping[str, Any], cross: Mapping[str, Any], verdict: str) -> float:
    stability = float(single.get("frame_to_frame_log_scale_std", 0.5))
    single_res = float(single.get("median_abs_log_depth_residual", 0.5))
    cross_res = float(cross.get("median_log_depth_residual", single_res))
    base = max(0.03, stability, 0.5 * single_res, 0.5 * cross_res)
    if verdict == "accepted":
        return float(min(max(base, 0.04), 0.20))
    if verdict == "weak":
        return float(min(max(base, 0.12), 0.35))
    return float(min(max(base, 0.30), 0.75))


def _anchor_confidence(single: Mapping[str, Any], cross: Mapping[str, Any], verdict: str) -> float:
    if verdict == "rejected":
        return 0.05
    stability = float(single.get("frame_to_frame_log_scale_std", 0.5))
    outlier = float(single.get("outlier_rate_log_gt_0p25", 0.5))
    cross_res = float(cross.get("median_log_depth_residual", 0.8))
    score = 1.0 - ((stability / 0.30) + outlier + (cross_res / 1.0)) / 3.0
    if verdict == "weak":
        score *= 0.6
    return float(max(0.05, min(0.95, score)))


def _anchor_blockers(verdict: str, single: Mapping[str, Any], cross: Mapping[str, Any]) -> list[str]:
    if verdict == "accepted":
        return []
    reasons = []
    stability = float(single.get("frame_to_frame_log_scale_std", 1.0))
    outlier = float(single.get("outlier_rate_log_gt_0p25", 1.0))
    if stability > FRAME_STABILITY_ACCEPT:
        reasons.append(f"anchor_frame_scale_instability:{stability:.3f}")
    if outlier > OUTLIER_ACCEPT:
        reasons.append(f"anchor_single_frame_outlier_rate_high:{outlier:.3f}")
    if cross.get("status") != "computed":
        reasons.append(f"anchor_cross_view_not_computed:{cross.get('status')}")
    else:
        cross_res = float(cross.get("median_log_depth_residual", 9.0))
        inbounds = float(cross.get("inbounds_ratio", 0.0))
        if cross_res > CROSS_VIEW_ACCEPT:
            reasons.append(f"anchor_cross_view_residual_high:{cross_res:.3f}")
        if inbounds < INBOUNDS_ACCEPT:
            reasons.append(f"anchor_cross_view_inbounds_low:{inbounds:.3f}")
    return reasons


def _anchor_risk_signature(single: Mapping[str, Any], cross: Mapping[str, Any]) -> dict[str, Any]:
    stability = float(single.get("frame_to_frame_log_scale_std", 1.0))
    outlier = float(single.get("outlier_rate_log_gt_0p25", 1.0))
    cross_status = str(cross.get("status"))
    cross_resid = (
        float(cross.get("median_log_depth_residual", CROSS_VIEW_WEAK))
        if cross_status == "computed"
        else CROSS_VIEW_WEAK
    )
    inbounds = float(cross.get("inbounds_ratio", 0.0))
    samples = int(cross.get("sample_count", 0) or 0)
    components = {
        "frame_scale_instability": stability / FRAME_STABILITY_ACCEPT,
        "single_frame_outlier_rate": outlier / OUTLIER_ACCEPT,
        "cross_view_log_residual": cross_resid / CROSS_VIEW_ACCEPT,
        "cross_view_inbounds_deficit": max(0.0, (INBOUNDS_ACCEPT - inbounds) / INBOUNDS_ACCEPT),
        "cross_view_sample_deficit": max(0.0, (MIN_CROSS_VIEW_MATCHES - samples) / MIN_CROSS_VIEW_MATCHES),
    }
    risk = max(float(v) for v in components.values())
    return {
        "risk_score": float(risk),
        "nominal_support_passes": bool(risk <= 1.0),
        "components": components,
        "basis": (
            "max normalized residual; <=1 means inside pre-calibration anchor "
            "support, not metric truth"
        ),
    }


def _consensus(anchor_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        return {"status": "no_usable_anchors", "blockers": ("missing_numpy",)}

    usable = [
        a for a in anchor_reports
        if a.get("independent_metric_anchor", True)
        and a.get("status") in {"accepted", "weak"}
    ]
    if not usable:
        return _calibration_guard({
            "status": "no_usable_anchors",
            "scale_mean": 1.0,
            "scale_std": 0.30,
            "relative_scale_uncertainty": 0.30,
            "confidence": 0.0,
            "uncertainty_basis": "all_independent_anchor_families_absent_or_rejected",
            "self_audit_anchor_ids": tuple(
                str(a.get("anchor_id"))
                for a in anchor_reports
                if not a.get("independent_metric_anchor", True)
            ),
        })
    accepted = [a for a in usable if a.get("status") == "accepted"]
    pool = accepted if accepted else usable
    scales = np.asarray([float(a["estimated_metric_scale"]) for a in pool], dtype=np.float64)
    rels = np.asarray([float(a.get("relative_scale_uncertainty", 0.30)) for a in pool], dtype=np.float64)
    ids = [str(a["anchor_id"]) for a in pool]
    if scales.size == 1:
        rel = max(float(rels[0]), 0.12 if not accepted else 0.06)
        return _calibration_guard({
            "status": "single_anchor_consensus" if accepted else "single_weak_anchor",
            "scale_mean": float(scales[0]),
            "scale_std": float(scales[0] * rel),
            "relative_scale_uncertainty": float(rel),
            "confidence": float(pool[0].get("confidence", 0.2)),
            "accepted_anchor_ids": tuple(str(a["anchor_id"]) for a in accepted),
            "weak_anchor_ids": tuple(str(a["anchor_id"]) for a in usable if a.get("status") == "weak"),
            "outlier_anchor_ids": (),
            "residual_after_optimization": float(pool[0].get("multi_view_metric_consistency", {}).get("median_log_depth_residual", 0.0) or 0.0),
            "uncertainty_basis": "single learned metric anchor; floor keeps pseudo-label uncertainty",
        })
    med = float(np.median(scales))
    log_dev = np.abs(np.log(scales / med))
    inliers = log_dev <= max(0.18, float(np.median(log_dev) * 3.0))
    if int(np.count_nonzero(inliers)) < 2:
        disagreement = float(np.max(log_dev)) if log_dev.size else 0.30
        return _calibration_guard({
            "status": "weak_anchor_disagreement",
            "scale_mean": 1.0,
            "scale_std": float(max(0.30, disagreement)),
            "relative_scale_uncertainty": float(max(0.30, disagreement)),
            "confidence": 0.1,
            "accepted_anchor_ids": tuple(str(a["anchor_id"]) for a in accepted),
            "weak_anchor_ids": tuple(str(a["anchor_id"]) for a in usable if a.get("status") == "weak"),
            "outlier_anchor_ids": tuple(ids),
            "residual_after_optimization": disagreement,
            "uncertainty_basis": "independent learned anchors disagree; geometry remains at candidate gauge",
        })
    in_scales = scales[inliers]
    in_rels = rels[inliers]
    mad = float(np.median(np.abs(np.log(in_scales / np.median(in_scales)))))
    rel = max(float(np.median(in_rels)), 1.4826 * mad, 0.05)
    scale_mean = float(np.median(in_scales))
    return _calibration_guard({
        "status": "robust_anchor_consensus",
        "scale_mean": scale_mean,
        "scale_std": float(scale_mean * rel),
        "relative_scale_uncertainty": rel,
        "confidence": float(min(0.95, max(a.get("confidence", 0.0) for a in pool))),
        "accepted_anchor_ids": tuple(str(a["anchor_id"]) for a in accepted),
        "weak_anchor_ids": tuple(str(a["anchor_id"]) for a in usable if a.get("status") == "weak"),
        "outlier_anchor_ids": tuple(ids[i] for i, ok in enumerate(inliers) if not bool(ok)),
        "residual_after_optimization": float(max(a.get("multi_view_metric_consistency", {}).get("median_log_depth_residual", 0.0) or 0.0 for a in pool)),
        "uncertainty_basis": "median inlier anchor scale plus anchor disagreement MAD",
    })


def _calibration_guard(consensus: dict[str, Any]) -> dict[str, Any]:
    """Prevent uncalibrated learned anchors from silently setting metric scale."""
    if consensus.get("status") == "no_usable_anchors" or CALIBRATED_LEARNED_SCALE_ENABLED:
        return consensus
    raw = dict(consensus)
    rel = float(
        max(
            consensus.get("relative_scale_uncertainty", 0.0) or 0.0,
            (consensus.get("scale_std", 0.0) or 0.0) / max(consensus.get("scale_mean", 1.0) or 1.0, 1e-9),
            0.30,
        )
    )
    weak_ids = tuple(consensus.get("weak_anchor_ids", ())) + tuple(consensus.get("accepted_anchor_ids", ()))
    return {
        "status": "uncalibrated_anchor_reportage_only",
        "scale_mean": 1.0,
        "scale_std": rel,
        "relative_scale_uncertainty": rel,
        "confidence": min(float(consensus.get("confidence", 0.0) or 0.0), 0.10),
        "accepted_anchor_ids": (),
        "weak_anchor_ids": weak_ids,
        "outlier_anchor_ids": tuple(consensus.get("outlier_anchor_ids", ())),
        "residual_after_optimization": consensus.get("residual_after_optimization"),
        "uncertainty_basis": (
            "learned anchor scales are reportage until measured calibration "
            "shows predicted uncertainty covers true scale error"
        ),
        "raw_uncalibrated_consensus": raw,
    }


def _relative_iqr(values: Any, np: Any) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    if med <= 0.0:
        return 0.0
    return float((np.percentile(values, 75.0) - np.percentile(values, 25.0)) / med)


def _evidence_summary(ev: ScaleEvidence) -> dict[str, Any]:
    return {
        "evidence_id": ev.evidence_id,
        "evidence_type": ev.evidence_type.value,
        "measured": ev.measured,
        "source": ev.source,
        "confidence": ev.confidence,
        "scale_mean": ev.scale_mean,
        "scale_std": ev.scale_std,
        "residual_after_optimization": ev.residual_after_optimization,
        "provenance": dict(ev.provenance),
    }


__all__ = ["run_metric_anchor_council"]
