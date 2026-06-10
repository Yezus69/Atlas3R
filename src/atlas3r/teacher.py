"""Top-level Atlas3R teacher spine (M3 -> M8).

``python -m atlas3r.teacher`` runs the full teacher over every canonical asset:

  M1 asset availability
  -> M2 measured reference packets (reference_metric)
  -> M3 geometry adapter (external monocular artifacts)
  -> M4 visibility / residual graph
  -> M5 scale posterior + metric classification
  -> M7 ray-fused static map + occupancy grid
  -> M8 validation + final honest category

Every stage is wrapped: a missing input becomes that stage's
missing/blocked status and an exact blocker -- never a crash, never fabricated
data. Per-track and summary reports are written under ``runs/teacher/``.

Package import stays dependency-free; ``numpy`` etc. are imported lazily by the
stage modules.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import RobotEnvelopeConfig, load_robot_envelope
from .contracts import TrackType, VideoAsset
from .export import export_teacher_artifacts
from .geometry_adapter import load_geometry_artifacts, load_measured_packets_from_m2
from .m1 import DEFAULT_MANIFEST_PATH, _jsonable, _resolve_path, _write_json, load_canonical_assets
from .mapping import fuse_static_map
from .refine import refine_scene
from .scale import estimate_scale_posterior
from .static_dynamic import infer_static_dynamic
from .validation import validate_and_accept
from .visibility import build_visibility_graph
from .visualize import write_visual_proof

DEFAULT_OUTPUT_DIR = Path("runs/teacher")
DEFAULT_ARTIFACTS_DIR = "external/teacher_artifacts"
DEFAULT_M1_DIR = "runs/m1"
DEFAULT_M2_DIR = "runs/m2"
# Score static/dynamic with at least this many samples per frame so the per-pixel
# state arrays align 1:1 with the packet rays (packets hold <=2048 rays), which
# lets fuse_static_map exclude dynamic pixels without fabricated alignment.
STATIC_DYNAMIC_SAMPLES_PER_FRAME = 4096
# Robot-scale tolerance for the band agreement: a measured band voxel matches the
# nearest OBSERVED candidate voxel within this distance. Two independently sparse
# (sampled-ray) fields rarely hit the exact same voxel, so an exact-voxel match is
# dominated by sampling noise; a ~10 cm tolerance is the right granularity for a
# floor-cleaning occupancy comparison. Both exact and tolerant coverage are
# reported so nothing is hidden.
BAND_MATCH_TOLERANCE_M = 0.10


def run_teacher(
    manifest: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    root: str | Path | None = None,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    m1_dir: str | Path = DEFAULT_M1_DIR,
    m2_dir: str | Path = DEFAULT_M2_DIR,
) -> dict[str, Any]:
    root_path = Path.cwd() if root is None else Path(root)
    output_path = _resolve_path(Path(output_dir), root_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Robot collision envelope (config-driven; bounds the 3D occupancy field).
    envelope, envelope_meta = load_robot_envelope(root=root_path)

    try:
        assets = load_canonical_assets(manifest, repo_root=root_path)
    except (FileNotFoundError, ValueError) as exc:
        summary = {
            "module": "Atlas3R Teacher",
            "status": "blocked_no_manifest",
            "error": str(exc),
            "tracks": [],
        }
        _write_json(output_path / "teacher_summary.json", summary)
        return summary

    per_track: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}
    for asset in assets:
        track_report = _run_track(
            asset,
            root_path,
            artifacts_dir=artifacts_dir,
            m1_dir=m1_dir,
            m2_dir=m2_dir,
            output_dir=output_path,
            envelope=envelope,
        )
        report_path = output_path / f"{asset.asset_id}_teacher_report.json"
        _write_json(report_path, track_report)
        report_paths[asset.asset_id] = str(report_path)
        per_track.append(track_report)

    summary = {
        "module": "Atlas3R Teacher",
        "status": "complete",
        "robot_envelope": envelope_meta,
        "track_reports": report_paths,
        "tracks": [
            {
                "asset_id": tr["asset_id"],
                "track_type": tr["track_type"],
                "final_category": tr["final_category"],
                "monocular_candidate_category": (
                    tr.get("reference_metric_subresults", {})
                    .get("monocular_candidate", {})
                    .get("final_category")
                ),
                "measured_baseline_category": (
                    tr.get("reference_metric_subresults", {})
                    .get("measured_baseline", {})
                    .get("final_category")
                ),
                "exact_blockers": tr["exact_blockers"],
            }
            for tr in per_track
        ],
    }
    _write_json(output_path / "teacher_summary.json", summary)
    _print_summary(summary)
    return summary


def _run_track(
    asset: VideoAsset,
    root: Path,
    *,
    artifacts_dir: str | Path,
    m1_dir: str | Path,
    m2_dir: str | Path,
    output_dir: Path,
    envelope: RobotEnvelopeConfig,
) -> dict[str, Any]:
    asset_id = asset.asset_id
    is_reference = asset.track_type is TrackType.REFERENCE_METRIC
    blockers: list[str] = []
    asset_out_dir = output_dir / asset_id
    report_path = output_dir / f"{asset_id}_teacher_report.json"

    report: dict[str, Any] = {
        "module": "Atlas3R Teacher",
        "asset_id": asset_id,
        "track_type": asset.track_type.value,
        "asset_availability": {},
        "geometry_source_status": {},
        "packet_creation_status": {},
        "refined_pose_depth_map_status": {},
        "static_dynamic_movable_status": {},
        "visibility_residual_status": {},
        "scale_evidence": {},
        "map_occupancy_status": {},
        "map_mesh_occupancy_artifacts": {},
        "visual_proof_artifacts": {},
        "validation_status": {},
        "reference_metric_subresults": {},
        "final_category": "rejected",
        "exact_blockers": [],
    }

    # (a) asset availability from M1
    availability = _stage(
        blockers, "asset_availability",
        lambda: _asset_availability(asset_id, root, m1_dir),
    )
    report["asset_availability"] = availability

    # (b) measured reference (M2) -- BASELINE / EVALUATION only.
    measured_packets: list[Any] = []
    measured_status = _stage(
        blockers, "measured_reference",
        lambda: _load_measured(asset_id, root, m2_dir) if is_reference else _not_applicable("not_reference_metric_track"),
    )
    if isinstance(measured_status, Mapping) and measured_status.get("status") == "loaded":
        measured_packets = list(measured_status.get("_packets", []))
    measured_report = {k: v for k, v in measured_status.items() if k != "_packets"} if isinstance(measured_status, Mapping) else measured_status

    # (c) monocular DA3 candidate (BOTH tracks).
    geometry_status = _stage(
        blockers, "geometry_source",
        lambda: _load_geometry(asset_id, root, artifacts_dir),
    )
    monocular_packets = list(geometry_status.get("_packets", [])) if isinstance(geometry_status, Mapping) else []
    geometry_soft_evidence = geometry_status.get("_scale_evidence", []) if isinstance(geometry_status, Mapping) else []
    geometry_report = {k: v for k, v in geometry_status.items() if not str(k).startswith("_")} if isinstance(geometry_status, Mapping) else geometry_status

    report["geometry_source_status"] = {
        "measured_reference": _provenance_labeled(measured_report, measured_packets, "measured_reference"),
        "monocular_artifact": _provenance_labeled(geometry_report, monocular_packets, "monocular_DA3"),
    }

    # (d) REFINE the MONOCULAR candidate ONLY. Measured depth/pose is NEVER fed
    # into monocular refinement (architecture: candidate must be proven
    # separately). phone_room is metric (learned prior) so scale is fixed; the
    # measured-reference track's monocular candidate has a free gauge unless its
    # own learned prior anchors it -- refine with fixed scale when a soft metric
    # prior is present, free otherwise.
    monocular_has_metric_prior = bool(geometry_soft_evidence)
    refined_monocular = monocular_packets
    refine_report: dict[str, Any] = {"status": "skipped_no_monocular_packets"}
    if monocular_packets:
        refine_result = _stage(
            blockers, "refine_monocular",
            lambda: _refine(monocular_packets, fix_global_scale=monocular_has_metric_prior),
        )
        if isinstance(refine_result, Mapping):
            refined_monocular = refine_result.get("_packets", monocular_packets)
            refine_report = {k: v for k, v in refine_result.items() if not str(k).startswith("_")}
    report["refined_pose_depth_map_status"] = _refine_summary(refine_report)

    # (e) STATIC / DYNAMIC / MOVABLE / UNKNOWN on the REFINED monocular candidate.
    static_dynamic_states: list[Any] = []
    sd_report: dict[str, Any] = {"status": "skipped_no_monocular_packets"}
    if refined_monocular:
        sd_result = _stage(
            blockers, "static_dynamic",
            lambda: _static_dynamic(refined_monocular, asset_id, root, artifacts_dir),
        )
        if isinstance(sd_result, Mapping):
            static_dynamic_states = sd_result.get("_states", [])
            sd_report = {k: v for k, v in sd_result.items() if not str(k).startswith("_")}
    report["static_dynamic_movable_status"] = _static_dynamic_summary(sd_report)

    # Visibility on the refined monocular candidate (the working candidate set).
    candidate_packets = refined_monocular
    visibility_report = _stage(
        blockers, "visibility_residual",
        lambda: _visibility(candidate_packets) if candidate_packets else {"status": "blocked_no_packets", "blockers": ("no_candidate_packets_for_visibility",)},
    )
    report["visibility_residual_status"] = _strip_private(visibility_report)

    # Independent epipolar pose audit on the candidate (REPORTAGE ONLY: a
    # different algorithm class audits the reconstruction's relative poses;
    # abstention is authority loss, never a pass; it does not gate acceptance
    # until detection-limit calibration assigns it measured authority).
    report["epipolar_audit_status"] = _stage(
        blockers, "epipolar_audit",
        lambda: _epipolar_audit_summary(candidate_packets, asset_id, root)
        if candidate_packets
        else {"status": "blocked_no_packets", "authority": "none"},
    )

    # ------------------------------------------------------------------
    # MEASURED BASELINE pipeline FIRST (reference_metric only): the measured
    # RGB-D/pose packets, evaluated for the metric category. This is the ONLY path
    # that may reach measured_metric, AND it yields the measured 3D occupancy
    # field used as ground truth for the candidate's band validation.
    # ------------------------------------------------------------------
    baseline_result = None
    measured_comparison_field = None
    if is_reference and measured_packets:
        baseline_scale_evidence = _collect_scale_evidence(measured_status, is_reference)
        baseline_result = _run_pipeline(
            asset_id=asset_id,
            track_type=asset.track_type.value,
            packets=measured_packets,
            scale_evidence=baseline_scale_evidence,
            static_dynamic_states=None,  # baseline is measured GT; no SD inference
            measured_reference=measured_packets,
            packet_source="measured_reference",
            out_dir=asset_out_dir / "measured_baseline",
            report_path=report_path,
            visibility_report=_strip_private(visibility_report),
            blockers=blockers,
            label="measured_baseline",
            envelope=envelope,
        )
        measured_comparison_field = baseline_result.get("_comparison_field")

    # ------------------------------------------------------------------
    # MONOCULAR CANDIDATE pipeline (scale -> map(per-voxel class) -> export ->
    # visual proof -> validation). Run for BOTH tracks. The measured 3D field (if
    # any) is passed for READ-ONLY band agreement -- never into its construction.
    # ------------------------------------------------------------------
    candidate_result = _run_pipeline(
        asset_id=asset_id,
        track_type=asset.track_type.value,
        packets=candidate_packets,
        scale_evidence=list(geometry_soft_evidence),
        static_dynamic_states=static_dynamic_states,
        measured_reference=None,  # NEVER feed measured into the candidate
        packet_source="monocular_artifact",
        out_dir=asset_out_dir,
        report_path=report_path,
        visibility_report=_strip_private(visibility_report),
        blockers=blockers,
        label="monocular_candidate",
        envelope=envelope,
        measured_comparison_field=measured_comparison_field,
        measured_packets_for_band=(measured_packets if (is_reference and measured_packets) else None),
    )

    # Honest comparison of the monocular candidate against the measured baseline.
    monocular_vs_measured = None
    if is_reference and measured_packets and candidate_packets:
        monocular_vs_measured = _compare_candidate_to_measured(
            candidate_packets, measured_packets
        )

    report["packet_creation_status"] = {
        "measured_packet_count": len(measured_packets),
        "monocular_packet_count": len(monocular_packets),
        "refined_monocular_packet_count": len(refined_monocular),
        "candidate_packet_source": "monocular_artifact",
        "monocular_vs_measured": monocular_vs_measured,
        "status": "ready" if candidate_packets or measured_packets else "no_packets_available",
    }
    if not candidate_packets and not measured_packets:
        blockers.append(f"no_packets_for_{asset_id}")

    # Surface the candidate pipeline fields at top level (back-compat keys) and
    # record both sub-results explicitly.
    report["scale_evidence"] = candidate_result["scale_evidence_block"]
    report["map_occupancy_status"] = candidate_result["map_status"]
    report["map_mesh_occupancy_artifacts"] = candidate_result["export_artifacts"]
    report["visual_proof_artifacts"] = candidate_result["visual_proof"]
    report["validation_status"] = candidate_result["validation_status"]

    report["reference_metric_subresults"] = {
        "monocular_candidate": _subresult_summary(candidate_result),
        "measured_baseline": _subresult_summary(baseline_result) if baseline_result else {
            "status": "not_applicable" if not is_reference else "no_measured_packets",
            "note": (
                "phone_room has no measured reference; strongest monocular path only"
                if not is_reference
                else "reference_metric track had no measured packets to baseline"
            ),
        },
        "comparison_monocular_vs_measured": monocular_vs_measured,
        "band3d_agreement_monocular_vs_measured": candidate_result.get("band3d_agreement"),
        "truth": (
            "measured_metric is reached ONLY via measured_baseline; the monocular "
            "candidate carries its own honest category + error-vs-measured + "
            "per-voxel band agreement vs the measured 3D field"
        ),
    }

    # ------------------------------------------------------------------
    # FINAL CATEGORY. reference_metric: measured baseline decides the headline
    # metric category (measured_metric when it passes); phone_room: the monocular
    # candidate's own honest category (metric_pseudo_label / non_metric / rejected).
    # ------------------------------------------------------------------
    if is_reference and baseline_result is not None:
        final_category = baseline_result["final_category"]
    else:
        final_category = candidate_result["final_category"]
    report["final_category"] = final_category

    report["geometry_source_status"]["per_result_provenance"] = {
        "measured_reference": "measured_reference" if measured_packets else "unavailable",
        "monocular_candidate": candidate_result["provenance_label"],
        "headline_final_category_source": (
            "measured_baseline" if (is_reference and baseline_result is not None) else "monocular_candidate"
        ),
    }

    report["exact_blockers"] = _dedupe(blockers)
    return report


def _run_pipeline(
    *,
    asset_id: str,
    track_type: str,
    packets: Sequence[Any],
    scale_evidence: Sequence[Any],
    static_dynamic_states: Sequence[Any] | None,
    measured_reference: Sequence[Any] | None,
    packet_source: str,
    out_dir: Path,
    report_path: Path,
    visibility_report: Mapping[str, Any],
    blockers: list[str],
    label: str,
    envelope: RobotEnvelopeConfig,
    measured_comparison_field: Any = None,
    measured_packets_for_band: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run scale -> map(per-voxel class) -> export -> visual proof -> validation.

    Returns a dict of the per-stage sub-reports plus ``final_category`` /
    ``provenance_label`` / ``band3d_agreement`` and the private band 3D field
    ``_voxel_3d``. Each stage is wrapped; a missing input is a blocked status with
    an exact blocker, never a fabricated result.
    """
    local_blockers: list[str] = []
    result: dict[str, Any] = {
        "label": label,
        "packet_source": packet_source,
        "packet_count": len(packets),
        "final_category": "rejected",
        "band3d_agreement": None,
        "_voxel_3d": None,
    }

    measured_evidence_present = any(getattr(e, "measured", False) for e in scale_evidence)
    scale_result = _stage(
        local_blockers, f"{label}_scale_posterior",
        lambda: _scale(packets, scale_evidence) if packets else {"status": "blocked_no_packets", "blockers": ("no_packets_for_scale",)},
    )
    scale_posterior = scale_result.get("_posterior") if isinstance(scale_result, Mapping) else None
    provenance_label = _provenance_from_packets(packets, scale_posterior)

    result["scale_evidence_block"] = {
        "scale_evidence_count": len(scale_evidence),
        "measured_evidence_present": measured_evidence_present,
        "scale_evidence_records": [_jsonable(e) for e in scale_evidence],
        "classification": _strip_private(scale_result),
        "provenance_label": provenance_label,
    }
    result["provenance_label"] = provenance_label

    if scale_posterior is None:
        result["map_status"] = {"status": "blocked_no_scale_posterior"}
        result["export_artifacts"] = {"status": "blocked_no_scale_posterior"}
        result["visual_proof"] = {"status": "blocked_no_scale_posterior"}
        result["validation_status"] = {"status": "blocked_no_scale_posterior"}
        result["final_category"] = "rejected"
        local_blockers.append(f"no_scale_posterior_for_{label}")
        blockers.extend(local_blockers)
        return result

    # MAP + per-voxel 3D occupancy (dynamic tagged, never fused into static). The
    # candidate occupancy-estimation policy (free-carve truncation + gravity support)
    # is applied ONLY to the monocular candidate -- never to the measured GT
    # baseline, which stays a raw yardstick.
    apply_fusion_policy = label == "monocular_candidate"
    map_result = _stage(
        local_blockers, f"{label}_map_occupancy",
        lambda: _map(packets, scale_posterior, static_dynamic_states, envelope, apply_fusion_policy),
    )
    voxel_map = map_result.get("_voxel_map") if isinstance(map_result, Mapping) else None
    occupancy_grid = map_result.get("_grid") if isinstance(map_result, Mapping) else None
    voxel_3d = map_result.get("_voxel_3d") if isinstance(map_result, Mapping) else None
    comparison_field = map_result.get("_comparison_field") if isinstance(map_result, Mapping) else None
    map_report = map_result.get("_map_report", {}) if isinstance(map_result, Mapping) else {}
    result["map_status"] = _strip_private(map_result)
    result["_voxel_3d"] = voxel_3d
    result["_comparison_field"] = comparison_field

    # Per-voxel band agreement of THIS field vs the measured 3D field (read-only;
    # the measured field never enters this field's construction). The measured
    # band is the eval region; this candidate's FULL field is sampled there. Only
    # the monocular candidate is compared; the measured baseline IS the reference.
    band3d_agreement = _resolve_band3d_agreement(
        label, comparison_field, measured_comparison_field, packets, measured_packets_for_band,
    )
    result["band3d_agreement"] = band3d_agreement

    # EXPORT artifacts (point cloud / mesh / trajectory / occupancy npz + 3D npz).
    export_result = _stage(
        local_blockers, f"{label}_export",
        lambda: _export(
            asset_id, packets, voxel_map, occupancy_grid, voxel_3d,
            static_dynamic_states, scale_posterior, out_dir, map_report,
        ),
    )
    result["export_artifacts"] = _strip_private(export_result) if isinstance(export_result, Mapping) else export_result

    # VISUAL PROOF (top-down + per-channel PNG + index.md).
    visual_result = _stage(
        local_blockers, f"{label}_visual_proof",
        lambda: _visual_proof(
            asset_id, track_type, packets, occupancy_grid, map_report,
            scale_posterior, packet_source, provenance_label, out_dir,
            report_path, export_result if isinstance(export_result, Mapping) else {},
            voxel_3d,
        ),
    )
    result["visual_proof"] = visual_result

    # VALIDATION + final honest category.
    validation_result = _stage(
        local_blockers, f"{label}_validation",
        lambda: _validate(
            asset_id, packets, scale_posterior,
            visibility_report,
            map_report if isinstance(map_report, Mapping) else {},
            measured_reference,
            static_dynamic_states,
            band3d_agreement,
        ),
    )
    final_category = validation_result.get("final_category", "rejected") if isinstance(validation_result, Mapping) else "rejected"
    result["validation_status"] = _strip_private(validation_result)
    result["final_category"] = final_category

    if isinstance(validation_result, Mapping):
        for b in validation_result.get("blockers", ()):  # type: ignore[union-attr]
            local_blockers.append(f"{label}:{b}")

    blockers.extend(local_blockers)
    return result


# ---------------------------------------------------------------------------
# stage wrappers (each returns a dict; private keys prefixed with "_")
# ---------------------------------------------------------------------------


def _stage(blockers: list[str], name: str, fn) -> Any:
    try:
        return fn()
    except Exception as exc:  # robust: a stage failure must not crash the run
        blockers.append(f"{name}_stage_exception:{type(exc).__name__}")
        return {
            "status": f"{name}_stage_exception",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
            "blockers": (f"{name}_stage_exception",),
        }


def _asset_availability(asset_id: str, root: Path, m1_dir: str | Path) -> dict[str, Any]:
    m1_root = _resolve_path(Path(m1_dir), root)
    registry_path = m1_root / "asset_registry_report.json"
    inspection_path = m1_root / f"{asset_id}_inspection.json"
    if not inspection_path.exists():
        return {
            "status": "missing_m1_inspection",
            "m1_report_dir": str(m1_root),
            "next_command": "python -m atlas3r.m1",
            "blockers": (f"missing_m1_inspection:{asset_id}",),
        }
    try:
        data = json.loads(inspection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "corrupt_m1_inspection", "error": str(exc), "blockers": ("corrupt_m1_inspection",)}
    frame_count = int(data.get("frame_count", 0) or 0)
    hard = tuple(data.get("hard_rejection_reasons", ()))
    return {
        "status": "available" if frame_count > 0 and not hard else "m1_unusable",
        "frame_count": frame_count,
        "hard_rejection_reasons": hard,
        "registry_report": str(registry_path) if registry_path.exists() else None,
        "blockers": () if frame_count > 0 and not hard else ("m1_asset_unusable",),
    }


def _load_measured(asset_id: str, root: Path, m2_dir: str | Path) -> dict[str, Any]:
    packets, report = load_measured_packets_from_m2(asset_id, root, m2_dir=m2_dir)
    return {**report, "_packets": packets}


def _load_geometry(asset_id: str, root: Path, artifacts_dir: str | Path) -> dict[str, Any]:
    packets, report = load_geometry_artifacts(asset_id, root, artifacts_dir=artifacts_dir)
    return {**report, "_packets": packets}


def _visibility(packets: Sequence[Any]) -> dict[str, Any]:
    _graph, report = build_visibility_graph(packets)
    return report


def _epipolar_audit_summary(
    packets: Sequence[Any], asset_id: str, root: Path
) -> dict[str, Any]:
    """Compact epipolar-audit block for the teacher report (per-pair rows are
    dropped; the full audit lives in the standalone CLI output)."""
    from .epipolar_audit import audit_scene

    report = audit_scene(packets, asset_id, root)
    return {k: v for k, v in report.items() if k != "pairs"}


def _scale(packets: Sequence[Any], scale_evidence: Sequence[Any]) -> dict[str, Any]:
    posterior, report = estimate_scale_posterior(packets, scale_evidence)
    return {**report, "_posterior": posterior}


def _map(
    packets: Sequence[Any],
    scale_posterior: Any,
    static_dynamic_states: Sequence[Any] | None = None,
    envelope: RobotEnvelopeConfig | None = None,
    apply_fusion_policy: bool = False,
) -> dict[str, Any]:
    voxel_map, grid, voxel_3d, comparison_field, report = fuse_static_map(
        packets, scale_posterior,
        static_dynamic_states=static_dynamic_states,
        envelope=envelope,
        apply_fusion_policy=apply_fusion_policy,
    )
    return {
        **report,
        "_map_report": report,
        "_voxel_map": voxel_map,
        "_grid": grid,
        "_voxel_3d": voxel_3d,
        "_comparison_field": comparison_field,
        "voxel_map_produced": voxel_map is not None,
        "occupancy_grid_produced": grid is not None,
        "voxel_occupancy_3d_produced": voxel_3d is not None,
    }


def _refine(packets: Sequence[Any], *, fix_global_scale: bool) -> dict[str, Any]:
    refined, report = refine_scene(packets, fix_global_scale=fix_global_scale)
    return {**report, "_packets": refined}


def _static_dynamic(
    packets: Sequence[Any], asset_id: str, root: Path, artifacts_dir: str | Path
) -> dict[str, Any]:
    masks_dir = _resolve_path(Path(artifacts_dir), root) / asset_id / "masks"
    states, report = infer_static_dynamic(
        packets,
        masks_dir=masks_dir,
        max_samples_per_frame=STATIC_DYNAMIC_SAMPLES_PER_FRAME,
    )
    return {**report, "_states": states}


def _export(
    asset_id: str,
    packets: Sequence[Any],
    voxel_map: Any,
    occupancy_grid: Any,
    voxel_occupancy_3d: Any,
    static_dynamic_states: Sequence[Any] | None,
    scale_posterior: Any,
    out_dir: Path,
    map_report: Mapping[str, Any],
) -> dict[str, Any]:
    return export_teacher_artifacts(
        asset_id, packets, voxel_map, occupancy_grid,
        static_dynamic_states, scale_posterior, out_dir,
        map_report=map_report,
        voxel_occupancy_3d=voxel_occupancy_3d,
        floor_align_rotation=map_report.get("floor_align_rotation") if isinstance(map_report, Mapping) else None,
    )


def _visual_proof(
    asset_id: str,
    track_type: str,
    packets: Sequence[Any],
    occupancy_grid: Any,
    map_report: Mapping[str, Any],
    scale_posterior: Any,
    packet_source: str,
    provenance_label: str,
    out_dir: Path,
    report_path: Path,
    export_result: Mapping[str, Any],
    voxel_occupancy_3d: Any = None,
) -> dict[str, Any]:
    import numpy as np  # lazy; trajectory stacking only

    trajectory = None
    if packets:
        centers = []
        for packet in sorted(packets, key=lambda p: int(p.frame_id)):
            T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
            centers.append(T[:3, 3])
        trajectory = np.asarray(centers, dtype=np.float64)
        # The occupancy grids are in the FLOOR-ALIGNED frame; rotate the camera
        # centres into the same frame so the top-down overlay matches the map.
        r_align = map_report.get("floor_align_rotation") if isinstance(map_report, Mapping) else None
        if r_align is not None and trajectory.shape[0]:
            trajectory = trajectory @ np.asarray(r_align, dtype=np.float64).T

    grid_sub = map_report.get("occupancy_grid", {}) if isinstance(map_report, Mapping) else {}
    plane_axes = grid_sub.get("plane_axes") if isinstance(grid_sub, Mapping) else None
    floor_axis = grid_sub.get("floor_axis") if isinstance(grid_sub, Mapping) else None
    status_label = scale_posterior.metric_acceptance_status.value if scale_posterior is not None else "unknown"

    provenance = {
        "provenance_label": provenance_label,
        "packet_source": packet_source,
        "track_type": track_type,
        "plane_axes": plane_axes,
        "floor_axis": floor_axis,
        "exports": export_result.get("artifacts") if isinstance(export_result, Mapping) else None,
        "teacher_report": str(report_path),
        "map_status": map_report.get("status") if isinstance(map_report, Mapping) else None,
        "missing_command": "python -m atlas3r.teacher",
        "blockers": map_report.get("blockers") if isinstance(map_report, Mapping) else None,
    }
    return write_visual_proof(
        asset_id, occupancy_grid, trajectory, out_dir, status_label, provenance,
        voxel_occupancy_3d=voxel_occupancy_3d,
    )


def _validate(
    asset_id: str,
    packets: Sequence[Any],
    scale_posterior: Any,
    visibility_report: Mapping[str, Any],
    map_report: Mapping[str, Any],
    measured_reference: Sequence[Any] | None,
    static_dynamic_states: Sequence[Any] | None = None,
    band3d_agreement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _report, final_category, validation_dict = validate_and_accept(
        asset_id, packets, scale_posterior, visibility_report, map_report,
        measured_reference, static_dynamic_states,
        band3d_agreement=band3d_agreement,
    )
    return {**validation_dict, "final_category": final_category}


def _collect_scale_evidence(measured_status: Any, is_reference: bool) -> list[Any]:
    """Pull ScaleEvidence from the M2 report only for the measured track.

    The monocular artifacts explicitly declare ``metric_evidence: false`` and
    supply no measured anchor, so they contribute no scale evidence here.
    """
    if not is_reference or not isinstance(measured_status, Mapping):
        return []
    # The measured loader rebuilds packets; scale evidence lives in the M2
    # JSON report which we re-read to construct ScaleEvidence objects.
    records = measured_status.get("scale_evidence_objects")
    if isinstance(records, list):
        return records
    return _read_m2_scale_evidence(measured_status)


def _read_m2_scale_evidence(measured_status: Mapping[str, Any]) -> list[Any]:
    from .contracts import ScaleEvidence, ScaleEvidenceType

    report_path = measured_status.get("report_path")
    if not report_path or not Path(report_path).exists():
        return []
    try:
        data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[Any] = []
    for record in data.get("scale_evidence", ()):
        if not isinstance(record, Mapping):
            continue
        try:
            provenance = record.get("provenance") or {"source": "m2_reference"}
            out.append(
                ScaleEvidence(
                    evidence_id=str(record["evidence_id"]),
                    evidence_type=ScaleEvidenceType(str(record["evidence_type"])),
                    measured=bool(record.get("measured", False)),
                    source=str(record.get("source", "m2_reference")),
                    frame_ids=tuple(int(f) for f in record.get("frame_ids", ())),
                    confidence=float(record.get("confidence", 1.0)),
                    provenance=dict(provenance) if provenance else {"source": "m2_reference"},
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# provenance labels + report summaries + candidate-vs-measured comparison
# ---------------------------------------------------------------------------


def _provenance_from_packets(packets: Sequence[Any], scale_posterior: Any) -> str:
    """Architecture per-result provenance label from packet source + evidence.

    measured_reference | monocular_DA3 | learned_metric_prior | manual_anchor |
    unavailable. A learned metric-depth prior is detected on the scale posterior's
    soft evidence; never a fabricated upgrade.
    """
    if not packets:
        return "unavailable"
    has_learned_prior = False
    if scale_posterior is not None:
        for ev in getattr(scale_posterior, "scale_sources", ()) or ():
            etype = getattr(getattr(ev, "evidence_type", None), "value", None)
            if etype == "learned_metric_depth_prior":
                has_learned_prior = True
    first = packets[0]
    source = str(getattr(first, "source", ""))
    if source.startswith("measured_reference"):
        return "measured_reference"
    if source.startswith("external_artifact"):
        return "learned_metric_prior" if has_learned_prior else "monocular_DA3"
    return "unavailable"


def _provenance_labeled(
    report: Any, packets: Sequence[Any], default_label: str
) -> Any:
    """Attach a per-result provenance label to a geometry-source sub-report."""
    if not isinstance(report, Mapping):
        return report
    label = "unavailable"
    if packets:
        first = packets[0]
        source = str(getattr(first, "source", ""))
        if source.startswith("measured_reference"):
            label = "measured_reference"
        elif source.startswith("external_artifact"):
            label = "learned_metric_prior" if report.get("learned_metric_depth_prior") else "monocular_DA3"
    elif isinstance(report, Mapping) and report.get("status") in {"loaded"}:
        label = default_label
    return {**report, "per_result_provenance": label}


def _refine_summary(refine_report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, honest summary of the refinement (cost before/after = proof)."""
    if not isinstance(refine_report, Mapping):
        return {"status": "unknown"}
    keys = (
        "status", "method", "frames_used", "edges_used", "samples_used", "nfev",
        "cost_before", "cost_after", "cost_reduction_fraction", "global_scale",
        "fix_global_scale", "residual_summary_before", "residual_summary_after",
        "provenance", "blockers",
    )
    out = {k: refine_report[k] for k in keys if k in refine_report}
    out["refinement_is_real"] = bool(
        refine_report.get("status") == "refined"
        and float(refine_report.get("cost_after", 1.0)) < float(refine_report.get("cost_before", 0.0))
    )
    return out


def _static_dynamic_summary(sd_report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(sd_report, Mapping):
        return {"status": "unknown"}
    keys = (
        "status", "packet_count", "evidence_families_used", "mask_evidence",
        "provenance", "totals", "dynamic_excluded_from_static_fusion",
        "movable_distinct_from_dynamic", "blockers",
    )
    return {k: sd_report[k] for k in keys if k in sd_report}


def _subresult_summary(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {"status": "absent"}
    map_status = result.get("map_status", {})
    val = result.get("validation_status", {})
    export = result.get("export_artifacts", {})
    visual = result.get("visual_proof", {})
    return {
        "label": result.get("label"),
        "packet_source": result.get("packet_source"),
        "packet_count": result.get("packet_count"),
        "provenance_label": result.get("provenance_label"),
        "final_category": result.get("final_category"),
        "scale_status": (result.get("scale_evidence_block", {}) or {}).get("classification", {}).get("status"),
        "map_status": map_status.get("status") if isinstance(map_status, Mapping) else map_status,
        "accepted_for_metric_training": val.get("accepted_for_metric_training") if isinstance(val, Mapping) else None,
        "held_out_render_error": val.get("held_out_render_error") if isinstance(val, Mapping) else None,
        "free_space_contradiction_rate": val.get("free_space_contradiction_rate") if isinstance(val, Mapping) else None,
        "export_status": export.get("status") if isinstance(export, Mapping) else export,
        "visual_proof_status": visual.get("status") if isinstance(visual, Mapping) else visual,
    }


def _compare_candidate_to_measured(
    candidate_packets: Sequence[Any],
    measured_packets: Sequence[Any],
) -> dict[str, Any]:
    """Honest comparison of the monocular candidate against the measured baseline.

    Camera centres (``T_world_camera[:3,3]``) of frames present in BOTH sets are
    aligned with a Sim(3) Umeyama fit (scale+rotation+translation; the monocular
    gauge is free so scale is part of the alignment), and the residual trajectory
    error is reported. When too few frames overlap for a Sim(3) fit, the
    nearest-frame translation discrepancy is reported instead with an explicit
    note -- never a fabricated metric.
    """
    import numpy as np  # lazy

    cand_by_frame = {int(p.frame_id): p for p in candidate_packets}
    meas_by_frame = {int(p.frame_id): p for p in measured_packets}
    common = sorted(set(cand_by_frame) & set(meas_by_frame))

    def _center(p):
        T = np.asarray(p.T_world_camera, dtype=np.float64).reshape((4, 4))
        return T[:3, 3]

    if len(common) >= 3:
        src = np.asarray([_center(cand_by_frame[f]) for f in common], dtype=np.float64)
        dst = np.asarray([_center(meas_by_frame[f]) for f in common], dtype=np.float64)
        s, R, t, aligned, rmse = _umeyama_align(src, dst, np)
        per_frame = np.linalg.norm(aligned - dst, axis=1)
        return {
            "method": "umeyama_sim3_camera_center_alignment",
            "common_frame_ids": common,
            "common_frame_count": len(common),
            "estimated_scale_monocular_to_measured": float(s),
            "trajectory_rmse_m_after_alignment": float(rmse),
            "trajectory_median_error_m_after_alignment": float(np.median(per_frame)),
            "trajectory_max_error_m_after_alignment": float(np.max(per_frame)),
            "note": (
                "monocular gauge is free; Sim(3) scale folded into alignment. "
                "Error is the residual camera-center discrepancy vs measured GT."
            ),
        }

    # Too few overlapping frames for a Sim(3) fit: nearest-frame translation gap.
    if cand_by_frame and meas_by_frame:
        meas_frames = sorted(meas_by_frame)
        gaps = []
        for f, p in cand_by_frame.items():
            nearest = min(meas_frames, key=lambda mf: abs(mf - f))
            gaps.append(float(np.linalg.norm(_center(p) - _center(meas_by_frame[nearest]))))
        return {
            "method": "nearest_frame_translation_discrepancy",
            "common_frame_count": len(common),
            "candidate_frame_count": len(cand_by_frame),
            "measured_frame_count": len(meas_by_frame),
            "nearest_frame_translation_median_m": float(np.median(gaps)),
            "note": (
                "fewer than 3 shared frame_ids for a Sim(3) Umeyama fit; reporting "
                "nearest-frame camera-center gap. This is NOT scale-aligned -- the "
                "monocular gauge is free -- so treat as a coarse upper bound only."
            ),
        }
    return {
        "method": "no_comparison_possible",
        "note": "no overlapping or comparable frames between candidate and measured",
    }


def _resolve_band3d_agreement(
    label: str,
    candidate_field: Mapping[str, Any] | None,
    measured_field: Mapping[str, Any] | None,
    packets: Sequence[Any],
    measured_packets_for_band: Sequence[Any] | None,
) -> dict[str, Any] | None:
    """Decide and compute the per-voxel band agreement vs the measured 3D field.

    Honest, never fabricated: the measured baseline IS the reference (no self
    comparison); a track with no measured 3D reference (phone_room) returns an
    explicit ``missing_measured_3d_reference`` status; a comparison error is
    surfaced rather than swallowed.
    """
    if label == "measured_baseline":
        return None
    if candidate_field is None:
        return {
            "status": "no_candidate_3d_field",
            "note": "no VoxelOccupancyGrid3D was built for this result",
        }
    if measured_field is None or not measured_packets_for_band:
        return {
            "status": "missing_measured_3d_reference",
            "note": (
                "no measured 3D reference for this track; per-voxel band agreement "
                "is not computable (never fabricated)"
            ),
        }
    try:
        return _band3d_agreement(
            candidate_field, measured_field, packets, measured_packets_for_band
        )
    except Exception as exc:  # robust: comparison must not crash the run
        return {
            "status": "band3d_agreement_error",
            "error": f"{type(exc).__name__}:{exc}",
        }


def _rotation_align(a, b, np):
    """Shortest-arc rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    sin = float(np.linalg.norm(v))
    if sin < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (vx @ vx)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _band3d_agreement(
    candidate_field: Mapping[str, Any],
    measured_field: Mapping[str, Any],
    candidate_packets: Sequence[Any],
    measured_packets: Sequence[Any],
) -> dict[str, Any]:
    """Per-voxel agreement of the candidate vs the measured 3D field in the band.

    The eval region is the MEASURED collision band (the robot-relevant ground
    truth). The candidate has a free gauge, so the measured band voxels are mapped
    into the candidate frame and the candidate's FULL-volume class field is sampled
    there by nearest voxel. The alignment is content-independent (no occupancy
    leak): a Sim(3) from SHARED CAMERA CENTRES fixes scale + yaw + translation, and
    its rotation is then REFINED to align the two reconstructions' FLOOR PLANES
    (their RANSAC normals -- a structural prior, not occupancy), because the
    camera-only Sim(3) leaves a residual floor tilt between two independently
    floor-aligned reconstructions. Returns concrete MEASURED numbers; an overlap
    too small for a Sim(3) fit yields an explicit insufficient-overlap status.
    """
    import numpy as np  # lazy

    def _center(p: Any) -> Any:
        T = np.asarray(p.T_world_camera, dtype=np.float64).reshape((4, 4))
        return T[:3, 3]

    # Each field was built in its OWN per-reconstruction floor-aligned frame
    # (mapping up-aligns candidate and measured independently). Bring the camera
    # centres into the SAME aligned frame as the field they will be matched against,
    # so the Sim(3) and the sampled occupancy share one frame. Identity R_up (the
    # honest fallback) leaves the centres untouched.
    R_up_c = np.asarray(candidate_field.get("R_up", np.eye(3)), dtype=np.float64)
    R_up_m = np.asarray(measured_field.get("R_up", np.eye(3)), dtype=np.float64)
    cand_by = {int(p.frame_id): R_up_c @ _center(p) for p in candidate_packets}
    meas_by = {int(p.frame_id): R_up_m @ _center(p) for p in measured_packets}
    common = sorted(set(cand_by) & set(meas_by))
    if len(common) < 3:
        return {
            "status": "insufficient_overlap_for_sim3_band_comparison",
            "common_frame_count": len(common),
            "note": (
                "fewer than 3 shared frame_ids for a Sim(3) fit; no scale-aligned "
                "band agreement (never fabricated)"
            ),
        }

    # Sim(3) from shared camera centres (scale + yaw + translation), then refine
    # the rotation so the two reconstructions' FLOOR PLANES coincide (their RANSAC
    # normals -- a structural prior, not occupancy content). The camera-only Sim(3)
    # leaves a residual floor tilt between two independently floor-aligned
    # reconstructions; this refinement removes it. Content-independent: occupancy
    # never informs the transform, so the measured field never leaks in.
    src = np.asarray([cand_by[f] for f in common], dtype=np.float64)
    dst = np.asarray([meas_by[f] for f in common], dtype=np.float64)
    s, r_sim, _t_sim, _aligned, rmse_cam_sim = _umeyama_align(src, dst, np)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    n_c = np.asarray(candidate_field.get("floor_normal", (0.0, 0.0, 1.0)), dtype=np.float64)
    n_m = np.asarray(measured_field.get("floor_normal", (0.0, 0.0, 1.0)), dtype=np.float64)
    nc_mapped = r_sim @ (n_c / (np.linalg.norm(n_c) + 1e-12))
    n_m_u = n_m / (np.linalg.norm(n_m) + 1e-12)
    if float(np.dot(nc_mapped, n_m_u)) < 0:
        n_m_u = -n_m_u
    tilt_before_deg = float(np.degrees(np.arccos(np.clip(np.dot(nc_mapped, n_m_u), -1.0, 1.0))))
    r_corr = _rotation_align(nc_mapped, n_m_u, np)
    R = r_corr @ r_sim
    t = mu_dst - s * (R @ mu_src)
    aligned = (s * (R @ src.T)).T + t[None, :]
    rmse_cam = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))

    FREE, OCC, MOV, DYN = 0, 1, 2, 3  # noqa: N806 (class label order)

    # Measured band eval region (the robot-relevant ground truth slab).
    m_cls = np.asarray(measured_field["class"])
    m_touched = np.asarray(measured_field["touched"])
    m_org = np.asarray(measured_field["origin"], dtype=np.float64)
    m_v = float(measured_field["voxel"])
    m_dims = measured_field["dims"]
    m_fa = int(measured_field["floor_axis"])
    bmin = float(measured_field["band_min_m"])
    bmax = float(measured_field["band_max_m"])
    gx = m_org[0] + (np.arange(m_dims[0]) + 0.5) * m_v
    gy = m_org[1] + (np.arange(m_dims[1]) + 0.5) * m_v
    gz = m_org[2] + (np.arange(m_dims[2]) + 0.5) * m_v
    grid_x, grid_y, grid_z = np.meshgrid(gx, gy, gz, indexing="ij")
    centers = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    in_band = (centers[:, m_fa] >= bmin) & (centers[:, m_fa] < bmax)
    sel = in_band & m_touched.ravel()
    pts_meas = centers[sel]
    meas_sel = m_cls.ravel()[sel]
    meas_band_touched = int(pts_meas.shape[0])
    if meas_band_touched == 0:
        return {
            "status": "no_measured_band_voxels",
            "estimated_scale_monocular_to_measured": float(s),
            "camera_center_rmse_m": rmse_cam,
            "note": "the measured 3D band had no observed voxels to compare against",
        }

    # Inverse transform: measured -> candidate frame, sample candidate full field.
    # Two sparse sampled-ray fields rarely hit the exact same voxel, so each
    # measured band voxel matches the nearest OBSERVED candidate voxel within a
    # robot-scale tolerance (exact-voxel coverage is reported too).
    p_cand = ((1.0 / s) * (R.T @ (pts_meas - t[None, :]).T)).T
    cf_cls = np.asarray(candidate_field["class"])
    cf_touched = np.asarray(candidate_field["touched"])
    cf_org = np.asarray(candidate_field["origin"], dtype=np.float64)
    cf_v = float(candidate_field["voxel"])
    cd0, cd1, cd2 = (int(candidate_field["dims"][0]), int(candidate_field["dims"][1]), int(candidate_field["dims"][2]))
    ci = np.floor((p_cand - cf_org[None, :]) / cf_v).astype(np.int64)
    in_b = (
        (ci[:, 0] >= 0) & (ci[:, 0] < cd0)
        & (ci[:, 1] >= 0) & (ci[:, 1] < cd1)
        & (ci[:, 2] >= 0) & (ci[:, 2] < cd2)
    )
    in_bounds_count = int(np.count_nonzero(in_b))
    rad = max(1, int(round(BAND_MATCH_TOLERANCE_M / cf_v)))
    n = pts_meas.shape[0]
    cand_at = np.full(n, -1, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    exact = 0
    for k in range(n):
        if not bool(in_b[k]):
            continue
        i, j, kz = int(ci[k, 0]), int(ci[k, 1]), int(ci[k, 2])
        i0, i1 = max(i - rad, 0), min(i + rad + 1, cd0)
        j0, j1 = max(j - rad, 0), min(j + rad + 1, cd1)
        l0, l1 = max(kz - rad, 0), min(kz + rad + 1, cd2)
        sub_t = cf_touched[i0:i1, j0:j1, l0:l1]
        if not sub_t.any():
            continue
        # Majority class among OBSERVED candidate voxels in the tolerance box.
        # Majority (not nearest-single) is robust to a thin floor surface sheet
        # bridging to free-above-floor samples in two sparse fields.
        sub_c = cf_cls[i0:i1, j0:j1, l0:l1][sub_t].astype(np.int64)
        counts = np.bincount(sub_c, minlength=5)
        cand_at[k] = int(np.argmax(counts))
        matched[k] = True
        if bool(cf_touched[i, j, kz]):
            exact += 1
    co = matched
    n_co = int(np.count_nonzero(co))
    if n_co == 0:
        return {
            "status": "no_co_observed_band_voxels",
            "estimated_scale_monocular_to_measured": float(s),
            "camera_center_rmse_m": rmse_cam,
            "floor_tilt_before_refine_deg": tilt_before_deg,
            "measured_band_touched_voxels": meas_band_touched,
            "measured_band_voxels_in_candidate_bounds": in_bounds_count,
            "match_tolerance_m": float(BAND_MATCH_TOLERANCE_M),
            "note": "no measured band voxel mapped onto an observed candidate voxel within tolerance",
        }

    meas_co = meas_sel[co]
    cand_co = cand_at[co]

    labels = ["free", "occupied_static", "movable_static", "dynamic", "unknown"]
    confusion: dict[str, int] = {}
    for mj, mn in enumerate(labels):
        for cj, cn in enumerate(labels):
            cnt = int(np.count_nonzero((meas_co == mj) & (cand_co == cj)))
            if cnt:
                confusion[f"meas_{mn}__cand_{cn}"] = cnt

    agreement = float(np.mean(meas_co == cand_co))
    meas_occ = meas_co == OCC
    cand_occ = cand_co == OCC
    inter = int(np.count_nonzero(meas_occ & cand_occ))
    union = int(np.count_nonzero(meas_occ | cand_occ))
    occ_iou = float(inter / union) if union else 0.0

    # Robot-critical: candidate calls free where the measured GT sees an obstacle.
    meas_solid = (meas_co == OCC) | (meas_co == MOV)
    n_meas_solid = int(np.count_nonzero(meas_solid))
    free_contra = int(np.count_nonzero(meas_solid & (cand_co == FREE)))
    free_contra_rate = float(free_contra / n_meas_solid) if n_meas_solid else 0.0

    # Dynamic leakage: candidate static-occupied where measured is dynamic, or
    # candidate dynamic where the measured GT is a solid static obstacle.
    dyn_leak = int(np.count_nonzero((meas_co == DYN) & (cand_co == OCC))) + int(
        np.count_nonzero(meas_solid & (cand_co == DYN))
    )
    dyn_leak_rate = float(dyn_leak / n_co) if n_co else 0.0
    coverage = float(n_co / meas_band_touched) if meas_band_touched else 0.0

    return {
        "status": "computed",
        "method": "camera_sim3_plus_floor_normal_refine_inverse_sample_candidate_full_at_measured_band",
        "common_frame_count": len(common),
        "estimated_scale_monocular_to_measured": float(s),
        "camera_center_rmse_m": rmse_cam,
        "camera_center_rmse_m_sim3_only": float(rmse_cam_sim),
        "floor_tilt_before_refine_deg": tilt_before_deg,
        "co_observed_band_voxels": n_co,
        "exact_voxel_co_observed": int(exact),
        "measured_band_voxels_in_candidate_bounds": in_bounds_count,
        "measured_band_touched_voxels": meas_band_touched,
        "match_tolerance_m": float(BAND_MATCH_TOLERANCE_M),
        "coverage_of_measured_band": coverage,
        "per_class_agreement": agreement,
        "occupied_static_iou": occ_iou,
        "measured_solid_voxels_co_observed": n_meas_solid,
        "free_space_contradiction_rate": free_contra_rate,
        "free_space_contradiction_voxels": free_contra,
        "dynamic_leakage_rate": dyn_leak_rate,
        "dynamic_leakage_voxels": dyn_leak,
        "confusion_counts": confusion,
        "note": (
            "eval region = measured collision band; candidate FULL field sampled "
            "there via inverse of a camera-center Sim(3) whose rotation is refined "
            "to align the two floor planes. free_space_contradiction = candidate "
            "calls free where the measured GT sees an obstacle (robot-critical)."
        ),
    }


def _umeyama_align(src, dst, np):
    """Sim(3) Umeyama: return ``(scale, R, t, aligned_src, rmse)``."""
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = float((D * np.diag(S)).sum() / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - scale * (R @ mu_src)
    aligned = (scale * (R @ src.T)).T + t
    rmse = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return scale, R, t, aligned, rmse


def _not_applicable(reason: str) -> dict[str, Any]:
    return {"status": "not_applicable", "reason": reason, "_packets": [], "blockers": ()}


def _strip_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if not str(k).startswith("_")}
    return value


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Atlas3R teacher spine complete.")
    for track in summary.get("tracks", ()):  # type: ignore[union-attr]
        blockers = track.get("exact_blockers", [])
        blocker_str = "; ".join(blockers) if blockers else "none"
        print(
            f"  {track['asset_id']} [{track['track_type']}] -> "
            f"{track['final_category']} (blockers: {blocker_str})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--m1-dir", default=DEFAULT_M1_DIR)
    parser.add_argument("--m2-dir", default=DEFAULT_M2_DIR)
    args = parser.parse_args(argv)

    run_teacher(
        args.manifest,
        args.output_dir,
        artifacts_dir=args.artifacts_dir,
        m1_dir=args.m1_dir,
        m2_dir=args.m2_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
