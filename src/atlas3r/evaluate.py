"""``python -m atlas3r.evaluate`` -- repeatable evaluation harness.

This module is a READ-AND-AGGREGATE pass over the teacher's existing honest
outputs. It runs the teacher on both canonical tracks (or consumes the latest
existing run with ``--no-run``) and emits ONE versioned, diffable scorecard plus a
human-readable summary. It establishes the measured baseline so every future change
is provably better or worse.

Hard rules (mirroring AGENTS.md Truth Invariants):

- It invents no metric, runs no model of its own, fabricates no number, and adds no
  synthetic data. Every value is plucked verbatim from a teacher report field.
- Missing / non-computed states are surfaced LOUDLY and verbatim. When a track has no
  measured 3D reference (e.g. ``phone_room``) the measured comparison is reported as
  DID NOT RUN with the teacher's exact status string -- never blank, never a
  fabricated number, never silently treated as "passed".
- A field the scorecard expects but the report omits is recorded as ``not_reported``;
  no replacement is computed.

The canonical ``scorecard.json`` is deterministic: the same teacher reports + git
state produce a byte-identical file (the stamp is the git commit ISO timestamp plus a
content SHA-256). Wall-clock time appears only in the human ``.md`` summary header so
it never perturbs the diff-critical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCORECARD_VERSION = 1

# A field the scorecard expects but the report does not contain. Never replaced with
# a computed value (HARD CONSTRAINT).
NOT_REPORTED = "not_reported"

DEFAULT_TEACHER_DIR = Path("runs/teacher")
DEFAULT_EVAL_DIR = Path("runs/eval")
DEFAULT_CONFIG_PATH = Path("configs/robot_envelope.json")

SCORECARD_NAME = "scorecard.json"
PREV_NAME = "scorecard.prev.json"
DIFF_NAME = "scorecard_diff.json"
SUMMARY_NAME = "scorecard_summary.md"

# Canonical tracks, in stable order so the scorecard is deterministic. The gate spans
# a DIFFICULTY SPREAD of measured scenes so it reflects realistic motion, not just a
# gentle best case, ordered easy -> hardest:
#   reference_metric       TUM freiburg1_xyz  gentle hand-held jitter   -> PASSES (cam RMSE ~0.08m, occ_iou ~0.13)
#   reference_metric_desk  TUM freiburg1_desk harder desk-orbit         -> FAILS  (cam RMSE ~0.25-0.33m, occ_iou ~0.0)
#   reference_metric_room  TUM freiburg1_room full room-LOOP            -> FAILS  (depth-limited band miss; occ_iou ~0.0)
#   phone_room             unanchored phone target, no measured GT      -> metric_pseudo_label (no band score)
# The two failing measured scenes are kept ON PURPOSE so the gate reflects realistic
# motion. NOTE on room: its 13-keyframe scorecard (cam RMSE 0.82m, scale 0.28) looks like
# a loop-closure blow-up but is mostly keyframe UNDER-SAMPLING -- at ~48 keyframes it
# reconstructs near-metric (cam RMSE 0.24-0.46m), better-posed than desk; the residual
# fail is the same depth-limited band miss. See docs/band_obstacle_recall_evidence.md Phase 5.
CANONICAL_TRACKS = ("reference_metric", "reference_metric_room", "reference_metric_desk", "phone_room")

# band3d_agreement statuses for which numeric per-class metrics exist. Anything else
# (missing_measured_3d_reference, insufficient_overlap_for_sim3_band_comparison, ...)
# is surfaced as status-only, verbatim.
BAND3D_COMPUTED_STATUS = "computed"


# ---------------------------------------------------------------------------
# Small read-only helpers
# ---------------------------------------------------------------------------
def _pluck(obj: Any, *path: str, default: Any = NOT_REPORTED) -> Any:
    """Walk a nested mapping by key path. Return ``default`` if any key is absent or a
    non-mapping is hit. A key present with value ``None`` returns ``None`` verbatim
    (``None`` is meaningful, e.g. a null camera comparison = "did not run")."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _run_git(args: list[str], root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path, root: Path) -> str:
    """Repo-relative POSIX path so the scorecard stays machine-independent."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Running / locating the teacher run
# ---------------------------------------------------------------------------
def run_teacher_subprocess(root: Path, teacher_dir: Path) -> dict[str, Any]:
    """Invoke ``python -m atlas3r.teacher`` (do NOT re-run its logic in-process; just
    drive its CLI, then aggregate its outputs)."""
    cmd = [sys.executable, "-m", "atlas3r.teacher", "--output-dir", str(teacher_dir)]
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    return {"command": cmd, "returncode": proc.returncode}


def load_teacher_run(root: Path, teacher_dir: Path) -> dict[str, Any]:
    """Resolve the latest teacher run from ``teacher_summary.json`` and load both
    per-track reports. Returns an explicit blocker dict if anything is missing -- never
    fabricates a run."""
    summary_path = (root / teacher_dir / "teacher_summary.json")
    if not summary_path.is_file():
        return {
            "status": "missing_teacher_run",
            "blocker": f"no teacher_summary.json at {_rel(summary_path, root)}; "
            "run `python -m atlas3r.teacher` first (or drop --no-run)",
            "summary_path": _rel(summary_path, root),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    track_report_paths = summary.get("track_reports", {})
    reports: dict[str, dict[str, Any]] = {}
    report_bytes: dict[str, bytes] = {}
    report_rel: dict[str, str] = {}
    missing: list[str] = []
    for asset_id, raw_path in track_report_paths.items():
        rp = Path(raw_path)
        if not rp.is_absolute():
            rp = root / rp
        if not rp.is_file():
            missing.append(f"{asset_id}:{raw_path}")
            continue
        data = rp.read_bytes()
        report_bytes[asset_id] = data
        report_rel[asset_id] = _rel(rp, root)
        reports[asset_id] = json.loads(data.decode("utf-8"))
    return {
        "status": "loaded" if not missing else "partial",
        "summary": summary,
        "summary_path": _rel(summary_path, root),
        "reports": reports,
        "report_bytes": report_bytes,
        "report_rel": report_rel,
        "missing_reports": missing,
    }


# ---------------------------------------------------------------------------
# Per-track aggregation -- pure field extraction, no computation
# ---------------------------------------------------------------------------
def extract_scale_posterior(report: dict[str, Any]) -> dict[str, Any]:
    classification = _pluck(report, "scale_evidence", "classification", default={})
    provenance = _pluck(report, "map_mesh_occupancy_artifacts", "provenance", default={})
    return {
        "scale_mean": _pluck(provenance, "scale_mean"),
        "scale_std": _pluck(provenance, "scale_std"),
        "relative_scale_uncertainty": _pluck(classification, "relative_scale_uncertainty"),
        "metric_acceptance_status": _pluck(classification, "status"),
        "measured_evidence_present": _pluck(report, "scale_evidence", "measured_evidence_present"),
        "scale_evidence_count": _pluck(report, "scale_evidence", "scale_evidence_count"),
    }


def extract_camera_center_error(report: dict[str, Any]) -> dict[str, Any]:
    """Camera-center Sim(3) error vs the measured baseline.

    The teacher emits ``packet_creation_status.monocular_vs_measured`` as a bare ``null``
    when there is no measured reference (e.g. phone_room) -- it carries NO status/reason
    string of its own for this case. So we surface the teacher field VERBATIM (the null)
    and carry the teacher's actual narration about the absence
    (``reference_metric_subresults.measured_baseline.note``). ``status`` is the only
    harness-chosen label here and is explicitly tagged as harness-derived -- it is never
    dressed up as a teacher string, and no number is fabricated."""
    comp = _pluck(report, "packet_creation_status", "monocular_vs_measured", default=NOT_REPORTED)
    if comp is None:
        return {
            "status": "not_computed",
            "status_origin": "harness_derived_from_null_teacher_field",
            "teacher_field": "packet_creation_status.monocular_vs_measured",
            "teacher_field_value": None,
            "teacher_note": _pluck(
                report, "reference_metric_subresults", "measured_baseline", "note"
            ),
        }
    if comp == NOT_REPORTED or not isinstance(comp, dict):
        return {"status": NOT_REPORTED}
    return {
        "status": "computed",
        "method": _pluck(comp, "method"),
        "trajectory_rmse_m": _pluck(comp, "trajectory_rmse_m_after_alignment"),
        "trajectory_median_m": _pluck(comp, "trajectory_median_error_m_after_alignment"),
        "trajectory_max_m": _pluck(comp, "trajectory_max_error_m_after_alignment"),
        "estimated_scale_monocular_to_measured": _pluck(comp, "estimated_scale_monocular_to_measured"),
        "common_frame_count": _pluck(comp, "common_frame_count"),
        "note": _pluck(comp, "note"),
    }


def extract_band3d_agreement(report: dict[str, Any]) -> dict[str, Any]:
    """Per-voxel band agreement vs the measured 3D field. When the teacher could not
    compute it (no measured GT, or too-small overlap) only the verbatim status + note
    are reported -- NEVER a number. This is reportage; it never gates acceptance."""
    summary = _pluck(report, "validation_status", "band3d_agreement_summary", default={})
    full = _pluck(report, "validation_status", "band3d_agreement", default={})
    status = _pluck(summary, "status", default=_pluck(full, "status"))
    if status != BAND3D_COMPUTED_STATUS:
        return {
            "status": status,
            "note": _pluck(full, "note"),
            "measured_comparison_ran": False,
        }
    return {
        "status": status,
        "measured_comparison_ran": True,
        "per_class_agreement": _pluck(summary, "per_class_agreement"),
        "occupied_static_iou": _pluck(summary, "occupied_static_iou"),
        "occupied_recall_any_within_tolerance": _pluck(summary, "occupied_recall_any_within_tolerance"),
        "free_space_contradiction_rate": _pluck(summary, "free_space_contradiction_rate"),
        "dynamic_leakage_rate": _pluck(summary, "dynamic_leakage_rate"),
        "coverage_of_measured_band": _pluck(summary, "coverage_of_measured_band"),
        "estimated_scale_monocular_to_measured": _pluck(summary, "estimated_scale_monocular_to_measured"),
        "co_observed_band_voxels": _pluck(summary, "co_observed_band_voxels"),
        # Resolution-invariant metric-distance agreement (reportage; not in
        # the numeric diff law until a pre-registered promotion).
        "solid_distance_agreement": _pluck(full, "solid_distance_agreement"),
    }


def extract_map(report: dict[str, Any]) -> dict[str, Any]:
    mos = _pluck(report, "map_occupancy_status", default={})
    band = _pluck(mos, "voxel_occupancy_3d", default={})
    field_summary = _pluck(
        report, "map_mesh_occupancy_artifacts", "voxel_occupancy_3d_summary", default={}
    )
    return {
        "voxel_band_3d": {
            "free_fraction": _pluck(band, "free_fraction"),
            "occupied_static_fraction": _pluck(band, "occupied_static_fraction"),
            "movable_static_fraction": _pluck(band, "movable_static_fraction"),
            "dynamic_fraction": _pluck(band, "dynamic_fraction"),
            "unknown_fraction": _pluck(band, "unknown_fraction"),
        },
        "full_grid": {
            "free_fraction": _pluck(mos, "free_fraction"),
            "occupied_fraction": _pluck(mos, "occupied_fraction"),
            "unknown_fraction": _pluck(mos, "unknown_fraction"),
        },
        "free_space_contradiction_rate": _pluck(mos, "free_space_contradiction_rate"),
        # Resolution-invariant companion at the calibration scale (reportage).
        "free_space_contradiction_rate_at_reference_scale": _pluck(
            mos, "free_space_contradiction_rate_at_reference_scale"
        ),
        "free_space_contradiction_basis": _pluck(mos, "free_space_contradiction_basis"),
        "mean_map_confidence": _pluck(field_summary, "mean_map_confidence"),
    }


def extract_floor(report: dict[str, Any]) -> dict[str, Any]:
    floor = _pluck(report, "map_occupancy_status", "floor", default={})
    return {
        "method": _pluck(floor, "method"),
        "inlier_ratio": _pluck(floor, "inlier_ratio"),
        "floor_axis": _pluck(floor, "floor_axis"),
        "up_alignment_applied": _pluck(floor, "up_alignment_applied"),
        "floor_tilt_to_band_axis_deg_before": _pluck(floor, "floor_tilt_to_band_axis_deg_before"),
        "floor_tilt_to_band_axis_deg_after": _pluck(floor, "floor_tilt_to_band_axis_deg_after"),
        "blockers": _pluck(report, "map_occupancy_status", "blockers", default=[]),
    }


def extract_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    sub = _pluck(report, "reference_metric_subresults", default={})
    measured_baseline = _pluck(sub, "measured_baseline", default={})
    # final_category when the measured baseline ran; else its verbatim status
    # (e.g. "not_applicable" for phone_room -- never silently blanked).
    measured_baseline_category = _pluck(
        measured_baseline, "final_category", default=_pluck(measured_baseline, "status")
    )
    return {
        "final_category": _pluck(report, "final_category"),
        "accepted_for_metric_training": _pluck(report, "validation_status", "accepted_for_metric_training"),
        "rejection_reasons": _pluck(report, "validation_status", "rejection_reasons", default=[]),
        "exact_blockers": _pluck(report, "exact_blockers", default=[]),
        "measured_baseline_category": measured_baseline_category,
        "monocular_candidate_category": _pluck(sub, "monocular_candidate", "final_category"),
    }


def extract_gate_cascade(report: dict[str, Any]) -> dict[str, Any]:
    """GT-free acceptance cascade verdicts (Stage 0 evidence mass / Stage 1
    gravity). Absent on pre-cascade reports -> verbatim absence, never zeros."""
    cascade = _pluck(report, "validation_status", "gate_cascade")
    if not isinstance(cascade, dict):
        return {"status": "absent_pre_cascade_report"}
    s0 = cascade.get("stage0_evidence_mass", {}) or {}
    s1 = cascade.get("stage1_gravity_alignment", {}) or {}
    return {
        "applies_to_this_path": cascade.get("applies_to_this_path"),
        "stage0_passes": s0.get("passes"),
        "stage0_median_inbounds_ratio": s0.get("median_reprojection_inbounds_ratio"),
        "stage0_depth_residual_edge_fraction": s0.get("depth_residual_edge_fraction"),
        "stage0_mean_confidence_weight": s0.get("mean_confidence_weight"),
        "stage1_passes": s1.get("passes"),
        "stage1_up_alignment_applied": s1.get("up_alignment_applied"),
        "stage1_floor_inlier_ratio": s1.get("floor_inlier_ratio"),
        "threshold_authority": s0.get("threshold_authority"),
    }


def extract_prerefine_severity(report: dict[str, Any]) -> dict[str, Any]:
    """Pre-refine cross-frame log-depth residual p90: REPORTAGE severity signal.

    Measured calibration (runs/_diag/signal_calibration_table.json, 2026-06-09):
    perfect rank correlation with camera-center Sim(3) RMSE over 7 GT-labeled
    configs spanning 3 scenes x 2 backbones (Spearman 1.0; 1.0 after partialling
    out coverage). Candidate for gate promotion pending leave-one-scene-out
    validation -- until then it gates NOTHING."""
    rb = _pluck(report, "refined_pose_depth_map_status", "residual_summary_before")
    if not isinstance(rb, dict):
        return {"status": "absent"}
    return {
        "prerefine_p90_log_depth_residual": rb.get("p90_log_depth_residual"),
        "prerefine_median_log_depth_residual": rb.get("median_log_depth_residual"),
        "authority": "candidate_pending_loso_validation_gates_nothing",
    }


def extract_plane_ledger(report: dict[str, Any]) -> dict[str, Any]:
    """Plane-ledger rigid-world drift audit (reportage only -- never gates)."""
    ledger = _pluck(report, "plane_ledger_status")
    if not isinstance(ledger, dict):
        return {"status": "absent_pre_ledger_report"}
    keep = ("status", "authority", "n_tracks_qualifying", "n_tracks_total",
            "signals", "direction_authority")
    out = {k: ledger.get(k) for k in keep if k in ledger}
    da = out.get("direction_authority")
    if isinstance(da, dict):  # compact: drop axis vectors, keep conditioning
        out["direction_authority"] = {
            "normal_span_singular_values_relative": da.get("normal_span_singular_values_relative"),
        }
    return out


def extract_epipolar_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Independent epipolar pose audit (reportage only -- never gates)."""
    audit = _pluck(report, "epipolar_audit_status")
    if not isinstance(audit, dict):
        return {"status": "absent_pre_audit_report"}
    keep = (
        "status", "authority", "intrinsics_source", "n_valid_pairs",
        "rotation_deviation_deg", "translation_direction_deviation_deg",
        "auditor_cycle_residual_deg", "deviation_authority", "abstained_pairs",
    )
    return {k: audit.get(k) for k in keep if k in audit}


def extract_track(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": _pluck(report, "asset_id"),
        "track_type": _pluck(report, "track_type"),
        "acceptance": extract_acceptance(report),
        "gate_cascade": extract_gate_cascade(report),
        "prerefine_severity": extract_prerefine_severity(report),
        "plane_ledger": extract_plane_ledger(report),
        "epipolar_audit": extract_epipolar_audit(report),
        "scale_posterior": extract_scale_posterior(report),
        "camera_center_sim3_error": extract_camera_center_error(report),
        "band3d_agreement": extract_band3d_agreement(report),
        "map": extract_map(report),
        "floor": extract_floor(report),
    }


# ---------------------------------------------------------------------------
# Stamping / provenance
# ---------------------------------------------------------------------------
def build_robot_envelope(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    """RobotEnvelopeConfig actually used by the run. Read the canonical config plus the
    run's derived band, dropping any absolute machine path so the stamp stays
    deterministic and portable."""
    env: dict[str, Any] = {"config_path": _rel(root / DEFAULT_CONFIG_PATH, root)}
    cfg_path = root / DEFAULT_CONFIG_PATH
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("collision_height_m", "margin_m", "voxel_size_m"):
            env[key] = cfg.get(key, NOT_REPORTED)
    else:
        env["config_status"] = "missing_config"
    # The run summary carries the envelope the teacher ACTUALLY used (its
    # provenance dict from load_robot_envelope, including any env-override path and
    # the candidate occupancy-estimation policy). Surface those so the scorecard is
    # self-documenting: the policy materially shapes the candidate band, and an
    # ATLAS3R_ROBOT_ENVELOPE_CONFIG override would otherwise be invisible here.
    run_env = _pluck(run, "summary", "robot_envelope", default={})
    if isinstance(run_env, dict):
        env["band_height_m"] = run_env.get("band_height_m", NOT_REPORTED)
        env["present"] = run_env.get("present", NOT_REPORTED)
        env["source"] = run_env.get("source", NOT_REPORTED)
        # env_override is the raw ATLAS3R_ROBOT_ENVELOPE_CONFIG value (None for the
        # canonical run); the absolute resolved path is deliberately NOT stamped so
        # the canonical scorecard stays machine-portable and byte-deterministic.
        env["env_override"] = run_env.get("env_override")
        for key in (
            "free_carve_margin_m",
            "occupancy_support_height_m",
            "occupancy_support_min_count",
            "occupancy_support_overrides_free",
            "occupancy_close_voxels",
        ):
            env[key] = run_env.get(key, NOT_REPORTED)
    return env


def build_backbone_provenance(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Monocular geometry-backbone provenance already in the reports (backbone-agnostic:
    MapAnything, DA3, ...). Asserted identical across tracks; divergence is surfaced
    rather than hidden."""
    seen: dict[str, dict[str, Any]] = {}
    for asset_id, report in reports.items():
        mono = _pluck(report, "geometry_source_status", "monocular_artifact", default={})
        seen[asset_id] = {
            "backbone_name": _pluck(mono, "backbone_name"),
            "method": _pluck(mono, "method"),
            "metric_evidence": _pluck(mono, "metric_evidence"),
            "learned_metric_depth_prior": _pluck(mono, "learned_metric_depth_prior"),
            "provenance_label": _pluck(
                report, "map_mesh_occupancy_artifacts", "provenance", "provenance_label"
            ),
        }
    distinct = {json.dumps(v, sort_keys=True) for v in seen.values()}
    if len(distinct) <= 1 and seen:
        prov = dict(next(iter(seen.values())))
        prov["consistent_across_tracks"] = True
        return prov
    return {"consistent_across_tracks": False, "per_track": seen}


def build_asset_provenance(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for asset_id, report in reports.items():
        avail = _pluck(report, "asset_availability", default={})
        out[asset_id] = {
            "status": _pluck(avail, "status"),
            "frame_count": _pluck(avail, "frame_count"),
        }
    return out


def build_scorecard(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    reports = run["reports"]
    report_bytes = run["report_bytes"]
    report_rel = run["report_rel"]

    tracks: dict[str, Any] = {}
    track_status: dict[str, Any] = {}
    for asset_id in CANONICAL_TRACKS:
        if asset_id in reports:
            tracks[asset_id] = extract_track(reports[asset_id])
            track_status[asset_id] = "aggregated"
        else:
            # A canonical track with no report: surfaced loudly, never blanked.
            tracks[asset_id] = {"status": "missing_track_report"}
            track_status[asset_id] = "missing_track_report"

    source_reports = {
        asset_id: {"path": report_rel[asset_id], "sha256": _sha256_bytes(report_bytes[asset_id])}
        for asset_id in sorted(report_bytes)
    }

    git_commit = _run_git(["rev-parse", "HEAD"], root)
    git_commit_iso = _run_git(["show", "-s", "--format=%cI", "HEAD"], root)
    # NOTE: working-tree dirtiness is deliberately NOT in the canonical scorecard. It
    # reflects transient state (unrelated untracked/edited files) that is not a teacher
    # input, so embedding it would let an unrelated file break the byte-identity
    # guarantee. It is surfaced in the non-canonical .md header instead (see evaluate()).

    meta = {
        "scorecard_version": SCORECARD_VERSION,
        "module": "Atlas3R Evaluate",
        "git_commit": git_commit if git_commit else NOT_REPORTED,
        "git_commit_iso": git_commit_iso if git_commit_iso else NOT_REPORTED,
        "robot_envelope": build_robot_envelope(root, run),
        "backbone_provenance": build_backbone_provenance(reports),
        "asset_provenance": build_asset_provenance(reports),
        "source_run_summary": run.get("summary_path", NOT_REPORTED),
        "source_reports": source_reports,
        "track_status": track_status,
        "tracks_present": [a for a in CANONICAL_TRACKS if a in reports],
        "missing_reports": run.get("missing_reports", []),
    }

    scorecard = {"meta": meta, "tracks": tracks}
    # Bind the scorecard to its exact content. Computed over the canonical JSON with the
    # hash field itself excluded, so two runs over the same inputs hash identically.
    scorecard["meta"]["scorecard_sha256"] = _canonical_sha256(scorecard)
    return scorecard


def _canonical_sha256(scorecard: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(scorecard))
    clone.get("meta", {}).pop("scorecard_sha256", None)
    payload = json.dumps(clone, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(payload)


# ---------------------------------------------------------------------------
# Diff against the previous scorecard
# ---------------------------------------------------------------------------
# Numeric metric paths (dotted, into a track block) diffed as deltas.
NUMERIC_METRICS: tuple[str, ...] = (
    "scale_posterior.scale_mean",
    "scale_posterior.scale_std",
    "scale_posterior.relative_scale_uncertainty",
    "camera_center_sim3_error.trajectory_rmse_m",
    "camera_center_sim3_error.trajectory_median_m",
    "camera_center_sim3_error.trajectory_max_m",
    "camera_center_sim3_error.estimated_scale_monocular_to_measured",
    "band3d_agreement.per_class_agreement",
    "band3d_agreement.occupied_static_iou",
    "band3d_agreement.free_space_contradiction_rate",
    "band3d_agreement.dynamic_leakage_rate",
    "band3d_agreement.coverage_of_measured_band",
    "band3d_agreement.estimated_scale_monocular_to_measured",
    # Resolution-invariant distance instruments (metric-law amendment,
    # docs/band_obstacle_recall_evidence.md Phase 12: the adoption bar keys on
    # solid_f1_at_0.05m; the voted metrics above stay reported for continuity
    # but no longer gate adoption -- they diverge under grid refinement).
    "band3d_agreement.solid_distance_agreement.solid_f1_at_5cm",
    "band3d_agreement.solid_distance_agreement.solid_precision_at_5cm",
    "band3d_agreement.solid_distance_agreement.solid_recall_at_5cm",
    "band3d_agreement.solid_distance_agreement.solid_f1_at_10cm",
    "band3d_agreement.solid_distance_agreement.solid_precision_at_10cm",
    "band3d_agreement.solid_distance_agreement.solid_recall_at_10cm",
    "band3d_agreement.solid_distance_agreement.median_solid_distance_m",
    "map.free_space_contradiction_rate_at_reference_scale",
    "map.voxel_band_3d.free_fraction",
    "map.voxel_band_3d.occupied_static_fraction",
    "map.voxel_band_3d.movable_static_fraction",
    "map.voxel_band_3d.dynamic_fraction",
    "map.voxel_band_3d.unknown_fraction",
    "map.full_grid.free_fraction",
    "map.full_grid.occupied_fraction",
    "map.full_grid.unknown_fraction",
    "map.free_space_contradiction_rate",
    "map.mean_map_confidence",
    "floor.inlier_ratio",
    "floor.floor_tilt_to_band_axis_deg_before",
    "floor.floor_tilt_to_band_axis_deg_after",
)
# Categorical / status paths diffed as old -> new.
CATEGORICAL_METRICS: tuple[str, ...] = (
    "acceptance.final_category",
    "acceptance.accepted_for_metric_training",
    "acceptance.measured_baseline_category",
    "acceptance.monocular_candidate_category",
    "scale_posterior.metric_acceptance_status",
    "camera_center_sim3_error.status",
    "band3d_agreement.status",
    "floor.method",
)


def _dotted(track: dict[str, Any], dotted: str) -> Any:
    return _pluck(track, *dotted.split("."), default=NOT_REPORTED)


def diff_scorecards(new: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    if prev is None:
        return {"status": "baseline_run", "note": "no previous scorecard; nothing to diff against"}
    # Recompute both content hashes rather than trusting the stored field, so the
    # identical flag reflects actual content (immune to a stale/tampered stored hash).
    new_hash = _canonical_sha256(new)
    prev_hash = _canonical_sha256(prev)
    diff: dict[str, Any] = {
        "status": "diffed",
        "previous_commit": _pluck(prev, "meta", "git_commit"),
        "previous_sha256": prev_hash,
        "current_commit": _pluck(new, "meta", "git_commit"),
        "current_sha256": new_hash,
        "identical": new_hash == prev_hash,
        "tracks": {},
    }
    for asset_id in CANONICAL_TRACKS:
        new_t = _pluck(new, "tracks", asset_id, default={})
        prev_t = _pluck(prev, "tracks", asset_id, default={})
        numeric: dict[str, Any] = {}
        for path in NUMERIC_METRICS:
            nv, pv = _dotted(new_t, path), _dotted(prev_t, path)
            if nv == pv:
                continue
            if isinstance(nv, (int, float)) and isinstance(pv, (int, float)) and not isinstance(nv, bool):
                numeric[path] = {"old": pv, "new": nv, "delta": nv - pv}
            else:
                numeric[path] = {"old": pv, "new": nv, "delta": "non_numeric_change"}
        categorical: dict[str, Any] = {}
        for path in CATEGORICAL_METRICS:
            nv, pv = _dotted(new_t, path), _dotted(prev_t, path)
            if nv != pv:
                categorical[path] = {"old": pv, "new": nv}
        diff["tracks"][asset_id] = {"numeric": numeric, "categorical": categorical}
    return diff


# ---------------------------------------------------------------------------
# Human-readable markdown summary
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "null"
    return str(value)


def _fmt_delta(delta: Any) -> str:
    if isinstance(delta, (int, float)) and not isinstance(delta, bool):
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.6g}"
    return str(delta)


def render_summary(
    scorecard: dict[str, Any], diff: dict[str, Any], generated_at: str, git_tree_dirty: Any
) -> str:
    meta = scorecard["meta"]
    lines: list[str] = []
    lines.append("# Atlas3R Teacher Scorecard")
    lines.append("")
    lines.append(f"- generated_at (wall-clock, non-canonical): {generated_at}")
    lines.append(f"- git_commit: {_fmt(meta.get('git_commit'))}")
    lines.append(f"- git_commit_iso: {_fmt(meta.get('git_commit_iso'))}")
    lines.append(f"- git_tree_dirty (non-canonical, excluded from scorecard hash): {_fmt(git_tree_dirty)}")
    lines.append(f"- scorecard_sha256: {_fmt(meta.get('scorecard_sha256'))}")
    env = meta.get("robot_envelope", {})
    lines.append(
        "- robot_envelope: "
        f"collision_height_m={_fmt(env.get('collision_height_m'))}, "
        f"margin_m={_fmt(env.get('margin_m'))}, "
        f"voxel_size_m={_fmt(env.get('voxel_size_m'))}, "
        f"band_height_m={_fmt(env.get('band_height_m'))}"
    )
    bb = meta.get("backbone_provenance", {})
    lines.append(f"- geometry_backbone: {_fmt(bb.get('method'))} (metric_evidence={_fmt(bb.get('metric_evidence'))})")
    assets = meta.get("asset_provenance", {})
    for asset_id, info in assets.items():
        lines.append(
            f"- asset[{asset_id}]: status={_fmt(info.get('status'))}, frame_count={_fmt(info.get('frame_count'))}"
        )
    if meta.get("missing_reports"):
        lines.append(f"- [!] MISSING REPORTS: {meta['missing_reports']}")
    lines.append("")

    for asset_id in CANONICAL_TRACKS:
        track = scorecard["tracks"].get(asset_id, {})
        lines.append(f"## {asset_id}")
        if track.get("status") == "missing_track_report":
            lines.append("")
            lines.append("[!] NO TEACHER REPORT FOR THIS CANONICAL TRACK -- nothing aggregated.")
            lines.append("")
            continue
        lines.append("")
        acc = track.get("acceptance", {})
        lines.append(f"- final_category: **{_fmt(acc.get('final_category'))}**")
        lines.append(f"- accepted_for_metric_training: {_fmt(acc.get('accepted_for_metric_training'))}")
        lines.append(f"- measured_baseline_category: {_fmt(acc.get('measured_baseline_category'))}")
        lines.append(f"- monocular_candidate_category: {_fmt(acc.get('monocular_candidate_category'))}")
        lines.append(f"- rejection_reasons: {_fmt(acc.get('rejection_reasons'))}")
        lines.append(f"- exact_blockers: {_fmt(acc.get('exact_blockers'))}")

        gc = track.get("gate_cascade", {})
        lines.append("")
        lines.append("### gt-free gate cascade (candidate path)")
        if gc.get("status"):
            lines.append(f"- status: {_fmt(gc.get('status'))}")
        else:
            lines.append(
                f"- stage0 evidence mass: passes={_fmt(gc.get('stage0_passes'))} "
                f"(inbounds={_fmt(gc.get('stage0_median_inbounds_ratio'))}, "
                f"edge_fraction={_fmt(gc.get('stage0_depth_residual_edge_fraction'))}, "
                f"confidence={_fmt(gc.get('stage0_mean_confidence_weight'))})"
            )
            lines.append(
                f"- stage1 gravity: passes={_fmt(gc.get('stage1_passes'))} "
                f"(up_alignment_applied={_fmt(gc.get('stage1_up_alignment_applied'))}, "
                f"floor_inlier={_fmt(gc.get('stage1_floor_inlier_ratio'))})"
            )
            lines.append(f"- threshold_authority: {_fmt(gc.get('threshold_authority'))}")

        ps = track.get("prerefine_severity", {})
        if ps.get("status") != "absent":
            lines.append(
                f"- prerefine severity (reportage, gates nothing): "
                f"p90_log_depth_residual={_fmt(ps.get('prerefine_p90_log_depth_residual'))} "
                f"({_fmt(ps.get('authority'))})"
            )

        pl = track.get("plane_ledger", {})
        lines.append("")
        lines.append("### plane ledger (rigid-world drift audit, reportage only)")
        if pl.get("status") == "audited":
            sig = pl.get("signals") or {}
            da = pl.get("direction_authority") or {}
            lines.append(
                f"- audited: {_fmt(pl.get('n_tracks_qualifying'))}/{_fmt(pl.get('n_tracks_total'))} "
                f"qualifying tracks; normal-span conditioning="
                f"{_fmt(da.get('normal_span_singular_values_relative'))}"
            )
            lines.append(
                f"- offset drift p90 (span fraction): {_fmt(sig.get('ledger_offset_drift_p90_span_fraction'))} "
                f"(noise floor {_fmt(sig.get('ledger_noise_floor_p90_span_fraction'))}); "
                f"normal drift p90: {_fmt(sig.get('ledger_normal_drift_p90_deg'))} deg; "
                f"scale ramp p90 |log ratio|: {_fmt(sig.get('ledger_scale_ramp_p90_abs_log_ratio'))}"
            )
        else:
            lines.append(
                f"- [!] {_fmt(pl.get('status'))} (authority={_fmt(pl.get('authority'))}; "
                f"abstention is authority loss, never a pass)"
            )

        ea = track.get("epipolar_audit", {})
        lines.append("")
        lines.append("### epipolar audit (independent auditor, reportage only)")
        if ea.get("status") == "audited":
            rot = ea.get("rotation_deviation_deg") or {}
            cyc = ea.get("auditor_cycle_residual_deg") or {}
            lines.append(
                f"- audited: {_fmt(ea.get('n_valid_pairs'))} valid pairs, "
                f"authority={_fmt(ea.get('authority'))}, "
                f"intrinsics={_fmt(ea.get('intrinsics_source'))}"
            )
            lines.append(
                f"- rotation deviation deg: median={_fmt(rot.get('median'))} "
                f"p90={_fmt(rot.get('p90'))} vs auditor noise floor p90="
                f"{_fmt(cyc.get('p90') if isinstance(cyc, dict) else None)} "
                f"-> {_fmt(ea.get('deviation_authority'))}"
            )
        else:
            lines.append(
                f"- [!] {_fmt(ea.get('status'))} (authority={_fmt(ea.get('authority'))}; "
                f"abstention is authority loss, never a pass)"
            )

        sp = track.get("scale_posterior", {})
        lines.append("")
        lines.append("### scale posterior")
        lines.append(f"- scale_mean: {_fmt(sp.get('scale_mean'))}")
        lines.append(f"- scale_std: {_fmt(sp.get('scale_std'))}")
        lines.append(f"- relative_scale_uncertainty: {_fmt(sp.get('relative_scale_uncertainty'))}")
        lines.append(f"- metric_acceptance_status: {_fmt(sp.get('metric_acceptance_status'))}")
        lines.append(f"- measured_evidence_present: {_fmt(sp.get('measured_evidence_present'))}")

        cc = track.get("camera_center_sim3_error", {})
        lines.append("")
        lines.append("### camera-center Sim(3) error vs measured")
        if cc.get("status") == "computed":
            lines.append(f"- method: {_fmt(cc.get('method'))}")
            lines.append(f"- trajectory_rmse_m: {_fmt(cc.get('trajectory_rmse_m'))}")
            lines.append(f"- trajectory_median_m: {_fmt(cc.get('trajectory_median_m'))}")
            lines.append(f"- trajectory_max_m: {_fmt(cc.get('trajectory_max_m'))}")
            lines.append(
                f"- estimated_scale_monocular_to_measured: {_fmt(cc.get('estimated_scale_monocular_to_measured'))}"
            )
        else:
            lines.append(
                f"- [!] DID NOT RUN -- status: **{_fmt(cc.get('status'))}** "
                f"({_fmt(cc.get('status_origin'))}; "
                f"teacher field {_fmt(cc.get('teacher_field'))} = {_fmt(cc.get('teacher_field_value'))})"
            )
            if cc.get("teacher_note") not in (None, NOT_REPORTED):
                lines.append(f"- teacher_note (verbatim): {_fmt(cc.get('teacher_note'))}")

        b3 = track.get("band3d_agreement", {})
        lines.append("")
        lines.append("### band3d agreement vs measured 3D field (reportage only)")
        if b3.get("measured_comparison_ran"):
            lines.append(f"- status: {_fmt(b3.get('status'))}")
            lines.append(f"- per_class_agreement: {_fmt(b3.get('per_class_agreement'))}")
            lines.append(f"- occupied_static_iou: {_fmt(b3.get('occupied_static_iou'))}")
            lines.append(f"- free_space_contradiction_rate: {_fmt(b3.get('free_space_contradiction_rate'))}")
            lines.append(f"- dynamic_leakage_rate: {_fmt(b3.get('dynamic_leakage_rate'))}")
            lines.append(f"- coverage_of_measured_band: {_fmt(b3.get('coverage_of_measured_band'))}")
            lines.append(
                f"- estimated_scale_monocular_to_measured: {_fmt(b3.get('estimated_scale_monocular_to_measured'))}"
            )
        else:
            lines.append(
                f"- [!] MEASURED COMPARISON DID NOT RUN -- status: **{_fmt(b3.get('status'))}**"
            )
            lines.append(f"- note: {_fmt(b3.get('note'))}")

        mp = track.get("map", {})
        band = mp.get("voxel_band_3d", {})
        full = mp.get("full_grid", {})
        lines.append("")
        lines.append("### map (occupancy)")
        lines.append(
            "- voxel band 3D fractions: "
            f"free={_fmt(band.get('free_fraction'))}, "
            f"occupied_static={_fmt(band.get('occupied_static_fraction'))}, "
            f"movable_static={_fmt(band.get('movable_static_fraction'))}, "
            f"dynamic={_fmt(band.get('dynamic_fraction'))}, "
            f"unknown={_fmt(band.get('unknown_fraction'))}"
        )
        lines.append(
            "- full grid fractions: "
            f"free={_fmt(full.get('free_fraction'))}, "
            f"occupied={_fmt(full.get('occupied_fraction'))}, "
            f"unknown={_fmt(full.get('unknown_fraction'))}"
        )
        lines.append(f"- free_space_contradiction_rate: {_fmt(mp.get('free_space_contradiction_rate'))}")
        lines.append(f"- mean_map_confidence: {_fmt(mp.get('mean_map_confidence'))}")

        fl = track.get("floor", {})
        lines.append("")
        lines.append("### floor estimate health")
        lines.append(f"- method: {_fmt(fl.get('method'))}")
        lines.append(f"- inlier_ratio: {_fmt(fl.get('inlier_ratio'))}")
        lines.append(f"- up_alignment_applied: {_fmt(fl.get('up_alignment_applied'))}")
        lines.append(
            f"- floor_tilt_to_band_axis_deg: before={_fmt(fl.get('floor_tilt_to_band_axis_deg_before'))} "
            f"-> after={_fmt(fl.get('floor_tilt_to_band_axis_deg_after'))}"
        )
        lines.append(f"- blockers: {_fmt(fl.get('blockers'))}")
        lines.append("")

    lines.append("## diff vs previous scorecard")
    lines.append("")
    if diff.get("status") == "baseline_run":
        lines.append("baseline run -- no previous scorecard to diff against.")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"- previous_commit: {_fmt(diff.get('previous_commit'))}")
    lines.append(f"- current_commit: {_fmt(diff.get('current_commit'))}")
    lines.append(f"- identical_to_previous: {_fmt(diff.get('identical'))}")
    lines.append("")
    any_change = False
    for asset_id in CANONICAL_TRACKS:
        td = diff.get("tracks", {}).get(asset_id, {})
        numeric = td.get("numeric", {})
        categorical = td.get("categorical", {})
        if not numeric and not categorical:
            continue
        any_change = True
        lines.append(f"### {asset_id} changes")
        for path, ch in categorical.items():
            lines.append(f"- {path}: {_fmt(ch['old'])} -> {_fmt(ch['new'])}")
        for path, ch in numeric.items():
            lines.append(
                f"- {path}: {_fmt(ch['old'])} -> {_fmt(ch['new'])} (delta {_fmt_delta(ch['delta'])})"
            )
        lines.append("")
    if not any_change:
        lines.append("no metric changes vs previous scorecard.")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate(
    root: Path,
    teacher_dir: Path,
    eval_dir: Path,
    *,
    no_run: bool,
    generated_at: str,
) -> dict[str, Any]:
    if not no_run:
        run_teacher_subprocess(root, teacher_dir)

    run = load_teacher_run(root, teacher_dir)
    if run["status"] in {"missing_teacher_run"}:
        return {"status": run["status"], "blocker": run.get("blocker")}

    scorecard = build_scorecard(root, run)

    eval_path = root / eval_dir
    scorecard_path = eval_path / SCORECARD_NAME
    prev_path = eval_path / PREV_NAME

    prev = None
    if scorecard_path.is_file():
        prev = json.loads(scorecard_path.read_text(encoding="utf-8"))

    diff = diff_scorecards(scorecard, prev)

    # Rotate the current scorecard to .prev BEFORE overwriting, so there is always
    # exactly one prior to diff against next time.
    if scorecard_path.is_file():
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        prev_path.write_text(scorecard_path.read_text(encoding="utf-8"), encoding="utf-8")

    _write_json(scorecard_path, scorecard)
    _write_json(eval_path / DIFF_NAME, diff)
    # Working-tree dirtiness is non-canonical (transient, not a teacher input): it is
    # shown in the human .md header but never embedded in the hashed scorecard.json.
    git_porcelain = _run_git(["status", "--porcelain"], root)
    git_tree_dirty = bool(git_porcelain) if git_porcelain is not None else NOT_REPORTED
    summary_md = render_summary(scorecard, diff, generated_at, git_tree_dirty)
    (eval_path / SUMMARY_NAME).write_text(summary_md, encoding="utf-8")

    return {
        "status": "ok",
        "scorecard_path": _rel(scorecard_path, root),
        "summary_path": _rel(eval_path / SUMMARY_NAME, root),
        "diff_path": _rel(eval_path / DIFF_NAME, root),
        "scorecard_sha256": scorecard["meta"]["scorecard_sha256"],
        "diff": diff,
        "summary_md": summary_md,
    }


def _now_iso() -> str:
    # Wall-clock, used ONLY for the human summary header (never the canonical JSON).
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas3R teacher evaluation harness (read-and-aggregate).")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="aggregate the latest existing teacher run instead of running the teacher first",
    )
    parser.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR))
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--root", default=None, help="repo root (default: current working directory)")
    args = parser.parse_args(argv)

    root = Path.cwd() if args.root is None else Path(args.root)
    result = evaluate(
        root,
        Path(args.teacher_dir),
        Path(args.eval_dir),
        no_run=args.no_run,
        generated_at=_now_iso(),
    )

    if result["status"] != "ok":
        sys.stderr.write(f"[evaluate] blocked: {result.get('blocker')}\n")
        return 1

    sys.stdout.write(result["summary_md"])
    sys.stdout.write(
        f"\n[evaluate] wrote {result['scorecard_path']} "
        f"(sha256={result['scorecard_sha256'][:12]}...), "
        f"{result['summary_path']}, {result['diff_path']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
