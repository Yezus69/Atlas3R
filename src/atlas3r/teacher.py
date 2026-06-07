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

from .contracts import TrackType, VideoAsset
from .geometry_adapter import load_geometry_artifacts, load_measured_packets_from_m2
from .m1 import DEFAULT_MANIFEST_PATH, _jsonable, _resolve_path, _write_json, load_canonical_assets
from .mapping import fuse_static_map
from .scale import estimate_scale_posterior
from .validation import validate_and_accept
from .visibility import build_visibility_graph

DEFAULT_OUTPUT_DIR = Path("runs/teacher")
DEFAULT_ARTIFACTS_DIR = "external/teacher_artifacts"
DEFAULT_M1_DIR = "runs/m1"
DEFAULT_M2_DIR = "runs/m2"


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
        )
        report_path = output_path / f"{asset.asset_id}_teacher_report.json"
        _write_json(report_path, track_report)
        report_paths[asset.asset_id] = str(report_path)
        per_track.append(track_report)

    summary = {
        "module": "Atlas3R Teacher",
        "status": "complete",
        "track_reports": report_paths,
        "tracks": [
            {
                "asset_id": tr["asset_id"],
                "track_type": tr["track_type"],
                "final_category": tr["final_category"],
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
) -> dict[str, Any]:
    asset_id = asset.asset_id
    is_reference = asset.track_type is TrackType.REFERENCE_METRIC
    blockers: list[str] = []

    report: dict[str, Any] = {
        "module": "Atlas3R Teacher",
        "asset_id": asset_id,
        "track_type": asset.track_type.value,
        "asset_availability": {},
        "geometry_source_status": {},
        "packet_creation_status": {},
        "visibility_residual_status": {},
        "scale_evidence": {},
        "map_occupancy_status": {},
        "validation_status": {},
        "final_category": "rejected",
        "exact_blockers": [],
    }

    # (a) asset availability from M1
    availability = _stage(
        blockers, "asset_availability",
        lambda: _asset_availability(asset_id, root, m1_dir),
    )
    report["asset_availability"] = availability

    # (b) measured reference (M2) + (c) geometry source (M3)
    measured_packets = []
    measured_status = _stage(
        blockers, "measured_reference",
        lambda: _load_measured(asset_id, root, m2_dir) if is_reference else _not_applicable("not_reference_metric_track"),
    )
    if isinstance(measured_status, Mapping) and measured_status.get("status") == "loaded":
        measured_packets = measured_status.get("_packets", [])
    measured_report = {k: v for k, v in measured_status.items() if k != "_packets"} if isinstance(measured_status, Mapping) else measured_status

    geometry_status = _stage(
        blockers, "geometry_source",
        lambda: _load_geometry(asset_id, root, artifacts_dir),
    )
    monocular_packets = geometry_status.get("_packets", []) if isinstance(geometry_status, Mapping) else []
    geometry_soft_evidence = geometry_status.get("_scale_evidence", []) if isinstance(geometry_status, Mapping) else []
    geometry_report = {k: v for k, v in geometry_status.items() if not str(k).startswith("_")} if isinstance(geometry_status, Mapping) else geometry_status

    report["geometry_source_status"] = {
        "measured_reference": measured_report,
        "monocular_artifact": geometry_report,
    }

    # Choose the working packet set: for reference_metric prefer measured
    # packets; for phone_room use monocular artifacts. Record the comparison.
    if is_reference and measured_packets:
        working_packets = measured_packets
        packet_source = "measured_reference"
    elif monocular_packets:
        working_packets = monocular_packets
        packet_source = "monocular_artifact"
    else:
        working_packets = []
        packet_source = "none"

    monocular_vs_measured = None
    if is_reference and measured_packets and monocular_packets:
        monocular_vs_measured = {
            "measured_packet_count": len(measured_packets),
            "monocular_packet_count": len(monocular_packets),
            "comparison": "monocular_artifact_available_alongside_measured_reference",
            "note": "monocular is non-metric (free gauge); measured reference is metric GT",
        }

    report["packet_creation_status"] = {
        "packet_source": packet_source,
        "working_packet_count": len(working_packets),
        "measured_packet_count": len(measured_packets),
        "monocular_packet_count": len(monocular_packets),
        "monocular_vs_measured": monocular_vs_measured,
        "status": "ready" if working_packets else "no_packets_available",
    }
    if not working_packets:
        blockers.append(f"no_packets_for_{asset_id}")

    # (e) visibility / residual
    visibility_report = _stage(
        blockers, "visibility_residual",
        lambda: _visibility(working_packets),
    )
    report["visibility_residual_status"] = _strip_private(visibility_report)

    # (f) scale evidence + posterior. The scale evidence must match the working
    # packet set: measured evidence anchors the measured reference packets;
    # a learned metric-depth prior (soft, measured=False) anchors the monocular
    # packets and can back at most a metric_pseudo_label. A monocular set with no
    # such prior stays unanchored (non_metric_pseudo_label).
    if packet_source == "monocular_artifact":
        scale_evidence = list(geometry_soft_evidence)
    else:
        scale_evidence = _collect_scale_evidence(measured_status, is_reference)
    scale_result = _stage(
        blockers, "scale_posterior",
        lambda: _scale(working_packets, scale_evidence),
    )
    scale_posterior = scale_result.get("_posterior") if isinstance(scale_result, Mapping) else None
    report["scale_evidence"] = {
        "scale_evidence_count": len(scale_evidence),
        "measured_evidence_present": any(getattr(e, "measured", False) for e in scale_evidence),
        "scale_evidence_records": [_jsonable(e) for e in scale_evidence],
        "classification": _strip_private(scale_result),
    }

    if scale_posterior is None:
        report["map_occupancy_status"] = {"status": "blocked_no_scale_posterior"}
        report["validation_status"] = {"status": "blocked_no_scale_posterior"}
        report["final_category"] = "rejected"
        blockers.append(f"no_scale_posterior_for_{asset_id}")
        report["exact_blockers"] = _dedupe(blockers)
        return report

    # (g) map / occupancy
    map_result = _stage(
        blockers, "map_occupancy",
        lambda: _map(working_packets, scale_posterior),
    )
    map_report = map_result.get("_map_report", map_result) if isinstance(map_result, Mapping) else map_result
    report["map_occupancy_status"] = _strip_private(map_result)

    # (h) validation + final category
    validation_result = _stage(
        blockers, "validation",
        lambda: _validate(
            asset_id, working_packets, scale_posterior,
            _strip_private(visibility_report),
            map_report if isinstance(map_report, Mapping) else {},
            measured_packets if is_reference else None,
        ),
    )
    final_category = validation_result.get("final_category", "rejected") if isinstance(validation_result, Mapping) else "rejected"
    report["validation_status"] = _strip_private(validation_result)
    report["final_category"] = final_category

    # Collect blockers reported by validation itself.
    if isinstance(validation_result, Mapping):
        for b in validation_result.get("blockers", ()):  # type: ignore[union-attr]
            blockers.append(str(b))

    report["exact_blockers"] = _dedupe(blockers)
    return report


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


def _scale(packets: Sequence[Any], scale_evidence: Sequence[Any]) -> dict[str, Any]:
    posterior, report = estimate_scale_posterior(packets, scale_evidence)
    return {**report, "_posterior": posterior}


def _map(packets: Sequence[Any], scale_posterior: Any) -> dict[str, Any]:
    voxel_map, grid, report = fuse_static_map(packets, scale_posterior)
    return {
        **report,
        "_map_report": report,
        "voxel_map_produced": voxel_map is not None,
        "occupancy_grid_produced": grid is not None,
    }


def _validate(
    asset_id: str,
    packets: Sequence[Any],
    scale_posterior: Any,
    visibility_report: Mapping[str, Any],
    map_report: Mapping[str, Any],
    measured_reference: Sequence[Any] | None,
) -> dict[str, Any]:
    _report, final_category, validation_dict = validate_and_accept(
        asset_id, packets, scale_posterior, visibility_report, map_report, measured_reference
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
