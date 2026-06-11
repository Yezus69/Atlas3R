"""Dataset emitter: TrainingSample / DatasetManifest (ARCHITECTURE.md API Contracts).

Packages ACCEPTED teacher scenes into the unit a student trainer consumes:
label field + per-frame views + trust channels + provenance, under the binding
honesty rules of the contract:

- only ``accepted_for_metric_training`` scenes are emitted; the emitter REFUSES
  rejected scenes loudly and there is no force flag;
- no measured-GT-derived value appears in any trainer-consumable field;
- the label field is the contract-validated ``voxel_occupancy_3d.npz`` carried
  verbatim (re-validated through :class:`VoxelOccupancyGrid3D` at emit time, so
  an unknown-as-free flip cannot ship);
- every trust channel carries an explicit calibration-status marker
  (``confidence_calibration``, ``scale_status``);
- a scene with no license record (or a non-commercial-clean one) is excluded
  with a named reason -- one NC source poisons a sellable dataset.

The manifest is deterministic (no timestamps inside hashed content) and
content-hashed per file, so corruption is detectable by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
# License names that poison a sellable dataset (matched case-insensitively
# against the recorded license name).
_NON_COMMERCIAL_MARKERS = ("nc", "noncommercial", "non-commercial", "sharealike", "by-sa")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown_commit"


def _license_verdict(track: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(license_record, refusal_reason)`` -- exactly one is None."""
    record = (track.get("metadata") or {}).get("license")
    if not isinstance(record, dict) or not record.get("name"):
        return None, "missing_license_record"
    name = str(record["name"]).lower().replace(" ", "")
    if any(marker in name for marker in _NON_COMMERCIAL_MARKERS):
        return None, f"license_not_commercial_clean:{record['name']}"
    return record, None


def _revalidate_label_field(npz_path: Path) -> dict[str, Any]:
    """Re-validate the exported label field through the contract before it ships.

    Reconstructs :class:`VoxelOccupancyGrid3D` from the npz arrays so the
    contract's per-voxel invariants (unknown never free, dynamic never static,
    movable never free) run on EXACTLY the bytes a trainer would read."""
    import numpy as np  # lazy

    from .contracts import VOXEL_EVIDENCE_COUNT_FIELDS, VoxelOccupancyGrid3D

    z = np.load(npz_path)
    evidence = {
        f: (np.asarray(z[f]) if f in z.files else None)
        for f in VOXEL_EVIDENCE_COUNT_FIELDS
    }
    grid = VoxelOccupancyGrid3D(
        grid_frame=str(z["grid_frame"]),
        voxel_size_m=float(z["voxel_size_m"]),
        origin_world=tuple(float(v) for v in z["origin_world"]),
        floor_axis=int(z["floor_axis"]),
        band_min_m=float(z["band_min_m"]),
        band_max_m=float(z["band_max_m"]),
        P_free=z["P_free"],
        P_occupied_static=z["P_occupied_static"],
        P_movable_static=z["P_movable_static"],
        P_dynamic=z["P_dynamic"],
        P_unknown=z["P_unknown"],
        map_confidence=z["map_confidence"],
        scale_uncertainty=float(z["scale_uncertainty"]),
        acceptance_category=str(z["acceptance_category"]),
        **evidence,
    )
    return {
        "grid_frame": grid.grid_frame,
        "voxel_size_m": grid.voxel_size_m,
        "origin_world": [float(v) for v in grid.origin_world],
        "floor_axis": grid.floor_axis,
        "band_min_m": grid.band_min_m,
        "band_max_m": grid.band_max_m,
        "confidence_calibration": (
            str(z["confidence_calibration"])
            if "confidence_calibration" in z.files else "uncalibrated_heuristic"
        ),
        "scale_uncertainty": float(z["scale_uncertainty"]),
        "acceptance_category": str(z["acceptance_category"]),
    }


def _emit_sample(
    root: Path,
    asset_id: str,
    track: dict[str, Any],
    report: dict[str, Any],
    teacher_dir: Path,
    artifacts_dir: Path,
    out_dir: Path,
    commit: str,
) -> dict[str, Any]:
    """Emit one TrainingSample; returns the manifest entry or a refusal record."""
    validation = report.get("validation_status") or {}
    if not bool(validation.get("accepted_for_metric_training")):
        return {
            "asset_id": asset_id,
            "emitted": False,
            "reason": "scene_not_accepted_for_metric_training",
            "rejection_reasons": list(validation.get("rejection_reasons") or ()),
        }
    license_record, refusal = _license_verdict(track)
    if refusal is not None:
        return {"asset_id": asset_id, "emitted": False, "reason": refusal}

    scene_dir = teacher_dir / asset_id
    npz_src = scene_dir / "voxel_occupancy_3d.npz"
    traj_src = scene_dir / "camera_trajectory.json"
    poses_src = artifacts_dir / asset_id / "poses.json"
    missing = [str(p) for p in (npz_src, traj_src, poses_src) if not p.exists()]
    if missing:
        return {"asset_id": asset_id, "emitted": False,
                "reason": "missing_inputs", "missing": missing}

    label_summary = _revalidate_label_field(npz_src)  # raises on invariant breach

    sample_id = f"{asset_id}@{commit[:7]}"
    sample_dir = out_dir / sample_id
    rgb_dir = sample_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    label_dst = sample_dir / "voxel_occupancy_3d.npz"
    shutil.copyfile(npz_src, label_dst)

    poses = json.loads(poses_src.read_text(encoding="utf-8"))
    rgb_by_frame: dict[int, Path] = {}
    for fr in poses.get("frames", ()):  # source RGB paths live in poses.json
        src = fr.get("source_frame_path")
        if src:
            rgb_by_frame[int(fr["frame_id"])] = root / src

    per_frame_intr: dict[str, Any] = {}
    pfi_path = artifacts_dir / asset_id / "per_frame_intrinsics.json"
    if pfi_path.exists():
        per_frame_intr = json.loads(pfi_path.read_text(encoding="utf-8"))
    intr_global = json.loads(
        (artifacts_dir / asset_id / "intrinsics.json").read_text(encoding="utf-8")
    )
    intr_w = int(intr_global.get("width_px", 0))
    intr_h = int(intr_global.get("height_px", 0))

    backbone_manifest = json.loads(
        (artifacts_dir / asset_id / "backbone_manifest.json").read_text(encoding="utf-8")
    )
    stability = (backbone_manifest.get("depth_source") or {}).get("stability") or {}
    pose_provenance = poses.get("pose_provenance") or {}

    traj = json.loads(traj_src.read_text(encoding="utf-8"))
    frames_out: list[dict[str, Any]] = []
    dropped_no_rgb: list[int] = []
    file_hashes: dict[str, str] = {"voxel_occupancy_3d.npz": _sha256(label_dst)}
    for entry in traj.get("frames", ()):  # poses already in the floor-aligned grid frame
        fid = int(entry["frame_id"])
        rgb_src = rgb_by_frame.get(fid)
        if rgb_src is None or not rgb_src.exists():
            dropped_no_rgb.append(fid)  # honest drop, never a fabricated view
            continue
        rgb_name = f"rgb/{fid}{rgb_src.suffix}"
        rgb_dst = sample_dir / rgb_name
        shutil.copyfile(rgb_src, rgb_dst)
        file_hashes[rgb_name] = _sha256(rgb_dst)
        intr = per_frame_intr.get(str(fid)) or {
            k: intr_global[k] for k in ("fx", "fy", "cx", "cy")
        }
        frames_out.append({
            "frame_id": fid,
            "rgb_path": rgb_name,
            "rgb_sha256": file_hashes[rgb_name],
            "intrinsics": {
                "fx": float(intr["fx"]), "fy": float(intr["fy"]),
                "cx": float(intr["cx"]), "cy": float(intr["cy"]),
                "width_px": intr_w, "height_px": intr_h,
            },
            "T_grid_camera": entry["T_world_camera"],
        })

    gate = (validation.get("gate_cascade") or {}).get("stage2b_prerefine_residual") or {}
    sample = {
        "sample_id": sample_id,
        "asset_id": asset_id,
        "teacher_commit": commit,
        "recipe": backbone_manifest.get("method", "unknown"),
        "robot_envelope": (report.get("map_occupancy_status") or {}).get("robot_envelope"),
        "acceptance": {
            "category": validation.get("final_category"),
            "accepted_for_metric_training": True,
            "gate_provenance_class": gate.get("pose_provenance_class"),
            "validation_report_path": str(
                (teacher_dir / f"{asset_id}_teacher_report.json").relative_to(root)
            ),
        },
        "label_field": {"path": "voxel_occupancy_3d.npz",
                        "sha256": file_hashes["voxel_occupancy_3d.npz"],
                        **label_summary},
        "frames": frames_out,
        "frames_dropped_no_rgb": dropped_no_rgb,
        "intrinsics_resolution_note": (
            "intrinsics are stated at their OWN width_px/height_px (the producing"
            " pipeline's processed resolution); rescale to the RGB resolution"
            " before projecting"
        ),
        "scale": {
            "claimed_scale": 1.0,
            "scale_uncertainty": label_summary["scale_uncertainty"],
            "scale_status": "per_backbone_constant_prior",
        },
        "provenance": {
            "source_uri_or_path": track.get("source_uri_or_path"),
            "license": license_record,
            "backbone": backbone_manifest.get("backbone_name"),
            "pose_source": pose_provenance.get("pose_source")
            or poses.get("source"),
            "stability": {"tau": stability.get("tau"),
                          "k": (report.get("map_occupancy_status") or {})
                          .get("robot_envelope", {}).get("verified_surface_min_count")},
            "domain": "indoor",
        },
    }
    sample_json = sample_dir / "sample.json"
    sample_json.write_text(json.dumps(sample, indent=1, sort_keys=True), encoding="utf-8")
    file_hashes["sample.json"] = _sha256(sample_json)
    return {
        "asset_id": asset_id,
        "emitted": True,
        "sample_id": sample_id,
        "sample_dir": str(sample_dir.relative_to(out_dir)),
        "file_sha256": file_hashes,
        "acceptance_category": str(validation.get("final_category")),
        "license_name": license_record["name"],
        "domain": "indoor",
        "frame_count": len(frames_out),
    }


def emit_training_dataset(
    root: Path,
    teacher_dir: Path,
    manifest_path: Path,
    artifacts_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Emit TrainingSamples for every accepted scene + the DatasetManifest."""
    asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = {t["asset_id"]: t for t in asset_manifest.get("tracks", ())}
    commit = _git_commit(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    histogram: dict[str, int] = {}
    for asset_id in sorted(tracks):
        report_path = teacher_dir / f"{asset_id}_teacher_report.json"
        if not report_path.exists():
            refusals.append({"asset_id": asset_id, "emitted": False,
                             "reason": "no_teacher_report"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = _emit_sample(
            root, asset_id, tracks[asset_id], report, teacher_dir, artifacts_dir,
            out_dir, commit,
        )
        if result.get("emitted"):
            entries.append(result)
        else:
            refusals.append(result)
            for reason in result.get("rejection_reasons") or [result["reason"]]:
                key = str(reason).split(":")[0]
                histogram[key] = histogram.get(key, 0) + 1

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "teacher_commit": commit,
        "samples": [
            {k: e[k] for k in ("sample_id", "asset_id", "sample_dir", "file_sha256",
                               "acceptance_category", "license_name", "domain")}
            for e in entries
        ],
        "accounting": {
            "scenes_attempted": len(entries) + len(refusals),
            "scenes_accepted": len(entries),
            "scenes_rejected": len(refusals),
            "rejection_reasons_histogram": dict(sorted(histogram.items())),
            "refusals": refusals,
        },
        "splits": {},
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--teacher-dir", default=Path("runs/teacher"), type=Path)
    parser.add_argument("--manifest", default=Path("config/canonical_assets.json"), type=Path)
    parser.add_argument("--artifacts-dir", default=Path("external/teacher_artifacts"), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else root / p

    manifest = emit_training_dataset(
        root, _abs(args.teacher_dir), _abs(args.manifest), _abs(args.artifacts_dir),
        _abs(args.out_dir),
    )
    print(json.dumps({
        "status": "emitted",
        "scenes_accepted": manifest["accounting"]["scenes_accepted"],
        "scenes_rejected": manifest["accounting"]["scenes_rejected"],
        "samples": [s["sample_id"] for s in manifest["samples"]],
    }, indent=2))


if __name__ == "__main__":
    main()
