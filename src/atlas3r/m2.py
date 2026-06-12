"""M2 measured metric reference ingestion.

This module is intentionally narrow. It supports a local TUM-style RGB-D
directory for the canonical ``reference_metric`` asset and emits reports that
stay useful when measured artifacts or scale-critical metadata are absent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .contracts import (
    ContractValidationError,
    DepthConvention,
    FrameRayPacket,
    ScaleEvidence,
    ScaleEvidenceType,
    TrackType,
    VideoAsset,
    VideoAssetStatus,
)
from .m1 import (
    DEFAULT_MANIFEST_PATH,
    _build_input_cache_state,
    _cache_hit,
    _evenly_spaced_ids,
    _jsonable,
    _resolve_asset_source,
    _resolve_path,
    _write_cache_state,
    _write_json,
    load_canonical_assets,
)

DEFAULT_OUTPUT_DIR = Path("runs/m2")
DEFAULT_M1_REPORT_DIR = Path("runs/m1")
DEFAULT_CACHE_STATE_NAME = ".cache_state.json"

RGB_INDEX_FILE = "rgb.txt"
DEPTH_INDEX_FILE = "depth.txt"
GROUNDTRUTH_FILE = "groundtruth.txt"
ASSOCIATION_FILE_CANDIDATES = (
    "associations.txt",
    "associate.txt",
    "rgb_depth_associations.txt",
)
SIDECAR_METADATA_CANDIDATES = (
    "atlas3r_reference_metadata.json",
    "m2_reference_metadata.json",
    "reference_metadata.json",
    "metadata.json",
)
SUPPORTED_DEPTH_CONVENTIONS = {
    DepthConvention.OPTICAL_Z,
    DepthConvention.RADIAL_RANGE,
}
SUPPORTED_POSE_CONVENTIONS = {"T_world_camera", "T_camera_world"}
REQUIRED_INTRINSIC_FIELDS = ("fx", "fy", "cx", "cy", "width_px", "height_px")
_T = TypeVar("_T")


@dataclass(frozen=True)
class TimestampedFile:
    timestamp_s: float
    relative_path: str
    path: Path


@dataclass(frozen=True)
class PoseEntry:
    timestamp_s: float
    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class RgbDepthPair:
    frame_id: int
    rgb: TimestampedFile
    depth: TimestampedFile


@dataclass(frozen=True)
class RgbPosePair:
    frame_id: int
    rgb: TimestampedFile
    pose: PoseEntry


@dataclass(frozen=True)
class ReferenceAssociation:
    frame_id: int
    rgb: TimestampedFile
    depth: TimestampedFile
    pose: PoseEntry


@dataclass(frozen=True)
class PinholeCameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int
    height_px: int
    model_name: str = "pinhole"
    pixel_coordinate_convention: str = "top_left_pixel_centers"

    def __post_init__(self) -> None:
        for field_name in ("fx", "fy"):
            _require_positive_number(getattr(self, field_name), f"intrinsics.{field_name}")
        for field_name in ("cx", "cy"):
            _require_finite_number(getattr(self, field_name), f"intrinsics.{field_name}")
        for field_name in ("width_px", "height_px"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ContractValidationError(f"intrinsics.{field_name} must be a positive integer")

    def ray_for_pixel(self, pixel_uv: Sequence[float]) -> tuple[float, float, float]:
        u, v = _pixel_uv(pixel_uv)
        x = (u - float(self.cx)) / float(self.fx)
        y = (v - float(self.cy)) / float(self.fy)
        norm = math.sqrt(x * x + y * y + 1.0)
        return (x / norm, y / norm, 1.0 / norm)

    def unproject(self, pixel_uv: object, radial_depth_m: float) -> tuple[float, float, float]:
        _require_positive_number(radial_depth_m, "radial_depth_m")
        ray = self.ray_for_pixel(pixel_uv)  # type: ignore[arg-type]
        depth = float(radial_depth_m)
        return (depth * ray[0], depth * ray[1], depth * ray[2])

    def project(self, X_camera: object) -> dict[str, Any]:
        x, y, z = _camera_xyz(X_camera)
        if z <= 0:
            return {"pixel_uv": None, "radial_depth_m": None, "valid": False}
        u = float(self.fx) * x / z + float(self.cx)
        v = float(self.fy) * y / z + float(self.cy)
        radial_depth_m = math.sqrt(x * x + y * y + z * z)
        valid = 0.0 <= u < float(self.width_px) and 0.0 <= v < float(self.height_px)
        return {"pixel_uv": (u, v), "radial_depth_m": radial_depth_m, "valid": valid}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "pixel_coordinate_convention": self.pixel_coordinate_convention,
        }


def run_m2(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    repo_root: str | Path | None = None,
    m1_report_dir: str | Path = DEFAULT_M1_REPORT_DIR,
    max_packets: int = 8,
    max_rays_per_packet: int = 2048,
    force: bool = False,
) -> dict[str, Any]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    output_path = _resolve_path(Path(output_dir), root)
    output_path.mkdir(parents=True, exist_ok=True)

    assets = load_canonical_assets(manifest_path, repo_root=root)
    report_paths = {
        asset.asset_id: output_path / f"{asset.asset_id}_measured_reference_report.json"
        for asset in assets
    }
    registry_path = output_path / "measured_reference_registry_report.json"
    m1_root = _resolve_path(Path(m1_report_dir), root)
    m1_inputs: list[Path] = [m1_root / "asset_registry_report.json"]
    for asset in assets:
        m1_inputs.append(m1_root / f"{asset.asset_id}_inspection.json")
        m1_inputs.append(m1_root / f"{asset.asset_id}_keyframes.json")
    cache_state = _build_input_cache_state(
        "m2",
        manifest_path,
        assets,
        root,
        {
            "m1_report_dir": str(m1_root),
            "max_packets": int(max_packets),
            "max_rays_per_packet": int(max_rays_per_packet),
        },
        extra_paths=m1_inputs,
    )
    cache_path = output_path / DEFAULT_CACHE_STATE_NAME
    cache_outputs = [registry_path, *report_paths.values()]
    if not force and _cache_hit(cache_path, cache_state, cache_outputs):
        return {
            "registry_report": registry_path,
            "measured_reference_reports": report_paths,
            "reports": [
                json.loads(path.read_text(encoding="utf-8"))
                for path in report_paths.values()
            ],
            "cache_status": "hit",
        }
    reports: list[dict[str, Any]] = []

    for asset in assets:
        if asset.track_type is TrackType.REFERENCE_METRIC:
            report = inspect_reference_metric(
                asset,
                repo_root=root,
                output_dir=output_path,
                m1_report_dir=m1_report_dir,
                max_packets=max_packets,
                max_rays_per_packet=max_rays_per_packet,
            )
        elif asset.track_type is TrackType.PHONE_ROOM:
            report = _phone_no_measured_evidence_report(asset, root)
        else:
            report = _non_reference_report(asset, root)

        report_path = output_path / f"{asset.asset_id}_measured_reference_report.json"
        _write_json(report_path, report)
        reports.append(report)

    registry_report = {
        "manifest_path": str(_resolve_path(Path(manifest_path), root)),
        "report_dir": str(output_path),
        "scale_posterior": None,
        "scale_posterior_status": "not_produced_in_m2",
        "accepted_for_metric_training": False,
        "assets": [
            {
                "asset_id": report["asset_id"],
                "track_type": report["track_type"],
                "status": report["status"],
                "blocking_reasons": report.get("blocking_reasons", []),
                "scale_evidence_count": len(report.get("scale_evidence", [])),
                "frame_ray_packet_count": report.get("frame_ray_packets", {}).get("packet_count", 0),
                "report": f"{report['asset_id']}_measured_reference_report.json",
            }
            for report in reports
        ],
    }
    _write_json(registry_path, registry_report)
    _write_cache_state(cache_path, cache_state, cache_outputs)

    return {
        "registry_report": registry_path,
        "measured_reference_reports": report_paths,
        "reports": reports,
        "cache_status": "miss",
    }


def inspect_reference_metric(
    asset: VideoAsset,
    *,
    repo_root: str | Path | None,
    output_dir: Path,
    m1_report_dir: str | Path,
    max_packets: int,
    max_rays_per_packet: int,
) -> dict[str, Any]:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    resolved = _resolve_asset_source(asset, root)
    base_report = _base_report(asset, root, resolved)

    if resolved is None:
        return {
            **base_report,
            "status": VideoAssetStatus.UNSUPPORTED.value,
            "reference_format": None,
            "blocking_reasons": ("remote_or_non_file_asset_not_available_locally",),
            "evidence_summary": {},
            "missing_fields": (),
            "scale_evidence": (),
            "scale_posterior": None,
            "scale_posterior_status": "not_produced_in_m2",
            "accepted_for_metric_training": False,
            "metric_training_status": "not_accepted_in_m2",
            "frame_ray_packets": _empty_packet_report("not_created_due_to_unsupported_asset"),
        }
    if not resolved.exists():
        return {
            **base_report,
            "status": VideoAssetStatus.MISSING_ASSET.value,
            "reference_format": "tum_rgbd_directory",
            "blocking_reasons": ("missing_asset",),
            "evidence_summary": {"resolved_path": str(resolved), "path_exists": False},
            "missing_fields": (),
            "scale_evidence": (),
            "scale_posterior": None,
            "scale_posterior_status": "not_produced_in_m2",
            "accepted_for_metric_training": False,
            "metric_training_status": "not_accepted_in_m2",
            "frame_ray_packets": _empty_packet_report("not_created_due_to_missing_asset"),
        }
    if not resolved.is_dir():
        return {
            **base_report,
            "status": VideoAssetStatus.UNSUPPORTED.value,
            "reference_format": "tum_rgbd_directory",
            "blocking_reasons": ("reference_metric_source_must_be_local_tum_rgbd_directory",),
            "evidence_summary": {"resolved_path": str(resolved), "path_exists": True, "is_dir": False},
            "missing_fields": (),
            "scale_evidence": (),
            "scale_posterior": None,
            "scale_posterior_status": "not_produced_in_m2",
            "accepted_for_metric_training": False,
            "metric_training_status": "not_accepted_in_m2",
            "frame_ray_packets": _empty_packet_report("not_created_due_to_unsupported_asset"),
        }

    config, metadata_sources, metadata_errors = _merged_reference_metadata(asset, resolved, root)
    file_set = _discover_tum_file_set(resolved, config)

    rgb_entries, rgb_parse_errors = _parse_tum_file_index(file_set["rgb_index"], resolved)
    depth_entries, depth_parse_errors = _parse_tum_file_index(file_set["depth_index"], resolved)
    pose_entries, pose_parse_errors = _parse_tum_groundtruth(file_set["groundtruth"])
    association_rows, association_parse_errors = _parse_association_file(file_set["association"])

    missing_fields: list[str] = []
    invalid_fields: list[dict[str, str]] = []
    intrinsics = _extract_intrinsics(config, missing_fields, invalid_fields) if depth_entries else None
    depth_scale_to_meters = (
        _extract_positive_float(config, "depth_scale_to_meters", "m2_reference.depth_scale_to_meters", missing_fields, invalid_fields)
        if depth_entries
        else None
    )
    source_depth_convention = (
        _extract_depth_convention(config, missing_fields, invalid_fields)
        if depth_entries
        else None
    )
    timestamp_tolerance_s = (
        _extract_non_negative_float(config, "timestamp_tolerance_s", "m2_reference.timestamp_tolerance_s", missing_fields, invalid_fields)
        if pose_entries or (depth_entries and not association_rows)
        else None
    )
    pose_convention = (
        _extract_pose_convention(config, missing_fields, invalid_fields)
        if pose_entries
        else None
    )
    pose_translation_units = (
        _extract_pose_translation_units(config, missing_fields, invalid_fields)
        if pose_entries
        else None
    )

    blocking_reasons = _blocking_reasons(
        rgb_entries=rgb_entries,
        depth_entries=depth_entries,
        pose_entries=pose_entries,
        rgb_parse_errors=rgb_parse_errors,
        depth_parse_errors=depth_parse_errors,
        pose_parse_errors=pose_parse_errors,
        metadata_errors=metadata_errors,
        missing_fields=missing_fields,
        invalid_fields=invalid_fields,
    )

    rgb_depth_pairs = _associate_rgb_depth(
        rgb_entries,
        depth_entries,
        association_rows,
        timestamp_tolerance_s,
    )
    rgb_pose_pairs = _associate_rgb_pose(rgb_entries, pose_entries, timestamp_tolerance_s)
    complete_associations = _complete_associations(rgb_depth_pairs, pose_entries, timestamp_tolerance_s)

    if depth_entries and pose_entries and rgb_depth_pairs and not complete_associations:
        blocking_reasons.append("pose_present_but_not_associated_to_rgb_depth")
    if rgb_entries and depth_entries and not rgb_depth_pairs:
        blocking_reasons.append("depth_present_but_not_associated_to_rgb")

    selected_complete = _select_associations(
        complete_associations,
        asset.asset_id,
        root,
        m1_report_dir,
        max_packets=max_packets,
    )
    selected_depth_pairs = _select_pairs(rgb_depth_pairs, max_packets)
    selected_pose_pairs = _select_pairs(rgb_pose_pairs, max_packets)

    scale_evidence: list[ScaleEvidence] = []
    selected_depth_pairs_with_files = tuple(
        pair for pair in selected_depth_pairs if pair.rgb.path.exists() and pair.depth.path.exists()
    )
    selected_pose_pairs_with_files = tuple(
        pair for pair in selected_pose_pairs if pair.rgb.path.exists()
    )

    if _depth_metadata_complete(intrinsics, depth_scale_to_meters, source_depth_convention) and selected_depth_pairs_with_files:
        scale_evidence.append(
            ScaleEvidence(
                evidence_id=f"{asset.asset_id}_measured_depth",
                evidence_type=ScaleEvidenceType.MEASURED_DEPTH,
                measured=True,
                source="tum_rgbd_depth",
                frame_ids=tuple(pair.frame_id for pair in selected_depth_pairs_with_files),
                confidence=1.0,
                provenance={
                    "format": "tum_rgbd_directory",
                    "depth_index_file": str(file_set["depth_index"]),
                    "depth_scale_to_meters_source": _field_source(config, "depth_scale_to_meters"),
                    "source_depth_convention": source_depth_convention.value if source_depth_convention else None,
                    "bounded_to_selected_frames": True,
                },
            )
        )
    if _pose_metadata_complete(pose_convention, pose_translation_units) and selected_pose_pairs_with_files:
        scale_evidence.append(
            ScaleEvidence(
                evidence_id=f"{asset.asset_id}_measured_pose",
                evidence_type=ScaleEvidenceType.MEASURED_POSE,
                measured=True,
                source="tum_groundtruth_trajectory",
                frame_ids=tuple(pair.frame_id for pair in selected_pose_pairs_with_files),
                confidence=1.0,
                provenance={
                    "format": "tum_rgbd_directory",
                    "groundtruth_file": str(file_set["groundtruth"]),
                    "pose_convention": pose_convention,
                    "pose_translation_units": pose_translation_units,
                    "timestamp_tolerance_s": timestamp_tolerance_s,
                    "bounded_to_selected_frames": True,
                },
            )
        )

    packet_report = _empty_packet_report("not_created_due_to_incomplete_measured_evidence")
    if (
        selected_complete
        and intrinsics is not None
        and depth_scale_to_meters is not None
        and source_depth_convention is not None
        and pose_convention is not None
        and pose_translation_units == "meters"
    ):
        packet_report = _create_packet_sidecar(
            asset,
            selected_complete,
            intrinsics,
            depth_scale_to_meters,
            source_depth_convention,
            pose_convention,
            output_dir,
            max_rays_per_packet=max_rays_per_packet,
        )
        if packet_report["packet_count"] == 0 and "packet_creation_failed" not in blocking_reasons:
            blocking_reasons.append("packet_creation_failed")

    if packet_report["packet_count"] > 0:
        status = VideoAssetStatus.AVAILABLE.value
        blocking_reasons = [reason for reason in blocking_reasons if reason not in {"missing_measured_depth", "missing_measured_pose"}]
    else:
        status = "missing_external_artifact" if resolved.exists() else VideoAssetStatus.MISSING_ASSET.value

    return {
        **base_report,
        "status": status,
        "reference_format": "tum_rgbd_directory",
        "evidence_summary": {
            "resolved_path": str(resolved),
            "path_exists": True,
            "metadata_sources": metadata_sources,
            "metadata_errors": metadata_errors,
            "files": {
                "rgb_index": _file_summary(file_set["rgb_index"], rgb_entries, rgb_parse_errors),
                "depth_index": _file_summary(file_set["depth_index"], depth_entries, depth_parse_errors),
                "groundtruth_trajectory": _pose_file_summary(file_set["groundtruth"], pose_entries, pose_parse_errors),
                "association_file": _association_file_summary(file_set["association"], association_rows, association_parse_errors),
            },
            "local_file_availability": {
                "missing_rgb_files": _missing_entry_files(rgb_entries)[:8],
                "missing_depth_files": _missing_entry_files(depth_entries)[:8],
            },
            "metadata_fields": {
                "intrinsics_present": intrinsics is not None,
                "depth_scale_to_meters_present": depth_scale_to_meters is not None,
                "source_depth_convention": source_depth_convention.value if source_depth_convention else None,
                "timestamp_tolerance_s": timestamp_tolerance_s,
                "pose_convention": pose_convention,
                "pose_translation_units": pose_translation_units,
            },
            "association_counts": {
                "rgb_depth": len(rgb_depth_pairs),
                "rgb_pose": len(rgb_pose_pairs),
                "rgb_depth_pose": len(complete_associations),
            },
        },
        "missing_fields": tuple(_dedupe(missing_fields)),
        "invalid_fields": tuple(invalid_fields),
        "blocking_reasons": tuple(_dedupe(blocking_reasons)),
        "scale_evidence": tuple(scale_evidence),
        "scale_posterior": None,
        "scale_posterior_status": "not_produced_in_m2",
        "accepted_for_metric_training": False,
        "metric_training_status": "not_accepted_in_m2",
        "frame_ray_packets": packet_report,
    }


def _create_packet_sidecar(
    asset: VideoAsset,
    associations: Sequence[ReferenceAssociation],
    camera: PinholeCameraModel,
    depth_scale_to_meters: float,
    source_depth_convention: DepthConvention,
    pose_convention: str,
    output_dir: Path,
    *,
    max_rays_per_packet: int,
) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:
        return {
            **_empty_packet_report("not_created_due_to_missing_decoder_dependency"),
            "missing_dependency": exc.name,
        }

    packet_summaries: list[dict[str, Any]] = []
    skipped_packets: list[dict[str, Any]] = []
    sidecar_arrays: dict[str, Any] = {}
    packets: list[FrameRayPacket] = []

    for association in associations:
        if not association.rgb.path.exists():
            skipped_packets.append({"frame_id": association.frame_id, "reason": "missing_rgb_file", "path": str(association.rgb.path)})
            continue
        if not association.depth.path.exists():
            skipped_packets.append({"frame_id": association.frame_id, "reason": "missing_depth_file", "path": str(association.depth.path)})
            continue
        try:
            with Image.open(association.depth.path) as image:
                raw_depth = np.asarray(image)
        except OSError as exc:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "depth_decode_failed", "error": str(exc)})
            continue

        if raw_depth.ndim != 2:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "depth_image_must_be_single_channel", "shape": tuple(raw_depth.shape)})
            continue
        height_px, width_px = raw_depth.shape
        if width_px != camera.width_px or height_px != camera.height_px:
            skipped_packets.append(
                {
                    "frame_id": association.frame_id,
                    "reason": "depth_image_size_does_not_match_intrinsics",
                    "depth_shape": (int(height_px), int(width_px)),
                    "intrinsics_size": (camera.height_px, camera.width_px),
                }
            )
            continue

        depth_m = raw_depth.astype(np.float64) * float(depth_scale_to_meters)
        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "no_positive_measured_depth_pixels"})
            continue

        sampled_yx = _sample_valid_pixels(valid_mask, max_rays_per_packet, np)
        sampled_uv = np.stack((sampled_yx[:, 1], sampled_yx[:, 0]), axis=1).astype(np.float64)
        rays = _unit_rays_for_pixels(camera, sampled_uv, np)
        source_depth_m = depth_m[sampled_yx[:, 0], sampled_yx[:, 1]]
        try:
            radial_depth_m = _source_depth_to_radial(source_depth_m, rays[:, 2], source_depth_convention, np)
        except ContractValidationError as exc:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "depth_convention_conversion_failed", "error": str(exc)})
            continue
        if not bool(np.all(np.isfinite(radial_depth_m) & (radial_depth_m > 0.0))):
            skipped_packets.append({"frame_id": association.frame_id, "reason": "invalid_radial_depth_after_conversion"})
            continue

        try:
            T_world_camera = _pose_to_T_world_camera(
                association.pose.translation_m,
                association.pose.quaternion_xyzw,
                pose_convention,
            )
        except ContractValidationError as exc:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "pose_transform_construction_failed", "error": str(exc)})
            continue
        packet_key = f"frame_{association.frame_id}"
        rays_packet = rays.reshape((rays.shape[0], 1, 3))
        depth_packet = radial_depth_m.reshape((radial_depth_m.shape[0], 1))
        confidence_packet = np.ones_like(depth_packet, dtype=np.float64)
        try:
            packet = FrameRayPacket(
                asset_id=asset.asset_id,
                frame_id=association.frame_id,
                T_world_camera=T_world_camera,
                rays_camera=rays_packet,
                radial_depth_m=depth_packet,
                confidence=confidence_packet,
                camera_model=camera,
                source="measured_reference:tum_rgbd",
                uncertainty={
                    "depth": "measured_depth_pixels_scaled_to_meters_no_posterior",
                    "pose": "measured_groundtruth_pose_no_optimization",
                },
                provenance={
                    "rgb_timestamp_s": association.rgb.timestamp_s,
                    "rgb_file": str(association.rgb.path),
                    "depth_timestamp_s": association.depth.timestamp_s,
                    "depth_file": str(association.depth.path),
                    "pose_timestamp_s": association.pose.timestamp_s,
                    "pose_convention": pose_convention,
                    "depth_scale_to_meters": depth_scale_to_meters,
                    "source_depth_convention": source_depth_convention.value,
                    "packet_sampling": "bounded_valid_pixel_sample",
                    "sidecar_array_prefix": packet_key,
                },
                source_depth_convention=source_depth_convention,
                intrinsics=camera.to_metadata(),
                camera_confidence=1.0,
            )
        except ContractValidationError as exc:
            skipped_packets.append({"frame_id": association.frame_id, "reason": "frame_ray_packet_contract_rejected", "error": str(exc)})
            continue
        packets.append(packet)

        sidecar_arrays[f"{packet_key}_sampled_pixel_uv"] = sampled_uv.astype(np.float32)
        sidecar_arrays[f"{packet_key}_rays_camera"] = rays_packet.astype(np.float32)
        sidecar_arrays[f"{packet_key}_radial_depth_m"] = depth_packet.astype(np.float32)
        sidecar_arrays[f"{packet_key}_confidence"] = confidence_packet.astype(np.float32)
        sidecar_arrays[f"{packet_key}_T_world_camera"] = np.asarray(T_world_camera, dtype=np.float64)

        packet_summaries.append(
            {
                "asset_id": packet.asset_id,
                "frame_id": packet.frame_id,
                "rgb_timestamp_s": association.rgb.timestamp_s,
                "depth_timestamp_s": association.depth.timestamp_s,
                "pose_timestamp_s": association.pose.timestamp_s,
                "sampled_ray_count": int(rays.shape[0]),
                "source_valid_depth_pixel_count": valid_count,
                "source_depth_pixel_count": int(valid_mask.size),
                "source_valid_depth_fraction": valid_count / int(valid_mask.size),
                "array_shapes": {
                    "rays_camera": tuple(rays_packet.shape),
                    "radial_depth_m": tuple(depth_packet.shape),
                    "confidence": tuple(confidence_packet.shape),
                },
                "camera_model": camera.to_metadata(),
                "source_depth_convention": source_depth_convention.value,
                "pose_convention": pose_convention,
                "provenance": packet.provenance,
            }
        )

    if sidecar_arrays:
        sidecar_path = output_dir / f"{asset.asset_id}_measured_frame_ray_packets.npz"
        np.savez_compressed(sidecar_path, **sidecar_arrays)
        sidecar_artifact = str(sidecar_path)
    else:
        sidecar_artifact = None

    return {
        "packet_creation_status": "created" if packets else "not_created_no_valid_selected_packets",
        "packet_count": len(packets),
        "selected_frame_ids": tuple(packet.frame_id for packet in packets),
        "sidecar_artifact": sidecar_artifact,
        "sidecar_format": "npz_compressed" if sidecar_artifact else None,
        "json_payload_policy": "metadata_only_arrays_in_sidecar",
        "packet_summaries": tuple(packet_summaries),
        "skipped_packets": tuple(skipped_packets),
    }


def _merged_reference_metadata(
    asset: VideoAsset,
    asset_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    merged: dict[str, Any] = {}
    sources: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for sidecar_path in _sidecar_metadata_paths(asset.metadata, asset_root, repo_root):
        if not sidecar_path.exists():
            continue
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(sidecar_path), "error": str(exc)})
            continue
        if not isinstance(data, Mapping):
            errors.append({"path": str(sidecar_path), "error": "metadata_sidecar_must_be_json_object"})
            continue
        sidecar_config = _reference_config_from_mapping(data)
        _deep_update(merged, sidecar_config)
        sources.append({"kind": "sidecar", "path": str(sidecar_path)})
        break

    manifest_config = _reference_config_from_mapping(asset.metadata)
    if manifest_config:
        _deep_update(merged, manifest_config)
        sources.append({"kind": "manifest", "path": "config/canonical_assets.json", "field": f"tracks[{asset.asset_id}].metadata"})

    return merged, tuple(sources), tuple(errors)


def _sidecar_metadata_paths(metadata: Mapping[str, Any], asset_root: Path, repo_root: Path) -> tuple[Path, ...]:
    config = _reference_config_from_mapping(metadata)
    candidates: list[Path] = []
    metadata_file = config.get("metadata_file")
    if isinstance(metadata_file, str) and metadata_file.strip():
        path = Path(metadata_file)
        candidates.append(_resolve_path(path, repo_root) if not path.is_absolute() else path)
    candidates.extend(asset_root / name for name in SIDECAR_METADATA_CANDIDATES)
    return tuple(candidates)


def _reference_config_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("m2_reference", "reference_evidence", "reference_metric"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            _deep_update(result, dict(nested))

    for key in (
        "format",
        "rgb_index_file",
        "depth_index_file",
        "groundtruth_file",
        "association_file",
        "metadata_file",
        "intrinsics",
        "depth_scale_to_meters",
        "source_depth_convention",
        "timestamp_tolerance_s",
        "pose_convention",
        "pose_translation_units",
    ):
        if key in value:
            result[key] = value[key]
    return result


def _discover_tum_file_set(asset_root: Path, config: Mapping[str, Any]) -> dict[str, Path | None]:
    rgb_index = _metadata_path(config, "rgb_index_file", asset_root) or asset_root / RGB_INDEX_FILE
    depth_index = _metadata_path(config, "depth_index_file", asset_root) or asset_root / DEPTH_INDEX_FILE
    groundtruth = _metadata_path(config, "groundtruth_file", asset_root) or asset_root / GROUNDTRUTH_FILE
    association = _metadata_path(config, "association_file", asset_root)
    if association is None:
        for name in ASSOCIATION_FILE_CANDIDATES:
            candidate = asset_root / name
            if candidate.exists():
                association = candidate
                break
    return {
        "rgb_index": rgb_index,
        "depth_index": depth_index,
        "groundtruth": groundtruth,
        "association": association,
    }


def _metadata_path(config: Mapping[str, Any], key: str, asset_root: Path) -> Path | None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else asset_root / path


def _parse_tum_file_index(index_path: Path | None, asset_root: Path) -> tuple[tuple[TimestampedFile, ...], tuple[dict[str, str], ...]]:
    if index_path is None or not index_path.exists():
        return (), ()
    entries: list[TimestampedFile] = []
    errors: list[dict[str, str]] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            errors.append({"line": str(line_number), "error": "expected_timestamp_and_relative_path"})
            continue
        try:
            timestamp_s = float(parts[0])
        except ValueError:
            errors.append({"line": str(line_number), "error": "timestamp_must_be_float"})
            continue
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            errors.append({"line": str(line_number), "error": "timestamp_must_be_non_negative_finite"})
            continue
        relative_path = parts[1]
        entries.append(TimestampedFile(timestamp_s, relative_path, _entry_path(asset_root, relative_path)))
    return tuple(entries), tuple(errors)


def _parse_tum_groundtruth(index_path: Path | None) -> tuple[tuple[PoseEntry, ...], tuple[dict[str, str], ...]]:
    if index_path is None or not index_path.exists():
        return (), ()
    entries: list[PoseEntry] = []
    errors: list[dict[str, str]] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            errors.append({"line": str(line_number), "error": "expected_timestamp_tx_ty_tz_qx_qy_qz_qw"})
            continue
        try:
            values = tuple(float(part) for part in parts[:8])
        except ValueError:
            errors.append({"line": str(line_number), "error": "trajectory_values_must_be_float"})
            continue
        if any(not math.isfinite(value) for value in values):
            errors.append({"line": str(line_number), "error": "trajectory_values_must_be_finite"})
            continue
        timestamp_s, tx, ty, tz, qx, qy, qz, qw = values
        if timestamp_s < 0:
            errors.append({"line": str(line_number), "error": "timestamp_must_be_non_negative"})
            continue
        entries.append(PoseEntry(timestamp_s, (tx, ty, tz), (qx, qy, qz, qw)))
    return tuple(entries), tuple(errors)


def _parse_association_file(index_path: Path | None) -> tuple[tuple[tuple[float, str, float, str], ...], tuple[dict[str, str], ...]]:
    if index_path is None or not index_path.exists():
        return (), ()
    rows: list[tuple[float, str, float, str]] = []
    errors: list[dict[str, str]] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            errors.append({"line": str(line_number), "error": "expected_rgb_timestamp_rgb_path_depth_timestamp_depth_path"})
            continue
        try:
            rgb_timestamp_s = float(parts[0])
            depth_timestamp_s = float(parts[2])
        except ValueError:
            errors.append({"line": str(line_number), "error": "association_timestamps_must_be_float"})
            continue
        if min(rgb_timestamp_s, depth_timestamp_s) < 0 or not all(math.isfinite(value) for value in (rgb_timestamp_s, depth_timestamp_s)):
            errors.append({"line": str(line_number), "error": "association_timestamps_must_be_non_negative_finite"})
            continue
        rows.append((rgb_timestamp_s, parts[1], depth_timestamp_s, parts[3]))
    return tuple(rows), tuple(errors)


def _associate_rgb_depth(
    rgb_entries: Sequence[TimestampedFile],
    depth_entries: Sequence[TimestampedFile],
    association_rows: Sequence[tuple[float, str, float, str]],
    timestamp_tolerance_s: float | None,
) -> tuple[RgbDepthPair, ...]:
    if not rgb_entries or not depth_entries:
        return ()
    rgb_frame_ids = _frame_ids_by_rgb(rgb_entries)
    pairs: list[RgbDepthPair] = []
    if association_rows:
        rgb_by_path = {_normalize_rel(entry.relative_path): entry for entry in rgb_entries}
        depth_by_path = {_normalize_rel(entry.relative_path): entry for entry in depth_entries}
        for rgb_timestamp_s, rgb_path, depth_timestamp_s, depth_path in association_rows:
            rgb = rgb_by_path.get(_normalize_rel(rgb_path)) or _nearest_timestamp(rgb_entries, rgb_timestamp_s, 1e-9)
            depth = depth_by_path.get(_normalize_rel(depth_path)) or _nearest_timestamp(depth_entries, depth_timestamp_s, 1e-9)
            if rgb is not None and depth is not None:
                pairs.append(RgbDepthPair(rgb_frame_ids[rgb.relative_path], rgb, depth))
        return tuple(_dedupe_pairs(pairs))

    if timestamp_tolerance_s is None:
        return ()
    for rgb in rgb_entries:
        depth = _nearest_timestamp(depth_entries, rgb.timestamp_s, timestamp_tolerance_s)
        if depth is not None:
            pairs.append(RgbDepthPair(rgb_frame_ids[rgb.relative_path], rgb, depth))
    return tuple(pairs)


def _associate_rgb_pose(
    rgb_entries: Sequence[TimestampedFile],
    pose_entries: Sequence[PoseEntry],
    timestamp_tolerance_s: float | None,
) -> tuple[RgbPosePair, ...]:
    if not rgb_entries or not pose_entries or timestamp_tolerance_s is None:
        return ()
    rgb_frame_ids = _frame_ids_by_rgb(rgb_entries)
    pairs = []
    for rgb in rgb_entries:
        pose = _nearest_timestamp(pose_entries, rgb.timestamp_s, timestamp_tolerance_s)
        if pose is not None:
            pairs.append(RgbPosePair(rgb_frame_ids[rgb.relative_path], rgb, pose))
    return tuple(pairs)


def _complete_associations(
    rgb_depth_pairs: Sequence[RgbDepthPair],
    pose_entries: Sequence[PoseEntry],
    timestamp_tolerance_s: float | None,
) -> tuple[ReferenceAssociation, ...]:
    if not rgb_depth_pairs or not pose_entries or timestamp_tolerance_s is None:
        return ()
    associations: list[ReferenceAssociation] = []
    for pair in rgb_depth_pairs:
        pose = _nearest_timestamp(pose_entries, pair.rgb.timestamp_s, timestamp_tolerance_s)
        if pose is not None:
            associations.append(ReferenceAssociation(pair.frame_id, pair.rgb, pair.depth, pose))
    return tuple(associations)


def _select_associations(
    associations: Sequence[ReferenceAssociation],
    asset_id: str,
    repo_root: Path,
    m1_report_dir: str | Path,
    *,
    max_packets: int,
) -> tuple[ReferenceAssociation, ...]:
    if not associations:
        return ()
    keyframe_ids = _load_m1_keyframe_ids(asset_id, repo_root, m1_report_dir)
    by_frame_id = {association.frame_id: association for association in associations}
    if keyframe_ids:
        selected = [by_frame_id[frame_id] for frame_id in keyframe_ids if frame_id in by_frame_id]
        if selected:
            return tuple(selected[:max_packets])
    return tuple(associations[index] for index in _evenly_spaced_ids(len(associations), max_packets))


def _select_pairs(pairs: Sequence[_T], max_count: int) -> tuple[_T, ...]:
    if not pairs:
        return ()
    return tuple(pairs[index] for index in _evenly_spaced_ids(len(pairs), max_count))


def _load_m1_keyframe_ids(asset_id: str, repo_root: Path, m1_report_dir: str | Path) -> tuple[int, ...]:
    report_dir = _resolve_path(Path(m1_report_dir), repo_root)
    report_path = report_dir / f"{asset_id}_keyframes.json"
    if not report_path.exists():
        return ()
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    frame_ids = data.get("selected_frame_ids")
    if not isinstance(frame_ids, list):
        return ()
    result = []
    for frame_id in frame_ids:
        if isinstance(frame_id, int) and frame_id >= 0:
            result.append(frame_id)
    return tuple(result)


def _extract_intrinsics(
    config: Mapping[str, Any],
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> PinholeCameraModel | None:
    intrinsics = config.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        missing_fields.extend(f"m2_reference.intrinsics.{field}" for field in REQUIRED_INTRINSIC_FIELDS)
        return None

    parsed: dict[str, float | int] = {}
    for field in REQUIRED_INTRINSIC_FIELDS:
        if field not in intrinsics:
            missing_fields.append(f"m2_reference.intrinsics.{field}")
            continue
        value = intrinsics[field]
        try:
            if field in {"width_px", "height_px"}:
                parsed[field] = _parse_positive_int(value)
            else:
                parsed[field] = float(value)
        except (TypeError, ValueError):
            invalid_fields.append({"field": f"m2_reference.intrinsics.{field}", "error": "must_be_positive_integer" if field in {"width_px", "height_px"} else "must_be_numeric"})

    if len(parsed) != len(REQUIRED_INTRINSIC_FIELDS):
        return None
    try:
        return PinholeCameraModel(
            fx=float(parsed["fx"]),
            fy=float(parsed["fy"]),
            cx=float(parsed["cx"]),
            cy=float(parsed["cy"]),
            width_px=int(parsed["width_px"]),
            height_px=int(parsed["height_px"]),
        )
    except ContractValidationError as exc:
        invalid_fields.append({"field": "m2_reference.intrinsics", "error": str(exc)})
        return None


def _extract_positive_float(
    config: Mapping[str, Any],
    key: str,
    field_name: str,
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> float | None:
    if key not in config:
        missing_fields.append(field_name)
        return None
    try:
        value = float(config[key])
    except (TypeError, ValueError):
        invalid_fields.append({"field": field_name, "error": "must_be_positive_finite_number"})
        return None
    if not math.isfinite(value) or value <= 0:
        invalid_fields.append({"field": field_name, "error": "must_be_positive_finite_number"})
        return None
    return value


def _extract_non_negative_float(
    config: Mapping[str, Any],
    key: str,
    field_name: str,
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> float | None:
    if key not in config:
        missing_fields.append(field_name)
        return None
    try:
        value = float(config[key])
    except (TypeError, ValueError):
        invalid_fields.append({"field": field_name, "error": "must_be_non_negative_finite_number"})
        return None
    if not math.isfinite(value) or value < 0:
        invalid_fields.append({"field": field_name, "error": "must_be_non_negative_finite_number"})
        return None
    return value


def _extract_depth_convention(
    config: Mapping[str, Any],
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> DepthConvention | None:
    if "source_depth_convention" not in config:
        missing_fields.append("m2_reference.source_depth_convention")
        return None
    try:
        convention = DepthConvention(str(config["source_depth_convention"]))
    except ValueError:
        invalid_fields.append({"field": "m2_reference.source_depth_convention", "error": "must_be_radial_range_or_optical_z_for_m2"})
        return None
    if convention not in SUPPORTED_DEPTH_CONVENTIONS:
        invalid_fields.append({"field": "m2_reference.source_depth_convention", "error": "m2_supports_only_radial_range_or_optical_z"})
        return None
    return convention


def _extract_pose_convention(
    config: Mapping[str, Any],
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> str | None:
    value = config.get("pose_convention")
    if not isinstance(value, str) or not value.strip():
        missing_fields.append("m2_reference.pose_convention")
        return None
    if value not in SUPPORTED_POSE_CONVENTIONS:
        invalid_fields.append({"field": "m2_reference.pose_convention", "error": "must_be_T_world_camera_or_T_camera_world"})
        return None
    return value


def _extract_pose_translation_units(
    config: Mapping[str, Any],
    missing_fields: list[str],
    invalid_fields: list[dict[str, str]],
) -> str | None:
    value = config.get("pose_translation_units")
    if not isinstance(value, str) or not value.strip():
        missing_fields.append("m2_reference.pose_translation_units")
        return None
    if value != "meters":
        invalid_fields.append({"field": "m2_reference.pose_translation_units", "error": "m2_accepts_only_meters"})
        return None
    return value


def _blocking_reasons(
    *,
    rgb_entries: Sequence[TimestampedFile],
    depth_entries: Sequence[TimestampedFile],
    pose_entries: Sequence[PoseEntry],
    rgb_parse_errors: Sequence[Mapping[str, str]],
    depth_parse_errors: Sequence[Mapping[str, str]],
    pose_parse_errors: Sequence[Mapping[str, str]],
    metadata_errors: Sequence[Mapping[str, str]],
    missing_fields: Sequence[str],
    invalid_fields: Sequence[Mapping[str, str]],
) -> list[str]:
    reasons: list[str] = []
    if not rgb_entries:
        reasons.append("missing_rgb_timestamp_index")
    if not depth_entries:
        reasons.append("missing_measured_depth")
    if not pose_entries:
        reasons.append("missing_measured_pose")
    if rgb_parse_errors:
        reasons.append("rgb_timestamp_index_parse_errors")
    if depth_parse_errors:
        reasons.append("depth_timestamp_index_parse_errors")
    if pose_parse_errors:
        reasons.append("groundtruth_trajectory_parse_errors")
    if metadata_errors:
        reasons.append("metadata_sidecar_parse_errors")
    if missing_fields:
        reasons.append("missing_scale_critical_metadata")
    if invalid_fields:
        reasons.append("invalid_scale_critical_metadata")
    return reasons


def _pose_to_T_world_camera(
    translation_m: Sequence[float],
    quaternion_xyzw: Sequence[float],
    pose_convention: str,
) -> tuple[tuple[float, float, float, float], ...]:
    rotation = _quaternion_xyzw_to_rotation(quaternion_xyzw)
    tx, ty, tz = (float(value) for value in translation_m)
    if pose_convention == "T_world_camera":
        rows = (
            (rotation[0][0], rotation[0][1], rotation[0][2], tx),
            (rotation[1][0], rotation[1][1], rotation[1][2], ty),
            (rotation[2][0], rotation[2][1], rotation[2][2], tz),
            (0.0, 0.0, 0.0, 1.0),
        )
        return rows
    if pose_convention == "T_camera_world":
        rt = _transpose3(rotation)
        t_world_camera = (
            -(rt[0][0] * tx + rt[0][1] * ty + rt[0][2] * tz),
            -(rt[1][0] * tx + rt[1][1] * ty + rt[1][2] * tz),
            -(rt[2][0] * tx + rt[2][1] * ty + rt[2][2] * tz),
        )
        return (
            (rt[0][0], rt[0][1], rt[0][2], t_world_camera[0]),
            (rt[1][0], rt[1][1], rt[1][2], t_world_camera[1]),
            (rt[2][0], rt[2][1], rt[2][2], t_world_camera[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    raise ContractValidationError("pose_convention must be T_world_camera or T_camera_world")


def _quaternion_xyzw_to_rotation(quaternion_xyzw: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    if len(quaternion_xyzw) != 4:
        raise ContractValidationError("quaternion_xyzw must contain four values")
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0:
        raise ContractValidationError("quaternion_xyzw must have positive finite norm")
    qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def _unit_rays_for_pixels(camera: PinholeCameraModel, sampled_uv: Any, np: Any) -> Any:
    u = sampled_uv[:, 0]
    v = sampled_uv[:, 1]
    x = (u - float(camera.cx)) / float(camera.fx)
    y = (v - float(camera.cy)) / float(camera.fy)
    z = np.ones_like(x)
    rays = np.stack((x, y, z), axis=1)
    norms = np.linalg.norm(rays, axis=1)
    if bool(np.any(norms <= 0.0)):
        raise ContractValidationError("intrinsics produced invalid camera rays")
    return rays / norms[:, None]


def _source_depth_to_radial(source_depth_m: Any, ray_z: Any, convention: DepthConvention, np: Any) -> Any:
    if convention is DepthConvention.RADIAL_RANGE:
        return source_depth_m
    if convention is DepthConvention.OPTICAL_Z:
        if bool(np.any(ray_z <= 1e-12)):
            raise ContractValidationError("optical-z depth conversion requires positive ray z components")
        return source_depth_m / ray_z
    raise ContractValidationError("M2 measured packets support only optical_z or radial_range depth")


def _sample_valid_pixels(valid_mask: Any, max_rays_per_packet: int, np: Any) -> Any:
    if max_rays_per_packet <= 0:
        raise ContractValidationError("max_rays_per_packet must be positive")
    coords = np.argwhere(valid_mask)
    if coords.shape[0] <= max_rays_per_packet:
        return coords
    indices = np.linspace(0, coords.shape[0] - 1, num=max_rays_per_packet)
    return coords[np.unique(np.rint(indices).astype(np.int64))]


def _nearest_timestamp(entries: Sequence[_T], timestamp_s: float, tolerance_s: float | None) -> _T | None:
    if not entries:
        return None
    best = min(entries, key=lambda entry: abs(float(getattr(entry, "timestamp_s")) - timestamp_s))
    delta = abs(float(getattr(best, "timestamp_s")) - timestamp_s)
    if tolerance_s is not None and delta > tolerance_s:
        return None
    return best


def _frame_ids_by_rgb(rgb_entries: Sequence[TimestampedFile]) -> dict[str, int]:
    return {entry.relative_path: index for index, entry in enumerate(rgb_entries)}


def _dedupe_pairs(pairs: Sequence[RgbDepthPair]) -> tuple[RgbDepthPair, ...]:
    seen = set()
    result = []
    for pair in pairs:
        key = pair.frame_id
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
    return tuple(result)


def _entry_path(asset_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else asset_root / path


def _normalize_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _file_summary(path: Path | None, entries: Sequence[Any], parse_errors: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else None,
        "present": bool(path is not None and path.exists()),
        "entry_count": len(entries),
        "parse_error_count": len(parse_errors),
        "parse_errors": tuple(parse_errors[:8]),
    }


def _pose_file_summary(path: Path | None, entries: Sequence[PoseEntry], parse_errors: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return _file_summary(path, entries, parse_errors)


def _association_file_summary(
    path: Path | None,
    rows: Sequence[tuple[float, str, float, str]],
    parse_errors: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return _file_summary(path, rows, parse_errors)


def _missing_entry_files(entries: Sequence[TimestampedFile]) -> tuple[str, ...]:
    return tuple(str(entry.path) for entry in entries if not entry.path.exists())


def _base_report(asset: VideoAsset, repo_root: Path, resolved: Path | None) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "track_type": asset.track_type.value,
        "source_uri_or_path": asset.source_uri_or_path,
        "resolved_path": str(resolved) if resolved is not None else None,
        "repo_root": str(repo_root),
        "module": "M2 - Metric Reference Adapter",
    }


def _phone_no_measured_evidence_report(asset: VideoAsset, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_asset_source(asset, repo_root)
    return {
        **_base_report(asset, repo_root, resolved),
        "status": "no_measured_evidence_supplied",
        "reference_format": None,
        "blocking_reasons": ("phone_room_has_no_measured_metric_evidence_supplied",),
        "evidence_summary": {
            "resolved_path": str(resolved) if resolved is not None else None,
            "phone_room_metric_scale": "unanchored",
            "measured_depth": False,
            "measured_pose": False,
            "manual_distance": False,
        },
        "missing_fields": (),
        "scale_evidence": (),
        "scale_posterior": None,
        "scale_posterior_status": "not_produced_in_m2",
        "accepted_for_metric_training": False,
        "metric_training_status": "not_accepted_in_m2",
        "frame_ray_packets": _empty_packet_report("not_created_phone_room_has_no_measured_evidence"),
    }


def _non_reference_report(asset: VideoAsset, repo_root: Path) -> dict[str, Any]:
    resolved = _resolve_asset_source(asset, repo_root)
    return {
        **_base_report(asset, repo_root, resolved),
        "status": "not_reference_metric_track",
        "reference_format": None,
        "blocking_reasons": ("m2_only_loads_reference_metric",),
        "evidence_summary": {},
        "missing_fields": (),
        "scale_evidence": (),
        "scale_posterior": None,
        "scale_posterior_status": "not_produced_in_m2",
        "accepted_for_metric_training": False,
        "metric_training_status": "not_accepted_in_m2",
        "frame_ray_packets": _empty_packet_report("not_created_not_reference_metric"),
    }


def _empty_packet_report(status: str) -> dict[str, Any]:
    return {
        "packet_creation_status": status,
        "packet_count": 0,
        "selected_frame_ids": (),
        "sidecar_artifact": None,
        "sidecar_format": None,
        "json_payload_policy": "metadata_only_arrays_in_sidecar_when_created",
        "packet_summaries": (),
        "skipped_packets": (),
    }


def _depth_metadata_complete(
    intrinsics: PinholeCameraModel | None,
    depth_scale_to_meters: float | None,
    source_depth_convention: DepthConvention | None,
) -> bool:
    return (
        intrinsics is not None
        and depth_scale_to_meters is not None
        and source_depth_convention in SUPPORTED_DEPTH_CONVENTIONS
    )


def _pose_metadata_complete(pose_convention: str | None, pose_translation_units: str | None) -> bool:
    return pose_convention in SUPPORTED_POSE_CONVENTIONS and pose_translation_units == "meters"


def _field_source(config: Mapping[str, Any], key: str) -> str:
    return "sidecar_or_manifest" if key in config else "missing"


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return (
        (float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0])),
        (float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1])),
        (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])),
    )


def _pixel_uv(pixel_uv: Sequence[float]) -> tuple[float, float]:
    if not isinstance(pixel_uv, Sequence) or len(pixel_uv) != 2:
        raise ContractValidationError("pixel_uv must contain u and v")
    u, v = (float(pixel_uv[0]), float(pixel_uv[1]))
    if not all(math.isfinite(value) for value in (u, v)):
        raise ContractValidationError("pixel_uv must contain finite values")
    return u, v


def _camera_xyz(X_camera: object) -> tuple[float, float, float]:
    if not isinstance(X_camera, Sequence) or len(X_camera) != 3:
        raise ContractValidationError("X_camera must contain x, y, z")
    x, y, z = (float(X_camera[0]), float(X_camera[1]), float(X_camera[2]))
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ContractValidationError("X_camera must contain finite values")
    return x, y, z


def _require_finite_number(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractValidationError(f"{field_name} must be finite")


def _require_positive_number(value: float, field_name: str) -> None:
    _require_finite_number(value, field_name)
    if float(value) <= 0:
        raise ContractValidationError(f"{field_name} must be positive")


def _parse_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool_is_not_int")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        raise ValueError("must_be_positive_integer")
    if result <= 0:
        raise ValueError("must_be_positive_integer")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--m1-report-dir", default=str(DEFAULT_M1_REPORT_DIR))
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--max-rays-per-packet", type=int, default=2048)
    parser.add_argument("--force", action="store_true", help="ignore the input cache")
    args = parser.parse_args(argv)

    result = run_m2(
        args.manifest,
        args.output_dir,
        m1_report_dir=args.m1_report_dir,
        max_packets=args.max_packets,
        max_rays_per_packet=args.max_rays_per_packet,
        force=args.force,
    )
    summary = {
        "registry_report": result["registry_report"],
        "measured_reference_reports": result["measured_reference_reports"],
        "cache_status": result.get("cache_status", "unknown"),
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
