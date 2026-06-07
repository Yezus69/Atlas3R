"""M3 geometry adapter.

Normalizes external monocular-backbone artifacts (ViPE/DA3/MegaSaM-style or the
local ``tools/run_geometry_backbone.py`` output) into canonical
``FrameRayPacket`` objects, and rebuilds in-memory measured ``FrameRayPacket``
objects from the M2 sidecar/report so the teacher can consume both tracks
through one shape.

Hard rules honored here:
- No heavy ML deps at import time. ``numpy`` is imported lazily inside functions.
- Never fabricate packets. Missing artifacts -> explicit
  ``missing_external_artifact`` status with the exact next command.
- Reuse the M2 geometry helpers (``PinholeCameraModel``,
  ``_pose_to_T_world_camera`` is not needed here because artifact poses are
  already 4x4 ``T_world_camera``; ``_source_depth_to_radial`` and
  ``_unit_rays_for_pixels`` are reused for depth conversion and ray building).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    ContractValidationError,
    DepthConvention,
    FrameRayPacket,
)
from .m1 import _resolve_path
from .m2 import (
    PinholeCameraModel,
    _source_depth_to_radial,
    _unit_rays_for_pixels,
)

DEFAULT_ARTIFACTS_DIR = "external/teacher_artifacts"
DEFAULT_M2_DIR = "runs/m2"
DEFAULT_MAX_RAYS_PER_PACKET = 2048
# Robust clip band for monocular dense depth: dense models emit a handful of
# enormous outlier pixels (>1e6). They are not fabricated away -- they are
# excluded from the valid sample with an explicit recorded fraction.
_DEPTH_OUTLIER_HIGH_PERCENTILE = 99.0


def load_geometry_artifacts(
    asset_id: str,
    root: str | Path,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    max_rays_per_packet: int = DEFAULT_MAX_RAYS_PER_PACKET,
) -> tuple[list[FrameRayPacket], dict[str, Any]]:
    """Load an external geometry backbone artifact set for ``asset_id``.

    Returns ``(packets, adapter_report)``. When the artifact directory or its
    manifest is missing, ``packets`` is empty and the report carries
    ``status="missing_external_artifact"`` with the exact regeneration command.
    """
    root_path = Path(root)
    artifacts_root = _resolve_path(Path(artifacts_dir), root_path)
    asset_dir = artifacts_root / asset_id
    manifest_path = asset_dir / "backbone_manifest.json"

    base_report: dict[str, Any] = {
        "module": "M3 - Geometry Adapter",
        "asset_id": asset_id,
        "artifacts_dir": str(asset_dir),
        "source_kind": "external_monocular_backbone_artifact",
    }

    next_command = (
        f"python tools/run_geometry_backbone.py --asset-id {asset_id} "
        f"--artifacts-dir {artifacts_dir}"
    )

    if not asset_dir.is_dir() or not manifest_path.exists():
        missing = []
        if not asset_dir.is_dir():
            missing.append(str(asset_dir))
        if not manifest_path.exists():
            missing.append(str(manifest_path))
        return [], {
            **base_report,
            "status": "missing_external_artifact",
            "packet_count": 0,
            "missing_paths": tuple(missing),
            "next_command": next_command,
            "blockers": (f"missing_external_artifact:{asset_id}",),
        }

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        return [], {
            **base_report,
            "status": "missing_decoder_dependency",
            "packet_count": 0,
            "missing_dependency": exc.name,
            "next_command": next_command,
            "blockers": (f"missing_numpy_for_geometry_adapter:{exc.name}",),
        }

    manifest, manifest_error = _load_json_mapping(manifest_path)
    intrinsics_payload, intrinsics_error = _load_json_mapping(asset_dir / "intrinsics.json")
    depth_meta, depth_meta_error = _load_json_mapping(asset_dir / "depth_meta.json")
    poses_payload, poses_error = _load_json_mapping(asset_dir / "poses.json")

    parse_errors = [
        err
        for err in (manifest_error, intrinsics_error, depth_meta_error, poses_error)
        if err is not None
    ]
    if parse_errors:
        return [], {
            **base_report,
            "status": "corrupt_external_artifact",
            "packet_count": 0,
            "parse_errors": tuple(parse_errors),
            "next_command": next_command,
            "blockers": ("corrupt_external_artifact_json",),
        }

    camera, camera_error = _build_camera_model(intrinsics_payload)
    if camera is None:
        return [], {
            **base_report,
            "status": "invalid_intrinsics",
            "packet_count": 0,
            "intrinsics_error": camera_error,
            "next_command": next_command,
            "blockers": ("invalid_external_intrinsics",),
        }

    convention, convention_error = _depth_convention_from_meta(depth_meta)
    if convention is None:
        return [], {
            **base_report,
            "status": "invalid_depth_convention",
            "packet_count": 0,
            "depth_meta_error": convention_error,
            "next_command": next_command,
            "blockers": ("invalid_external_depth_convention",),
        }

    scale_to_meters = depth_meta.get("scale_to_meters_or_null")
    units = depth_meta.get("units")
    metric_evidence = bool(manifest.get("metric_evidence", False))
    backbone_name = str(manifest.get("backbone_name") or "external_backbone")
    method = str(manifest.get("method") or backbone_name)
    pose_convention = str(poses_payload.get("convention") or "T_world_camera")

    pose_by_frame = _poses_by_frame(poses_payload)

    depth_dir = asset_dir / "depth"
    confidence_dir = asset_dir / "confidence"

    packets: list[FrameRayPacket] = []
    packet_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    frame_ids = sorted(pose_by_frame.keys())
    for frame_id in frame_ids:
        depth_path = depth_dir / f"{frame_id}.npy"
        if not depth_path.exists():
            skipped.append({"frame_id": frame_id, "reason": "missing_depth_npy", "path": str(depth_path)})
            continue
        T_world_camera = pose_by_frame[frame_id]
        if pose_convention != "T_world_camera":
            skipped.append(
                {
                    "frame_id": frame_id,
                    "reason": "unsupported_pose_convention",
                    "pose_convention": pose_convention,
                }
            )
            continue

        try:
            raw_depth = np.load(depth_path)
        except (OSError, ValueError) as exc:
            skipped.append({"frame_id": frame_id, "reason": "depth_load_failed", "error": str(exc)})
            continue
        if raw_depth.ndim != 2:
            skipped.append({"frame_id": frame_id, "reason": "depth_must_be_2d", "shape": tuple(int(s) for s in raw_depth.shape)})
            continue
        height_px, width_px = raw_depth.shape
        if width_px != camera.width_px or height_px != camera.height_px:
            skipped.append(
                {
                    "frame_id": frame_id,
                    "reason": "depth_size_does_not_match_intrinsics",
                    "depth_shape": (int(height_px), int(width_px)),
                    "intrinsics_size": (camera.height_px, camera.width_px),
                }
            )
            continue

        depth = raw_depth.astype(np.float64)
        if scale_to_meters is not None:
            try:
                depth = depth * float(scale_to_meters)
            except (TypeError, ValueError):
                skipped.append({"frame_id": frame_id, "reason": "invalid_scale_to_meters"})
                continue

        finite_positive = np.isfinite(depth) & (depth > 0.0)
        positive_count = int(np.count_nonzero(finite_positive))
        if positive_count == 0:
            skipped.append({"frame_id": frame_id, "reason": "no_positive_depth_pixels"})
            continue

        # Robust outlier exclusion: keep depths at or below the high percentile
        # of positive depths. This removes the dense-model blow-up pixels
        # without inventing any values.
        positive_values = depth[finite_positive]
        high_cut = float(np.percentile(positive_values, _DEPTH_OUTLIER_HIGH_PERCENTILE))
        if not (high_cut > 0.0):
            high_cut = float(np.max(positive_values))
        valid_mask = finite_positive & (depth <= high_cut)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            valid_mask = finite_positive
            valid_count = positive_count

        confidence_values = _load_confidence(confidence_dir / f"{frame_id}.npy", valid_mask, np)

        sampled_yx = _sample_valid_pixels(valid_mask, max_rays_per_packet, np)
        sampled_uv = np.stack((sampled_yx[:, 1], sampled_yx[:, 0]), axis=1).astype(np.float64)
        try:
            rays = _unit_rays_for_pixels(camera, sampled_uv, np)
        except ContractValidationError as exc:
            skipped.append({"frame_id": frame_id, "reason": "ray_construction_failed", "error": str(exc)})
            continue

        source_depth = depth[sampled_yx[:, 0], sampled_yx[:, 1]]
        try:
            radial_depth = _source_depth_to_radial(source_depth, rays[:, 2], convention, np)
        except ContractValidationError as exc:
            skipped.append({"frame_id": frame_id, "reason": "depth_convention_conversion_failed", "error": str(exc)})
            continue
        radial_depth = np.asarray(radial_depth, dtype=np.float64)
        if not bool(np.all(np.isfinite(radial_depth) & (radial_depth > 0.0))):
            skipped.append({"frame_id": frame_id, "reason": "invalid_radial_depth_after_conversion"})
            continue

        if confidence_values is None:
            conf_sampled = np.full((sampled_yx.shape[0],), 0.5, dtype=np.float64)
        else:
            conf_sampled = np.clip(
                confidence_values[sampled_yx[:, 0], sampled_yx[:, 1]].astype(np.float64),
                0.0,
                1.0,
            )

        rays_packet = rays.reshape((rays.shape[0], 1, 3))
        depth_packet = radial_depth.reshape((radial_depth.shape[0], 1))
        confidence_packet = conf_sampled.reshape((conf_sampled.shape[0], 1))

        source = f"external_artifact:{backbone_name}"
        try:
            packet = FrameRayPacket(
                asset_id=asset_id,
                frame_id=int(frame_id),
                T_world_camera=[[float(v) for v in row] for row in T_world_camera],
                rays_camera=rays_packet,
                radial_depth_m=depth_packet,
                confidence=confidence_packet,
                camera_model=camera,
                source=source,
                uncertainty={
                    "depth": "monocular_dense_depth_no_posterior",
                    "scale": "free_global_gauge" if not metric_evidence else "external_metric_claimed",
                    "depth_units": str(units) if units is not None else "unknown_units",
                },
                provenance={
                    "backbone_name": backbone_name,
                    "method": method,
                    "metric_evidence": metric_evidence,
                    "depth_npy": str(depth_path),
                    "depth_scale_to_meters": scale_to_meters,
                    "source_depth_convention": convention.value,
                    "pose_convention": pose_convention,
                    "manifest": str(manifest_path),
                    "packet_sampling": "bounded_robust_valid_pixel_sample",
                    "outlier_high_percentile": _DEPTH_OUTLIER_HIGH_PERCENTILE,
                },
                source_depth_convention=convention,
                intrinsics=camera.to_metadata(),
                camera_confidence=_clamp01(float(manifest.get("camera_confidence", 0.5))),
            )
        except ContractValidationError as exc:
            skipped.append({"frame_id": frame_id, "reason": "frame_ray_packet_contract_rejected", "error": str(exc)})
            continue

        packets.append(packet)
        packet_summaries.append(
            {
                "frame_id": int(frame_id),
                "sampled_ray_count": int(rays.shape[0]),
                "source_valid_depth_pixel_count": valid_count,
                "source_positive_depth_pixel_count": positive_count,
                "source_depth_pixel_count": int(valid_mask.size),
                "outlier_high_cut_m": high_cut,
                "source_depth_convention": convention.value,
            }
        )

    # Soft (learned) metric scale evidence. A learned metric-depth backbone
    # (e.g. DA3METRIC) supplies a metric *prior*, not a measurement: it is
    # honest soft evidence (evidence_type=learned_metric_depth_prior,
    # measured=False) that can back at most a metric_pseudo_label -- never
    # measured_metric. We emit it only when the manifest explicitly declares a
    # learned metric prior AND the depth units are meters; otherwise the
    # reconstruction stays unanchored (non_metric_pseudo_label). Nothing is
    # fabricated: measured stays False and the source names the model.
    scale_evidence_objects: list[Any] = []
    learned_metric_prior = bool(manifest.get("learned_metric_depth_prior", False))
    if packets and learned_metric_prior and str(units) == "meters":
        from .contracts import ScaleEvidence, ScaleEvidenceType

        soft_conf = _clamp01(float(manifest.get("scale_evidence_confidence", 0.5)))
        try:
            scale_evidence_objects.append(
                ScaleEvidence(
                    evidence_id=f"{asset_id}_learned_metric_depth_prior",
                    evidence_type=ScaleEvidenceType.LEARNED_METRIC_DEPTH_PRIOR,
                    measured=False,
                    source=f"learned_metric_depth:{backbone_name}",
                    frame_ids=tuple(p.frame_id for p in packets),
                    confidence=soft_conf,
                    provenance={
                        "backbone_name": backbone_name,
                        "method": method,
                        "model_name": str(manifest.get("model_name") or backbone_name),
                        "units": "meters",
                        "evidence_class": "soft_learned_metric_prior_not_measured",
                    },
                )
            )
        except ContractValidationError:
            scale_evidence_objects = []

    status = "loaded" if packets else "no_valid_packets_from_artifact"
    report = {
        **base_report,
        "status": status,
        "backbone_name": backbone_name,
        "method": method,
        "metric_evidence": metric_evidence,
        "learned_metric_depth_prior": learned_metric_prior,
        "soft_scale_evidence_count": len(scale_evidence_objects),
        "depth_units": str(units) if units is not None else None,
        "depth_scale_to_meters": scale_to_meters,
        "source_depth_convention": convention.value,
        "pose_convention": pose_convention,
        "camera_model": camera.to_metadata(),
        "packet_count": len(packets),
        "frame_ids": tuple(p.frame_id for p in packets),
        "packet_summaries": tuple(packet_summaries),
        "skipped_frames": tuple(skipped),
        "next_command": next_command,
        "blockers": () if packets else ("external_artifact_present_but_no_valid_packets",),
        "_scale_evidence": scale_evidence_objects,
    }
    return packets, report


def load_measured_packets_from_m2(
    asset_id: str,
    root: str | Path,
    *,
    m2_dir: str | Path = DEFAULT_M2_DIR,
) -> tuple[list[FrameRayPacket], dict[str, Any]]:
    """Rebuild in-memory measured ``FrameRayPacket`` objects from M2 output.

    Loads the ``{asset_id}_measured_frame_ray_packets.npz`` sidecar (sparse
    ``[N,1,3]`` rays / ``[N,1]`` depth) and the JSON report metadata (camera
    intrinsics, source depth convention, provenance), then reconstructs
    contract-valid packets. The npz alone is lossy; the JSON supplies
    ``camera_model``/``uncertainty``/``provenance``/``source_depth_convention``.
    """
    root_path = Path(root)
    m2_root = _resolve_path(Path(m2_dir), root_path)
    npz_path = m2_root / f"{asset_id}_measured_frame_ray_packets.npz"
    report_path = m2_root / f"{asset_id}_measured_reference_report.json"

    base_report: dict[str, Any] = {
        "module": "M3 - Measured Packet Loader",
        "asset_id": asset_id,
        "npz_path": str(npz_path),
        "report_path": str(report_path),
        "source_kind": "m2_measured_reference",
    }
    next_command = "python -m atlas3r.m2"

    if not report_path.exists():
        return [], {
            **base_report,
            "status": "missing_m2_report",
            "packet_count": 0,
            "next_command": next_command,
            "blockers": (f"missing_m2_report:{asset_id}",),
        }

    report_payload, report_error = _load_json_mapping(report_path)
    if report_error is not None:
        return [], {
            **base_report,
            "status": "corrupt_m2_report",
            "packet_count": 0,
            "parse_error": report_error,
            "next_command": next_command,
            "blockers": ("corrupt_m2_report_json",),
        }

    frp_meta = report_payload.get("frame_ray_packets")
    if not isinstance(frp_meta, Mapping) or int(frp_meta.get("packet_count", 0) or 0) == 0:
        return [], {
            **base_report,
            "status": "no_measured_packets_in_m2",
            "packet_count": 0,
            "m2_status": report_payload.get("status"),
            "next_command": next_command,
            "blockers": ("m2_produced_no_measured_packets",),
        }

    if not npz_path.exists():
        return [], {
            **base_report,
            "status": "missing_measured_sidecar",
            "packet_count": 0,
            "next_command": next_command,
            "blockers": (f"missing_measured_sidecar:{asset_id}",),
        }

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        return [], {
            **base_report,
            "status": "missing_decoder_dependency",
            "packet_count": 0,
            "missing_dependency": exc.name,
            "next_command": next_command,
            "blockers": (f"missing_numpy_for_measured_loader:{exc.name}",),
        }

    summaries_by_frame: dict[int, Mapping[str, Any]] = {}
    for summary in frp_meta.get("packet_summaries", ()):  # type: ignore[union-attr]
        if isinstance(summary, Mapping) and "frame_id" in summary:
            summaries_by_frame[int(summary["frame_id"])] = summary

    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return [], {
            **base_report,
            "status": "corrupt_measured_sidecar",
            "packet_count": 0,
            "error": str(exc),
            "next_command": next_command,
            "blockers": ("corrupt_measured_sidecar_npz",),
        }

    frame_ids = sorted(
        int(key[len("frame_"):-len("_rays_camera")])
        for key in archive.files
        if key.startswith("frame_") and key.endswith("_rays_camera")
    )

    packets: list[FrameRayPacket] = []
    skipped: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        prefix = f"frame_{frame_id}"
        try:
            rays_packet = archive[f"{prefix}_rays_camera"].astype(np.float64)
            depth_packet = archive[f"{prefix}_radial_depth_m"].astype(np.float64)
            confidence_packet = archive[f"{prefix}_confidence"].astype(np.float64)
            T_world_camera = archive[f"{prefix}_T_world_camera"].astype(np.float64)
        except KeyError as exc:
            skipped.append({"frame_id": frame_id, "reason": "missing_sidecar_array", "key": str(exc)})
            continue

        summary = summaries_by_frame.get(frame_id, {})
        camera, camera_error = _build_camera_model(summary.get("camera_model", {}))
        if camera is None:
            skipped.append({"frame_id": frame_id, "reason": "missing_or_invalid_camera_model", "error": camera_error})
            continue
        convention_value = summary.get("source_depth_convention") or report_payload.get(
            "default_source_depth_convention"
        )
        try:
            convention = DepthConvention(str(convention_value))
        except ValueError:
            skipped.append({"frame_id": frame_id, "reason": "invalid_source_depth_convention", "value": convention_value})
            continue

        # Renormalize rays defensively: float32 round-trip can drift the unit
        # norm just past the 1e-5 tolerance. This is numeric cleanup of an
        # already-unit ray, not fabricated geometry.
        rays_flat = rays_packet.reshape((-1, 3))
        norms = np.linalg.norm(rays_flat, axis=1)
        if bool(np.any(norms <= 0.0)):
            skipped.append({"frame_id": frame_id, "reason": "degenerate_ray_norm"})
            continue
        rays_flat = rays_flat / norms[:, None]
        rays_packet = rays_flat.reshape(rays_packet.shape)
        confidence_packet = np.clip(confidence_packet, 0.0, 1.0)

        provenance = summary.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            provenance = {"source": "m2_measured_reference_sidecar", "frame_id": frame_id}

        try:
            packet = FrameRayPacket(
                asset_id=asset_id,
                frame_id=int(frame_id),
                T_world_camera=[[float(v) for v in row] for row in T_world_camera.tolist()],
                rays_camera=rays_packet,
                radial_depth_m=depth_packet,
                confidence=confidence_packet,
                camera_model=camera,
                source="measured_reference:tum_rgbd",
                uncertainty={
                    "depth": "measured_depth_pixels_scaled_to_meters_no_posterior",
                    "pose": "measured_groundtruth_pose_no_optimization",
                },
                provenance=dict(provenance),
                source_depth_convention=convention,
                intrinsics=camera.to_metadata(),
                camera_confidence=1.0,
            )
        except ContractValidationError as exc:
            skipped.append({"frame_id": frame_id, "reason": "frame_ray_packet_contract_rejected", "error": str(exc)})
            continue
        packets.append(packet)

    status = "loaded" if packets else "no_measured_packets_rebuilt"
    report = {
        **base_report,
        "status": status,
        "packet_count": len(packets),
        "frame_ids": tuple(p.frame_id for p in packets),
        "skipped_frames": tuple(skipped),
        "next_command": next_command,
        "blockers": () if packets else ("measured_sidecar_present_but_no_packets_rebuilt",),
    }
    return packets, report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_json_mapping(path: Path) -> tuple[dict[str, Any], dict[str, str] | None]:
    if not path.exists():
        return {}, {"path": str(path), "error": "missing_file"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"path": str(path), "error": str(exc)}
    if not isinstance(data, Mapping):
        return {}, {"path": str(path), "error": "json_must_be_object"}
    return dict(data), None


def _build_camera_model(payload: Mapping[str, Any]) -> tuple[PinholeCameraModel | None, str | None]:
    if not isinstance(payload, Mapping) or not payload:
        return None, "missing_intrinsics_payload"
    required = ("fx", "fy", "cx", "cy", "width_px", "height_px")
    for key in required:
        if key not in payload:
            return None, f"missing_intrinsics_field:{key}"
    try:
        camera = PinholeCameraModel(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            width_px=int(payload["width_px"]),
            height_px=int(payload["height_px"]),
        )
    except (TypeError, ValueError, ContractValidationError) as exc:
        return None, str(exc)
    return camera, None


def _depth_convention_from_meta(depth_meta: Mapping[str, Any]) -> tuple[DepthConvention | None, str | None]:
    value = depth_meta.get("depth_convention")
    if not isinstance(value, str) or not value.strip():
        return None, "missing_depth_convention"
    try:
        convention = DepthConvention(value)
    except ValueError:
        return None, f"unknown_depth_convention:{value}"
    if convention is DepthConvention.UNKNOWN:
        return None, "depth_convention_unknown_not_allowed"
    if convention in {DepthConvention.INVERSE_DEPTH, DepthConvention.DISPARITY}:
        # The adapter only converts radial_range / optical_z via the reused M2
        # helper; refuse to fabricate a conversion we have not implemented.
        return None, f"unsupported_depth_convention_for_adapter:{value}"
    return convention, None


def _poses_by_frame(poses_payload: Mapping[str, Any]) -> dict[int, Sequence[Sequence[float]]]:
    result: dict[int, Sequence[Sequence[float]]] = {}
    frames = poses_payload.get("frames")
    if not isinstance(frames, Sequence):
        return result
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        frame_id = frame.get("frame_id")
        matrix = frame.get("T_world_camera")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
            continue
        if not isinstance(matrix, Sequence) or len(matrix) != 4:
            continue
        result[int(frame_id)] = matrix
    return result


def _load_confidence(path: Path, valid_mask: Any, np: Any) -> Any | None:
    if not path.exists():
        return None
    try:
        conf = np.load(path)
    except (OSError, ValueError):
        return None
    if conf.shape != valid_mask.shape:
        return None
    return conf


def _sample_valid_pixels(valid_mask: Any, max_rays_per_packet: int, np: Any) -> Any:
    if max_rays_per_packet <= 0:
        raise ContractValidationError("max_rays_per_packet must be positive")
    coords = np.argwhere(valid_mask)
    if coords.shape[0] <= max_rays_per_packet:
        return coords
    indices = np.linspace(0, coords.shape[0] - 1, num=max_rays_per_packet)
    return coords[np.unique(np.rint(indices).astype(np.int64))]


def _clamp01(value: float) -> float:
    if not (value == value):  # NaN guard
        return 0.5
    return max(0.0, min(1.0, float(value)))
