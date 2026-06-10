"""Injected-corruption detection-limit calibration (the "measured authority"
principle of the GT-free acceptance cascade, ARCHITECTURE.md Module 11).

Inject KNOWN corruptions into a FINISHED candidate reconstruction and re-score
the GT-free signal suite. A signal that cannot detect an injected corruption has
no authority on that scene; its clean reading is `no_authority`, never evidence
of correctness. A signal that responds sharply while reading clean on the
uncorrupted reconstruction certifies a per-scene detection limit: "corruptions
of the tested families at or above this size would have been visible; none are."

Honesty rules baked into this module:
- Corruptions are applied to the finished reconstruction and NO refinement
  repair pass is granted between injection and scoring. Several families are
  chosen deliberately OUTSIDE the refiner's parametric span (per-frame SE(3) +
  per-frame log-depth affine) so the response curve measures signal sensitivity
  to error shapes a repair pass could not absorb either:
    * regional_depth_bias  -- smooth spatially-varying depth bias (a per-frame
      affine cannot represent it)
    * ray_field_focal_bias -- ray-direction (intrinsics-class) error
- Magnitudes are SCENE-RELATIVE (fractions of the trajectory span, log-depth
  units, degrees). Absolute metric units would fabricate a ruler the gauge
  freedom does not grant (a "10 cm" injection on a reconstruction whose scale is
  3.5x off is not 10 cm).
- `global_tilt_control` is a NEGATIVE control: internal-consistency signals must
  NOT respond to a rigid world-frame tilt (multiview geometry is gauge-blind to
  it); only the gravity/floor stage may. The certificate records, per scene,
  that tilt coverage rests on the gravity gate alone.
- Every injected packet carries the full injection record in its provenance and
  a ``:injected`` source suffix -- an injected packet can never silently
  masquerade as a real reconstruction.
- The certificate covers ONLY the tested corruption families. Errors shaped
  unlike every tested family are explicitly out of scope and the report says so.

``numpy`` is imported lazily inside functions (repo convention).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from .contracts import ContractValidationError, DepthConvention, FrameRayPacket

CORRUPTION_FAMILIES = (
    "pose_drift_translation",   # cumulative translation ramp, fraction of trajectory span
    "pose_drift_rotation",      # cumulative rotation ramp, degrees end-to-end
    "scale_drift_ramp",         # per-frame depth scale ramp, end-to-end factor
    "regional_depth_bias",      # smooth angular-region log-depth bias (OUT of refine span)
    "ray_field_focal_bias",     # ray-direction focal-class error (OUT of refine span)
    "global_tilt_control",      # rigid world tilt -- NEGATIVE control for internal signals
)

# Magnitude grids per family (scene-relative / dimensionless / degrees).
DEFAULT_MAGNITUDES: dict[str, tuple[float, ...]] = {
    "pose_drift_translation": (0.05, 0.10, 0.20, 0.40),
    "pose_drift_rotation": (1.0, 2.0, 5.0, 10.0),
    "scale_drift_ramp": (1.2, 1.5, 2.0),
    "regional_depth_bias": (0.10, 0.20, 0.40),
    "ray_field_focal_bias": (0.05, 0.10),
    "global_tilt_control": (15.0, 30.0, 37.0),  # 37 deg ~= phone_room's measured tilt
}

MAGNITUDE_UNITS: dict[str, str] = {
    "pose_drift_translation": "fraction_of_trajectory_span_end_to_end",
    "pose_drift_rotation": "degrees_end_to_end",
    "scale_drift_ramp": "depth_scale_factor_end_to_end",
    "regional_depth_bias": "peak_abs_log_depth_bias",
    "ray_field_focal_bias": "fractional_tangent_field_scaling",
    "global_tilt_control": "degrees_rigid_world_tilt",
}

IN_REFINE_SPAN: dict[str, bool] = {
    # Whether a refine repair pass COULD parametrically absorb this family.
    # (No repair pass is granted either way; this is recorded for honesty.)
    "pose_drift_translation": True,
    "pose_drift_rotation": True,
    "scale_drift_ramp": True,   # per-frame beta_i can express it
    "regional_depth_bias": False,
    "ray_field_focal_bias": False,
    "global_tilt_control": True,  # a rigid gauge move; also gauge-invisible
}


def trajectory_span_m(packets: Sequence[FrameRayPacket]) -> float:
    """Largest pairwise camera-center distance, in reconstruction units.

    This is the scene-relative ruler for translation corruption magnitudes.
    Named `_m` for unit bookkeeping, but the value is METRIC ONLY when the
    reconstruction itself is; under gauge freedom it is reconstruction units.
    """
    import numpy as np

    centers = np.asarray(
        [np.asarray(p.T_world_camera, dtype=np.float64).reshape(4, 4)[:3, 3] for p in packets]
    )
    if centers.shape[0] < 2:
        return 0.0
    diffs = centers[:, None, :] - centers[None, :, :]
    return float(np.sqrt((diffs ** 2).sum(axis=2)).max())


def _rng_for(asset_id: str, family: str, magnitude: float, seed: int):
    import numpy as np

    digest = hashlib.sha256(
        f"{asset_id}|{family}|{magnitude:.6f}|{seed}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def inject_corruption(
    packets: Sequence[FrameRayPacket],
    family: str,
    magnitude: float,
    seed: int,
) -> tuple[list[FrameRayPacket], dict[str, Any]]:
    """Return ``(corrupted_packets, injection_record)``.

    Deterministic for a given (asset, family, magnitude, seed). Raises on an
    unknown family rather than silently no-op'ing (a no-op injection would
    fabricate a detection-limit data point).
    """
    if family not in CORRUPTION_FAMILIES:
        raise ValueError(f"unknown corruption family: {family!r}")
    import numpy as np

    ordered = sorted(packets, key=lambda p: int(p.frame_id))
    n = len(ordered)
    if n < 2:
        raise ValueError("need at least two packets to inject a corruption ramp")

    asset_id = ordered[0].asset_id
    rng = _rng_for(asset_id, family, magnitude, seed)
    span = trajectory_span_m(ordered)

    record: dict[str, Any] = {
        "family": family,
        "magnitude": float(magnitude),
        "magnitude_units": MAGNITUDE_UNITS[family],
        "seed": int(seed),
        "trajectory_span_reconstruction_units": span,
        "in_refine_parametric_span": IN_REFINE_SPAN[family],
        "repair_pass_granted": False,
        "negative_control": family == "global_tilt_control",
    }

    # Family-level randomized direction parameters (seeded, recorded).
    if family in ("pose_drift_translation", "pose_drift_rotation", "global_tilt_control"):
        axis = rng.normal(size=3)
        if family == "global_tilt_control":
            # Tilt about a WORLD-horizontal axis (perpendicular to world z).
            axis[2] = 0.0
            if np.linalg.norm(axis) < 1e-9:
                axis = np.array([1.0, 0.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        record["direction_axis"] = [float(v) for v in axis]
    if family == "regional_depth_bias":
        # Region center/width in angular (tangent) coordinates: a smooth bump
        # fixed in the image field across frames -- the shape of a backbone
        # failing on a consistent image region; NOT representable by per-frame
        # affine log-depth corrections.
        center = rng.uniform(-0.35, 0.35, size=2)
        sigma = float(rng.uniform(0.12, 0.25))
        sign = float(rng.choice([-1.0, 1.0]))
        record["region_center_tangent"] = [float(center[0]), float(center[1])]
        record["region_sigma_tangent"] = sigma
        record["bias_sign"] = sign
    if family == "ray_field_focal_bias":
        sign = float(rng.choice([-1.0, 1.0]))
        record["bias_sign"] = sign

    corrupted: list[FrameRayPacket] = []
    for i, packet in enumerate(ordered):
        ramp = i / (n - 1)  # 0 at the first frame, 1 at the last
        T0 = np.asarray(packet.T_world_camera, dtype=np.float64).reshape(4, 4)
        rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape(-1, 3)
        depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape(-1)

        T_new = T0.copy()
        rays_new = rays
        depth_new = depth

        if family == "pose_drift_translation":
            axis = np.asarray(record["direction_axis"])
            T_new[:3, 3] = T0[:3, 3] + axis * (magnitude * span * ramp)
        elif family == "pose_drift_rotation":
            axis = np.asarray(record["direction_axis"])
            angle = np.deg2rad(magnitude * ramp)
            R = _axis_angle_rotation(axis, angle, np)
            T_new[:3, :3] = R @ T0[:3, :3]
            T_new[:3, 3] = R @ T0[:3, 3]
        elif family == "global_tilt_control":
            axis = np.asarray(record["direction_axis"])
            angle = np.deg2rad(magnitude)  # rigid: same for every frame
            R = _axis_angle_rotation(axis, angle, np)
            T_new[:3, :3] = R @ T0[:3, :3]
            T_new[:3, 3] = R @ T0[:3, 3]
        elif family == "scale_drift_ramp":
            depth_new = depth * float(magnitude) ** ramp
        elif family == "regional_depth_bias":
            cx, cy = record["region_center_tangent"]
            sigma = record["region_sigma_tangent"]
            z = np.maximum(np.abs(rays[:, 2]), 1e-9)
            tx, ty = rays[:, 0] / z, rays[:, 1] / z
            bump = np.exp(-(((tx - cx) ** 2 + (ty - cy) ** 2) / (2.0 * sigma ** 2)))
            depth_new = depth * np.exp(record["bias_sign"] * magnitude * bump)
        elif family == "ray_field_focal_bias":
            scale = 1.0 + record["bias_sign"] * magnitude
            rays_new = rays.copy()
            rays_new[:, 0] *= scale
            rays_new[:, 1] *= scale

        new_packet = _rebuild_injected_packet(
            packet=packet, T_new=T_new, rays_new=rays_new,
            depth_new=depth_new, record=record, np=np,
        )
        corrupted.append(new_packet)
    return corrupted, record


def _axis_angle_rotation(axis: Any, angle: float, np: Any) -> Any:
    a = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rebuild_injected_packet(
    *,
    packet: FrameRayPacket,
    T_new: Any,
    rays_new: Any,
    depth_new: Any,
    record: dict[str, Any],
    np: Any,
) -> FrameRayPacket:
    conf = np.asarray(packet.confidence, dtype=np.float64).reshape(-1)
    T = np.asarray(T_new, dtype=np.float64).reshape(4, 4).copy()
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])

    keep = (
        np.isfinite(depth_new) & (depth_new > 0.0)
        & np.all(np.isfinite(rays_new), axis=1)
    )
    rays_k = np.asarray(rays_new, dtype=np.float64)[keep]
    depth_k = np.asarray(depth_new, dtype=np.float64)[keep]
    conf_k = np.clip(conf[keep], 0.0, 1.0)

    norms = np.linalg.norm(rays_k, axis=1)
    rays_unit = rays_k / np.maximum(norms[:, None], 1e-12)

    new_provenance = dict(packet.provenance)
    new_provenance["injected_corruption"] = dict(record)
    new_uncertainty = dict(packet.uncertainty)
    new_uncertainty["injected_corruption"] = (
        f"{record['family']}@{record['magnitude']}(seed={record['seed']})"
    )

    try:
        return FrameRayPacket(
            asset_id=packet.asset_id,
            frame_id=int(packet.frame_id),
            T_world_camera=[[float(v) for v in row] for row in T],
            rays_camera=rays_unit.reshape((rays_unit.shape[0], 1, 3)),
            radial_depth_m=depth_k.reshape((depth_k.shape[0], 1)),
            confidence=conf_k.reshape((conf_k.shape[0], 1)),
            camera_model=packet.camera_model,
            source=f"{packet.source}:injected",
            uncertainty=new_uncertainty,
            provenance=new_provenance,
            source_depth_convention=DepthConvention.RADIAL_RANGE,
            intrinsics=packet.intrinsics,
            rolling_shutter_model=packet.rolling_shutter_model,
            depth_residual_field=packet.depth_residual_field,
            camera_confidence=packet.camera_confidence,
        )
    except ContractValidationError as exc:  # pragma: no cover - corruption kept contract-valid
        raise ValueError(
            f"injected packet failed contract validation ({record['family']}"
            f"@{record['magnitude']}): {exc}"
        ) from exc


__all__ = [
    "CORRUPTION_FAMILIES",
    "DEFAULT_MAGNITUDES",
    "MAGNITUDE_UNITS",
    "IN_REFINE_SPAN",
    "inject_corruption",
    "trajectory_span_m",
]
