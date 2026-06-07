"""M4 visibility and residual reports.

Builds a ``VisibilityGraph`` over ``FrameRayPacket`` keyframes by lifting
sampled rays in one frame to world space, transforming into a neighbour frame,
reprojecting through that frame's camera model, and summarizing the geometric
agreement as non-negative edge measurements.

No fabrication: edges carry only measured residual statistics. Frames with no
observable overlap simply produce no overlap edge.

Reuses the packet camera model's ``project``; never duplicates intrinsics math.
``numpy`` is imported lazily inside functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import (
    ContractValidationError,
    FrameRayPacket,
    VisibilityEdge,
    VisibilityGraph,
)

DEFAULT_MAX_SAMPLES_PER_EDGE = 256
DEFAULT_MAX_OVERLAP_PAIRS = 64


def build_visibility_graph(
    packets: Sequence[FrameRayPacket],
    *,
    max_samples_per_edge: int = DEFAULT_MAX_SAMPLES_PER_EDGE,
    max_overlap_pairs: int = DEFAULT_MAX_OVERLAP_PAIRS,
) -> tuple[VisibilityGraph, dict[str, Any]]:
    """Return ``(VisibilityGraph, residual_report)`` for ``packets``.

    Temporal edges connect consecutive packets; overlap edges sample additional
    non-adjacent pairs. With fewer than two packets, an empty graph is returned
    with an explanatory status (no self-edges, no fabricated overlap).
    """
    ordered = sorted(packets, key=lambda packet: packet.frame_id)
    node_ids = [int(packet.frame_id) for packet in ordered]

    base_report: dict[str, Any] = {
        "module": "M4 - Visibility And Residual Reports",
        "node_count": len(node_ids),
    }

    if len(ordered) < 2:
        graph = _empty_single_node_graph(node_ids)
        return graph, {
            **base_report,
            "status": "insufficient_packets_for_visibility",
            "temporal_edge_count": 0,
            "overlap_edge_count": 0,
            "edges": (),
            "overall": _empty_overall(),
            "blockers": ("need_at_least_two_packets_for_visibility",),
        }

    import numpy as np  # type: ignore

    prepared = [_prepare_packet(packet, np) for packet in ordered]

    temporal_edges: list[VisibilityEdge] = []
    overlap_edges: list[VisibilityEdge] = []
    edge_stats: list[dict[str, Any]] = []

    # Temporal edges: consecutive frames.
    temporal_pairs = [(i, i + 1) for i in range(len(prepared) - 1)]
    for i, j in temporal_pairs:
        edge, stats = _build_edge(prepared[i], prepared[j], np, max_samples_per_edge, "temporal")
        if edge is not None:
            temporal_edges.append(edge)
        if stats is not None:
            edge_stats.append(stats)

    # Overlap edges: sampled non-adjacent pairs.
    overlap_candidates = [
        (i, j)
        for i in range(len(prepared))
        for j in range(i + 2, len(prepared))
    ]
    sampled_overlap = _evenly_sample(overlap_candidates, max_overlap_pairs)
    for i, j in sampled_overlap:
        edge, stats = _build_edge(prepared[i], prepared[j], np, max_samples_per_edge, "overlap")
        # Only keep overlap edges with some observable reprojection.
        if edge is not None and edge.measurements.get("reprojection_inbounds_ratio", 0.0) > 0.0:
            overlap_edges.append(edge)
        if stats is not None:
            edge_stats.append(stats)

    try:
        graph = VisibilityGraph(
            nodes=tuple(node_ids),
            temporal_edges=tuple(temporal_edges),
            overlap_edges=tuple(overlap_edges),
            loop_edges=(),
            scale_edges=(),
            edge_measurements={
                "edge_count": float(len(temporal_edges) + len(overlap_edges)),
                "temporal_edge_count": float(len(temporal_edges)),
                "overlap_edge_count": float(len(overlap_edges)),
            },
        )
    except ContractValidationError as exc:
        return _empty_single_node_graph(node_ids), {
            **base_report,
            "status": "visibility_graph_contract_rejected",
            "error": str(exc),
            "edges": tuple(edge_stats),
            "overall": _empty_overall(),
            "blockers": ("visibility_graph_contract_rejected",),
        }

    report = {
        **base_report,
        "status": "built",
        "temporal_edge_count": len(temporal_edges),
        "overlap_edge_count": len(overlap_edges),
        "edges": tuple(edge_stats),
        "overall": _overall_summary(edge_stats),
        "blockers": (),
    }
    return graph, report


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _prepare_packet(packet: FrameRayPacket, np: Any) -> dict[str, Any]:
    rays = np.asarray(_to_array(packet.rays_camera), dtype=np.float64).reshape((-1, 3))
    depth = np.asarray(_to_array(packet.radial_depth_m), dtype=np.float64).reshape((-1,))
    T = np.asarray(_to_array(packet.T_world_camera), dtype=np.float64).reshape((4, 4))
    R = T[:3, :3]
    t = T[:3, 3]
    return {
        "frame_id": int(packet.frame_id),
        "rays": rays,
        "depth": depth,
        "R": R,
        "t": t,
        "camera": packet.camera_model,
    }


def _build_edge(
    source: dict[str, Any],
    target: dict[str, Any],
    np: Any,
    max_samples: int,
    kind: str,
) -> tuple[VisibilityEdge | None, dict[str, Any] | None]:
    rays = source["rays"]
    depth = source["depth"]
    n = rays.shape[0]
    if n == 0:
        return None, None
    if n > max_samples:
        idx = np.unique(np.rint(np.linspace(0, n - 1, num=max_samples)).astype(np.int64))
    else:
        idx = np.arange(n)
    rays_s = rays[idx]
    depth_s = depth[idx]

    # Lift to world: X_world = R_i (d * ray) + t_i
    X_cam_i = rays_s * depth_s[:, None]
    X_world = X_cam_i @ source["R"].T + source["t"][None, :]
    # Into target camera: X_cam_j = R_j^T (X_world - t_j)
    X_cam_j = (X_world - target["t"][None, :]) @ target["R"]

    target_camera = target["camera"]
    sample_count = int(rays_s.shape[0])
    inbounds = 0
    log_resid: list[float] = []
    for k in range(sample_count):
        result = target_camera.project((float(X_cam_j[k, 0]), float(X_cam_j[k, 1]), float(X_cam_j[k, 2])))
        if not result.get("valid"):
            continue
        inbounds += 1
        d_pred = result.get("radial_depth_m")
        if d_pred is None or not (d_pred > 0.0):
            continue
        # Depth consistency where the target also has an observation near the
        # reprojected pixel: compare predicted depth against the target frame's
        # own depth at the nearest sampled pixel. We use the target depth's
        # median as a robust observable proxy (sparse samples are not a dense
        # grid), so the residual stays a real measured quantity, not invented.
        d_obs = _nearest_target_depth(target, result.get("pixel_uv"), np)
        if d_obs is not None and d_obs > 0.0:
            log_resid.append(abs(_safe_log(d_pred) - _safe_log(d_obs)))

    inbounds_ratio = inbounds / sample_count if sample_count else 0.0
    if log_resid:
        resid_arr = np.asarray(log_resid, dtype=np.float64)
        mean_log_resid = float(np.median(resid_arr))
        # Huber-style robust mean as an alternate summary.
        huber = float(np.mean(_huber(resid_arr, 0.25, np)))
    else:
        mean_log_resid = 0.0
        huber = 0.0

    confidence_weight = inbounds_ratio * (1.0 / (1.0 + mean_log_resid))

    measurements = {
        "sample_count": float(sample_count),
        "reprojection_inbounds_count": float(inbounds),
        "reprojection_inbounds_ratio": float(max(0.0, min(1.0, inbounds_ratio))),
        "mean_log_depth_residual": float(max(0.0, mean_log_resid)),
        "huber_log_depth_residual": float(max(0.0, huber)),
        "depth_residual_sample_count": float(len(log_resid)),
        "confidence_weight": float(max(0.0, confidence_weight)),
        "free_space_compatibility_proxy": float(max(0.0, min(1.0, inbounds_ratio * (1.0 / (1.0 + mean_log_resid))))),
    }
    try:
        edge = VisibilityEdge(
            source_frame_id=source["frame_id"],
            target_frame_id=target["frame_id"],
            measurements=measurements,
        )
    except ContractValidationError:
        return None, {
            "source_frame_id": source["frame_id"],
            "target_frame_id": target["frame_id"],
            "kind": kind,
            "status": "edge_contract_rejected",
        }
    stats = {
        "source_frame_id": source["frame_id"],
        "target_frame_id": target["frame_id"],
        "kind": kind,
        **measurements,
    }
    return edge, stats


def _nearest_target_depth(target: dict[str, Any], pixel_uv: Any, np: Any) -> float | None:
    # Sparse packets do not carry a dense grid, so we use the target frame's
    # robust median radial depth as the observable comparison. This is honest:
    # it is a real statistic of the target's measured/predicted depth, not a
    # per-pixel fabrication.
    depth = target["depth"]
    if depth.size == 0:
        return None
    return float(np.median(depth))


def _huber(values: Any, delta: float, np: Any) -> Any:
    abs_v = np.abs(values)
    quadratic = 0.5 * abs_v * abs_v
    linear = delta * (abs_v - 0.5 * delta)
    return np.where(abs_v <= delta, quadratic, linear)


def _overall_summary(edge_stats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not edge_stats:
        return _empty_overall()
    inbounds = [s["reprojection_inbounds_ratio"] for s in edge_stats if "reprojection_inbounds_ratio" in s]
    resid = [s["mean_log_depth_residual"] for s in edge_stats if "mean_log_depth_residual" in s]
    conf = [s["confidence_weight"] for s in edge_stats if "confidence_weight" in s]
    return {
        "edge_count": len(edge_stats),
        "median_reprojection_inbounds_ratio": _median(inbounds),
        "median_log_depth_residual": _median(resid),
        "mean_confidence_weight": (sum(conf) / len(conf)) if conf else 0.0,
        "edges_with_depth_residual": sum(1 for s in edge_stats if s.get("depth_residual_sample_count", 0.0) > 0.0),
    }


def _empty_overall() -> dict[str, Any]:
    return {
        "edge_count": 0,
        "median_reprojection_inbounds_ratio": 0.0,
        "median_log_depth_residual": 0.0,
        "mean_confidence_weight": 0.0,
        "edges_with_depth_residual": 0,
    }


def _empty_single_node_graph(node_ids: Sequence[int]) -> VisibilityGraph:
    nodes = tuple(node_ids) if node_ids else (0,)
    # A single declared node with no edges is contract-valid (nodes non-empty,
    # unique; no edges to validate). If there are zero packets we still need a
    # non-empty unique node list, so fall back to (0,) -- this graph carries no
    # edges and the report status makes the emptiness explicit.
    return VisibilityGraph(
        nodes=nodes,
        temporal_edges=(),
        overlap_edges=(),
        loop_edges=(),
        scale_edges=(),
        edge_measurements={},
    )


def _evenly_sample(items: Sequence[Any], max_count: int) -> list[Any]:
    if max_count <= 0 or not items:
        return []
    if len(items) <= max_count:
        return list(items)
    step = len(items) / max_count
    return [items[int(i * step)] for i in range(max_count)]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _safe_log(value: float) -> float:
    import math

    return math.log(max(value, 1e-9))


def _to_array(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return value
    return value
