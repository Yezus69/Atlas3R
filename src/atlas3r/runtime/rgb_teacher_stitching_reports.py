"""JSON report helpers for RGB teacher stitching diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.rgb_teacher_stitching_types import StitchGraph


def stitch_edge_records(graph: StitchGraph) -> list[dict[str, object]]:
    return [
        {
            "source_window_index": edge.source_window_index,
            "target_window_index": edge.target_window_index,
            "overlap_frame_ids": list(edge.overlap_frame_ids),
            "accepted": edge.accepted,
            "rejection_reason": edge.rejection_reason,
            "sim3_source_to_global": edge.sim3_source_to_global.to_record(),
            "metrics": edge.metrics,
        }
        for edge in graph.edges
    ]


def stitch_graph_record(graph: StitchGraph) -> dict[str, object]:
    return {
        "nodes": [
            {
                "window_index": node.window_index,
                "frame_ids": list(node.frame_ids),
                "pseudo_submap_id": node.pseudo_submap_id,
                "accepted": node.accepted,
                "rejection_reason": node.rejection_reason,
                "confidence_scale": node.confidence_scale,
                "sim3_local_to_global": node.sim3_local_to_global.to_record(),
            }
            for node in graph.nodes
        ],
        "edges": stitch_edge_records(graph),
    }


def observation_submap_id(observation: DepthObservation) -> int | None:
    stitching = observation.pose.diagnostics.get("stitching")
    if not isinstance(stitching, Mapping):
        return None
    value = stitching.get("pseudo_submap_id")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["observation_submap_id", "stitch_edge_records", "stitch_graph_record"]
