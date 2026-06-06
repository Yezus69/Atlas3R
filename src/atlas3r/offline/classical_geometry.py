"""Shared structures for classical SfM witness artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from atlas3r.offline.colmap_import import ColmapSparseModel
from atlas3r.offline.run_manifest import write_json

CLASSICAL_LABEL_TYPE = "classical_sfm_proposal"


@dataclass(frozen=True)
class ClassicalWitnessResult:
    status: str
    source: str = "none"
    available: bool = False
    reason: str = ""
    install_hint: str = ""
    registered_image_count: int = 0
    sparse_point_count: int = 0
    dense_point_count: int = 0
    dense_mesh_vertex_count: int = 0
    dense_mesh_face_count: int = 0
    stage_failed: str | None = None
    command: tuple[str, ...] = ()
    returncode: int | None = None
    stderr_tail: str | None = None
    likely_reason: str | None = None
    next_debug_action: str | None = None
    model_text_path: str | None = None
    cameras_jsonl_path: str | None = None
    images_jsonl_path: str | None = None
    points_npz_path: str | None = None
    sparse_points_ply_path: str | None = None
    dense_fused_ply_path: str | None = None
    dense_mesh_ply_path: str | None = None
    status_path: str = "classical/classical_status.json"
    report_path: str = "classical/classical_report.md"
    run_manifest_path: str | None = None
    command_log_path: str | None = None
    stdout_tail_path: str | None = None
    stderr_tail_path: str | None = None
    sparse_summary_path: str | None = None
    glomap_status: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    sparse_model: ColmapSparseModel | None = field(default=None, repr=False, compare=False)

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        values = (
            self.status_path,
            self.report_path,
            self.run_manifest_path,
            self.command_log_path,
            self.stdout_tail_path,
            self.stderr_tail_path,
            self.sparse_summary_path,
            self.cameras_jsonl_path,
            self.images_jsonl_path,
            self.points_npz_path,
            self.sparse_points_ply_path,
            self.dense_fused_ply_path,
            self.dense_mesh_ply_path,
        )
        return tuple(path for path in values if path is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "install_hint": self.install_hint,
            "registered_image_count": self.registered_image_count,
            "sparse_point_count": self.sparse_point_count,
            "dense_point_count": self.dense_point_count,
            "dense_mesh_vertex_count": self.dense_mesh_vertex_count,
            "dense_mesh_face_count": self.dense_mesh_face_count,
            "stage_failed": self.stage_failed,
            "command": list(self.command),
            "returncode": self.returncode,
            "stderr_tail": self.stderr_tail,
            "likely_reason": self.likely_reason,
            "next_debug_action": self.next_debug_action,
            "model_text_path": self.model_text_path,
            "artifacts": {
                "status": self.status_path,
                "report": self.report_path,
                "run_manifest": self.run_manifest_path,
                "commands": self.command_log_path,
                "stdout_tail": self.stdout_tail_path,
                "stderr_tail": self.stderr_tail_path,
                "sparse_summary": self.sparse_summary_path,
                "cameras_jsonl": self.cameras_jsonl_path,
                "images_jsonl": self.images_jsonl_path,
                "points_npz": self.points_npz_path,
                "sparse_points_ply": self.sparse_points_ply_path,
                "dense_fused_ply": self.dense_fused_ply_path,
                "dense_mesh_ply": self.dense_mesh_ply_path,
            },
            "glomap": self.glomap_status,
            "truth_boundary": classical_truth_boundary(self.source),
            "metadata": self.metadata,
        }


def classical_truth_boundary(source: str) -> dict[str, object]:
    source_key = "glomap_sfm_unanchored" if source == "glomap" else "colmap_sfm_unanchored"
    return {
        "label_type": CLASSICAL_LABEL_TYPE,
        "measured_geometry": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "physical_accuracy_claim": False,
        "scale_status": "sfm_scale_unanchored",
        "metric_scale_source": source_key,
        "training_quality": False,
        "accuracy_report": False,
        "usable_for_training": False,
        "realtime_claim": False,
    }


def write_classical_status(
    run_dir: str | Path, result: ClassicalWitnessResult
) -> ClassicalWitnessResult:
    root = Path(run_dir)
    classical_dir = root / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)
    write_json(classical_dir / "classical_status.json", result.to_dict())
    (classical_dir / "classical_report.md").write_text(
        classical_report_markdown(result), encoding="utf-8"
    )
    return result


def classical_report_markdown(result: ClassicalWitnessResult) -> str:
    lines = [
        "# Classical Geometry Witness Report",
        "",
        f"- Status: {result.status}",
        f"- Source: {result.source}",
        f"- Available: {result.available}",
        f"- Reason: {result.reason}",
        f"- Registered images: {result.registered_image_count}",
        f"- Sparse points: {result.sparse_point_count}",
        f"- Dense fused points: {result.dense_point_count}",
        f"- Dense mesh vertices: {result.dense_mesh_vertex_count}",
        f"- Dense mesh faces: {result.dense_mesh_face_count}",
        f"- Stage failed: {result.stage_failed}",
        f"- Likely reason: {result.likely_reason}",
        f"- Next debug action: {result.next_debug_action}",
        "- Physical accuracy claim: false",
        "- Training-quality claim: false",
        "",
        "COLMAP/GLOMAP output is an unanchored classical SfM proposal. It is not "
        "measured geometry, physical ground truth, or a training-quality label.",
        "",
    ]
    if result.command:
        lines.extend(["## Failed Command", "", f"`{' '.join(result.command)}`", ""])
    if result.stderr_tail:
        tail = result.stderr_tail[-2000:]
        lines.extend(["## Stderr Tail", "", "```text", tail, "```", ""])
    if result.glomap_status:
        lines.extend(
            [
                "## GLOMAP",
                "",
                "```json",
                json.dumps(result.glomap_status, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
