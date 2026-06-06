"""Process and artifact helpers for the COLMAP/GLOMAP witness."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from atlas3r.offline.classical_geometry import ClassicalWitnessResult, classical_truth_boundary
from atlas3r.offline.colmap_import import ColmapSparseModel, write_sparse_points_ply
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import write_json, write_jsonl

ColmapMatcher = Literal["sequential", "exhaustive"]
ColmapCameraModel = Literal["SIMPLE_RADIAL", "PINHOLE", "OPENCV"]


@dataclass(frozen=True)
class ColmapWitnessOptions:
    enable_colmap: bool = False
    colmap_exe: str = "colmap"
    colmap_camera_model: ColmapCameraModel = "SIMPLE_RADIAL"
    colmap_matcher: ColmapMatcher = "sequential"
    colmap_max_images: int | None = None
    colmap_image_stride: int = 1
    colmap_use_gpu: int = 1
    colmap_run_dense: bool = False
    colmap_run_poisson: bool = False
    colmap_timeout_s: int = 7200
    enable_glomap: bool = False
    glomap_exe: str = "glomap"
    colmap_proposal_cache: str | None = None
    glomap_proposal_cache: str | None = None


def colmap_command_plan(
    colmap_exe: str,
    *,
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    camera_model: ColmapCameraModel,
    matcher: ColmapMatcher,
    use_gpu: int,
) -> list[tuple[str, list[str]]]:
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    gpu = "1" if int(use_gpu) else "0"
    return [
        (
            "feature_extractor",
            [
                colmap_exe,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--ImageReader.camera_model",
                camera_model,
                "--SiftExtraction.use_gpu",
                gpu,
            ],
        ),
        (
            matcher_command,
            [
                colmap_exe,
                matcher_command,
                "--database_path",
                str(database_path),
                "--SiftMatching.use_gpu",
                gpu,
            ],
        ),
        (
            "mapper",
            [
                colmap_exe,
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse_dir),
            ],
        ),
    ]


def run_dense_if_requested(
    runner: CommandRunner,
    colmap_exe: str,
    *,
    image_dir: Path,
    sparse_model_dir: Path,
    dense_dir: Path,
    classical_dir: Path,
    options: ColmapWitnessOptions,
) -> dict[str, str]:
    if not options.colmap_run_dense:
        return {}
    steps = [
        [
            colmap_exe,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ],
        [
            colmap_exe,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ],
        [
            colmap_exe,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(classical_dir / "colmap_dense_fused.ply"),
        ],
    ]
    for name, cmd in zip(
        ("image_undistorter", "patch_match_stereo", "stereo_fusion"),
        steps,
        strict=True,
    ):
        if runner.run(name, cmd) is not None:
            return {}
    paths = {"dense_fused": "classical/colmap_dense_fused.ply"}
    if options.colmap_run_poisson:
        mesh_cmd = [
            colmap_exe,
            "poisson_mesher",
            "--input_path",
            str(classical_dir / "colmap_dense_fused.ply"),
            "--output_path",
            str(classical_dir / "colmap_dense_mesh.ply"),
        ]
        if runner.run("poisson_mesher", mesh_cmd) is None:
            paths["dense_mesh"] = "classical/colmap_dense_mesh.ply"
    return paths


def run_glomap_if_requested(
    classical_dir: Path,
    options: ColmapWitnessOptions,
    *,
    image_dir: Path,
    database: Path,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    if not options.enable_glomap:
        return {"status": "disabled"}
    exe = resolve_executable(options.glomap_exe, "glomap")
    if exe is None:
        return {
            "status": "unavailable",
            "reason": f"GLOMAP executable not found: {options.glomap_exe}",
            "install_hint": "Install GLOMAP externally or pass --glomap-exe.",
        }
    output = classical_dir / "glomap" / "sparse"
    output.mkdir(parents=True, exist_ok=True)
    runner = CommandRunner(classical_dir, commands, timeout_s=options.colmap_timeout_s)
    failure = runner.run(
        "glomap_mapper",
        [
            exe,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(output),
        ],
    )
    if failure is not None:
        return {"status": "failed", **failure}
    return {"status": "ran", "output_path": "classical/glomap/sparse"}


class CommandRunner:
    def __init__(self, classical_dir: Path, commands: list[dict[str, object]], *, timeout_s: int):
        self.classical_dir = classical_dir
        self.commands = commands
        self.timeout_s = timeout_s

    def run(self, stage: str, command: list[str]) -> dict[str, object] | None:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            row = {
                "stage": stage,
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": tail(completed.stdout),
                "stderr_tail": tail(completed.stderr),
            }
        except subprocess.TimeoutExpired as exc:
            row = {
                "stage": stage,
                "command": command,
                "returncode": None,
                "stdout_tail": tail(exc.stdout),
                "stderr_tail": tail(exc.stderr),
                "timeout_s": self.timeout_s,
            }
        self.commands.append(row)
        write_jsonl(self.classical_dir / "colmap_commands.jsonl", self.commands)
        (self.classical_dir / "colmap_stdout_tail.txt").write_text(
            str(row.get("stdout_tail", "")), encoding="utf-8"
        )
        (self.classical_dir / "colmap_stderr_tail.txt").write_text(
            str(row.get("stderr_tail", "")), encoding="utf-8"
        )
        return row if row.get("returncode") != 0 else None


def resolve_executable(value: str, name: str) -> str | None:
    path = Path(value)
    if path.is_file():
        return str(path)
    found = shutil.which(value)
    if found is not None:
        return found
    for candidate in common_executable_paths(name):
        if candidate.is_file():
            return str(candidate)
    return None


def common_executable_paths(name: str) -> tuple[Path, ...]:
    exe = f"{name}.exe"
    bat = f"{name}.bat"
    cwd = Path.cwd()
    return (
        Path("C:/Program Files/COLMAP") / exe,
        Path("C:/Program Files/COLMAP") / bat,
        Path("C:/Program Files/GLOMAP") / exe,
        Path("C:/Program Files/GLOMAP") / bat,
        cwd / "external" / name / exe,
        cwd / "external" / name / bat,
        cwd.parent / "homebrain" / "external" / name / exe,
        cwd.parent / "homebrain" / "external" / name / bat,
    )


def best_sparse_model_dir(sparse_dir: Path) -> Path:
    children = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not children:
        return sparse_dir
    return sorted(children, key=lambda path: path.name)[0]


def with_required_artifacts(
    run_dir: str | Path, result: ClassicalWitnessResult
) -> ClassicalWitnessResult:
    root = Path(run_dir)
    classical_dir = root / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)
    paths = _RequiredPaths.from_result(result)
    if paths.run_manifest_path is None:
        write_json(
            classical_dir / "colmap_run_manifest.json",
            {
                "format_name": "atlas3r_colmap_run_manifest",
                "format_version": 1,
                "status": result.status,
                "reason": result.reason,
                "source": result.source,
            },
        )
        paths.run_manifest_path = "classical/colmap_run_manifest.json"
    if paths.command_log_path is None:
        write_jsonl(classical_dir / "colmap_commands.jsonl", [])
        paths.command_log_path = "classical/colmap_commands.jsonl"
    if paths.stdout_tail_path is None:
        (classical_dir / "colmap_stdout_tail.txt").write_text("", encoding="utf-8")
        paths.stdout_tail_path = "classical/colmap_stdout_tail.txt"
    if paths.stderr_tail_path is None:
        (classical_dir / "colmap_stderr_tail.txt").write_text(
            result.stderr_tail or "", encoding="utf-8"
        )
        paths.stderr_tail_path = "classical/colmap_stderr_tail.txt"
    if paths.sparse_summary_path is None:
        _write_placeholder_sparse_summary(classical_dir, result)
        paths.sparse_summary_path = "classical/colmap_sparse_summary.json"
    if paths.cameras_jsonl_path is None:
        write_jsonl(classical_dir / "colmap_cameras.jsonl", [])
        paths.cameras_jsonl_path = "classical/colmap_cameras.jsonl"
    if paths.images_jsonl_path is None:
        write_jsonl(classical_dir / "colmap_images.jsonl", [])
        paths.images_jsonl_path = "classical/colmap_images.jsonl"
    if paths.points_npz_path is None:
        _write_empty_points_npz(classical_dir, result)
        paths.points_npz_path = "classical/colmap_points3d.npz"
    if paths.sparse_points_ply_path is None:
        write_sparse_points_ply(
            classical_dir / "colmap_sparse_points.ply",
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            metric_scale_source="colmap_sfm_unanchored",
        )
        paths.sparse_points_ply_path = "classical/colmap_sparse_points.ply"
    return replace(
        result,
        run_manifest_path=paths.run_manifest_path,
        command_log_path=paths.command_log_path,
        stdout_tail_path=paths.stdout_tail_path,
        stderr_tail_path=paths.stderr_tail_path,
        sparse_summary_path=paths.sparse_summary_path,
        cameras_jsonl_path=paths.cameras_jsonl_path,
        images_jsonl_path=paths.images_jsonl_path,
        points_npz_path=paths.points_npz_path,
        sparse_points_ply_path=paths.sparse_points_ply_path,
    )


def run_manifest_payload(
    options: ColmapWitnessOptions,
    selected: tuple[KeyframeRecord, ...],
    colmap_exe: str,
    image_dir: Path,
    database: Path,
) -> dict[str, object]:
    return {
        "format_name": "atlas3r_colmap_run_manifest",
        "format_version": 1,
        "colmap_exe": colmap_exe,
        "camera_model": options.colmap_camera_model,
        "matcher": options.colmap_matcher,
        "max_images": options.colmap_max_images,
        "image_stride": options.colmap_image_stride,
        "use_gpu": int(options.colmap_use_gpu),
        "run_dense": options.colmap_run_dense,
        "run_poisson": options.colmap_run_poisson,
        "timeout_s": options.colmap_timeout_s,
        "image_path": str(image_dir),
        "database_path": str(database),
        "selected_frame_ids": [item.frame_id for item in selected],
    }


def colmap_install_hint() -> str:
    return (
        "Install COLMAP externally and ensure `colmap` is on PATH, or pass --colmap-exe "
        "with the full executable path. Do not vendor COLMAP into this repository."
    )


def tail(text: object, limit: int = 8000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return str(text)[-limit:]


@dataclass
class _RequiredPaths:
    run_manifest_path: str | None
    command_log_path: str | None
    stdout_tail_path: str | None
    stderr_tail_path: str | None
    sparse_summary_path: str | None
    cameras_jsonl_path: str | None
    images_jsonl_path: str | None
    points_npz_path: str | None
    sparse_points_ply_path: str | None

    @classmethod
    def from_result(cls, result: ClassicalWitnessResult) -> _RequiredPaths:
        return cls(
            run_manifest_path=result.run_manifest_path,
            command_log_path=result.command_log_path,
            stdout_tail_path=result.stdout_tail_path,
            stderr_tail_path=result.stderr_tail_path,
            sparse_summary_path=result.sparse_summary_path,
            cameras_jsonl_path=result.cameras_jsonl_path,
            images_jsonl_path=result.images_jsonl_path,
            points_npz_path=result.points_npz_path,
            sparse_points_ply_path=result.sparse_points_ply_path,
        )


def _write_placeholder_sparse_summary(classical_dir: Path, result: ClassicalWitnessResult) -> None:
    write_json(
        classical_dir / "colmap_sparse_summary.json",
        {
            "format_name": "atlas3r_colmap_sparse_summary",
            "format_version": 1,
            "status": result.status,
            "reason": result.reason,
            "registered_image_count": result.registered_image_count,
            "sparse_point_count": result.sparse_point_count,
            "metric_scale_source": "colmap_sfm_unanchored",
            "truth_boundary": classical_truth_boundary("colmap"),
        },
    )


def _write_empty_points_npz(classical_dir: Path, result: ClassicalWitnessResult) -> None:
    np.savez_compressed(
        classical_dir / "colmap_points3d.npz",
        points_world_m=np.zeros((0, 3), dtype=np.float32),
        colors_u8=np.zeros((0, 3), dtype=np.uint8),
        reprojection_error=np.zeros((0,), dtype=np.float32),
        track_length=np.zeros((0,), dtype=np.int32),
        metadata_json=json.dumps(
            {
                "format_name": "atlas3r_colmap_sparse_points",
                "format_version": 1,
                "point_count": 0,
                "status": result.status,
                "metric_scale_source": "colmap_sfm_unanchored",
            },
            sort_keys=True,
        ),
    )


def write_sparse_summary(
    classical_dir: Path, model: ColmapSparseModel, model_text_path: Path
) -> str:
    write_json(
        classical_dir / "colmap_sparse_summary.json",
        {
            "format_name": "atlas3r_colmap_sparse_summary",
            "format_version": 1,
            "registered_image_count": model.registered_image_count,
            "sparse_point_count": model.sparse_point_count,
            "camera_count": len(model.cameras),
            "model_text_path": str(model_text_path),
            "metric_scale_source": "colmap_sfm_unanchored",
            "truth_boundary": classical_truth_boundary("colmap"),
        },
    )
    return "classical/colmap_sparse_summary.json"
