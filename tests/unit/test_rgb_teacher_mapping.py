import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas3r.mapping.mesh_chunks import load_mesh_chunk_npz
from atlas3r.runtime.rgb_teacher_conversion import (
    opencv_camera_from_world_to_T_world_camera,
    teacher_prediction_to_observations,
)
from atlas3r.runtime.rgb_teacher_inputs import RGBTeacherFrame
from atlas3r.runtime.rgb_teacher_mapping import RGBTeacherMapConfig, run_rgb_teacher_mapping
from atlas3r.teachers.external.contracts import ExternalTeacherDependencyError

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class RGBTeacherMappingTest(unittest.TestCase):
    def test_opencv_camera_from_world_extrinsic_converts_to_T_world_camera(self) -> None:
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        T_camera_world = np.linalg.inv(T_world_camera)

        converted = opencv_camera_from_world_to_T_world_camera(
            T_camera_world[np.newaxis, :, :],
            clip_length=1,
        )

        np.testing.assert_allclose(converted[0], T_world_camera, atol=1e-6)

    def test_teacher_prediction_without_pose_is_rejected(self) -> None:
        frame = _rgb_teacher_frame()
        raw = {
            "depth": np.ones((1, 4, 4), dtype=np.float32),
            "depth_conf": np.ones((1, 4, 4), dtype=np.float32),
            "intrinsic": frame.K[np.newaxis, :, :],
        }

        with self.assertRaisesRegex(ValueError, "missing camera pose"):
            teacher_prediction_to_observations(
                raw,
                frames=(frame,),
                teacher_name="vggt",
                metric_scale_source="teacher_scale_unverified",
            )

    def test_fixture_image_folder_maps_nonzero_mesh_chunks_and_truth_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=4)
            output = root / "out"

            summary = run_rgb_teacher_mapping(
                RGBTeacherMapConfig(
                    input=input_dir,
                    output=output,
                    teacher="fixture-vggt",
                    max_frames=4,
                    frame_stride=1,
                    teacher_window_size=4,
                    teacher_window_overlap=0,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    pixel_stride=1,
                    export_mesh_chunks=True,
                    mesh_format="ply",
                    rgb_only=True,
                )
            )
            manifest = json.loads(
                (output / "mesh_chunks" / "mesh_chunk_manifest.json").read_text(encoding="utf-8")
            )
            updates = _jsonl(output / "mesh_chunks" / "mesh_chunk_updates.jsonl")
            first_chunk = load_mesh_chunk_npz(
                output / "mesh_chunks" / str(updates[0]["payload_npz"])
            )
            ply_counts = _ply_header_counts(output / "mesh_chunks" / str(updates[0]["payload_ply"]))
            pseudo_recording_exists = (
                output / "rgb_teacher_recording" / "atlas3r_recording.json"
            ).is_file()

        self.assertGreater(summary["mesh_chunk_count"], 0)
        self.assertGreater(summary["total_vertex_count"], 0)
        self.assertGreater(summary["total_triangle_count"], 0)
        self.assertEqual(summary["pseudo_depth_count"], 4)
        self.assertTrue(summary["pseudo_depth_used"])
        self.assertTrue(summary["pseudo_pose_used"])
        self.assertFalse(summary["truth_boundary"]["measured_depth_used"])  # type: ignore[index]
        self.assertFalse(summary["truth_boundary"]["measured_pose_used"])  # type: ignore[index]
        self.assertTrue(manifest["truth_boundary"]["teacher_geometry_used"])  # type: ignore[index]
        self.assertFalse(manifest["truth_boundary"]["measured_depth_used"])  # type: ignore[index]
        self.assertTrue(pseudo_recording_exists)
        self.assertGreater(first_chunk.vertex_count, 0)
        self.assertEqual(ply_counts["vertex"], first_chunk.vertex_count)

    def test_unavailable_real_vggt_path_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=1)

            with patch("atlas3r.teachers.external.vggt.find_spec", return_value=None):
                with self.assertRaisesRegex(
                    ExternalTeacherDependencyError,
                    "VGGT.*ATLAS3R_VGGT_REPO",
                ):
                    run_rgb_teacher_mapping(
                        RGBTeacherMapConfig(
                            input=input_dir,
                            output=root / "out",
                            teacher="vggt",
                            max_frames=1,
                            frame_stride=1,
                            teacher_window_size=1,
                            teacher_window_overlap=0,
                        )
                    )

    def test_runtime_map_rgb_teacher_cli_help_and_fixture_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_npz_image_folder(root / "images", frame_count=3)
            output = root / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            help_result = subprocess.run(
                [sys.executable, "-m", "atlas3r", "runtime", "map-rgb-teacher", "--help"],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            run_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "runtime",
                    "map-rgb-teacher",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--teacher",
                    "fixture-vggt",
                    "--max-frames",
                    "3",
                    "--frame-stride",
                    "1",
                    "--teacher-window-size",
                    "3",
                    "--teacher-window-overlap",
                    "0",
                    "--voxel-size-m",
                    "0.25",
                    "--truncation-voxels",
                    "3.0",
                    "--pixel-stride",
                    "1",
                    "--export-mesh-chunks",
                    "--mesh-format",
                    "ply",
                    "--rgb-only",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            summary = json.loads((output / "rgb_teacher_summary.json").read_text("utf-8"))

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--teacher-window-size", help_result.stdout)
        self.assertEqual(run_result.returncode, 0, run_result.stderr)
        self.assertGreater(summary["mesh_chunk_count"], 0)
        self.assertFalse(summary["truth_boundary"]["measured_pose_used"])  # type: ignore[index]


def _rgb_teacher_frame() -> RGBTeacherFrame:
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return RGBTeacherFrame(
        frame_id=0,
        timestamp_s=0.0,
        rgb_u8=np.zeros((4, 4, 3), dtype=np.uint8),
        K=K,
        source_path="unit",
        source_metadata={},
    )


def _write_npz_image_folder(path: Path, *, frame_count: int) -> Path:
    path.mkdir(parents=True)
    K = np.asarray([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.savez(path / "intrinsics.npz", K=K)
    for frame_id in range(frame_count):
        rgb = np.full((8, 8, 3), frame_id * 20, dtype=np.uint8)
        np.savez(path / f"frame_{frame_id:06d}.npz", rgb_u8=rgb)
    return path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ply_header_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "end_header":
            break
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            counts[parts[1]] = int(parts[2])
    return counts


if __name__ == "__main__":
    unittest.main()
