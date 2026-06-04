import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.api import CameraModel, MeshChunk, ObjectInstance, PoseEstimate
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.io.session import load_session, validate_session
from atlas3r.visualization.session_preview import write_session_preview

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class SessionInspectTest(unittest.TestCase):
    def test_load_session_reconstructs_phase_0b_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session_path)

            session = validate_session(session_path)

            self.assertEqual(session.frame_count, 3)
            self.assertEqual(session.metadata["session_type"], "synthetic_cube_room")
            self.assertTrue(all(isinstance(pose, PoseEstimate) for pose in session.poses))
            self.assertTrue(all(isinstance(camera, CameraModel) for camera in session.cameras))
            self.assertTrue(
                all(isinstance(instance, ObjectInstance) for instance in session.objects)
            )
            self.assertTrue(all(isinstance(chunk, MeshChunk) for chunk in session.mesh_chunks))
            self.assertEqual(session.mesh_chunks[0].chunk_id, "cube_room_ground_truth")

    def test_missing_required_session_files_raise_path_named_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session_path)
            (session_path / "metadata.json").unlink()

            with self.assertRaisesRegex(ValueError, r"metadata\.json"):
                load_session(session_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session_path)
            (session_path / "mesh_chunks" / "index.json").unlink()

            with self.assertRaisesRegex(ValueError, r"mesh_chunks.*index\.json"):
                load_session(session_path)

    def test_depth_files_are_discovered_in_frame_id_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session_path)

            session = load_session(session_path)

            self.assertEqual(
                [path.name for path in session.depth_files],
                ["frame_000000.npz", "frame_000001.npz", "frame_000002.npz"],
            )

    def test_write_session_preview_creates_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "session.atlas3r"
            preview = root / "preview"
            write_synthetic_cube_room_session(session_path)

            written = write_session_preview(session_path, preview)

            self.assertEqual(
                [path.name for path in written],
                [
                    "index.html",
                    "top_down.svg",
                    "depth_frame_000000.svg",
                    "object_mask_frame_000000.svg",
                ],
            )
            for name in [path.name for path in written]:
                self.assertTrue((preview / name).is_file())

    def test_top_down_svg_contains_expected_scene_labels_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "session.atlas3r"
            preview = root / "preview"
            write_synthetic_cube_room_session(session_path)
            write_session_preview(session_path, preview)

            top_down = (preview / "top_down.svg").read_text(encoding="utf-8")

            self.assertIn(">0<", top_down)
            self.assertIn(">1<", top_down)
            self.assertIn(">2<", top_down)
            self.assertIn("synthetic_cube", top_down)
            self.assertIn("<rect", top_down)
            self.assertIn("<polygon", top_down)
            self.assertIn("<circle", top_down)

    def test_depth_and_object_mask_svg_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "session.atlas3r"
            first_preview = root / "preview_a"
            second_preview = root / "preview_b"
            write_synthetic_cube_room_session(session_path)

            write_session_preview(session_path, first_preview)
            write_session_preview(session_path, second_preview)

            self.assertEqual(
                (first_preview / "depth_frame_000000.svg").read_text(encoding="utf-8"),
                (second_preview / "depth_frame_000000.svg").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (first_preview / "object_mask_frame_000000.svg").read_text(encoding="utf-8"),
                (second_preview / "object_mask_frame_000000.svg").read_text(encoding="utf-8"),
            )

    def test_cli_inspect_session_succeeds_after_smoke_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "cli_session.atlas3r"
            preview = root / "preview"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            smoke_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "synthetic-cube-room",
                    "--output",
                    str(session_path),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(smoke_result.returncode, 0, smoke_result.stderr)

            inspect_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "inspect",
                    "session",
                    "--input",
                    str(session_path),
                    "--output",
                    str(preview),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(inspect_result.returncode, 0, inspect_result.stderr)
            self.assertIn("index.html", inspect_result.stdout)
            self.assertTrue((preview / "index.html").is_file())
            self.assertTrue((preview / "top_down.svg").is_file())

    def test_status_handoff_points_to_phase_6c(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6C - Accelerated Incremental Mapper Prototype", next_task)
        self.assertIn("bounded incremental mapper", next_task)
        self.assertIn("Phase 6B CPU TSDF", next_task)


class SessionPreviewIgnoreTest(unittest.TestCase):
    def test_generated_session_folder_matches_gitignore_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "generated.atlas3r"
            write_synthetic_cube_room_session(session_path)
            copied_path = root / "copy.atlas3r"
            shutil.copytree(session_path, copied_path)

            self.assertTrue(copied_path.is_dir())
            self.assertTrue(
                (ROOT / ".gitignore").read_text(encoding="utf-8").find("*.atlas3r/") >= 0
            )


if __name__ == "__main__":
    unittest.main()
