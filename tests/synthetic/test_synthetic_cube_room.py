import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.api import CameraModel, MeshChunk, PoseEstimate, WorldMap
from atlas3r.camera.pinhole import project_points_camera, unproject_pixels
from atlas3r.data.synthetic_cube_room import (
    create_synthetic_cube_room_scene,
    write_synthetic_cube_room_session,
)
from atlas3r.pose.transforms import invert_transform, transform_points

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class SyntheticCubeRoomTest(unittest.TestCase):
    def test_synthetic_intrinsics_and_poses_validate_with_contracts(self) -> None:
        scene = create_synthetic_cube_room_scene()

        self.assertIsInstance(scene.camera, CameraModel)
        self.assertEqual(scene.camera.width, 32)
        self.assertEqual(scene.camera.height, 24)
        for frame in scene.frames:
            self.assertIsInstance(frame.pose, PoseEstimate)
            self.assertEqual(frame.camera, scene.camera)
            np.testing.assert_allclose(
                frame.pose.camera_center_world_m,
                frame.pose.T_world_camera[:3, 3],
                atol=1e-6,
            )
            self.assertEqual(frame.depth_m.shape, (scene.camera.height, scene.camera.width))

    def test_analytic_depth_agrees_with_projection_and_unprojection(self) -> None:
        scene = create_synthetic_cube_room_scene()
        samples = ((0, 16, 12), (1, 15, 12), (2, 16, 11))

        for frame_index, u, v in samples:
            frame = scene.frames[frame_index]
            depth = np.array([frame.depth_m[v, u]], dtype=np.float64)
            pixels = np.array([[float(u), float(v)]], dtype=np.float64)

            point_camera = unproject_pixels(pixels, depth, frame.camera.K)
            point_world = transform_points(frame.pose.T_world_camera, point_camera)
            np.testing.assert_allclose(point_world[0], frame.point_world_m[v, u], atol=1e-5)

            T_camera_world = invert_transform(frame.pose.T_world_camera)
            point_camera_round_trip = transform_points(T_camera_world, point_world)
            pixels_round_trip, depth_round_trip = project_points_camera(
                point_camera_round_trip, frame.camera.K
            )
            np.testing.assert_allclose(pixels_round_trip, pixels, atol=1e-9)
            np.testing.assert_allclose(depth_round_trip, depth, atol=1e-7)

    def test_object_masks_align_with_generated_cube_geometry(self) -> None:
        scene = create_synthetic_cube_room_scene()
        bounds = scene.object_bounds_m
        tolerance = 2e-5

        for frame in scene.frames:
            self.assertGreater(int(np.count_nonzero(frame.object_mask)), 0)
            points = frame.point_world_m[frame.object_mask].astype(np.float64)
            self.assertTrue(np.all(points >= bounds.min_corner_m - tolerance))
            self.assertTrue(np.all(points <= bounds.max_corner_m + tolerance))
            on_surface = np.any(
                np.isclose(points, bounds.min_corner_m, atol=tolerance)
                | np.isclose(points, bounds.max_corner_m, atol=tolerance),
                axis=1,
            )
            self.assertTrue(np.all(on_surface))
            self.assertTrue(np.all(frame.object_id[frame.object_mask] == 1))

    def test_ground_truth_mesh_validates_as_mesh_chunk_and_world_map(self) -> None:
        scene = create_synthetic_cube_room_scene()

        self.assertIsInstance(scene.mesh_chunk, MeshChunk)
        self.assertEqual(scene.mesh_chunk.vertices_m.shape, (16, 3))
        self.assertEqual(scene.mesh_chunk.faces.shape, (24, 3))
        self.assertIsInstance(scene.world_map, WorldMap)
        self.assertEqual(scene.world_map.mesh_chunks[scene.mesh_chunk.chunk_id], scene.mesh_chunk)
        self.assertIn(1, scene.world_map.objects)
        self.assertEqual(set(scene.world_map.keyframes), {0, 1, 2})

    def test_session_writer_creates_expected_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "session.atlas3r"
            write_synthetic_cube_room_session(session)

            self.assertTrue((session / "metadata.json").is_file())
            self.assertTrue((session / "poses.jsonl").is_file())
            self.assertTrue((session / "cameras.jsonl").is_file())
            self.assertTrue((session / "objects.jsonl").is_file())
            self.assertTrue(
                (session / "mesh_chunks" / "chunk_cube_room_ground_truth_v1.json").is_file()
            )
            self.assertTrue((session / "mesh_chunks" / "index.json").is_file())
            self.assertTrue((session / "depth" / "frame_000000.npz").is_file())
            self.assertTrue((session / "logs" / "runtime_profile.json").is_file())

            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["session_type"], "synthetic_cube_room")
            self.assertEqual(metadata["frame_count"], 3)
            with np.load(session / "depth" / "frame_000000.npz") as depth_file:
                self.assertEqual(depth_file["depth_m"].shape, (24, 32))
                self.assertEqual(depth_file["object_mask"].shape, (24, 32))

    def test_smoke_command_creates_expected_session_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "cli_session.atlas3r"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "synthetic-cube-room",
                    "--output",
                    str(session),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("synthetic cube-room", result.stdout)
            self.assertTrue((session / "metadata.json").is_file())
            self.assertTrue((session / "poses.jsonl").is_file())
            self.assertTrue((session / "cameras.jsonl").is_file())
            self.assertTrue((session / "objects.jsonl").is_file())
            self.assertTrue((session / "depth" / "frame_000002.npz").is_file())
            self.assertTrue(
                (session / "mesh_chunks" / "chunk_cube_room_ground_truth_v1.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
