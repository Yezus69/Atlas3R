from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.vipe_artifact_io import DenseSlamMap
from atlas3r.offline.vipe_import import (
    VipeImportOptions,
    _cloud_from_slam_map,
    import_vipe_world_map,
)
from tests.helpers import write_ppm_sequence


class VipeImportTest(unittest.TestCase):
    def test_import_vipe_npz_fixture_writes_observed_map_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vipe_output = _write_fake_vipe_output(root / "vipe_output")
            frames = root / "frames"
            write_ppm_sequence(frames, count=2, width=2, height=2)
            previous = _write_previous_map_stub(root / "previous_map")
            output = root / "runs" / "room_walk_001_vipe_import"

            result = import_vipe_world_map(
                VipeImportOptions(
                    vipe_output=vipe_output,
                    frames=frames,
                    output=output,
                    point_stride=1,
                    max_points=100,
                    voxel_size_m=0.25,
                    previous_map=previous,
                )
            )

            self.assertEqual(result.status, "available")
            self.assertEqual(result.point_count, 8)
            self.assertGreater(result.occupied_voxel_count, 0)
            self.assertGreater(result.mesh_triangle_count, 0)
            self.assertEqual(result.trajectory_count, 2)
            self.assertTrue((output / "fused_points.ply").is_file())
            self.assertTrue((output / "occupancy_grid.npz").is_file())
            self.assertTrue((output / "observed_voxel_mesh.ply").is_file())
            self.assertTrue((output / "comparison_against_previous_map.json").is_file())

            with np.load(output / "fused_points.npz") as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                source_ids = payload["depth_source_id"]
                sigma = payload["point_sigma_m"]
            self.assertEqual(
                metadata["truth_boundary"]["label_type"], "teacher_pseudo_vipe_near_metric"
            )
            self.assertFalse(metadata["truth_boundary"]["measured_geometry"])
            self.assertFalse(metadata["truth_boundary"]["physical_accuracy_claim"])
            self.assertFalse(metadata["truth_boundary"]["training_quality"])
            self.assertTrue(metadata["truth_boundary"]["observed_only"])
            self.assertEqual(set(source_ids.tolist()), {6})
            self.assertTrue(np.all(sigma >= 0.02))

            trajectory = json.loads((output / "camera_trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory["poses"][1]["pose_source"], "vipe")
            np.testing.assert_allclose(
                trajectory["poses"][1]["camera_center_world_m"], [0.5, 0.0, 0.0]
            )

            comparison = json.loads(
                (output / "comparison_against_previous_map.json").read_text(encoding="utf-8")
            )
            self.assertFalse(comparison["ground_truth_available"])
            self.assertFalse(comparison["physical_accuracy_claim"])

    def test_import_vipe_cli_help_is_dependency_safe(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "offline", "import-vipe", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--vipe-output", result.stdout)
        self.assertIn("--previous-map", result.stdout)

    def test_slam_map_fallback_cloud_preserves_uncertainty_and_sources(self) -> None:
        slam_map = DenseSlamMap(
            points_world_m=np.asarray([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float32),
            colors_u8=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            source_frame_ids=np.asarray([3, 5], dtype=np.int64),
        )

        cloud = _cloud_from_slam_map(slam_map)

        self.assertEqual(cloud.depth_source, "vipe_slam_dense_map_fallback")
        self.assertEqual(cloud.per_frame_point_counts, {"3": 1, "5": 1})
        self.assertEqual(set(cloud.depth_source_id.tolist()), {6})
        self.assertTrue(np.all(cloud.point_sigma_m >= 0.02))


def _write_fake_vipe_output(root: Path) -> Path:
    (root / "pose").mkdir(parents=True)
    (root / "intrinsics").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    transforms[1, 0, 3] = 0.5
    inds = np.asarray([0, 1], dtype=np.int64)
    np.savez_compressed(root / "pose" / "frames.npz", data=transforms, inds=inds)
    intrinsics = np.asarray([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    np.savez_compressed(root / "intrinsics" / "frames.npz", data=intrinsics, inds=inds)
    depths = np.asarray(
        [
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(root / "depth" / "frames.npz", depths=depths, inds=inds)
    return root


def _write_previous_map_stub(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "map_quality.json").write_text(
        json.dumps(
            {
                "fused_point_count": 10,
                "occupied_voxel_count": 3,
                "observed_mesh_triangle_count": 4,
                "camera_trajectory_count": 2,
                "bbox_size_m": [0.1, 0.1, 0.1],
                "inspectable_map_available": True,
                "truth_boundary": {"measured_geometry": False},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    unittest.main()
