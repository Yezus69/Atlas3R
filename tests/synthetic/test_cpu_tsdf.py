import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.data.synthetic_cube_room import create_synthetic_cube_room_scene
from atlas3r.mapping.cpu_tsdf import (
    evaluate_surface_against_synthetic_cube_room,
    extract_tsdf_surface,
    integrate_synthetic_cube_room_scene,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class CpuTSDFReferenceTest(unittest.TestCase):
    def test_integration_and_surface_extraction_are_deterministic(self) -> None:
        scene = create_synthetic_cube_room_scene()

        first_volume = integrate_synthetic_cube_room_scene(scene)
        second_volume = integrate_synthetic_cube_room_scene(scene)
        first_surface = extract_tsdf_surface(first_volume)
        second_surface = extract_tsdf_surface(second_volume)

        np.testing.assert_array_equal(first_volume.tsdf, second_volume.tsdf)
        np.testing.assert_array_equal(first_volume.weight, second_volume.weight)
        np.testing.assert_array_equal(first_surface.points_world_m, second_surface.points_world_m)
        np.testing.assert_array_equal(first_surface.confidence, second_surface.confidence)
        np.testing.assert_array_equal(first_surface.uncertainty_m, second_surface.uncertainty_m)

    def test_surface_output_carries_required_confidence_uncertainty_metadata(self) -> None:
        scene = create_synthetic_cube_room_scene()
        volume = integrate_synthetic_cube_room_scene(scene)
        surface = extract_tsdf_surface(volume)

        self.assertEqual(surface.metadata["source_frame_ids"], [0, 1, 2])
        self.assertEqual(surface.metadata["coordinate_frame"], "synthetic_world")
        self.assertEqual(surface.metadata["metric_scale_source"], "known_anchor")
        self.assertEqual(surface.metadata["voxel_size_m"], 0.1)
        self.assertIn("observed_coverage_estimate", surface.metadata)
        self.assertIn("uncertainty_summary_m", surface.metadata)
        self.assertEqual(surface.confidence.shape, (surface.points_world_m.shape[0],))
        self.assertEqual(surface.uncertainty_m.shape, (surface.points_world_m.shape[0],))
        self.assertTrue(np.all(surface.confidence > 0.0))
        self.assertTrue(np.all(surface.uncertainty_m >= 0.0))

    def test_surface_overlaps_synthetic_room_and_object_bounds_at_voxel_scale(self) -> None:
        scene = create_synthetic_cube_room_scene()
        volume = integrate_synthetic_cube_room_scene(scene)
        surface = extract_tsdf_surface(volume)
        metrics = evaluate_surface_against_synthetic_cube_room(scene, surface)
        voxel_size_m = float(metrics["voxel_size_m"])
        surface_min = np.asarray(metrics["surface_bounds_min_m"], dtype=np.float64)
        surface_max = np.asarray(metrics["surface_bounds_max_m"], dtype=np.float64)

        self.assertEqual(metrics["surface_points_inside_room_bounds_ratio"], 1.0)
        self.assertEqual(metrics["surface_points_within_one_voxel_of_gt_surface_ratio"], 1.0)
        self.assertLessEqual(
            metrics["surface_to_gt_box_surface_max_distance_m"], voxel_size_m + 1e-6
        )
        self.assertGreater(metrics["room_surface_point_count"], 0)
        self.assertGreater(metrics["object_surface_point_count"], 0)
        self.assertLessEqual(surface_min[0], scene.room_bounds_m.min_corner_m[0] + voxel_size_m)
        self.assertLessEqual(surface_min[1], scene.room_bounds_m.min_corner_m[1] + voxel_size_m)
        self.assertGreaterEqual(surface_max[0], scene.room_bounds_m.max_corner_m[0] - voxel_size_m)
        self.assertGreaterEqual(surface_max[1], scene.room_bounds_m.max_corner_m[1] - voxel_size_m)
        self.assertGreaterEqual(surface_max[2], scene.room_bounds_m.max_corner_m[2] - voxel_size_m)
        self.assertIn("not an accuracy report", " ".join(metrics["known_limitations"]))

    def test_tsdf_cube_room_cli_writes_surface_metrics_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf_smoke"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "tsdf-cube-room",
                    "--output",
                    str(output),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("surface_points.npz", result.stdout)
            self.assertIn("metrics.json", result.stdout)
            self.assertTrue((output / "synthetic_cube_room.atlas3r" / "metadata.json").is_file())
            self.assertTrue((output / "tsdf_grid.npz").is_file())
            self.assertTrue((output / "surface_points.npz").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertTrue((output / "metrics.json").is_file())

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["artifact_type"], "phase_0d_cpu_tsdf_surface_points")
            self.assertEqual(
                metrics["metric_family"],
                "phase_0d_synthetic_axis_aligned_box_reference",
            )
            with np.load(output / "surface_points.npz") as surface_file:
                self.assertEqual(surface_file["points_world_m"].shape[1], 3)
                self.assertEqual(
                    surface_file["confidence"].shape[0],
                    surface_file["points_world_m"].shape[0],
                )


if __name__ == "__main__":
    unittest.main()
