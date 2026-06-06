from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.classical_map_comparison import (
    compare_point_sets,
    write_classical_map_comparison,
)
from atlas3r.offline.fused_world_map import FusedWorldMapResult
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.trajectory_alignment import TrajectoryAlignmentResult


class ClassicalMapComparisonTest(unittest.TestCase):
    def test_nearest_neighbor_and_bbox_metrics_compute(self) -> None:
        map_points = np.asarray([[0, 0, 0], [1, 1, 1], [2, 1, 1], [0, 1, 1]], dtype=np.float32)
        classical_points = np.asarray([[0, 0, 0], [1.1, 1, 1], [0, 1, 1]], dtype=np.float32)

        metrics = compare_point_sets(map_points, classical_points, near_threshold_m=0.2)

        self.assertEqual(metrics["status"], "available")
        self.assertLessEqual(float(metrics["nearest_neighbor_distance_p95_m"]), 0.15)
        self.assertGreater(float(metrics["bbox_overlap_ratio"]), 0.0)
        self.assertGreater(float(metrics["classical_points_near_map_ratio"]), 0.5)

    def test_empty_inputs_are_unavailable(self) -> None:
        metrics = compare_point_sets(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            near_threshold_m=0.2,
        )

        self.assertEqual(metrics["status"], "unavailable")

    def test_write_comparison_records_unavailable_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_classical_map_comparison(
                tmp,
                alignment=TrajectoryAlignmentResult(status="unavailable", reason="fixture"),
                raw_world_map=FusedWorldMapResult(status="disabled"),
                optimizer=MapConsistencyOptimizerResult(status="disabled"),
                best_map=_BestMapStub(),
                voxel_size_m=0.05,
            )

            self.assertEqual(result.status, "unavailable")
            self.assertTrue((Path(tmp) / "diagnostics" / "classical_map_comparison.json").is_file())


class _BestMapStub:
    fused_points_npz_path = None


if __name__ == "__main__":
    unittest.main()
