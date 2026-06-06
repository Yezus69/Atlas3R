from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.fused_world_map_artifacts import FusedPointCloud
from atlas3r.offline.topdown_preview import write_topdown_preview


class TopdownPreviewTest(unittest.TestCase):
    def test_svg_writes_points_trajectory_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topdown_preview.svg"
            cloud = FusedPointCloud(
                points_world_m=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32),
                colors_u8=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
                confidence=np.ones((2,), dtype=np.float32),
                source_frame_ids=np.array([0, 1], dtype=np.int64),
                source_keyframe_ids=np.array([0, 1], dtype=np.int64),
                depth_source_id=np.array([1, 1], dtype=np.int32),
                disagreement_rel=np.zeros((2,), dtype=np.float32),
                point_sigma_m=np.full((2,), 0.1, dtype=np.float32),
                depth_source="consensus",
                metric_scale_source="depth_pro_vggt_soft_metric_prior",
                valid_depth_ratio=1.0,
                rejected_low_confidence_ratio=0.0,
                rejected_high_disagreement_ratio=0.0,
                mapped_disagreement_mean=None,
                mapped_disagreement_p50=None,
                mapped_disagreement_p95=None,
                per_frame_point_counts={},
            )

            write_topdown_preview(
                path,
                cloud,
                [
                    {"camera_center_world_m": [0.0, 0.0, 0.0]},
                    {"camera_center_world_m": [1.0, 0.0, 1.0]},
                ],
            )

            svg = path.read_text(encoding="utf-8")
            self.assertIn('id="map-points"', svg)
            self.assertIn('id="camera-trajectory"', svg)
            self.assertIn("unanchored soft-metric map, not physical ground truth", svg)
            self.assertIn('class="map-point"', svg)


if __name__ == "__main__":
    unittest.main()
