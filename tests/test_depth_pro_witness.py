from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.models.adapters.depth_pro_adapter import (
    DepthProBatchPrediction,
    DepthProFramePrediction,
)
from atlas3r.offline.depth_pro_witness import (
    focal_px_to_K,
    load_depth_pro_proposal_cache,
    normalize_depth_pro_prediction,
    run_depth_pro_witness,
)
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_fake_depth_pro_cache


class DepthProWitnessTest(unittest.TestCase):
    def test_replay_cache_loads_without_heavy_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = write_fake_depth_pro_cache(Path(tmp) / "cache", frame_ids=(0, 1))

            result = load_depth_pro_proposal_cache(cache_dir)

            self.assertEqual(result.runtime_status, "replayed")
            self.assertTrue(result.has_depth)
            self.assertEqual(len(result.camera_records), 2)
            self.assertEqual(len(result.depth_records), 2)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("depth_pro", sys.modules)

    def test_unavailable_status_has_install_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failures = []

            result = run_depth_pro_witness(
                ensure_run_tree(Path(tmp) / "run"),
                frame_cache=_empty_frame_cache(),
                keyframes=(),
                options=_enabled_options(),
                failure_points=failures,
            )

            self.assertFalse(result.available)
            self.assertIn("ml-depth-pro", result.install_hint)
            self.assertTrue(failures)

    def test_fake_prediction_normalizes_depth_intrinsics_and_masks(self) -> None:
        prediction = DepthProBatchPrediction(
            frame_predictions=(
                DepthProFramePrediction(
                    frame_id=3,
                    depth_m=np.array([[2.0, np.nan], [-1.0, 4.0]], dtype=np.float32),
                    focal_px=10.0,
                    runtime_ms=1.0,
                    output_shapes={"depth": [2, 2]},
                ),
            ),
            model_source="fixture",
            checkpoint="fixture",
            device="cpu",
            image_size=None,
            runtime_ms=1.0,
        )

        cameras, depths, arrays, frames = normalize_depth_pro_prediction(
            prediction, keyframe_index={3: 0}
        )

        self.assertEqual(cameras[0]["teacher_name"], "depth_pro")
        self.assertEqual(cameras[0]["focal_px"], 10.0)
        np.testing.assert_allclose(cameras[0]["K"], focal_px_to_K(10.0, width=2, height=2))
        self.assertEqual(len(depths), 1)
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(
            arrays["depth_pro_valid_mask_000000"], np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        self.assertFalse(depths[0]["truth_boundary"]["measured_geometry"])
        self.assertTrue(depths[0]["confidence_derived"])

    def test_focal_conversion_uses_center_principal_point(self) -> None:
        K = focal_px_to_K(8.0, width=5, height=3)

        np.testing.assert_allclose(
            K,
            np.array([[8.0, 0.0, 2.0], [0.0, 8.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )


def _empty_frame_cache() -> object:
    class EmptyFrameCache:
        frames: tuple[object, ...] = ()
        records: tuple[object, ...] = ()

    return EmptyFrameCache()


def _enabled_options() -> object:
    class Options:
        enabled = True
        proposal_cache = None
        repo_path = None
        checkpoint = None
        device = "cpu"
        image_size = None
        max_keyframes = None

    return Options()


if __name__ == "__main__":
    unittest.main()
