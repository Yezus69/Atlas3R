import unittest

import numpy as np

from atlas3r.models.student import StudentClipInput
from atlas3r.training.synthetic_depth_dataset import (
    SyntheticDepthSample,
    generate_synthetic_depth_samples,
    sample_to_student_clip,
)


class SyntheticDepthDatasetTest(unittest.TestCase):
    def test_samples_are_deterministic_for_seed(self) -> None:
        first = generate_synthetic_depth_samples(count=3, width=16, height=12, seed=17)
        second = generate_synthetic_depth_samples(count=3, width=16, height=12, seed=17)

        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        for a, b in zip(first, second, strict=True):
            self.assertEqual(a.metadata, b.metadata)
            np.testing.assert_array_equal(a.rgb_u8, b.rgb_u8)
            np.testing.assert_array_equal(a.rgb_model, b.rgb_model)
            np.testing.assert_array_equal(a.depth_m, b.depth_m)
            np.testing.assert_array_equal(a.object_mask, b.object_mask)
            np.testing.assert_array_equal(a.K, b.K)
            np.testing.assert_array_equal(a.T_world_camera, b.T_world_camera)

    def test_sample_shapes_dtypes_ranges_and_variation_are_valid(self) -> None:
        samples = generate_synthetic_depth_samples(count=4, width=32, height=24, seed=5)
        sample = samples[0]

        self.assertIsInstance(sample, SyntheticDepthSample)
        self.assertEqual(sample.rgb_u8.shape, (24, 32, 3))
        self.assertEqual(sample.rgb_u8.dtype, np.uint8)
        self.assertEqual(sample.rgb_model.shape, (3, 24, 32))
        self.assertEqual(sample.rgb_model.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(sample.rgb_model)), 0.0)
        self.assertLessEqual(float(np.max(sample.rgb_model)), 1.0)
        self.assertEqual(sample.depth_m.shape, (24, 32))
        self.assertEqual(sample.depth_m.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(sample.depth_m)))
        self.assertTrue(np.all(sample.depth_m > 0.0))
        self.assertTrue(np.all(sample.depth_sigma_m >= 0.0))
        self.assertTrue(np.all((sample.confidence >= 0.0) & (sample.confidence <= 1.0)))
        self.assertEqual(sample.object_mask.dtype, np.bool_)
        self.assertTrue(bool(np.any(sample.object_mask)))
        self.assertFalse(bool(np.all(sample.object_mask)))
        self.assertGreater(np.unique(sample.rgb_u8.reshape(-1, 3), axis=0).shape[0], 8)
        self.assertEqual(sample.K.shape, (3, 3))
        self.assertEqual(sample.T_world_camera.shape, (4, 4))
        self.assertEqual(sample.camera_center_world_m.shape, (3,))
        self.assertIs(sample.metadata["synthetic_only"], True)
        self.assertEqual(sample.metadata["metric_depth"], "analytic_ray_box_intersection")

    def test_camera_centers_vary_across_samples(self) -> None:
        samples = generate_synthetic_depth_samples(count=4, width=16, height=12, seed=11)
        centers = np.stack([sample.camera_center_world_m for sample in samples], axis=0)

        self.assertGreater(float(np.max(np.std(centers, axis=0))), 0.01)

    def test_sample_to_student_clip_returns_valid_boundary_record(self) -> None:
        sample = generate_synthetic_depth_samples(count=1, width=16, height=12, seed=23)[0]

        clip = sample_to_student_clip(sample)

        self.assertIsInstance(clip, StudentClipInput)
        self.assertEqual(clip.frame_ids, (sample.frame_id,))
        self.assertEqual(clip.images_rgb.shape, (1, 1, 3, 12, 16))
        self.assertEqual(clip.intrinsics.shape, (1, 1, 3, 3))
        self.assertEqual(clip.T_world_camera_prior.shape, (1, 1, 4, 4))
        self.assertIs(clip.metadata["synthetic_only"], True)
        np.testing.assert_array_equal(clip.images_rgb[0, 0], sample.rgb_model)
        np.testing.assert_array_equal(clip.intrinsics[0, 0], sample.K)

    def test_invalid_generation_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "count"):
            generate_synthetic_depth_samples(count=0)
        with self.assertRaisesRegex(ValueError, "width"):
            generate_synthetic_depth_samples(count=1, width=0)
        with self.assertRaisesRegex(ValueError, "height"):
            generate_synthetic_depth_samples(count=1, height=0)


if __name__ == "__main__":
    unittest.main()
