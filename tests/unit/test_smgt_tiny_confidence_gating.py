import unittest

import numpy as np

from atlas3r.runtime.rgb_student_gating import (
    StudentMapGateConfig,
    resolve_student_map_valid_policy,
    sanitize_student_depth,
    student_mapping_valid_mask,
)


class SMGTTinyConfidenceGatingTest(unittest.TestCase):
    def test_sigmoid_low_confidence_below_threshold_is_rejected(self) -> None:
        result = student_mapping_valid_mask(
            depth_m=np.full((2, 2), 1.0, dtype=np.float32),
            depth_sigma_m=np.full((2, 2), 0.1, dtype=np.float32),
            confidence=np.full((2, 2), 0.29, dtype=np.float32),
            dynamic_probability=np.zeros((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05, confidence_threshold=0.30),
        )

        self.assertEqual(int(np.count_nonzero(result.valid_mask)), 0)
        self.assertEqual(result.stats["confidence_gated_valid_pixel_count"], 0)

    def test_confidence_threshold_changes_mapped_pixel_count(self) -> None:
        confidence = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

        loose = student_mapping_valid_mask(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_sigma_m=np.full((2, 2), 0.1, dtype=np.float32),
            confidence=confidence,
            dynamic_probability=np.zeros((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05, confidence_threshold=0.30),
        )
        strict = student_mapping_valid_mask(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_sigma_m=np.full((2, 2), 0.1, dtype=np.float32),
            confidence=confidence,
            dynamic_probability=np.zeros((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05, confidence_threshold=0.70),
        )

        self.assertEqual(int(np.count_nonzero(loose.valid_mask)), 3)
        self.assertEqual(int(np.count_nonzero(strict.valid_mask)), 1)

    def test_all_positive_policy_is_marked_unsafe(self) -> None:
        result = student_mapping_valid_mask(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_sigma_m=np.full((2, 2), 10.0, dtype=np.float32),
            confidence=np.zeros((2, 2), dtype=np.float32),
            dynamic_probability=np.ones((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05, policy="all_positive"),
        )

        self.assertEqual(int(np.count_nonzero(result.valid_mask)), 4)
        self.assertTrue(result.stats["student_map_valid_policy_unsafe"])
        self.assertIn("unsafe", str(result.stats["student_map_valid_policy_warning"]))

    def test_zero_nan_and_negative_depth_are_rejected_and_sanitized(self) -> None:
        depth = np.asarray([[1.0, 0.0], [np.nan, -1.0]], dtype=np.float32)
        result = student_mapping_valid_mask(
            depth_m=depth,
            depth_sigma_m=np.full((2, 2), 0.1, dtype=np.float32),
            confidence=np.ones((2, 2), dtype=np.float32),
            dynamic_probability=np.zeros((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05),
        )
        sanitized = sanitize_student_depth(depth)

        self.assertEqual(int(np.count_nonzero(result.valid_mask)), 1)
        self.assertTrue(np.all(np.isfinite(sanitized)))
        self.assertTrue(np.all(sanitized >= 0.0))

    def test_high_sigma_rejected_when_threshold_provided(self) -> None:
        result = student_mapping_valid_mask(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_sigma_m=np.asarray([[0.1, 0.9], [1.1, 2.0]], dtype=np.float32),
            confidence=np.ones((2, 2), dtype=np.float32),
            dynamic_probability=np.zeros((2, 2), dtype=np.float32),
            config=StudentMapGateConfig(
                min_depth_m=0.05,
                max_sigma_m=1.0,
                policy="confidence_sigma",
            ),
        )

        self.assertEqual(int(np.count_nonzero(result.valid_mask)), 2)
        self.assertEqual(result.stats["sigma_gated_valid_pixel_count"], 2)

    def test_high_dynamic_probability_is_rejected(self) -> None:
        result = student_mapping_valid_mask(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_sigma_m=np.full((2, 2), 0.1, dtype=np.float32),
            confidence=np.ones((2, 2), dtype=np.float32),
            dynamic_probability=np.asarray([[0.0, 0.49], [0.5, 0.9]], dtype=np.float32),
            config=StudentMapGateConfig(min_depth_m=0.05, dynamic_threshold=0.50),
        )

        self.assertEqual(int(np.count_nonzero(result.valid_mask)), 2)
        self.assertEqual(result.stats["dynamic_rejected_pixel_count"], 2)

    def test_default_policy_uses_sigma_gate_only_when_sigma_threshold_exists(self) -> None:
        self.assertEqual(resolve_student_map_valid_policy(None, None), "confidence")
        self.assertEqual(resolve_student_map_valid_policy(None, 1.0), "confidence_sigma")


if __name__ == "__main__":
    unittest.main()
