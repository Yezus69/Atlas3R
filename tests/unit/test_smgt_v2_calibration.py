import unittest

import numpy as np

from atlas3r.training.smgt_v2_eval import select_smgt_v2_gate_thresholds


class SMGTV2CalibrationTest(unittest.TestCase):
    def test_threshold_selection_maps_subset_and_rejects_high_error_pixels(self) -> None:
        good_count = 600
        bad_count = 400
        target = np.ones(good_count + bad_count, dtype=np.float32)
        pred = target.copy()
        pred[:good_count] += 0.02
        pred[good_count:] += 0.80
        confidence = np.concatenate(
            [
                np.full(good_count, 0.90, dtype=np.float32),
                np.full(bad_count, 0.10, dtype=np.float32),
            ]
        )
        sigma = np.concatenate(
            [
                np.full(good_count, 0.05, dtype=np.float32),
                np.full(bad_count, 2.0, dtype=np.float32),
            ]
        )
        valid = np.ones_like(target, dtype=np.bool_)

        calibration = select_smgt_v2_gate_thresholds(
            confidence,
            sigma,
            pred,
            target,
            valid,
            target_min_mapped_ratio=0.10,
            target_max_mapped_ratio=0.70,
        )

        self.assertGreater(calibration["mapped_pixel_ratio"], 0.10)
        self.assertLess(calibration["mapped_pixel_ratio"], 0.70)
        self.assertLess(calibration["mapped_absrel"], calibration["rejected_absrel"])
        self.assertTrue(calibration["mapped_pixels_lower_error_than_rejected"])
        self.assertTrue(calibration["rejects_high_error_better_than_random"])


if __name__ == "__main__":
    unittest.main()
