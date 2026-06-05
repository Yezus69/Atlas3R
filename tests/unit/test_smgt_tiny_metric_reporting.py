import unittest

from atlas3r.training.smgt_tiny_metric_reporting import summarize_smgt_tiny_training_metrics


class SMGTTinyMetricReportingTest(unittest.TestCase):
    def test_negative_loss_crossing_does_not_report_misleading_percent(self) -> None:
        history = [
            _metrics(loss_total=1.0, depth_absrel=0.4, depth_rmse_m=0.8, pose_center=0.5),
        ] * 100 + [
            _metrics(loss_total=-0.5, depth_absrel=0.2, depth_rmse_m=0.4, pose_center=0.2),
        ] * 100

        summary = summarize_smgt_tiny_training_metrics(history)

        self.assertTrue(summary["loss_total_crossed_zero"])
        self.assertFalse(summary["loss_total_percent_meaningful"])
        self.assertIsNone(summary["loss_total_decrease_percent"])
        self.assertEqual(summary["loss_total_delta"], -1.5)
        self.assertEqual(summary["loss_total_reduction_absolute"], 1.5)
        self.assertTrue(summary["training_quality_pass"])

    def test_first_final_metric_windows_are_computed(self) -> None:
        history = [
            _metrics(
                loss_total=float(index),
                depth_absrel=float(index),
                depth_rmse_m=2.0,
                pose_center=4.0,
            )
            for index in range(101)
        ]

        summary = summarize_smgt_tiny_training_metrics(history)
        first = summary["first_100"]  # type: ignore[index]
        final = summary["final_100"]  # type: ignore[index]

        self.assertEqual(summary["window_size"], 100)
        self.assertAlmostEqual(first["loss_total"], 49.5)  # type: ignore[index]
        self.assertAlmostEqual(final["loss_total"], 50.5)  # type: ignore[index]
        self.assertAlmostEqual(first["depth_absrel"], 49.5)  # type: ignore[index]
        self.assertAlmostEqual(final["depth_absrel"], 50.5)  # type: ignore[index]

    def test_positive_same_sign_loss_can_report_percent(self) -> None:
        summary = summarize_smgt_tiny_training_metrics(
            [
                _metrics(loss_total=2.0, depth_absrel=0.4, depth_rmse_m=0.8, pose_center=0.5),
            ]
            * 100
            + [
                _metrics(loss_total=1.0, depth_absrel=0.2, depth_rmse_m=0.4, pose_center=0.2),
            ]
            * 100
        )

        self.assertTrue(summary["loss_total_percent_meaningful"])
        self.assertEqual(summary["loss_total_decrease_percent"], 50.0)


def _metrics(
    *,
    loss_total: float,
    depth_absrel: float,
    depth_rmse_m: float,
    pose_center: float,
) -> dict[str, float]:
    return {
        "loss_total": loss_total,
        "depth_absrel": depth_absrel,
        "depth_rmse_m": depth_rmse_m,
        "pose_center_mean_m": pose_center,
        "pose_relative_translation_mean_m": pose_center,
        "pose_rotation_mean_deg": pose_center,
        "confidence_brier": depth_absrel,
    }


if __name__ == "__main__":
    unittest.main()
