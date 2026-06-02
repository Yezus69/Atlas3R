import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas3r.mapping.checkpoint_tsdf_smoke import write_checkpoint_tsdf_smoke
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.tsdf_output_inspection import tsdf_output_folder_inspection_record
from atlas3r.training.checkpoint_inference import (
    TINY_DEPTH_POSE_OBSERVATION_SOURCE,
    depth_observations_from_tiny_prediction,
    load_tiny_depth_pose_checkpoint,
    predict_tiny_depth_pose_student_clip,
)
from atlas3r.training.synthetic_depth_dataset import (
    generate_synthetic_depth_samples,
    sample_to_student_clip,
)
from atlas3r.training.synthetic_overfit import SyntheticOverfitConfig, run_synthetic_overfit
from atlas3r.training.torch_runtime import TorchDependencyError, torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class CheckpointInferenceBridgeTorchOptionalTest(unittest.TestCase):
    def test_missing_torch_checkpoint_loader_reports_train_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint_last.pt"
            checkpoint_path.write_bytes(b"placeholder")

            with patch("atlas3r.training.torch_runtime.torch_available", return_value=False):
                with self.assertRaisesRegex(TorchDependencyError, "train extra"):
                    load_tiny_depth_pose_checkpoint(checkpoint_path)

    def test_checkpoint_prediction_converts_to_valid_depth_observation(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = _write_tiny_checkpoint(Path(tmp) / "train_run")
            checkpoint = load_tiny_depth_pose_checkpoint(checkpoint_path, device="cpu")
            sample = generate_synthetic_depth_samples(count=1, width=16, height=12, seed=41)[0]

            prediction = predict_tiny_depth_pose_student_clip(
                checkpoint,
                sample_to_student_clip(sample),
            )
            observations = depth_observations_from_tiny_prediction(
                prediction,
                rgb_u8_by_frame_id={sample.frame_id: sample.rgb_u8},
            )

        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertIsInstance(observation, DepthObservation)
        self.assertEqual(observation.source, TINY_DEPTH_POSE_OBSERVATION_SOURCE)
        self.assertEqual(observation.depth_m.shape, sample.depth_m.shape)
        self.assertEqual(observation.depth_sigma_m.shape, sample.depth_m.shape)
        self.assertEqual(observation.confidence.shape, sample.depth_m.shape)
        self.assertTrue(np.all(observation.depth_m >= 0.0))
        self.assertTrue(np.all(observation.depth_sigma_m >= 0.0))
        self.assertEqual(observation.pose.scale_source, "rgb_prior")
        self.assertEqual(
            observation.pose.diagnostics["truth_boundary"],
            checkpoint.truth_boundary,
        )
        self.assertIs(checkpoint.truth_boundary["training_mvp"], True)
        self.assertIs(checkpoint.truth_boundary["synthetic_only"], True)
        self.assertIs(checkpoint.truth_boundary["usable_for_mapping"], False)
        self.assertIs(checkpoint.truth_boundary["accuracy_report"], False)

    def test_synthetic_checkpoint_tsdf_smoke_writes_comparison_artifacts(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = _write_tiny_checkpoint(root / "train_run")
            output = root / "checkpoint_tsdf"

            written_paths = write_checkpoint_tsdf_smoke(
                checkpoint_path,
                output,
                width=16,
                height=12,
                seed=7,
                device="cpu",
            )

            self.assertIn(output / "prediction_preview.html", written_paths)
            self.assertIn(output / "target_tsdf" / "metadata.json", written_paths)
            tsdf_output_folder_inspection_record(output, mode="surface")
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            target_metadata = json.loads(
                (output / "target_tsdf" / "metadata.json").read_text(encoding="utf-8")
            )
            with np.load(output / "surface_points.npz") as surface:
                self.assertTrue(np.all(surface["uncertainty_m"] >= 0.0))

        self.assertEqual(
            metadata["artifact_type"],
            "phase_4b_checkpoint_predicted_cpu_tsdf_surface_points",
        )
        self.assertEqual(metadata["metric_scale_source"], "rgb_prior")
        self.assertIn("input_uncertainty_summary_m", metadata)
        self.assertEqual(metadata["truth_boundary"]["synthetic_only"], True)
        self.assertFalse(metadata["truth_boundary"]["usable_for_mapping"])
        self.assertEqual(
            metrics["metric_family"],
            "phase_4b_checkpoint_synthetic_depth_tsdf_bridge",
        )
        self.assertFalse(metrics["accuracy_report"])
        self.assertIn("depth_metrics_m", metrics)
        self.assertEqual(target_metadata["metric_scale_source"], "known_anchor")

    def test_npz_checkpoint_tsdf_smoke_runs_without_target_metrics(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = _write_tiny_checkpoint(root / "train_run")
            sample = generate_synthetic_depth_samples(count=1, width=16, height=12, seed=43)[0]
            clip_path = root / "clip.npz"
            np.savez(clip_path, rgb_u8=sample.rgb_u8[np.newaxis, ...], K=sample.K)

            write_checkpoint_tsdf_smoke(
                checkpoint_path,
                root / "npz_tsdf",
                input_npz=clip_path,
                device="cpu",
            )
            metrics = json.loads((root / "npz_tsdf" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["metric_family"], "not_evaluated")
        self.assertFalse(metrics["accuracy_report"])
        self.assertIn("do not carry target metric depth", metrics["reason"])

    def test_cli_checkpoint_tsdf_smoke_reports_written_outputs(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = _write_tiny_checkpoint(root / "train_run")
            output = root / "cli_checkpoint_tsdf"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "checkpoint-tsdf",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output",
                    str(output),
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--device",
                    "cpu",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("checkpoint TSDF smoke outputs", result.stdout)
            self.assertIn("prediction_preview.html", result.stdout)
            self.assertTrue((output / "metadata.json").is_file())


def _write_tiny_checkpoint(output: Path) -> Path:
    run_synthetic_overfit(
        SyntheticOverfitConfig(
            output=output,
            steps=1,
            batch_size=1,
            num_samples=2,
            width=16,
            height=12,
            seed=5,
            device="cpu",
            log_every=1,
        )
    )
    return output / "checkpoint_last.pt"


if __name__ == "__main__":
    unittest.main()
