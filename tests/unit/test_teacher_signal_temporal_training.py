import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.teachers.map_eval import (
    TeacherSignalInspectConfig,
    TeacherSignalMapConfig,
    inspect_teacher_signals,
    map_teacher_signals,
)
from atlas3r.teachers.measured_tum import (
    LocalTeacherIngestConfig,
    MeasuredTumTeacherForgeConfig,
    forge_measured_tum_teacher_signal_cache,
    ingest_local_teacher_signal_cache,
)
from atlas3r.teachers.signals import load_teacher_signal_manifest
from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TeacherSignalTemporalTrainingTest(unittest.TestCase):
    def test_cli_help_includes_teacher_signals_temporal(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "train", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("teacher-signals-temporal", result.stdout)

    def test_dataset_loads_measured_fixture_shapes(self) -> None:
        self._require_torch()
        from atlas3r.training.teacher_signal_dataset import TeacherSignalTemporalDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_test_clip_cache(root / "clip_cache")
            teacher_cache = root / "teacher_cache"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=teacher_cache)
            )
            dataset = TeacherSignalTemporalDataset([teacher_cache])
            sample = dataset[0]

        self.assertEqual(tuple(sample["images_rgb"].shape), (2, 3, 4, 4))
        self.assertEqual(tuple(sample["intrinsics"].shape), (2, 3, 3))
        self.assertEqual(tuple(sample["T_world_camera"].shape), (2, 4, 4))
        self.assertEqual(tuple(sample["target"]["depth_m"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(sample["target"]["depth_sigma_m"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(sample["target"]["confidence"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(sample["target"]["valid_mask"].shape), (2, 1, 4, 4))
        self.assertTrue(bool(sample["target"]["teacher_is_measured"]))
        self.assertEqual(sample["metadata"]["teacher_source_type"], "measured_rgbd_pose")

    def test_dataset_mixed_measured_and_pseudo_preserves_metadata(self) -> None:
        self._require_torch()
        from atlas3r.training.teacher_signal_dataset import TeacherSignalTemporalDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_test_clip_cache(root / "clip_cache")
            measured_cache = root / "measured"
            pseudo_cache = root / "pseudo"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=measured_cache)
            )
            raw_input = root / "raw"
            raw_input.mkdir()
            shutil.copyfile(
                measured_cache / "signals" / "clip_000000.npz",
                raw_input / "clip_000000.npz",
            )
            ingest_local_teacher_signal_cache(
                LocalTeacherIngestConfig(
                    clip_cache=clip_manifest,
                    input=raw_input,
                    output=pseudo_cache,
                    teacher_name="pseudo_fixture",
                    teacher_version="fixture-v1",
                    pseudo_label=True,
                )
            )
            dataset = TeacherSignalTemporalDataset([measured_cache, pseudo_cache])
            measured = dataset[0]
            pseudo = dataset[1]
            summary = dataset.cache_summary()

        self.assertTrue(bool(measured["target"]["teacher_is_measured"]))
        self.assertFalse(bool(pseudo["target"]["teacher_is_measured"]))
        self.assertEqual(pseudo["metadata"]["teacher_name"], "pseudo_fixture")
        self.assertEqual(summary["measured_record_count"], 1)
        self.assertEqual(summary["pseudo_record_count"], 1)

    def test_weighted_loss_uses_confidence_sigma_and_rejects_all_invalid(self) -> None:
        self._require_torch()
        import torch

        from atlas3r.training.teacher_signal_losses import (
            TeacherSignalLossConfig,
            teacher_signal_temporal_loss,
        )

        prediction = {
            "depth_m": torch.tensor([[[[[2.0, 1.0]]]]], dtype=torch.float32),
            "depth_sigma_m": torch.full((1, 1, 1, 1, 2), 0.05),
            "confidence": torch.full((1, 1, 1, 1, 2), 0.8),
            "relative_translation_center_from_camera": torch.zeros((1, 1, 3)),
        }
        target = {
            "depth_m": torch.ones((1, 1, 1, 1, 2), dtype=torch.float32),
            "depth_sigma_m": torch.tensor([[[[[0.01, 0.5]]]]], dtype=torch.float32),
            "confidence": torch.tensor([[[[[1.0, 1.0]]]]], dtype=torch.float32),
            "valid_mask": torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
            "teacher_is_measured": torch.tensor([True]),
            "relative_translation_center_from_camera": torch.zeros((1, 1, 3)),
        }
        cfg = TeacherSignalLossConfig(max_pixel_weight=1_000_000.0)
        high_weight_loss, metrics = teacher_signal_temporal_loss(prediction, target, config=cfg)
        target["depth_sigma_m"] = torch.tensor([[[[[0.5, 0.01]]]]], dtype=torch.float32)
        low_weight_loss, _metrics = teacher_signal_temporal_loss(prediction, target, config=cfg)
        target["valid_mask"] = torch.zeros((1, 1, 1, 1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "no valid teacher pixels"):
            teacher_signal_temporal_loss(prediction, target, config=cfg)
        self.assertGreater(
            float(high_weight_loss.detach().cpu()), float(low_weight_loss.detach().cpu())
        )
        self.assertEqual(metrics["valid_depth_pixels"], 2.0)

    def test_temporal_v1_output_shapes_are_finite(self) -> None:
        self._require_torch()
        import torch

        from atlas3r.training.tiny_temporal_geometry_model import TemporalMetricNetV1

        model = TemporalMetricNetV1(hidden_channels=4, bottleneck_channels=6)
        images = torch.zeros((2, 3, 3, 8, 10), dtype=torch.float32)
        K = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(2, 3, 1, 1)
        K[:, :, 0, 0] = 8.0
        K[:, :, 1, 1] = 8.0
        prediction = model(images, K)

        self.assertEqual(tuple(prediction["depth_m"].shape), (2, 3, 1, 8, 10))
        self.assertEqual(tuple(prediction["depth_sigma_m"].shape), (2, 3, 1, 8, 10))
        self.assertEqual(tuple(prediction["confidence"].shape), (2, 3, 1, 8, 10))
        self.assertEqual(
            tuple(prediction["relative_translation_center_from_camera"].shape),
            (2, 3, 3),
        )
        for value in prediction.values():
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_cpu_train_export_inspect_and_map_smoke(self) -> None:
        self._require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_clip = _write_test_clip_cache(root / "train_clip", clip_count=2)
            val_clip = _write_test_clip_cache(root / "val_clip", clip_count=2, overlap=True)
            train_teacher = root / "teacher_train"
            val_teacher = root / "teacher_val"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=train_clip, output=train_teacher)
            )
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=val_clip, output=val_teacher)
            )
            run_dir = root / "run"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            train_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "train",
                    "teacher-signals-temporal",
                    "--teacher-cache",
                    str(train_teacher),
                    "--val-teacher-cache",
                    str(val_teacher),
                    "--output",
                    str(run_dir),
                    "--model",
                    "temporal-v1",
                    "--steps",
                    "2",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--log-every",
                    "1",
                    "--val-every",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--preview-every",
                    "1",
                    "--hidden-channels",
                    "4",
                    "--bottleneck-channels",
                    "6",
                    "--max-runtime-minutes",
                    "5",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(train_result.returncode, 0, train_result.stderr)
            for filename in (
                "config.json",
                "metrics.jsonl",
                "validation_metrics.jsonl",
                "summary.json",
                "checkpoint_last.pt",
                "checkpoint_best.pt",
                "prediction_sample.npz",
                "prediction_preview.html",
                "prediction_preview.svg",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)

            student_cache = root / "student_teacher"
            export_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "teachers",
                    "run-student-temporal",
                    "--checkpoint",
                    str(run_dir / "checkpoint_best.pt"),
                    "--clip-cache",
                    str(val_clip),
                    "--output",
                    str(student_cache),
                    "--device",
                    "cpu",
                    "--max-clips",
                    "2",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(export_result.returncode, 0, export_result.stderr)
            manifest = load_teacher_signal_manifest(student_cache, validate_payloads=True)
            inspect_result = inspect_teacher_signals(
                TeacherSignalInspectConfig(
                    clip_cache=val_clip,
                    teacher_cache=student_cache,
                    output=root / "inspect",
                    max_clips=2,
                )
            )
            map_result = map_teacher_signals(
                TeacherSignalMapConfig(
                    teacher_cache=student_cache,
                    output=root / "map",
                    max_clips=2,
                    voxel_size_m=0.25,
                )
            )

        truth = manifest["truth_boundary"]  # type: ignore[index]
        self.assertFalse(bool(truth["measured_geometry"]))  # type: ignore[index]
        self.assertTrue(bool(truth["pseudo_label"]))  # type: ignore[index]
        self.assertEqual(manifest["teacher_name"], "atlas3r_temporal_v1_student")
        self.assertEqual(inspect_result["summary"]["selected_signal_count"], 2)  # type: ignore[index]
        self.assertEqual(map_result["map_summary"]["observation_count_before_dedupe"], 4)  # type: ignore[index]
        self.assertEqual(map_result["map_summary"]["observation_count_after_dedupe"], 3)  # type: ignore[index]

    def _require_torch(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")


def _write_test_clip_cache(root: Path, *, clip_count: int = 1, overlap: bool = False) -> Path:
    unit_dir = Path(__file__).resolve().parent
    if str(unit_dir) not in sys.path:
        sys.path.insert(0, str(unit_dir))
    from test_teacher_signals import _write_clip_cache

    return _write_clip_cache(root, clip_count=clip_count, overlap=overlap)


if __name__ == "__main__":
    unittest.main()
