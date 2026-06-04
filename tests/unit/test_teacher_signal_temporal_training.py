import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.forge.clip_cache import write_json_file
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
from atlas3r.teachers.signals import (
    load_teacher_signal_manifest,
    read_teacher_signal_payload,
    write_teacher_signal_payload,
)
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
        self.assertEqual(
            tuple(sample["target"]["relative_rotation_6d_center_from_camera"].shape),
            (2, 6),
        )
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

    def test_dataset_accepts_multi_sequence_caches_and_summarizes_counts(self) -> None:
        self._require_torch()
        from atlas3r.training.teacher_signal_dataset import TeacherSignalTemporalDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_a = _write_test_clip_cache(root / "clip_a")
            clip_b = _write_test_clip_cache(root / "clip_b")
            _rewrite_clip_cache_identity(
                clip_b,
                dataset_name="TUM RGB-D fixture",
                sequence_name="freiburg2_fixture",
            )
            teacher_a = root / "teacher_a"
            teacher_b = root / "teacher_b"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_a, output=teacher_a)
            )
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_b, output=teacher_b)
            )

            dataset = TeacherSignalTemporalDataset([teacher_a, teacher_b])
            summary = dataset.cache_summary()
            sample_b = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            summary["source_sequence_names"],
            ["teacher_signal_fixture", "freiburg2_fixture"],
        )
        self.assertEqual(len(summary["cache_records"]), 2)
        self.assertEqual(len(summary["sequence_records"]), 2)
        self.assertEqual(sample_b["metadata"]["source_sequence_name"], "freiburg2_fixture")
        self.assertEqual(summary["record_count"], 2)

    def test_dataset_collates_mixed_optional_pointmaps(self) -> None:
        self._require_torch()
        import torch

        from atlas3r.training.teacher_signal_dataset import TeacherSignalTemporalDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_test_clip_cache(root / "clip_cache")
            measured_cache = root / "measured"
            pseudo_cache = root / "pseudo"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=measured_cache)
            )
            measured_payload = read_teacher_signal_payload(measured_cache)
            measured_payload["pointmap_camera_m"] = np.zeros((2, 4, 4, 3), dtype=np.float32)
            write_teacher_signal_payload(
                measured_cache / "signals" / "clip_000000.npz",
                measured_payload,
                clip_length=2,
                height=4,
                width=4,
            )
            raw_input = root / "raw"
            raw_input.mkdir()
            pseudo_payload = {
                key: value for key, value in measured_payload.items() if key != "pointmap_camera_m"
            }
            np.savez_compressed(raw_input / "clip_000000.npz", **pseudo_payload)
            ingest_local_teacher_signal_cache(
                LocalTeacherIngestConfig(
                    clip_cache=clip_manifest,
                    input=raw_input,
                    output=pseudo_cache,
                    teacher_name="pseudo_without_pointmap",
                    teacher_version="fixture-v1",
                    pseudo_label=True,
                )
            )
            dataset = TeacherSignalTemporalDataset([measured_cache, pseudo_cache])
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))

        self.assertEqual(tuple(batch["target"]["pointmap_camera_m"].shape), (2, 2, 3, 4, 4))
        self.assertEqual(batch["target"]["pointmap_camera_valid"].tolist(), [True, False])

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
            "relative_rotation_6d_center_from_camera": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
            ),
        }
        target = {
            "depth_m": torch.ones((1, 1, 1, 1, 2), dtype=torch.float32),
            "depth_sigma_m": torch.tensor([[[[[0.01, 0.5]]]]], dtype=torch.float32),
            "confidence": torch.tensor([[[[[1.0, 1.0]]]]], dtype=torch.float32),
            "valid_mask": torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
            "teacher_is_measured": torch.tensor([True]),
            "relative_translation_center_from_camera": torch.zeros((1, 1, 3)),
            "relative_rotation_6d_center_from_camera": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
            ),
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
        self.assertEqual(metrics["relative_rotation_mean_deg"], 0.0)

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
        self.assertEqual(
            tuple(prediction["relative_rotation_6d_center_from_camera"].shape),
            (2, 3, 6),
        )
        for value in prediction.values():
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_temporal_v1_cuda_amp_step_updates_weights(self) -> None:
        self._require_torch()
        import torch

        if not bool(torch.cuda.is_available()):
            self.skipTest("CUDA is not available")
        from atlas3r.training.teacher_signal_losses import (
            TeacherSignalLossConfig,
            teacher_signal_temporal_loss,
        )
        from atlas3r.training.tiny_temporal_geometry_model import TemporalMetricNetV1

        device = "cuda"
        model = TemporalMetricNetV1(hidden_channels=4, bottleneck_channels=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        images = torch.zeros((1, 2, 3, 8, 8), dtype=torch.float32, device=device)
        intrinsics = (
            torch.eye(3, dtype=torch.float32, device=device)
            .view(1, 1, 3, 3)
            .repeat(
                1,
                2,
                1,
                1,
            )
        )
        intrinsics[:, :, 0, 0] = 8.0
        intrinsics[:, :, 1, 1] = 8.0
        target = {
            "depth_m": torch.ones((1, 2, 1, 8, 8), dtype=torch.float32, device=device),
            "depth_sigma_m": torch.full((1, 2, 1, 8, 8), 0.05, device=device),
            "confidence": torch.ones((1, 2, 1, 8, 8), dtype=torch.float32, device=device),
            "valid_mask": torch.ones((1, 2, 1, 8, 8), dtype=torch.bool, device=device),
            "teacher_is_measured": torch.tensor([True], device=device),
            "relative_translation_center_from_camera": torch.zeros(
                (1, 2, 3),
                dtype=torch.float32,
                device=device,
            ),
            "relative_rotation_6d_center_from_camera": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
                dtype=torch.float32,
                device=device,
            ),
            "intrinsics": intrinsics,
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(images, intrinsics)
            loss, _metrics = teacher_signal_temporal_loss(
                prediction,
                target,
                config=TeacherSignalLossConfig(
                    relative_translation_weight=1.0,
                    relative_rotation_weight=0.1,
                    se3_pose_weight=1.0,
                ),
            )
        before = model.depth_head.weight.detach().clone()
        scaler.scale(loss).backward()
        gradients = [param.grad for param in model.parameters() if param.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        scaler.step(optimizer)
        scaler.update()

        delta = (model.depth_head.weight.detach() - before).abs().sum()
        self.assertGreater(float(delta.cpu()), 0.0)

    def test_temporal_v1_loader_accepts_old_checkpoint_without_rotation_head(self) -> None:
        self._require_torch()
        import torch

        from atlas3r.training.teacher_signal_temporal_artifacts import (
            load_teacher_signal_temporal_checkpoint,
        )
        from atlas3r.training.tiny_temporal_geometry_model import (
            TemporalMetricNetV1,
            temporal_v1_truth_boundary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_checkpoint.pt"
            model = TemporalMetricNetV1(hidden_channels=4, bottleneck_channels=6)
            state = {
                key: value
                for key, value in model.state_dict().items()
                if not key.startswith("rotation_head.")
            }
            torch.save(
                {
                    "format_name": "atlas3r_teacher_signal_temporal_checkpoint",
                    "format_version": 1,
                    "step": 1,
                    "model_state_dict": state,
                    "optimizer_state_dict": {},
                    "config": {},
                    "metrics": {},
                    "validation_metrics": {},
                    "model_config": {
                        "model_name": "TemporalMetricNetV1",
                        "hidden_channels": 4,
                        "bottleneck_channels": 6,
                    },
                    "loss_config": {},
                    "teacher_caches": [],
                    "val_teacher_caches": [],
                    "truth_boundary": temporal_v1_truth_boundary(),
                },
                path,
            )

            loaded = load_teacher_signal_temporal_checkpoint(path, device="cpu")

        self.assertFalse(bool(loaded["has_trained_rotation_head"]))

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
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            validation_lines = [
                json.loads(line)
                for line in (run_dir / "validation_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertIn("teacher_signal_fixture", summary["best_validation_per_sequence"])
            self.assertIn("per_sequence", validation_lines[-1])

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


def _rewrite_clip_cache_identity(
    manifest_path: Path,
    *,
    dataset_name: str,
    sequence_name: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dataset_name"] = dataset_name
    manifest["source_sequence_name"] = sequence_name
    write_json_file(manifest_path, manifest)


if __name__ == "__main__":
    unittest.main()
