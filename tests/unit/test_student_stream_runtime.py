import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.runtime.student_map_observations import observation_from_prediction
from atlas3r.runtime.student_map_runtime import StudentMapRuntimeConfig, run_stream_student_map
from atlas3r.runtime.student_stream import build_stream_window, load_unique_frame_stream
from atlas3r.teachers.measured_tum import (
    MeasuredTumTeacherForgeConfig,
    forge_measured_tum_teacher_signal_cache,
)
from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class StudentStreamRuntimeTest(unittest.TestCase):
    def test_unique_chronological_stream_dedupes_overlapping_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_manifest = _write_test_clip_cache(
                Path(tmp) / "clip_cache", clip_count=2, overlap=True
            )
            stream = load_unique_frame_stream(clip_manifest)

        self.assertEqual([frame.frame_id for frame in stream], [10, 11, 12])
        self.assertEqual([frame.stream_index for frame in stream], [0, 1, 2])
        self.assertEqual(stream[1].duplicate_source_count, 2)

    def test_window_builder_pads_boundaries_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_manifest = _write_test_clip_cache(
                Path(tmp) / "clip_cache", clip_count=2, overlap=True
            )
            stream = load_unique_frame_stream(clip_manifest)

        first = build_stream_window(stream, stream_index=0, window_size=3)
        last = build_stream_window(stream, stream_index=2, window_size=3)

        self.assertEqual(first.frame_ids, (10, 10, 11))
        self.assertEqual(first.padded_positions, (0,))
        self.assertEqual(last.frame_ids, (11, 12, 12))
        self.assertEqual(last.padded_positions, (2,))
        self.assertEqual(first.padding_policy, "edge_reuse_clamp")

    def test_oracle_depth_observation_preserves_source_pose(self) -> None:
        self._require_torch()
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            clip_manifest = _write_test_clip_cache(
                Path(tmp) / "clip_cache", clip_count=2, overlap=True
            )
            stream = load_unique_frame_stream(clip_manifest)
            window = build_stream_window(stream, stream_index=1, window_size=3)
            prediction = {
                "depth_m": torch.ones((1, 3, 1, 4, 4), dtype=torch.float32),
                "depth_sigma_m": torch.full((1, 3, 1, 4, 4), 0.05, dtype=torch.float32),
                "confidence": torch.full((1, 3, 1, 4, 4), 0.75, dtype=torch.float32),
                "relative_translation_center_from_camera": torch.zeros(
                    (1, 3, 3), dtype=torch.float32
                ),
            }
            observation, _record = observation_from_prediction(
                prediction=prediction,
                window=window,
                pose_mode="oracle",
                checkpoint_path=Path("checkpoint_best.pt"),
                checkpoint=_checkpoint_record(),
            )

        np.testing.assert_allclose(observation.pose.T_world_camera, window.output.T_world_camera)
        self.assertEqual(observation.frame_id, window.output.frame_id)
        self.assertEqual(observation.source, "phase5e_temporal_student_checkpoint")

    def test_runtime_command_help_works(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "atlas3r", "runtime", "stream-student-map", "--help"],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--pose-mode", result.stdout)

    def test_fixture_checkpoint_stream_map_run_writes_reports_and_ply(self) -> None:
        self._require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_test_clip_cache(root / "clip_cache", clip_count=2, overlap=True)
            teacher_cache = root / "teacher_cache"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=teacher_cache)
            )
            checkpoint = _write_test_checkpoint(root / "checkpoint_best.pt")
            output = root / "runtime"

            result = run_stream_student_map(
                StudentMapRuntimeConfig(
                    checkpoint=checkpoint,
                    clip_cache=clip_manifest,
                    teacher_cache=teacher_cache,
                    output=output,
                    device="cpu",
                    max_frames=3,
                    window_size=3,
                    pose_mode="oracle",
                    voxel_size_m=0.25,
                )
            )
            mode_dir = output / "oracle"
            summary = json.loads((mode_dir / "summary.json").read_text(encoding="utf-8"))
            quality = json.loads((mode_dir / "quality_report.json").read_text(encoding="utf-8"))
            latency = json.loads((mode_dir / "latency_report.json").read_text(encoding="utf-8"))
            with np.load(mode_dir / "tsdf" / "surface_points.npz", allow_pickle=False) as surface:
                vertex_count = int(surface["points_world_m"].shape[0])
            ply_lines = (mode_dir / "point_cloud.ply").read_text(encoding="utf-8").splitlines()
            file_exists = {
                relative: (mode_dir / relative).is_file()
                for relative in (
                    "runtime_events.jsonl",
                    "observations_summary.jsonl",
                    "quality_report.json",
                    "per_frame_quality.jsonl",
                    "latency_report.json",
                    "summary.json",
                    "tsdf/metadata.json",
                    "tsdf/metrics.json",
                    "tsdf/mesh_chunk_sidecar.json",
                    "tsdf/world_map_sidecar.json",
                    "point_cloud.ply",
                    "map_preview.html",
                )
            }

        self.assertEqual(result["requested_pose_mode"], "oracle")
        for relative, exists in file_exists.items():
            self.assertTrue(exists, relative)
        self.assertFalse(bool(summary["accuracy_report"]))
        self.assertFalse(bool(summary["performance_report"]))
        self.assertFalse(bool(summary["realtime_claim"]))
        self.assertFalse(bool(summary["mapping_ready"]))
        self.assertFalse(bool(quality["accuracy_report"]))
        self.assertFalse(bool(latency["performance_report"]))
        self.assertIn(f"element vertex {vertex_count}", ply_lines)
        self.assertEqual(ply_lines[0], "ply")
        self.assertEqual(ply_lines[1], "format ascii 1.0")
        self.assertGreater(vertex_count, 0)

    def _require_torch(self) -> None:
        if not torch_available():
            self.skipTest("optional torch dependency is not installed")


def _write_test_clip_cache(root: Path, *, clip_count: int = 1, overlap: bool = False) -> Path:
    unit_dir = Path(__file__).resolve().parent
    if str(unit_dir) not in sys.path:
        sys.path.insert(0, str(unit_dir))
    from test_teacher_signals import _write_clip_cache

    return _write_clip_cache(root, clip_count=clip_count, overlap=overlap)


def _checkpoint_record() -> dict[str, object]:
    return {
        "format_name": "atlas3r_teacher_signal_temporal_checkpoint",
        "step": 1,
        "truth_boundary": {
            "mapping": False,
            "accuracy_report": False,
            "performance_report": False,
        },
    }


def _write_test_checkpoint(path: Path) -> Path:
    import torch

    from atlas3r.training.teacher_signal_losses import TeacherSignalLossConfig
    from atlas3r.training.teacher_signal_temporal_artifacts import (
        write_teacher_signal_temporal_checkpoint,
    )
    from atlas3r.training.tiny_temporal_geometry_model import (
        TemporalMetricNetV1,
        temporal_v1_truth_boundary,
    )

    model = TemporalMetricNetV1(hidden_channels=4, bottleneck_channels=6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.depth_head.bias.fill_(math.log(math.exp(1.0) - 1.0))
        model.sigma_head.bias.fill_(math.log(math.exp(0.05) - 1.0))
        model.confidence_head.bias.fill_(2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    write_teacher_signal_temporal_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_record={
            "teacher_caches": [],
            "val_teacher_caches": [],
            "output": str(path.parent),
            "model": "temporal-v1",
        },
        metrics={},
        validation_metrics={},
        truth_boundary=temporal_v1_truth_boundary(),
        loss_config=TeacherSignalLossConfig(),
    )
    return path


if __name__ == "__main__":
    unittest.main()
