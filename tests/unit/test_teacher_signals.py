import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.forge.clip_cache import (
    CLIP_CACHE_FORMAT_NAME,
    CLIP_CACHE_FORMAT_VERSION,
    write_clip_payload,
    write_json_file,
)
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
    TEACHER_SIGNAL_FORMAT_NAME,
    TEACHER_SIGNAL_FORMAT_VERSION,
    load_teacher_signal_manifest,
    read_teacher_signal_payload,
    validate_teacher_signal_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TeacherSignalTest(unittest.TestCase):
    def test_measured_forge_writes_valid_signal_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            output = root / "teacher_cache"

            result = forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(
                    clip_cache=clip_manifest,
                    output=output,
                    sigma_m=0.02,
                    max_clips=1,
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)
            payload = read_teacher_signal_payload(output)

        self.assertEqual(result["signal_count"], 1)
        self.assertEqual(manifest["format_name"], TEACHER_SIGNAL_FORMAT_NAME)
        self.assertTrue(manifest["truth_boundary"]["measured_geometry"])  # type: ignore[index]
        self.assertFalse(manifest["truth_boundary"]["pseudo_label"])  # type: ignore[index]
        source_metadata = manifest["source_metadata"]  # type: ignore[index]
        self.assertEqual(source_metadata["pose_source_type"], "measured_tum_groundtruth")  # type: ignore[index]
        self.assertEqual(source_metadata["pose_confidence"], 1.0)  # type: ignore[index]
        self.assertEqual(payload["depth_sigma_m"][0, 0, 0], 0.0)
        self.assertEqual(payload["confidence"][0, 0, 0], 0.0)
        self.assertAlmostEqual(float(payload["depth_sigma_m"][0, 0, 1]), 0.02)

    def test_manifest_rejects_payload_path_traversal(self) -> None:
        manifest = _valid_teacher_manifest()
        manifest["signals"][0]["payload_path"] = "../bad.npz"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "unsafe path"):
            validate_teacher_signal_manifest(
                manifest,
                cache_root=".",
                validate_payloads=False,
                source_clip_manifest=None,
            )

    def test_manifest_rejects_bad_pose_source_metadata_when_present(self) -> None:
        manifest = _valid_teacher_manifest()
        manifest["source_metadata"] = {"pose_source": "fixture", "pose_confidence": 1.5}

        with self.assertRaisesRegex(ValueError, "pose_confidence"):
            validate_teacher_signal_manifest(
                manifest,
                cache_root=".",
                validate_payloads=False,
                source_clip_manifest=None,
            )

    def test_local_raw_ingest_maps_payload_by_source_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache", clip_count=2)
            source_teacher = root / "source_teacher"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=source_teacher)
            )
            raw_input = root / "raw_input"
            raw_input.mkdir()
            shutil.copyfile(
                source_teacher / "signals" / "clip_000001.npz",
                raw_input / "clip_000001.npz",
            )

            output = root / "ingested"
            result = ingest_local_teacher_signal_cache(
                LocalTeacherIngestConfig(
                    clip_cache=clip_manifest,
                    input=raw_input,
                    output=output,
                    teacher_name="external_fixture_depth",
                    teacher_version="fixture-v1",
                    pseudo_label=True,
                )
            )
            manifest = load_teacher_signal_manifest(output, validate_payloads=True)

        self.assertEqual(result["teacher_name"], "external_fixture_depth")
        self.assertEqual(result["signal_count"], 1)
        self.assertEqual(manifest["teacher_source_type"], "local_folder")
        self.assertEqual(manifest["signals"][0]["source_clip_id"], 1)  # type: ignore[index]
        self.assertEqual(manifest["signals"][0]["frame_ids"], [20, 21])  # type: ignore[index]
        self.assertTrue(manifest["truth_boundary"]["pseudo_label"])  # type: ignore[index]

    def test_local_raw_ingest_rejects_bad_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            source_teacher = root / "source_teacher"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=source_teacher)
            )
            raw_input = root / "raw_input"
            raw_input.mkdir()
            shutil.copyfile(
                source_teacher / "signals" / "clip_000000.npz",
                raw_input / "bad_name.npz",
            )

            with self.assertRaisesRegex(ValueError, "raw NPZ filename"):
                ingest_local_teacher_signal_cache(
                    LocalTeacherIngestConfig(
                        clip_cache=clip_manifest,
                        input=raw_input,
                        output=root / "ingested",
                        teacher_name="external_fixture_depth",
                        teacher_version="fixture-v1",
                        pseudo_label=True,
                    )
                )

    def test_local_raw_ingest_rejects_duplicate_source_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            source_teacher = root / "source_teacher"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=source_teacher)
            )
            raw_input = root / "raw_input"
            raw_input.mkdir()
            source_payload = source_teacher / "signals" / "clip_000000.npz"
            shutil.copyfile(source_payload, raw_input / "clip_000000.npz")
            shutil.copyfile(source_payload, raw_input / "source_clip_000000.npz")

            with self.assertRaisesRegex(ValueError, "duplicate source clip id 0"):
                ingest_local_teacher_signal_cache(
                    LocalTeacherIngestConfig(
                        clip_cache=clip_manifest,
                        input=raw_input,
                        output=root / "ingested",
                        teacher_name="external_fixture_depth",
                        teacher_version="fixture-v1",
                        pseudo_label=True,
                    )
                )

    def test_local_raw_ingest_rejects_out_of_range_source_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            source_teacher = root / "source_teacher"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=source_teacher)
            )
            raw_input = root / "raw_input"
            raw_input.mkdir()
            shutil.copyfile(
                source_teacher / "signals" / "clip_000000.npz",
                raw_input / "clip_000001.npz",
            )

            with self.assertRaisesRegex(ValueError, "outside the source clip-cache range"):
                ingest_local_teacher_signal_cache(
                    LocalTeacherIngestConfig(
                        clip_cache=clip_manifest,
                        input=raw_input,
                        output=root / "ingested",
                        teacher_name="external_fixture_depth",
                        teacher_version="fixture-v1",
                        pseudo_label=True,
                    )
                )

    def test_local_raw_ingest_rejects_payload_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache", clip_count=2)
            source_teacher = root / "source_teacher"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=source_teacher)
            )
            raw_input = root / "raw_input"
            raw_input.mkdir()
            shutil.copyfile(
                source_teacher / "signals" / "clip_000000.npz",
                raw_input / "clip_000001.npz",
            )

            with self.assertRaisesRegex(ValueError, "frame_ids: must match payload"):
                ingest_local_teacher_signal_cache(
                    LocalTeacherIngestConfig(
                        clip_cache=clip_manifest,
                        input=raw_input,
                        output=root / "ingested",
                        teacher_name="external_fixture_depth",
                        teacher_version="fixture-v1",
                        pseudo_label=True,
                    )
                )

    def test_manifest_relative_source_clip_path_resolves_from_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            teacher_cache = root / "teacher_cache"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=teacher_cache)
            )
            manifest = load_teacher_signal_manifest(teacher_cache, validate_payloads=True)
            manifest["source_clip_cache_manifest_path"] = (
                "../clip_cache/atlas3r_clip_cache_manifest.json"
            )
            write_json_file(teacher_cache / "atlas3r_teacher_signal_manifest.json", manifest)
            other_cwd = root / "other_cwd"
            other_cwd.mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                loaded = load_teacher_signal_manifest(teacher_cache, validate_payloads=True)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(loaded["signal_count"], 1)

    def test_manifest_rejects_non_dict_signal_entries(self) -> None:
        manifest = _valid_teacher_manifest()
        manifest["signals"] = ["not-a-mapping"]

        with self.assertRaisesRegex(ValueError, r"manifest\.signals\[0\]: must be a mapping"):
            validate_teacher_signal_manifest(
                manifest,
                cache_root=".",
                validate_payloads=False,
                source_clip_manifest=None,
            )

    def test_inspect_signals_reports_zero_measured_depth_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache")
            teacher_cache = root / "teacher_cache"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=teacher_cache)
            )
            output = root / "inspection"

            result = inspect_teacher_signals(
                TeacherSignalInspectConfig(
                    clip_cache=clip_manifest,
                    teacher_cache=teacher_cache,
                    output=output,
                    max_clips=1,
                )
            )
            summary = result["summary"]
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "per_clip_metrics.jsonl").is_file())
            self.assertFalse((output / "preview.html").exists())
            self.assertFalse((output / "preview.svg").exists())
            self.assertNotIn("preview_html_path", result)
            self.assertNotIn("preview_svg_path", result)

        aggregate = summary["aggregate"]  # type: ignore[index]
        self.assertLessEqual(float(aggregate["depth_rmse_m"]), 1e-8)  # type: ignore[index]
        self.assertEqual(float(aggregate["within_1mm_percent"]), 100.0)  # type: ignore[index]

    def test_map_signals_writes_tsdf_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_manifest = _write_clip_cache(root / "clip_cache", clip_count=2, overlap=True)
            teacher_cache = root / "teacher_cache"
            forge_measured_tum_teacher_signal_cache(
                MeasuredTumTeacherForgeConfig(clip_cache=clip_manifest, output=teacher_cache)
            )
            output = root / "map"

            result = map_teacher_signals(
                TeacherSignalMapConfig(
                    teacher_cache=teacher_cache,
                    output=output,
                    max_clips=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                )
            )
            summary = result["map_summary"]
            self.assertTrue((output / "teacher_tsdf" / "tsdf_grid.npz").is_file())
            self.assertTrue((output / "teacher_tsdf" / "surface_points.npz").is_file())
            self.assertTrue((output / "map_summary.json").is_file())

        self.assertGreater(int(summary["surface_point_count"]), 0)  # type: ignore[index]
        self.assertEqual(summary["voxel_size_m"], 0.25)  # type: ignore[index]
        self.assertEqual(summary["observation_count_before_dedupe"], 4)  # type: ignore[index]
        self.assertEqual(summary["observation_count_after_dedupe"], 3)  # type: ignore[index]
        self.assertEqual(summary["duplicate_frame_count"], 1)  # type: ignore[index]
        self.assertEqual(summary["source_frame_ids"], [10, 11, 12])  # type: ignore[index]
        self.assertIn("first occurrence wins", summary["dedupe_policy"])  # type: ignore[index]

    def test_teachers_cli_help_works(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        commands = [
            ["teachers", "forge-measured-tum", "--help"],
            ["teachers", "ingest-local", "--help"],
            ["teachers", "inspect-signals", "--help"],
            ["teachers", "map-signals", "--help"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "atlas3r", *command],
                    check=False,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)

    def test_status_handoff_points_to_phase6d(self) -> None:
        text = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        self.assertIn("Phase 6D - Live-Ready Incremental Mapper Scheduler", text)
        self.assertIn("bounded memory", text)
        self.assertIn("phase6c_true_incremental_tsdf_backend_report.md", text)


def _write_clip_cache(root: Path, *, clip_count: int = 1, overlap: bool = False) -> Path:
    root.mkdir(parents=True)
    clips_dir = root / "clips"
    clips_dir.mkdir()
    for clip_id in range(clip_count):
        frame_ids, timestamps = _clip_frame_metadata(clip_id, overlap=overlap)
        payload = _clip_payload(frame_ids=frame_ids, timestamps=timestamps)
        write_clip_payload(
            clips_dir / f"clip_{clip_id:06d}.npz",
            payload,
            clip_length=2,
            height=4,
            width=4,
        )
    manifest = _clip_manifest(root, clip_count=clip_count, overlap=overlap)
    write_json_file(root / "atlas3r_clip_cache_manifest.json", manifest)
    return root / "atlas3r_clip_cache_manifest.json"


def _clip_manifest(root: Path, *, clip_count: int, overlap: bool) -> dict[str, object]:
    clips: list[dict[str, object]] = []
    for clip_id in range(clip_count):
        frame_ids, timestamps = _clip_frame_metadata(clip_id, overlap=overlap)
        clips.append(
            {
                "clip_id": clip_id,
                "payload_path": f"clips/clip_{clip_id:06d}.npz",
                "frame_ids": list(frame_ids),
                "timestamps_s": list(timestamps),
                "center_index": 1,
                "center_frame_id": frame_ids[1],
            }
        )
    return {
        "format_name": CLIP_CACHE_FORMAT_NAME,
        "format_version": CLIP_CACHE_FORMAT_VERSION,
        "source_dataset_name": "fixture",
        "source_sequence_name": "teacher_signal_fixture",
        "source_manifest_path": str(root / "source_manifest.json"),
        "split": "val",
        "clip_length": 2,
        "stride": 1,
        "image_width": 4,
        "image_height": 4,
        "max_frame_gap_s": 0.1,
        "clip_count": clip_count,
        "clips": clips,
        "teacher_source_metadata": {"fixture": True},
        "truth_boundary": {
            "diagnostic_only": True,
            "accuracy_report": False,
            "performance_report": False,
            "teacher_source": "tum_rgbd_sensor_depth_pose",
        },
    }


def _clip_frame_metadata(
    clip_id: int,
    *,
    overlap: bool,
) -> tuple[tuple[int, int], tuple[float, float]]:
    if overlap:
        frame_start = 10 + clip_id
        time_start = 1.0 + clip_id * 0.033
    else:
        frame_start = 10 + clip_id * 10
        time_start = 1.0 + clip_id
    return (frame_start, frame_start + 1), (time_start, time_start + 0.033)


def _clip_payload(
    *,
    frame_ids: tuple[int, int] = (10, 11),
    timestamps: tuple[float, float] = (1.0, 1.033),
) -> dict[str, np.ndarray]:
    K = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    depth = np.ones((2, 4, 4), dtype=np.float32)
    valid = np.ones((2, 4, 4), dtype=np.bool_)
    if 10 in frame_ids:
        invalid_index = frame_ids.index(10)
        depth[invalid_index, 0, 0] = 0.0
        valid[invalid_index, 0, 0] = False
    transforms = np.repeat(np.eye(4, dtype=np.float32)[np.newaxis, :, :], 2, axis=0)
    return {
        "images_rgb_u8": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "depth_m": depth,
        "valid_depth_mask": valid,
        "K": np.repeat(K[np.newaxis, :, :], 2, axis=0),
        "T_world_camera": transforms,
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "timestamps_s": np.array(timestamps, dtype=np.float64),
        "center_index": np.array(1, dtype=np.int32),
    }


def _valid_teacher_manifest() -> dict[str, object]:
    return {
        "format_name": TEACHER_SIGNAL_FORMAT_NAME,
        "format_version": TEACHER_SIGNAL_FORMAT_VERSION,
        "source_clip_cache_manifest_path": "clip.json",
        "source_dataset_name": "fixture",
        "source_sequence_name": "fixture",
        "split": "val",
        "clip_length": 2,
        "image_width": 4,
        "image_height": 4,
        "teacher_name": "fixture_teacher",
        "teacher_version": "1",
        "teacher_source_type": "fixture",
        "signal_count": 1,
        "signals": [
            {
                "signal_id": 0,
                "source_clip_id": 0,
                "payload_path": "signals/clip_000000.npz",
                "frame_ids": [10, 11],
                "timestamps_s": [1.0, 1.033],
            }
        ],
        "truth_boundary": {
            "diagnostic_only": True,
            "accuracy_report": False,
            "performance_report": False,
            "teacher_source": "fixture_teacher",
            "measured_geometry": False,
            "pseudo_label": True,
        },
    }


if __name__ == "__main__":
    unittest.main()
