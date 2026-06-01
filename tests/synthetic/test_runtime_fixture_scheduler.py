import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.io.teacher_cache import load_teacher_prediction_cache
from atlas3r.mapping.tsdf_output_inspection import format_tsdf_output_folder_inspection
from atlas3r.runtime import (
    RUNTIME_EVENT_LOG_FORMAT_NAME,
    RUNTIME_EVENTS_FILENAME,
    RUNTIME_SUMMARY_FILENAME,
    RuntimeStage,
    write_runtime_fixture_smoke,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RuntimeFixtureSchedulerTest(unittest.TestCase):
    def test_runtime_fixture_event_log_is_deterministic_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "runtime_a"
            second = root / "runtime_b"

            first_result = write_runtime_fixture_smoke(first)
            second_result = write_runtime_fixture_smoke(second)

            self.assertEqual(
                first_result.event_log_path.read_text(encoding="utf-8"),
                second_result.event_log_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                first_result.summary_path.read_text(encoding="utf-8"),
                second_result.summary_path.read_text(encoding="utf-8"),
            )

            events = _jsonl(first / RUNTIME_EVENTS_FILENAME)
            summary = json.loads((first / RUNTIME_SUMMARY_FILENAME).read_text(encoding="utf-8"))

            self.assertEqual(summary["event_log"]["event_count"], len(events))
            self.assertEqual(summary["bounded_memory"]["configured_frame_array_bound"], 1)
            self.assertEqual(summary["bounded_memory"]["peak_frame_arrays_in_memory"], 1)
            self.assertFalse(summary["bounded_memory"]["all_frame_arrays_accumulated"])
            self.assertTrue(summary["bounded_memory"]["bounded_memory_check_passed"])
            self.assertEqual(summary["frame_ids"], [0, 1, 2])

            for index, event in enumerate(events):
                self.assertEqual(event["format_name"], RUNTIME_EVENT_LOG_FORMAT_NAME)
                self.assertEqual(event["event_index"], index)
                self.assertEqual(event["timestamp_ns"], index * 1_000_000)
                self.assertEqual(event["latency_ns"], 0)
                self.assertFalse(event["dropped_frame"])
                self.assertIn("stage_name", event)
                self.assertIn("frame_id", event)
                self.assertIn("paths", event)
                self.assertIn("metadata", event)
                counters = event["memory_counters"]
                self.assertIsInstance(counters, dict)
                self.assertLessEqual(counters["frame_arrays_in_memory"], 1)
                self.assertLessEqual(counters["peak_frame_arrays_in_memory"], 1)
                self.assertIn("dropped_frame_count", counters)

            stages = [event["stage_name"] for event in events]
            self.assertIn(RuntimeStage.SESSION_WRITE.value, stages)
            self.assertIn(RuntimeStage.ADAPTER_CACHE_FRAME.value, stages)
            self.assertIn(RuntimeStage.TSDF_REPLAY_FRAME.value, stages)
            replay_frame_ids = [
                event["frame_id"]
                for event in events
                if event["stage_name"] == RuntimeStage.TSDF_REPLAY_FRAME.value
            ]
            self.assertEqual(replay_frame_ids, [0, 1, 2])

    def test_runtime_fixture_writes_cache_and_tsdf_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "runtime"

            result = write_runtime_fixture_smoke(output)

            self.assertTrue(result.event_log_path.is_file())
            self.assertTrue(result.summary_path.is_file())
            self.assertTrue((output / "synthetic_cube_room.atlas3r" / "metadata.json").is_file())
            cache = load_teacher_prediction_cache(output / "teacher_cache")
            self.assertTrue(cache.metadata["arrays"]["stored"])
            self.assertTrue((output / "teacher_cache" / "arrays" / "frame_000000.npz").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "tsdf_grid.npz").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "surface_points.npz").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "metadata.json").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "metrics.json").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "mesh_chunk_sidecar.json").is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "world_map_sidecar.json").is_file())

            inspection = json.loads(
                format_tsdf_output_folder_inspection(output / "teacher_cache_tsdf")
            )
            self.assertTrue(inspection["cross_checks"]["passed"])
            self.assertFalse(inspection["truth_boundary"]["accuracy_report"])

    def test_cli_runtime_fixture_smoke_writes_outputs_and_reports_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "runtime"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "runtime-fixture",
                    "--output",
                    str(output),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(RUNTIME_EVENTS_FILENAME, result.stdout)
            self.assertIn(RUNTIME_SUMMARY_FILENAME, result.stdout)
            self.assertTrue((output / RUNTIME_EVENTS_FILENAME).is_file())
            self.assertTrue((output / "teacher_cache_tsdf" / "metadata.json").is_file())

            bad_output = root / "not_a_directory"
            bad_output.write_text("file blocks output folder\n", encoding="utf-8")
            failure = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "runtime-fixture",
                    "--output",
                    str(bad_output),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(failure.returncode, 2)
            self.assertIn(str(bad_output), failure.stderr)
            self.assertIn("directory", failure.stderr)
            self.assertNotIn("Traceback", failure.stderr)


if __name__ == "__main__":
    unittest.main()
