import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.runtime import (
    RUNTIME_EVENTS_FILENAME,
    RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME,
    RUNTIME_SUMMARY_FILENAME,
    RuntimeStage,
    format_runtime_fixture_inspection,
    runtime_fixture_inspection_record,
    write_runtime_fixture_smoke,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class RuntimeFixtureInspectionTest(unittest.TestCase):
    def test_runtime_fixture_inspection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_output = root / "runtime_a"
            second_output = root / "runtime_b"
            write_runtime_fixture_smoke(first_output)
            write_runtime_fixture_smoke(second_output)

            first = format_runtime_fixture_inspection(first_output)
            second = format_runtime_fixture_inspection(second_output)
            inspection = json.loads(first)

            self.assertEqual(first, second)
            self.assertEqual(inspection["format_name"], RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME)
            self.assertTrue(inspection["cross_checks"]["passed"])
            self.assertTrue(
                inspection["cross_checks"]["teacher_cache_tsdf_complete_inspection_passed"]
            )
            self.assertEqual(inspection["summary"]["frame_ids"], [0, 1, 2])
            self.assertEqual(inspection["teacher_cache"]["adapter"], "fixture-cube-room")
            self.assertTrue(inspection["teacher_cache"]["arrays_stored"])
            self.assertEqual(inspection["teacher_cache"]["array_payload_count"], 3)
            self.assertEqual(
                [item["path"] for item in inspection["teacher_cache"]["array_payloads"]],
                [
                    "teacher_cache/arrays/frame_000000.npz",
                    "teacher_cache/arrays/frame_000001.npz",
                    "teacher_cache/arrays/frame_000002.npz",
                ],
            )
            self.assertEqual(inspection["teacher_cache_tsdf"]["inspect_mode"], "complete")
            self.assertTrue(inspection["teacher_cache_tsdf"]["cross_checks"]["passed"])
            self.assertFalse(inspection["diagnostic_boundary"]["performance_report"])
            self.assertFalse(inspection["diagnostic_boundary"]["accuracy_report"])
            self.assertIn("diagnostic only", inspection["diagnostic_boundary"]["note"])
            self.assertIn("not a performance report", inspection["diagnostic_boundary"]["note"])
            self.assertIn("not an accuracy report", inspection["diagnostic_boundary"]["note"])

            event_log = inspection["event_log"]
            self.assertEqual(event_log["dropped_frame_count"], 0)
            self.assertEqual(event_log["bounded_memory"]["peak_frame_arrays_in_memory"], 1)
            self.assertEqual(
                event_log["stage_counts"][RuntimeStage.TSDF_REPLAY_FRAME.value],
                3,
            )
            self.assertIn(
                "teacher_cache/arrays/frame_000000.npz",
                event_log["relative_paths_checked"],
            )

    def test_runtime_fixture_inspection_reports_malformed_event_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "runtime"
            write_runtime_fixture_smoke(output)
            events_path = output / RUNTIME_EVENTS_FILENAME
            records = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            for record in records:
                if record["stage_name"] == RuntimeStage.ADAPTER_CACHE_FRAME.value:
                    record["paths"]["array_payload"] = str(
                        output / "teacher_cache" / "arrays" / "frame_000000.npz"
                    )
                    break
            events_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"runtime_events\.jsonl.*paths\.array_payload.*relative",
            ):
                runtime_fixture_inspection_record(output)

    def test_runtime_fixture_inspection_cross_checks_summary_event_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "runtime"
            write_runtime_fixture_smoke(output)
            summary_path = output / RUNTIME_SUMMARY_FILENAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["event_log"]["event_count"] = 1
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"runtime_summary\.json.*event_log\.event_count",
            ):
                runtime_fixture_inspection_record(output)

    def test_cli_inspect_runtime_fixture_outputs_json_and_reports_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "runtime"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            write_runtime_fixture_smoke(output)

            command = [
                sys.executable,
                "-m",
                "atlas3r",
                "inspect",
                "runtime-fixture",
                "--input",
                str(output),
            ]
            first = subprocess.run(
                command,
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            second = subprocess.run(
                command,
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            inspection = json.loads(first.stdout)
            self.assertEqual(inspection["format_name"], RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME)
            self.assertNotIn("Traceback", first.stderr)

            (output / RUNTIME_SUMMARY_FILENAME).unlink()
            failure = subprocess.run(
                command,
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(failure.returncode, 2)
            self.assertIn(RUNTIME_SUMMARY_FILENAME, failure.stderr)
            self.assertNotIn("Traceback", failure.stderr)


if __name__ == "__main__":
    unittest.main()
