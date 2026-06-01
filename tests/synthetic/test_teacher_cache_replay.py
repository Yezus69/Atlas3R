import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.mapping.mesh_sidecar import MESH_SIDECAR_FILENAME, load_tsdf_surface_mesh_sidecar
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.models.adapters import run_adapter_to_cache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TeacherCacheTSDFReplayTest(unittest.TestCase):
    def test_replay_source_does_not_suppress_observation_type_errors(self) -> None:
        replay_source = (SRC / "atlas3r" / "mapping" / "teacher_cache_replay.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("type: ignore[arg-type]", replay_source)
        self.assertNotIn("_grid_shape_xyz", replay_source)
        self.assertNotIn("_voxel_centers", replay_source)

    def test_fixture_array_cache_replays_into_cpu_tsdf_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            first_output = root / "replay_a"
            second_output = root / "replay_b"
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
                store_arrays=True,
            )

            write_teacher_cache_tsdf_replay(cache, first_output)
            write_teacher_cache_tsdf_replay(cache, second_output)

            self.assertEqual(
                (first_output / "metadata.json").read_text(encoding="utf-8"),
                (second_output / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (first_output / "metrics.json").read_text(encoding="utf-8"),
                (second_output / "metrics.json").read_text(encoding="utf-8"),
            )
            with np.load(first_output / "surface_points.npz") as first_surface:
                with np.load(second_output / "surface_points.npz") as second_surface:
                    np.testing.assert_array_equal(
                        first_surface["points_world_m"],
                        second_surface["points_world_m"],
                    )
                    np.testing.assert_array_equal(
                        first_surface["confidence"],
                        second_surface["confidence"],
                    )
                    np.testing.assert_array_equal(
                        first_surface["uncertainty_m"],
                        second_surface["uncertainty_m"],
                    )
                    self.assertEqual(first_surface["points_world_m"].shape[1], 3)
                    self.assertEqual(
                        first_surface["confidence"].shape[0],
                        first_surface["points_world_m"].shape[0],
                    )
                    self.assertTrue(np.all(first_surface["confidence"] > 0.0))
                    self.assertTrue(np.all(first_surface["uncertainty_m"] >= 0.0))

            metadata = json.loads((first_output / "metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((first_output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["artifact_type"],
                "phase_1d_teacher_cache_cpu_tsdf_surface_points",
            )
            self.assertEqual(metadata["coordinate_frame"], "synthetic_world")
            self.assertEqual(metadata["metric_scale_source"], "known_anchor")
            self.assertEqual(metadata["metric_scale_sources"], ["known_anchor"])
            self.assertEqual(metadata["source_frame_ids"], [0, 1, 2])
            self.assertEqual(metadata["teacher_adapter"], "fixture-cube-room")
            self.assertTrue(metadata["cache_arrays_stored"])
            self.assertIn("input_confidence_summaries", metadata)
            self.assertIn("input_uncertainty_summaries_m", metadata)
            self.assertIn("not an accuracy report", metadata["accuracy_note"])
            self.assertEqual(
                metrics["metric_family"],
                "phase_1d_teacher_cache_replay_synthetic_cube_room_reference",
            )
            self.assertFalse(metrics["accuracy_report"])
            self.assertEqual(metrics["surface_points_inside_room_bounds_ratio"], 1.0)
            self.assertIn("not an accuracy report", " ".join(metrics["known_limitations"]))

    def test_replay_rejects_summaries_only_cache_with_path_named_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
            )

            with self.assertRaisesRegex(ValueError, r"metadata\.json.*arrays\.stored=true"):
                write_teacher_cache_tsdf_replay(cache, root / "replay")

    def test_replay_reports_missing_required_payload_array_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
                store_arrays=True,
            )
            payload_path = cache / "arrays" / "frame_000000.npz"
            with np.load(payload_path) as payload_file:
                payload = {name: payload_file[name] for name in payload_file.files}
            del payload["depth_sigma_m"]
            np.savez(payload_path, **payload)

            with self.assertRaisesRegex(
                ValueError,
                r"frame_000000\.npz.*missing required array payload key depth_sigma_m",
            ):
                write_teacher_cache_tsdf_replay(cache, root / "replay")

    def test_replay_reports_missing_pose_summary_field_with_path_named_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
                store_arrays=True,
            )
            summaries_path = cache / "frame_summaries.jsonl"
            lines = summaries_path.read_text(encoding="utf-8").splitlines()
            first_summary = json.loads(lines[0])
            del first_summary["pose"]["T_world_camera"]
            lines[0] = json.dumps(first_summary, sort_keys=True)
            summaries_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"frame_summaries\.jsonl:1.*T_world_camera",
            ):
                write_teacher_cache_tsdf_replay(cache, root / "replay")

    def test_cli_teacher_cache_tsdf_smoke_writes_deterministic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            first_output = root / "replay_a"
            second_output = root / "replay_b"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
                store_arrays=True,
            )

            command = [
                sys.executable,
                "-m",
                "atlas3r",
                "smoke",
                "teacher-cache-tsdf",
                "--input",
                str(cache),
                "--write-mesh-sidecar",
                "--output",
            ]
            first = subprocess.run(
                [*command, str(first_output)],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            second = subprocess.run(
                [*command, str(second_output)],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("metadata.json", first.stdout)
            self.assertIn("metrics.json", first.stdout)
            self.assertIn(MESH_SIDECAR_FILENAME, first.stdout)
            self.assertTrue((first_output / "tsdf_grid.npz").is_file())
            self.assertTrue((first_output / "surface_points.npz").is_file())
            self.assertTrue((first_output / MESH_SIDECAR_FILENAME).is_file())
            load_tsdf_surface_mesh_sidecar(first_output / MESH_SIDECAR_FILENAME)
            self.assertEqual(
                (first_output / "metadata.json").read_text(encoding="utf-8"),
                (second_output / "metadata.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (first_output / MESH_SIDECAR_FILENAME).read_text(encoding="utf-8"),
                (second_output / MESH_SIDECAR_FILENAME).read_text(encoding="utf-8"),
            )

    def test_cli_teacher_cache_tsdf_reports_path_named_replay_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "teacher-cache-tsdf",
                    "--input",
                    str(cache),
                    "--output",
                    str(root / "replay"),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("metadata.json", result.stderr)
            self.assertIn("arrays.stored=true", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
