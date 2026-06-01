import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.mapping.cpu_tsdf import write_tsdf_cube_room_smoke
from atlas3r.mapping.mesh_sidecar import MESH_SIDECAR_FILENAME
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.mapping.tsdf_output_inspection import (
    TSDF_OUTPUT_INSPECTION_FORMAT_NAME,
    format_tsdf_output_folder_inspection,
    tsdf_output_folder_inspection_record,
)
from atlas3r.mapping.world_map_sidecar import WORLD_MAP_SIDECAR_FILENAME
from atlas3r.models.adapters import run_adapter_to_cache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TSDFOutputInspectionTest(unittest.TestCase):
    def test_tsdf_smoke_output_folder_inspection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_output = root / "tsdf_a"
            second_output = root / "tsdf_b"
            write_tsdf_cube_room_smoke(first_output, write_world_map_sidecar=True)
            write_tsdf_cube_room_smoke(second_output, write_world_map_sidecar=True)

            first = format_tsdf_output_folder_inspection(first_output)
            second = format_tsdf_output_folder_inspection(second_output)
            inspection = json.loads(first)

            self.assertEqual(first, second)
            self.assertEqual(inspection["format_name"], TSDF_OUTPUT_INSPECTION_FORMAT_NAME)
            self.assertEqual(inspection["inspect_mode"], "complete")
            self.assertEqual(
                inspection["surface"]["artifact_type"], "phase_0d_cpu_tsdf_surface_points"
            )
            self.assertTrue(inspection["artifacts"]["mesh_chunk_sidecar_json"]["present"])
            self.assertTrue(inspection["artifacts"]["world_map_sidecar_json"]["present"])
            self.assertEqual(
                inspection["mesh_chunk"]["chunk_id"],
                "phase_0d_cpu_tsdf_surface_reference",
            )
            self.assertEqual(
                inspection["world_map"]["map_id"],
                "phase_2a_cpu_tsdf_world_map_reference",
            )
            self.assertTrue(inspection["cross_checks"]["passed"])
            self.assertTrue(inspection["truth_boundary"]["low_fidelity_reference_only"])
            self.assertTrue(inspection["truth_boundary"]["observed_only"])
            self.assertTrue(inspection["truth_boundary"]["not_completed"])
            self.assertFalse(inspection["truth_boundary"]["accuracy_report"])

    def test_teacher_cache_replay_output_folder_inspection_is_deterministic(self) -> None:
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
            write_teacher_cache_tsdf_replay(cache, first_output, write_world_map_sidecar=True)
            write_teacher_cache_tsdf_replay(cache, second_output, write_world_map_sidecar=True)

            first = format_tsdf_output_folder_inspection(first_output)
            second = format_tsdf_output_folder_inspection(second_output)
            inspection = json.loads(first)

            self.assertEqual(first, second)
            self.assertEqual(
                inspection["surface"]["artifact_type"],
                "phase_1d_teacher_cache_cpu_tsdf_surface_points",
            )
            self.assertEqual(
                inspection["mesh_chunk"]["chunk_id"],
                "phase_1d_teacher_cache_tsdf_surface_reference",
            )
            self.assertEqual(inspection["surface"]["coordinate_frame"], "synthetic_world")
            self.assertEqual(inspection["surface"]["metric_scale_source"], "known_anchor")
            self.assertEqual(inspection["surface"]["source_frame_ids"], [0, 1, 2])

    def test_metadata_mismatch_rejects_with_path_named_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output, write_world_map_sidecar=True)
            mesh_path = output / MESH_SIDECAR_FILENAME
            record = json.loads(mesh_path.read_text(encoding="utf-8"))
            record["metadata"]["coordinate_frame"] = "wrong_world"
            mesh_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"mesh_chunk_sidecar\.json.*coordinate_frame",
            ):
                tsdf_output_folder_inspection_record(output)

    def test_required_sidecar_modes_report_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "surface_only"
            write_tsdf_cube_room_smoke(output)

            surface_only = tsdf_output_folder_inspection_record(output, mode="surface")
            self.assertFalse(surface_only["artifacts"]["mesh_chunk_sidecar_json"]["present"])
            self.assertFalse(surface_only["artifacts"]["world_map_sidecar_json"]["present"])

            with self.assertRaisesRegex(
                ValueError,
                r"mesh_chunk_sidecar\.json.*missing required MeshChunk sidecar.*mesh",
            ):
                tsdf_output_folder_inspection_record(output, mode="mesh")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "mesh_only"
            write_tsdf_cube_room_smoke(output, write_mesh_sidecar=True)

            with self.assertRaisesRegex(
                ValueError,
                r"world_map_sidecar\.json.*missing required WorldMap sidecar.*complete",
            ):
                tsdf_output_folder_inspection_record(output)

    def test_cli_inspect_tsdf_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            write_tsdf_cube_room_smoke(output, write_world_map_sidecar=True)

            command = [
                sys.executable,
                "-m",
                "atlas3r",
                "inspect",
                "tsdf-output",
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
            self.assertEqual(inspection["format_name"], TSDF_OUTPUT_INSPECTION_FORMAT_NAME)
            self.assertIn(MESH_SIDECAR_FILENAME, inspection["cross_checks"]["artifacts_checked"])
            self.assertIn(
                WORLD_MAP_SIDECAR_FILENAME, inspection["cross_checks"]["artifacts_checked"]
            )
            self.assertNotIn("Traceback", first.stderr)


if __name__ == "__main__":
    unittest.main()
