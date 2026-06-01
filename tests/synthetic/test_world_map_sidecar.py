import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.api import MeshChunk, WorldMap
from atlas3r.data.synthetic_cube_room import write_synthetic_cube_room_session
from atlas3r.mapping.cpu_tsdf import write_tsdf_cube_room_smoke
from atlas3r.mapping.mesh_sidecar import MESH_SIDECAR_FILENAME, load_tsdf_surface_mesh_sidecar
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.mapping.world_map_sidecar import (
    WORLD_MAP_SIDECAR_FILENAME,
    WORLD_MAP_SIDECAR_FORMAT_NAME,
    load_tsdf_world_map_sidecar,
    world_map_from_mesh_sidecar,
    write_tsdf_world_map_sidecar_from_artifacts,
)
from atlas3r.models.adapters import run_adapter_to_cache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class WorldMapSidecarTest(unittest.TestCase):
    def test_mesh_sidecar_world_map_validates_contracts_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output, write_mesh_sidecar=True)

            path = write_tsdf_world_map_sidecar_from_artifacts(output)
            world_map = load_tsdf_world_map_sidecar(path)
            mesh = load_tsdf_surface_mesh_sidecar(output / MESH_SIDECAR_FILENAME)
            record = json.loads(path.read_text(encoding="utf-8"))

            self.assertIsInstance(world_map, WorldMap)
            self.assertEqual(record["format_name"], WORLD_MAP_SIDECAR_FORMAT_NAME)
            self.assertEqual(world_map.map_id, "phase_2a_cpu_tsdf_world_map_reference")
            self.assertEqual(world_map.created_at_ns, 0)
            self.assertEqual(world_map.world_frame_name, "synthetic_world")
            self.assertEqual(world_map.scale_source, "known_anchor")
            self.assertEqual(world_map.objects, {})
            self.assertEqual(world_map.keyframes, {})
            self.assertEqual(list(world_map.mesh_chunks), [mesh.chunk_id])
            self.assertIsInstance(world_map.mesh_chunks[mesh.chunk_id], MeshChunk)

            metadata = world_map.metadata
            self.assertEqual(metadata["coordinate_frame"], "synthetic_world")
            self.assertEqual(metadata["source_frame_ids"], [0, 1, 2])
            self.assertEqual(metadata["voxel_size_m"], mesh.voxel_size_m)
            self.assertEqual(metadata["metric_scale_source"], mesh.scale_source)
            self.assertEqual(
                metadata["observed_coverage_estimate"],
                record["metadata"]["observed_coverage_estimate"],
            )
            self.assertEqual(metadata["mean_uncertainty_m"], mesh.mean_uncertainty_m)
            self.assertEqual(metadata["p95_uncertainty_m"], mesh.p95_uncertainty_m)
            self.assertEqual(metadata["object_count"], 0)
            self.assertFalse(metadata["object_meshes_invented"])
            self.assertIn("low_fidelity_reference_only", metadata["flags"])
            self.assertIn("observed_surface_samples", metadata["flags"])
            self.assertIn("not_completed_surface", metadata["flags"])
            self.assertIn("not_accuracy_report", metadata["flags"])
            self.assertIn("not an accuracy report", metadata["accuracy_note"])
            self.assertEqual(
                world_map.global_confidence,
                metadata["confidence_summary"]["mean"],
            )

    def test_tsdf_smoke_world_map_sidecar_is_deterministic_and_keeps_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_output = root / "tsdf_a"
            second_output = root / "tsdf_b"

            first_paths = write_tsdf_cube_room_smoke(
                first_output,
                write_world_map_sidecar=True,
            )
            second_paths = write_tsdf_cube_room_smoke(
                second_output,
                write_world_map_sidecar=True,
            )

            self.assertIn(first_output / MESH_SIDECAR_FILENAME, first_paths)
            self.assertIn(first_output / WORLD_MAP_SIDECAR_FILENAME, first_paths)
            self.assertIn(second_output / WORLD_MAP_SIDECAR_FILENAME, second_paths)
            for name in [
                "tsdf_grid.npz",
                "surface_points.npz",
                "metadata.json",
                "metrics.json",
                MESH_SIDECAR_FILENAME,
                WORLD_MAP_SIDECAR_FILENAME,
            ]:
                self.assertTrue((first_output / name).is_file())
            self.assertEqual(
                (first_output / WORLD_MAP_SIDECAR_FILENAME).read_text(encoding="utf-8"),
                (second_output / WORLD_MAP_SIDECAR_FILENAME).read_text(encoding="utf-8"),
            )
            self.assertIsInstance(
                load_tsdf_world_map_sidecar(first_output / WORLD_MAP_SIDECAR_FILENAME),
                WorldMap,
            )

    def test_teacher_cache_replay_writes_deterministic_world_map_sidecar(self) -> None:
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

            first_paths = write_teacher_cache_tsdf_replay(
                cache,
                first_output,
                write_world_map_sidecar=True,
            )
            write_teacher_cache_tsdf_replay(
                cache,
                second_output,
                write_world_map_sidecar=True,
            )

            self.assertIn(first_output / MESH_SIDECAR_FILENAME, first_paths)
            self.assertIn(first_output / WORLD_MAP_SIDECAR_FILENAME, first_paths)
            self.assertEqual(
                (first_output / WORLD_MAP_SIDECAR_FILENAME).read_text(encoding="utf-8"),
                (second_output / WORLD_MAP_SIDECAR_FILENAME).read_text(encoding="utf-8"),
            )
            world_map = load_tsdf_world_map_sidecar(first_output / WORLD_MAP_SIDECAR_FILENAME)
            metadata = world_map.metadata
            source_mesh_metadata = metadata["source_mesh_metadata"]
            self.assertEqual(
                source_mesh_metadata["source_surface_artifact_type"],
                "phase_1d_teacher_cache_cpu_tsdf_surface_points",
            )
            self.assertEqual(
                source_mesh_metadata["source_surface_metadata"]["teacher_adapter"],
                "fixture-cube-room",
            )
            self.assertIn(
                "input_confidence_summaries",
                source_mesh_metadata["source_surface_metadata"],
            )
            mesh = load_tsdf_surface_mesh_sidecar(first_output / MESH_SIDECAR_FILENAME)
            self.assertEqual(metadata["mean_uncertainty_m"], mesh.mean_uncertainty_m)
            self.assertEqual(metadata["p95_uncertainty_m"], mesh.p95_uncertainty_m)
            self.assertGreaterEqual(metadata["p95_uncertainty_m"], metadata["mean_uncertainty_m"])

    def test_world_map_writer_reports_missing_and_malformed_mesh_sidecar_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            with self.assertRaisesRegex(
                ValueError,
                r"mesh_chunk_sidecar\.json.*missing required MeshChunk sidecar",
            ):
                write_tsdf_world_map_sidecar_from_artifacts(output)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output, write_mesh_sidecar=True)
            mesh_path = output / MESH_SIDECAR_FILENAME
            record = json.loads(mesh_path.read_text(encoding="utf-8"))
            record["mesh_chunk"]["flags"].remove("not_accuracy_report")
            mesh_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"mesh_chunk_sidecar\.json.*truth-boundary flags.*not_accuracy_report",
            ):
                world_map_from_mesh_sidecar(mesh_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output, write_mesh_sidecar=True)
            mesh_path = output / MESH_SIDECAR_FILENAME
            record = json.loads(mesh_path.read_text(encoding="utf-8"))
            del record["metadata"]["observed_coverage_estimate"]
            mesh_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"mesh_chunk_sidecar\.json.*observed_coverage_estimate",
            ):
                world_map_from_mesh_sidecar(mesh_path)

    def test_cli_smoke_and_inspect_world_map_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf_smoke"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            smoke_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "tsdf-cube-room",
                    "--output",
                    str(output),
                    "--write-world-map-sidecar",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            inspect_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "inspect",
                    "world-map",
                    "--input",
                    str(output / WORLD_MAP_SIDECAR_FILENAME),
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(smoke_result.returncode, 0, smoke_result.stderr)
            self.assertIn(MESH_SIDECAR_FILENAME, smoke_result.stdout)
            self.assertIn(WORLD_MAP_SIDECAR_FILENAME, smoke_result.stdout)
            self.assertTrue((output / WORLD_MAP_SIDECAR_FILENAME).is_file())
            self.assertEqual(inspect_result.returncode, 0, inspect_result.stderr)
            self.assertIn("phase_2a_cpu_tsdf_world_map_reference", inspect_result.stdout)
            self.assertIn("not_accuracy_report", inspect_result.stdout)
            self.assertNotIn("Traceback", inspect_result.stderr)


if __name__ == "__main__":
    unittest.main()
