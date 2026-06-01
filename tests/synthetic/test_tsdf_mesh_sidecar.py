import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.api import MeshChunk, SurfaceSource
from atlas3r.data.synthetic_cube_room import (
    create_synthetic_cube_room_scene,
    write_synthetic_cube_room_session,
)
from atlas3r.mapping.cpu_tsdf import (
    extract_tsdf_surface,
    integrate_synthetic_cube_room_scene,
    write_tsdf_cube_room_smoke,
)
from atlas3r.mapping.mesh_sidecar import (
    MESH_SIDECAR_FILENAME,
    MESH_SIDECAR_FORMAT_NAME,
    load_tsdf_surface_mesh_sidecar,
    mesh_chunk_from_tsdf_surface,
    tsdf_surface_mesh_sidecar_record,
    write_tsdf_surface_mesh_sidecar_from_artifacts,
)
from atlas3r.mapping.teacher_cache_replay import write_teacher_cache_tsdf_replay
from atlas3r.models.adapters import run_adapter_to_cache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TSDFMeshSidecarTest(unittest.TestCase):
    def test_surface_mesh_sidecar_validates_mesh_chunk_contract_and_metadata(self) -> None:
        scene = create_synthetic_cube_room_scene()
        surface = extract_tsdf_surface(integrate_synthetic_cube_room_scene(scene))

        mesh = mesh_chunk_from_tsdf_surface(surface)
        record = tsdf_surface_mesh_sidecar_record(surface)

        self.assertIsInstance(mesh, MeshChunk)
        self.assertEqual(record["format_name"], MESH_SIDECAR_FORMAT_NAME)
        self.assertEqual(mesh.chunk_id, "cpu_tsdf_surface_reference")
        self.assertEqual(mesh.source_frame_ids, [0, 1, 2])
        self.assertEqual(mesh.scale_source, "known_anchor")
        self.assertEqual(mesh.voxel_size_m, surface.metadata["voxel_size_m"])
        self.assertEqual(
            mesh.mean_uncertainty_m,
            surface.metadata["uncertainty_summary_m"]["mean"],
        )
        self.assertEqual(
            mesh.p95_uncertainty_m,
            surface.metadata["uncertainty_summary_m"]["p95"],
        )
        self.assertEqual(mesh.faces.shape[0], surface.points_world_m.shape[0])
        self.assertEqual(mesh.vertices_m.shape[0], mesh.faces.shape[0] * 3)
        self.assertTrue(np.all(mesh.surface_source_per_face == int(SurfaceSource.OBSERVED_SURFACE)))
        self.assertIn("low_fidelity_reference_only", mesh.flags)
        self.assertIn("observed_surface_samples", mesh.flags)
        self.assertIn("not_completed_surface", mesh.flags)
        self.assertNotIn("completed_surface", mesh.flags)

        metadata = record["metadata"]
        self.assertEqual(metadata["coordinate_frame"], "synthetic_world")
        self.assertEqual(metadata["metric_scale_source"], "known_anchor")
        self.assertEqual(
            metadata["observed_coverage_estimate"],
            surface.metadata["observed_coverage_estimate"],
        )
        self.assertEqual(metadata["source_surface_sample_count"], surface.points_world_m.shape[0])
        self.assertIn("not an accuracy report", metadata["accuracy_note"])

        sample_attributes = record["sample_attributes"]
        self.assertEqual(len(sample_attributes["face_confidence"]), mesh.faces.shape[0])
        self.assertEqual(len(sample_attributes["face_uncertainty_m"]), mesh.faces.shape[0])
        self.assertEqual(len(sample_attributes["vertex_confidence"]), mesh.vertices_m.shape[0])
        self.assertGreater(sample_attributes["face_confidence"][0], 0.0)
        self.assertGreaterEqual(sample_attributes["face_uncertainty_m"][0], 0.0)

    def test_tsdf_smoke_mesh_sidecar_is_deterministic_and_keeps_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_output = root / "tsdf_a"
            second_output = root / "tsdf_b"

            first_paths = write_tsdf_cube_room_smoke(first_output, write_mesh_sidecar=True)
            second_paths = write_tsdf_cube_room_smoke(second_output, write_mesh_sidecar=True)

            self.assertIn(first_output / MESH_SIDECAR_FILENAME, first_paths)
            self.assertIn(second_output / MESH_SIDECAR_FILENAME, second_paths)
            for name in [
                "tsdf_grid.npz",
                "surface_points.npz",
                "metadata.json",
                "metrics.json",
                MESH_SIDECAR_FILENAME,
            ]:
                self.assertTrue((first_output / name).is_file())
            self.assertTrue((first_output / "synthetic_cube_room.atlas3r").is_dir())
            self.assertEqual(
                (first_output / MESH_SIDECAR_FILENAME).read_text(encoding="utf-8"),
                (second_output / MESH_SIDECAR_FILENAME).read_text(encoding="utf-8"),
            )
            mesh = load_tsdf_surface_mesh_sidecar(first_output / MESH_SIDECAR_FILENAME)
            self.assertIsInstance(mesh, MeshChunk)

    def test_teacher_cache_replay_can_write_observed_mesh_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "session.atlas3r"
            cache = root / "teacher_cache"
            output = root / "replay"
            write_synthetic_cube_room_session(session)
            run_adapter_to_cache(
                adapter_name="fixture-cube-room",
                input_session=session,
                output_cache=cache,
                store_arrays=True,
            )

            written = write_teacher_cache_tsdf_replay(cache, output, write_mesh_sidecar=True)

            self.assertIn(output / MESH_SIDECAR_FILENAME, written)
            record = json.loads((output / MESH_SIDECAR_FILENAME).read_text(encoding="utf-8"))
            mesh = load_tsdf_surface_mesh_sidecar(output / MESH_SIDECAR_FILENAME)
            self.assertIsInstance(mesh, MeshChunk)
            self.assertEqual(mesh.chunk_id, "phase_1d_teacher_cache_tsdf_surface_reference")
            self.assertEqual(record["metadata"]["coordinate_frame"], "synthetic_world")
            self.assertEqual(
                record["metadata"]["source_surface_artifact_type"],
                "phase_1d_teacher_cache_cpu_tsdf_surface_points",
            )
            self.assertEqual(
                record["metadata"]["source_surface_metadata"]["teacher_adapter"],
                "fixture-cube-room",
            )
            self.assertIn(
                "input_confidence_summaries", record["metadata"]["source_surface_metadata"]
            )

    def test_sidecar_writer_reports_missing_and_malformed_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "coordinate_frame": "synthetic_world",
                        "metric_scale_source": "known_anchor",
                        "source_frame_ids": [0],
                        "voxel_size_m": 0.1,
                        "observed_coverage_estimate": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"surface_points\.npz.*missing"):
                write_tsdf_surface_mesh_sidecar_from_artifacts(output)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output)
            with np.load(output / "surface_points.npz") as surface_file:
                payload = {name: surface_file[name] for name in surface_file.files}
            del payload["confidence"]
            np.savez(output / "surface_points.npz", **payload)

            with self.assertRaisesRegex(
                ValueError,
                r"surface_points\.npz.*missing required array confidence",
            ):
                write_tsdf_surface_mesh_sidecar_from_artifacts(output)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf"
            write_tsdf_cube_room_smoke(output)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            del metadata["observed_coverage_estimate"]
            (output / "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"metadata\.json.*observed_coverage_estimate",
            ):
                write_tsdf_surface_mesh_sidecar_from_artifacts(output)

    def test_cli_tsdf_cube_room_writes_mesh_sidecar_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tsdf_smoke"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "smoke",
                    "tsdf-cube-room",
                    "--output",
                    str(output),
                    "--write-mesh-sidecar",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(MESH_SIDECAR_FILENAME, result.stdout)
            self.assertTrue((output / MESH_SIDECAR_FILENAME).is_file())
            self.assertIsInstance(
                load_tsdf_surface_mesh_sidecar(output / MESH_SIDECAR_FILENAME),
                MeshChunk,
            )


if __name__ == "__main__":
    unittest.main()
