from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from tests.helpers import (
    write_fake_colmap_text_model,
    write_fake_depth_pro_cache,
    write_fake_vggt_cache,
    write_ppm_sequence,
)


class OfflineBuildWorldTracerTest(unittest.TestCase):
    def test_tiny_ppm_runs_full_build_world_tracer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            write_ppm_sequence(input_dir, count=4, width=4, height=3)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "4",
                    "--keyframe-stride",
                    "2",
                    "--keyframe-max-count",
                    "3",
                    "--debug-geometry-mode",
                    "flat-depth",
                    "--export-world-map",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            required = [
                "run_manifest.json",
                "frames/frame_index.jsonl",
                "keyframes/keyframes.json",
                "teachers/teacher_status.json",
                "proposals/proposal_manifest.json",
                "world/world_state.json",
                "world/camera_ledger.json",
                "world/scale_ledger.json",
                "world_map/world_map_manifest.json",
                "world_map/camera_trajectory.json",
                "world_map/fused_points.npz",
                "world_map/fused_points.ply",
                "world_map/occupancy_grid.npz",
                "world_map/occupancy_grid_metadata.json",
                "world_map/observed_voxel_mesh.ply",
                "world_map/map_quality.json",
                "world_map/map_quality.md",
                "geometry/geometry_preview.npz",
                "geometry/geometry_preview.ply",
                "objects/object_ledger.json",
                "diagnostics/render_repair_diagnostics.json",
                "diagnostics/failure_points.json",
                "quality_report.json",
                "quality_report.md",
                "training_cache/training_cache_manifest.json",
            ]
            for relative in required:
                self.assertTrue((output / relative).is_file(), relative)
            with np.load(output / "geometry" / "geometry_preview.npz", allow_pickle=False) as data:
                point_count = int(data["points_world_m"].shape[0])
                metadata = json.loads(str(data["metadata_json"].item()))
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            map_quality = json.loads(
                (output / "world_map" / "map_quality.json").read_text(encoding="utf-8")
            )
            training = json.loads(
                (output / "training_cache" / "training_cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            failures = json.loads(
                (output / "diagnostics" / "failure_points.json").read_text(encoding="utf-8")
            )
            self.assertGreater(point_count, 0)
            self.assertEqual(metadata["metric_scale_source"], "debug_flat_depth")
            self.assertFalse(metadata["measured_geometry"])
            self.assertFalse(quality["physical_accuracy"])
            self.assertTrue(quality["world_map"]["inspectable_map_available"])
            self.assertTrue(map_quality["inspectable_map_available"])
            self.assertFalse(training["usable_for_training"])
            self.assertEqual(
                training["refs"]["fused_world_map_manifest"],
                "world_map/world_map_manifest.json",
            )
            self.assertGreater(len(failures["failure_points"]), 0)

    def test_png_folder_without_vggt_decodes_but_geometry_is_unavailable(self) -> None:
        if not any(find_spec(name) is not None for name in ("PIL", "imageio", "cv2")):
            self.skipTest("no optional PNG decoder is installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "pngs"
            output = root / "run"
            _write_png(input_dir / "000.png", np.zeros((3, 4, 3), dtype=np.uint8))
            _write_png(input_dir / "001.png", np.full((3, 4, 3), 128, dtype=np.uint8))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "2",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "2",
                    "--debug-geometry-mode",
                    "none",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            frames = (output / "frames" / "frame_index.jsonl").read_text(encoding="utf-8")
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            self.assertIn("frame_000000.ppm", frames)
            self.assertEqual(quality["geometry"]["point_count"], 0)
            self.assertEqual(quality["geometry"]["source_teacher"], "none")

    def test_replayed_vggt_full_pipeline_produces_teacher_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            cache_dir = write_fake_vggt_cache(root / "cache", frame_ids=(0, 1))
            write_ppm_sequence(input_dir, count=2, width=4, height=3)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "2",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "2",
                    "--debug-geometry-mode",
                    "none",
                    "--vggt-proposal-cache",
                    str(cache_dir),
                    "--export-world-map",
                    "--map-depth-source",
                    "vggt",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            proposal_manifest = json.loads(
                (output / "proposals" / "proposal_manifest.json").read_text(encoding="utf-8")
            )
            world = json.loads((output / "world" / "world_state.json").read_text(encoding="utf-8"))
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            training = json.loads(
                (output / "training_cache" / "training_cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            with np.load(output / "geometry" / "geometry_preview.npz", allow_pickle=False) as data:
                point_count = int(data["points_world_m"].shape[0])
                metadata = json.loads(str(data["metadata_json"].item()))

            self.assertEqual(proposal_manifest["teacher_counts"]["vggt_depth_proposals"], 2)
            self.assertEqual(world["pose_status"], "proposed")
            self.assertEqual(world["depth_status"], "proposed")
            self.assertEqual(world["map_status"], "preview")
            self.assertEqual(world["proposal_source"], "vggt")
            self.assertGreater(point_count, 0)
            self.assertEqual(metadata["label_type"], "teacher_pseudo")
            self.assertEqual(quality["geometry"]["source_teacher"], "vggt")
            self.assertFalse(quality["physical_accuracy"])
            self.assertFalse(training["usable_for_training"])
            self.assertTrue((output / "geometry" / "geometry_preview.ply").is_file())
            self.assertTrue((output / "world_map" / "fused_points.ply").is_file())
            self.assertTrue((output / "world_map" / "observed_voxel_mesh.ply").is_file())
            self.assertTrue(quality["world_map"]["inspectable_map_available"])
            self.assertGreater(quality["world_map"]["occupied_voxel_count"], 0)

    def test_replayed_vggt_and_depth_pro_write_disagreement_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            vggt_cache = write_fake_vggt_cache(
                root / "vggt_cache", frame_ids=(0, 1), width=4, height=3, depth_m=2.0
            )
            depth_pro_cache = write_fake_depth_pro_cache(
                root / "depth_pro_cache",
                frame_ids=(0, 1),
                width=4,
                height=3,
                depth_m=2.0,
            )
            write_ppm_sequence(input_dir, count=2, width=4, height=3)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "2",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "2",
                    "--debug-geometry-mode",
                    "none",
                    "--vggt-proposal-cache",
                    str(vggt_cache),
                    "--depth-pro-proposal-cache",
                    str(depth_pro_cache),
                    "--export-world-map",
                    "--map-depth-source",
                    "consensus",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            proposal_manifest = json.loads(
                (output / "proposals" / "proposal_manifest.json").read_text(encoding="utf-8")
            )
            world = json.loads((output / "world" / "world_state.json").read_text(encoding="utf-8"))
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            disagreement = json.loads(
                (output / "diagnostics" / "teacher_disagreement.json").read_text(encoding="utf-8")
            )
            with np.load(
                output / "diagnostics" / "disagreement_maps.npz", allow_pickle=False
            ) as data:
                overlap_count = int(data["valid_overlap_mask"].sum())
            with np.load(output / "geometry" / "geometry_preview.npz", allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                point_count = int(data["points_world_m"].shape[0])

            self.assertEqual(proposal_manifest["teacher_counts"]["vggt_depth_proposals"], 2)
            self.assertEqual(proposal_manifest["teacher_counts"]["depth_pro_depth_proposals"], 2)
            self.assertEqual(disagreement["status"], "available")
            self.assertGreater(overlap_count, 0)
            self.assertIn("depth_pro", world["available_witnesses"])
            self.assertEqual(quality["teacher_disagreement"]["status"], "available")
            self.assertEqual(metadata["source_teacher"], "vggt_pose_consensus_depth_diagnostic")
            self.assertGreater(point_count, 0)
            self.assertTrue((output / "geometry" / "geometry_preview.ply").is_file())
            self.assertTrue((output / "world_map" / "fused_points.ply").is_file())
            self.assertTrue((output / "world_map" / "occupancy_grid.npz").is_file())
            self.assertTrue((output / "world_map" / "observed_voxel_mesh.ply").is_file())
            self.assertEqual(quality["world_map"]["depth_source"], "consensus")
            self.assertTrue(quality["world_map"]["inspectable_map_available"])
            self.assertFalse(quality["physical_accuracy"])

    def test_replayed_depth_pro_only_explains_missing_global_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            depth_pro_cache = write_fake_depth_pro_cache(root / "depth_pro_cache", frame_ids=(0, 1))
            write_ppm_sequence(input_dir, count=2)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "2",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "2",
                    "--debug-geometry-mode",
                    "none",
                    "--depth-pro-proposal-cache",
                    str(depth_pro_cache),
                    "--export-world-map",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            world = json.loads((output / "world" / "world_state.json").read_text(encoding="utf-8"))
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            with np.load(output / "geometry" / "geometry_preview.npz", allow_pickle=False) as data:
                point_count = int(data["points_world_m"].shape[0])

            self.assertEqual(world["pose_status"], "missing_global_pose")
            self.assertEqual(world["map_status"], "none")
            self.assertEqual(point_count, 0)
            self.assertEqual(quality["depth_pro"]["depth_proposals"], 2)
            self.assertEqual(quality["world_map"]["point_count"], 0)
            self.assertFalse(quality["world_map"]["inspectable_map_available"])
            self.assertFalse((output / "geometry" / "geometry_preview.ply").is_file())
            self.assertFalse((output / "world_map" / "fused_points.ply").is_file())

    def test_replayed_colmap_sparse_model_flows_through_build_world(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output = root / "run"
            vggt_cache = write_fake_vggt_cache(
                root / "vggt_cache", frame_ids=(0, 1, 2, 3), width=4, height=3
            )
            depth_pro_cache = write_fake_depth_pro_cache(
                root / "depth_pro_cache", frame_ids=(0, 1, 2, 3), width=4, height=3
            )
            colmap_cache = write_fake_colmap_text_model(root / "colmap_cache")
            write_ppm_sequence(input_dir, count=4, width=4, height=3)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "offline",
                    "build-world",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output),
                    "--max-frames",
                    "4",
                    "--keyframe-stride",
                    "1",
                    "--keyframe-max-count",
                    "4",
                    "--vggt-proposal-cache",
                    str(vggt_cache),
                    "--depth-pro-proposal-cache",
                    str(depth_pro_cache),
                    "--colmap-proposal-cache",
                    str(colmap_cache),
                    "--export-world-map",
                    "--export-best-world-map",
                    "--map-depth-source",
                    "consensus",
                    "--map-write-occupancy",
                    "--map-write-observed-mesh",
                    "--write-ply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            classical = json.loads(
                (output / "classical" / "classical_status.json").read_text(encoding="utf-8")
            )
            alignment = json.loads(
                (output / "classical" / "trajectory_alignment.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(
                (output / "diagnostics" / "classical_map_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))

            self.assertEqual(classical["status"], "available")
            self.assertEqual(classical["registered_image_count"], 4)
            self.assertEqual(classical["sparse_point_count"], 4)
            self.assertEqual(alignment["status"], "available")
            self.assertEqual(alignment["common_frame_count"], 4)
            self.assertTrue((output / "classical" / "aligned_colmap_sparse_points.ply").is_file())
            self.assertEqual(comparison["status"], "available")
            self.assertEqual(quality["classical_geometry_witness"]["status"], "available")
            self.assertTrue((output / "world_map_best" / "fused_points.ply").is_file())


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if find_spec("PIL") is not None:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    if find_spec("imageio") is not None:
        import imageio.v3 as iio

        iio.imwrite(path, rgb)
        return
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    unittest.main()
