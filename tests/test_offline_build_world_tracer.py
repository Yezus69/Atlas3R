from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from tests.helpers import write_fake_vggt_cache, write_ppm_sequence


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
            self.assertFalse(training["usable_for_training"])
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
