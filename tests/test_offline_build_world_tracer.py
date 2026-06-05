from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import write_ppm_sequence


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


if __name__ == "__main__":
    unittest.main()
