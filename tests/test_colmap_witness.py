from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.colmap_witness import (
    ColmapWitnessOptions,
    colmap_command_plan,
    run_colmap_witness,
)
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_fake_colmap_text_model, write_ppm_sequence


class ColmapWitnessTest(unittest.TestCase):
    def test_unavailable_executable_status_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, frames, keyframes, failures = _prepared_run(root)

            result = run_colmap_witness(
                run_dir,
                frame_cache=frames,
                keyframes=keyframes.keyframes,
                options=ColmapWitnessOptions(
                    enable_colmap=True,
                    colmap_exe=str(root / "missing_colmap.exe"),
                ),
                failure_points=failures,
            )

            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.registered_image_count, 0)
            self.assertTrue((run_dir / "classical" / "classical_status.json").is_file())

    def test_command_plan_uses_requested_matcher_and_gpu(self) -> None:
        plan = colmap_command_plan(
            "colmap",
            image_dir=Path("images"),
            database_path=Path("database.db"),
            sparse_dir=Path("sparse"),
            camera_model="PINHOLE",
            matcher="exhaustive",
            use_gpu=0,
        )

        stages = [stage for stage, _ in plan]
        self.assertIn("exhaustive_matcher", stages)
        self.assertIn("--ImageReader.camera_model", plan[0][1])
        self.assertIn("PINHOLE", plan[0][1])
        self.assertIn("0", plan[0][1])

    def test_subprocess_failure_writes_stage_and_stderr_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, frames, keyframes, failures = _prepared_run(root)

            result = run_colmap_witness(
                run_dir,
                frame_cache=frames,
                keyframes=keyframes.keyframes,
                options=ColmapWitnessOptions(
                    enable_colmap=True,
                    colmap_exe=sys.executable,
                    colmap_timeout_s=30,
                ),
                failure_points=failures,
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.stage_failed, "feature_extractor")
            status = json.loads(
                (run_dir / "classical" / "classical_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["stage_failed"], "feature_extractor")
            self.assertTrue((run_dir / "classical" / "colmap_stderr_tail.txt").is_file())

    def test_replay_cache_loads_sparse_model_without_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, frames, keyframes, failures = _prepared_run(root)
            cache = write_fake_colmap_text_model(root / "cache")

            result = run_colmap_witness(
                run_dir,
                frame_cache=frames,
                keyframes=keyframes.keyframes,
                options=ColmapWitnessOptions(colmap_proposal_cache=str(cache)),
                failure_points=failures,
            )

            self.assertEqual(result.status, "available")
            self.assertEqual(result.registered_image_count, 4)
            self.assertEqual(result.sparse_point_count, 4)
            self.assertIsNotNone(result.sparse_model)
            self.assertTrue((run_dir / "classical" / "colmap_sparse_points.ply").is_file())


def _prepared_run(root: Path):
    input_dir = root / "input"
    run_dir = ensure_run_tree(root / "run")
    write_ppm_sequence(input_dir, count=4, width=4, height=3)
    failures = []
    frames = build_frame_cache(input_dir, run_dir, max_frames=4, failure_points=failures)
    keyframes = select_keyframes(
        frames.records,
        run_dir,
        keyframe_stride=1,
        keyframe_max_count=4,
        failure_points=failures,
    )
    return run_dir, frames, keyframes, failures


if __name__ == "__main__":
    unittest.main()
