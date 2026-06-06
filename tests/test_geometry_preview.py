from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.geometry_preview import (
    lift_depth_to_world_points,
    read_geometry_metadata,
    write_geometry_preview,
)
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import ProposalCacheResult, write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_vggt_cache, write_ppm_sequence


class GeometryPreviewTest(unittest.TestCase):
    def test_pixel_depth_pose_lifts_to_world_points(self) -> None:
        K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        T_world_camera = np.eye(4, dtype=np.float32)

        points = lift_depth_to_world_points(K, depth, T_world_camera)

        expected = np.array(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(points, expected)

    def test_empty_geometry_writes_failure_reason_and_loadable_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=1)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=1, failure_points=failures)
            proposals = ProposalCacheResult(
                status="partial",
                manifest_path="proposals/proposal_manifest.json",
                streams=(),
                debug_depth_records=(),
                depth_proposal_available=False,
                debug_geometry_mode="none",
            )

            result = write_geometry_preview(
                run_dir,
                frame_cache=frames,
                proposal_cache=proposals,
                write_ply=True,
                failure_points=failures,
            )

            self.assertEqual(result.point_count, 0)
            metadata = read_geometry_metadata(run_dir / result.geometry_npz_path)
            self.assertEqual(metadata["status"], "unavailable")
            self.assertIsNone(result.geometry_ply_path)
            self.assertTrue(failures)

    def test_vggt_depth_pose_lifts_nonzero_teacher_geometry_and_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            cache_dir = write_fake_vggt_cache(root / "cache", frame_ids=(0,), width=4, height=3)
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=1, width=4, height=3)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=1, failure_points=failures)
            keyframes = select_keyframes(
                frames.records,
                run_dir,
                keyframe_stride=1,
                keyframe_max_count=1,
                failure_points=failures,
            )
            vggt = load_vggt_proposal_cache(cache_dir)
            statuses = write_teacher_statuses(run_dir, failures, vggt_result=vggt)
            proposals = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
                vggt_result=vggt,
            )

            result = write_geometry_preview(
                run_dir,
                frame_cache=frames,
                proposal_cache=proposals,
                write_ply=True,
                failure_points=failures,
            )

            metadata = read_geometry_metadata(run_dir / result.geometry_npz_path)
            self.assertGreater(result.point_count, 0)
            self.assertEqual(result.source_teacher, "vggt")
            self.assertEqual(metadata["label_type"], "teacher_pseudo")
            self.assertFalse(metadata["measured_geometry"])
            self.assertFalse(metadata["predicted_completion"])
            self.assertTrue((run_dir / "geometry" / "geometry_preview.ply").is_file())

    def test_invalid_vggt_depth_is_masked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            cache_dir = write_fake_vggt_cache(root / "cache", frame_ids=(0,), width=4, height=3)
            with np.load(cache_dir / "vggt_depths.npz", allow_pickle=False) as payload:
                arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
                metadata_json = str(payload["metadata_json"].item())
            arrays["vggt_depth_000000"] = np.full((3, 4), -1.0, dtype=np.float32)
            np.savez_compressed(
                cache_dir / "vggt_depths.npz",
                **arrays,
                metadata_json=metadata_json,
            )
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=1, width=4, height=3)
            failures = []
            frames = build_frame_cache(input_dir, run_dir, max_frames=1, failure_points=failures)
            keyframes = select_keyframes(
                frames.records,
                run_dir,
                keyframe_stride=1,
                keyframe_max_count=1,
                failure_points=failures,
            )
            vggt = load_vggt_proposal_cache(cache_dir)
            statuses = write_teacher_statuses(run_dir, failures, vggt_result=vggt)
            proposals = write_proposal_cache(
                run_dir,
                teacher_statuses=statuses,
                frame_records=frames.records,
                keyframes=keyframes.keyframes,
                debug_geometry_mode="none",
                vggt_result=vggt,
            )

            result = write_geometry_preview(
                run_dir,
                frame_cache=frames,
                proposal_cache=proposals,
                write_ply=True,
                failure_points=failures,
            )

            self.assertEqual(result.point_count, 0)
            self.assertIsNone(result.geometry_ply_path)


if __name__ == "__main__":
    unittest.main()
