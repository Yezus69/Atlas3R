from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.classical_validated_map import try_write_classical_validated_map
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.fused_world_map import FusedWorldMapOptions
from atlas3r.offline.fused_world_map_artifacts import FusedPointCloud
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from tests.helpers import write_ppm_sequence


class ClassicalValidatedMapTest(unittest.TestCase):
    def test_accepts_validation_that_removes_outlier_without_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _prepared_run(Path(tmp))
            _write_aligned_points(
                run["run_dir"], np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
            )

            result = try_write_classical_validated_map(
                run["run_dir"],
                input_path=str(run["input_dir"]),
                frame_cache=run["frames"],
                keyframes=run["keyframes"].keyframes,
                proposal_cache=run["proposals"],
                source_cloud=_cloud(include_many_outliers=False),
                trajectory=[],
                rejected_pose_count=0,
                map_options=FusedWorldMapOptions(voxel_size_m=0.05),
                selected_source="raw_consensus",
                classical_comparison=_ComparisonStub("agrees", "agrees"),
            )

            self.assertTrue(result.accepted)
            self.assertGreaterEqual(result.retained_point_ratio, 0.5)
            self.assertTrue(
                (run["run_dir"] / "world_map_classical_validated" / "fused_points.ply").is_file()
            )

    def test_anti_cheat_rejects_filter_that_deletes_most_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _prepared_run(Path(tmp))
            _write_aligned_points(run["run_dir"], np.asarray([[0, 0, 0]], dtype=np.float32))

            result = try_write_classical_validated_map(
                run["run_dir"],
                input_path=str(run["input_dir"]),
                frame_cache=run["frames"],
                keyframes=run["keyframes"].keyframes,
                proposal_cache=run["proposals"],
                source_cloud=_cloud(include_many_outliers=True),
                trajectory=[],
                rejected_pose_count=0,
                map_options=FusedWorldMapOptions(voxel_size_m=0.05),
                selected_source="raw_consensus",
                classical_comparison=_ComparisonStub("agrees", "agrees"),
            )

            self.assertFalse(result.accepted)
            self.assertEqual(result.status, "rejected")

    def test_bad_alignment_does_not_validate_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _prepared_run(Path(tmp))
            _write_aligned_points(run["run_dir"], np.asarray([[0, 0, 0]], dtype=np.float32))

            result = try_write_classical_validated_map(
                run["run_dir"],
                input_path=str(run["input_dir"]),
                frame_cache=run["frames"],
                keyframes=run["keyframes"].keyframes,
                proposal_cache=run["proposals"],
                source_cloud=_cloud(include_many_outliers=False),
                trajectory=[],
                rejected_pose_count=0,
                map_options=FusedWorldMapOptions(voxel_size_m=0.05),
                selected_source="raw_consensus",
                classical_comparison=_ComparisonStub("disagrees", "agrees"),
            )

            self.assertFalse(result.accepted)
            self.assertEqual(result.status, "skipped")


class _ComparisonStub:
    status = "available"
    aligned_sparse_points_path = "classical/aligned_colmap_sparse_points.npz"

    def __init__(self, trajectory_status: str, map_status: str):
        self.trajectory_agreement_status = trajectory_status
        self.map_agreement_status = map_status


def _cloud(*, include_many_outliers: bool) -> FusedPointCloud:
    near = np.asarray(
        [[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [0.2, 0, 0], [0.25, 0, 0]],
        dtype=np.float32,
    )
    outliers = (
        np.asarray(
            [
                [10.0, 0, 0],
                [11.0, 0, 0],
                [12.0, 0, 0],
                [13.0, 0, 0],
                [14.0, 0, 0],
                [15.0, 0, 0],
                [16.0, 0, 0],
                [17.0, 0, 0],
            ],
            dtype=np.float32,
        )
        if include_many_outliers
        else np.asarray([[10.0, 0, 0]], dtype=np.float32)
    )
    points = np.concatenate([near, outliers], axis=0)
    count = points.shape[0]
    return FusedPointCloud(
        points_world_m=points,
        colors_u8=np.full((count, 3), 120, dtype=np.uint8),
        confidence=np.ones((count,), dtype=np.float32),
        source_frame_ids=np.zeros((count,), dtype=np.int64),
        source_keyframe_ids=np.zeros((count,), dtype=np.int64),
        depth_source_id=np.zeros((count,), dtype=np.int32),
        disagreement_rel=np.zeros((count,), dtype=np.float32),
        point_sigma_m=np.full((count,), 0.05, dtype=np.float32),
        depth_source="fixture",
        metric_scale_source="fixture",
        valid_depth_ratio=1.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=None,
        mapped_disagreement_p50=None,
        mapped_disagreement_p95=None,
        per_frame_point_counts={},
    )


def _write_aligned_points(run_dir: Path, points: np.ndarray) -> None:
    path = run_dir / "classical" / "aligned_colmap_sparse_points.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points_world_m=points.astype(np.float32))


def _prepared_run(root: Path) -> dict[str, object]:
    input_dir = root / "input"
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
    statuses = write_teacher_statuses(run_dir, failures)
    proposals = write_proposal_cache(
        run_dir,
        teacher_statuses=statuses,
        frame_records=frames.records,
        keyframes=keyframes.keyframes,
        debug_geometry_mode="none",
    )
    return {
        "input_dir": input_dir,
        "run_dir": run_dir,
        "frames": frames,
        "keyframes": keyframes,
        "proposals": proposals,
    }


if __name__ == "__main__":
    unittest.main()
