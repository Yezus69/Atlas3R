from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.depth_pro_witness import load_depth_pro_proposal_cache
from atlas3r.offline.disagreement import write_teacher_disagreement
from atlas3r.offline.frame_cache import build_frame_cache
from atlas3r.offline.fused_world_map import (
    FusedWorldMapOptions,
    build_fused_point_cloud,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    write_fused_world_map,
)
from atlas3r.offline.keyframes import select_keyframes
from atlas3r.offline.proposal_cache import write_proposal_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.teacher_witnesses import write_teacher_statuses
from atlas3r.offline.vggt_witness import load_vggt_proposal_cache
from tests.helpers import write_fake_depth_pro_cache, write_fake_vggt_cache, write_ppm_sequence


class FusedWorldMapTest(unittest.TestCase):
    def test_vggt_depth_pose_lifts_points_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _prepared_run(Path(tmp), frame_ids=(0,), width=2, height=2)

            result = write_fused_world_map(
                run.run_dir,
                input_path=str(run.input_dir),
                frame_cache=run.frames,
                keyframes=run.keyframes.keyframes,
                proposal_cache=run.proposals,
                disagreement=None,
                options=FusedWorldMapOptions(
                    export_world_map=True,
                    point_stride=1,
                    write_occupancy=True,
                    write_observed_mesh=True,
                ),
                failure_points=run.failures,
            )

            self.assertGreater(result.point_count, 0)
            self.assertGreater(result.occupied_voxel_count, 0)
            self.assertGreater(result.mesh_triangle_count, 0)
            self.assertEqual(result.trajectory_count, 1)
            self.assertTrue(result.inspectable_map_available)
            with np.load(run.run_dir / "world_map" / "fused_points.npz") as payload:
                points = payload["points_world_m"]
                metadata = json.loads(str(payload["metadata_json"].item()))
            np.testing.assert_allclose(points[0], [-0.5, -0.5, 2.0], atol=1e-6)
            self.assertEqual(metadata["truth_boundary"]["label_type"], "teacher_pseudo_fused_map")
            ply = (run.run_dir / "world_map" / "fused_points.ply").read_text(encoding="utf-8")
            self.assertIn(f"element vertex {result.point_count}", ply)
            trajectory = json.loads(
                (run.run_dir / "world_map" / "camera_trajectory.json").read_text(encoding="utf-8")
            )
            np.testing.assert_allclose(
                trajectory["poses"][0]["camera_center_world_m"], [0.0, 0.0, 0.0]
            )

    def test_invalid_depth_low_confidence_and_stride_cap_filter_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = _prepared_run(root / "invalid", frame_ids=(0,), width=4, height=4)
            invalid.proposals.depth_arrays["vggt_depth_000000"] = np.full(
                (4, 4), -1.0, dtype=np.float32
            )
            invalid_cloud = build_fused_point_cloud(
                invalid.run_dir,
                frame_cache=invalid.frames,
                proposal_cache=invalid.proposals,
                disagreement=None,
                options=FusedWorldMapOptions(export_world_map=True, point_stride=1),
            )
            self.assertEqual(invalid_cloud.points_world_m.shape[0], 0)

            low_conf = _prepared_run(root / "low_conf", frame_ids=(0,), width=4, height=4)
            low_conf.proposals.depth_arrays["vggt_confidence_000000"] = np.full(
                (4, 4), 0.1, dtype=np.float32
            )
            low_conf_cloud = build_fused_point_cloud(
                low_conf.run_dir,
                frame_cache=low_conf.frames,
                proposal_cache=low_conf.proposals,
                disagreement=None,
                options=FusedWorldMapOptions(
                    export_world_map=True, point_stride=1, min_confidence=0.25
                ),
            )
            self.assertEqual(low_conf_cloud.points_world_m.shape[0], 0)
            self.assertGreater(low_conf_cloud.rejected_low_confidence_ratio, 0.0)

            capped = _prepared_run(root / "capped", frame_ids=(0,), width=4, height=4)
            capped_cloud = build_fused_point_cloud(
                capped.run_dir,
                frame_cache=capped.frames,
                proposal_cache=capped.proposals,
                disagreement=None,
                options=FusedWorldMapOptions(
                    export_world_map=True,
                    point_stride=2,
                    max_points=3,
                    min_confidence=0.0,
                ),
            )
            self.assertEqual(capped_cloud.points_world_m.shape[0], 3)

    def test_high_disagreement_rejects_consensus_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _prepared_run(
                Path(tmp),
                frame_ids=(0,),
                width=4,
                height=4,
                vggt_depth=2.0,
                depth_pro_depth=4.0,
            )

            cloud = build_fused_point_cloud(
                run.run_dir,
                frame_cache=run.frames,
                proposal_cache=run.proposals,
                disagreement=run.disagreement,
                options=FusedWorldMapOptions(
                    export_world_map=True,
                    point_stride=1,
                    min_confidence=0.0,
                    max_relative_disagreement=0.25,
                    depth_source="consensus",
                ),
            )

            self.assertEqual(cloud.points_world_m.shape[0], 0)
            self.assertGreater(cloud.rejected_high_disagreement_ratio, 0.0)

    def test_sparse_occupancy_aggregates_without_filling_unknown_space(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[10, 20, 30], [30, 40, 50], [100, 120, 140]], dtype=np.uint8)
        confidence = np.array([0.5, 1.0, 0.25], dtype=np.float32)

        grid = build_sparse_occupancy(points, colors, confidence, voxel_size_m=0.1)

        self.assertEqual(grid.occupied_voxel_count, 2)
        self.assertEqual(grid.occupancy_count.tolist(), [2, 1])
        np.testing.assert_allclose(grid.confidence_mean, [0.75, 0.25])
        self.assertEqual(grid.color_mean_u8.tolist()[0], [20, 30, 40])

    def test_observed_voxel_mesh_omits_internal_faces_and_empty_mesh_is_explicit(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        confidence = np.ones((2,), dtype=np.float32)
        grid = build_sparse_occupancy(points, colors, confidence, voxel_size_m=1.0)

        mesh = build_observed_voxel_mesh(grid)
        empty = build_observed_voxel_mesh(
            build_sparse_occupancy(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
                voxel_size_m=1.0,
            )
        )

        self.assertEqual(mesh.triangle_count, 20)
        self.assertEqual(empty.triangle_count, 0)
        self.assertEqual(empty.vertices_world_m.shape[0], 0)


class _PreparedRun:
    def __init__(
        self,
        *,
        input_dir: Path,
        run_dir: Path,
        frames: object,
        keyframes: object,
        proposals: object,
        failures: list[object],
        disagreement: object | None,
    ) -> None:
        self.input_dir = input_dir
        self.run_dir = run_dir
        self.frames = frames
        self.keyframes = keyframes
        self.proposals = proposals
        self.failures = failures
        self.disagreement = disagreement


def _prepared_run(
    root: Path,
    *,
    frame_ids: tuple[int, ...],
    width: int,
    height: int,
    vggt_depth: float = 2.0,
    depth_pro_depth: float | None = None,
) -> _PreparedRun:
    input_dir = root / "input"
    run_dir = ensure_run_tree(root / "run")
    write_ppm_sequence(input_dir, count=max(frame_ids) + 1, width=width, height=height)
    failures = []
    frames = build_frame_cache(
        input_dir,
        run_dir,
        max_frames=max(frame_ids) + 1,
        failure_points=failures,
    )
    keyframes = select_keyframes(
        frames.records,
        run_dir,
        keyframe_stride=1,
        keyframe_max_count=max(frame_ids) + 1,
        failure_points=failures,
    )
    vggt = load_vggt_proposal_cache(
        write_fake_vggt_cache(
            root / "vggt_cache",
            frame_ids=frame_ids,
            width=width,
            height=height,
            depth_m=vggt_depth,
        )
    )
    depth_pro = (
        None
        if depth_pro_depth is None
        else load_depth_pro_proposal_cache(
            write_fake_depth_pro_cache(
                root / "depth_pro_cache",
                frame_ids=frame_ids,
                width=width,
                height=height,
                depth_m=depth_pro_depth,
            )
        )
    )
    statuses = write_teacher_statuses(
        run_dir, failures, vggt_result=vggt, depth_pro_result=depth_pro
    )
    proposals = write_proposal_cache(
        run_dir,
        teacher_statuses=statuses,
        frame_records=frames.records,
        keyframes=keyframes.keyframes,
        debug_geometry_mode="none",
        vggt_result=vggt,
        depth_pro_result=depth_pro,
    )
    disagreement = (
        None if depth_pro is None else write_teacher_disagreement(run_dir, proposal_cache=proposals)
    )
    return _PreparedRun(
        input_dir=input_dir,
        run_dir=run_dir,
        frames=frames,
        keyframes=keyframes,
        proposals=proposals,
        failures=failures,
        disagreement=disagreement,
    )


if __name__ == "__main__":
    unittest.main()
