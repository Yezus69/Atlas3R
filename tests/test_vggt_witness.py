from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.run_manifest import ensure_run_tree
from atlas3r.offline.stitching import estimate_sim3_umeyama
from atlas3r.offline.vggt_witness import (
    load_vggt_proposal_cache,
    run_vggt_witness,
    stitch_vggt_records,
)
from tests.helpers import write_fake_vggt_cache


class VggtWitnessTest(unittest.TestCase):
    def test_replay_cache_loads_without_heavy_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = write_fake_vggt_cache(Path(tmp) / "cache", frame_ids=(0, 1))

            result = load_vggt_proposal_cache(cache_dir)

            self.assertEqual(result.runtime_status, "replayed")
            self.assertTrue(result.has_geometry)
            self.assertEqual(len(result.camera_records), 2)
            self.assertEqual(len(result.depth_records), 2)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("vggt", sys.modules)

    def test_unavailable_vggt_status_has_install_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            failures = []

            result = run_vggt_witness(
                ensure_run_tree(Path(tmp) / "run"),
                frame_cache=_empty_frame_cache(),
                keyframes=(),
                options=_enabled_options(),
                failure_points=failures,
            )

            self.assertFalse(result.available)
            self.assertIn("Install VGGT", result.install_hint)
            self.assertTrue(failures)

    def test_synthetic_sim3_transform_is_recovered(self) -> None:
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        angle = np.deg2rad(30.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = (2.0 * (source @ rotation.T) + np.array([1.0, -0.5, 0.25])).astype(np.float32)

        estimate = estimate_sim3_umeyama(source, target, residual_threshold_m=1e-4)

        self.assertTrue(estimate.accepted)
        self.assertAlmostEqual(estimate.scale, 2.0, places=5)
        np.testing.assert_allclose(estimate.rotation, rotation, atol=1e-5)

    def test_stitching_scales_later_window_depth(self) -> None:
        camera_records, depth_records, arrays = _two_window_records(scale_gap=2.0)

        stitched, stitched_arrays, stats = stitch_vggt_records(
            camera_records, depth_records, arrays, stitch_mode="overlap-sim3"
        )

        self.assertEqual(stats["accepted_edge_count"], 1)
        self.assertEqual(stats["rejected_edge_count"], 0)
        self.assertGreater(float(stitched[3]["depth_scale_applied"]), 1.5)
        np.testing.assert_allclose(stitched_arrays["depth_3"], np.full((2, 2), 4.0), atol=1e-5)

    def test_bad_overlap_creates_new_submap(self) -> None:
        camera_records, depth_records, arrays = _two_window_records(scale_gap=2.0)
        camera_records[3]["T_window_camera"] = _pose(100.0, 100.0, 0.0).tolist()

        stitched, _arrays, stats = stitch_vggt_records(
            camera_records, depth_records, arrays, stitch_mode="overlap-sim3"
        )

        self.assertEqual(stats["accepted_edge_count"], 0)
        self.assertEqual(stats["rejected_edge_count"], 1)
        self.assertGreater(int(stitched[3]["pseudo_submap_id"]), 0)


def _empty_frame_cache() -> object:
    class EmptyFrameCache:
        frames: tuple[object, ...] = ()
        records: tuple[object, ...] = ()

    return EmptyFrameCache()


def _enabled_options() -> object:
    class Options:
        enabled = True
        proposal_cache = None
        repo_path = None
        checkpoint = None
        device = "cpu"
        image_size = 518
        window_size = 24
        window_overlap = 8
        max_keyframes = None
        stitch_mode = "overlap-sim3"

    return Options()


def _two_window_records(
    *, scale_gap: float
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    frame_centers = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (1.0, 1.0, 0.0),
    }
    records: list[dict[str, object]] = []
    depth_records: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, frame_id in enumerate((0, 1, 2)):
        _append_stitch_record(
            records, depth_records, arrays, index, frame_id, 0, frame_centers[frame_id]
        )
    for frame_id in (0, 1, 2, 3):
        index = len(records)
        local = tuple(value / scale_gap for value in frame_centers[frame_id])
        _append_stitch_record(records, depth_records, arrays, index, frame_id, 1, local)
    return records, depth_records, arrays


def _append_stitch_record(
    records: list[dict[str, object]],
    depth_records: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
    index: int,
    frame_id: int,
    window_id: int,
    center: tuple[float, float, float],
) -> None:
    depth_key = f"depth_{index}"
    sigma_key = f"sigma_{index}"
    records.append(
        {
            "frame_id": frame_id,
            "window_id": window_id,
            "T_window_camera": _pose(*center).tolist(),
            "T_world_camera": _pose(*center).tolist(),
            "camera_center_world_m": list(center),
            "depth_key": depth_key,
            "depth_sigma_key": sigma_key,
            "pseudo_submap_id": 0,
            "depth_scale_applied": 1.0,
        }
    )
    depth_records.append({"frame_id": frame_id, "depth_key": depth_key})
    arrays[depth_key] = np.full((2, 2), 2.0, dtype=np.float32)
    arrays[sigma_key] = np.full((2, 2), 0.1, dtype=np.float32)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return transform


if __name__ == "__main__":
    unittest.main()
