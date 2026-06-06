from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.colmap_import import (
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    ColmapSparseModel,
)
from atlas3r.offline.trajectory_alignment import (
    align_colmap_to_vggt,
    estimate_sim3,
    write_trajectory_alignment,
)


class TrajectoryAlignmentTest(unittest.TestCase):
    def test_sim3_recovers_known_scale_and_translation(self) -> None:
        source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        target = source * 2.0 + np.asarray([1.0, -2.0, 0.5], dtype=np.float32)

        transform = estimate_sim3(source, target)

        self.assertAlmostEqual(float(transform["scale"]), 2.0, places=5)
        np.testing.assert_allclose(transform["translation"], [1.0, -2.0, 0.5], atol=1e-5)

    def test_alignment_writes_aligned_outputs_and_reduces_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = _model()
            vggt = _vggt_records(scale=2.0, translation=np.asarray([1.0, 0.0, 0.0]))

            result = write_trajectory_alignment(tmp, model=model, vggt_camera_records=vggt)

            self.assertEqual(result.status, "available")
            self.assertEqual(result.common_frame_count, 4)
            self.assertIsNotNone(result.camera_center_rmse_m)
            self.assertLess(result.camera_center_rmse_m, 1e-5)
            self.assertTrue((Path(tmp) / "classical" / "trajectory_alignment.json").is_file())
            self.assertTrue(
                (Path(tmp) / "classical" / "aligned_colmap_sparse_points.ply").is_file()
            )

    def test_insufficient_common_frames_fails_clearly(self) -> None:
        result = align_colmap_to_vggt(_model(), vggt_camera_records=_vggt_records()[:2])

        self.assertEqual(result.status, "unavailable")
        self.assertIn("need at least", result.reason)

    def test_nonfinite_inputs_rejected(self) -> None:
        source = np.asarray([[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]], dtype=np.float32)
        target = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_sim3(source, target)


def _model() -> ColmapSparseModel:
    centers = [
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    ]
    images = []
    for index, center in enumerate(centers):
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[:3, 3] = center
        images.append(
            ColmapImage(
                image_id=index + 1,
                qvec=(1.0, 0.0, 0.0, 0.0),
                tvec=tuple((-center).tolist()),
                camera_id=1,
                name=f"frame_{index:06d}.ppm",
                T_world_camera=T_world_camera,
                camera_center_world_m=center,
            )
        )
    points = (
        ColmapPoint3D(1, (0.0, 0.0, 1.0), (255, 0, 0), 0.1, 2),
        ColmapPoint3D(2, (1.0, 0.0, 1.0), (0, 255, 0), 0.1, 2),
    )
    return ColmapSparseModel(
        cameras=(ColmapCamera(1, "SIMPLE_RADIAL", 4, 3, (4.0, 2.0, 1.0, 0.0)),),
        images=tuple(images),
        points3d=points,
        source_path="fixture",
    )


def _vggt_records(
    *, scale: float = 1.0, translation: np.ndarray | None = None
) -> tuple[dict[str, object], ...]:
    offset = (
        np.zeros((3,), dtype=np.float32) if translation is None else translation.astype(np.float32)
    )
    rows = []
    for index, center in enumerate(
        [
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        ]
    ):
        rows.append(
            {"frame_id": index, "camera_center_world_m": (center * scale + offset).tolist()}
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
