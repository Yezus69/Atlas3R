from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.offline.colmap_import import read_colmap_text_model


class ColmapImportTest(unittest.TestCase):
    def test_parse_text_model_and_convert_pose_to_T_world_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_text_model(root)

            model = read_colmap_text_model(root)

            self.assertEqual(model.registered_image_count, 2)
            self.assertEqual(model.sparse_point_count, 2)
            first = model.images[0]
            self.assertEqual(first.frame_id, 0)
            np.testing.assert_allclose(first.camera_center_world_m, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(first.T_world_camera[:3, 3], [1.0, 2.0, 3.0])
            self.assertEqual(model.points3d[0].rgb, (255, 10, 20))
            self.assertEqual(model.points3d[0].track_length, 2)
            self.assertAlmostEqual(model.points3d[0].error, 0.5)

    def test_rejects_malformed_images_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_text_model(root)
            (root / "images.txt").write_text("1 1 0 0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "malformed images"):
                read_colmap_text_model(root)


def _write_text_model(root: Path) -> None:
    (root / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list",
                "1 SIMPLE_RADIAL 640 480 500 320 240 0.01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "images.txt").write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 -1 -2 -3 1 frame_000000.ppm",
                "0 0 1 1 1 2",
                "2 1 0 0 0 -2 -2 -3 1 frame_000001.ppm",
                "0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "points3D.txt").write_text(
        "\n".join(
            [
                "# Point list",
                "1 0 0 0 255 10 20 0.5 1 0 2 0",
                "2 1 0 0 30 40 50 1.5 1 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
