from __future__ import annotations

import math
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from atlas3r.offline.frame_cache import InputDataError, build_frame_cache
from atlas3r.offline.run_manifest import ensure_run_tree
from tests.helpers import write_ppm_sequence


class FrameCacheTest(unittest.TestCase):
    def test_frame_index_writes_stable_ids_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            run_dir = ensure_run_tree(root / "run")
            write_ppm_sequence(input_dir, count=3)
            failures = []

            result = build_frame_cache(input_dir, run_dir, max_frames=3, failure_points=failures)

            self.assertEqual(result.status, "complete")
            self.assertEqual([record.frame_id for record in result.records], [0, 1, 2])
            self.assertEqual(result.records[0].frame_path, "frames/images/frame_000000.ppm")
            self.assertEqual(result.records[0].camera.source, "guessed_from_image_size")
            for record in result.records:
                quality = record.quality
                self.assertTrue(math.isfinite(quality.blur_score))
                self.assertTrue(math.isfinite(quality.exposure_score))
                self.assertTrue(math.isfinite(quality.visual_change_score))

    def test_png_folder_decodes_when_optional_decoder_exists(self) -> None:
        if not any(find_spec(name) is not None for name in ("PIL", "imageio", "cv2")):
            self.skipTest("no optional PNG decoder is installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "pngs"
            run_dir = ensure_run_tree(root / "run")
            _write_png(input_dir / "000.png", np.zeros((3, 4, 3), dtype=np.uint8))
            _write_png(input_dir / "001.png", np.full((3, 4, 3), 128, dtype=np.uint8))
            failures = []

            result = build_frame_cache(input_dir, run_dir, max_frames=2, failure_points=failures)

            self.assertEqual(result.status, "complete")
            self.assertEqual([record.frame_id for record in result.records], [0, 1])
            self.assertEqual(result.records[0].frame_path, "frames/images/frame_000000.ppm")
            self.assertIn(result.records[0].decoder_name, {"pillow", "imageio.v3", "opencv"})
            self.assertTrue((run_dir / "frames" / "images" / "frame_000000.ppm").is_file())

    def test_bad_mp4_records_decoder_failure_without_crashing_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "bad.mp4"
            video.write_bytes(b"not a real mp4")
            run_dir = ensure_run_tree(root / "run")
            failures = []

            result = build_frame_cache(video, run_dir, max_frames=2, failure_points=failures)

            self.assertEqual(result.status, "unavailable")
            self.assertEqual(result.records, ())
            self.assertTrue(any(item.code == "video_decoder_unavailable" for item in failures))

    def test_missing_input_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_run_tree(Path(tmp) / "run")
            with self.assertRaisesRegex(InputDataError, "input path does not exist"):
                build_frame_cache(
                    Path(tmp) / "missing",
                    run_dir,
                    max_frames=1,
                    failure_points=[],
                )


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
