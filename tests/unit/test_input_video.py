from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from atlas3r.input.metadata import camera_from_focal
from atlas3r.input.video import inspect_video_input, load_ppm_sequence_frames, require_video_decoder


class InputVideoTest(unittest.TestCase):
    def test_ppm_sequence_frame_source_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_ppm(root / "000.ppm", np.zeros((2, 3, 3), dtype=np.uint8))
            _write_ppm(root / "001.ppm", np.full((2, 3, 3), 255, dtype=np.uint8))
            camera = camera_from_focal(width=3, height=2, fx=10.0, fy=10.0)

            inspection = inspect_video_input(root)
            frames = load_ppm_sequence_frames(root, camera)

            self.assertEqual(inspection.kind, "image_directory")
            self.assertEqual(inspection.frame_count, 2)
            self.assertEqual([frame.frame_id for frame in frames], [0, 1])
            self.assertEqual(frames[0].camera_metadata["source_format"], "ppm_sequence")

    def test_video_decoder_probe_is_explicit(self) -> None:
        if any(find_spec(name) is not None for name in ("imageio", "cv2")):
            require_video_decoder()
            return
        with self.assertRaisesRegex(RuntimeError, "video decoding requires imageio or opencv"):
            require_video_decoder()


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("expected RGB")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


if __name__ == "__main__":
    unittest.main()
