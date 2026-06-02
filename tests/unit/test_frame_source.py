import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas3r.api import FramePacket
from atlas3r.data.frame_source import (
    NPZFrameSource,
    PPMSequenceFrameSource,
    load_npz_clip_frames,
    load_ppm_sequence_frames,
    write_frame_source_smoke_fixture,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _K() -> np.ndarray:
    return np.array(
        [
            [64.0, 0.0, 1.0],
            [0.0, 64.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rgb_clip() -> np.ndarray:
    return np.array(
        [
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            ],
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            ],
        ],
        dtype=np.uint8,
    )


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise AssertionError("test PPM helper expects RGB")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


class FrameSourceTest(unittest.TestCase):
    def test_valid_npz_clip_to_frame_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.npz"
            rgb_u8 = _rgb_clip()
            np.savez(clip_path, rgb_u8=rgb_u8, K=_K())

            packets = tuple(NPZFrameSource(clip_path, frame_id_start=10).frames())

        self.assertEqual(len(packets), 2)
        self.assertTrue(all(isinstance(packet, FramePacket) for packet in packets))
        self.assertEqual([packet.frame_id for packet in packets], [10, 11])
        self.assertEqual([packet.timestamp_ns for packet in packets], [0, 0])
        self.assertEqual(packets[0].camera_metadata["source_format"], "npz")
        np.testing.assert_array_equal(packets[0].rgb_u8, rgb_u8[0])
        expected_rgb_model = np.moveaxis(rgb_u8[0], 2, 0).astype(np.float32) / 255.0
        np.testing.assert_array_equal(packets[0].rgb_model, expected_rgb_model)
        np.testing.assert_allclose(packets[0].K_original, _K())
        np.testing.assert_allclose(packets[0].K_model, _K())
        np.testing.assert_array_equal(packets[0].resize_transform, np.eye(3, dtype=np.float32))

    def test_valid_ppm_sequence_to_frame_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sequence_dir = Path(tmp)
            second = _rgb_clip()[1]
            first = _rgb_clip()[0]
            _write_ppm(sequence_dir / "b.ppm", second)
            _write_ppm(sequence_dir / "a.ppm", first)
            np.savez(sequence_dir / "intrinsics.npz", K=np.stack([_K(), _K()]))

            packets = tuple(PPMSequenceFrameSource(sequence_dir, frame_id_start=3).frames())

        self.assertEqual([packet.frame_id for packet in packets], [3, 4])
        np.testing.assert_array_equal(packets[0].rgb_u8, first)
        np.testing.assert_array_equal(packets[1].rgb_u8, second)
        self.assertEqual(packets[0].camera_metadata["source_format"], "ppm_sequence")
        self.assertTrue(str(packets[0].camera_metadata["source_file"]).endswith("a.ppm"))
        np.testing.assert_allclose(packets[1].K_model, _K())

    def test_supplied_ppm_intrinsics_are_accepted_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sequence_dir = Path(tmp)
            _write_ppm(sequence_dir / "000.ppm", _rgb_clip()[0])

            packets = load_ppm_sequence_frames(sequence_dir, K=_K())

        self.assertEqual(len(packets), 1)
        np.testing.assert_allclose(packets[0].K_original, _K())

    def test_smoke_fixture_writes_and_validates_npz_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)
            clip_path = Path(tmp) / "frame_source_smoke.npz"
            self.assertTrue(clip_path.is_file())

        self.assertEqual([packet.frame_id for packet in packets], [0, 1])
        self.assertEqual(packets[0].camera_metadata["source_format"], "npz")

    def test_invalid_rgb_layout_and_channel_count_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "bad_layout.npz"
            np.savez(clip_path, rgb_u8=np.zeros((2, 3, 3), dtype=np.uint8), K=_K())
            with self.assertRaisesRegex(ValueError, "rgb_u8.*TxHxWx3"):
                load_npz_clip_frames(clip_path)

            bad_channels = Path(tmp) / "bad_channels.npz"
            np.savez(bad_channels, rgb_u8=np.zeros((1, 2, 3, 1), dtype=np.uint8), K=_K())
            with self.assertRaisesRegex(ValueError, "rgb_u8.*TxHxWx3"):
                load_npz_clip_frames(bad_channels)

    def test_invalid_intrinsics_rejected_through_existing_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "bad_K.npz"
            K = _K()
            K[0, 0] = 0.0
            np.savez(clip_path, rgb_u8=_rgb_clip(), K=K)

            with self.assertRaisesRegex(ValueError, "K.*focal lengths"):
                load_npz_clip_frames(clip_path)

    def test_frame_source_imports_do_not_load_heavy_or_student_dependencies(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r.data.frame_source\n"
            "heavy = {'cv2', 'av', 'imageio', 'PIL'} & set(sys.modules)\n"
            "student_loaded = 'atlas3r.models.student' in sys.modules\n"
            "if heavy or student_loaded:\n"
            "    raise SystemExit("
            "'unexpected imports: ' + ', '.join(sorted(heavy)) + "
            "('; student' if student_loaded else '')"
            ")\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
