import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas3r.data import student_clip_from_frame_packets, write_frame_source_smoke_fixture
from atlas3r.models.student import ShapeOnlyStudentModel, StudentClipInput

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class StudentClipBridgeTest(unittest.TestCase):
    def test_smoke_frame_packets_convert_to_student_clip_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)

        clip = student_clip_from_frame_packets(packets, batch_id="unit-clip")

        self.assertIsInstance(clip, StudentClipInput)
        self.assertEqual(clip.frame_ids, (0, 1))
        self.assertEqual(clip.images_rgb.shape, (1, 2, 3, 2, 2))
        self.assertEqual(clip.intrinsics.shape, (1, 2, 3, 3))
        self.assertEqual(clip.metadata["batch_id"], "unit-clip")
        self.assertEqual(clip.metadata["source"], "frame_packets")
        self.assertEqual(clip.metadata["frame_count"], 2)
        self.assertEqual(clip.metadata["image_layout"], "B,T,3,H,W")
        self.assertIs(clip.metadata["input_order_preserved"], True)
        self.assertEqual(clip.metadata["source_formats"], ("npz",))
        np.testing.assert_allclose(clip.images_rgb[0, 0], packets[0].rgb_model)
        np.testing.assert_allclose(clip.intrinsics[0, 0], packets[0].K_model)

    def test_frame_id_and_image_order_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)

        clip = student_clip_from_frame_packets((packets[1], packets[0]))

        self.assertEqual(clip.frame_ids, (1, 0))
        np.testing.assert_allclose(clip.images_rgb[0, 0], packets[1].rgb_model)
        np.testing.assert_allclose(clip.images_rgb[0, 1], packets[0].rgb_model)

    def test_output_arrays_do_not_share_memory_with_frame_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)
        first_rgb_before = packets[0].rgb_model.copy()
        first_k_before = packets[0].K_model.copy()

        clip = student_clip_from_frame_packets(packets)
        clip.images_rgb[0, 0, 0, 0, 0] = -1.0
        clip.intrinsics[0, 0, 0, 0] = 999.0

        np.testing.assert_array_equal(packets[0].rgb_model, first_rgb_before)
        np.testing.assert_array_equal(packets[0].K_model, first_k_before)

    def test_duplicate_frame_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)
        duplicate = replace(packets[1], frame_id=packets[0].frame_id)

        with self.assertRaisesRegex(ValueError, r"frames\[1\]\.frame_id.*duplicate"):
            student_clip_from_frame_packets((packets[0], duplicate))

    def test_mismatched_rgb_model_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)
        narrow_rgb_model = packets[1].rgb_model[:, :, :1]
        mismatched = replace(packets[1], rgb_model=narrow_rgb_model)

        with self.assertRaisesRegex(ValueError, r"frames\[1\]\.rgb_model.*match"):
            student_clip_from_frame_packets((packets[0], mismatched))

    def test_empty_sequences_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frames.*at least one"):
            student_clip_from_frame_packets(())

    def test_non_frame_packet_items_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)

        with self.assertRaisesRegex(ValueError, r"frames\[1\].*FramePacket"):
            student_clip_from_frame_packets((packets[0], object()))

    def test_shape_only_student_forward_accepts_bridge_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packets = write_frame_source_smoke_fixture(tmp)
        clip = student_clip_from_frame_packets(packets)

        output = ShapeOnlyStudentModel().forward(clip)

        self.assertEqual(output.frame_ids, clip.frame_ids)
        self.assertEqual(output.depth_m.shape, (1, 2, 2, 2))
        self.assertIs(output.truth_boundary["shape_only"], True)
        self.assertIs(output.truth_boundary["learned_inference"], False)
        self.assertIs(output.truth_boundary["usable_for_mapping"], False)

    def test_student_clip_imports_do_not_load_heavy_dependencies(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r.data.student_clip\n"
            "import atlas3r.data as data\n"
            "getattr(data, 'student_clip_from_frame_packets')\n"
            "heavy = {'torch', 'tensorflow', 'jax', 'cv2', 'av', 'imageio', 'PIL'} & "
            "set(sys.modules)\n"
            "unexpected = heavy | ({'atlas3r.mapping'} & set(sys.modules)) | "
            "({'atlas3r.runtime'} & set(sys.modules))\n"
            "message = 'unexpected imports: ' + ', '.join(sorted(unexpected))\n"
            "raise SystemExit(message if unexpected else 0)\n"
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
