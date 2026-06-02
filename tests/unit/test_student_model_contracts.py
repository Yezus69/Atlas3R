import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from atlas3r.models.student import (
    CAMERA_COORDINATE_FRAME,
    ShapeOnlyStudentModel,
    StudentClipInput,
    StudentForwardOutput,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _K() -> np.ndarray:
    return np.array(
        [
            [120.0, 0.0, 2.0],
            [0.0, 120.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _clip_input(
    *,
    batch_size: int = 2,
    frame_count: int = 3,
    height: int = 4,
    width: int = 5,
) -> StudentClipInput:
    return StudentClipInput(
        frame_ids=tuple(range(10, 10 + frame_count)),
        images_rgb=np.zeros((batch_size, frame_count, 3, height, width), dtype=np.uint8),
        intrinsics=_K(),
        metadata={"source": "unit-test"},
    )


def _output() -> StudentForwardOutput:
    return ShapeOnlyStudentModel().forward(_clip_input())


class StudentModelContractsTest(unittest.TestCase):
    def test_valid_student_clip_input_is_accepted(self) -> None:
        clip = _clip_input()

        self.assertEqual(clip.batch_size, 2)
        self.assertEqual(clip.frame_count, 3)
        self.assertEqual(clip.height, 4)
        self.assertEqual(clip.width, 5)
        self.assertEqual(clip.coordinate_frame, CAMERA_COORDINATE_FRAME)
        self.assertEqual(clip.batched_intrinsics().shape, (2, 3, 3, 3))

    def test_invalid_image_layout_and_channel_count_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "images_rgb.*BxTx3xHxW"):
            StudentClipInput(
                frame_ids=(0,),
                images_rgb=np.zeros((1, 3, 4, 5), dtype=np.uint8),
                intrinsics=_K(),
            )

        with self.assertRaisesRegex(ValueError, "images_rgb.*channel count"):
            StudentClipInput(
                frame_ids=(0,),
                images_rgb=np.zeros((1, 1, 1, 4, 5), dtype=np.uint8),
                intrinsics=_K(),
            )

    def test_invalid_intrinsics_are_rejected(self) -> None:
        K = _K()
        K[0, 0] = 0.0

        with self.assertRaisesRegex(ValueError, "intrinsics.*focal lengths"):
            StudentClipInput(
                frame_ids=(0, 1),
                images_rgb=np.zeros((1, 2, 3, 4, 5), dtype=np.uint8),
                intrinsics=K,
            )

    def test_shape_only_forward_returns_expected_shapes(self) -> None:
        clip = _clip_input(batch_size=2, frame_count=3, height=4, width=5)
        output = ShapeOnlyStudentModel().forward(clip)

        self.assertEqual(output.depth_m.shape, (2, 3, 4, 5))
        self.assertEqual(output.depth_sigma_m.shape, (2, 3, 4, 5))
        self.assertEqual(output.confidence.shape, (2, 3, 4, 5))
        self.assertEqual(output.dynamic_probability.shape, (2, 3, 4, 5))
        self.assertEqual(output.normals_camera.shape, (2, 3, 3, 4, 5))
        self.assertEqual(output.pointmap_camera_m.shape, (2, 3, 3, 4, 5))
        self.assertEqual(output.T_world_camera.shape, (2, 3, 4, 4))
        self.assertEqual(output.intrinsics.shape, (2, 3, 3, 3))
        np.testing.assert_array_equal(output.intrinsics[0, 0], _K())

    def test_output_validation_rejects_bad_confidence_and_probability_ranges(self) -> None:
        output = _output()
        bad_confidence = output.confidence.copy()
        bad_confidence[0, 0, 0, 0] = 1.1

        with self.assertRaisesRegex(ValueError, r"confidence.*\[0, 1\]"):
            StudentForwardOutput(
                frame_ids=output.frame_ids,
                depth_m=output.depth_m,
                depth_sigma_m=output.depth_sigma_m,
                confidence=bad_confidence,
                dynamic_probability=output.dynamic_probability,
                normals_camera=output.normals_camera,
                pointmap_camera_m=output.pointmap_camera_m,
                T_world_camera=output.T_world_camera,
                intrinsics=output.intrinsics,
                truth_boundary=output.truth_boundary,
            )

        bad_dynamic_probability = output.dynamic_probability.copy()
        bad_dynamic_probability[0, 0, 0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, r"dynamic_probability.*\[0, 1\]"):
            StudentForwardOutput(
                frame_ids=output.frame_ids,
                depth_m=output.depth_m,
                depth_sigma_m=output.depth_sigma_m,
                confidence=output.confidence,
                dynamic_probability=bad_dynamic_probability,
                normals_camera=output.normals_camera,
                pointmap_camera_m=output.pointmap_camera_m,
                T_world_camera=output.T_world_camera,
                intrinsics=output.intrinsics,
                truth_boundary=output.truth_boundary,
            )

    def test_identity_transforms_pass_output_validation(self) -> None:
        output = _output()

        expected_identity = np.eye(4, dtype=np.float32)
        for batch_index in range(output.T_world_camera.shape[0]):
            for frame_index in range(output.T_world_camera.shape[1]):
                np.testing.assert_array_equal(
                    output.T_world_camera[batch_index, frame_index],
                    expected_identity,
                )

    def test_truth_boundary_marks_shape_only_stub(self) -> None:
        output = _output()

        self.assertIs(output.truth_boundary["shape_only"], True)
        self.assertIs(output.truth_boundary["learned_inference"], False)
        self.assertIs(output.truth_boundary["usable_for_mapping"], False)
        self.assertIs(output.truth_boundary["performance_report"], False)
        self.assertIs(output.truth_boundary["accuracy_report"], False)

    def test_student_package_import_does_not_load_heavy_ml_dependencies(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r.models.student\n"
            "heavy = {'torch', 'tensorflow', 'jax', 'cv2'} & set(sys.modules)\n"
            "message = 'imported heavy modules: ' + ', '.join(sorted(heavy))\n"
            "raise SystemExit(message if heavy else 0)\n"
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
