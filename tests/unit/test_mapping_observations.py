import unittest

import numpy as np

from atlas3r.data.synthetic_cube_room import create_synthetic_cube_room_scene
from atlas3r.mapping.observations import (
    DepthObservation,
    depth_observation_from_synthetic_frame,
)


class DepthObservationTest(unittest.TestCase):
    def _valid_observation(self) -> DepthObservation:
        scene = create_synthetic_cube_room_scene()
        return depth_observation_from_synthetic_frame(scene.frames[0])

    def test_valid_synthetic_frame_observation_is_accepted(self) -> None:
        scene = create_synthetic_cube_room_scene()
        frame = scene.frames[0]

        observation = depth_observation_from_synthetic_frame(frame)

        self.assertEqual(observation.frame_id, frame.frame_id)
        self.assertIs(observation.camera, frame.camera)
        self.assertIs(observation.pose, frame.pose)
        np.testing.assert_array_equal(observation.depth_m, frame.depth_m)
        np.testing.assert_array_equal(observation.depth_sigma_m, frame.depth_sigma_m)
        np.testing.assert_array_equal(observation.confidence, frame.confidence)
        self.assertTrue(np.all(observation.static_mask))
        np.testing.assert_array_equal(observation.object_id, frame.object_id)
        self.assertEqual(observation.source, "synthetic_cube_room")

    def test_invalid_depth_shape_is_rejected(self) -> None:
        observation = self._valid_observation()

        with self.assertRaisesRegex(ValueError, "depth_m.*shape"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m[:-1, :],
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
            )

    def test_negative_depth_is_rejected(self) -> None:
        observation = self._valid_observation()
        depth_m = observation.depth_m.copy()
        depth_m[0, 0] = -0.1

        with self.assertRaisesRegex(ValueError, "depth_m.*non-negative"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
            )

    def test_confidence_outside_unit_range_is_rejected(self) -> None:
        observation = self._valid_observation()
        confidence = observation.confidence.copy()
        confidence[0, 0] = 1.1

        with self.assertRaisesRegex(ValueError, r"confidence.*\[0, 1\]"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=confidence,
            )

    def test_optional_object_id_wrong_shape_or_dtype_is_rejected(self) -> None:
        observation = self._valid_observation()

        with self.assertRaisesRegex(ValueError, "object_id.*shape"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
                object_id=observation.object_id[:-1, :],
            )

        with self.assertRaisesRegex(ValueError, "object_id.*integer"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
                object_id=observation.object_id.astype(np.float32),
            )

    def test_optional_rgb_wrong_shape_or_dtype_is_rejected(self) -> None:
        observation = self._valid_observation()
        height, width = observation.depth_m.shape

        with self.assertRaisesRegex(ValueError, "rgb_u8.*uint8 HxWx3"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
                rgb_u8=np.zeros((height, width, 3), dtype=np.float32),
            )

        with self.assertRaisesRegex(ValueError, "rgb_u8.*uint8 HxWx3"):
            DepthObservation(
                frame_id=observation.frame_id,
                camera=observation.camera,
                pose=observation.pose,
                depth_m=observation.depth_m,
                depth_sigma_m=observation.depth_sigma_m,
                confidence=observation.confidence,
                rgb_u8=np.zeros((height, width), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
