from __future__ import annotations

import unittest

import numpy as np

from atlas3r.contracts import (
    COORDINATE_FRAME_NAME,
    CameraModel,
    DepthProposal,
    FramePacket,
    MapArtifact,
    ObjectMaskProposal,
    PoseEstimate,
    TeacherProposal,
    TruthBoundary,
    WorldState,
)


class ContractSerializationTest(unittest.TestCase):
    def test_core_contracts_validate_and_roundtrip(self) -> None:
        K = np.array([[50.0, 0.0, 1.0], [0.0, 50.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        camera = CameraModel(width=3, height=2, K=K, confidence=1.0, source="metadata")
        pose = PoseEstimate(
            frame_id=7,
            timestamp_ns=100,
            T_world_camera=np.eye(4, dtype=np.float32),
            covariance_6x6=np.eye(6, dtype=np.float32),
            confidence=0.8,
            tracking_state="OK",
            scale_source="known_anchor",
            diagnostics={"source": "test"},
        )
        truth = TruthBoundary.teacher_pseudo("depth_pro_metric_depth")
        depth = DepthProposal(
            frame_id=7,
            teacher_name="depth_pro",
            camera=camera,
            pose=pose,
            depth_m=np.ones((2, 3), dtype=np.float32),
            depth_sigma_m=np.full((2, 3), 0.1, dtype=np.float32),
            confidence=np.full((2, 3), 0.9, dtype=np.float32),
            truth_boundary=truth,
        )
        mask = ObjectMaskProposal(
            frame_id=7,
            teacher_name="sam_dino",
            mask_u8=np.ones((2, 3), dtype=np.uint8),
            confidence=0.5,
            label="object",
            object_id=1,
            truth_boundary=truth,
        )
        artifact = MapArtifact(
            artifact_type="voxel",
            path="artifacts/world.npz",
            coordinate_frame=COORDINATE_FRAME_NAME,
            source_frame_ids=(7,),
            voxel_size_m=0.05,
            observed_coverage_estimate=0.2,
            mean_uncertainty_m=0.1,
            p95_uncertainty_m=0.2,
            truth_boundary=truth,
            metadata={"format": "npz"},
        )
        proposal = TeacherProposal(
            teacher_name="depth_pro",
            frame_ids=(7,),
            depth_proposals=(depth,),
            object_masks=(mask,),
            truth_boundary=truth,
            status="unavailable",
            metadata={"witness": True},
        )
        world = WorldState(
            world_id="world-test",
            poses=(pose,),
            cameras=(camera,),
            teacher_proposals=(proposal,),
            map_artifacts=(artifact,),
            truth_boundary=truth,
            metadata={},
        )

        self.assertEqual(CameraModel.from_dict(camera.to_dict()).width, 3)
        self.assertEqual(PoseEstimate.from_dict(pose.to_dict()).frame_id, 7)
        self.assertEqual(DepthProposal.from_dict(depth.to_dict()).teacher_name, "depth_pro")
        self.assertEqual(TeacherProposal.from_dict(proposal.to_dict()).frame_ids, (7,))
        self.assertEqual(WorldState.from_dict(world.to_dict()).world_id, "world-test")

    def test_frame_packet_normalizes_rgb_model(self) -> None:
        K = np.eye(3, dtype=np.float32)
        frame = FramePacket(
            frame_id=0,
            timestamp_ns=0,
            rgb_u8=np.full((2, 3, 3), 255, dtype=np.uint8),
            K_original=K,
            K_model=K,
            resize_transform=K,
            camera_metadata={"source_format": "test"},
        )

        self.assertEqual(frame.rgb_model_f32().shape, (3, 2, 3))
        self.assertEqual(FramePacket.from_dict(frame.to_dict()).width, 3)

    def test_pseudo_truth_cannot_claim_measured_geometry(self) -> None:
        with self.assertRaises(ValueError):
            TruthBoundary(
                label_type="unanchored_mp4_pseudo",
                metric_scale_source="unanchored_rgb_prior",
                measured_geometry=True,
            )


if __name__ == "__main__":
    unittest.main()
