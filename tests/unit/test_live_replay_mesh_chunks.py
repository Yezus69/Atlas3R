import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas3r.mapping.mesh_chunks import load_mesh_chunk_npz
from atlas3r.runtime.live_replay import LiveReplayConfig, run_live_replay_recording
from tests.unit.test_live_replay_scheduler import _write_replay_recording

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class LiveReplayMeshChunksTest(unittest.TestCase):
    def test_live_replay_writes_loadable_mesh_chunks_npz_and_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            _write_replay_recording(recording, frame_count=4)
            output = root / "live"

            summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=recording,
                    output=output,
                    max_frames=4,
                    map_keyframe_stride=1,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    pixel_stride=1,
                    export_point_cloud=True,
                    export_mesh_chunks=True,
                    mesh_format="ply",
                )
            )
            manifest = json.loads(
                (output / "mesh_chunks" / "mesh_chunk_manifest.json").read_text(encoding="utf-8")
            )
            updates = _jsonl(output / "mesh_chunks" / "mesh_chunk_updates.jsonl")
            first_npz = output / "mesh_chunks" / str(updates[0]["payload_npz"])
            first_ply = output / "mesh_chunks" / str(updates[0]["payload_ply"])
            loaded_chunk = load_mesh_chunk_npz(first_npz)
            ply_counts = _ply_header_counts(first_ply)
            live_events = _jsonl(output / "live_replay_events.jsonl")

        self.assertGreater(summary["mesh_chunk_count"], 0)
        self.assertGreater(summary["mesh_chunk_update_count"], 0)
        self.assertGreater(summary["total_vertex_count"], 0)
        self.assertGreater(summary["total_triangle_count"], 0)
        self.assertEqual(summary["mesh_format"], "ply")
        self.assertTrue(summary["truth_boundary"]["observed_only"])
        self.assertFalse(summary["truth_boundary"]["rgb_only_mapping_ready"])
        self.assertEqual(manifest["format_name"], "atlas3r_observed_mesh_chunk_manifest")
        self.assertGreater(manifest["total_triangle_count"], 0)
        self.assertEqual([event["event_index"] for event in updates], list(range(len(updates))))
        self.assertTrue(any(event["stage_name"] == "mesh_chunk_update" for event in live_events))
        self.assertGreater(loaded_chunk.vertex_count, 0)
        self.assertGreater(loaded_chunk.triangle_count, 0)
        self.assertEqual(ply_counts["vertex"], loaded_chunk.vertex_count)
        self.assertEqual(ply_counts["face"], loaded_chunk.triangle_count)
        versions_by_chunk: dict[str, list[int]] = {}
        for event in updates:
            versions_by_chunk.setdefault(str(event["chunk_id"]), []).append(int(event["version"]))
            self.assertTrue(event["observed_only"])
            self.assertFalse(event["hidden_geometry_measured"])
        for versions in versions_by_chunk.values():
            self.assertEqual(versions, sorted(versions))

    def test_missing_pose_or_depth_emits_no_fake_mesh_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_depth = root / "missing_depth"
            missing_pose = root / "missing_pose"
            _write_replay_recording(missing_depth, frame_count=2, depth_present=False)
            _write_replay_recording(missing_pose, frame_count=2, pose_present=False)

            depth_summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=missing_depth,
                    output=root / "missing_depth_run",
                    max_frames=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    export_mesh_chunks=True,
                    mesh_format="ply",
                )
            )
            pose_summary = run_live_replay_recording(
                LiveReplayConfig(
                    recording=missing_pose,
                    output=root / "missing_pose_run",
                    max_frames=2,
                    voxel_size_m=0.25,
                    truncation_voxels=3.0,
                    export_mesh_chunks=True,
                    mesh_format="ply",
                )
            )

        self.assertEqual(depth_summary["mesh_chunk_count"], 0)
        self.assertEqual(depth_summary["total_triangle_count"], 0)
        self.assertEqual(pose_summary["mesh_chunk_count"], 0)
        self.assertEqual(pose_summary["total_triangle_count"], 0)

    def test_live_replay_mesh_flags_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "recording"
            output = root / "live"
            _write_replay_recording(recording, frame_count=3)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atlas3r",
                    "runtime",
                    "live-replay-recording",
                    "--recording",
                    str(recording),
                    "--output",
                    str(output),
                    "--max-frames",
                    "3",
                    "--voxel-size-m",
                    "0.25",
                    "--truncation-voxels",
                    "3.0",
                    "--pixel-stride",
                    "1",
                    "--export-mesh-chunks",
                    "--mesh-format",
                    "ply",
                ],
                check=False,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            summary = json.loads((output / "live_replay_summary.json").read_text("utf-8"))
            manifest_exists = (output / "mesh_chunks" / "mesh_chunk_manifest.json").is_file()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(summary["total_triangle_count"], 0)
        self.assertTrue(manifest_exists)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ply_header_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "end_header":
            break
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            counts[parts[1]] = int(parts[2])
    return counts


if __name__ == "__main__":
    unittest.main()
