import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6EStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6f_mesh_updates(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6F - Live Mesh Chunk Updates", next_task)
        self.assertIn("runtime live-replay-recording", next_task)
        self.assertIn("phase6e_live_replay_scheduler_report.md", progress)


if __name__ == "__main__":
    unittest.main()
