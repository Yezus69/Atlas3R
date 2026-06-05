import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6DStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6e_scheduler(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6E - Live Capture Replay Scheduler", next_task)
        self.assertIn("actual camera adapter", next_task)
        self.assertIn("phase6d_sparse_block_tsdf_live_replay_report.md", progress)


if __name__ == "__main__":
    unittest.main()
