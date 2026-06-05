import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6CStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6d_scheduler(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6D - Live-Ready Incremental Mapper Scheduler", next_task)
        self.assertIn("bounded memory", next_task)
        self.assertIn("phase6c_true_incremental_tsdf_backend_report.md", progress)


if __name__ == "__main__":
    unittest.main()
