import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6BStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6c_accelerated_mapper(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6C - Accelerated Incremental Mapper Prototype", next_task)
        self.assertIn("bounded incremental mapper", next_task)
        self.assertIn("phase6b_real_capture_incremental_mapper_report.md", progress)


if __name__ == "__main__":
    unittest.main()
