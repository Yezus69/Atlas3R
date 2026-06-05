import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Phase6EStatusTest(unittest.TestCase):
    def test_next_task_handoff_points_to_phase6g_acceleration(self) -> None:
        next_task = (ROOT / "docs" / "status" / "next_task.md").read_text(encoding="utf-8")
        progress = (ROOT / "docs" / "status" / "progress.md").read_text(encoding="utf-8")

        self.assertIn("Phase 6G - Accelerate Sparse Mapping And Mesh Chunk Updates", next_task)
        self.assertIn("runtime live-replay-recording", next_task)
        self.assertIn("phase6f_live_mesh_chunks_report.md", progress)


if __name__ == "__main__":
    unittest.main()
