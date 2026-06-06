from __future__ import annotations

import sys
import unittest

from atlas3r.teachers import TeacherUnavailableError, get_teacher_adapter, get_teacher_status
from atlas3r.teachers.registry import list_teacher_statuses


class TeacherRegistryTest(unittest.TestCase):
    def test_known_teachers_are_dependency_safe_and_unavailable(self) -> None:
        statuses = list_teacher_statuses()

        self.assertGreaterEqual(len(statuses), 5)
        for status in statuses:
            self.assertTrue(status.install_hint)
            if status.name not in {"vggt", "depth_pro"}:
                self.assertFalse(status.available)
        self.assertNotIn("torch", sys.modules)

    def test_unavailable_adapter_raises_with_install_hint(self) -> None:
        adapter = get_teacher_adapter("Depth-Pro")
        status = get_teacher_status("depth_pro")

        self.assertEqual(status.name, "depth_pro")
        with self.assertRaises(TeacherUnavailableError) as ctx:
            adapter.predict(())
        self.assertIn("Install Depth Pro", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
