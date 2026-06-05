from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atlas3r.offline.geometry_preview import GeometryPreviewResult
from atlas3r.offline.render_repair import write_render_repair_diagnostics
from atlas3r.offline.run_manifest import ensure_run_tree


class RenderRepairTest(unittest.TestCase):
    def test_missing_geometry_is_unavailable_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_run_tree(Path(tmp) / "run")
            failures = []
            geometry = GeometryPreviewResult(
                status="unavailable",
                geometry_npz_path="geometry/geometry_preview.npz",
                geometry_ply_path=None,
                point_count=0,
                observed_only=True,
                predicted_completion=False,
                measured_geometry=False,
                metric_scale_source="unknown",
            )

            result = write_render_repair_diagnostics(
                run_dir, geometry=geometry, keyframes=(), failure_points=failures
            )

            payload = json.loads((run_dir / result.diagnostics_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "unavailable")
            self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
