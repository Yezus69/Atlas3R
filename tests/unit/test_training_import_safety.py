import os
import subprocess
import sys
import unittest
from pathlib import Path

from atlas3r.training.torch_runtime import torch_available

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class TrainingImportSafetyTest(unittest.TestCase):
    def test_torch_available_returns_bool_without_importing_torch(self) -> None:
        self.assertIsInstance(torch_available(), bool)

    def test_training_package_import_does_not_load_torch_or_heavy_dependencies(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        code = (
            "import sys\n"
            "import atlas3r\n"
            "import atlas3r.data\n"
            "import atlas3r.models.student\n"
            "import atlas3r.training\n"
            "import atlas3r.training.checkpoint_inference\n"
            "import atlas3r.mapping.checkpoint_tsdf_smoke\n"
            "from atlas3r.training.torch_runtime import torch_available\n"
            "available = torch_available()\n"
            "unexpected = {'torch', 'tensorflow', 'jax', 'cv2', 'PIL', 'imageio', 'av'} & "
            "set(sys.modules)\n"
            "message = 'unexpected imports: ' + ', '.join(sorted(unexpected))\n"
            "raise SystemExit(message if unexpected else 0)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
