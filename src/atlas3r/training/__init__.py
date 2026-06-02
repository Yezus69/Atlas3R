"""Training losses, distillation, and trainer entry points.

The package import stays dependency-safe: optional PyTorch modules are loaded
only when a training runner or model module explicitly requires them.
"""

from atlas3r.training.synthetic_depth_dataset import (
    SyntheticDepthSample,
    generate_synthetic_depth_samples,
    sample_to_student_clip,
)
from atlas3r.training.synthetic_overfit import SyntheticOverfitConfig, run_synthetic_overfit
from atlas3r.training.torch_runtime import (
    TorchDependencyError,
    require_torch,
    select_device,
    torch_available,
)

__all__ = [
    "SyntheticDepthSample",
    "SyntheticOverfitConfig",
    "TorchDependencyError",
    "generate_synthetic_depth_samples",
    "require_torch",
    "run_synthetic_overfit",
    "sample_to_student_clip",
    "select_device",
    "torch_available",
]
