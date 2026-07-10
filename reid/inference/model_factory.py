import os
import sys
import types
import collections.abc

from .utils import get_device

# Mock torch._six to support newer PyTorch versions (PyTorch 2.x+)
torch_six_mock = types.ModuleType("torch._six")
torch_six_mock.container_abcs = collections.abc
sys.modules["torch._six"] = torch_six_mock

import torch

from ..model.make_model import make_model
from .config import InferenceConfig


def build_model_from_config(config: InferenceConfig) -> torch.nn.Module:
    """Builds a model instance based on the InferenceConfig and loads the checkpoint weights."""
    # Convert configuration to the mock yacs node expected by make_model
    yacs_cfg = config.to_yacs_mock()

    # Instantiate model.
    # Pass a dummy class number (e.g. 1000), since during test/inference
    # the classifier layer is not used and not loaded from checkpoint.
    model = make_model(yacs_cfg, num_class=1000)

    # Load parameters
    if config.checkpoint_path:
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint weight file not found at: {config.checkpoint_path}"
            )
        model.load_param(config.checkpoint_path)

    # Place on device
    device = torch.device(get_device(config.device))
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    return model
