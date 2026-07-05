import os
import glob
from typing import List, Union
from .config import InferenceConfig

def discover_checkpoints(model_dir: str) -> List[str]:
    """Recursively discover all .pth checkpoints under model_dir."""
    if not os.path.isdir(model_dir):
        return []
    # Find all .pth files recursively
    pth_pattern = os.path.join(model_dir, "**", "*.pth")
    checkpoints = glob.glob(pth_pattern, recursive=True)
    # Sort them to have a deterministic order
    return sorted(checkpoints)

def get_config_for_checkpoint(checkpoint_path: str, device: str = "cuda", fp16: bool = True) -> InferenceConfig:
    """Dynamically determine InferenceConfig based on the checkpoint path/directory naming."""
    path_lower = os.path.abspath(checkpoint_path).lower()
    
    # Defaults matching the standard stage 2 configurations
    backbone = "resnet101_ibn_a"
    image_size = (384, 384)
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    transformer_type = "None"
    stride_size = (32, 32)
    neck_feat = "after"
    
    if "transreid" in path_lower or "transformer" in path_lower:
        backbone = "transformer"
        image_size = (256, 256)
        pixel_mean = (0.5, 0.5, 0.5)
        pixel_std = (0.5, 0.5, 0.5)
        transformer_type = "vit_base_patch16_224_TransReID"
        stride_size = (14, 14)
    elif "densenet169" in path_lower or "densenet" in path_lower:
        backbone = "densenet169_ibn_a"
    elif "resnext101" in path_lower or "resnext" in path_lower:
        backbone = "resnext101_ibn_a"
    elif "resnest101" in path_lower or "resnest" in path_lower or "s101_384" in path_lower:
        backbone = "resnest101"
    elif "se_resnet101" in path_lower or "se_resnet" in path_lower:
        backbone = "se_resnet101_ibn_a"
    elif "101a_384" in path_lower or "resnet101_ibn_a" in path_lower:
        backbone = "resnet101_ibn_a"
    elif "resnet50_ibn_a" in path_lower:
        backbone = "resnet50_ibn_a"
    elif "resnet50" in path_lower:
        backbone = "resnet50"
        
    return InferenceConfig(
        backbone=backbone,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        device=device,
        fp16=fp16,
        checkpoint_path=checkpoint_path,
        transformer_type=transformer_type,
        stride_size=stride_size,
        neck_feat=neck_feat
    )

def resolve_checkpoints_and_configs(
    model_dir: str = "trained_models",
    model_paths: Union[str, List[str]] = None,
    device: str = "cuda",
    fp16: bool = True
) -> List[InferenceConfig]:
    """Resolves and returns a list of InferenceConfigs for the selected/discovered checkpoints."""
    resolved_paths = []
    
    if model_paths is not None:
        if isinstance(model_paths, str):
            resolved_paths = [model_paths]
        else:
            resolved_paths = list(model_paths)
    else:
        resolved_paths = discover_checkpoints(model_dir)
        if not resolved_paths:
            raise FileNotFoundError(f"No checkpoints (.pth) discovered under model directory: {model_dir}")
            
    configs = []
    for path in resolved_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        configs.append(get_config_for_checkpoint(path, device=device, fp16=fp16))
        
    return configs
