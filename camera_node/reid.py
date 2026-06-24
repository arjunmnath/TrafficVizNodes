"""
DMT Vehicle ReID Feature Extractor

Uses backbone architectures from AICITY2021_Track2_DMT (1st place, mAP 0.7445)
to extract discriminative vehicle re-identification embeddings.

Supported backbones:
  - resnet101_ibn_a     (2048-dim, 384×384 input)
  - resnext101_ibn_a    (2048-dim, 384×384 input)
  - se_resnet101_ibn_a  (2048-dim, 384×384 input)
  - densenet169_ibn_a   (1664-dim, 384×384 input)
  - resnest101          (2048-dim, 384×384 input)
  - transformer         (768-dim,  256×256 input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from functools import partial
from typing import Optional

# DMT backbone imports
from camera_node.dmt_backbones.resnet_ibn_a import resnet101_ibn_a
from camera_node.dmt_backbones.resnext_ibn import resnext101_ibn_a
from camera_node.dmt_backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from camera_node.dmt_backbones.densenet_ibn import densenet169_ibn_a
from camera_node.dmt_backbones.resnest import resnest101
from camera_node.dmt_backbones.vit_pytorch import vit_base_patch16_224_TransReID


# ---------- Model configuration registry ---------- #

MODEL_REGISTRY = {
    "resnet101_ibn_a": {
        "in_planes": 2048,
        "input_size": (384, 384),
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "is_transformer": False,
    },
    "resnext101_ibn_a": {
        "in_planes": 2048,
        "input_size": (384, 384),
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "is_transformer": False,
    },
    "se_resnet101_ibn_a": {
        "in_planes": 2048,
        "input_size": (384, 384),
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "is_transformer": False,
    },
    "densenet169_ibn_a": {
        "in_planes": 1664,
        "input_size": (384, 384),
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "is_transformer": False,
    },
    "resnest101": {
        "in_planes": 2048,
        "input_size": (384, 384),
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "is_transformer": False,
    },
    "transformer": {
        "in_planes": 768,
        "input_size": (256, 256),
        "pixel_mean": [0.5, 0.5, 0.5],
        "pixel_std": [0.5, 0.5, 0.5],
        "is_transformer": True,
    },
}


# ---------- Initialization helpers (from DMT) ---------- #

def _weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# ---------- CNN ReID wrapper (mirrors DMT Backbone) ---------- #

class _CNNReIDBackbone(nn.Module):
    """
    Wraps a CNN backbone with Global Average Pooling + BNNeck.
    During eval, returns the BNNeck-processed feature vector.
    """

    def __init__(self, model_name: str, in_planes: int):
        super().__init__()
        self.in_planes = in_planes

        # Build backbone
        if model_name == "resnet101_ibn_a":
            self.base = resnet101_ibn_a(last_stride=1, frozen_stages=-1)
        elif model_name == "resnext101_ibn_a":
            self.base = resnext101_ibn_a()
        elif model_name == "se_resnet101_ibn_a":
            self.base = se_resnet101_ibn_a(last_stride=1, frozen_stages=-1)
        elif model_name == "densenet169_ibn_a":
            self.base = densenet169_ibn_a()
        elif model_name == "resnest101":
            self.base = resnest101(last_stride=1)
        else:
            raise ValueError(f"Unknown CNN backbone: {model_name}")

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # BNNeck (batch-norm neck for feature normalization)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

    def forward(self, x):
        feat_map = self.base(x)
        global_feat = self.gap(feat_map)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat = self.bottleneck(global_feat)
        return feat

    def load_param(self, trained_path: str):
        """Load trained DMT checkpoint, skipping classifier weights."""
        param_dict = torch.load(trained_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for key in param_dict:
            if "classifier" in key or "arcface" in key:
                continue
            clean_key = key.replace("module.", "")
            if clean_key in self.state_dict():
                self.state_dict()[clean_key].copy_(param_dict[key])


# ---------- Transformer ReID wrapper (mirrors DMT build_transformer) ---------- #

class _TransformerReIDBackbone(nn.Module):
    """
    Wraps TransReID ViT-Base with BNNeck.
    During eval, returns the BNNeck-processed feature vector.
    """

    def __init__(self, in_planes: int, img_size: tuple):
        super().__init__()
        self.in_planes = in_planes
        self.base = vit_base_patch16_224_TransReID(
            img_size=img_size,
            stride_size=[14, 14],
            drop_path_rate=0.1,
            camera=0,
            view=0,
            local_feature=False,
            aie_xishu=2.5,
        )

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

    def forward(self, x):
        global_feat = self.base(x, cam_label=None, view_label=None)
        feat = self.bottleneck(global_feat)
        return feat

    def load_param(self, trained_path: str):
        """Load trained DMT checkpoint, skipping classifier weights."""
        param_dict = torch.load(trained_path, map_location="cpu")
        for key in param_dict:
            if "classifier" in key or "arcface" in key or "gap" in key:
                continue
            clean_key = key.replace("module.", "")
            if clean_key in self.state_dict():
                self.state_dict()[clean_key].copy_(param_dict[key])


# ---------- Public API ---------- #

class ReIDFeatureExtractor:
    """
    Vehicle ReID feature extractor using DMT trained models.

    Args:
        model_name: One of the keys in MODEL_REGISTRY.
        model_path: Path to the trained .pth checkpoint file.
        device: Device string ('cuda', 'cpu', or None for auto-detect).
        flip_augment: If True, average features from original + horizontally flipped
                      image for better accuracy (2x inference cost).
    """

    def __init__(
        self,
        model_name: str = "resnet101_ibn_a",
        model_path: str = "",
        device: Optional[str] = None,
        flip_augment: bool = False,
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.flip_augment = flip_augment
        cfg = MODEL_REGISTRY[model_name]
        self.in_planes = cfg["in_planes"]
        self.input_size = cfg["input_size"]

        # Device selection
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Build model
        if cfg["is_transformer"]:
            self.model = _TransformerReIDBackbone(
                in_planes=cfg["in_planes"],
                img_size=cfg["input_size"],
            )
        else:
            self.model = _CNNReIDBackbone(
                model_name=model_name,
                in_planes=cfg["in_planes"],
            )

        # Load trained weights
        if model_path:
            self.model.load_param(model_path)

        self.model = self.model.to(self.device).eval()

        # Preprocessing transform
        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"]),
        ])

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the output embedding."""
        return self.in_planes

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a ReID feature embedding from a BGR image crop.

        Args:
            crop: NumPy array of shape (H, W, 3) in BGR format (from OpenCV).

        Returns:
            L2-normalized feature vector of shape (embedding_dim,).
        """
        if crop.size == 0:
            return np.zeros(self.in_planes, dtype=np.float32)

        # BGR → RGB → PIL
        img = Image.fromarray(crop[..., ::-1])
        img_t = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img_t)

            if self.flip_augment:
                img_flip = torch.flip(img_t, [3])  # Horizontal flip
                feat_flip = self.model(img_flip)
                feat = (feat + feat_flip) / 2.0

        # L2 normalize
        feat = F.normalize(feat, p=2, dim=1)

        return feat.squeeze(0).cpu().numpy()
