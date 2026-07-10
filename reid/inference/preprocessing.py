import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F_t
from PIL import Image
from typing import Union, List, Any


def to_pil_image(img: Any, is_bgr: bool = True) -> Image.Image:
    """Converts various image formats (PIL, OpenCV, numpy, Tensor, path) into a standard RGB PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    elif isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image path not found: {img}")
        return Image.open(img).convert("RGB")

    elif isinstance(img, np.ndarray):
        # Handle numpy arrays
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                if is_bgr:
                    # OpenCV BGR image
                    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    return Image.fromarray(img)
            else:
                raise ValueError(f"Expected 3 channels in HWC numpy array, got shape {img.shape}")
        elif len(img.shape) == 2:
            # Grayscale numpy array
            return Image.fromarray(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported numpy array shape: {img.shape}")

    elif isinstance(img, torch.Tensor):
        # Handle torch tensors
        t = img.clone().detach().cpu()
        # Remove batch dimension if shape is (1, C, H, W)
        if len(t.shape) == 4 and t.shape[0] == 1:
            t = t.squeeze(0)

        if len(t.shape) == 3:
            # (C, H, W)
            if t.shape[0] in [1, 3]:
                if t.dtype in (torch.float32, torch.float64):
                    # Clamp to [0, 1] then rescale to [0, 255]
                    t = torch.clamp(t, 0.0, 1.0)
                    t = (t * 255.0).byte()
                return F_t.to_pil_image(t).convert("RGB")
            # (H, W, C)
            elif t.shape[2] in [1, 3]:
                if t.dtype in (torch.float32, torch.float64):
                    t = torch.clamp(t, 0.0, 1.0)
                    t = (t * 255.0).byte()
                t = t.permute(2, 0, 1)
                return F_t.to_pil_image(t).convert("RGB")
            else:
                raise ValueError(f"Expected 3 channels in tensor, got shape {t.shape}")
        elif len(t.shape) == 2:
            if t.dtype in (torch.float32, torch.float64):
                t = torch.clamp(t, 0.0, 1.0)
                t = (t * 255.0).byte()
            return F_t.to_pil_image(t).convert("RGB")
        else:
            raise ValueError(f"Unsupported torch Tensor shape: {img.shape}")

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def preprocess_images(
    images: Union[Any, List[Any]],
    image_size: tuple,
    pixel_mean: list,
    pixel_std: list,
    is_bgr: bool = True,
) -> torch.Tensor:
    """Preprocesses a single image or a list/batch of images into a preprocessed float32 PyTorch tensor.

    Performs resize (interpolation=BICUBIC), tensor conversion, and normalization.
    """
    # Create the validation transform exactly matching original datasets/make_dataloader.py
    # PIL.Image.BICUBIC is 3
    transform = T.Compose(
        [
            T.Resize(image_size, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std),
        ]
    )

    if not isinstance(images, (list, tuple)):
        # Single image
        pil_img = to_pil_image(images, is_bgr=is_bgr)
        tensor_img = transform(pil_img)  # shape (C, H, W)
        return tensor_img.unsqueeze(0)  # shape (1, C, H, W)
    else:
        # Batch of images
        tensor_list = []
        for img in images:
            pil_img = to_pil_image(img, is_bgr=is_bgr)
            tensor_list.append(transform(pil_img))
        return torch.stack(tensor_list, dim=0)  # shape (B, C, H, W)
