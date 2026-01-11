from typing import Optional

import torch
from torchvision import transforms as T


def build_transform(image_size: Optional[int] = None):
    """Build transform that resizes to square (for metrics needing matched sizes)."""
    t: list = []
    if image_size is not None:
        t.append(T.Resize((image_size, image_size)))
    t.append(T.ToTensor())
    return T.Compose(t)


def build_transform_fid(image_size: Optional[int] = None):
    """Build transform for FID that preserves aspect ratio (reference ArtFID behavior)."""
    t: list = []
    if image_size is not None:
        t.append(T.Resize(image_size))  # Single int preserves aspect ratio
    t.append(T.ToTensor())
    return T.Compose(t)


def to_tensor(img, image_size: Optional[int] = None) -> torch.Tensor:
    return build_transform(image_size)(img)


def to_pil(t: torch.Tensor):
    return T.ToPILImage()(t.clamp(0, 1))
