from typing import Optional, Tuple

import torch
from torchvision import transforms as T


def build_transform(image_size: Optional[int] = None):
    t: list = []
    if image_size is not None:
        t.append(T.Resize((image_size, image_size)))
    t.append(T.ToTensor())
    return T.Compose(t)


def to_tensor(img, image_size: Optional[int] = None) -> torch.Tensor:
    return build_transform(image_size)(img)


def to_pil(t: torch.Tensor):
    return T.ToPILImage()(t.clamp(0, 1))

