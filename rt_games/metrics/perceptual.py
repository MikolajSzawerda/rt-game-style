from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import lpips
from piq import ssim

from rt_games.data.transforms import build_transform
from rt_games.models.vgg import build_vgg
from rt_games.utils.cache import ModelCache
from rt_games.utils.registry import METRICS_REGISTRY


def _to_tensor(img: Image.Image, size: Optional[int], device: str) -> torch.Tensor:
    return build_transform(size)(img).unsqueeze(0).to(device)


def _load_pair(content_path: Path, stylized_path: Path, size: Optional[int], device: str):
    content = Image.open(content_path).convert("RGB")
    stylized = Image.open(stylized_path).convert("RGB")
    return _to_tensor(content, size, device), _to_tensor(stylized, size, device)


@METRICS_REGISTRY.register("lpips")
def lpips_content(content_path: Path, stylized_path: Path, device: str = "cuda", size: Optional[int] = None, net: str = "alex"):
    loss_fn = lpips.LPIPS(net=net).to(device)
    c, s = _load_pair(content_path, stylized_path, size, device)
    with torch.no_grad():
        val = loss_fn(c * 2 - 1, s * 2 - 1)
    return float(val.mean().item())


@METRICS_REGISTRY.register("ssim")
def ssim_score(content_path: Path, stylized_path: Path, device: str = "cuda", size: Optional[int] = None):
    c, s = _load_pair(content_path, stylized_path, size, device)
    with torch.no_grad():
        val = ssim(s, c, data_range=1.0)
    return float(val.mean().item())


@METRICS_REGISTRY.register("content_loss")
def content_loss(content_path: Path, stylized_path: Path, device: str = "cuda", size: Optional[int] = None):
    vgg = ModelCache.get_vgg(build_vgg, device)
    c, s = _load_pair(content_path, stylized_path, size, device)
    with torch.no_grad():
        c_feats = vgg(c)["relu2_2"]
        s_feats = vgg(s)["relu2_2"]
    loss = F.mse_loss(s_feats, c_feats)
    return float(loss.item())

