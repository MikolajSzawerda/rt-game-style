from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from rt_games.config import DEFAULT_CONFIG
from rt_games.data.transforms import build_transform
from rt_games.models.inception import load_art_inception, load_inception
from rt_games.models.vgg import build_vgg
from rt_games.utils.cache import ModelCache
from rt_games.utils.registry import METRICS_REGISTRY

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    n, c, h, w = feat.size()
    f = feat.view(n, c, -1)
    g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    return g


@METRICS_REGISTRY.register("gram_loss")
def gram_loss(style_path: Path, stylized_path: Path, device: str = "cuda", size: Optional[int] = None):
    vgg = ModelCache.get_vgg(build_vgg, device)
    t = build_transform(size)
    with torch.no_grad():
        style = t(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        stylized = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)
        s_feats = gram_matrix(vgg(style)["relu3_3"])
        y_feats = gram_matrix(vgg(stylized)["relu3_3"])
        loss = F.mse_loss(y_feats, s_feats)
    return float(loss.item())


def _load_inception(device: str, use_art: bool, cfg=DEFAULT_CONFIG):
    if use_art:
        return ModelCache.get_inception(
            lambda d: load_art_inception(d, cfg.art_inception_url), device, art=True
        )
    return ModelCache.get_inception(lambda d: load_inception(d), device, art=False)


@METRICS_REGISTRY.register("fid")
def fid_score(
    real_dir: Path,
    fake_dir: Path,
    device: str = "cuda",
    use_art_inception: bool = True,
    sample_limit: Optional[int] = None,
):
    if calculate_metrics is None:
        raise ImportError("torch-fidelity is required for FID.")
    kwargs = {
        "input1": str(real_dir),
        "input2": str(fake_dir),
        "device": device,
        "fid": True,
        "feature_extractor": _load_inception(device, use_art_inception),
    }
    if sample_limit is not None:
        kwargs["sample_limit"] = sample_limit
    result = calculate_metrics(**kwargs)
    return float(result["frechet_inception_distance"])


def _get_features(paths: List[Path], model: nn.Module, device: str, size: Optional[int]) -> torch.Tensor:
    t = build_transform(size)
    feats = []
    with torch.no_grad():
        for p in paths:
            x = t(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model._forward_impl(x)[0].view(1, -1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@METRICS_REGISTRY.register("sifid")
def sifid_score(
    style_dir: Path,
    stylized_dir: Path,
    device: str = "cuda",
    size: Optional[int] = None,
    use_art_inception: bool = True,
    sample_limit: Optional[int] = None,
):
    model = _load_inception(device, use_art_inception)
    style_paths = sorted(list(style_dir.glob("*")))
    stylized_paths = sorted(list(stylized_dir.glob("*")))
    if sample_limit:
        style_paths = style_paths[:sample_limit]
        stylized_paths = stylized_paths[:sample_limit]
    n = min(len(style_paths), len(stylized_paths))
    if n == 0:
        raise ValueError("No images for SIFID.")
    style_feats = _get_features(style_paths[:n], model, device, size)
    styl_feats = _get_features(stylized_paths[:n], model, device, size)
    mu1, mu2 = style_feats.mean(0), styl_feats.mean(0)
    sigma1 = torch.from_numpy(np.cov(style_feats.numpy(), rowvar=False))
    sigma2 = torch.from_numpy(np.cov(styl_feats.numpy(), rowvar=False))
    diff = mu1 - mu2
    # FID formula
    covmean = torch.linalg.sqrtm((sigma1 @ sigma2).numpy()).real
    covmean = torch.from_numpy(covmean)
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid.item())


@METRICS_REGISTRY.register("cfsd")
def cfsd(content_path: Path, stylized_path: Path, device: str = "cuda", size: Optional[int] = None):
    """
    Simplified CFSD: compare patch cosine similarity on VGG relu3_3.
    """
    vgg = ModelCache.get_vgg(build_vgg, device)
    t = build_transform(size)
    with torch.no_grad():
        c = t(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        s = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)
        c_feat = vgg(c)["relu3_3"]
        s_feat = vgg(s)["relu3_3"]
    # unfold patches
    patch = 3
    c_unf = F.unfold(c_feat, kernel_size=patch, stride=patch).transpose(1, 2)  # (N, P, C*k*k)
    s_unf = F.unfold(s_feat, kernel_size=patch, stride=patch).transpose(1, 2)
    c_norm = F.normalize(c_unf, dim=-1)
    s_norm = F.normalize(s_unf, dim=-1)
    sim = (c_norm * s_norm).sum(-1)  # cosine similarity per patch
    return float(1.0 - sim.mean().item())


class RGBuvHistBlock(nn.Module):
    """
    Lightweight RGBuv histogram distance adapted from StyleID.
    """

    def __init__(self, h=64, sigma=0.02):
        super().__init__()
        self.h = h
        self.sigma = sigma
        bin_vals = torch.linspace(-1.0, 1.0, steps=h)
        self.register_buffer("bin_vals", bin_vals)

    def forward(self, x: torch.Tensor):
        # x: (N,3,H,W) in [0,1]
        x = x * 2 - 1  # to [-1,1]
        n, c, h, w = x.shape
        x = x.view(n, c, -1)
        x_exp = x.unsqueeze(-1)  # (N,3,HW,1)
        bin_diff = (x_exp - self.bin_vals) ** 2
        weights = torch.exp(-bin_diff / (2 * self.sigma ** 2))
        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        return hist


@METRICS_REGISTRY.register("histogan")
def histogan_distance(
    style_path: Path,
    stylized_path: Path,
    device: str = "cuda",
    size: Optional[int] = None,
):
    t = build_transform(size)
    block = RGBuvHistBlock().to(device)
    with torch.no_grad():
        s = t(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        y = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)
        h_s = block(s)
        h_y = block(y)
        dist = F.l1_loss(h_s, h_y)
    return float(dist.item())

