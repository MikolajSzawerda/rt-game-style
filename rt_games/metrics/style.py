from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch import nn


from rt_games.config import DEFAULT_CONFIG
from rt_games.data.transforms import build_transform, build_transform_fid
from rt_games.models.art_inception import ArtInception3
from rt_games.models.inception import InceptionV3, load_art_inception, load_inception
from rt_games.models.vgg import build_vgg
from rt_games.utils.cache import ModelCache
from rt_games.utils.registry import METRICS_REGISTRY

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    n, c, h, w = feat.size()
    f = feat.view(n, c, -1)
    g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    return g


@METRICS_REGISTRY.register("gram_loss")
def gram_loss(
    style_path: Path,
    stylized_path: Path,
    device: str = "cuda",
    size: Optional[int] = None,
    layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
):
    vgg = ModelCache.get_vgg(build_vgg, device)
    t = build_transform(size)
    with torch.no_grad():
        style = t(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        stylized = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)
        s_feats = vgg(style)
        y_feats = vgg(stylized)
        total = 0.0
        for layer in layers:
            total += F.mse_loss(
                gram_matrix(y_feats[layer]), gram_matrix(s_feats[layer])
            )
        loss = total / len(layers)
    return float(loss.item())


def _load_inception(device: str, use_art: bool, cfg=DEFAULT_CONFIG):
    if use_art:
        return ModelCache.get_inception(
            lambda d: load_art_inception(d, cfg.art_inception_url), device, art=True
        )
    return ModelCache.get_inception(lambda d: load_inception(d), device, art=False)


def _compute_frechet_distance(
    real_feats: torch.Tensor, fake_feats: torch.Tensor, eps: float = 1e-6
) -> float:
    """
    Compute Frechet distance between two sets of features.

    Uses scipy.linalg.sqrtm for matrix square root computation.

    Args:
        real_feats: Features from real images, shape (N, D)
        fake_feats: Features from fake images, shape (M, D)
        eps: Small value for numerical stability

    Returns:
        Frechet distance value
    """
    n_real, d = real_feats.shape
    n_fake = fake_feats.shape[0]

    mu1 = real_feats.mean(0).numpy()
    mu2 = fake_feats.mean(0).numpy()

    diff = mu1 - mu2

    # Handle edge case: single sample cannot compute proper covariance
    # In this case, just return squared mean difference
    if n_real < 2 or n_fake < 2:
        return float(diff.dot(diff))

    sigma1 = np.cov(real_feats.numpy(), rowvar=False)
    sigma2 = np.cov(fake_feats.numpy(), rowvar=False)

    # Ensure covariance matrices are 2D
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert sigma1.shape == sigma2.shape, (
        f"Covariance matrices have different dimensions: {sigma1.shape} vs {sigma2.shape}"
    )

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        # Add epsilon to diagonal for numerical stability
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@METRICS_REGISTRY.register("fid")
def fid_score(
    real_dir: Path,
    fake_dir: Path,
    device: str = "cuda",
    use_art_inception: bool = True,
    sample_limit: Optional[int] = None,
    size: Optional[int] = 512,
):
    """
    Compute Frechet Inception Distance (FID) between two image directories.

    Args:
        real_dir: Directory containing real/style images
        fake_dir: Directory containing fake/stylized images
        device: Device to run on ('cpu' or 'cuda')
        use_art_inception: If True, use ArtInception3 (for ArtFID). If False,
                          use torch-fidelity with standard Inception.
        sample_limit: Optional limit on number of images to process
        size: Image size for preprocessing. Default 512 matches reference ArtFID.

    Returns:
        FID score (lower is better)
    """
    if use_art_inception:
        model = _load_inception(device, use_art_inception)
        real_paths = sorted(real_dir.glob("*"))
        fake_paths = sorted(fake_dir.glob("*"))
        if sample_limit:
            real_paths = real_paths[:sample_limit]
            fake_paths = fake_paths[:sample_limit]
        real_feats = _get_features(real_paths, model, device, size)
        fake_feats = _get_features(fake_paths, model, device, size)
        return _compute_frechet_distance(real_feats, fake_feats)
    else:
        if calculate_metrics is None:
            raise ImportError("torch-fidelity is required for FID.")
        kwargs = {
            "input1": str(real_dir),
            "input2": str(fake_dir),
            "device": device,
            "fid": True,
        }
        if sample_limit is not None:
            kwargs["sample_limit"] = sample_limit
        result = calculate_metrics(**kwargs)
        return float(result["frechet_inception_distance"])


def _get_activations_single(
    path: Path, model: nn.Module, device: str, size: Optional[int]
) -> torch.Tensor:
    """Get activations for a single image, flattened to (1, features)."""
    t = build_transform_fid(size)  # Use aspect-preserving resize for FID
    with torch.no_grad():
        x = t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        # ArtInception3 supports return_features=True which returns flattened 2048-dim
        if isinstance(model, ArtInception3):
            f = model(x, return_features=True)
            # Already flattened to (B, 2048)
            return f.cpu().float()
        f = model(x)
        # Handle InceptionV3 wrapper which returns a list of feature maps
        if isinstance(f, list):
            f = f[0]
        # Global average pool and flatten
        f = F.adaptive_avg_pool2d(f, (1, 1)).view(1, -1)
        return f.cpu().float()


def _get_activations_with_patches(
    path: Path, model: nn.Module, device: str, size: Optional[int]
) -> torch.Tensor:
    """
    Get patch-level activations for SIFID computation.
    Returns tensor of shape (num_patches, features) for covariance computation.
    """
    t = build_transform_fid(size)  # Use aspect-preserving resize for SIFID
    with torch.no_grad():
        x = t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        # ArtInception3 supports return_spatial=True for pre-pooling features
        if isinstance(model, ArtInception3):
            f = model(x, return_spatial=True)  # (1, 2048, H, W)
        else:
            f = model(x)
            # Handle InceptionV3 wrapper which returns a list of feature maps
            if isinstance(f, list):
                f = f[0]  # (1, C, H, W)
        # Reshape to (H*W, C) to get multiple samples for covariance
        n, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(h * w, c)
        return f.cpu().float()


def _get_features(
    paths: List[Path], model: nn.Module, device: str, size: Optional[int]
) -> torch.Tensor:
    feats = []
    for p in tqdm(paths, desc="Extracting features", leave=False):
        feats.append(_get_activations_single(p, model, device, size))
    return torch.cat(feats, dim=0)


@METRICS_REGISTRY.register("sifid")
def sifid_score(
    style_dir: Path,
    stylized_dir: Path,
    device: str = "cuda",
    size: Optional[int] = None,
    dims: int = 64,
    sample_limit: Optional[int] = None,
    eps: float = 1e-6,
):
    """
    Compute Single-Image FID (SIFID) between style and stylized image pairs.

    Uses early-layer features (dims=64 by default) at original resolution,
    following the original SIFID paper approach.

    Args:
        style_dir: Directory containing style images
        stylized_dir: Directory containing stylized images
        device: Device to run on ('cpu' or 'cuda')
        size: Optional size to resize images (None = original resolution)
        dims: Feature dimensionality (64, 192, 768, or 2048). Default 64 matches
              the reference SIFID implementation.
        sample_limit: Optional limit on number of image pairs to process
        eps: Small value for numerical stability

    Returns:
        Mean SIFID value across all image pairs
    """
    # SIFID uses standard InceptionV3 with early features, NOT ArtInception
    # Key: resize_input=False to preserve original resolution (reference behavior)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], resize_input=False, normalize_input=True)
    model.to(device)
    model.eval()

    style_paths = sorted(list(style_dir.glob("*")))
    stylized_paths = sorted(list(stylized_dir.glob("*")))
    if sample_limit:
        style_paths = style_paths[:sample_limit]
        stylized_paths = stylized_paths[:sample_limit]
    n = min(len(style_paths), len(stylized_paths))
    if n == 0:
        raise ValueError("No images for SIFID.")
    sifid_vals = []
    for s_path, y_path in tqdm(
        zip(style_paths[:n], stylized_paths[:n]),
        total=n,
        desc="Computing SIFID",
        leave=False,
    ):
        # Use patch-level features for proper covariance computation
        s_feat = _get_activations_with_patches(s_path, model, device, size)
        y_feat = _get_activations_with_patches(y_path, model, device, size)

        # Compute per-image FID using patch statistics
        fid_val = _compute_frechet_distance(s_feat, y_feat, eps=eps)
        sifid_vals.append(fid_val)
    return float(np.mean(sifid_vals))


@METRICS_REGISTRY.register("cfsd")
def cfsd(
    content_path: Path,
    stylized_path: Path,
    device: str = "cuda",
    size: Optional[int] = None,
):
    """
    Simplified CFSD: compare patch cosine similarity on VGG relu3_3.
    """
    vgg = ModelCache.get_vgg(build_vgg, device)
    t = build_transform(size)

    with torch.no_grad():
        c = t(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
        s = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)

        if s.shape != c.shape:
            s = F.interpolate(s, size=c.shape[2:], mode="bilinear", align_corners=False)

        c_feat = vgg(c)["relu3_3"]
        s_feat = vgg(s)["relu3_3"]
        
    # unfold patches
    patch = 3
    c_unf = F.unfold(c_feat, kernel_size=patch, stride=patch).transpose(
        1, 2
    )  # (N, P, C*k*k)
    s_unf = F.unfold(s_feat, kernel_size=patch, stride=patch).transpose(1, 2)
    c_norm = F.normalize(c_unf, dim=-1)
    s_norm = F.normalize(s_unf, dim=-1)
    sim = (c_norm * s_norm).sum(-1)  # cosine similarity per patch
    return float(1.0 - sim.mean().item())


class RGBuvHistBlock(nn.Module):
    """
    RGB-uv histogram from HistoGAN paper (Afifi et al., CVPR 2021).

    Uses log-chromaticity color representation and inverse-quadratic kernel,
    matching the reference implementation.

    Reference: https://github.com/mahmoudnafifi/HistoGAN
    """

    def __init__(
        self,
        h: int = 64,
        insz: int = 150,
        method: str = "inverse-quadratic",
        sigma: float = 0.02,
        intensity_scale: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize RGBuvHistBlock.

        Args:
            h: Histogram dimension size (default 64)
            insz: Maximum input size; larger images are resized (default 150)
            method: Kernel method ('inverse-quadratic', 'RBF', or 'thresholding')
            sigma: Kernel bandwidth for RBF/inverse-quadratic methods
            intensity_scale: Whether to use intensity scaling (I_y in paper)
            device: Device to run on
        """
        super().__init__()
        self.h = h
        self.insz = insz
        self.method = method
        self.sigma = sigma
        self.intensity_scale = intensity_scale
        self.device = device
        self.eps = 1e-6

        if method == "thresholding":
            self.bin_eps = 6.0 / h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RGB-uv histogram.

        Args:
            x: Input tensor of shape (N, 3, H, W) in [0, 1]

        Returns:
            Histogram tensor of shape (N, 3, h, h)
        """
        x = torch.clamp(x, 0, 1)

        # Resize if needed
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            x = F.interpolate(
                x, size=(self.insz, self.insz), mode="bilinear", align_corners=False
            )

        L = x.shape[0]
        hists = torch.zeros((L, 3, self.h, self.h), device=x.device)
        bin_edges = torch.linspace(-3, 3, self.h, device=x.device)

        for l in range(L):
            # I: (H*W, 3) pixels
            I = x[l].permute(1, 2, 0).reshape(-1, 3)
            II = I**2

            # Intensity scaling factor
            if self.intensity_scale:
                Iy = torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + self.eps).unsqueeze(1)
            else:
                Iy = torch.ones(I.shape[0], 1, device=x.device)

            # Log-chromaticity for each channel pair
            # Channel 0: log(R/G) vs log(R/B)
            # Channel 1: log(G/R) vs log(G/B)
            # Channel 2: log(B/R) vs log(B/G)
            for c in range(3):
                c1, c2 = (c + 1) % 3, (c + 2) % 3
                Iu = (
                    torch.log(I[:, c] + self.eps) - torch.log(I[:, c1] + self.eps)
                ).unsqueeze(1)
                Iv = (
                    torch.log(I[:, c] + self.eps) - torch.log(I[:, c2] + self.eps)
                ).unsqueeze(1)

                diff_u = torch.abs(Iu - bin_edges)
                diff_v = torch.abs(Iv - bin_edges)

                if self.method == "thresholding":
                    diff_u = (diff_u <= self.bin_eps / 2).float()
                    diff_v = (diff_v <= self.bin_eps / 2).float()
                elif self.method == "RBF":
                    diff_u = torch.exp(-((diff_u / self.sigma) ** 2))
                    diff_v = torch.exp(-((diff_v / self.sigma) ** 2))
                elif self.method == "inverse-quadratic":
                    diff_u = 1.0 / (1.0 + (diff_u / self.sigma) ** 2)
                    diff_v = 1.0 / (1.0 + (diff_v / self.sigma) ** 2)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                # Weighted outer product
                hists[l, c] = torch.mm((Iy * diff_u).t(), diff_v)

        # Normalize
        hists = hists / (hists.sum(dim=(1, 2, 3), keepdim=True) + self.eps)
        return hists


@METRICS_REGISTRY.register("histogan")
def histogan_distance(
    style_path: Path,
    stylized_path: Path,
    device: str = "cuda",
    size: Optional[int] = None,
):
    """
    Compute HistoGAN distance using Hellinger distance on RGB-uv histograms.

    Uses the log-chromaticity color representation from the HistoGAN paper
    (Afifi et al., CVPR 2021).

    Args:
        style_path: Path to style image
        stylized_path: Path to stylized image
        device: Device to run on
        size: Optional size to resize images

    Returns:
        Hellinger distance between RGB-uv histograms (lower is better)
    """
    t = build_transform(size)
    block = RGBuvHistBlock(device=device).to(device)

    with torch.no_grad():
        s = t(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
        y = t(Image.open(stylized_path).convert("RGB")).unsqueeze(0).to(device)
        h_s = block(s)
        h_y = block(y)

        # Hellinger distance (reference implementation)
        dist = (1.0 / np.sqrt(2.0)) * torch.sqrt(
            torch.sum((torch.sqrt(h_s) - torch.sqrt(h_y)) ** 2)
        )

    return float(dist.item())
