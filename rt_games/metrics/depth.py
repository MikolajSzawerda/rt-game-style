from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from rt_games.data.transforms import build_transform
from rt_games.models.midas import load_midas
from rt_games.utils.cache import ModelCache


def _load_depth_from_image(path: Path, size: Optional[int], device: str):
    img = Image.open(path)
    t = build_transform(size)
    return t(img).mean(dim=0, keepdim=True).unsqueeze(0).to(device)


def depth_error(
    original_dir: Path,
    stylized_dir: Path,
    depth_dir: Optional[Path] = None,
    device: str = "cuda",
    size: Optional[int] = None,
):
    originals = sorted(list(original_dir.glob("*")))
    stylized = sorted(list(stylized_dir.glob("*")))
    if len(originals) == 0:
        raise ValueError("No frames for depth error.")

    if depth_dir:
        depth_maps = sorted(list(depth_dir.glob("*")))
    else:
        depth_maps = None

    midas_model, midas_transform = ModelCache.get_midas(load_midas, device)

    total = 0.0
    count = 0
    for idx, orig_path in enumerate(originals):
        sty_path = stylized[idx] if idx < len(stylized) else None
        if sty_path is None:
            break
        if depth_maps and idx < len(depth_maps):
            depth_gt = _load_depth_from_image(depth_maps[idx], size, device)
        else:
            with torch.no_grad():
                inp = midas_transform(Image.open(orig_path)).to(device)
                depth_gt = midas_model(inp).unsqueeze(0)

        with torch.no_grad():
            sty_in = midas_transform(Image.open(sty_path)).to(device)
            depth_sty = midas_model(sty_in).unsqueeze(0)
        # scale-invariant log RMSE
        diff = torch.log(depth_sty + 1e-6) - torch.log(depth_gt + 1e-6)
        si = torch.sqrt((diff ** 2).mean() - diff.mean() ** 2)
        total += float(si.item())
        count += 1
    return total / max(count, 1)

