from pathlib import Path
from typing import Optional

import torch
from PIL import Image

import lpips

from rt_games.data.transforms import build_transform
from rt_games.utils.cache import ModelCache
from rt_games.utils.flow import compute_flow, load_raft
from rt_games.utils.warp import warp


def _load_frame(path: Path, size: Optional[int], device: str):
    t = build_transform(size)
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def warping_error(
    original_dir: Path,
    stylized_dir: Path,
    flow_dir: Optional[Path] = None,
    device: str = "cuda",
    size: Optional[int] = None,
    use_raft: bool = True,
):
    originals = sorted(list(original_dir.glob("*")))
    stylized = sorted(list(stylized_dir.glob("*")))
    if len(originals) < 2 or len(stylized) < 2:
        raise ValueError("Need at least two frames for temporal metrics.")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    total = 0.0
    count = 0
    raft_model = None

    for i in range(len(originals) - 1):
        c1 = _load_frame(originals[i], size, device)
        c2 = _load_frame(originals[i + 1], size, device)
        y1 = _load_frame(stylized[i], size, device)
        y2 = _load_frame(stylized[i + 1], size, device)

        if flow_dir:
            flow_path = flow_dir / f"{originals[i].stem}_flow.pt"
            flow = torch.load(flow_path, map_location=device)
        else:
            if use_raft:
                raft_model = ModelCache.get_flow(load_raft, device)
                flow = compute_flow(raft_model, c1, c2)
            else:
                raise ValueError("Flow data not provided and RAFT disabled.")

        y1_warped = warp(y1, flow)
        diff = (y1_warped - y2).abs().mean()
        total += float(diff.item())
        count += 1
    return total / max(count, 1)


def temporal_lpips(
    original_dir: Path,
    stylized_dir: Path,
    flow_dir: Optional[Path] = None,
    device: str = "cuda",
    size: Optional[int] = None,
    use_raft: bool = True,
):
    originals = sorted(list(original_dir.glob("*")))
    stylized = sorted(list(stylized_dir.glob("*")))
    if len(originals) < 2 or len(stylized) < 2:
        raise ValueError("Need at least two frames for temporal metrics.")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    total = 0.0
    count = 0
    raft_model = None

    for i in range(len(originals) - 1):
        c1 = _load_frame(originals[i], size, device)
        c2 = _load_frame(originals[i + 1], size, device)
        y1 = _load_frame(stylized[i], size, device)
        y2 = _load_frame(stylized[i + 1], size, device)

        if flow_dir:
            flow_path = flow_dir / f"{originals[i].stem}_flow.pt"
            flow = torch.load(flow_path, map_location=device)
        else:
            if use_raft:
                raft_model = ModelCache.get_flow(load_raft, device)
                flow = compute_flow(raft_model, c1, c2)
            else:
                raise ValueError("Flow data not provided and RAFT disabled.")

        y1_warped = warp(y1, flow)
        val = lpips_fn(y1_warped, y2)
        total += float(val.mean().item())
        count += 1
    return total / max(count, 1)

