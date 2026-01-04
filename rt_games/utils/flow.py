from pathlib import Path
from typing import Optional

import torch


def load_raft(device: str = "cuda", model_name: str = "raft-small"):
    """
    Load RAFT model from torch hub. Keeps download optional to avoid breaking
    in environments without internet; caller should handle exceptions.
    """
    raft = torch.hub.load("princeton-vl/RAFT", model_name, pretrained=True)
    raft = raft.to(device)
    raft.eval()
    return raft


def compute_flow(
    raft_model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    pad_to_multiple: int = 8,
):
    """
    Compute optical flow between two images (N,3,H,W) in [0,1].
    """
    # RAFT expects list of images
    with torch.no_grad():
        flow_low, flow_up = raft_model(img1, img2, iters=20, test_mode=True, pad=pad_to_multiple)
    return flow_up

