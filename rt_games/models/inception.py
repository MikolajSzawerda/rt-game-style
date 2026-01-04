from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models.inception import Inception3


def load_art_inception(device: str, checkpoint_url: str) -> nn.Module:
    """
    Load art_inception checkpoint (used by ArtFID and style FID in SOTA papers).
    """
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location=device)
    model = Inception3(aux_logits=False, transform_input=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Not ideal but we proceed; warn caller by printing.
        print(f"[warn] art_inception missing keys: {missing}, unexpected: {unexpected}")
    model.to(device)
    model.eval()
    return model


def load_inception(device: str) -> nn.Module:
    model = Inception3(aux_logits=False, transform_input=False, weights="IMAGENET1K_V1")
    model.to(device)
    model.eval()
    return model

