from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
import torch


VGG_LAYERS = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
}


class VGG16Features(nn.Module):
    """
    VGG16 feature extractor returning selected layers.
    """

    def __init__(self, layers: Dict[str, int] = VGG_LAYERS):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.model = nn.Sequential(*list(vgg.children())[: max(layers.values()) + 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        outputs = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            idx = int(name)
            for k, v in self.layers.items():
                if v == idx:
                    outputs[k] = x
        return outputs


def build_vgg(device: str = "cuda") -> VGG16Features:
    return VGG16Features().to(device).eval()

