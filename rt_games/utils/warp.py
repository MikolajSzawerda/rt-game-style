import torch
import torch.nn.functional as F


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp image (N,3,H,W) using flow (N,2,H,W) in pixel units.
    Adapted from common optical flow warping utilities.
    """
    n, c, h, w = img.size()
    # mesh grid
    yy, xx = torch.meshgrid(
        torch.arange(0, h, device=img.device),
        torch.arange(0, w, device=img.device),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0).expand(n, -1, -1) + flow[:, 0, :, :]
    grid_y = yy.unsqueeze(0).expand(n, -1, -1) + flow[:, 1, :, :]

    grid_x = 2.0 * grid_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=3)
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped
