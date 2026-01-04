import torch


def load_midas(device: str = "cuda", model_type: str = "DPT_Hybrid"):
    """
    Load MiDaS depth estimation model from torch hub.
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "dpt" in model_type.lower():
        tfm = transform.dpt_transform
    else:
        tfm = transform.small_transform
    return midas, tfm

