from pathlib import Path
from typing import Optional

from rt_games.metrics.perceptual import lpips_content
from rt_games.metrics.style import fid_score
from rt_games.data.io import validate_image_triplets


def artfid(
    content_dir: Path,
    style_dir: Path,
    stylized_dir: Path,
    device: str = "cuda",
    size: Optional[int] = 512,
    use_art_inception: bool = True,
) -> float:
    """
    ArtFID per definition: (1 + LPIPS_content) * (1 + FID_style)
    - LPIPS_content averaged over all matched (content, stylized)
    - FID_style between style set and stylized set
    """
    samples = validate_image_triplets(content_dir, style_dir, stylized_dir)
    lpips_vals = [
        lpips_content(s.content, s.stylized, device=device, size=size) for s in samples
    ]
    lpips_mean = sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0
    fid_val = fid_score(
        style_dir, stylized_dir, device=device, use_art_inception=use_art_inception
    )
    return (1.0 + lpips_mean) * (1.0 + fid_val)
