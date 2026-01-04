from pathlib import Path
from typing import Optional

from rt_games.metrics.perceptual import lpips_content
from rt_games.metrics.style import fid_score


def artfid(
    content_dir: Path,
    style_dir: Path,
    stylized_dir: Path,
    device: str = "cuda",
    size: Optional[int] = None,
    use_art_inception: bool = True,
) -> float:
    """
    ArtFID per definition: (1 + LPIPS_content) * (1 + FID_style)
    - LPIPS_content averaged over content vs stylized
    - FID_style between style set and stylized set
    """
    lpips_vals = []
    for content_path in sorted(content_dir.glob("*")):
        stylized_path = stylized_dir / f"{content_path.stem}_stylized_{style_dir.stem}{content_path.suffix}"
        if not stylized_path.exists():
            continue
        lp = lpips_content(content_path, stylized_path, device=device, size=size)
        lpips_vals.append(lp)
    lpips_mean = sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0
    fid_val = fid_score(style_dir, stylized_dir, device=device, use_art_inception=use_art_inception)
    return (1.0 + lpips_mean) * (1.0 + fid_val)

