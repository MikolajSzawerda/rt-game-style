from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class SamplePaths:
    content: Path
    stylized: Path
    style: Optional[Path] = None


def _list_images(root: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])


def validate_image_triplets(content_dir: Path, style_dir: Path, stylized_dir: Path) -> List[SamplePaths]:
    """
    Validate and align files using naming {content}_stylized_{style}.ext
    Returns list of SamplePaths with matched triplets.
    """
    content_map = {p.stem: p for p in _list_images(content_dir)}
    style_map = {p.stem: p for p in _list_images(style_dir)}
    samples: List[SamplePaths] = []

    for p in _list_images(stylized_dir):
        stem = p.stem
        if "_stylized_" not in stem:
            raise ValueError(f"Stylized file {p.name} missing '_stylized_' convention")
        content_stem, style_stem = stem.split("_stylized_", maxsplit=1)
        if content_stem not in content_map:
            raise ValueError(f"Content image {content_stem} not found for stylized {p.name}")
        if style_stem not in style_map:
            raise ValueError(f"Style image {style_stem} not found for stylized {p.name}")
        samples.append(
            SamplePaths(
                content=content_map[content_stem],
                stylized=p,
                style=style_map[style_stem],
            )
        )
    if not samples:
        raise ValueError("No stylized images found after validation.")
    return samples


def load_image(path: Path, size: Optional[int] = None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size))
    return img

