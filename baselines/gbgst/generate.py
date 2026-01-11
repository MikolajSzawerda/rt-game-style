#!/usr/bin/env python3
"""
GBGST batch stylization.

Generates stylized images using GBGST pretrained models.
Output: {content_stem}_stylized_{style_stem}.png
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add submodule to path
REPO_PATH = Path(__file__).parent / "repo" / "depth-aware-nst"
sys.path.insert(0, str(REPO_PATH))

from transformer_net import TransformerNet  # noqa: E402


@dataclass
class Args:
    content_dir: Path
    weights_dir: Path
    output_dir: Path
    style_dir: Path | None = None
    styles: list[str] = field(default_factory=list)
    content_images: list[str] = field(default_factory=list)
    no_cuda: bool = False
    content_scale: float | None = None


def parse_args() -> Args:
    import argparse

    p = argparse.ArgumentParser(description="GBGST batch stylization")
    p.add_argument("--content-dir", type=Path, required=True)
    p.add_argument("--style-dir", type=Path, default=None, help="(for naming only)")
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--styles", nargs="*", default=[])
    p.add_argument("--content-images", nargs="*", default=[])
    p.add_argument("--no-cuda", action="store_true", help="Use CPU instead of CUDA")
    p.add_argument("--content-scale", type=float, default=None)
    a = p.parse_args()
    return Args(
        content_dir=a.content_dir,
        style_dir=a.style_dir,
        weights_dir=a.weights_dir,
        output_dir=a.output_dir,
        styles=a.styles or [],
        content_images=a.content_images or [],
        no_cuda=a.no_cuda,
        content_scale=a.content_scale,
    )


def load_image(path: Path, scale: float | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if scale:
        img = img.resize(
            (int(img.width / scale), int(img.height / scale)), Image.LANCZOS
        )
    return img


def save_image(path: Path, tensor: torch.Tensor) -> None:
    img = tensor.clone().clamp(0, 255).cpu().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_model(weights_path: Path, device: torch.device) -> TransformerNet:
    model = TransformerNet()
    state = torch.load(weights_path, map_location=device, weights_only=True)

    # Remove deprecated InstanceNorm running stats
    for k in list(state.keys()):
        if re.search(r"in\d+\.running_(mean|var)$", k):
            del state[k]

    model.load_state_dict(state)
    return model.to(device).eval()


def stylize(
    model: TransformerNet, img: Image.Image, device: torch.device
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ]
    )
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x)[0].cpu()


def main():
    args = parse_args()

    # Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if use_cuda else "")
    )

    # Content images
    if args.content_images:
        content_paths = [args.content_dir / name for name in args.content_images]
    else:
        content_paths = sorted(
            p
            for p in args.content_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    if not content_paths:
        sys.exit(f"No content images in {args.content_dir}")

    # Weight files
    if args.styles:
        weight_paths = [args.weights_dir / f"{s}.pth" for s in args.styles]
    else:
        weight_paths = sorted(args.weights_dir.glob("*.pth"))
    if not weight_paths:
        sys.exit(f"No weights in {args.weights_dir}")

    print(f"Content: {len(content_paths)} images, Styles: {len(weight_paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(content_paths) * len(weight_paths)
    with tqdm(total=total, desc="Stylizing") as pbar:
        for wp in weight_paths:
            style_name = wp.stem
            model = load_model(wp, device)

            for cp in content_paths:
                img = load_image(cp, args.content_scale)
                out = stylize(model, img, device)
                out_path = args.output_dir / f"{cp.stem}_stylized_{style_name}.png"
                save_image(out_path, out)
                pbar.update(1)

    print(f"Done! {total} images → {args.output_dir}")


if __name__ == "__main__":
    main()
