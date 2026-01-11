#!/usr/bin/env python3
"""
AdaAttN batch stylization.

Generates stylized images using AdaAttN (arbitrary style transfer).
Output: {content_stem}_stylized_{style_stem}.png
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add submodule to path
REPO_PATH = Path(__file__).parent / "repo"
sys.path.insert(0, str(REPO_PATH))

from models import create_model  # noqa: E402
from util.util import tensor2im  # noqa: E402


@dataclass
class Args:
    content_dir: Path
    style_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    styles: list[str] = field(default_factory=list)
    content_images: list[str] = field(default_factory=list)
    no_cuda: bool = False
    image_size: int = 512


def parse_args() -> Args:
    import argparse

    p = argparse.ArgumentParser(description="AdaAttN batch stylization")
    p.add_argument("--content-dir", type=Path, required=True)
    p.add_argument("--style-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Dir with vgg_normalised.pth and AdaAttN/",
    )
    p.add_argument("--styles", nargs="*", default=[])
    p.add_argument("--content-images", nargs="*", default=[])
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--image-size", type=int, default=512)
    a = p.parse_args()
    return Args(**{k.replace("-", "_"): v for k, v in vars(a).items()})


def make_opt(args: Args) -> SimpleNamespace:
    """Create minimal opt object required by AdaAttN model."""
    gpu_ids = [] if args.no_cuda or not torch.cuda.is_available() else [0]

    return SimpleNamespace(
        # Device
        gpu_ids=gpu_ids,
        isTrain=False,
        # Model architecture
        model="adaattn",
        skip_connection_3=True,
        shallow_layer=True,
        # Paths
        image_encoder_path=str(args.checkpoint_dir / "vgg_normalised.pth"),
        checkpoints_dir=str(args.checkpoint_dir),
        name="AdaAttN",
        # Network init (not used at inference, but required)
        init_type="normal",
        init_gain=0.02,
        # Loading
        epoch="latest",
        load_iter=0,
        verbose=False,
        # Preprocessing (for cudnn.benchmark setting)
        preprocess="resize_and_crop",
    )


def load_image(path: Path, size: int) -> torch.Tensor:
    """Load image and transform to tensor [1, 3, H, W] in [0, 1]."""
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor to image file."""
    img_np = tensor2im(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np).save(path)


def main():
    args = parse_args()

    # Device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name()})" if use_cuda else "")
    )

    # Create model
    opt = make_opt(args)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # Gather content images
    if args.content_images:
        content_paths = [args.content_dir / name for name in args.content_images]
    else:
        content_paths = sorted(
            p
            for p in args.content_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    # Gather style images
    if args.styles:
        style_paths = []
        for s in args.styles:
            # Try with and without extension
            for ext in ["", ".jpg", ".jpeg", ".png"]:
                candidate = args.style_dir / f"{s}{ext}"
                if candidate.exists():
                    style_paths.append(candidate)
                    break
    else:
        style_paths = sorted(
            p
            for p in args.style_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not content_paths:
        sys.exit(f"No content images in {args.content_dir}")
    if not style_paths:
        sys.exit(f"No style images in {args.style_dir}")

    print(f"Content: {len(content_paths)} images, Styles: {len(style_paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(content_paths) * len(style_paths)
    with tqdm(total=total, desc="Stylizing") as pbar:
        for style_path in style_paths:
            style_tensor = load_image(style_path, args.image_size).to(device)
            style_name = style_path.stem

            for content_path in content_paths:
                content_tensor = load_image(content_path, args.image_size).to(device)

                # Run inference
                model.set_input({"c": content_tensor, "s": style_tensor, "name": ""})
                model.test()

                # Save result
                out_path = args.output_dir / f"{content_path.stem}_stylized_{style_name}.png"
                save_image(model.cs, out_path)
                pbar.update(1)

    print(f"Done! {total} images → {args.output_dir}")


if __name__ == "__main__":
    main()

