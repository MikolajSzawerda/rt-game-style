#!/usr/bin/env python3
"""
Download COCO dataset subsets for style transfer evaluation.

This script downloads and extracts COCO images for use as content images
in style transfer evaluation.
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",  # 5K images, ~1GB
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",  # 118K images, ~18GB
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",  # 40K images, ~6GB
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_coco(
    output_dir: Path,
    split: str = "val2017",
    keep_zip: bool = False,
) -> Path:
    """
    Download COCO dataset split.
    
    Args:
        output_dir: Directory to save the dataset
        split: Dataset split ("val2017", "train2017", "test2017")
        keep_zip: Whether to keep the zip file after extraction
        
    Returns:
        Path to the extracted images directory
    """
    if split not in COCO_URLS:
        raise ValueError(f"Unknown split: {split}. Choose from: {list(COCO_URLS.keys())}")
    
    url = COCO_URLS[split]
    output_dir = Path(output_dir)
    zip_path = output_dir / f"{split}.zip"
    images_dir = output_dir / split
    
    # Check if already downloaded
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"COCO {split} already exists at {images_dir}")
        return images_dir
    
    # Download
    if not zip_path.exists():
        print(f"Downloading COCO {split}...")
        download_file(url, zip_path)
    else:
        print(f"Using existing zip file: {zip_path}")
    
    # Extract
    extract_zip(zip_path, output_dir)
    
    # Optionally remove zip
    if not keep_zip:
        print(f"Removing {zip_path}...")
        zip_path.unlink()
    
    print(f"COCO {split} ready at {images_dir}")
    return images_dir


def main():
    parser = argparse.ArgumentParser(description="Download COCO dataset for style transfer evaluation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "coco",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--split",
        choices=list(COCO_URLS.keys()),
        default="val2017",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip file after extraction",
    )
    
    args = parser.parse_args()
    download_coco(args.output_dir, args.split, args.keep_zip)


if __name__ == "__main__":
    main()

