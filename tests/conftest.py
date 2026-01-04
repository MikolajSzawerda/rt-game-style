"""Shared pytest fixtures for rt-games tests."""

import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Create a temporary directory with test images."""
    return tmp_path


@pytest.fixture
def create_test_image(tmp_path):
    """Factory fixture to create test images."""

    def _create(name: str, size: tuple = (64, 64), color: str = "red") -> Path:
        img = Image.new("RGB", size, color=color)
        path = tmp_path / name
        img.save(path)
        return path

    return _create


@pytest.fixture
def sample_triplet(tmp_path):
    """Create a valid content/style/stylized triplet for testing."""
    content_dir = tmp_path / "content"
    style_dir = tmp_path / "style"
    stylized_dir = tmp_path / "stylized"

    for d in [content_dir, style_dir, stylized_dir]:
        d.mkdir()

    # Create test images
    Image.new("RGB", (100, 100), "red").save(content_dir / "cat.jpg")
    Image.new("RGB", (80, 80), "blue").save(style_dir / "starry.jpg")
    Image.new("RGB", (120, 90), "purple").save(stylized_dir / "cat_stylized_starry.jpg")

    return {
        "content_dir": content_dir,
        "style_dir": style_dir,
        "stylized_dir": stylized_dir,
    }


@pytest.fixture
def different_size_images(tmp_path):
    """Create two images with different sizes (for regression testing size handling)."""
    img1 = Image.new("RGB", (100, 200), color="red")
    img2 = Image.new("RGB", (300, 150), color="blue")

    p1 = tmp_path / "content.jpg"
    p2 = tmp_path / "stylized.jpg"
    img1.save(p1)
    img2.save(p2)
    return p1, p2
