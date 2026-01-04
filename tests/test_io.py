"""Unit tests for data/io.py."""

import pytest
from pathlib import Path
from PIL import Image

from rt_games.data.io import (
    _list_images,
    validate_image_triplets,
    load_image,
    SamplePaths,
)


class TestListImages:
    """Tests for _list_images function."""

    def test_lists_common_image_formats(self, tmp_path):
        """Should list png, jpg, jpeg, bmp, tiff files."""
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "c.jpeg").touch()
        (tmp_path / "d.bmp").touch()
        (tmp_path / "e.tiff").touch()

        result = _list_images(tmp_path)
        assert len(result) == 5
        assert all(isinstance(p, Path) for p in result)

    def test_ignores_non_image_files(self, tmp_path):
        """Should ignore non-image files."""
        (tmp_path / "image.png").touch()
        (tmp_path / "text.txt").touch()
        (tmp_path / "data.json").touch()

        result = _list_images(tmp_path)
        assert len(result) == 1
        assert result[0].name == "image.png"

    def test_returns_sorted_list(self, tmp_path):
        """Should return sorted list of paths."""
        (tmp_path / "c.png").touch()
        (tmp_path / "a.png").touch()
        (tmp_path / "b.png").touch()

        result = _list_images(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.png", "b.png", "c.png"]

    def test_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        result = _list_images(tmp_path)
        assert result == []

    def test_case_insensitive_extensions(self, tmp_path):
        """Should handle uppercase extensions."""
        (tmp_path / "a.PNG").touch()
        (tmp_path / "b.JPG").touch()

        result = _list_images(tmp_path)
        assert len(result) == 2


class TestValidateImageTriplets:
    """Tests for validate_image_triplets function."""

    def test_valid_triplet(self, sample_triplet):
        """Should successfully validate a correct triplet."""
        samples = validate_image_triplets(
            sample_triplet["content_dir"],
            sample_triplet["style_dir"],
            sample_triplet["stylized_dir"],
        )

        assert len(samples) == 1
        assert isinstance(samples[0], SamplePaths)
        assert samples[0].content.stem == "cat"
        assert samples[0].style.stem == "starry"
        assert "cat_stylized_starry" in samples[0].stylized.stem

    def test_missing_stylized_convention(self, tmp_path):
        """Should raise ValueError if stylized file lacks _stylized_ convention."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (10, 10)).save(content_dir / "cat.jpg")
        Image.new("RGB", (10, 10)).save(style_dir / "starry.jpg")
        Image.new("RGB", (10, 10)).save(
            stylized_dir / "bad_name.jpg"
        )  # Missing _stylized_

        with pytest.raises(ValueError, match="missing '_stylized_' convention"):
            validate_image_triplets(content_dir, style_dir, stylized_dir)

    def test_missing_content_image(self, tmp_path):
        """Should raise ValueError if content image is missing."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        # No content image for "dog"
        Image.new("RGB", (10, 10)).save(style_dir / "starry.jpg")
        Image.new("RGB", (10, 10)).save(stylized_dir / "dog_stylized_starry.jpg")

        with pytest.raises(ValueError, match="Content image .* not found"):
            validate_image_triplets(content_dir, style_dir, stylized_dir)

    def test_missing_style_image(self, tmp_path):
        """Should raise ValueError if style image is missing."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (10, 10)).save(content_dir / "cat.jpg")
        # No style image for "monet"
        Image.new("RGB", (10, 10)).save(stylized_dir / "cat_stylized_monet.jpg")

        with pytest.raises(ValueError, match="Style image .* not found"):
            validate_image_triplets(content_dir, style_dir, stylized_dir)

    def test_empty_stylized_directory(self, tmp_path):
        """Should raise ValueError if no stylized images found."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (10, 10)).save(content_dir / "cat.jpg")
        Image.new("RGB", (10, 10)).save(style_dir / "starry.jpg")
        # stylized_dir is empty

        with pytest.raises(ValueError, match="No stylized images found"):
            validate_image_triplets(content_dir, style_dir, stylized_dir)

    def test_multiple_triplets(self, tmp_path):
        """Should handle multiple valid triplets."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        # Two content images
        Image.new("RGB", (10, 10)).save(content_dir / "cat.jpg")
        Image.new("RGB", (10, 10)).save(content_dir / "dog.jpg")
        # Two style images
        Image.new("RGB", (10, 10)).save(style_dir / "starry.jpg")
        Image.new("RGB", (10, 10)).save(style_dir / "monet.jpg")
        # Four stylized images (all combinations)
        Image.new("RGB", (10, 10)).save(stylized_dir / "cat_stylized_starry.jpg")
        Image.new("RGB", (10, 10)).save(stylized_dir / "cat_stylized_monet.jpg")
        Image.new("RGB", (10, 10)).save(stylized_dir / "dog_stylized_starry.jpg")
        Image.new("RGB", (10, 10)).save(stylized_dir / "dog_stylized_monet.jpg")

        samples = validate_image_triplets(content_dir, style_dir, stylized_dir)
        assert len(samples) == 4


class TestLoadImage:
    """Tests for load_image function."""

    def test_loads_rgb_image(self, create_test_image):
        """Should load image as RGB."""
        path = create_test_image("test.png", size=(50, 50), color="red")
        img = load_image(path)
        assert img.mode == "RGB"

    def test_resizes_when_size_provided(self, create_test_image):
        """Should resize image when size is provided."""
        path = create_test_image("test.png", size=(100, 200))
        img = load_image(path, size=64)
        assert img.size == (64, 64)

    def test_no_resize_when_size_none(self, create_test_image):
        """Should keep original size when size is None."""
        path = create_test_image("test.png", size=(100, 200))
        img = load_image(path, size=None)
        assert img.size == (100, 200)
