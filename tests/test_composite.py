"""Tests for composite metrics (artfid)."""

import inspect

from PIL import Image


class TestArtFID:
    """Tests for artfid composite metric."""

    def test_artfid_handles_different_sized_images(self, tmp_path):
        """Regression test: artfid should handle different-sized images.

        This was a bug where size=None was passed to lpips_content,
        causing tensor size mismatches when content and stylized
        images had different dimensions.
        """
        from rt_games.metrics.composite import artfid

        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        content_dir.mkdir()
        style_dir.mkdir()
        stylized_dir.mkdir()

        # Create images with DIFFERENT sizes (the bug case)
        Image.new("RGB", (100, 200), "red").save(content_dir / "img1.jpg")
        Image.new("RGB", (80, 80), "blue").save(style_dir / "starry.jpg")
        # Stylized has different size than content
        Image.new("RGB", (150, 180), "purple").save(
            stylized_dir / "img1_stylized_starry.jpg"
        )

        # Should NOT raise RuntimeError about tensor size mismatch
        result = artfid(content_dir, style_dir, stylized_dir, device="cpu")
        assert isinstance(result, float)
        assert result > 0

    def test_artfid_default_size_is_512(self):
        """ArtFID should default to size=512 to match reference implementation."""
        from rt_games.metrics.composite import artfid

        sig = inspect.signature(artfid)
        assert sig.parameters["size"].default == 512

    def test_artfid_identical_images_returns_expected_range(self, tmp_path):
        """ArtFID of identical image sets should return reasonable value."""
        from rt_games.metrics.composite import artfid

        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        content_dir.mkdir()
        style_dir.mkdir()
        stylized_dir.mkdir()

        # Create identical images
        img = Image.new("RGB", (64, 64), "green")
        img.save(content_dir / "img1.jpg")
        img.save(style_dir / "style.jpg")
        img.save(stylized_dir / "img1_stylized_style.jpg")

        result = artfid(content_dir, style_dir, stylized_dir, device="cpu")
        # ArtFID = (1 + LPIPS) * (1 + FID), so minimum is ~1
        assert isinstance(result, float)
        assert result >= 1.0

    def test_artfid_formula_is_correct(self, tmp_path):
        """ArtFID should compute (1 + LPIPS_mean) * (1 + FID_style)."""
        from rt_games.metrics.composite import artfid
        from rt_games.metrics.perceptual import lpips_content
        from rt_games.metrics.style import fid_score

        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        content_dir.mkdir()
        style_dir.mkdir()
        stylized_dir.mkdir()

        # Create test images
        Image.new("RGB", (64, 64), "red").save(content_dir / "img1.jpg")
        Image.new("RGB", (64, 64), "blue").save(style_dir / "style.jpg")
        Image.new("RGB", (64, 64), "purple").save(
            stylized_dir / "img1_stylized_style.jpg"
        )

        # Compute components manually with size=512 (default)
        lpips_val = lpips_content(
            content_dir / "img1.jpg",
            stylized_dir / "img1_stylized_style.jpg",
            device="cpu",
            size=512,
        )
        fid_val = fid_score(
            style_dir, stylized_dir, device="cpu", use_art_inception=True
        )
        expected = (1.0 + lpips_val) * (1.0 + fid_val)

        # Compute via artfid function
        result = artfid(content_dir, style_dir, stylized_dir, device="cpu")

        assert abs(result - expected) < 0.01
