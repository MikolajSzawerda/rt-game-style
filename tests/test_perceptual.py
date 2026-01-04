"""Integration tests for perceptual metrics (LPIPS, SSIM, content_loss).

These tests ensure metrics handle image sizing correctly and return valid values.
"""

from PIL import Image


class TestLPIPS:
    """Tests for lpips_content metric."""

    def test_handles_different_sizes_with_explicit_size(self, different_size_images):
        """Regression test: different-sized images should be resized when size is provided."""
        from rt_games.metrics.perceptual import lpips_content

        content, stylized = different_size_images
        # Should not raise RuntimeError about tensor size mismatch
        result = lpips_content(content, stylized, device="cpu", size=64)
        assert isinstance(result, float)
        assert 0 <= result <= 1  # LPIPS is bounded [0, 1]

    def test_default_size_resizes_images(self, different_size_images):
        """Default size=512 should resize both images, preventing size mismatch."""
        from rt_games.metrics.perceptual import lpips_content

        content, stylized = different_size_images
        # Uses default size=512, should not crash
        result = lpips_content(content, stylized, device="cpu")
        assert isinstance(result, float)

    def test_identical_images_low_lpips(self, tmp_path):
        """LPIPS of identical images should be close to 0."""
        from rt_games.metrics.perceptual import lpips_content

        img = Image.new("RGB", (64, 64), color="green")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = lpips_content(p, p, device="cpu", size=64)
        assert result < 0.1  # Should be very low for identical images

    def test_different_images_higher_lpips(self, tmp_path):
        """LPIPS of very different images should be higher."""
        from rt_games.metrics.perceptual import lpips_content

        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="blue")
        p1 = tmp_path / "red.jpg"
        p2 = tmp_path / "blue.jpg"
        img1.save(p1)
        img2.save(p2)

        result = lpips_content(p1, p2, device="cpu", size=64)
        assert result > 0.1  # Should be noticeably different


class TestSSIM:
    """Tests for ssim_score metric."""

    def test_handles_different_sizes(self, different_size_images):
        """Regression test: different-sized images should be resized."""
        from rt_games.metrics.perceptual import ssim_score

        content, stylized = different_size_images
        result = ssim_score(content, stylized, device="cpu", size=64)
        assert isinstance(result, float)
        assert -1 <= result <= 1  # SSIM is bounded [-1, 1]

    def test_identical_images_high_ssim(self, tmp_path):
        """SSIM of identical images should be ~1.0."""
        from rt_games.metrics.perceptual import ssim_score

        img = Image.new("RGB", (64, 64), color="green")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = ssim_score(p, p, device="cpu", size=64)
        assert result > 0.99

    def test_different_images_lower_ssim(self, tmp_path):
        """SSIM of different images should be lower."""
        from rt_games.metrics.perceptual import ssim_score

        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="blue")
        p1 = tmp_path / "red.jpg"
        p2 = tmp_path / "blue.jpg"
        img1.save(p1)
        img2.save(p2)

        result = ssim_score(p1, p2, device="cpu", size=64)
        assert result < 0.9


class TestContentLoss:
    """Tests for content_loss metric."""

    def test_handles_different_sizes(self, different_size_images):
        """Regression test: different-sized images should be resized."""
        from rt_games.metrics.perceptual import content_loss

        content, stylized = different_size_images
        result = content_loss(content, stylized, device="cpu", size=64)
        assert isinstance(result, float)
        assert result >= 0  # MSE loss is non-negative

    def test_identical_images_zero_loss(self, tmp_path):
        """Content loss of identical images should be ~0."""
        from rt_games.metrics.perceptual import content_loss

        img = Image.new("RGB", (64, 64), color="green")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = content_loss(p, p, device="cpu", size=64)
        assert result < 0.01
