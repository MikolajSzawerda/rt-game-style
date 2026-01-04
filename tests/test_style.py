"""Tests for style metrics (gram_loss, FID, SIFID, CFSD, histogan)."""

import pytest
import torch
import numpy as np
from PIL import Image

from rt_games.metrics.style import (
    gram_matrix,
    gram_loss,
    _compute_frechet_distance,
    _get_activations_with_patches,
    sifid_score,
    cfsd,
    histogan_distance,
    RGBuvHistBlock,
)
from rt_games.utils.registry import METRICS_REGISTRY


class TestGramMatrix:
    """Tests for gram_matrix computation."""

    def test_gram_matrix_shape(self):
        """Gram matrix should have shape (N, C, C)."""
        feat = torch.randn(2, 64, 16, 16)  # (N, C, H, W)
        gram = gram_matrix(feat)
        assert gram.shape == (2, 64, 64)

    def test_gram_matrix_symmetric(self):
        """Gram matrix should be symmetric."""
        feat = torch.randn(1, 32, 8, 8)
        gram = gram_matrix(feat)
        assert torch.allclose(gram[0], gram[0].T, atol=1e-6)

    def test_gram_matrix_normalized(self):
        """Gram matrix should be normalized by (C * H * W)."""
        feat = torch.ones(1, 4, 2, 2)  # C=4, H=2, W=2
        gram = gram_matrix(feat)
        # Each element should be sum(1*1) / (4*2*2) = 4 / 16 = 0.25
        expected = torch.ones(1, 4, 4) * 0.25
        assert torch.allclose(gram, expected, atol=1e-6)


class TestFrechetDistance:
    """Tests for _compute_frechet_distance."""

    def test_identical_distributions_zero_distance(self):
        """FID of identical distributions should be ~0."""
        feats = torch.randn(100, 64)
        dist = _compute_frechet_distance(feats, feats)
        assert dist < 0.1

    def test_different_distributions_positive_distance(self):
        """FID of different distributions should be positive."""
        feats1 = torch.randn(100, 64)
        feats2 = torch.randn(100, 64) + 5  # Shifted mean
        dist = _compute_frechet_distance(feats1, feats2)
        assert dist > 0

    def test_handles_single_sample(self):
        """Should handle single-sample inputs without crashing."""
        feats1 = torch.randn(1, 64)
        feats2 = torch.randn(1, 64)
        # Should not raise an error
        dist = _compute_frechet_distance(feats1, feats2)
        assert isinstance(dist, float)

    def test_handles_few_samples(self):
        """Should handle small number of samples."""
        feats1 = torch.randn(5, 64)
        feats2 = torch.randn(5, 64)
        dist = _compute_frechet_distance(feats1, feats2)
        assert isinstance(dist, float)
        assert np.isfinite(dist)


class TestGramLoss:
    """Tests for gram_loss metric."""

    def test_identical_images_low_gram_loss(self, tmp_path):
        """Gram loss of identical images should be close to 0."""
        img = Image.new("RGB", (64, 64), color="blue")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = gram_loss(p, p, device="cpu", size=64)
        assert result < 0.01

    def test_different_images_positive_gram_loss(self, tmp_path):
        """Gram loss of different images should be positive."""
        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="blue")
        p1 = tmp_path / "style.jpg"
        p2 = tmp_path / "stylized.jpg"
        img1.save(p1)
        img2.save(p2)

        result = gram_loss(p1, p2, device="cpu", size=64)
        assert result > 0

    def test_registered_in_registry(self):
        """gram_loss should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("gram_loss")


class TestSIFID:
    """Tests for SIFID (Single-Image FID) computation."""

    def test_sifid_identical_images(self, tmp_path):
        """SIFID of identical image pairs should be low."""
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        style_dir.mkdir()
        stylized_dir.mkdir()

        # Create identical images
        img = Image.new("RGB", (64, 64), color="green")
        img.save(style_dir / "img1.jpg")
        img.save(stylized_dir / "img1.jpg")

        result = sifid_score(
            style_dir, stylized_dir, device="cpu", size=64, use_art_inception=False
        )
        # Identical images should have low SIFID
        assert result < 1.0

    def test_sifid_requires_images(self, tmp_path):
        """SIFID should raise ValueError if no images found."""
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        style_dir.mkdir()
        stylized_dir.mkdir()

        with pytest.raises(ValueError, match="No images for SIFID"):
            sifid_score(style_dir, stylized_dir, device="cpu")

    def test_sifid_registered_in_registry(self):
        """sifid_score should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("sifid")


class TestGetActivationsWithPatches:
    """Tests for _get_activations_with_patches function."""

    def test_returns_multiple_patches(self, tmp_path):
        """Should return tensor with multiple rows (patches)."""
        # Skip if we can't load inception easily in CPU mode
        pytest.importorskip("torchvision")

        img = Image.new("RGB", (64, 64), color="red")
        p = tmp_path / "test.jpg"
        img.save(p)

        from rt_games.models.inception import load_inception

        model = load_inception("cpu")
        feats = _get_activations_with_patches(p, model, "cpu", size=299)

        # Should have multiple patches (H*W rows)
        assert feats.dim() == 2
        assert feats.shape[0] > 1  # Multiple patches


class TestCFSD:
    """Tests for CFSD metric."""

    def test_cfsd_identical_images(self, tmp_path):
        """CFSD of identical images should be close to 0."""
        img = Image.new("RGB", (64, 64), color="blue")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = cfsd(p, p, device="cpu", size=64)
        assert result < 0.1

    def test_cfsd_returns_float(self, tmp_path):
        """CFSD should return a float value."""
        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="blue")
        p1 = tmp_path / "content.jpg"
        p2 = tmp_path / "stylized.jpg"
        img1.save(p1)
        img2.save(p2)

        result = cfsd(p1, p2, device="cpu", size=64)
        assert isinstance(result, float)
        assert 0 <= result <= 1  # CFSD is 1 - cosine_similarity

    def test_cfsd_registered_in_registry(self):
        """cfsd should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("cfsd")


class TestRGBuvHistBlock:
    """Tests for RGBuvHistBlock module."""

    def test_histogram_shape(self):
        """Histogram output should have shape (N, 3, h)."""
        block = RGBuvHistBlock(h=64)
        x = torch.rand(2, 3, 32, 32)  # (N, 3, H, W) in [0, 1]
        hist = block(x)
        assert hist.shape == (2, 3, 64)

    def test_histogram_sums_to_one(self):
        """Each channel histogram should sum to ~1."""
        block = RGBuvHistBlock(h=64)
        x = torch.rand(1, 3, 32, 32)
        hist = block(x)
        # Each histogram should sum to approximately 1
        sums = hist.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestHistoganDistance:
    """Tests for histogan_distance metric."""

    def test_identical_images_low_distance(self, tmp_path):
        """Histogram distance of identical images should be 0."""
        img = Image.new("RGB", (64, 64), color="blue")
        p = tmp_path / "same.jpg"
        img.save(p)

        result = histogan_distance(p, p, device="cpu", size=64)
        assert result < 0.01

    def test_different_images_positive_distance(self, tmp_path):
        """Histogram distance of different color images should be positive."""
        img1 = Image.new("RGB", (64, 64), color="red")
        img2 = Image.new("RGB", (64, 64), color="blue")
        p1 = tmp_path / "style.jpg"
        p2 = tmp_path / "stylized.jpg"
        img1.save(p1)
        img2.save(p2)

        result = histogan_distance(p1, p2, device="cpu", size=64)
        assert result > 0

    def test_histogan_registered_in_registry(self):
        """histogan_distance should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("histogan")


class TestFIDNotDuplicated:
    """Test that FID helper is not exposed in registry."""

    def test_fid_registered_once(self):
        """fid_score should be the only 'fid' in registry."""
        assert METRICS_REGISTRY.has("fid")
        fid_fn = METRICS_REGISTRY.get("fid")
        # Should be fid_score function, not _compute_frechet_distance
        assert fid_fn.__name__ == "fid_score"
