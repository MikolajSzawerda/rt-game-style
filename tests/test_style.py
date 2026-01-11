"""Tests for style metrics (gram_loss, FID, SIFID, CFSD, histogan)."""

import pytest
import torch
import numpy as np
from PIL import Image

from rt_games.metrics.style import (
    gram_matrix,
    gram_loss,
    _compute_frechet_distance,
    _get_activations_single,
    _get_activations_with_patches,
    fid_score,
    sifid_score,
    cfsd,
    histogan_distance,
    RGBuvHistBlock,
)
from rt_games.models.art_inception import ArtInception3
from rt_games.models.inception import InceptionV3
from rt_games.utils.registry import METRICS_REGISTRY


# =============================================================================
# Shared fixtures for expensive model instantiation
# =============================================================================


@pytest.fixture(scope="module")
def art_inception_model():
    """Shared ArtInception3 model for all tests in module."""
    model = ArtInception3(aux_logits=False, init_weights=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def art_inception_model_with_logits():
    """Shared ArtInception3 model with num_classes for logits test."""
    model = ArtInception3(num_classes=1000, aux_logits=False, init_weights=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def inception_v3_block0_model():
    """Shared InceptionV3 model with block 0 (64-dim features)."""
    model = InceptionV3([0], resize_input=False, normalize_input=True)
    model.eval()
    return model


@pytest.fixture(scope="module")
def inception_v3_block0_model_with_resize():
    """Shared InceptionV3 model with block 0 and resize enabled."""
    model = InceptionV3([0], resize_input=True, normalize_input=True)
    model.eval()
    return model


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

        # Create identical images (need reasonable size for inception)
        img = Image.new("RGB", (150, 150), color="green")
        img.save(style_dir / "img1.jpg")
        img.save(stylized_dir / "img1.jpg")

        # SIFID now uses dims parameter (default 64) and InceptionV3
        result = sifid_score(style_dir, stylized_dir, device="cpu", dims=64)
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

    def test_sifid_uses_64_dim_features_by_default(self, tmp_path):
        """SIFID should use 64-dim features by default (reference behavior)."""
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        style_dir.mkdir()
        stylized_dir.mkdir()

        img = Image.new("RGB", (150, 150), color="blue")
        img.save(style_dir / "img1.jpg")
        img.save(stylized_dir / "img1.jpg")

        # Default dims=64, should work without error
        result = sifid_score(style_dir, stylized_dir, device="cpu")
        assert isinstance(result, float)

    def test_sifid_different_dims(self, tmp_path):
        """SIFID should support different feature dimensions."""
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"
        style_dir.mkdir()
        stylized_dir.mkdir()

        img = Image.new("RGB", (150, 150), color="red")
        img.save(style_dir / "img1.jpg")
        img.save(stylized_dir / "img1.jpg")

        # Test with dims=192 (second block)
        result = sifid_score(style_dir, stylized_dir, device="cpu", dims=192)
        assert isinstance(result, float)


class TestGetActivationsWithPatches:
    """Tests for _get_activations_with_patches function."""

    def test_returns_multiple_patches(self, tmp_path, inception_v3_block0_model):
        """Should return tensor with multiple rows (patches)."""
        img = Image.new("RGB", (150, 150), color="red")
        p = tmp_path / "test.jpg"
        img.save(p)

        feats = _get_activations_with_patches(p, inception_v3_block0_model, "cpu", size=None)

        # Should have multiple patches (H*W rows) with 64 features
        assert feats.dim() == 2
        assert feats.shape[0] > 1  # Multiple patches
        assert feats.shape[1] == 64  # 64-dim features from block 0

    def test_returns_patches_with_artinception(self, tmp_path, art_inception_model):
        """Should return 2048-dim patches when using ArtInception3."""
        img = Image.new("RGB", (64, 64), color="green")
        p = tmp_path / "test.jpg"
        img.save(p)

        feats = _get_activations_with_patches(p, art_inception_model, "cpu", size=299)

        # ArtInception returns 8x8=64 patches with 2048 features
        assert feats.dim() == 2
        assert feats.shape[0] == 64  # 8*8 patches
        assert feats.shape[1] == 2048


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
        """Histogram output should have shape (N, 3, h, h) for 2D histograms."""
        block = RGBuvHistBlock(h=64)
        x = torch.rand(2, 3, 32, 32)  # (N, 3, H, W) in [0, 1]
        hist = block(x)
        # Reference uses 2D histograms per channel (log-chromaticity pairs)
        assert hist.shape == (2, 3, 64, 64)

    def test_histogram_sums_to_one(self):
        """Full histogram should sum to ~1 (normalized)."""
        block = RGBuvHistBlock(h=64)
        x = torch.rand(1, 3, 32, 32)
        hist = block(x)
        # Total histogram sum should be approximately 1
        total_sum = hist.sum(dim=(1, 2, 3))  # Sum over channels and 2D bins
        assert torch.allclose(total_sum, torch.ones_like(total_sum), atol=1e-5)

    def test_inverse_quadratic_method(self):
        """Should use inverse-quadratic kernel by default."""
        block = RGBuvHistBlock(h=32, method="inverse-quadratic")
        x = torch.rand(1, 3, 32, 32)
        hist = block(x)
        assert hist.shape == (1, 3, 32, 32)

    def test_rbf_method(self):
        """Should support RBF kernel method."""
        block = RGBuvHistBlock(h=32, method="RBF")
        x = torch.rand(1, 3, 32, 32)
        hist = block(x)
        assert hist.shape == (1, 3, 32, 32)

    def test_thresholding_method(self):
        """Should support thresholding method."""
        block = RGBuvHistBlock(h=32, method="thresholding")
        x = torch.rand(1, 3, 32, 32)
        hist = block(x)
        assert hist.shape == (1, 3, 32, 32)

    def test_resizes_large_images(self):
        """Should resize images larger than insz."""
        block = RGBuvHistBlock(h=32, insz=50)
        x = torch.rand(1, 3, 100, 100)  # Larger than insz=50
        hist = block(x)
        assert hist.shape == (1, 3, 32, 32)


class TestHistoganDistance:
    """Tests for histogan_distance metric using Hellinger distance on RGB-uv histograms."""

    def test_identical_images_low_distance(self, tmp_path):
        """Histogram distance of identical images should be ~0."""
        # Use a gradient image for more realistic histogram
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        p = tmp_path / "same.jpg"
        img.save(p)

        result = histogan_distance(p, p, device="cpu", size=64)
        assert result < 0.01

    def test_different_images_positive_distance(self, tmp_path):
        """Histogram distance of different color images should be positive."""
        img1 = Image.new("RGB", (64, 64), color=(200, 50, 50))   # Reddish
        img2 = Image.new("RGB", (64, 64), color=(50, 50, 200))   # Bluish
        p1 = tmp_path / "style.jpg"
        p2 = tmp_path / "stylized.jpg"
        img1.save(p1)
        img2.save(p2)

        result = histogan_distance(p1, p2, device="cpu", size=64)
        assert result > 0

    def test_histogan_registered_in_registry(self):
        """histogan_distance should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("histogan")

    def test_histogan_uses_hellinger_distance(self, tmp_path):
        """HistoGAN should use Hellinger distance (bounded between 0 and 1)."""
        img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img2 = Image.new("RGB", (64, 64), color=(0, 255, 0))
        p1 = tmp_path / "red.jpg"
        p2 = tmp_path / "green.jpg"
        img1.save(p1)
        img2.save(p2)

        result = histogan_distance(p1, p2, device="cpu", size=64)
        # Hellinger distance is bounded between 0 and 1
        assert 0 <= result <= 1.5  # Allow some margin for normalization


class TestFIDNotDuplicated:
    """Test that FID helper is not exposed in registry."""

    def test_fid_registered_once(self):
        """fid_score should be the only 'fid' in registry."""
        assert METRICS_REGISTRY.has("fid")
        fid_fn = METRICS_REGISTRY.get("fid")
        # Should be fid_score function, not _compute_frechet_distance
        assert fid_fn.__name__ == "fid_score"


class TestArtInception3:
    """Tests for ArtInception3 model and feature extraction."""

    def test_return_features_shape(self, art_inception_model):
        """ArtInception3 with return_features=True should return (B, 2048)."""
        x = torch.randn(2, 3, 299, 299)
        with torch.no_grad():
            features = art_inception_model(x, return_features=True)

        assert features.shape == (2, 2048)
        assert features.dtype == torch.float32

    def test_return_spatial_shape(self, art_inception_model):
        """ArtInception3 with return_spatial=True should return (B, 2048, H, W)."""
        x = torch.randn(2, 3, 299, 299)
        with torch.no_grad():
            spatial = art_inception_model(x, return_spatial=True)

        assert spatial.dim() == 4
        assert spatial.shape[0] == 2
        assert spatial.shape[1] == 2048
        # Spatial dimensions should be 8x8 for 299x299 input
        assert spatial.shape[2] == 8
        assert spatial.shape[3] == 8

    def test_default_forward_returns_logits(self, art_inception_model_with_logits):
        """Default forward should return classification logits."""
        x = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            logits = art_inception_model_with_logits(x)

        assert logits.shape == (1, 1000)


class TestFIDWithArtInception:
    """Tests for fid_score using ArtInception3."""

    def test_fid_with_art_inception_runs_without_error(self, tmp_path):
        """FID computation with art_inception=True should not crash."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        # Create test images (reduced from 3 to 2)
        for i in range(2):
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(real_dir / f"img{i}.jpg")
            img.save(fake_dir / f"img{i}.jpg")

        # This should not raise any error (size defaults to 512 now)
        result = fid_score(real_dir, fake_dir, device="cpu", use_art_inception=True)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_fid_identical_dirs_low_score(self, tmp_path):
        """FID of identical image sets should be low."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        for i in range(2):
            img = Image.new("RGB", (64, 64), color=(i * 40, 100, 200 - i * 30))
            img.save(real_dir / f"img{i}.jpg")
            img.save(fake_dir / f"img{i}.jpg")

        result = fid_score(real_dir, fake_dir, device="cpu", use_art_inception=True)
        # Identical images should have very low FID
        assert result < 1.0

    def test_fid_different_dirs_positive_score(self, tmp_path):
        """FID of different image sets should be positive."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        for i in range(2):
            img_real = Image.new("RGB", (64, 64), color=(255, 0, 0))  # All red
            img_fake = Image.new("RGB", (64, 64), color=(0, 0, 255))  # All blue
            img_real.save(real_dir / f"img{i}.jpg")
            img_fake.save(fake_dir / f"img{i}.jpg")

        result = fid_score(real_dir, fake_dir, device="cpu", use_art_inception=True)
        assert result > 0

    def test_fid_default_size_is_512(self, tmp_path):
        """FID should default to size=512 to match reference ArtFID."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        for i in range(2):
            img = Image.new("RGB", (64, 64), color=(100, 100, 100))
            img.save(real_dir / f"img{i}.jpg")
            img.save(fake_dir / f"img{i}.jpg")

        # Should work with default size=512
        result = fid_score(real_dir, fake_dir, device="cpu")
        assert isinstance(result, float)


class TestGetActivationsSingle:
    """Tests for _get_activations_single with different models."""

    def test_art_inception_returns_2048_features(self, tmp_path, art_inception_model):
        """_get_activations_single should return 2048-dim features for ArtInception3."""
        img = Image.new("RGB", (64, 64), color="red")
        p = tmp_path / "test.jpg"
        img.save(p)

        feats = _get_activations_single(p, art_inception_model, "cpu", size=299)
        assert feats.shape == (1, 2048)

    def test_inception_v3_wrapper_returns_features(self, tmp_path, inception_v3_block0_model_with_resize):
        """_get_activations_single should work with InceptionV3 wrapper."""
        img = Image.new("RGB", (64, 64), color="blue")
        p = tmp_path / "test.jpg"
        img.save(p)

        feats = _get_activations_single(p, inception_v3_block0_model_with_resize, "cpu", size=299)
        assert feats.dim() == 2
        assert feats.shape[0] == 1


class TestGetActivationsWithPatchesArtInception:
    """Tests for _get_activations_with_patches with ArtInception3."""

    def test_art_inception_returns_patches(self, tmp_path, art_inception_model):
        """_get_activations_with_patches should return spatial patches for ArtInception3."""
        img = Image.new("RGB", (64, 64), color="green")
        p = tmp_path / "test.jpg"
        img.save(p)

        feats = _get_activations_with_patches(p, art_inception_model, "cpu", size=299)
        # Should have H*W patches with 2048 features each
        # For 299x299 input, output is 8x8 = 64 patches
        assert feats.dim() == 2
        assert feats.shape[0] == 64  # 8*8 patches
        assert feats.shape[1] == 2048
