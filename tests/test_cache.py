"""Unit tests for the ModelCache utility."""

from rt_games.utils.cache import ModelCache


class TestModelCache:
    """Tests for ModelCache class."""

    def setup_method(self):
        """Reset cache before each test."""
        ModelCache._vgg = None
        ModelCache._midas = None
        ModelCache._inception = None
        ModelCache._art_inception = None
        ModelCache._flow = None
        ModelCache._lpips = None

    def test_get_lpips_caches_model(self):
        """LPIPS model should be cached and reused."""
        call_count = 0

        def factory(device):
            nonlocal call_count
            call_count += 1
            return f"lpips_model_{device}"

        # First call - should invoke factory
        result1 = ModelCache.get_lpips(factory, "cpu")
        assert call_count == 1
        assert result1 == "lpips_model_cpu"

        # Second call - should return cached model, not invoke factory again
        result2 = ModelCache.get_lpips(factory, "cpu")
        assert call_count == 1  # Still 1, not 2
        assert result2 == result1

    def test_get_vgg_caches_model(self):
        """VGG model should be cached and reused."""
        call_count = 0

        def factory(device):
            nonlocal call_count
            call_count += 1
            return f"vgg_model_{device}"

        result1 = ModelCache.get_vgg(factory, "cpu")
        result2 = ModelCache.get_vgg(factory, "cpu")

        assert call_count == 1
        assert result1 == result2

    def test_get_midas_caches_model(self):
        """MiDaS model should be cached and reused."""
        call_count = 0

        def factory(device):
            nonlocal call_count
            call_count += 1
            return "midas_model"

        result1 = ModelCache.get_midas(factory, "cpu")
        result2 = ModelCache.get_midas(factory, "cpu")

        assert call_count == 1
        assert result1 == result2

    def test_get_inception_caches_separately_by_art_flag(self):
        """Inception and ArtInception should be cached separately."""
        call_count = 0

        def factory(device):
            nonlocal call_count
            call_count += 1
            return f"inception_{call_count}"

        # Regular inception
        result1 = ModelCache.get_inception(factory, "cpu", art=False)
        ModelCache.get_inception(factory, "cpu", art=False)  # Second call
        assert call_count == 1

        # Art inception - should create new model
        result3 = ModelCache.get_inception(factory, "cpu", art=True)
        assert call_count == 2
        assert result3 != result1

        # Art inception again - should be cached
        result4 = ModelCache.get_inception(factory, "cpu", art=True)
        assert call_count == 2
        assert result4 == result3

    def test_get_flow_caches_model(self):
        """Flow model should be cached and reused."""
        call_count = 0

        def factory(device):
            nonlocal call_count
            call_count += 1
            return "flow_model"

        result1 = ModelCache.get_flow(factory, "cpu")
        result2 = ModelCache.get_flow(factory, "cpu")

        assert call_count == 1
        assert result1 == result2


class TestLPIPSCachingIntegration:
    """Integration tests to verify LPIPS is actually cached in practice."""

    def setup_method(self):
        """Reset LPIPS cache before each test."""
        ModelCache._lpips = None

    def test_lpips_content_uses_cache(self, tmp_path):
        """lpips_content should use cached LPIPS model across multiple calls."""
        from PIL import Image
        from rt_games.metrics.perceptual import lpips_content

        # Create test images
        img1 = Image.new("RGB", (64, 64), "red")
        img2 = Image.new("RGB", (64, 64), "blue")
        p1 = tmp_path / "img1.jpg"
        p2 = tmp_path / "img2.jpg"
        img1.save(p1)
        img2.save(p2)

        # First call
        lpips_content(p1, p2, device="cpu", size=64)

        # Cache should now be populated
        assert ModelCache._lpips is not None
        cached_model = ModelCache._lpips

        # Second call should reuse same model instance
        lpips_content(p1, p2, device="cpu", size=64)
        assert ModelCache._lpips is cached_model  # Same object
