"""Tests for temporal metrics (warping_error, temporal_lpips)."""

import pytest
from PIL import Image

from rt_games.utils.registry import METRICS_REGISTRY


class TestTemporalMetricsRegistry:
    """Tests for temporal metrics registration."""

    def test_warping_error_registered(self):
        """warping_error should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("warping_error")

    def test_temporal_lpips_registered(self):
        """temporal_lpips should be registered in METRICS_REGISTRY."""
        assert METRICS_REGISTRY.has("temporal_lpips")

    def test_warping_error_function_name(self):
        """Registered warping_error should be the correct function."""
        fn = METRICS_REGISTRY.get("warping_error")
        assert fn.__name__ == "warping_error"

    def test_temporal_lpips_function_name(self):
        """Registered temporal_lpips should be the correct function."""
        fn = METRICS_REGISTRY.get("temporal_lpips")
        assert fn.__name__ == "temporal_lpips"


class TestWarpingError:
    """Tests for warping_error metric."""

    def test_requires_minimum_frames(self, tmp_path):
        """warping_error should require at least 2 frames."""
        from rt_games.metrics.temporal import warping_error

        original_dir = tmp_path / "original"
        stylized_dir = tmp_path / "stylized"
        original_dir.mkdir()
        stylized_dir.mkdir()

        # Only 1 frame
        Image.new("RGB", (64, 64)).save(original_dir / "frame_001.jpg")
        Image.new("RGB", (64, 64)).save(stylized_dir / "frame_001.jpg")

        with pytest.raises(ValueError, match="at least two frames"):
            warping_error(original_dir, stylized_dir, device="cpu")

    def test_empty_directories_raises(self, tmp_path):
        """warping_error should raise error for empty directories."""
        from rt_games.metrics.temporal import warping_error

        original_dir = tmp_path / "original"
        stylized_dir = tmp_path / "stylized"
        original_dir.mkdir()
        stylized_dir.mkdir()

        with pytest.raises(ValueError, match="at least two frames"):
            warping_error(original_dir, stylized_dir, device="cpu")


class TestTemporalLPIPS:
    """Tests for temporal_lpips metric."""

    def test_requires_minimum_frames(self, tmp_path):
        """temporal_lpips should require at least 2 frames."""
        from rt_games.metrics.temporal import temporal_lpips

        original_dir = tmp_path / "original"
        stylized_dir = tmp_path / "stylized"
        original_dir.mkdir()
        stylized_dir.mkdir()

        # Only 1 frame
        Image.new("RGB", (64, 64)).save(original_dir / "frame_001.jpg")
        Image.new("RGB", (64, 64)).save(stylized_dir / "frame_001.jpg")

        with pytest.raises(ValueError, match="at least two frames"):
            temporal_lpips(original_dir, stylized_dir, device="cpu")

    def test_empty_directories_raises(self, tmp_path):
        """temporal_lpips should raise error for empty directories."""
        from rt_games.metrics.temporal import temporal_lpips

        original_dir = tmp_path / "original"
        stylized_dir = tmp_path / "stylized"
        original_dir.mkdir()
        stylized_dir.mkdir()

        with pytest.raises(ValueError, match="at least two frames"):
            temporal_lpips(original_dir, stylized_dir, device="cpu")
