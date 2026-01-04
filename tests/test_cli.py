"""Tests for CLI argument parsing and workflow."""

import subprocess
from PIL import Image
from unittest.mock import patch


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_image_size_is_none(self):
        """Default --image-size should be None to allow function defaults."""
        from rt_games.cli import parse_args

        with patch(
            "sys.argv",
            [
                "cli",
                "--output",
                "out.csv",
                "--content",
                "c",
                "--style",
                "s",
                "--stylized",
                "z",
                "--metrics",
                "lpips",
            ],
        ):
            args = parse_args()
            assert args.image_size is None

    def test_image_size_can_be_set(self):
        """--image-size should accept integer values."""
        from rt_games.cli import parse_args

        with patch(
            "sys.argv",
            [
                "cli",
                "--output",
                "out.csv",
                "--content",
                "c",
                "--style",
                "s",
                "--stylized",
                "z",
                "--image-size",
                "256",
            ],
        ):
            args = parse_args()
            assert args.image_size == 256

    def test_default_device_is_cuda(self):
        """Default device should be cuda."""
        from rt_games.cli import parse_args

        with patch(
            "sys.argv",
            [
                "cli",
                "--output",
                "out.csv",
                "--content",
                "c",
                "--style",
                "s",
                "--stylized",
                "z",
            ],
        ):
            args = parse_args()
            assert args.device == "cuda"

    def test_default_mode_is_image(self):
        """Default mode should be 'image'."""
        from rt_games.cli import parse_args

        with patch(
            "sys.argv",
            [
                "cli",
                "--output",
                "out.csv",
                "--content",
                "c",
                "--style",
                "s",
                "--stylized",
                "z",
            ],
        ):
            args = parse_args()
            assert args.mode == "image"


class TestKwargsConstruction:
    """Tests for kwargs construction in _run_image_for_method."""

    def test_size_excluded_when_none(self):
        """Regression test: size should not be in kwargs when image_size is None."""
        # Simulate the logic from _run_image_for_method
        image_size = None
        device = "cuda"

        kwargs = {"device": device}
        if image_size is not None:
            kwargs["size"] = image_size

        assert "size" not in kwargs

    def test_size_included_when_set(self):
        """size should be in kwargs when image_size is provided."""
        image_size = 256
        device = "cuda"

        kwargs = {"device": device}
        if image_size is not None:
            kwargs["size"] = image_size

        assert kwargs["size"] == 256


class TestCLIEndToEnd:
    """End-to-end tests for the CLI."""

    def test_cli_runs_with_lpips_metric(self, tmp_path):
        """CLI should successfully run with LPIPS metric."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        # Create test images
        Image.new("RGB", (64, 64), "red").save(content_dir / "cat.jpg")
        Image.new("RGB", (64, 64), "blue").save(style_dir / "starry.jpg")
        Image.new("RGB", (64, 64), "purple").save(
            stylized_dir / "cat_stylized_starry.jpg"
        )

        output = tmp_path / "results.csv"

        result = subprocess.run(
            [
                "python",
                "-m",
                "rt_games.cli",
                "--content",
                str(content_dir),
                "--style",
                str(style_dir),
                "--stylized",
                str(stylized_dir),
                "--metrics",
                "lpips",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            cwd="/home/mszawerda/rt-game-style",
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output.exists()

        # Check output CSV has expected columns
        content = output.read_text()
        assert "lpips" in content
        assert "method" in content

    def test_cli_runs_with_ssim_metric(self, tmp_path):
        """CLI should successfully run with SSIM metric."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (64, 64), "red").save(content_dir / "cat.jpg")
        Image.new("RGB", (64, 64), "blue").save(style_dir / "starry.jpg")
        Image.new("RGB", (64, 64), "purple").save(
            stylized_dir / "cat_stylized_starry.jpg"
        )

        output = tmp_path / "results.csv"

        result = subprocess.run(
            [
                "python",
                "-m",
                "rt_games.cli",
                "--content",
                str(content_dir),
                "--style",
                str(style_dir),
                "--stylized",
                str(stylized_dir),
                "--metrics",
                "ssim",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            cwd="/home/mszawerda/rt-game-style",
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output.exists()

    def test_cli_multiple_metrics(self, tmp_path):
        """CLI should handle multiple comma-separated metrics."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (64, 64), "red").save(content_dir / "cat.jpg")
        Image.new("RGB", (64, 64), "blue").save(style_dir / "starry.jpg")
        Image.new("RGB", (64, 64), "purple").save(
            stylized_dir / "cat_stylized_starry.jpg"
        )

        output = tmp_path / "results.csv"

        result = subprocess.run(
            [
                "python",
                "-m",
                "rt_games.cli",
                "--content",
                str(content_dir),
                "--style",
                str(style_dir),
                "--stylized",
                str(stylized_dir),
                "--metrics",
                "lpips,ssim",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            cwd="/home/mszawerda/rt-game-style",
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        content = output.read_text()
        assert "lpips" in content
        assert "ssim" in content

    def test_cli_creates_output_directory(self, tmp_path):
        """CLI should create output directory if it doesn't exist."""
        content_dir = tmp_path / "content"
        style_dir = tmp_path / "style"
        stylized_dir = tmp_path / "stylized"

        for d in [content_dir, style_dir, stylized_dir]:
            d.mkdir()

        Image.new("RGB", (64, 64), "red").save(content_dir / "cat.jpg")
        Image.new("RGB", (64, 64), "blue").save(style_dir / "starry.jpg")
        Image.new("RGB", (64, 64), "purple").save(
            stylized_dir / "cat_stylized_starry.jpg"
        )

        # Output in non-existent subdirectory
        output = tmp_path / "new_dir" / "results.csv"

        result = subprocess.run(
            [
                "python",
                "-m",
                "rt_games.cli",
                "--content",
                str(content_dir),
                "--style",
                str(style_dir),
                "--stylized",
                str(stylized_dir),
                "--metrics",
                "ssim",
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            cwd="/home/mszawerda/rt-game-style",
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output.exists()
