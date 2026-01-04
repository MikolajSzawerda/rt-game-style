# rt-games — Style Transfer Evaluation Toolkit

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Coverage](docs/coverage.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

Python CLI for evaluating style transfer models (image + game/video) with SOTA metrics. You provide frames; the toolkit aligns them, runs metrics, and writes CSVs.

## Installation

```bash
# Using pip
pip install torch torchvision  # install PyTorch first (https://pytorch.org/get-started)
pip install -e .

# Using uv
uv sync
```

**Python 3.10+** required.

---

## Quick Start

```bash
# Evaluate a single method (image mode)
python -m rt_games.cli \
  --content evaluation/content \
  --style evaluation/style \
  --stylized evaluation/methods/AdaIN \
  --metrics lpips,ssim,gram_loss,fid \
  --output results/adain.csv

# Evaluate all methods in a directory
python -m rt_games.cli \
  --content evaluation/content \
  --style evaluation/style \
  --methods-dir evaluation/methods \
  --output results/all_methods.csv

# Temporal/video metrics
python -m rt_games.cli \
  --mode temporal \
  --original scenes/game1/original \
  --stylized scenes/game1/stylized \
  --metrics warping_error,temporal_lpips,depth_error \
  --output results/temporal.csv
```

---

## Available Metrics

| Metric | Category | Compares | Description |
|--------|----------|----------|-------------|
| `lpips` | Content | content ↔ stylized | Learned perceptual similarity (AlexNet) |
| `ssim` | Content | content ↔ stylized | Structural similarity index |
| `content_loss` | Content | content ↔ stylized | VGG feature MSE (relu2_2) |
| `gram_loss` | Style | style ↔ stylized | Gram matrix MSE across VGG layers |
| `histogan` | Style | style ↔ stylized | RGB-uv histogram distance |
| `cfsd` | Content | content ↔ stylized | Patch-wise cosine similarity (VGG relu3_3) |
| `fid` | Style | style_dir ↔ stylized_dir | Fréchet Inception Distance (dataset-level) |
| `sifid` | Style | style ↔ stylized | Single-image FID (per-image, averaged) |
| `artfid` | Composite | all three | `(1 + LPIPS) × (1 + FID)` |
| `warping_error` | Temporal | frame_t ↔ frame_t+1 | Flow-based temporal consistency |
| `temporal_lpips` | Temporal | frame_t ↔ frame_t+1 | Perceptual temporal consistency |
| `depth_error` | Temporal | original ↔ stylized | Scale-invariant depth preservation |

---

## Model Dependencies by Metric

Each metric may download/load one or more pretrained models. First run may take longer due to downloads.

| Metric | Model(s) Downloaded | Size (approx) | Source |
|--------|---------------------|---------------|--------|
| `lpips` | AlexNet (LPIPS weights) | ~9 MB | lpips library |
| `ssim` | — | — | Pure computation (piq) |
| `content_loss` | VGG16 (ImageNet) | ~528 MB | torchvision |
| `gram_loss` | VGG16 (ImageNet) | ~528 MB | torchvision |
| `histogan` | — | — | Pure computation |
| `cfsd` | VGG16 (ImageNet) | ~528 MB | torchvision |
| `fid` | art_inception¹ | ~104 MB | HuggingFace (matthias-wright) |
| `sifid` | art_inception¹ | ~104 MB | HuggingFace (matthias-wright) |
| `artfid` | AlexNet + art_inception | ~113 MB | lpips + HuggingFace |
| `warping_error` | RAFT (raft-small)² | ~20 MB | torch.hub (princeton-vl) |
| `temporal_lpips` | RAFT + AlexNet | ~29 MB | torch.hub + lpips |
| `depth_error` | MiDaS (DPT_Hybrid) | ~470 MB | torch.hub (intel-isl) |

¹ Uses art_inception by default for SOTA comparability. Set `use_art_inception=False` in config for standard InceptionV3.  
² Skipped if you provide precomputed flow via `--flow`.

**Note:** Models are cached after first load — subsequent runs reuse cached weights.

---

## CLI Reference

### Global Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | `image` / `temporal` | `image` | Evaluation mode |
| `--device` | str | `cuda` | Device (`cuda` / `cpu`) |
| `--image-size` | int | original | Resize images to this size |
| `--metrics` | str | all | Comma-separated list of metrics |
| `--output` | path | **required** | Output CSV path |
| `-q, --quiet` | flag | false | Suppress progress bars |

### Image Mode Options

| Flag | Type | Description |
|------|------|-------------|
| `--content` | path | Directory with content images |
| `--style` | path | Directory with style images |
| `--stylized` | path | Directory with stylized images (single method) |
| `--methods-dir` | path | Directory containing multiple method folders |

### Temporal Mode Options

| Flag | Type | Description |
|------|------|-------------|
| `--original` | path | Directory with original frames |
| `--stylized` | path | Directory with stylized frames |
| `--flow` | path | (optional) Precomputed optical flow (.pt files) |
| `--depth` | path | (optional) Precomputed depth maps |

---

## Expected File Layout

### Image Mode

```
evaluation/
├── content/                    # Content images
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── style/                      # Style reference images
│   ├── starry_night.jpg
│   ├── mosaic.jpg
│   └── ...
└── methods/
    ├── AdaIN/                  # Stylized outputs per method
    │   ├── 001_stylized_starry_night.jpg
    │   ├── 001_stylized_mosaic.jpg
    │   ├── 002_stylized_starry_night.jpg
    │   └── ...
    ├── StyleID/
    └── MambaST/
```

**⚠️ Naming Convention:** Stylized files MUST follow `{content_stem}_stylized_{style_stem}.{ext}`
- `001_stylized_starry_night.jpg` → content: `001.jpg`, style: `starry_night.jpg`

The CLI validates alignment and fails fast on mismatches.

### Temporal Mode

```
scenes/game1_scene1/
├── original/                   # Sequential original frames
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── stylized/                   # Corresponding stylized frames
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── flow/                       # (optional) Precomputed optical flow
│   ├── frame_0001_flow.pt      # Flow from frame_0001 → frame_0002
│   └── ...
└── depth/                      # (optional) Precomputed depth maps
    ├── frame_0001.png
    └── ...
```

**Flow format:** `{frame_stem}_flow.pt` containing a `(2, H, W)` tensor.

---

## Usage Examples

### Basic Image Evaluation

```bash
# All default metrics (lpips, ssim, content_loss, gram_loss, histogan, cfsd)
python -m rt_games.cli \
  --content data/content \
  --style data/style \
  --stylized data/stylized/my_method \
  --output results/my_method.csv
```

### Specific Metrics Only

```bash
# Just LPIPS and SSIM for quick content preservation check
python -m rt_games.cli \
  --content data/content \
  --style data/style \
  --stylized data/stylized/my_method \
  --metrics lpips,ssim \
  --output results/quick_check.csv
```

### Full Benchmark (All Image Metrics)

```bash
python -m rt_games.cli \
  --content data/content \
  --style data/style \
  --stylized data/stylized/my_method \
  --metrics lpips,ssim,content_loss,gram_loss,histogan,cfsd,fid,sifid,artfid \
  --output results/full_benchmark.csv
```

### Compare Multiple Methods

```bash
# Evaluate all subdirectories under methods/
python -m rt_games.cli \
  --content data/content \
  --style data/style \
  --methods-dir data/methods \
  --metrics lpips,gram_loss,artfid \
  --output results/comparison.csv
```

Output CSV will have one row per method:
```csv
method,lpips,gram_loss,artfid
AdaIN,0.342,0.0021,15.23
StyleID,0.298,0.0018,12.87
MambaST,0.315,0.0019,14.01
```

### Temporal Video Evaluation

```bash
# With RAFT flow estimation (automatic)
python -m rt_games.cli \
  --mode temporal \
  --original scenes/game1/original \
  --stylized scenes/game1/stylized \
  --metrics warping_error,temporal_lpips \
  --output results/temporal.csv

# With precomputed flow (faster, recommended for games)
python -m rt_games.cli \
  --mode temporal \
  --original scenes/game1/original \
  --stylized scenes/game1/stylized \
  --flow scenes/game1/flow \
  --metrics warping_error,temporal_lpips \
  --output results/temporal.csv

# Depth preservation check
python -m rt_games.cli \
  --mode temporal \
  --original scenes/game1/original \
  --stylized scenes/game1/stylized \
  --depth scenes/game1/depth \
  --metrics depth_error \
  --output results/depth.csv
```

### CPU-only Evaluation

```bash
python -m rt_games.cli \
  --device cpu \
  --content data/content \
  --style data/style \
  --stylized data/stylized/my_method \
  --metrics lpips,ssim \
  --output results/cpu_eval.csv
```

### Resize for Faster Evaluation

```bash
# Resize all images to 256×256 for quick iteration
python -m rt_games.cli \
  --image-size 256 \
  --content data/content \
  --style data/style \
  --stylized data/stylized/my_method \
  --output results/quick.csv
```

---

## Notes

- **FID/ArtFID** use the `art_inception` checkpoint for comparability with SOTA papers (StyleID, MambaST, AttenST).
- **Model caching** avoids reloading VGG/MiDaS/Inception between images.
- **Unity/Game engines:** Export frames in consistent color space (sRGB after tone mapping). If possible, export engine motion vectors and depth to skip RAFT/MiDaS overhead.
- **Supported image formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

---

## Metric Details

See **[METRICS.md](METRICS.md)** for full documentation including:
- Paper references for each metric
- Mathematical definitions
- Model architecture details
- Evaluation protocols from published papers
