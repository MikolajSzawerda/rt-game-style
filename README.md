## rt-games metrics toolkit (evaluation-only)

Python CLI to evaluate style transfer models (image + game/video) with SOTA metrics used in `paper.md`, MambaST, StyleID, AttenST. You provide frames; the toolkit aligns them, runs metrics, and writes CSVs.

### Metrics

See **[METRICS.md](METRICS.md)** for full documentation with paper sources and model dependencies.

| Category | Metrics |
|----------|---------|
| Content/style | LPIPS, SSIM, Content Loss (VGG), Gram Loss |
| Style fidelity | FID_style (art_inception), SIFID, HistoGAN distance, CFSD |
| Composite | ArtFID = (1 + LPIPS_content) × (1 + FID_style) |
| Temporal/game | Warping Error (flow-based), Temporal LPIPS, Depth Error (MiDaS) |

### Expected input layout
Image metrics:
```
evaluation/
├── content/                    # 001.jpg, 002.jpg, ...
├── style/                      # starry_night.jpg, ...
└── methods/
    ├── AdaIN/
    │   ├── 001_stylized_starry_night.jpg
    ├── StyleID/
    └── MambaST/
```
- Naming: `{content_stem}_stylized_{style_stem}.{ext}` to align stylized → content & style.

Temporal/game metrics:
```
scenes/game1_scene1/
├── original/   # sequential frames
├── stylized/
├── flow/       # optional (engine or RAFT, saved as *_flow.pt)
└── depth/      # optional (engine depth or MiDaS cache)
```

### CLI examples
```bash
# Image metrics
python -m rt_games.cli \
  --content evaluation/content \
  --style evaluation/style \
  --stylized evaluation/methods/AdaIN \
  --metrics lpips,ssim,artfid,cfsd,gram_loss,histogan \
  --output results/AdaIN.csv

# Temporal metrics
python -m rt_games.cli \
  --mode temporal \
  --original scenes/game1_scene1/original \
  --stylized scenes/game1_scene1/stylized \
  --metrics warping_error,temporal_lpips,depth_error \
  --output results/temporal.csv

# Batch all methods under evaluation/methods
python -m rt_games.cli \
  --content evaluation/content \
  --style evaluation/style \
  --methods-dir evaluation/methods \
  --output results/all_methods.csv
```

### Notes
- FID/ArtFID use the art_inception checkpoint for comparability with SOTA papers.
- Model caching avoids reloading VGG/MiDaS/Inception; heavy metrics (FID/SIFID/CFSD, flow/depth) support sampling/optional precomputed data.
- Unity: export frames in a consistent color space (sRGB after tone mapping or linear—be consistent for original/stylized). If possible, export engine motion vectors and depth to skip RAFT/MiDaS.
- Validation: stylized filenames must match the naming convention; the CLI fails fast on misalignment.

