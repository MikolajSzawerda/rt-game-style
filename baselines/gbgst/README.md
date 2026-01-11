# GBGST Baseline

**G-buffer Guided Style Transfer for Computer Games**

This baseline wraps the style transfer network from:

> Ioannou & Maddock, "Towards Real-time G-buffer-Guided Style Transfer in Computer Games"  
> IEEE Transactions on Games, 2024

**Source repository:** https://github.com/ioannouE/GBGST

## Quick Start

```bash
# Full setup (init submodule + sync deps + link weights)
just setup

# Or step by step:
just init-submodule    # Clone GBGST repo as submodule
just sync              # Install Python dependencies
just link-weights      # Symlink weights from submodule

# Generate stylized images
just generate

# Generate with specific style
just generate-style starry_night
```

## Available Styles

Pretrained models available from the GBGST repository:

- `starry_night` - Van Gogh's Starry Night
- `mosaic` - Mosaic pattern
- `feathers` - Feathers texture
- `wave` - Hokusai's Great Wave
- `composition_vii` - Kandinsky's Composition VII

## Directory Structure

```
gbgst/
├── justfile           # Task runner recipes
├── pyproject.toml     # Python dependencies
├── generate.py        # Batch stylization wrapper
├── repo/              # Git submodule → github.com/ioannouE/GBGST
│   ├── depth-aware-nst/
│   │   └── transformer_net_light.py  # Model architecture
│   └── saved_models/                  # Pretrained weights
│       ├── starry_night.pth
│       └── ...
├── weights/           # Symlinks to repo/saved_models/*.pth
└── outputs/           # Test outputs
```

## Submodule Management

```bash
# Check submodule status
just submodule-status

# Update to latest version
just submodule-update

# If cloning rt-games fresh, initialize submodule:
git submodule update --init --recursive
```

## Alternative: Direct Download

If you don't want to use the submodule, you can download weights directly:

```bash
just download-weights
```

This downloads `.pth` files from GitHub instead of symlinking from the submodule.

## Model Architecture

TransformerNetLight is a fast feed-forward network:
- **Encoder**: 3 convolutional layers with instance normalization
- **Transform**: 2 residual blocks
- **Decoder**: 3 upsampling layers

The model runs at ~30 FPS on modern GPUs for 1080p images.

## Citation

```bibtex
@article{ioannou2024gbgst,
  title={Towards Real-time G-buffer-Guided Style Transfer in Computer Games},
  author={Ioannou, Eleftherios and Maddock, Steve},
  journal={IEEE Transactions on Games},
  year={2024},
  publisher={IEEE}
}
```

## License

The GBGST code is from the [original repository](https://github.com/ioannouE/GBGST).
This wrapper script (`generate.py`) is MIT licensed.
