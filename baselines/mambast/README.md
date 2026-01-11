# MambaST Baseline

**State Space Model for Efficient Style Transfer (WACV 2025)**

This baseline wraps the style transfer network from:

> Botti et al., "Mamba-ST: State Space Model for Efficient Style Transfer"  
> WACV 2025

**Source repository:** https://github.com/FilippoBotti/MambaST

## Key Features

Like AdaAttN, MambaST is an **arbitrary style transfer** model:
- **Single model** works with any style image
- No retraining needed for new styles
- Uses State Space Model (Mamba) for efficient style transfer
- Faster than transformer-based methods with comparable quality

## Quick Start

```bash
# Full setup (init submodule + sync deps + download weights)
just setup

# Or step by step:
just init-submodule    # Clone MambaST repo as submodule
just sync              # Install Python dependencies
just download-weights  # Download pretrained weights (~200MB)

# Generate stylized images
just gen starry_night 10

# List available styles
just list-styles
```

## Directory Structure

```
mambast/
├── justfile           # Task runner recipes
├── pyproject.toml     # Python dependencies
├── generate.py        # Batch stylization wrapper
├── README.md
└── repo/              # Git submodule → github.com/FilippoBotti/MambaST
    ├── models/
    │   ├── MambaST.py
    │   ├── mamba.py
    │   └── models_helper.py
    └── util/
        └── utils.py
```

## Weights Structure

After running `just download-weights`:

```
../data/weights/mambast/
├── vgg_normalised.pth          # VGG encoder (pretrained)
├── decoder_iter_160000.pth     # Decoder network
├── mamba_iter_160000.pth       # Mamba network
└── embedding_iter_160000.pth   # Patch embedding network
```

## Manual Weight Download

If automatic download fails, download manually:

1. Go to [Google Drive](https://drive.google.com/drive/folders/1pVhJFwk2f3arP7zUDFAe5_PJrPSG1gc2)
2. Download all `.pth` files
3. Place in `../data/weights/mambast/`

## Requirements

**CUDA Required**: MambaST uses `mamba-ssm` which requires CUDA for compilation.

```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Comparison with Other Methods

| Feature | GBGST | AdaAttN | MambaST |
|---------|-------|---------|---------|
| Model type | Per-style trained | Arbitrary style | Arbitrary style |
| Weights per style | ~6MB each | ~150MB total | ~200MB total |
| New style | Requires training | Just provide image | Just provide image |
| Architecture | CNN | Attention | State Space Model |
| Speed | Fastest | Slower | Fast |

## Model Architecture

MambaST uses a State Space Model design:

1. **VGG Encoder**: Extract features from content and style
2. **Patch Embedding**: Project features to embedding space
3. **Mamba Encoder/Decoder**: Process style and content with SSM
4. **CNN Decoder**: Reconstruct stylized image

## Citation

```bibtex
@inproceedings{botti2025mamba,
  author={Botti, Filippo and Ergasti, Alex and Rossi, Leonardo and Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={Mamba-ST: State Space Model for Efficient Style Transfer}, 
  year={2025},
  pages={7797-7806},
  doi={10.1109/WACV61041.2025.00757}
}
```

## License

The MambaST code is from the [original repository](https://github.com/FilippoBotti/MambaST).
This wrapper script (`generate.py`) is MIT licensed.

