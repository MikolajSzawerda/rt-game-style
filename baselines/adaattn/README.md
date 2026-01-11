# AdaAttN Baseline

**Attention-based Arbitrary Neural Style Transfer**

This baseline wraps the style transfer network from:

> Liu et al., "AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer"  
> ICCV 2021

**Source repository:** https://github.com/Huage001/AdaAttN

## Key Features

Unlike per-style models (e.g., GBGST), AdaAttN is an **arbitrary style transfer** model:
- **Single model** works with any style image
- No retraining needed for new styles
- Uses attention mechanism to transfer style features

## Quick Start

```bash
# Full setup (init submodule + sync deps + download weights)
just setup

# Or step by step:
just init-submodule    # Clone AdaAttN repo as submodule
just sync              # Install Python dependencies
just download-weights  # Download pretrained weights (~150MB)

# Generate stylized images
just gen starry_night 10

# List available styles
just list-styles
```

## Directory Structure

```
adaattn/
├── justfile           # Task runner recipes
├── pyproject.toml     # Python dependencies
├── generate.py        # Batch stylization wrapper
├── README.md
└── repo/              # Git submodule → github.com/Huage001/AdaAttN
    ├── models/
    │   ├── adaattn_model.py
    │   └── networks.py
    └── util/
        └── util.py
```

## Weights Structure

After running `just download-weights`:

```
../data/weights/adaattn/
├── vgg_normalised.pth          # VGG encoder (pretrained)
└── AdaAttN/
    ├── latest_net_decoder.pth      # Decoder network
    ├── latest_net_transformer.pth  # Transformer network
    └── latest_net_adaattn_3.pth    # AdaAttN module (skip connection)
```

## Manual Weight Download

If automatic download fails, download manually:

1. Go to [Google Drive](https://drive.google.com/file/d/1LngbEPLHVaVYnOvCTLf0M_yzgxOoEjjP/view)
2. Download `AdaAttN_model.zip`
3. Extract to `../data/weights/adaattn/`

## Comparison with GBGST

| Feature | GBGST | AdaAttN |
|---------|-------|---------|
| Model type | Per-style trained | Arbitrary style |
| Weights per style | ~6MB each | ~150MB total |
| New style | Requires training | Just provide image |
| Speed | Faster (simpler network) | Slower (attention) |
| Quality | Good for trained styles | Good for any style |

## Model Architecture

AdaAttN uses attention-adaptive normalization:

1. **VGG Encoder**: Extract features from content and style
2. **AdaAttN Module**: Compute attention between content and style features
3. **Transformer**: Merge multi-scale attended features
4. **Decoder**: Reconstruct stylized image

## Citation

```bibtex
@inproceedings{liu2021adaattn,
  title={AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer},
  author={Liu, Songhua and Lin, Tianwei and He, Dongliang and Li, Fu and Wang, Meiling and Li, Xin and Sun, Zhengxing and Li, Qian and Ding, Errui},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}
```

## License

The AdaAttN code is from the [original repository](https://github.com/Huage001/AdaAttN) under Apache-2.0 license.
This wrapper script (`generate.py`) is MIT licensed.

