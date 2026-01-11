# Shared Utilities

Shared download scripts and utilities used across all baselines.

## Directory Structure

```
baselines/
├── data/                        # All data (gitignored)
│   ├── datasets/                # Evaluation datasets
│   │   ├── content/             # Sample content images
│   │   ├── style/               # Style reference images
│   │   └── coco/
│   │       └── val2017/         # COCO validation (~5K images)
│   ├── weights/                 # Model weights per baseline
│   │   └── gbgst/
│   └── outputs/                 # Generated outputs per baseline
│       ├── gbgst/
│       └── gbgst_coco/
├── gbgst/                       # Baseline code (tracked)
├── shared/                      # This directory (tracked)
│   ├── download_coco.py
│   ├── pyproject.toml
│   └── README.md
└── justfile
```

## Usage

Download datasets using the parent justfile:

```bash
# From baselines/ directory
just download-samples   # Content + style images (~5 each)
just download-coco      # COCO val2017 (~5K images, 1GB)
just data-status        # Show what's downloaded
```

Or use uv directly:

```bash
cd shared
uv sync
uv run python download_coco.py --output-dir ../data/datasets/coco --split val2017
```

## Notes

- **COCO val2017**: 5K images, ~1GB download
- **Data directory**: All data lives in `baselines/data/` and is gitignored
- **No data in code directories**: Keep baseline code directories clean
