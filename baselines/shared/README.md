# Shared Resources

This directory contains shared datasets and utilities used across all baselines.

## Directory Structure

```
shared/
├── data/                    # Downloaded datasets
│   ├── coco/               # COCO dataset
│   │   └── val2017/        # Validation images
│   ├── wikiart/            # WikiArt style images
│   └── samples/            # Sample images for quick testing
├── download_coco.py        # COCO download script
└── README.md
```

## Usage

Download datasets using the parent justfile:

```bash
# From baselines/ directory
just download-coco
just download-samples
```

Or use the Python script directly:

```bash
python shared/download_coco.py --split val2017
```

## Notes

- **COCO val2017**: 5K images, ~1GB download
- **WikiArt**: Requires manual download due to licensing
- Data directories are gitignored to avoid bloating the repo

