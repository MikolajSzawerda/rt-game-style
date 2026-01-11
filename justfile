# rt-games justfile - Style Transfer Evaluation
# 
# Quick start:
#   just download-coco          # Get COCO val2017
#   just download-styles        # Get style images
#   just setup gbgst            # Setup baseline
#   just gen gbgst starry_night 100   # Generate 100 images
#   just eval gbgst_starry_night_100  # Evaluate outputs

set dotenv-load

# =============================================================================
# PATHS - All data in baselines/data/
# =============================================================================

data_dir := "baselines/data"
coco_dir := data_dir / "coco/val2017"
style_dir := data_dir / "style"
weights_dir := data_dir / "weights"
outputs_dir := data_dir / "outputs"

# =============================================================================
# DATA - Download datasets
# =============================================================================

# Download COCO val2017 (~5K images, 1GB)
download-coco:
    cd baselines && just download-coco

# Download style images (starry_night, mosaic, etc.)
download-styles:
    cd baselines && just download-styles

# Show what's downloaded
data-status:
    cd baselines && just data-status

# =============================================================================
# BASELINES - Setup and generation
# =============================================================================

baselines := "gbgst"

# Setup a baseline (install deps, download weights)
setup model:
    cd baselines/{{model}} && just setup

# Generate: method, style, n_images from COCO
# Output goes to: baselines/data/outputs/{method}_{style}_{n}/
# Example: just gen gbgst starry_night 100
gen model style n:
    cd baselines/{{model}} && just gen {{style}} {{n}}

# =============================================================================
# EVALUATION - Run metrics on generated outputs
# =============================================================================

# Evaluate outputs by folder name
# Example: just eval gbgst_starry_night_100
eval output_name:
    uv run python -m rt_games.cli \
        --content {{coco_dir}} \
        --style {{style_dir}} \
        --stylized {{outputs_dir}}/{{output_name}} \
        --output results/{{output_name}}.csv

# Evaluate with specific metrics
eval-metrics output_name metrics:
    uv run python -m rt_games.cli \
        --content {{coco_dir}} \
        --style {{style_dir}} \
        --stylized {{outputs_dir}}/{{output_name}} \
        --metrics {{metrics}} \
        --output results/{{output_name}}.csv

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
    uv run pytest tests/ -v

test-cov:
    uv run pytest tests/ --cov=rt_games --cov-report=term-missing

lint:
    uv run ruff check rt_games/

fmt:
    uv run ruff format rt_games/
