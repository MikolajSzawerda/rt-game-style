# rt-games justfile - Style Transfer Evaluation Toolkit
# Run `just --list` to see available recipes

set dotenv-load

# Default recipe - show help
default:
    @just --list

# =============================================================================
# BASELINES - Style transfer model management
# =============================================================================

# List all available baselines
baselines := "gbgst"

# Run a specific baseline's justfile recipe
[group('baselines')]
baseline model *args:
    cd baselines/{{model}} && just {{args}}

# Setup a specific baseline (install deps, download weights)
[group('baselines')]
setup model:
    @echo "=== Setting up {{model}} ==="
    just baseline {{model}} setup

# Setup all baselines
[group('baselines')]
setup-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in {{baselines}}; do
        echo "=== Setting up $model ==="
        just baseline $model setup
    done

# Generate stylized outputs from a specific baseline
[group('baselines')]
generate model:
    @echo "=== Generating with {{model}} ==="
    just baseline {{model}} generate

# Generate stylized outputs from all baselines
[group('baselines')]
generate-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in {{baselines}}; do
        echo "=== Generating with $model ==="
        just baseline $model generate
    done

# =============================================================================
# SHARED DATA - Download datasets
# =============================================================================

# Download COCO validation set for evaluation
[group('data')]
download-coco:
    cd baselines && just download-coco

# Download WikiArt style images
[group('data')]
download-wikiart:
    cd baselines && just download-wikiart

# =============================================================================
# EVALUATION - Run metrics on stylized outputs
# =============================================================================

# Evaluate all methods in evaluation/methods/
[group('eval')]
eval-all:
    uv run python -m rt_games.cli \
        --content evaluation/content \
        --style evaluation/style \
        --methods-dir evaluation/methods \
        --output results/all_methods.csv

# Evaluate a specific method
[group('eval')]
eval method:
    uv run python -m rt_games.cli \
        --content evaluation/content \
        --style evaluation/style \
        --stylized evaluation/methods/{{method}} \
        --output results/{{method}}.csv

# Evaluate with specific metrics only
[group('eval')]
eval-metrics method *metrics:
    uv run python -m rt_games.cli \
        --content evaluation/content \
        --style evaluation/style \
        --stylized evaluation/methods/{{method}} \
        --metrics {{metrics}} \
        --output results/{{method}}.csv

# Temporal evaluation for video/game sequences
[group('eval')]
eval-temporal scene:
    uv run python -m rt_games.cli \
        --mode temporal \
        --original evaluation/scenes/{{scene}}/original \
        --stylized evaluation/scenes/{{scene}}/stylized \
        --metrics warping_error,temporal_lpips,depth_error \
        --output results/temporal_{{scene}}.csv

# =============================================================================
# DEVELOPMENT
# =============================================================================

# Run tests
[group('dev')]
test:
    uv run pytest tests/ -v

# Run tests with coverage
[group('dev')]
test-cov:
    uv run pytest tests/ --cov=rt_games --cov-report=term-missing

# Lint code
[group('dev')]
lint:
    uv run ruff check rt_games/

# Format code
[group('dev')]
fmt:
    uv run ruff format rt_games/

# =============================================================================
# FULL PIPELINE - End-to-end workflow
# =============================================================================

# Full pipeline: setup baseline, generate, evaluate
[group('pipeline')]
pipeline model:
    @echo "=== Full pipeline for {{model}} ==="
    just setup {{model}}
    just generate {{model}}
    just eval {{model}}

# Run full pipeline for all baselines
[group('pipeline')]
pipeline-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in {{baselines}}; do
        echo "=== Full pipeline for $model ==="
        just pipeline $model
    done

