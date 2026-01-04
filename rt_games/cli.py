import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rt_games.config import DEFAULT_CONFIG
from rt_games.data.io import validate_image_triplets
from rt_games.metrics import composite, depth, perceptual, style, temporal
from rt_games.metrics.style import fid_score, sifid_score
from rt_games.utils.registry import METRICS_REGISTRY


DATASET_METRICS = {"fid", "sifid", "artfid"}
TEMPORAL_METRICS = {"warping_error", "temporal_lpips", "depth_error"}


def parse_args():
    p = argparse.ArgumentParser(description="rt-games metrics CLI")
    p.add_argument("--mode", choices=["image", "temporal"], default="image")
    p.add_argument("--content", type=Path, help="content folder")
    p.add_argument("--style", type=Path, help="style folder")
    p.add_argument("--stylized", type=Path, help="stylized folder for single method")
    p.add_argument("--methods-dir", type=Path, help="root containing multiple method folders")
    p.add_argument("--original", type=Path, help="original frames (temporal)")
    p.add_argument("--flow", type=Path, help="flow folder (optional)")
    p.add_argument("--depth", type=Path, help="depth folder (optional)")
    p.add_argument("--metrics", type=str, help="comma-separated metrics to run")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _run_image_for_method(method_dir: Path, args, cfg) -> Dict[str, float]:
    content_dir = args.content
    style_dir = args.style
    stylized_dir = method_dir
    samples = validate_image_triplets(content_dir, style_dir, stylized_dir)
    metrics = [m.strip().lower() for m in args.metrics.split(",")] if args.metrics else METRICS_REGISTRY.names()
    results: Dict[str, float] = {}

    # per-image averages
    per_image_metrics = [m for m in metrics if m not in DATASET_METRICS]
    for name in per_image_metrics:
        fn = METRICS_REGISTRY.get(name)
        vals = []
        for s in samples:
            kwargs = {"device": args.device, "size": args.image_size}
            if name in ("gram_loss", "histogan"):
                vals.append(fn(s.style, s.stylized, **kwargs))
            elif name in ("cfsd", "lpips", "ssim", "content_loss"):
                vals.append(fn(s.content, s.stylized, **kwargs))
            else:
                vals.append(fn(s.content, s.stylized, **kwargs))
        results[name] = sum(vals) / len(vals)

    # dataset metrics
    if "fid" in metrics:
        results["fid"] = fid_score(style_dir, stylized_dir, device=args.device, use_art_inception=cfg.use_art_inception)
    if "sifid" in metrics:
        results["sifid"] = sifid_score(style_dir, stylized_dir, device=args.device, use_art_inception=cfg.use_art_inception)
    if "artfid" in metrics:
        results["artfid"] = composite.artfid(
            content_dir, style_dir, stylized_dir, device=args.device, size=args.image_size, use_art_inception=cfg.use_art_inception
        )
    return results


def run_image_mode(args, cfg):
    rows = []
    if args.methods_dir:
        for method_dir in sorted(args.methods_dir.iterdir()):
            if method_dir.is_dir():
                res = _run_image_for_method(method_dir, args, cfg)
                res["method"] = method_dir.name
                rows.append(res)
    else:
        res = _run_image_for_method(args.stylized, args, cfg)
        res["method"] = args.stylized.name
        rows.append(res)
    return rows


def run_temporal_mode(args, cfg):
    metrics = [m.strip().lower() for m in args.metrics.split(",")] if args.metrics else list(TEMPORAL_METRICS)
    rows = []
    res: Dict[str, float] = {"method": args.stylized.name if args.stylized else "unknown"}
    for name in metrics:
        if name == "warping_error":
            res[name] = temporal.warping_error(args.original, args.stylized, args.flow, device=args.device, size=args.image_size)
        elif name == "temporal_lpips":
            res[name] = temporal.temporal_lpips(args.original, args.stylized, args.flow, device=args.device, size=args.image_size)
        elif name == "depth_error":
            res[name] = depth.depth_error(args.original, args.stylized, args.depth, device=args.device, size=args.image_size)
    rows.append(res)
    return rows


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.image_size = args.image_size
    cfg.device = args.device
    if args.mode == "image":
        rows = run_image_mode(args, cfg)
    else:
        rows = run_temporal_mode(args, cfg)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()

