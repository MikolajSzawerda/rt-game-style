"""
Default configuration for rt_games evaluation toolkit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalConfig:
    device: str = "cuda"
    image_size: Optional[int] = None  # None -> keep original size
    batch_size: int = 4
    num_workers: int = 2
    fid_samples: Optional[int] = None  # limit for heavy metrics
    sifid_samples: Optional[int] = None
    cfsd_samples: Optional[int] = None
    lpips_net: str = "alex"
    use_art_inception: bool = True
    art_inception_url: str = (
        "https://huggingface.co/matthias-wright/art_inception/resolve/main/"
        "art_inception.pth"
    )
    cache_dir: Path = Path(".cache/rt_games")
    flow_model: str = "raft"
    midas_model: str = "DPT_Hybrid"
    force_cpu_flow: bool = False
    force_cpu_depth: bool = False
    verbose: bool = False
    metrics: Optional[List[str]] = None


DEFAULT_CONFIG = EvalConfig()

