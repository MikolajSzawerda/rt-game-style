import sys
import os
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import cv2
import torch
from tqdm import tqdm
import kornia
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


FRAMEWORK_PATH = Path(__file__).parent / ".." / ".." / "rt_games"
sys.path.insert(0, str(FRAMEWORK_PATH))
from models.midas import load_midas  # noqa: E402

# Add submodule to path
REPO_PATH = Path(__file__).parent / "repo" / "depth-aware-nst"
sys.path.insert(0, str(REPO_PATH))

from transformer_net_light import TransformerNetLight  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Args:
    video_path: Path
    weights_path: Path
    output_path: Path
    independent_stylization: bool = False
    no_cuda: bool = False


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="GBST video stylization")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--independent",
        action="store_true",
        help="Skip using previous stylization for frames",
    )
    p.add_argument("--no-cuda", action="store_true", help="Use PCU instead of GPU")

    a = p.parse_args()

    return Args(
        video_path=a.video,
        weights_path=BASE_DIR / a.weights,
        output_path=a.output,
        independent_stylization=a.independent,
        no_cuda=a.no_cuda,
    )


def load_model(weights_path: Path, device: torch.device) -> TransformerNetLight:
    model = TransformerNetLight()
    state = torch.load(weights_path, map_location=device, weights_only=True)

    for k in list(state.keys()):
        if re.search(r"in\d+\.running_(mean|var)$", k):
            del state[k]

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    return model

def load_raft(device: str = "cuda"):
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    
    return model, weights.transforms()


def compute_flow(raft_model, transforms, img1, img2):
    img1_trans, img2_trans = transforms(img1, img2)
    
    with torch.no_grad():
        list_of_flows = raft_model(img1_trans, img2_trans)
        flow_up = list_of_flows[-1]

    flow_up = torch.nn.functional.interpolate(
        flow_up, size=img1.shape[-2:], mode="bilinear", align_corners=False
    )
    return flow_up


def get_depth_tensor(model, transform, frame_rgb_np, device, target_size):
    input_batch = transform(frame_rgb_np).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        depth = F.interpolate(
            prediction.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
    
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    return depth


def get_normals_from_depth(depth_tensor, intrinsics=None):
    B, C, H, W = depth_tensor.shape
    
    if intrinsics is None:
        intrinsics = torch.eye(3, device=depth_tensor.device).unsqueeze(0)
        intrinsics[:, 0, 0] = W
        intrinsics[:, 1, 1] = H
        intrinsics[:, 0, 2] = W / 2
        intrinsics[:, 1, 2] = H / 2

    normals = kornia.geometry.depth.depth_to_normals(depth_tensor, intrinsics)
    
    return normals


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if use_cuda else "")
    )

    model = load_model(args.weights_path, device)
    model.eval()

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        sys.exit(f"Error: Could not open video {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(args.output_path), fourcc, fps, (w, h))

    previous_tensor = None
    previous_stylized = None

    if not args.independent_stylization:
        midas_model, midas_transform = load_midas(device)
        raft_model, flow_transforms = load_raft(device)

    with torch.no_grad():
        for i in tqdm(range(min(total_frames, 30)), desc="Processing Video"):
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
            current_tensor = current_tensor.to(device).unsqueeze(0)

            stylized_tensor = model.process_image(current_tensor)

            if previous_stylized is None or args.independent_stylization:
                output_tensor = stylized_tensor.unsqueeze(0)
            else:
                depth = get_depth_tensor(midas_model, midas_transform, frame_rgb, device, (h, w))
                normals = get_normals_from_depth(depth)
                motion = compute_flow(raft_model, flow_transforms, previous_tensor, current_tensor)

                output_tensor = model.process_image_motion_vectors(current_tensor, stylized_tensor.unsqueeze(0), previous_stylized, motion, normals, depth)

            previous_tensor = current_tensor
            previous_stylized = output_tensor

            out_img = (
                output_tensor.detach()
                .squeeze(0)
                .cpu()
                .float()
                .clamp(0, 255)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )
            out_video.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    cap.release()
    out_video.release()
    print(f"Done! Saved to {args.output_path}")


if __name__ == "__main__":
    main()
