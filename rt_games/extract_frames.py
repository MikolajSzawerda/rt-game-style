import argparse
from dataclasses import dataclass, field
from pathlib import Path
import os
import cv2
from tqdm import tqdm


@dataclass
class Args:
    video_path: Path
    output_dir: Path
    frames_to_extract: list[int] = field(default_factory=list)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Extract frames from video")

    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "-f",
        "--frames",
        nargs="+",
        type=int,
        required=True,
        help="List of frame numbers to extract (separated by spaces)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default="extracted_frames",
        help="Output folder path (default: extracted_frames)",
    )

    a = p.parse_args()

    return Args(video_path=a.video, frames_to_extract=a.frames, output_dir=a.output)


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frames = [f for f in args.frames_to_extract if 0 <= f < total_frames]

    for frame_no in tqdm(valid_frames, desc="Extracting Frames", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()

        if success:
            file_name = f"frame_{frame_no}.jpg"
            save_path = os.path.join(args.output_dir, file_name)
            cv2.imwrite(save_path, frame)
        else:
            tqdm.write(f"Error: Could not read frame {frame_no}")

    cap.release()

    print(f"Done! {len(valid_frames)} frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
