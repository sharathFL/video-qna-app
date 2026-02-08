#!/usr/bin/env python3
"""
Extract frames from videos at 1 frame per second.
Multiclass: preserves all 8 class folders (no mapping to safe/unsafe).
Output: data/frames_multiclass/ so binary data/frames/ stays separate.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# Dataset root (same as binary extract_frames)
DATASET_ROOT = Path("/workspace/data/Safe and Unsafe Behaviours Dataset")
if not DATASET_ROOT.exists():
    fallback_path = Path("/workspace/data/Video_Dataset _for_Safe_and_Unsafe_Behaviours/Safe_and _Unsafe_Behaviours_Dataset")
    if fallback_path.exists():
        DATASET_ROOT = fallback_path

# Multiclass: separate output so binary data/frames/ is untouched
OUTPUT_ROOT = Path("/workspace/data/frames_multiclass")


def extract_frames_from_video(video_path, output_dir, video_id):
    """Extract 1 frame per second from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Invalid FPS for {video_path}")
        cap.release()
        return 0

    frame_interval = int(fps)
    frame_count = 0
    saved_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{frame_count // frame_interval:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def process_split(split_name):
    """Process train or test split; keep each class folder as-is (8 classes)."""
    split_dir = DATASET_ROOT / split_name
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return

    print(f"\nProcessing {split_name} split (multiclass)...")

    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        print(f"  Processing {class_name}")

        video_files = list(class_folder.glob("*.mp4"))
        total_frames = 0

        for video_path in tqdm(video_files, desc=f"    {class_name}"):
            video_id = video_path.stem
            output_dir = OUTPUT_ROOT / split_name / class_name / video_id

            frames_saved = extract_frames_from_video(video_path, output_dir, video_id)
            total_frames += frames_saved

        print(f"    Total frames extracted: {total_frames}")


def main():
    """Main extraction pipeline."""
    print("Starting multiclass frame extraction...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output root: {OUTPUT_ROOT} (separate from binary data/frames/)")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    process_split("train")
    process_split("test")

    print("\nMulticlass frame extraction complete!")
    print(f"Frames saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
