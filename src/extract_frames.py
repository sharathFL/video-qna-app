#!/usr/bin/env python3
"""
Extract frames from videos at 1 frame per second.
Maps 8 classes to binary SAFE/UNSAFE labels.
Preserves train/test split.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# Dataset root inside container
# Per context specification: /workspace/data/Safe and Unsafe Behaviours Dataset
DATASET_ROOT = Path("/workspace/data/Safe and Unsafe Behaviours Dataset")
# Fallback to actual directory structure if standard path doesn't exist
if not DATASET_ROOT.exists():
    fallback_path = Path("/workspace/data/Video_Dataset _for_Safe_and_Unsafe_Behaviours/Safe_and _Unsafe_Behaviours_Dataset")
    if fallback_path.exists():
        DATASET_ROOT = fallback_path

OUTPUT_ROOT = Path("/workspace/data/frames")

# Label mapping: UNSAFE classes (0-3) -> "unsafe", SAFE classes (4-7) -> "safe"
UNSAFE_CLASSES = {
    "0_safe_walkway_violation": "unsafe",
    "1_unauthorized_intervention": "unsafe",
    "2_opened_panel_cover": "unsafe",
    "3_carrying_overload_with_forklift": "unsafe",
}

SAFE_CLASSES = {
    "4_safe_walkway": "safe",
    "5_authorized_intervention": "safe",
    "6_closed_panel_cover": "safe",
    "7_safe_carrying": "safe",
}

CLASS_TO_LABEL = {**{k: "unsafe" for k in UNSAFE_CLASSES}, **{k: "safe" for k in SAFE_CLASSES}}


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
    
    frame_interval = int(fps)  # Extract 1 frame per second
    frame_count = 0
    saved_count = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame every second
        if frame_count % frame_interval == 0:
            frame_filename = f"{frame_count // frame_interval:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def process_split(split_name):
    """Process train or test split."""
    split_dir = DATASET_ROOT / split_name
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return
    
    print(f"\nProcessing {split_name} split...")
    
    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        label = CLASS_TO_LABEL.get(class_name)
        
        if label is None:
            print(f"Warning: Unknown class {class_name}, skipping")
            continue
        
        print(f"  Processing {class_name} -> {label}")
        
        video_files = list(class_folder.glob("*.mp4"))
        total_frames = 0
        
        for video_path in tqdm(video_files, desc=f"    {class_name}"):
            video_id = video_path.stem
            output_dir = OUTPUT_ROOT / split_name / label / video_id
            
            frames_saved = extract_frames_from_video(video_path, output_dir, video_id)
            total_frames += frames_saved
        
        print(f"    Total frames extracted: {total_frames}")


def main():
    """Main extraction pipeline."""
    print("Starting frame extraction...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Process train and test splits
    process_split("train")
    process_split("test")
    
    print("\nFrame extraction complete!")
    print(f"Frames saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
