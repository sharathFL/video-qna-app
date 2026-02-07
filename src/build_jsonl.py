#!/usr/bin/env python3
"""
Build JSONL training samples from extracted frames.
Each frame becomes one sample with image path, prompt, and response.
"""

import json
from pathlib import Path
from tqdm import tqdm

FRAMES_ROOT = Path("/workspace/data/frames")
OUTPUT_JSONL_TRAIN = Path("/workspace/data/train.jsonl")
OUTPUT_JSONL_TEST = Path("/workspace/data/test.jsonl")

PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."


def build_jsonl_for_split(split_name, output_path):
    """Build JSONL file for train or test split."""
    split_dir = FRAMES_ROOT / split_name
    
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return
    
    samples = []
    
    # Process safe and unsafe folders
    for label in ["safe", "unsafe"]:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        
        print(f"Processing {split_name}/{label}...")
        
        # Iterate through video folders
        for video_dir in sorted(label_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            
            # Get all frame images
            frame_files = sorted(video_dir.glob("*.jpg"))
            
            for frame_path in frame_files:
                # Use relative path from frames root
                relative_path = frame_path.relative_to(FRAMES_ROOT)
                
                sample = {
                    "image": str(relative_path),
                    "prompt": PROMPT,
                    "response": label.upper()  # "SAFE" or "UNSAFE"
                }
                samples.append(sample)
    
    # Write JSONL file
    print(f"Writing {len(samples)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        for sample in tqdm(samples, desc="Writing samples"):
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Created {output_path} with {len(samples)} samples")


def main():
    """Main JSONL building pipeline."""
    print("Building JSONL training samples...")
    print(f"Frames root: {FRAMES_ROOT}")
    
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")
    
    # Build train and test JSONL files
    build_jsonl_for_split("train", OUTPUT_JSONL_TRAIN)
    build_jsonl_for_split("test", OUTPUT_JSONL_TEST)
    
    print("\nJSONL building complete!")


if __name__ == "__main__":
    main()
