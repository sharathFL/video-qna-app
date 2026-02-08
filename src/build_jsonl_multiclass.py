#!/usr/bin/env python3
"""
Build JSONL training samples from multiclass extracted frames.
Each frame gets one sample with image path, prompt, and response = class name.
Outputs: data/train_multiclass.jsonl, data/test_multiclass.jsonl (separate from binary).
"""

import json
from pathlib import Path
from tqdm import tqdm

# Multiclass paths (separate from binary)
FRAMES_ROOT = Path("/workspace/data/frames_multiclass")
OUTPUT_JSONL_TRAIN = Path("/workspace/data/train_multiclass.jsonl")
OUTPUT_JSONL_TEST = Path("/workspace/data/test_multiclass.jsonl")


def get_class_names_from_frames():
    """Discover class folder names from frames_multiclass/train (or test) so prompt matches labels."""
    for split in ("train", "test"):
        split_dir = FRAMES_ROOT / split
        if split_dir.exists():
            names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            if names:
                return names
    return []


def make_prompt(class_names):
    return (
        "You are a workplace safety inspector reviewing CCTV footage. "
        "Classify the behavior into exactly one of: "
        + ", ".join(class_names)
        + ". Answer with only that exact class name."
    )


def build_jsonl_for_split(split_name, output_path, prompt):
    """Build JSONL for train or test from frames_multiclass/<split>/<class_name>/<video_id>/*.jpg."""
    split_dir = FRAMES_ROOT / split_name

    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return

    samples = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            for frame_path in sorted(video_dir.glob("*.jpg")):
                relative_path = frame_path.relative_to(FRAMES_ROOT)
                sample = {
                    "image": str(relative_path),
                    "prompt": prompt,
                    "response": class_name,
                }
                samples.append(sample)

    print(f"Writing {len(samples)} samples to {output_path}...")
    with open(output_path, "w") as f:
        for sample in tqdm(samples, desc="Writing samples"):
            f.write(json.dumps(sample) + "\n")

    print(f"✓ Created {output_path} with {len(samples)} samples")


def main():
    """Main JSONL building pipeline."""
    print("Building multiclass JSONL training samples...")
    print(f"Frames root: {FRAMES_ROOT}")
    print(f"Outputs: {OUTPUT_JSONL_TRAIN}, {OUTPUT_JSONL_TEST} (separate from binary train/test.jsonl)")

    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(
            f"Frames root not found: {FRAMES_ROOT}. Run extract_frames_multiclass.py first."
        )

    class_names = get_class_names_from_frames()
    if not class_names:
        raise FileNotFoundError(f"No class folders found under {FRAMES_ROOT}/train or test")
    prompt = make_prompt(class_names)
    print(f"Classes ({len(class_names)}): {class_names}")

    build_jsonl_for_split("train", OUTPUT_JSONL_TRAIN, prompt)
    build_jsonl_for_split("test", OUTPUT_JSONL_TEST, prompt)

    print("\nMulticlass JSONL building complete!")


if __name__ == "__main__":
    main()
