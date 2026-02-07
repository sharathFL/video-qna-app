#!/usr/bin/env python3
"""
Evaluate trained SmolVLM2 (or other VLM) on train / eval / test splits.
Computes accuracy and plots it for all stages.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Paths (match train_smolvlm.py)
FRAMES_ROOT = Path("/workspace/data/frames")
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/outputs")
PLOTS_DIR = Path("/workspace/outputs/plots")

PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."
SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def normalize_pred(raw: str) -> str:
    """Map model output to SAFE or UNSAFE."""
    raw = (raw or "").strip().upper()
    if "UNSAFE" in raw:
        return "UNSAFE"
    if "SAFE" in raw:
        return "SAFE"
    return raw[:20] if raw else ""


def load_samples(jsonl_path: Path) -> List[dict]:
    out = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_inference(
    model,
    processor,
    samples: List[dict],
    frames_root: Path,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device=None,
) -> Tuple[int, int]:
    """Run model on samples. Returns (correct, total)."""
    if device is None:
        device = next(model.parameters()).device
    if max_samples is not None:
        samples = samples[:max_samples]
    correct = 0
    total = len(samples)
    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = samples[i : i + batch_size]
            for sample in batch:
                image_path = frames_root / sample["image"]
                if not image_path.exists():
                    total -= 1
                    continue
                img = Image.open(image_path).convert("RGB")
                gt = (sample.get("response") or "").strip().upper()
                if gt not in ("SAFE", "UNSAFE"):
                    total -= 1
                    continue
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": PROMPT},
                            ],
                        }
                    ]
                    inp = processor.apply_chat_template(
                        [messages],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
                    out = model.generate(**inp, max_new_tokens=10, do_sample=False)
                    gen = processor.batch_decode(
                        out[:, inp["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    pred = normalize_pred(gen[0] if gen else "")
                    if pred == gt:
                        correct += 1
                except Exception:
                    pass
    return correct, total


def plot_accuracy(results: dict[str, float], output_path: Path) -> None:
    """Bar chart of accuracy per stage."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    stages = list(results.keys())
    accs = [results[s] * 100 for s in stages]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, accs, color=colors[: len(stages)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by stage (Safe/Unsafe)")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on train/eval/test and plot accuracy.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory with adapter (adapter_model.safetensors or checkpoint-*)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint dir (e.g. outputs/checkpoint-200). Default: use model-dir or latest checkpoint-*",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=SMOLVLM2_2B,
        help="Base model name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for quick eval)",
    )
    parser.add_argument(
        "--eval-from-train",
        type=float,
        default=0.1,
        help="If no eval.jsonl, use this fraction of train as eval (default 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting train into eval",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Output plot path (default: outputs/plots/accuracy.png)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.checkpoint:
        adapter_path = Path(args.checkpoint)
    else:
        checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
        if checkpoints:
            adapter_path = checkpoints[-1]
            print(f"Using latest checkpoint: {adapter_path}")
        else:
            adapter_path = model_dir
    if not (adapter_path / "adapter_model.safetensors").exists() and not (adapter_path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"No adapter in {adapter_path}")

    print(f"Loading base model {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    device = next(model.parameters()).device
    results = {}
    counts = {}

    eval_samples = []
    # Train
    train_path = DATA_DIR / "train.jsonl"
    if train_path.exists():
        train_samples = load_samples(train_path)
        if args.eval_from_train and args.eval_from_train > 0:
            random.seed(args.seed)
            n_eval = max(1, int(len(train_samples) * args.eval_from_train))
            eval_indices = set(random.sample(range(len(train_samples)), n_eval))
            train_only = [s for i, s in enumerate(train_samples) if i not in eval_indices]
            eval_samples = [train_samples[i] for i in sorted(eval_indices)]
        else:
            train_only = train_samples
        c, t = run_inference(model, processor, train_only, FRAMES_ROOT, max_samples=args.max_samples, device=device)
        results["train"] = c / t if t else 0.0
        counts["train"] = (c, t)
        print(f"Train: {c}/{t} = {results['train']*100:.2f}%")
    else:
        print("No train.jsonl; skipping train.")

    # Eval (dedicated file or split from train)
    eval_path = DATA_DIR / "eval.jsonl"
    if not eval_path.exists():
        eval_path = DATA_DIR / "val.jsonl"
    if eval_path.exists():
        eval_samples = load_samples(eval_path)
    if eval_samples:
        c, t = run_inference(model, processor, eval_samples, FRAMES_ROOT, max_samples=args.max_samples, device=device)
        results["eval"] = c / t if t else 0.0
        counts["eval"] = (c, t)
        print(f"Eval:  {c}/{t} = {results['eval']*100:.2f}%")
    else:
        print("No eval split; skipping eval.")

    # Test
    test_path = DATA_DIR / "test.jsonl"
    if test_path.exists():
        test_samples = load_samples(test_path)
        c, t = run_inference(model, processor, test_samples, FRAMES_ROOT, max_samples=args.max_samples, device=device)
        results["test"] = c / t if t else 0.0
        counts["test"] = (c, t)
        print(f"Test:  {c}/{t} = {results['test']*100:.2f}%")
    else:
        print("No test.jsonl; skipping test.")

    if not results:
        print("No splits evaluated.")
        return

    out_plot = Path(args.out_plot) if args.out_plot else PLOTS_DIR / "accuracy.png"
    plot_accuracy(results, out_plot)

    # Save metrics JSON
    metrics_path = out_plot.parent / "accuracy_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "accuracy": results,
                "counts": {k: {"correct": c, "total": t} for k, (c, t) in counts.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
