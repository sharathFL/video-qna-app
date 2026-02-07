#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen2-VL-7B-Instruct.
Binary classification: SAFE or UNSAFE.
"""

import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import TrainerCallback
import os
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback: use processor directly
    process_vision_info = None

# Paths
FRAMES_ROOT = Path("/workspace/data/frames")
TRAIN_JSONL = Path("/workspace/data/train.jsonl")
OUTPUT_DIR = Path("/workspace/outputs")
PLOTS_DIR = Path("/workspace/outputs/plots")
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME_2B = "Qwen/Qwen2-VL-2B-Instruct"  # smaller model for dry-run-gpu
LIVE_DASHBOARD_PORT = 8080
PLOT_UPDATE_EVERY_N_STEPS = 10
INFERENCE_PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."


def _run_http_server(port, directory, log_file_path=None):
    """Serve directory on port; optionally serve GET /log.txt with tail of log_file_path."""
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            log_path = getattr(self.server, "log_file_path", None)
            path_only = self.path.split("?")[0]
            if path_only == "/log.txt" and log_path:
                log_path = Path(log_path)
                try:
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        tail = "".join(lines[-250:])  # last 250 lines
                    else:
                        tail = "Waiting for training log...\n"
                except Exception as e:
                    tail = f"Log read error: {e}\n"
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(tail.encode("utf-8"))
                return
            return super().do_GET()

        def log_message(self, format, *args):
            pass  # quiet

    class ReuseServer(socketserver.TCPServer):
        allow_reuse_address = True
    with ReuseServer(("0.0.0.0", port), Handler) as httpd:
        httpd.log_file_path = log_file_path
        httpd.serve_forever()


def _start_live_server(port, directory, log_file_path=None):
    import threading
    t = threading.Thread(target=_run_http_server, args=(port, directory, log_file_path), daemon=True)
    t.start()
    return t


class LiveDashboardCallback(TrainerCallback):
    """Live training dashboard: loss plot + sample inputs/outputs every N steps. Served in browser."""
    
    def __init__(self, output_dir, processor, dataset, update_every_n=PLOT_UPDATE_EVERY_N_STEPS, sample_indices=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = processor
        self.dataset = dataset
        self.update_every_n = update_every_n
        self.sample_indices = sample_indices or [0, min(100, len(dataset) - 1), min(500, len(dataset) - 1)]
        self.train_losses = []
        self.log_steps = []
        self.trainer_ref = None
        self.last_step_updated = -1
        self.sample_results = []  # list of {step, gt, pred, image_path}
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Waiting for first loss values...", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            plt.savefig(self.output_dir / "training_loss.png", dpi=150)
            plt.close()
        self._write_index(step=0, message="Training started. Refresh in a few steps.")
        print(f"  (plots update every {self.update_every_n} steps)")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.trainer_ref is None and model is not None:
            self.trainer_ref = model
    
    def _write_index(self, step, message="", sample_results=None):
        try:
            cache_bust = step  # so browser refetches images
            sample_results = sample_results or self.sample_results
            rows = "".join(
                f'<tr><td><img src="{r.get("image_path", "")}?v={cache_bust}" width="120" alt=""/></td><td>{r.get("ground_truth", "")}</td><td>{r.get("prediction", "")}</td><td>Step {r.get("step", 0)}</td></tr>'
                for r in sample_results
            )
            input_grid_img = '<img src="input_grid.png?v=%d" width="600" alt="Input samples" onerror="this.alt=\'Waiting for first update...\'"/>' % cache_bust
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Safe/Unsafe VLM Training</title>
<meta http-equiv="refresh" content="5"/>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
.section {{ margin: 24px 0; }}
img {{ border-radius: 8px; border: 2px solid #333; }}
table {{ border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #444; }}
th {{ background: #16213e; }}
a {{ color: #e94560; }}
</style></head><body>
<h1>Safe/Unsafe VLM — Live Training</h1>
<p>Step: <b>{step}</b> | {message} | <em>Auto-refresh every 5s</em></p>
<div class="section">
<h2>Training loss</h2>
<img src="training_loss.png?v={cache_bust}" width="900" alt="Loss curve" onerror="this.style.display='none'"/>
</div>
<div class="section">
<h2>Sample inputs &amp; outputs</h2>
<table><tr><th>Input frame</th><th>Ground truth</th><th>Model prediction</th><th>Step</th></tr>
{rows if rows else "<tr><td colspan='4'>Run a few steps to see samples.</td></tr>"}
</table>
</div>
<div class="section"><h2>Latest input grid</h2>
{input_grid_img}
</div>
</body></html>"""
            (self.output_dir / "index.html").write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"Dashboard write skipped: {e}")
    
    def _save_loss_plot(self):
        if not self.train_losses or not HAS_MATPLOTLIB:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_steps, self.train_losses, "b-o", markersize=3, linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss (Safe/Unsafe VLM)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_loss.png", dpi=150)
        plt.close()
    
    def _save_input_grid_and_run_inference(self, step):
        """Save sample input images as a grid and run inference for dashboard."""
        model = self.trainer_ref
        if model is None or not HAS_MATPLOTLIB:
            return
        n = len(self.sample_indices)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        results = []
        for i, idx in enumerate(self.sample_indices):
            if idx >= len(self.dataset):
                continue
            sample = self.dataset.samples[idx]
            image_path = self.dataset.frames_root / sample["image"]
            if not image_path.exists():
                continue
            img = Image.open(image_path).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(f"Sample {idx}")
            axes[i].axis("off")
            gt = sample.get("response", "?")
            pred = "—"
            try:
                with torch.no_grad():
                    model.eval()
                    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": INFERENCE_PROMPT}]}]
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images = [c["image"] for m in messages for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "image"]
                    inp = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
                    dev = next(model.parameters()).device
                    inp = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inp.items()}
                    out = model.generate(**inp, max_new_tokens=10, do_sample=False)
                    gen = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred = (gen[0].strip().upper() if gen else "—")
                    if "UNSAFE" in pred:
                        pred = "UNSAFE"
                    elif "SAFE" in pred:
                        pred = "SAFE"
                    else:
                        pred = pred[:20] if pred else "—"
                    model.train()
            except Exception:
                pred = "error"
            results.append({"step": step, "ground_truth": gt, "prediction": pred, "image_path": f"sample_{i}.png"})
            img.save(self.output_dir / f"sample_{i}.png")
        plt.tight_layout()
        plt.savefig(self.output_dir / "input_grid.png", dpi=120)
        plt.close()
        self.sample_results = results
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.log_steps.append(state.global_step)
        step = state.global_step
        if step - self.last_step_updated >= self.update_every_n:
            self.last_step_updated = step
            self._save_loss_plot()
            self._save_input_grid_and_run_inference(step)
            self._write_index(step, message="Updated.", sample_results=self.sample_results)
    
    def on_train_end(self, args, state, control, **kwargs):
        self._save_loss_plot()
        self._write_index(state.global_step, message="Training finished.", sample_results=self.sample_results)
        print(f"\n✓ Dashboard and plots in {self.output_dir}")

# Training config
EPOCHS = 2
BATCH_SIZE = 1  # Reduced for memory efficiency
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size
MAX_LENGTH = 512


class SafetyDataset(Dataset):
    """Dataset for safety classification."""
    
    def __init__(self, jsonl_path, frames_root, processor):
        self.frames_root = frames_root
        self.processor = processor
        self.samples = []
        
        print(f"Loading samples from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.frames_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        prompt = sample["prompt"]
        response = sample["response"]  # "SAFE" or "UNSAFE"
        
        # Format for Qwen2-VL: image + text prompt + response
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
        
        # Extract images from messages for Qwen2-VL
        images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item["image"])
        
        # Apply chat template to get text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Process with processor - images and text separately
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels: mask prompt tokens, keep response tokens
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Find where the assistant response starts
        # In Qwen2-VL format, assistant tokens come after user content
        # We'll mask everything except the response tokens
        # Simple approach: mask all tokens initially, then unmask response portion
        labels.fill_(-100)
        
        # Find assistant response tokens (after "<|im_start|>assistant\n")
        assistant_start = None
        for i, token_id in enumerate(input_ids):
            # This is approximate - actual tokenization may vary
            # We'll use a simpler approach: keep last few tokens as response
            pass
        
        # Simpler approach: keep last tokens as response (assuming response is short)
        # For "SAFE" or "UNSAFE", this should be ~2-5 tokens
        response_length = 5  # Approximate tokens for "SAFE" or "UNSAFE"
        if len(labels) > response_length:
            labels[-response_length:] = input_ids[-response_length:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels
        }


def setup_processor_only():
    """Load only the processor (no model). Use for --dry-run to test data pipeline."""
    print(f"Loading processor only: {MODEL_NAME}")
    processor = Qwen2VLProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    return processor


def setup_model_and_processor(model_name=None):
    """Setup model with QLoRA and processor. model_name: use this instead of MODEL_NAME (e.g. 2B for dry-run)."""
    name = model_name or MODEL_NAME
    print(f"Loading model: {name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Force model onto GPU (RTX A4000); "cuda:0" pins to first visible GPU
    device_map = "cuda:0" if torch.cuda.is_available() else None
    if device_map:
        print(f"Using device_map={device_map} (training on GPU)")
    # Use SDPA to avoid "too many indices for tensor of dimension 1" in eager attention (get_rope_index)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map=device_map or "auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    except AttributeError as e:
        if "set_submodule" in str(e):
            print("Warning: Quantization failed (PyTorch/transformers). Loading in fp16 (needs more GPU memory)...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                name,
                device_map=device_map or "auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        else:
            raise
    
    # Prepare for k-bit training with memory optimization
    # Use gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Skip prepare_model_for_kbit_training for 4-bit models to avoid OOM
    # This function tries to convert params to float32 which causes memory issues
    # With bitsandbytes 4-bit, we can often skip this step
    print("Skipping prepare_model_for_kbit_training to avoid OOM (4-bit models may work without it)")
    # model = prepare_model_for_kbit_training(model)  # Commented out to avoid OOM
    
    # LoRA config - reduced rank for memory efficiency
    lora_config = LoraConfig(
        r=8,  # Reduced from 16 to save memory
        lora_alpha=16,  # Reduced proportionally
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Verify model is on GPU
    try:
        _dev = next(model.parameters()).device
        print(f"Model device: {_dev} (expect cuda:0 for RTX A4000)")
        if _dev.type != "cuda":
            raise RuntimeError(f"Model is on {_dev}; GPU required. Set CUDA_VISIBLE_DEVICES=0 and run with Docker GPU support.")
    except StopIteration:
        pass
    # Load processor (same tokenizer/processor family as model)
    processor = Qwen2VLProcessor.from_pretrained(
        name,
        trust_remote_code=True
    )
    
    return model, processor


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="VLM Safety Classification Training")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip loading the model; only load processor, build dataset, and run a few batches to verify the data pipeline.",
    )
    parser.add_argument(
        "--dry-run-gpu",
        action="store_true",
        help="Load 2B model and run 2 training steps on GPU to verify full pipeline (no full 7B load).",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use DeepSpeed ZeRO-2 with optimizer offload to CPU to reduce GPU memory. Run with: accelerate launch src/train_vlm.py --cpu-offload ...",
    )
    args_cli = parser.parse_args()

    import os
    # Limit main-process CPU threads so UI stays responsive; workers use multiple cores
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(2)

    print("=" * 60)
    print("VLM Safety Classification Training")
    print("=" * 60)

    dry_run_gpu = getattr(args_cli, "dry_run_gpu", False)
    if dry_run_gpu:
        print("\n*** DRY RUN GPU: 2B model + 2 steps on GPU ***\n")

    if args_cli.dry_run:
        print("\n*** DRY RUN: testing data pipeline only (no model load) ***\n")
        processor = setup_processor_only()
        train_dataset = SafetyDataset(TRAIN_JSONL, FRAMES_ROOT, processor)
        _pad_id = getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", 0)

        def data_collator(features):
            batch = {}
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
            batch["labels"] = torch.stack([f["labels"] for f in features])
            batch["attention_mask"] = (batch["input_ids"] != _pad_id).long()
            if features[0].get("pixel_values") is not None:
                batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features], dim=0)
            if features[0].get("image_grid_thw") is not None:
                thw_list = [f["image_grid_thw"] for f in features]
                batch["image_grid_thw"] = torch.cat(thw_list, dim=0)
            return batch

        from torch.utils.data import DataLoader
        loader = DataLoader(train_dataset, batch_size=min(2, len(train_dataset)), collate_fn=data_collator, num_workers=0)
        for i, batch in enumerate(loader):
            pv = batch.get("pixel_values")
            pv_str = pv.shape if pv is not None else "N/A"
            print(f"  Batch {i + 1}: input_ids {batch['input_ids'].shape}, labels {batch['labels'].shape}, "
                  f"pixel_values {pv_str}")
            if i >= 2:
                break
        print("\n✓ Dry run OK: data pipeline works. Run without --dry-run to train with the model.")
        return

    # Start live dashboard (serves plots + GET /log.txt for model loading progress)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_log_path = OUTPUT_DIR / "train.log"
    _start_live_server(LIVE_DASHBOARD_PORT, PLOTS_DIR, log_file_path=str(train_log_path))
    (PLOTS_DIR / "index.html").write_text("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Safe/Unsafe VLM Training</title>
<meta http-equiv="refresh" content="60"/>
<style>
body{font-family:sans-serif;margin:20px;background:#1a1a2e;color:#eee;}
h1{color:#e94560;}
h2{color:#a0a0c0;font-size:1rem;}
.log{background:#0f0f1a;padding:14px;border-radius:8px;overflow:auto;max-height:70vh;font-family:monospace;font-size:13px;white-space:pre-wrap;border:1px solid #333;}
</style></head>
<body>
<h1>Safe/Unsafe VLM — Live Training</h1>
<p>Status: <b>Loading model...</b></p>
<h2>Model loading progress (updates every 2s)</h2>
<div class="log" id="log">Waiting for log...</div>
<script>
function refreshLog(){
  fetch('/log.txt?t='+Date.now()).then(r=>r.text()).then(t=>{
    var el=document.getElementById('log');
    el.textContent=(t&&t.trim())?t:'No output yet. Start training with output redirected to train.log.';
    el.scrollTop=el.scrollHeight;
  }).catch(function(){
    document.getElementById('log').textContent='Could not load log. Is the server running?';
  });
}
refreshLog();
setInterval(refreshLog, 2000);
</script>
</body></html>""", encoding="utf-8")
    print(f"\n✓ Live dashboard: http://localhost:{LIVE_DASHBOARD_PORT}/")
    
    # Set memory optimization and ensure GPU is used
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Require GPU (RTX A4000)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available! Training requires GPU. "
            "Run with: docker compose run --gpus all (or ensure NVIDIA_VISIBLE_DEVICES is set)."
        )
    torch.cuda.empty_cache()
    print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Setup model and processor (use 2B for dry-run-gpu so it loads quickly)
    model, processor = setup_model_and_processor(
        model_name=MODEL_NAME_2B if dry_run_gpu else None
    )
    
    # Load datasets
    train_dataset = SafetyDataset(TRAIN_JSONL, FRAMES_ROOT, processor)
    
    # Training arguments: use multiple cores for data loading (keeps main process lighter, UI responsive)
    _n_cpu = os.cpu_count() or 4
    num_workers = 0 if dry_run_gpu else min(8, max(4, _n_cpu - 2))  # leave 2 cores for system/UI
    if num_workers > 0:
        print(f"✓ DataLoader workers: {num_workers} (multi-core data loading; OMP_NUM_THREADS=2 in main process for UI)")
    training_args_dict = dict(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS if not dry_run_gpu else 1,
        max_steps=2 if dry_run_gpu else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
    )
    if num_workers > 1:
        training_args_dict["dataloader_prefetch_factor"] = 2
    if getattr(args_cli, "cpu_offload", False):
        _ds_config = Path(__file__).resolve().parent.parent / "configs" / "ds_zero2_cpu_offload.json"
        if not _ds_config.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {_ds_config}. Install deepspeed and use: accelerate launch src/train_vlm.py --cpu-offload ...")
        training_args_dict["deepspeed"] = str(_ds_config)
        print("✓ CPU offload enabled: DeepSpeed ZeRO-2 (optimizer offload to CPU).")
        print("  Tip: run with 'accelerate launch src/train_vlm.py --cpu-offload ...' so DeepSpeed is used correctly.")
    training_args = TrainingArguments(**training_args_dict)
    
    # Pad token for building a proper 2D attention mask (Qwen2-VL processor can return
    # a non-standard format that causes "too many indices for tensor of dimension 1")
    _pad_id = getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", 0)
    
    # Custom data collator
    def data_collator(features):
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        # 2D attention mask (batch, seq_len): 1 = valid token, 0 = padding
        batch["attention_mask"] = (batch["input_ids"] != _pad_id).long()
        
        # Handle vision inputs
        if features[0].get("pixel_values") is not None:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features], dim=0)
        if features[0].get("image_grid_thw") is not None:
            # Model expects one (N, 3) tensor: for t, h, w in grid_thw iterates rows
            thw_list = [f["image_grid_thw"] for f in features]
            batch["image_grid_thw"] = torch.cat(thw_list, dim=0)
        
        return batch
    
    # Live dashboard callback (loss + sample inputs/outputs every N steps)
    live_callback = LiveDashboardCallback(
        PLOTS_DIR,
        processor=processor,
        dataset=train_dataset,
        update_every_n=PLOT_UPDATE_EVERY_N_STEPS,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[live_callback],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    if dry_run_gpu:
        print("\n✓ Dry run GPU complete: 2 steps finished on GPU. Pipeline is OK. Run without --dry-run-gpu for full training.")
        return
    
    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
