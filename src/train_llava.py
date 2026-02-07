#!/usr/bin/env python3
"""
Train LLaVA 1.5 7B for SAFE/UNSAFE classification.
Uses same data as train_vlm.py. Alternative to Qwen2-VL-7B.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

FRAMES_ROOT = Path("/workspace/data/frames")
TRAIN_JSONL = Path("/workspace/data/train.jsonl")
OUTPUT_DIR = Path("/workspace/outputs")
PLOTS_DIR = Path("/workspace/outputs/plots")
LIVE_DASHBOARD_PORT = 8080

# LLaVA 1.5 7B (HF format)
LLAVA_7B = "llava-hf/llava-1.5-7b-hf"

PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."
EPOCHS = 2
BATCH_SIZE = 1
LR = 2e-5
GRAD_ACCUM = 8
RESPONSE_LABEL_TOKENS = 8


def _run_http_server(port, directory, log_file_path=None):
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
                    tail = "".join(open(log_path, "r", encoding="utf-8", errors="replace").readlines()[-250:]) if log_path.exists() else "Waiting for log...\n"
                except Exception as e:
                    tail = f"Log error: {e}\n"
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(tail.encode("utf-8"))
                return
            return super().do_GET()
        def log_message(self, format, *args):
            pass
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


class SafetyDatasetLLaVA(Dataset):
    def __init__(self, jsonl_path, frames_root, processor):
        self.frames_root = Path(frames_root)
        self.processor = processor
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.frames_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        prompt = sample["prompt"]
        response = sample["response"]
        # LLaVA 1.5 format: list of messages with content = text or image
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        inputs = self.processor.apply_chat_template(
            [messages],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels.fill_(-100)
        if len(labels) > RESPONSE_LABEL_TOKENS:
            labels[-RESPONSE_LABEL_TOKENS:] = input_ids[-RESPONSE_LABEL_TOKENS:]
        out = {"input_ids": input_ids, "labels": labels}
        if "attention_mask" in inputs:
            out["attention_mask"] = inputs["attention_mask"].squeeze(0)
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"].squeeze(0)
        return out


def collate_fn(features):
    batch = {}
    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    if features[0].get("attention_mask") is not None:
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
    if features[0].get("pixel_values") is not None:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_log_path = OUTPUT_DIR / "train.log"
    _start_live_server(LIVE_DASHBOARD_PORT, PLOTS_DIR, log_file_path=str(train_log_path))
    (PLOTS_DIR / "index.html").write_text("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LLaVA 7B Training</title><meta http-equiv="refresh" content="60"/>
<style>body{font-family:sans-serif;margin:20px;background:#1a1a2e;color:#eee;} h1{color:#e94560;} .log{background:#0f0f1a;padding:14px;border-radius:8px;overflow:auto;max-height:80vh;font-family:monospace;font-size:13px;white-space:pre-wrap;border:1px solid #333;}</style></head>
<body><h1>LLaVA 1.5 7B — Live Training</h1><p>Status: <b>Loading / training...</b></p><h2>Progress (updates every 2s)</h2><div class="log" id="log">Waiting for log...</div>
<script>function r(){ fetch('/log.txt?t='+Date.now()).then(x=>x.text()).then(t=>{ var e=document.getElementById('log'); e.textContent=(t&&t.trim())?t:'No output yet.'; e.scrollTop=e.scrollHeight; }).catch(()=>{ document.getElementById('log').textContent='Could not load log.'; }); } r(); setInterval(r, 2000);</script></body></html>""", encoding="utf-8")
    print(f"Live dashboard: http://localhost:{LIVE_DASHBOARD_PORT}/")
    sys.stdout.flush(); sys.stderr.flush()

    print(f"Loading {LLAVA_7B}...")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(LLAVA_7B)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_7B,
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else None,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    sys.stdout.flush(); sys.stderr.flush()
    print("Building dataset...")
    sys.stdout.flush()
    dataset = SafetyDatasetLLaVA(TRAIN_JSONL, FRAMES_ROOT, processor)
    sys.stdout.flush()
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=200,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0 if args.dry_run else 4,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate_fn)
    print("Starting training...")
    sys.stdout.flush()
    trainer.train()
    if not args.dry_run:
        trainer.save_model()
        processor.save_pretrained(OUTPUT_DIR)
    print("Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
