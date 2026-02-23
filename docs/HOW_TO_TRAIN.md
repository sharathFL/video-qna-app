# How to Train

This guide covers training the VLM for **binary** (safe/unsafe) or **multiclass** (8 behavior classes) classification on the Safe and Unsafe Behaviours Dataset.

## Prerequisites

- Dataset at `./data/Safe and Unsafe Behaviours Dataset/` with:
  - `train/` and `test/` each containing the class folders (see [Dataset structure](#dataset-structure) below)
- Docker with NVIDIA GPU (or a Python env with the same dependencies)
- For training inside Docker: start a container with data and outputs mounted, or use the same volumes as in `docker-compose.yml`

## Dataset structure

```
data/Safe and Unsafe Behaviours Dataset/
├── train/
│   ├── 0_safe_walkway_violation/
│   ├── 1_unauthorized_intervention/
│   ├── 2_opened_panel_cover/
│   ├── 3_carrying_overload_with_forklift/
│   ├── 4_safe_walkway/
│   ├── 5_authorized_intervention/
│   ├── 6_closed_panel_cover/
│   └── 7_safe_carrying/
└── test/
    └── (same structure)
```

**Binary mapping:** UNSAFE = 0–3, SAFE = 4–7 (used by binary pipeline).

---

## Binary training (safe/unsafe)

Outputs go to `data/frames/`, `data/train.jsonl`, `data/test.jsonl`, and `outputs/`. Training dashboard: **http://localhost:8080/** (when running inside the main container).

### 1. Extract frames

One frame per second from all videos, organized into `frames/train/safe`, `frames/train/unsafe`, and same under `frames/test/`.

```bash
cd /workspace/src
python extract_frames.py
```

### 2. Build JSONL

Creates `data/train.jsonl` and `data/test.jsonl` with image paths and SAFE/UNSAFE labels.

```bash
python build_jsonl.py
```

### 3. Train

**SmolVLM2 (recommended for less VRAM, faster runs):**

```bash
# Dry run (no training, sanity check)
python -u train_smolvlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --dry-run

# Full training; redirect to see live log in dashboard
python -u train_smolvlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct >> /workspace/outputs/train.log 2>&1
```

Optional flags:

- `--plot-every N` — update loss plot and sample predictions in the dashboard every N steps
- `--pred-every-n-frame N` — log predictions for every N-th frame (0, N, 2N, …)

Other SmolVLM2 models: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`. Requires `pip install num2words`.

**LLaVA 1.5 7B:**

```bash
python train_llava.py --dry-run   # test
python train_llava.py             # full training
```

**Qwen2-VL 7B (QLoRA, more VRAM):**

```bash
python train_vlm.py
```

First run downloads the model (~14GB) into the Hugging Face cache. Checkpoints are written to `/workspace/outputs/` (e.g. `checkpoint-2056`). Use that path for video/YouTube inference.

---

## Multiclass training (8 classes)

Uses separate dirs and outputs so it does not overwrite binary data or checkpoints.

| Binary                 | Multiclass                |
|------------------------|---------------------------|
| `data/frames/`         | `data/frames_multiclass/` |
| `data/train.jsonl`     | `data/train_multiclass.jsonl` |
| `data/test.jsonl`      | `data/test_multiclass.jsonl`  |
| `outputs/`             | `outputs_multiclass/`     |
| Dashboard 8080         | Dashboard **8081**        |

If you run in Docker, mount multiclass outputs, e.g. add to your run:

`- ./outputs_multiclass:/workspace/outputs_multiclass:rw`

### 1. Extract frames (8 class folders)

No mapping to safe/unsafe; class folder names are kept as labels.

```bash
cd /workspace/src
python extract_frames_multiclass.py
```

### 2. Build JSONL

Creates `data/train_multiclass.jsonl` and `data/test_multiclass.jsonl` with class names as labels.

```bash
python build_jsonl_multiclass.py
```

### 3. Train

```bash
python -u train_smolvlm_multiclass.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct >> /workspace/outputs_multiclass/train.log 2>&1
```

Dashboard: **http://localhost:8081/** (multiclass uses port 8081).

---

## Outputs summary

- **Binary:** Frames in `data/frames/`, JSONL in `data/`, checkpoints in `outputs/`
- **Multiclass:** Frames in `data/frames_multiclass/`, JSONL in `data/*_multiclass.jsonl`, checkpoints in `outputs_multiclass/`
- **Model cache:** `./cache/hf/` (or `/workspace/.cache/huggingface` in container)

---

## Troubleshooting

- **Out of memory:** Reduce batch size / increase gradient accumulation in the training script.
- **Model download:** Ensure network access and that the Hugging Face cache dir is writable; for gated models, set `HF_TOKEN` or log in with `huggingface-cli login`.
- **Dashboard not updating:** Training must be running in a container that exposes the dashboard port (8080 binary, 8081 multiclass); redirecting stdout to `train.log` allows the dashboard to tail the log.
