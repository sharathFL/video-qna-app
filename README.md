# Safe/Unsafe Workplace Behavior VLM Demo

GPU-accelerated VLM fine-tuning demo for classifying CCTV footage into SAFE/UNSAFE workplace handling behavior.

## Overview

This demo fine-tunes Qwen2-VL-7B-Instruct using QLoRA (4-bit) to perform binary classification on workplace safety videos.

## Prerequisites

- Docker with NVIDIA GPU support (tested with A1000)
- NVIDIA Container Toolkit installed
- Dataset mounted at `./data/Safe and Unsafe Behaviours Dataset/`

## Quick Start

### 1. Build and Start Containers

```bash
docker compose up -d
```

**Port map (bookmark this):** Open **http://localhost:8080/** in your browser. You get a single page listing what runs where:

| Port | Service |
|------|--------|
| **8080** | Port map (this page) |
| **8082** | Video inference (CCTV test videos) |
| **8083** | YouTube inference (paste a YouTube URL) |

If **8083** doesn’t load, ensure the YouTube service is up: `docker compose ps` and check that `vlm_youtube` is running.

To shell into the main container:
```bash
docker exec -it vlm_safe_unsafe bash
```

### 2. Extract Frames (Step 1)

Inside the container:

```bash
cd /workspace/src
python extract_frames.py
```

This extracts 1 frame per second from all videos and organizes them into:
- `frames/train/safe/` and `frames/train/unsafe/`
- `frames/test/safe/` and `frames/test/unsafe/`

### 3. Build Training JSONL (Step 2)

```bash
python build_jsonl.py
```

Creates `train.jsonl` and `test.jsonl` with training samples.

### 4. Train Model (Step 3)

```bash
python train_vlm.py
```

This will:
- Load Qwen2-VL-7B-Instruct with 4-bit quantization
- Apply QLoRA fine-tuning
- Train for 2 epochs
- Save model to `/workspace/outputs/`

**Note:** First run will download the model (~14GB), which may take time.

**Live dashboard:** When you run training (e.g. inside the container), the training dashboard uses port 8080 for binary or 8081 for multiclass. If you use `docker compose up` for inference only, port 8080 serves the **port map** (see above). Ensure ports are exposed in `docker-compose.yml`.

### 5. Run Inference

With `docker compose up`, inference is already running:
- **http://localhost:8080/** — Port map (links to 8082 and 8083)
- **http://localhost:8082/** — Video inference (CCTV test videos; max-vote per video)
- **http://localhost:8083/** — YouTube inference (paste a URL; no download)

To run frame inference manually (single image upload, port 8081):
```bash
python serve_inference.py --checkpoint /workspace/outputs/checkpoint-990
# Open http://localhost:8081/
```

Single image from CLI:
```bash
python infer.py /workspace/data/frames/test/safe/video_001/0001.jpg
```
Outputs: `SAFE` or `UNSAFE`

### Smaller VLMs (SmolVLM2, LLaVA 7B)

For **less VRAM** and **faster training**, use a smaller VLM with the same data:

**SmolVLM2 (2.2B, 500M, or 256M):**
```bash
# To see progress in the browser, redirect output to train.log then open http://localhost:8080/
python -u train_smolvlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --dry-run >> /workspace/outputs/train.log 2>&1
# Full training (same: use >> train.log 2>&1 to see live log in browser)
python -u train_smolvlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct >> /workspace/outputs/train.log 2>&1
# With plots + sample predictions every N steps (browser shows loss curve and input frames + predictions)
python -u train_smolvlm.py --plot-every 50 --pred-every-n-frame 500 >> /workspace/outputs/train.log 2>&1
```
Models: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`. Requires `num2words` (`pip install num2words`).  
`--plot-every N`: update loss plot and predictions in the dashboard every N steps. `--pred-every-n-frame N`: show predictions for every N-th frame (0, N, 2N, …).

**LLaVA 1.5 7B:**
```bash
python train_llava.py --dry-run   # test
python train_llava.py             # full training
```

### Multiclass training (8 classes)

To train on **all 8 behavior classes** (instead of binary safe/unsafe), use the multiclass pipeline. All intermediate data, models, logs, and plots are kept **separate** from the binary setup:

| Binary (safe/unsafe)     | Multiclass (8 classes)           |
|--------------------------|-----------------------------------|
| `data/frames/`           | `data/frames_multiclass/`        |
| `data/train.jsonl`       | `data/train_multiclass.jsonl`    |
| `data/test.jsonl`        | `data/test_multiclass.jsonl`     |
| `outputs/`               | `outputs_multiclass/`             |
| Dashboard port 8080      | Dashboard port **8081**          |

**Steps (inside container):**
```bash
cd /workspace/src
# 1. Extract frames keeping 8 class folders (no mapping to safe/unsafe)
python extract_frames_multiclass.py
# 2. Build JSONL with class names as labels
python build_jsonl_multiclass.py
# 3. Train; dashboard at http://localhost:8081/
python -u train_smolvlm_multiclass.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct >> /workspace/outputs_multiclass/train.log 2>&1
```

Mount `outputs_multiclass` if you run training in Docker: add `- ./outputs_multiclass:/workspace/outputs_multiclass:rw` to your run or compose volumes.

## Dataset Structure

Expected dataset structure:
```
data/
└── Safe and Unsafe Behaviours Dataset/
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

## Label Mapping

**UNSAFE:**
- 0_safe_walkway_violation
- 1_unauthorized_intervention
- 2_opened_panel_cover
- 3_carrying_overload_with_forklift

**SAFE:**
- 4_safe_walkway
- 5_authorized_intervention
- 6_closed_panel_cover
- 7_safe_carrying

## Outputs

- **Binary (safe/unsafe):** Frames `data/frames/`, JSONL `data/train.jsonl` / `data/test.jsonl`, model `outputs/`
- **Multiclass (8 classes):** Frames `data/frames_multiclass/`, JSONL `data/train_multiclass.jsonl` / `data/test_multiclass.jsonl`, model `outputs_multiclass/`
- **Model cache:** `/workspace/.cache/huggingface/`

## Troubleshooting

### GPU not detected
```bash
nvidia-smi  # Should show GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### Out of memory
- Reduce `BATCH_SIZE` in `train_vlm.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`

### Model download issues
- Check internet connection
- Verify HuggingFace cache directory is writable
- Model will be cached at `/workspace/.cache/huggingface/`

## Notes

- **Binary** (default): SAFE/UNSAFE via `extract_frames.py` → `build_jsonl.py` → `train_smolvlm.py` → `outputs/`
- **Multiclass**: 8 classes via `extract_frames_multiclass.py` → `build_jsonl_multiclass.py` → `train_smolvlm_multiclass.py` → `outputs_multiclass/`
- No explanations or reason codes
- Single-frame inference (no temporal aggregation)
- Demo quality (2-3 epochs)
