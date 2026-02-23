# Safe/Unsafe Workplace Behavior VLM Demo

GPU-accelerated VLM demo for classifying CCTV footage into SAFE/UNSAFE (and optional 8-class) workplace behavior. Uses Docker with NVIDIA GPU support.

## Prerequisites

- Docker with **NVIDIA GPU** support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Dataset (for inference/training): `./data/Safe and Unsafe Behaviours Dataset/` with `train/` and `test/` subfolders

## Bringing Up Services

Build images (first time or after Dockerfile changes):

```bash
docker compose build
```

### Start all services

```bash
docker compose up -d
```

### Start specific services

```bash
# Port map + video inference (8080, 8082)
docker compose up -d vlm

# YouTube inference (8083)
docker compose up -d youtube

# VLM QA / DINOv3 attention (8087)
docker compose up -d vlm_qa
```

### DINOv3 (vlm_qa) – gated model

The default model on 8087 is **DINOv3** (gated on Hugging Face). To use it:

1. Accept the license: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
2. Create a token at [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Start with your token:

```bash
HF_TOKEN=hf_your_token_here docker compose up -d vlm_qa
```

### Port map

| Port | Service |
|------|--------|
| **8080** | Port hub (links to other services) |
| **8082** | Video inference (CCTV test videos) |
| **8083** | YouTube inference (paste URL) |
| **8087** | VLM QA / DINOv3 attention maps |

Open **http://localhost:8080/** to see the hub and links.

### Useful commands

```bash
# Status
docker compose ps

# Logs
docker compose logs -f vlm_qa

# Restart one service
docker compose restart vlm_qa

# Shell into main VLM container
docker exec -it vlm_safe_unsafe bash
```

## Training

See **[docs/HOW_TO_TRAIN.md](docs/HOW_TO_TRAIN.md)** for:

- Binary (safe/unsafe) and multiclass (8 classes) pipelines
- Frame extraction, JSONL build, and training steps
- SmolVLM2, LLaVA, and Qwen2-VL options
- Dashboard ports and outputs

## Outputs

- **Inference:** Uses `./outputs/` (e.g. `checkpoint-2056` for video/YouTube).
- **Model cache:** `./cache/hf/` (Hugging Face models).
- **Training outputs:** See [docs/HOW_TO_TRAIN.md](docs/HOW_TO_TRAIN.md).
