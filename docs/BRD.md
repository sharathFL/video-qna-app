# Business Requirements Document (BRD)  
## Safe/Unsafe Workplace Behavior VLM Demo

**Version:** 1.0  
**Date:** February 16, 2025  
**Status:** Draft

---

## 1. Executive Summary

This document defines the business requirements for the **Safe/Unsafe Workplace Behavior VLM Demo**—a GPU-accelerated, vision-language model (VLM) system that classifies CCTV footage into **SAFE** or **UNSAFE** workplace handling behavior. The system supports fine-tuning of VLMs (Qwen2-VL-7B, SmolVLM2, LLaVA 1.5) using QLoRA, inference on local test videos and YouTube URLs, and an optional **multiclass** mode that classifies into eight distinct behavior categories. The solution is delivered as a containerized demo suitable for evaluation, prototyping, and integration into broader safety monitoring workflows.

---

## 2. Business Objectives

| Objective | Description |
|-----------|-------------|
| **Automate safety review** | Reduce manual review of CCTV footage by automatically flagging safe vs. unsafe workplace behavior. |
| **Demonstrate VLM capability** | Prove that fine-tuned VLMs can perform binary (and multiclass) behavior classification from video with acceptable accuracy for demo/prototype use. |
| **Support multiple inference entry points** | Allow users to run inference on (1) local CCTV test videos, (2) single image uploads, and (3) YouTube URLs without downloading video. |
| **Enable flexible training** | Support binary (SAFE/UNSAFE) and multiclass (8 classes) training pipelines with separate data, outputs, and dashboards to avoid interference. |
| **Lower barrier to experimentation** | Provide smaller VLM options (SmolVLM2 2.2B/500M/256M) for lower VRAM and faster iteration, alongside larger models (Qwen2-VL-7B, LLaVA 7B). |

---

## 3. Scope

### 3.1 In Scope

- **Data pipeline:** Frame extraction from dataset videos (1 fps), organization into train/test and safe/unsafe (or 8-class) folders, and JSONL generation for training.
- **Training:** Fine-tuning VLMs with QLoRA (4-bit) on the Safe and Unsafe Behaviours Dataset; support for binary and multiclass labels; live training dashboard (loss, sample predictions, plots).
- **Inference services:**
  - **Video inference (port 8082):** List and run inference on test videos from the dataset; per-video result via max-vote over frames; frame-wise SAFE/UNSAFE distribution.
  - **YouTube inference (port 8083):** Paste YouTube URL; stream and sample frames in memory; no persistent download; return prediction and frame distribution.
  - **Single-image inference (port 8081):** Upload one image or use CLI for one image; output SAFE or UNSAFE.
- **Port map (port 8080):** Single entry page listing all services and links.
- **Deployment:** Docker Compose setup with GPU support; separate compose files for main inference and multiclass training.
- **Outputs:** Trained checkpoints, plots, logs, and cached models; clear separation between binary and multiclass outputs.

### 3.2 Out of Scope

- **Production hardening:** No SLA, high-availability, or formal security review; demo/prototype quality only.
- **Explanations or reason codes:** Model answers with a single label (e.g., SAFE/UNSAFE or class name); no natural-language justification or explanation.
- **Temporal aggregation:** Single-frame classification with max-vote (or similar) per video; no dedicated temporal/video model.
- **Real-time streaming:** Inference is on pre-extracted or streamed frames at fixed sample rate (e.g., 1 fps), not live CCTV streams.
- **User authentication, multi-tenancy, or audit logging:** Not required for this demo.

---

## 4. Stakeholders

| Role | Responsibility |
|------|----------------|
| **Safety / Operations** | Define safe vs. unsafe behaviors; validate label mapping and sample predictions. |
| **Data / ML team** | Run training, tune hyperparameters, evaluate checkpoints, integrate with existing ML pipelines. |
| **DevOps / Platform** | Deploy Docker stack, manage GPU nodes, mount data and cache volumes. |
| **Product / Demo owners** | Use port map and inference UIs to showcase capability to internal or external audiences. |

---

## 5. Functional Requirements

### 5.1 Data Preparation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-D1 | System shall extract frames from dataset videos at a configurable rate (default 1 frame per second). | Must |
| FR-D2 | System shall organize frames into train/test and, for binary mode, safe/unsafe folders based on dataset folder naming (0–3 → UNSAFE, 4–7 → SAFE). | Must |
| FR-D3 | System shall support a multiclass pipeline that preserves all eight class folders (no mapping to safe/unsafe) in a separate directory tree and JSONL. | Must |
| FR-D4 | System shall produce training and test JSONL files compatible with the chosen VLM training script. | Must |

### 5.2 Training

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-T1 | System shall support fine-tuning with QLoRA (4-bit) for at least Qwen2-VL-7B, SmolVLM2 (2.2B/500M/256M), and LLaVA 1.5 7B. | Must |
| FR-T2 | System shall support binary (SAFE/UNSAFE) and multiclass (8 classes) training with separate outputs and dashboard ports. | Must |
| FR-T3 | Training shall expose a live dashboard (e.g., port 8080 for binary, 8081 for multiclass) showing loss, optional plots, and sample predictions. | Should |
| FR-T4 | Training shall save checkpoints and logs to a configurable output directory (e.g., `outputs/` for binary, `outputs_multiclass/` for multiclass). | Must |

### 5.3 Inference

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-I1 | System shall provide a port map page (default port 8080) listing all services and links (video inference, YouTube inference). | Must |
| FR-I2 | Video inference service shall list test videos from the dataset, run per-frame inference, and return a per-video label via max-vote over frames, with frame-wise distribution. | Must |
| FR-I3 | YouTube inference service shall accept a YouTube URL, stream/sample frames in memory (no persistent download), run inference, and return prediction and frame distribution. | Must |
| FR-I4 | Single-image inference shall accept one image (upload or path) and return SAFE or UNSAFE (or multiclass label when using a multiclass model). | Must |
| FR-I5 | Inference services shall use a configurable checkpoint path (e.g., default `checkpoint-2056`). | Must |

### 5.4 Deployment and Operations

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-O1 | System shall run in Docker with NVIDIA GPU support; dataset, source, outputs, and HuggingFace cache shall be mountable volumes. | Must |
| FR-O2 | Docker Compose shall start the port map and video inference service; a separate service shall run YouTube inference. | Must |
| FR-O3 | Optional Docker Compose file(s) shall support multiclass training (and optionally multiclass inference) without conflicting with binary services. | Should |

---

## 6. Non-Functional Requirements

| ID | Requirement | Notes |
|----|-------------|--------|
| NFR-1 | **Hardware:** NVIDIA GPU (e.g., A1000); NVIDIA Container Toolkit. | Documented in README. |
| NFR-2 | **Performance:** Training and inference run on GPU; first run may involve model download (~14GB for Qwen2-VL). | Demo quality; 2–3 epochs typical. |
| NFR-3 | **Usability:** Single entry point (port 8080) to discover and reach all services. | Port map page. |
| NFR-4 | **Maintainability:** Binary and multiclass pipelines use separate directories and ports to avoid interference. | outputs/ vs outputs_multiclass/; 8080 vs 8081/8084. |
| NFR-5 | **Reproducibility:** Data layout, label mapping, and training steps documented (README, dataset structure). | |

---

## 7. User Personas and Use Cases

### 7.1 Personas

- **Safety analyst:** Wants to quickly see whether a video or YouTube clip is classified as safe or unsafe and review frame-level distribution.
- **ML engineer:** Wants to run or re-run training, try different models (SmolVLM2 vs Qwen2-VL), and inspect loss and sample predictions.
- **Demo presenter:** Wants one page (port map) to open video and YouTube inference UIs for live demos.

### 7.2 Use Cases

| UC | Name | Actor | Flow |
|----|------|--------|------|
| UC1 | Run inference on CCTV test video | Safety analyst / Demo | Open port map → Video inference (8082) → Select or upload video → View SAFE/UNSAFE and frame distribution. |
| UC2 | Run inference on YouTube video | Safety analyst / Demo | Open port map → YouTube inference (8083) → Paste URL → View result and frame distribution (no download). |
| UC3 | Train binary model | ML engineer | Extract frames → Build JSONL → Run training (e.g., train_smolvlm.py) → Monitor dashboard → Use checkpoint for inference. |
| UC4 | Train multiclass model | ML engineer | Extract multiclass frames → Build multiclass JSONL → Run multiclass training → Use multiclass checkpoint for 8-class inference. |
| UC5 | Single-image classification | Integrator / Tester | Call serve_inference (8081) or CLI infer.py with image path → Get SAFE/UNSAFE (or class). |

---

## 8. Dataset and Label Mapping

### 8.1 Dataset Structure (Expected)

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

### 8.2 Binary Label Mapping

- **UNSAFE:** 0_safe_walkway_violation, 1_unauthorized_intervention, 2_opened_panel_cover, 3_carrying_overload_with_forklift  
- **SAFE:** 4_safe_walkway, 5_authorized_intervention, 6_closed_panel_cover, 7_safe_carrying  

Multiclass pipeline keeps all eight class names as labels (no binary mapping).

---

## 9. Service and Port Summary

| Port | Service | Description |
|------|---------|-------------|
| 8080 | Port map | Entry page listing all services and links. |
| 8081 | Single-image inference | Upload one image; SAFE/UNSAFE (or multiclass). |
| 8082 | Video inference | CCTV test videos; max-vote per video + frame distribution. |
| 8083 | YouTube inference | Paste YouTube URL; inference without download. |
| 8084 | Multiclass training dashboard | When using docker-compose.multiclass.yml. |

*(Binary training dashboard may use 8080 or 8081 depending on whether inference is running.)*

---

## 10. Assumptions and Constraints

### 10.1 Assumptions

- Dataset is available at the documented path and follows the expected folder structure and naming.
- Users have Docker, NVIDIA GPU, and NVIDIA Container Toolkit installed.
- HuggingFace model access (and any token for gated models) is configured as needed.
- Demo-level accuracy and latency are acceptable; no formal SLA or compliance requirements for this BRD.

### 10.2 Constraints

- Single-frame classification; no built-in temporal reasoning beyond max-vote over frames.
- No user management, authentication, or audit logging.
- Inference and training share GPU when run on same host; resource contention is possible.

---

## 11. Success Criteria

- Users can start the stack with `docker compose up -d` and access the port map and inference UIs without manual port lookup.
- Users can run binary and multiclass training using the documented steps and see loss (and optionally plots and sample predictions) in the dashboard.
- Users can obtain a SAFE/UNSAFE (or 8-class) prediction for a test video, a YouTube URL, or a single image using the provided services or CLI.
- Binary and multiclass pipelines coexist without overwriting each other’s data or checkpoints.

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **VLM** | Vision-Language Model; model that takes image (and optionally text) and produces text (e.g., SAFE/UNSAFE). |
| **QLoRA** | Quantized Low-Rank Adaptation; 4-bit quantized base model with trainable LoRA adapters. |
| **Max-vote** | Per-video label = most frequent per-frame label over sampled frames. |
| **Port map** | Single HTML page that lists services and their ports/URLs. |
| **Binary** | Two-class mode: SAFE vs. UNSAFE. |
| **Multiclass** | Eight-class mode: original dataset class names as labels. |

---

## 13. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-02-16 | — | Initial BRD. |
