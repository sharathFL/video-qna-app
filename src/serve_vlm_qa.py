#!/usr/bin/env python3
"""
General VLM QA server: ask questions about an image (e.g. "Is the person wearing a hardhat? yes/no").
Supports SmolVLM2 or LLaVA 7B, base models (no fine-tuned adapter). For use with stream feeds
(YouTube, webcam): send frames + questions and build answers from replies.
Serves on port 8087 by default.
"""

import argparse
import base64
import cgi
import io
import json
import subprocess
import sys
import threading
import traceback
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
from PIL import Image


# Lazy-loaded object detection pipeline (for bounding boxes on frame)
_detector = None
DETECTOR_MODEL = "facebook/detr-resnet-50"


def run_object_detection(image: Image.Image, top_k: int = 30, threshold: float = 0.25):
    """Run object detection on image. Returns list of {label, score, box} with box in normalized [0,1] coords (xmin, ymin, xmax, ymax)."""
    global _detector
    if _detector is None:
        from transformers import pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        for attempt, dev in enumerate([device_id, -1] if device_id != -1 else [-1]):
            try:
                _detector = pipeline("object-detection", model=DETECTOR_MODEL, device=dev)
                if attempt > 0:
                    print("Object detector loaded on CPU (GPU attempt failed).", file=sys.stderr)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e!r}" if str(e).strip() else f"{type(e).__name__}"
                print(f"Object detection not available (device={dev}): {msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                _detector = None
                if dev == -1:
                    return []
    try:
        w, h = image.size
        try:
            out = _detector(image, top_k=top_k, threshold=threshold)
        except TypeError:
            out = _detector(image, top_k=top_k)
        detections = []
        for item in out:
            b = item.get("box", {})
            xmin = b.get("xmin", 0) / w
            ymin = b.get("ymin", 0) / h
            xmax = b.get("xmax", w) / w
            ymax = b.get("ymax", h) / h
            detections.append({
                "label": item.get("label", "?"),
                "score": round(float(item.get("score", 0)), 2),
                "box": [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)],
            })
        return detections
    except Exception as e:
        msg = f"{type(e).__name__}: {e!r}" if not str(e).strip() else str(e)
        print(f"Detection failed: {msg}", file=sys.stderr)
        return []


def extract_one_frame_youtube(url: str, time_sec: float):
    """Extract a single frame from YouTube at time_sec. Returns PIL.Image or None."""
    url = (url or "").strip()
    if not url:
        return None
    time_sec = max(0.0, float(time_sec))
    cmd_ydl = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", "-",
        "--no-warnings",
        "--no-check-certificate",
        "--no-playlist",
        url,
    ]
    cmd_ffmpeg = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-ss", str(time_sec),
        "-i", "pipe:0",
        "-frames:v", "1",
        "-f", "image2pipe",
        "-c:v", "png",
        "pipe:1",
    ]
    try:
        p_ydl = subprocess.Popen(
            cmd_ydl,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        p_ff = subprocess.Popen(
            cmd_ffmpeg,
            stdin=p_ydl.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        p_ydl.stdout.close()
        png_data = p_ff.communicate(timeout=60)[0]
        p_ydl.terminate()
        p_ydl.wait(timeout=5)
        if not png_data:
            return None
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None

# Model choices (base, no adapter)
SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM2_500M = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
LLAVA_7B = "llava-hf/llava-1.5-7b-hf"
# DINOv3: self-supervised vision backbone (no language model). https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"

DEFAULT_PORT = 8087
MAX_NEW_TOKENS = 80


def load_smolvlm(model_name: str, device, progress_callback=None):
    import inspect
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if progress_callback:
        progress_callback(10, "Loading processor…")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if progress_callback:
        progress_callback(30, "Loading model…")
    # Load once: only pass attn_implementation if supported (avoids loading model twice on GPU on TypeError).
    kwargs = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "device_map": str(device) if device.type == "cuda" else None,
        "trust_remote_code": True,
    }
    try:
        sig = inspect.signature(AutoModelForImageTextToText.from_pretrained)
        if "attn_implementation" in sig.parameters:
            kwargs["attn_implementation"] = "eager"
    except Exception:
        pass
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    except TypeError:
        # Signature may have **kwargs; param exists but value not supported — clear GPU and retry without.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(model_name, **{k: v for k, v in kwargs.items() if k != "attn_implementation"})
        print("Note: attn_implementation not supported; attention map unavailable.", file=sys.stderr)
    if progress_callback:
        progress_callback(100, "Ready")
    return model, processor


def load_llava(device, progress_callback=None):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    if progress_callback:
        progress_callback(10, "Loading processor…")
    processor = AutoProcessor.from_pretrained(LLAVA_7B)
    if progress_callback:
        progress_callback(30, "Loading model…")
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_7B,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=str(device) if device.type == "cuda" else None,
    )
    if progress_callback:
        progress_callback(100, "Ready")
    return model, processor


def load_dinov3(device, progress_callback=None):
    """Load DINOv3 vision backbone (no language model). Produces dense features for downstream tasks."""
    from transformers import AutoImageProcessor, AutoModel
    if progress_callback:
        progress_callback(10, "Loading DINOv3 processor…")
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL)
    if progress_callback:
        progress_callback(30, "Loading DINOv3 backbone…")
    model = AutoModel.from_pretrained(DINOV3_MODEL)
    if device.type == "cuda":
        model = model.to(device)
    model.eval()
    if progress_callback:
        progress_callback(100, "Ready")
    return model, processor


def ask_dinov3(model, processor, image: Image.Image, question: str, device) -> str:
    """DINOv3 is vision-only: run image through backbone and return feature summary (no QA)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    inp = processor(images=image, return_tensors="pt")
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    pooled = getattr(out, "pooler_output", None) or (out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else None)
    patch_feats = getattr(out, "last_hidden_state", None)
    if pooled is not None:
        pooled = pooled.cpu().numpy()
    if patch_feats is not None:
        patch_feats = patch_feats.cpu().numpy()
    parts = ["DINOv3 (vision backbone): no language model. Image features extracted."]
    if pooled is not None:
        parts.append(" Pooled (CLS) shape: %s." % (list(pooled.shape),))
    if patch_feats is not None:
        parts.append(" Patch tokens shape: %s. Use /dinov3_features for raw features." % (list(patch_feats.shape),))
    return "".join(parts)


def ask_smolvlm(model, processor, image: Image.Image, question: str, device) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
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
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = processor.batch_decode(
        out[:, inp["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (gen[0] if gen else "").strip()


# Attention map (same model as QA; SmolVLM2 only)
ATTENTION_MAX_NEW_TOKENS = 80
PATCH_ROWS, PATCH_COLS = 3, 4
TOKENS_PER_PATCH = 64
PATCH_PIXELS = 512
IMAGE_TOKEN_COUNT = PATCH_ROWS * PATCH_COLS * TOKENS_PER_PATCH
_last_attention_run = {"attn": None, "tokens": [], "num_layers": 0, "num_heads": 0, "answer": "", "image_base64": None, "dinov3": False, "num_register_tokens": 0}


def run_dinov3_with_attentions(model, processor, image: Image.Image, device):
    """Run DINOv3 with output_attentions. Returns attn_list (one item per layer, each (num_heads, seq, seq)), num_layers, num_heads, num_register_tokens.
    Based on https://github.com/mselmangokmen/Dinov3-attention-map-extraction (CLS → patch attention)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    inp = processor(images=image, return_tensors="pt")
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    num_register_tokens = getattr(getattr(model, "config", None), "num_register_tokens", 4)
    try:
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
    except TypeError:
        with torch.no_grad():
            out = model(**inp)
    if not getattr(out, "attentions", None) or not out.attentions:
        return None, 0, 0, num_register_tokens
    # attentions: tuple of (batch, num_heads, seq, seq) per layer
    attn_list = []
    for layer_attn in out.attentions:
        a = layer_attn[0].float().cpu().numpy()  # (num_heads, seq, seq)
        attn_list.append(a)
    num_layers = len(attn_list)
    num_heads = attn_list[0].shape[0] if num_layers else 0
    return attn_list, num_layers, num_heads, num_register_tokens


def dinov3_attention_to_heatmap(attn_list, layer_idx, head_idx, num_register_tokens, out_h, out_w, threshold=0.0):
    """Build heatmap from DINOv3 CLS→patch attention (https://github.com/mselmangokmen/Dinov3-attention-map-extraction).
    attn_list: list of (num_heads, seq, seq). CLS=0, then num_register_tokens, then patch tokens. We use attention from CLS to patches."""
    if not attn_list:
        return None
    # layer_idx -1 = last layer (as in reference notebook "layer 11/12")
    if layer_idx < 0:
        layer_idx = len(attn_list) + layer_idx
    if layer_idx < 0 or layer_idx >= len(attn_list):
        return None
    attn = attn_list[layer_idx]  # (num_heads, seq, seq)
    num_heads, seq, _ = attn.shape
    patch_start = 1 + num_register_tokens
    n_patches = seq - patch_start
    if n_patches <= 0:
        return None
    if head_idx >= 0 and head_idx < num_heads:
        cls_to_patch = attn[head_idx, 0, patch_start:].astype(np.float32)
    else:
        cls_to_patch = attn[:, 0, patch_start:].mean(axis=0).astype(np.float32)
    side = int(round(np.sqrt(n_patches)))
    if side * side != n_patches:
        side = int(np.ceil(np.sqrt(n_patches)))
        cls_to_patch = np.pad(cls_to_patch, (0, side * side - n_patches), constant_values=0.0)[: side * side]
    grid = cls_to_patch.reshape(side, side)
    if threshold > 0:
        grid = np.where(grid >= threshold, grid, 0.0)
    if grid.shape[0] > 0 and grid.shape[1] > 0 and (out_h != side or out_w != side):
        # Upsample to (out_h, out_w) via repeat and crop
        ry = max(1, out_h // grid.shape[0])
        rx = max(1, out_w // grid.shape[1])
        grid = np.repeat(np.repeat(grid, ry, axis=0), rx, axis=1)[:out_h, :out_w]
        if grid.shape[0] < out_h or grid.shape[1] < out_w:
            padded = np.zeros((out_h, out_w), dtype=np.float32)
            padded[: grid.shape[0], : grid.shape[1]] = grid
            grid = padded
    return grid.astype(np.float32)


def run_with_attentions_smolvlm(model, processor, image: Image.Image, question: str, device):
    """Run SmolVLM with output_attentions; return answer, tokens, attn_list, num_layers, num_heads."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
    ]
    inp = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    prompt_length = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=ATTENTION_MAX_NEW_TOKENS,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    gen_ids = out.sequences[:, prompt_length:]
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    answer = (decoded[0] if decoded else "").strip()
    token_list = []
    for i in range(gen_ids.shape[1]):
        tok = processor.decode(gen_ids[0, i : i + 1], skip_special_tokens=False)
        token_list.append(tok if tok else "<unk>")
    if not getattr(out, "attentions", None) or not out.attentions:
        return answer, token_list, None, 0, 0
    num_steps = len(out.attentions)
    num_layers = len(out.attentions[0]) if num_steps else 0
    num_heads = out.attentions[0][0].shape[1] if num_layers else 0
    attn_list = []
    for step in range(num_steps):
        layer_list = []
        for layer in range(num_layers):
            a = out.attentions[step][layer][0].float().cpu().numpy()
            layer_list.append(a)
        attn_list.append(layer_list)
    return answer, token_list, attn_list, num_layers, num_heads


def _weighted_mean_axis0(arr, axis=0):
    """Weight by index (1,2,...,n) so later indices count more. arr shape (n, ...)."""
    n = arr.shape[axis]
    w = np.arange(1, n + 1, dtype=np.float32) / ((n * (n + 1)) / 2.0)
    # shape so w broadcasts: (1,...,1, n, 1,...,1) with n at position axis
    w = w.reshape((1,) * axis + (n,) + (1,) * (arr.ndim - axis - 1))
    return (arr * w).sum(axis=axis)


def _trimmed_mean_axis0(arr, trim_frac=0.25, axis=0):
    """Per position: sort values along axis, drop bottom and top trim_frac, mean of middle."""
    arr = np.asarray(arr, dtype=np.float32)
    sorted_arr = np.sort(arr, axis=axis)
    n = arr.shape[axis]
    drop = max(0, int(n * trim_frac))
    if drop * 2 >= n:
        return sorted_arr.take(n // 2, axis=axis)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(drop, n - drop)
    return sorted_arr[tuple(sl)].mean(axis=axis)


def attention_to_heatmap(attn_list, token_idx, layer_idx, head_idx, image_token_count, threshold=0.0):
    # layer_idx/head_idx: -3 trimmed mean, -2 weighted mean, -1 simple mean, >=0 single layer/head
    if token_idx >= len(attn_list):
        return None
    num_layers = len(attn_list[0])
    if num_layers == 0 or layer_idx >= num_layers:
        return None
    # For decoder: use last query position when q_len>1 (attention from the newly generated token)
    def _q_slice(layer_attn):
        q_len = layer_attn.shape[1]
        q_idx = -1 if q_len > 1 else 0
        return layer_attn[:, q_idx, :]
    layer_agg = layer_idx in (-3, -2, -1)
    if layer_idx >= 0:
        layer_attn = attn_list[token_idx][layer_idx]  # (num_heads, q_len, seq)
        num_heads = layer_attn.shape[0]
        if head_idx >= num_heads:
            return None
        q_len = layer_attn.shape[1]
        q_idx = -1 if q_len > 1 else 0
        if head_idx >= 0:
            attn = layer_attn[head_idx, q_idx, :].astype(np.float32)
        else:
            sl = _q_slice(layer_attn)
            if head_idx == -1:
                attn = sl.mean(axis=0).astype(np.float32)
            elif head_idx == -2:
                attn = _weighted_mean_axis0(sl, axis=0)
            else:  # -3
                attn = _trimmed_mean_axis0(sl, trim_frac=0.25, axis=0)
    else:
        stacked = np.stack([_q_slice(attn_list[token_idx][l]) for l in range(num_layers)], axis=0)
        nh = stacked.shape[1]
        if head_idx >= nh:
            return None
        if head_idx >= 0:
            single_head = stacked[:, head_idx, :]  # (num_layers, seq)
            if layer_idx == -1:
                attn = single_head.mean(axis=0).astype(np.float32)
            elif layer_idx == -2:
                attn = _weighted_mean_axis0(single_head, axis=0)
            else:
                attn = _trimmed_mean_axis0(single_head, trim_frac=0.25, axis=0)
        else:
            # average over both layers and heads
            flat = stacked.reshape(-1, stacked.shape[2])  # (L*H, seq)
            if layer_idx == -1 and head_idx == -1:
                attn = flat.mean(axis=0).astype(np.float32)
            elif layer_idx == -2 or head_idx == -2:
                attn = _weighted_mean_axis0(flat, axis=0)
            else:
                attn = _trimmed_mean_axis0(flat, trim_frac=0.25, axis=0)
    n = min(image_token_count, attn.shape[0])
    attn_img = attn[:n].copy()
    if threshold > 0:
        attn_img = np.where(attn_img >= threshold, attn_img, 0.0)
    num_patches = PATCH_ROWS * PATCH_COLS
    if attn_img.size < num_patches * TOKENS_PER_PATCH:
        attn_img = np.pad(attn_img, (0, num_patches * TOKENS_PER_PATCH - attn_img.size), constant_values=0.0)
    attn_img = attn_img[: num_patches * TOKENS_PER_PATCH].reshape(num_patches, TOKENS_PER_PATCH)
    patch_attn = attn_img.mean(axis=1)
    grid = patch_attn.reshape(PATCH_ROWS, PATCH_COLS)
    heatmap = np.repeat(np.repeat(grid, PATCH_PIXELS, axis=0), PATCH_PIXELS, axis=1)
    return heatmap


def attention_raw_vector(attn_list, token_idx, layer_idx, head_idx, image_token_count):
    """Return the 1D attention vector over image tokens (for distribution stats). Same aggregation as attention_to_heatmap."""
    if token_idx >= len(attn_list):
        return None
    num_layers = len(attn_list[0])
    if num_layers == 0 or layer_idx >= num_layers:
        return None
    def _q_slice(layer_attn):
        q_len = layer_attn.shape[1]
        q_idx = -1 if q_len > 1 else 0
        return layer_attn[:, q_idx, :]
    if layer_idx >= 0:
        layer_attn = attn_list[token_idx][layer_idx]
        num_heads = layer_attn.shape[0]
        if head_idx >= num_heads:
            return None
        q_len = layer_attn.shape[1]
        q_idx = -1 if q_len > 1 else 0
        if head_idx >= 0:
            attn = layer_attn[head_idx, q_idx, :].astype(np.float32)
        else:
            sl = _q_slice(layer_attn)
            if head_idx == -1:
                attn = sl.mean(axis=0).astype(np.float32)
            elif head_idx == -2:
                attn = _weighted_mean_axis0(sl, axis=0)
            else:
                attn = _trimmed_mean_axis0(sl, trim_frac=0.25, axis=0)
    else:
        stacked = np.stack([_q_slice(attn_list[token_idx][l]) for l in range(num_layers)], axis=0)
        nh = stacked.shape[1]
        if head_idx >= nh:
            return None
        if head_idx >= 0:
            single_head = stacked[:, head_idx, :]
            if layer_idx == -1:
                attn = single_head.mean(axis=0).astype(np.float32)
            elif layer_idx == -2:
                attn = _weighted_mean_axis0(single_head, axis=0)
            else:
                attn = _trimmed_mean_axis0(single_head, trim_frac=0.25, axis=0)
        else:
            flat = stacked.reshape(-1, stacked.shape[2])
            if layer_idx == -1 and head_idx == -1:
                attn = flat.mean(axis=0).astype(np.float32)
            elif layer_idx == -2 or head_idx == -2:
                attn = _weighted_mean_axis0(flat, axis=0)
            else:
                attn = _trimmed_mean_axis0(flat, trim_frac=0.25, axis=0)
    n = min(image_token_count, attn.shape[0])
    return attn[:n].copy()


def heatmap_to_bbox(heatmap, percentile=80, margin_frac=0.05):
    """Get axis-aligned bbox of the hot region (above percentile). Returns (x_min, y_min, x_max, y_max) in pixel coords."""
    if heatmap.size == 0:
        return None
    thresh = np.percentile(heatmap, percentile)
    above = heatmap >= thresh
    if not np.any(above):
        return None
    rows = np.any(above, axis=1)
    cols = np.any(above, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = heatmap.shape
    margin_x = max(1, int((x_max - x_min) * margin_frac))
    margin_y = max(1, int((y_max - y_min) * margin_frac))
    x_min = max(0, x_min - margin_x)
    x_max = min(w, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(h, y_max + margin_y)
    return (x_min, y_min, x_max, y_max)


def heatmap_to_bboxes(heatmap, percentile=80, margin_frac=0.05, max_boxes=5):
    """Get one or more axis-aligned bboxes for hot regions. Returns list of (x_min, y_min, x_max, y_max)."""
    main = heatmap_to_bbox(heatmap, percentile=percentile, margin_frac=margin_frac)
    if main is None:
        return []
    return [main]


def attention_distribution_stats(raw_attn, num_bins=32):
    """Return min, max, percentiles, and histogram counts for normalized distribution (sum=1; comparable across layers/heads)."""
    if raw_attn is None or raw_attn.size == 0:
        return None
    arr = np.asarray(raw_attn).ravel().astype(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    total = float(arr.sum())
    if total < 1e-12:
        total = 1.0
    normalized = arr / total
    vmin, vmax = float(np.min(normalized)), float(np.max(normalized))
    rng = (vmin, vmax + 1e-12) if vmax > vmin else (0.0, 1.0)
    counts, bin_edges = np.histogram(normalized, bins=num_bins, range=rng)
    return {
        "min": vmin,
        "max": vmax,
        "mean": float(np.mean(normalized)),
        "p5": float(np.percentile(normalized, 5)),
        "p25": float(np.percentile(normalized, 25)),
        "p50": float(np.percentile(normalized, 50)),
        "p75": float(np.percentile(normalized, 75)),
        "p95": float(np.percentile(normalized, 95)),
        "histogram": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def attention_histogram_base64(stats, width=280, height=120):
    """Draw a small histogram from distribution stats; return base64 PNG."""
    if not stats or "histogram" not in stats:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = stats["histogram"]
    edges = stats.get("bin_edges", [])
    if len(edges) != len(counts) + 1:
        edges = np.linspace(stats["min"], stats["max"], len(counts) + 1).tolist()
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)
    ax.bar(range(len(counts)), counts, color="orangered", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Norm. attention (bin)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution (normalized)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap_to_base64_png(heatmap, threshold=0.0, percentile_normalize=True, p_low=5, p_high=95):
    """Percentile-normalize by default so contrast is visible when values are similar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if threshold > 0:
        heatmap = np.where(heatmap >= threshold, heatmap, 0.0)
    vmin, vmax = heatmap.min(), heatmap.max()
    if percentile_normalize and heatmap.size > 0:
        vmin = np.percentile(heatmap, p_low)
        vmax = np.percentile(heatmap, p_high)
    if vmax - vmin > 1e-6:
        heatmap = np.clip((heatmap - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    else:
        heatmap = np.zeros_like(heatmap)
    fig, ax = plt.subplots(figsize=(heatmap.shape[1] / 100, heatmap.shape[0] / 100), dpi=100)
    ax.imshow(heatmap, cmap="hot", alpha=0.7, interpolation="bilinear")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def ask_llava(model, processor, image: Image.Image, question: str, device) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
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
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = processor.batch_decode(
        out[:, inp["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (gen[0] if gen else "").strip()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM QA — Upload, Webcam, or YouTube</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #1a1a2e; color: #eee; min-height: 100vh; overflow: hidden; }
    #loadOverlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; padding: 24px; }
    #loadOverlay.hidden { display: none; }
    #loadOverlay .msg { color: #a0a0a0; margin-bottom: 8px; }
    #mainContent { display: none; height: 100vh; }
    #mainContent.ready { display: flex; flex-direction: row; }
    .left-col { width: 380px; min-width: 380px; padding: 24px; overflow-y: auto; }
    .right-col { flex: 1; min-width: 0; display: flex; flex-direction: column; background: #0f0f1a; padding: 16px; }
    #ytFrameSection { flex: 1; display: flex; flex-direction: column; min-height: 0; }
    #ytFramePlaceholder { flex: 1; display: flex; align-items: center; justify-content: center; padding: 24px; }
    .right-col .frame-container { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
    .right-col .frame-container img, .right-col .frame-container canvas { max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; border-radius: 8px; border: 2px solid #333; }
    #ytFrameSection { display: flex; flex-direction: column; min-height: 0; flex: 1; }
    h1 { color: #e94560; margin-top: 0; font-size: 1.25rem; }
    .card { background: #16213e; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tabs button { padding: 10px 16px; border-radius: 8px; border: 1px solid #333; background: #0f0f1a; color: #ccc; cursor: pointer; }
    .tabs button.active { background: #e94560; color: #fff; border-color: #e94560; }
    .tabs button:hover:not(.active) { background: #252540; color: #eee; }
    .panel { display: none; }
    .panel.active { display: block; }
    label { display: block; margin-bottom: 8px; color: #a0a0a0; }
    input[type="file"], input[type="text"], input[type="number"], textarea { margin-bottom: 12px; color: #eee; width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #333; background: #0f0f1a; }
    textarea { min-height: 80px; resize: vertical; }
    button { background: #e94560; color: #fff; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #ff6b6b; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.secondary { background: #37474f; }
    button.secondary:hover { background: #455a64; }
    .result-box { min-height: 60px; margin-bottom: 20px; padding: 16px; border-radius: 8px; background: #0f0f1a; border: 1px solid #2a2a4a; white-space: pre-wrap; color: #a0a0a0; }
    .result-box.has-content { color: #eee; }
    .result-box.error { background: #2a1515; border-color: #6a2020; color: #ffabab; }
    #preview, #webcamPreview, #ytPreview { max-width: 100%; max-height: 240px; border-radius: 8px; border: 2px solid #333; margin-top: 8px; display: block; }
    #webcamVideo { width: 100%; max-height: 240px; border-radius: 8px; border: 2px solid #333; background: #000; }
    .hidden { display: none !important; }
    a { color: #e94560; }
    #plotlyFrame { width: 100%; height: 100%; min-height: 0; }
    .plotly-container { position: relative; flex: 1; min-height: 0; max-height: 100vh; display: flex; align-items: center; justify-content: center; background: #0f0f1a; }
  </style>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="loadOverlay">
    <p class="msg">Loading VLM…</p>
    <p id="loadPct">0%</p>
  </div>
  <div id="mainContent">
    <div class="left-col">
    <h1>VLM QA</h1>
    <p style="color:#888; font-size:0.9rem;">Choose source: <strong>Upload</strong>, <strong>Webcam</strong>, or <strong>YouTube</strong>.</p>
    <div class="card">
      <div class="tabs">
        <button type="button" id="tabUpload" class="active">Upload image</button>
        <button type="button" id="tabWebcam">Webcam</button>
        <button type="button" id="tabYouTube">YouTube</button>
      </div>

      <div id="panelUpload" class="panel active">
        <label for="file">Image</label>
        <input type="file" id="file" accept="image/*">
        <div id="previewWrap" class="hidden"><img id="preview" alt="Preview"></div>
      </div>

      <div id="panelWebcam" class="panel">
        <label>Camera</label>
        <video id="webcamVideo" autoplay playsinline muted></video>
        <p id="webcamStatus" style="color:#888; font-size:0.9rem; margin:8px 0;">Allow camera access to use webcam.</p>
        <button type="button" id="webcamStart" class="secondary">Start webcam</button>
        <button type="button" id="webcamCapture" class="secondary" disabled>Capture this frame</button>
        <div id="webcamCapturedWrap" class="hidden"><img id="webcamPreview" alt="Captured"></div>
      </div>

      <div id="panelYouTube" class="panel">
        <label for="ytUrl">YouTube URL</label>
        <input type="text" id="ytUrl" placeholder="https://www.youtube.com/watch?v=...">
        <label for="ytTime">Time (seconds)</label>
        <input type="number" id="ytTime" value="0" min="0" step="1" placeholder="0">
        <p style="color:#888; font-size:0.85rem; margin-top:4px;">Frame at this position will be used.</p>
      </div>

      <label for="result">Answer</label>
      <div id="result" class="result-box" aria-live="polite">—</div>

      <label for="question">Question</label>
      <textarea id="question" placeholder="e.g. Is the person wearing a hardhat? Answer yes or no."></textarea>
      <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:8px;">
        <button type="button" id="askBtn" disabled>Ask</button>
        <button type="button" id="btnViewAttention" class="secondary" title="Same image + question on port 8088, open attention heatmap">View attention map</button>
        <span id="attentionStatus" style="color:#888; font-size:0.85rem;"></span>
      </div>
    </div>
    <div class="card">
      <h2 style="font-size:1.1rem; margin-top:0;">Pipeline</h2>
      <p style="color:#888; font-size:0.85rem; margin-bottom:12px;">Run questions, then get real coordinates (DETR) and plot. Add comments to direct the pipeline flow.</p>
      <div id="pipelineSteps"></div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
        <button type="button" id="pipelineAddQuestion" class="secondary">Add question</button>
        <button type="button" id="pipelineAddGetCoords" class="secondary">Add: Get coordinates (detection)</button>
        <button type="button" id="pipelineAddComment" class="secondary">Add comment / direction</button>
        <button type="button" id="pipelineAddFindCoords" class="secondary">Add: Find coordinates &amp; plot</button>
      </div>
      <button type="button" id="pipelineRun" class="secondary">Run pipeline</button>
      <div id="pipelineResults" class="result-box" style="margin-top:12px; min-height:40px;" aria-live="polite">—</div>
    </div>
    <p style="color:#666; font-size:0.9rem;"><a href="/info">API info</a></p>
    </div>
    <div class="right-col">
      <div id="ytFrameSection">
        <p id="ytFramePlaceholder" style="color:#666; font-size:0.95rem; margin:0;">Use the YouTube tab and click Ask to see the frame. Then use &quot;Open in Plotly&quot; to zoom, pan, and annotate.</p>
        <div id="plotlyContainer" class="hidden plotly-container">
          <div id="plotlyFrame" class="hidden" style="position:absolute; inset:0;"></div>
          <img id="plotlyFallbackImg" class="hidden" alt="Frame" style="width:100%; height:100%; object-fit:contain; display:block;">
        </div>
        <div id="ytFrameOverlayOpts" class="hidden" style="margin-top:12px; flex-shrink:0;">
          <button type="button" id="btnOpenPlotly" class="secondary">Open in Plotly (zoom, pan, annotate)</button>
          <button type="button" id="btnBackToImage" class="secondary hidden">Back to image</button>
          <label style="display:inline-flex; align-items:center; gap:8px; cursor:pointer; margin-left:12px;">
            <input type="checkbox" id="overlayAnswerCheck">
            <span>Show answer on plot</span>
          </label>
        </div>
      </div>
    </div>
  </div>
  <script>
    (function() {
      var source = 'upload';
      var file = document.getElementById('file');
      var question = document.getElementById('question');
      var askBtn = document.getElementById('askBtn');
      var result = document.getElementById('result');
      var preview = document.getElementById('preview');
      var previewWrap = document.getElementById('previewWrap');
      var webcamVideo = document.getElementById('webcamVideo');
      var webcamStart = document.getElementById('webcamStart');
      var webcamCapture = document.getElementById('webcamCapture');
      var webcamStatus = document.getElementById('webcamStatus');
      var webcamPreview = document.getElementById('webcamPreview');
      var webcamCapturedWrap = document.getElementById('webcamCapturedWrap');
      var ytUrl = document.getElementById('ytUrl');
      var ytTime = document.getElementById('ytTime');
      var ytFrameSection = document.getElementById('ytFrameSection');
      var ytFramePlaceholder = document.getElementById('ytFramePlaceholder');
      var plotlyContainer = document.getElementById('plotlyContainer');
      var plotlyFrame = document.getElementById('plotlyFrame');
      var ytFrameOverlayOpts = document.getElementById('ytFrameOverlayOpts');
      var overlayAnswerCheck = document.getElementById('overlayAnswerCheck');
      var btnOpenPlotly = document.getElementById('btnOpenPlotly');
      var btnBackToImage = document.getElementById('btnBackToImage');
      var webcamStream = null;
      var capturedDataUrl = null;
      var lastYtFrameBase64 = null;
      var lastYtAnswer = null;
      var lastYtDetections = [];
      var pipelineSteps = [];
      var pipelineStepsEl = document.getElementById('pipelineSteps');
      var pipelineResultsEl = document.getElementById('pipelineResults');
      var pipelineRunBtn = document.getElementById('pipelineRun');
      var pipelineLastFrame = null;
      var pipelineLastDetections = null;

      function setSource(s) {
        source = s;
        document.querySelectorAll('.tabs button').forEach(function(b) { b.classList.remove('active'); });
        document.getElementById('tab' + (s === 'upload' ? 'Upload' : s === 'webcam' ? 'Webcam' : 'YouTube')).classList.add('active');
        document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
        document.getElementById('panel' + (s === 'upload' ? 'Upload' : s === 'webcam' ? 'Webcam' : 'YouTube')).classList.add('active');
        updateAskBtn();
      }
      document.getElementById('tabUpload').onclick = function() { setSource('upload'); };
      document.getElementById('tabWebcam').onclick = function() { setSource('webcam'); };
      document.getElementById('tabYouTube').onclick = function() { setSource('youtube'); };

      function updateAskBtn() {
        var q = question.value.trim();
        var ok = false;
        if (source === 'upload') ok = file.files && file.files[0] && q;
        else if (source === 'webcam') ok = capturedDataUrl && q;
        else if (source === 'youtube') ok = ytUrl.value.trim() && q;
        askBtn.disabled = !ok;
      }
      file.addEventListener('change', function() {
        if (file.files && file.files[0]) {
          previewWrap.classList.remove('hidden');
          preview.src = URL.createObjectURL(file.files[0]);
        }
        updateAskBtn();
      });
      question.addEventListener('input', updateAskBtn);
      ytUrl.addEventListener('input', updateAskBtn);
      ytTime.addEventListener('input', updateAskBtn);

      webcamStart.addEventListener('click', function() {
        if (webcamStream) {
          webcamStream.getTracks().forEach(function(t) { t.stop(); });
          webcamStream = null;
          webcamVideo.srcObject = null;
          webcamStart.textContent = 'Start webcam';
          webcamCapture.disabled = true;
          webcamStatus.textContent = 'Camera stopped.';
          return;
        }
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
          .then(function(stream) {
            webcamStream = stream;
            webcamVideo.srcObject = stream;
            webcamStart.textContent = 'Stop webcam';
            webcamCapture.disabled = false;
            webcamStatus.textContent = 'Camera on. Click "Capture this frame" then ask.';
          })
          .catch(function(e) {
            webcamStatus.textContent = 'Error: ' + (e.message || 'Could not access camera');
          });
      });
      webcamCapture.addEventListener('click', function() {
        if (!webcamStream || !webcamVideo.videoWidth) return;
        var c = document.createElement('canvas');
        c.width = webcamVideo.videoWidth;
        c.height = webcamVideo.videoHeight;
        c.getContext('2d').drawImage(webcamVideo, 0, 0);
        capturedDataUrl = c.toDataURL('image/jpeg', 0.9);
        webcamPreview.src = capturedDataUrl;
        webcamCapturedWrap.classList.remove('hidden');
        updateAskBtn();
      });

      overlayAnswerCheck.addEventListener('change', function() {
        if (lastYtFrameBase64 && plotlyFrame.classList.contains('hidden') === false) {
          renderPlotlyFrame();
        }
      });
      btnOpenPlotly.addEventListener('click', function() {
        if (!lastYtFrameBase64) return;
        var c = plotlyContainer;
        var w = c.clientWidth || 800;
        var h = c.clientHeight || 600;
        plotlyFrame.style.width = w + 'px';
        plotlyFrame.style.height = h + 'px';
        if (plotlyFallbackImg) plotlyFallbackImg.classList.add('hidden');
        plotlyFrame.classList.remove('hidden');
        btnOpenPlotly.classList.add('hidden');
        if (btnBackToImage) btnBackToImage.classList.remove('hidden');
        renderPlotlyFrame();
      });
      if (btnBackToImage) btnBackToImage.addEventListener('click', function() {
        plotlyFrame.classList.add('hidden');
        if (plotlyFallbackImg) plotlyFallbackImg.classList.remove('hidden');
        btnBackToImage.classList.add('hidden');
        if (btnOpenPlotly) btnOpenPlotly.classList.remove('hidden');
      });

      function showResult(text, isError) {
        result.textContent = text || '—';
        result.classList.toggle('has-content', !!text);
        result.classList.toggle('error', !!isError);
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      function isAnswerPositive(txt) {
        if (!txt || !String(txt).trim()) return false;
        var s = String(txt).toLowerCase().trim();
        if (/^no\\b|\\bno[,.]?\\s*$|^nope\\b|^negative\\b|there is no|there are no|there isn't|there aren't|i don't see|not present|\\bnone\\b|,\\s*no\\.?\\s*$|\\.\\s*no\\s*$/.test(s)) return false;
        if (/^yes\\b|\\byes[,.]?\\s*$|^yeah|^yep|there is a |there are |there is |i see |present|\\bperson\\b|\\btruck\\b|\\bcar\\b/.test(s)) return true;
        if (s === 'no' || s === 'n') return false;
        return false;
      }

      function renderPipelineSteps() {
        pipelineStepsEl.innerHTML = '';
        pipelineSteps.forEach(function(s, i) {
          var row = document.createElement('div');
          row.style.cssText = 'display:flex; align-items:center; gap:8px; margin-bottom:8px;';
          if (s.type === 'question') {
            var lab = document.createElement('span');
            lab.textContent = (i + 1) + '. Question:';
            lab.style.minWidth = '90px';
            var inp = document.createElement('input');
            inp.type = 'text';
            inp.placeholder = 'e.g. Is there a person?';
            inp.value = s.text || '';
            inp.style.flex = '1';
            inp.addEventListener('change', function() { s.text = inp.value.trim(); });
            inp.addEventListener('input', function() { s.text = inp.value.trim(); });
            var rm = document.createElement('button');
            rm.type = 'button';
            rm.textContent = 'Remove';
            rm.className = 'secondary';
            rm.style.flexShrink = '0';
            rm.addEventListener('click', function() { pipelineSteps.splice(i, 1); renderPipelineSteps(); });
            row.appendChild(lab);
            row.appendChild(inp);
            row.appendChild(rm);
          } else if (s.type === 'action' && s.action === 'get_coordinates') {
            var lab = document.createElement('span');
            lab.textContent = (i + 1) + '. Get coordinates (detection)';
            lab.style.flex = '1';
            var rm = document.createElement('button');
            rm.type = 'button';
            rm.textContent = 'Remove';
            rm.className = 'secondary';
            rm.addEventListener('click', function() { pipelineSteps.splice(i, 1); renderPipelineSteps(); });
            row.appendChild(lab);
            row.appendChild(rm);
          } else if (s.type === 'comment') {
            var lab = document.createElement('span');
            lab.textContent = (i + 1) + '. Comment:';
            lab.style.minWidth = '85px';
            var inp = document.createElement('input');
            inp.type = 'text';
            inp.placeholder = 'e.g. Only plot if previous answer is yes';
            inp.value = s.text || '';
            inp.style.flex = '1';
            inp.addEventListener('change', function() { s.text = inp.value.trim(); });
            inp.addEventListener('input', function() { s.text = inp.value.trim(); });
            var rm = document.createElement('button');
            rm.type = 'button';
            rm.textContent = 'Remove';
            rm.className = 'secondary';
            rm.addEventListener('click', function() { pipelineSteps.splice(i, 1); renderPipelineSteps(); });
            row.appendChild(lab);
            row.appendChild(inp);
            row.appendChild(rm);
          } else {
            var lab = document.createElement('span');
            lab.textContent = (i + 1) + '. Find coordinates & plot';
            lab.style.flex = '1';
            var rm = document.createElement('button');
            rm.type = 'button';
            rm.textContent = 'Remove';
            rm.className = 'secondary';
            rm.addEventListener('click', function() { pipelineSteps.splice(i, 1); renderPipelineSteps(); });
            row.appendChild(lab);
            row.appendChild(rm);
          }
          pipelineStepsEl.appendChild(row);
        });
      }
      document.getElementById('pipelineAddQuestion').addEventListener('click', function() {
        pipelineSteps.push({ type: 'question', text: '' });
        renderPipelineSteps();
      });
      document.getElementById('pipelineAddGetCoords').addEventListener('click', function() {
        pipelineSteps.push({ type: 'action', action: 'get_coordinates' });
        renderPipelineSteps();
      });
      document.getElementById('pipelineAddComment').addEventListener('click', function() {
        pipelineSteps.push({ type: 'comment', text: '' });
        renderPipelineSteps();
      });
      document.getElementById('pipelineAddFindCoords').addEventListener('click', function() {
        pipelineSteps.push({ type: 'action', action: 'find_coordinates' });
        renderPipelineSteps();
      });

      function getPlotFilterFromComments(steps) {
        var plotIdx = steps.length;
        for (var i = steps.length - 1; i >= 0; i--) {
          if (steps[i].type === 'action' && steps[i].action === 'find_coordinates') { plotIdx = i; break; }
        }
        var lastComment = '';
        for (var j = plotIdx - 1; j >= 0; j--) {
          if (steps[j].type === 'comment' && (steps[j].text || '').trim()) {
            lastComment = (steps[j].text || '').trim().toLowerCase();
            break;
          }
        }
        if (!lastComment) return null;
        var m = lastComment.match(/(?:only\\s+plot|plot\\s+only)\\s+(?:the\\s+)?(\\w+)(s)?\\b|only\\s+(?:the\\s+)?(\\w+)(s)?\\s*(?:in\\s+the\\s+next\\s+step)?\\b|just\\s+(?:the\\s+)?(\\w+)(s)?\\b|(?:plot\\s+)?only\\s+the\\s+(\\w+)(s)?\\b/i);
        if (!m) return null;
        var word = (m[1] || m[3] || m[5] || m[7] || '').toLowerCase();
        var plural = m[2] || m[4] || m[6] || m[8];
        if (!word) return null;
        var singular = word;
        if (plural) singular = word.replace(/s$/, '');
        if (word === 'people' || word === 'person') singular = 'person';
        else if (word === 'persons') singular = 'person';
        return [singular];
      }

      function runPipeline() {
        if (pipelineSteps.length === 0) {
          pipelineResultsEl.textContent = 'Add at least one step (question or Find coordinates).';
          pipelineResultsEl.classList.add('error');
          return;
        }
        var hasImage = false;
        if (source === 'youtube') hasImage = ytUrl.value.trim();
        else if (source === 'upload') hasImage = file.files && file.files[0];
        else if (source === 'webcam') hasImage = !!capturedDataUrl;
        if (!hasImage) {
          pipelineResultsEl.textContent = 'Choose a source and provide an image (YouTube URL, upload, or capture webcam) first.';
          pipelineResultsEl.classList.add('error');
          return;
        }
        pipelineRunBtn.disabled = true;
        pipelineResultsEl.textContent = 'Running pipeline…';
        pipelineResultsEl.classList.remove('error');
        pipelineLastFrame = null;
        pipelineLastDetections = null;
        var results = [];
        var idx = 0;
        function next() {
          if (idx >= pipelineSteps.length) {
            if (pipelineLastFrame) {
              lastYtFrameBase64 = pipelineLastFrame;
              lastYtAnswer = results[results.length - 1] || lastYtAnswer;
              var dets = pipelineLastDetections || [];
              var lastQuestionAnswer = '';
              for (var i = pipelineSteps.length - 1; i >= 0; i--) {
                if (pipelineSteps[i].type === 'question' && results[i]) {
                  var r = results[i];
                  var parts = r.split(/\\s*[→\\-–—]\\s*/);
                  lastQuestionAnswer = (parts.length > 1) ? parts[parts.length - 1].trim() : r.replace(/^Step\\s+\\d+:\\s*/i, '').trim();
                  break;
                }
              }
              if (lastQuestionAnswer !== '' && !isAnswerPositive(lastQuestionAnswer)) dets = [];
              else {
                var commentFilter = getPlotFilterFromComments(pipelineSteps);
                if (commentFilter && commentFilter.length > 0) {
                  dets = dets.filter(function(d) {
                    var L = (d.label || '').toLowerCase();
                    return commentFilter.some(function(c) { return c === L; });
                  });
                  results.push('Plot filter applied: showing only ' + commentFilter.join(', ') + '.');
                } else {
                  var prevStep = pipelineSteps[pipelineSteps.length - 2];
                  if (prevStep && prevStep.type === 'question' && (prevStep.text || '').toLowerCase().indexOf('person') >= 0) {
                    if (/\\byes\\b|\\bperson\\b|there is/i.test(lastQuestionAnswer)) {
                      var personOnly = dets.filter(function(d) { return (d.label || '').toLowerCase() === 'person'; });
                      if (personOnly.length > 0) dets = personOnly;
                    }
                  }
                }
              }
              lastYtDetections = dets;
              ytFramePlaceholder.classList.add('hidden');
              plotlyContainer.classList.remove('hidden');
              ytFrameOverlayOpts.classList.remove('hidden');
              if (plotlyFallbackImg) {
                plotlyFallbackImg.src = 'data:image/jpeg;base64,' + pipelineLastFrame;
                plotlyFallbackImg.classList.remove('hidden');
              }
              plotlyFrame.classList.add('hidden');
              var w = plotlyContainer.clientWidth || 800;
              var h = plotlyContainer.clientHeight || 600;
              plotlyFrame.style.width = w + 'px';
              plotlyFrame.style.height = h + 'px';
              if (plotlyFallbackImg) plotlyFallbackImg.classList.add('hidden');
              plotlyFrame.classList.remove('hidden');
              btnOpenPlotly.classList.add('hidden');
              if (btnBackToImage) btnBackToImage.classList.remove('hidden');
              renderPlotlyFrame();
            }
            pipelineResultsEl.textContent = results.join('\\n');
            pipelineResultsEl.classList.add('has-content');
            pipelineRunBtn.disabled = false;
            return;
          }
          var step = pipelineSteps[idx];
          if (step.type === 'comment') {
            var cmt = (step.text || '').trim();
            results.push('Step ' + (idx + 1) + ': [Comment] ' + (cmt || '(no text)'));
            idx++; next();
            return;
          }
          if (step.type === 'question') {
            var q = (step.text || '').trim();
            if (!q) { results.push('Step ' + (idx + 1) + ': (no question)'); idx++; next(); return; }
            if (source === 'youtube') {
              fetch('/ask_youtube', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: ytUrl.value.trim(), time_sec: parseFloat(ytTime.value) || 0, question: q })
              })
                .then(function(r) { return r.json(); })
                .then(function(d) {
                  if (d.error) results.push('Step ' + (idx + 1) + ': ' + d.error);
                  else {
                    results.push('Step ' + (idx + 1) + ': ' + q + ' → ' + (d.answer || ''));
                    if (d.frame_base64) { pipelineLastFrame = d.frame_base64; pipelineLastDetections = d.detections || []; }
                  }
                  idx++; next();
                })
                .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
            } else {
              var body;
              if (source === 'upload') {
                var fd = new FormData();
                fd.append('image', file.files[0]);
                fd.append('question', q);
                body = fd;
              } else {
                body = JSON.stringify({ image_base64: capturedDataUrl.split(',')[1], question: q });
              }
              var opts = source === 'upload' ? { method: 'POST', body: body } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body };
              fetch('/ask', opts)
                .then(function(r) { return r.json(); })
                .then(function(d) {
                  if (d.error) results.push('Step ' + (idx + 1) + ': ' + d.error);
                  else results.push('Step ' + (idx + 1) + ': ' + q + ' → ' + (d.answer || ''));
                  idx++; next();
                })
                .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
            }
            return;
          }
          if (step.type === 'action' && step.action === 'get_coordinates') {
            function fmtCoords(dets) {
              if (!dets || dets.length === 0) return 'No detections.';
              var lines = [];
              lines.push('Coordinates normalized 0-1, origin bottom-left (y_bottom = 1 - y_top):');
              dets.forEach(function(d, i) {
                var b = d.box || [];
                if (b.length < 4) return;
                var xmin = b[0], ymin = b[1], xmax = b[2], ymax = b[3];
                var yminBottom = (1 - ymax).toFixed(4);
                var ymaxBottom = (1 - ymin).toFixed(4);
                var label = (d.label || 'object') + (d.score != null ? ' ' + Math.round(d.score * 100) + '%' : '');
                lines.push((i + 1) + '. ' + label + ': [xmin=' + xmin.toFixed(4) + ', ymin_bottom=' + yminBottom + ', xmax=' + xmax.toFixed(4) + ', ymax_bottom=' + ymaxBottom + ']');
              });
              return lines.join('\\n');
            }
            if (source === 'youtube') {
              fetch('/detect_youtube', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: ytUrl.value.trim(), time_sec: parseFloat(ytTime.value) || 0 }) })
                .then(function(r) { return r.json(); })
                .then(function(d) {
                  if (d.error) results.push('Step ' + (idx + 1) + ': ' + d.error);
                  else {
                    pipelineLastFrame = d.frame_base64;
                    pipelineLastDetections = d.detections || [];
                    results.push('Step ' + (idx + 1) + ': ' + fmtCoords(pipelineLastDetections));
                  }
                  idx++; next();
                })
                .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
              return;
            }
            var detBody;
            if (source === 'upload') { var fd = new FormData(); fd.append('image', file.files[0]); detBody = fd; }
            else detBody = JSON.stringify({ image_base64: capturedDataUrl.split(',')[1] });
            var detOpts = source === 'upload' ? { method: 'POST', body: detBody } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: detBody };
            fetch('/detect', detOpts)
              .then(function(r) { return r.json(); })
              .then(function(d) {
                if (d.error) results.push('Step ' + (idx + 1) + ': ' + d.error);
                else {
                  pipelineLastFrame = d.frame_base64;
                  pipelineLastDetections = d.detections || [];
                  results.push('Step ' + (idx + 1) + ': ' + fmtCoords(pipelineLastDetections));
                }
                idx++; next();
              })
              .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
            return;
          }
          if (step.type === 'action' && step.action === 'find_coordinates') {
            if (source === 'youtube') {
              fetch('/detect_youtube', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: ytUrl.value.trim(), time_sec: parseFloat(ytTime.value) || 0 }) })
                .then(function(r) { return r.json(); })
                .then(function(d) {
                  if (d.error) {
                    results.push('Step ' + (idx + 1) + ': ' + d.error);
                  } else {
                    pipelineLastFrame = d.frame_base64 || pipelineLastFrame;
                    pipelineLastDetections = d.detections || [];
                    results.push('Step ' + (idx + 1) + ': Plotted ' + pipelineLastDetections.length + ' detection(s) on frame.');
                  }
                  idx++; next();
                })
                .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
              return;
            }
            if (source !== 'youtube') {
              var detBody;
              if (source === 'upload') {
                var fd = new FormData();
                fd.append('image', file.files[0]);
                detBody = fd;
              } else {
                detBody = JSON.stringify({ image_base64: capturedDataUrl.split(',')[1] });
              }
              var detOpts = source === 'upload' ? { method: 'POST', body: detBody } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: detBody };
              fetch('/detect', detOpts)
                .then(function(r) { return r.json(); })
                .then(function(d) {
                  if (d.error) results.push('Step ' + (idx + 1) + ': ' + d.error);
                  else {
                    pipelineLastFrame = d.frame_base64;
                    pipelineLastDetections = d.detections || [];
                    results.push('Step ' + (idx + 1) + ': Plotted ' + (pipelineLastDetections.length) + ' detection(s) on frame.');
                  }
                  idx++; next();
                })
                .catch(function(e) { results.push('Step ' + (idx + 1) + ': Error ' + e.message); idx++; next(); });
              return;
            }
            results.push('Step ' + (idx + 1) + ': No frame yet (run a question or Get coordinates step first).');
            idx++; next();
          } else {
            idx++; next();
          }
        }
        next();
      }
      pipelineRunBtn.addEventListener('click', runPipeline);

      var plotlyFallbackImg = document.getElementById('plotlyFallbackImg');
      function renderPlotlyFrame() {
        if (!lastYtFrameBase64) return;
        var imgSrc = 'data:image/jpeg;base64,' + lastYtFrameBase64;
        if (typeof Plotly === 'undefined') {
          plotlyFrame.classList.add('hidden');
          if (plotlyFallbackImg) { plotlyFallbackImg.src = imgSrc; plotlyFallbackImg.classList.remove('hidden'); }
          return;
        }
        var imgEl = new Image();
        imgEl.onerror = function() {
          console.error('Frame image failed to load');
          if (plotlyFallbackImg) { plotlyFallbackImg.src = imgSrc; plotlyFallbackImg.classList.remove('hidden'); }
          plotlyFrame.classList.add('hidden');
        };
        imgEl.onload = function() {
          var w = imgEl.naturalWidth;
          var h = imgEl.naturalHeight;
          var maxDim = 1200;
          var drawW = w;
          var drawH = h;
          if (w > maxDim || h > maxDim) {
            var s = Math.min(maxDim / w, maxDim / h);
            drawW = Math.round(w * s);
            drawH = Math.round(h * s);
          }
          var canvas = document.createElement('canvas');
          canvas.width = drawW;
          canvas.height = drawH;
          var ctx = canvas.getContext('2d');
          ctx.drawImage(imgEl, 0, 0, drawW, drawH);
          if (lastYtDetections && lastYtDetections.length > 0) {
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = Math.max(2, drawW / 400);
            ctx.font = 'bold ' + Math.max(11, drawW / 55) + 'px sans-serif';
            lastYtDetections.forEach(function(d) {
              var b = d.box;
              if (!b || b.length < 4) return;
              var x0 = b[0] * drawW, y0 = b[1] * drawH, x1 = b[2] * drawW, y1 = b[3] * drawH;
              ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
              var label = (d.label || '') + (d.score != null ? ' ' + Math.round(d.score * 100) + '%' : '');
              var tw = ctx.measureText(label).width;
              ctx.fillStyle = 'rgba(0,255,136,0.9)';
              ctx.fillRect(x0, y0 - 20, tw + 8, 20);
              ctx.fillStyle = '#000';
              ctx.fillText(label, x0 + 4, y0 - 5);
            });
          }
          var plotImgSrc = canvas.toDataURL('image/jpeg', 0.9);
          var container = plotlyFrame.parentElement;
          var cw = container ? container.clientWidth : 800;
          var ch = container ? container.clientHeight : 600;
          cw = Math.min(cw, window.innerWidth || 1920);
          ch = Math.min(ch, window.innerHeight || 1080);
          var layout = {
            margin: { l: 0, r: 0, t: 0, b: 0 },
            width: cw,
            height: ch,
            autosize: true,
            uirevision: 'frame',
            xaxis: { range: [0, drawW], domain: [0, 1], showgrid: false, zeroline: false, visible: false, constrain: 'domain' },
            yaxis: { range: [drawH, 0], domain: [0, 1], showgrid: false, zeroline: false, visible: false, scaleanchor: 'x', scaleratio: drawH / drawW, constrain: 'domain' },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: '#0f0f1a',
            images: [{
              source: plotImgSrc,
              xref: 'x', yref: 'y',
              x: 0, y: drawH, sizex: drawW, sizey: drawH,
              xanchor: 'left', yanchor: 'top',
              layer: 'below',
              sizing: 'stretch'
            }],
            shapes: [],
            annotations: []
          };
          if (overlayAnswerCheck.checked && lastYtAnswer) {
            layout.annotations.push({
              x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
              text: 'Answer: ' + lastYtAnswer,
              showarrow: false,
              font: { size: 14, color: '#fff' },
              bgcolor: 'rgba(0,0,0,0.7)',
              borderpad: 8,
              xanchor: 'left', yanchor: 'top'
            });
          }
          var data = [{ x: [0, drawW], y: [0, drawH], mode: 'markers', marker: { size: 0, opacity: 0 }, showlegend: false }];
          try {
            Plotly.newPlot(plotlyFrame, data, layout, {
              responsive: true,
              scrollZoom: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              displaylogo: false,
              toImageButtonOptions: { format: 'png', filename: 'frame_annotated', height: drawH, width: drawW }
            });
            if (plotlyFallbackImg) plotlyFallbackImg.classList.add('hidden');
            plotlyFrame.classList.remove('hidden');
          } catch (e) {
            console.error('Plotly error:', e);
            if (plotlyFallbackImg) { plotlyFallbackImg.src = imgSrc; plotlyFallbackImg.classList.remove('hidden'); }
            plotlyFrame.classList.add('hidden');
          }
        };
        imgEl.src = imgSrc;
      }
      askBtn.addEventListener('click', function() {
        var q = question.value.trim();
        if (!q) return;
        askBtn.disabled = true;
        result.textContent = 'Asking…';
        result.classList.add('has-content');
        result.classList.remove('error');

        if (source === 'youtube') {
          fetch('/ask_youtube', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              url: ytUrl.value.trim(),
              time_sec: parseFloat(ytTime.value) || 0,
              question: q
            })
          })
            .then(function(r) { return r.json(); })
            .then(function(d) {
              showResult(d.error || d.answer || '', !!d.error);
              if (!d.error && d.frame_base64) {
                lastYtFrameBase64 = d.frame_base64;
                lastYtAnswer = d.answer || null;
                lastYtDetections = isAnswerPositive(d.answer) ? (d.detections || []) : [];
                ytFramePlaceholder.classList.add('hidden');
                plotlyContainer.classList.remove('hidden');
                ytFrameOverlayOpts.classList.remove('hidden');
                if (plotlyFallbackImg) {
                  plotlyFallbackImg.src = 'data:image/jpeg;base64,' + d.frame_base64;
                  plotlyFallbackImg.classList.remove('hidden');
                }
                plotlyFrame.classList.add('hidden');
              } else {
                lastYtFrameBase64 = null;
                lastYtAnswer = null;
                lastYtDetections = [];
                ytFramePlaceholder.classList.remove('hidden');
                plotlyContainer.classList.add('hidden');
                ytFrameOverlayOpts.classList.add('hidden');
              }
            })
            .catch(function(e) {
              showResult('Error: ' + e.message, true);
            })
            .finally(function() { askBtn.disabled = false; updateAskBtn(); });
          return;
        }

        var body;
        if (source === 'upload') {
          var fd = new FormData();
          fd.append('image', file.files[0]);
          fd.append('question', q);
          body = fd;
        } else {
          body = JSON.stringify({
            image_base64: capturedDataUrl.split(',')[1],
            question: q
          });
          body = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body };
          fetch('/ask', body)
            .then(function(r) { return r.json(); })
            .then(function(d) {
              showResult(d.error || d.answer || '', !!d.error);
            })
            .catch(function(e) {
              showResult('Error: ' + e.message, true);
            })
            .finally(function() { askBtn.disabled = false; updateAskBtn(); });
          return;
        }
        fetch('/ask', { method: 'POST', body: body })
          .then(function(r) { return r.json(); })
          .then(function(d) {
            showResult(d.error || d.answer || '', !!d.error);
          })
          .catch(function(e) {
            showResult('Error: ' + e.message, true);
          })
          .finally(function() { askBtn.disabled = false; updateAskBtn(); });
      });

      document.getElementById('btnViewAttention').addEventListener('click', function() {
        var q = question.value.trim();
        if (!q) {
          document.getElementById('attentionStatus').textContent = 'Enter a question first.';
          return;
        }
        function doSend(imgB64) {
          if (!imgB64) {
            document.getElementById('attentionStatus').textContent = 'Select an image (upload, capture webcam, or Ask on YouTube first).';
            return;
          }
          document.getElementById('attentionStatus').textContent = 'Running with attention…';
          fetch('/run_attention', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: imgB64, question: q })
          })
            .then(function(r) { return r.json(); })
            .then(function(d) {
              document.getElementById('attentionStatus').textContent = '';
              if (d.error) {
                document.getElementById('attentionStatus').textContent = d.error;
                return;
              }
              window.open('/attention?from_8087=1', '_blank');
            })
            .catch(function(e) {
              document.getElementById('attentionStatus').textContent = 'Error: ' + e.message;
            });
        }
        if (source === 'youtube' && lastYtFrameBase64) {
          doSend(lastYtFrameBase64);
        } else if (source === 'webcam' && capturedDataUrl) {
          doSend(capturedDataUrl.split(',')[1]);
        } else if (source === 'upload' && file.files && file.files[0]) {
          var fr = new FileReader();
          fr.onload = function() {
            var dataUrl = fr.result;
            doSend(dataUrl.indexOf(',') >= 0 ? dataUrl.split(',')[1] : dataUrl);
          };
          fr.readAsDataURL(file.files[0]);
        } else {
          doSend(null);
        }
      });

      function checkReady() {
        fetch('/health').then(function(r) { return r.json(); }).then(function(d) {
          if (d.model_loaded) {
            document.getElementById('loadOverlay').classList.add('hidden');
            document.getElementById('mainContent').classList.add('ready');
          } else {
            document.getElementById('loadPct').textContent = (d.loading_progress || 0) + '%';
            setTimeout(checkReady, 1000);
          }
        }).catch(function() { setTimeout(checkReady, 1000); });
      }
      checkReady();
    })();
  </script>
</body>
</html>
"""

ATTENTION_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>VLM Attention Map</title>
<style>
body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#e0e0e0;margin:16px;}
.panel{background:#2a2a2a;padding:12px;border-radius:8px;margin-bottom:12px;}
button{padding:8px 14px;background:#0a7;color:#fff;border:none;border-radius:6px;cursor:pointer;}
button.secondary{background:#555;}
.tokens span{display:inline;padding:2px 4px;margin:1px;cursor:pointer;border-radius:3px;}
.tokens span:hover{background:#444}.tokens span.selected{background:#0a7;}
#imageContainer{position:relative;display:inline-block;}
#preview{display:block;max-width:100%;height:auto;}
#heatmapOverlay{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:fill;pointer-events:none;opacity:0.75;}
</style>
</head>
<body>
<h1>VLM Attention Map</h1>
<p style="color:#888">Uses the same model as the QA service (no extra load). <strong id="currentModelLabel">Current model:</strong> <span id="currentModelName" style="color:#8ac">—</span>. After running, click a word (SmolVLM2) or use CLS (DINOv3) to see where the model looked.</p>
  <div class="panel" style="font-size:0.85rem;color:#aaa">
  <strong>Layer &amp; Head:</strong> <strong>Layer</strong> = depth (early = low-level, later = semantic). <strong>-1</strong> = mean over all layers; <strong>-2</strong> = weighted average (later layers count more); <strong>-3</strong> = trimmed mean (drop top/bottom 25%%). Same for <strong>Head</strong>. <strong>Token</strong> = the word you clicked; the heatmap shows where the model looked.
</div>
<div class="panel">
  <label>Image</label><input type="file" id="file" accept="image/*"><br>
  <label>Question</label><input type="text" id="question" placeholder="e.g. Where is the white truck?" style="width:320px;margin-top:4px"><br>
  <button id="runBtn" style="margin-top:8px">Run</button><span id="runStatus" style="margin-left:8px;color:#888"></span>
</div>
<div class="panel" id="resultPanel" style="display:none">
  <strong>Answer:</strong> <span id="answer"></span>
  <div id="noAttnMsg" style="display:none;color:#888">Run with an image to get attention map (DINOv3: image only; SmolVLM2: image + question).</div>
  <div id="dinov3Hint" style="display:none;color:#8ac;font-size:0.85rem;margin-top:4px">DINOv3: CLS→patch attention (see <a href="https://github.com/mselmangokmen/Dinov3-attention-map-extraction" target="_blank">reference</a>). Layer = -1 (last) or 0..N-1, Head = -1 (mean) or 0..N-1.</div>
  <div class="tokens" id="tokens"></div>
  <div id="mapControls" style="margin-top:12px">
    Token index <input type="number" id="tokenIdx" min="0" value="0" style="width:50px" title="Which word (0=first)">
    Layer <input type="number" id="layer" min="-3" value="-1" style="width:50px" title="-3 trimmed, -2 weighted, -1 mean, 0..N single layer">
    Head <input type="number" id="head" min="-3" value="-1" style="width:50px" title="-3 trimmed, -2 weighted, -1 mean, 0..N single head">
    Threshold <input type="number" id="threshold" min="0" max="1" step="0.05" value="0" style="width:50px">
    Percentile <input type="number" id="percentile" min="50" max="99" value="80" style="width:50px" title="For bboxes: keep pixels above this percentile">
    <button id="mapBtn" class="secondary">Show attention map</button>
    <button id="extractBtn" class="secondary" style="margin-left:8px">Extract &amp; plot only this region</button>
    <button id="bboxBtn" class="secondary" style="margin-left:8px">Tag subject &amp; bboxes</button>
    <span id="mapExtractStatus" style="color:#c66;margin-left:8px;font-size:0.9rem;"></span>
  </div>
</div>
<div class="panel" id="imagePanel" style="display:none">
  <div id="imageContainer"><img id="preview" alt="preview"><img id="heatmapOverlay" alt="heatmap" style="display:none"></div>
  <div id="distributionPanel" style="margin-top:12px;display:none">
    <strong>Normalized attention distribution</strong> (as you flip layers/heads):<br>
    <img id="distributionHistogram" alt="distribution" style="max-width:100%;margin-top:4px;">
    <div id="distributionStats" style="font-size:0.8rem;color:#aaa;margin-top:4px;"></div>
  </div>
</div>
<div class="panel" id="extractPanel" style="display:none">
  <strong>Extracted region</strong> (attended area for the selected token, e.g. the white truck):<br>
  <img id="extractedImg" alt="extracted" style="max-width:100%;margin-top:8px;border:2px solid #0a7;">
</div>
<div class="panel" id="bboxPanel" style="display:none">
  <strong>Tagged subject &amp; bounding boxes</strong> (from attention heatmap):<br>
  <img id="bboxImg" alt="image with bboxes" style="max-width:100%;margin-top:8px;border:2px solid #0a7;">
</div>
<script>
var lastTokens=[],lastNumLayers=0,lastNumHeads=0;
function applyLastResult(d){
  if(!d||d.error)return;
  lastTokens=d.tokens||[];lastNumLayers=d.num_layers||0;lastNumHeads=d.num_heads||0;
  document.getElementById('answer').textContent=d.answer||'';
  var attnOk=d.attention_available&&lastNumLayers>0&&lastNumHeads>0;
  document.getElementById('mapControls').style.display='block';
  document.getElementById('noAttnMsg').style.display=attnOk?'none':'block';
  document.getElementById('dinov3Hint').style.display=(d.dinov3&&attnOk)?'block':'none';
  document.getElementById('mapExtractStatus').textContent='';
  var tokensEl=document.getElementById('tokens');tokensEl.innerHTML='';
  lastTokens.forEach(function(t,i){
    var s=document.createElement('span');s.textContent=t;s.dataset.index=i;
    s.onclick=function(){document.getElementById('tokenIdx').value=this.dataset.index;document.querySelectorAll('.tokens span.selected').forEach(function(x){x.classList.remove('selected');});this.classList.add('selected');};
    tokensEl.appendChild(s);
  });
  document.getElementById('tokenIdx').max=Math.max(0,lastTokens.length-1);
  document.getElementById('layer').max=Math.max(0,lastNumLayers-1);
  document.getElementById('head').max=Math.max(0,lastNumHeads-1);
  if(d.dinov3){document.getElementById('layer').min=-1; document.getElementById('layer').value=-1; document.getElementById('layer').title='DINOv3: -1 = last layer (recommended), 0..N-1';} else {document.getElementById('layer').min=-3;}
  document.getElementById('resultPanel').style.display='block';
  if(d.image_base64){document.getElementById('preview').src='data:image/png;base64,'+d.image_base64;document.getElementById('imagePanel').style.display='block';}
}
function parseLayerHead(elId, defaultVal){var v=document.getElementById(elId).value;var n=parseInt(v,10);return (v===''||isNaN(n))?defaultVal:n;}
document.getElementById('runBtn').onclick=function(){
  if(!document.getElementById('file').files.length){document.getElementById('runStatus').textContent='Select an image.';return;}
  var q=document.getElementById('question').value.trim();if(!q){document.getElementById('runStatus').textContent='Enter a question.';return;}
  document.getElementById('runStatus').textContent='Running…';this.disabled=true;
  var fd=new FormData();fd.append('image',document.getElementById('file').files[0]);fd.append('question',q);
  fetch('/run_attention',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(d){
    document.getElementById('runBtn').disabled=false;
    if(d.error){document.getElementById('runStatus').textContent=d.error;return;}
    document.getElementById('runStatus').textContent='Done.';applyLastResult(d);
  }).catch(function(e){document.getElementById('runBtn').disabled=false;document.getElementById('runStatus').textContent='Error: '+e.message;});
};
document.getElementById('mapBtn').onclick=function(){
  var statusEl=document.getElementById('mapExtractStatus');
  statusEl.textContent='';
  var tokenIdx=parseInt(document.getElementById('tokenIdx').value,10)||0,layer=parseLayerHead('layer',-1),head=parseLayerHead('head',-1),threshold=parseFloat(document.getElementById('threshold').value)||0;
  fetch('/attention_map?token_idx='+tokenIdx+'&layer='+layer+'&head='+head+'&threshold='+threshold).then(function(r){if(!r.ok)return r.text().then(function(t){throw new Error(t||r.status);});return r.json();}).then(function(d){
    if(d.error){document.getElementById('heatmapOverlay').style.display='none';document.getElementById('distributionPanel').style.display='none';statusEl.textContent=d.error;return;}
    var ov=document.getElementById('heatmapOverlay');
    ov.src='data:image/png;base64,'+d.heatmap;
    ov.style.display='block';
    var pr=document.getElementById('preview');
    ov.style.width=pr.offsetWidth+'px';ov.style.height=pr.offsetHeight+'px';
    var distPanel=document.getElementById('distributionPanel');
    if(d.distribution_histogram){document.getElementById('distributionHistogram').src='data:image/png;base64,'+d.distribution_histogram;distPanel.style.display='block';}else{distPanel.style.display='none';}
    var statsEl=document.getElementById('distributionStats');
    if(d.distribution){var s=d.distribution;statsEl.textContent='min='+(s.min!=null?s.min.toExponential(2):'')+' max='+(s.max!=null?s.max.toExponential(2):'')+' p50='+(s.p50!=null?s.p50.toExponential(2):'')+' p95='+(s.p95!=null?s.p95.toExponential(2):'');statsEl.style.display='block';}else{statsEl.style.display='none';}
  }).catch(function(e){document.getElementById('heatmapOverlay').style.display='none';statusEl.textContent='Network error: '+e.message;});
};
document.getElementById('extractBtn').onclick=function(){
  var statusEl=document.getElementById('mapExtractStatus');
  statusEl.textContent='';
  var tokenIdx=parseInt(document.getElementById('tokenIdx').value,10)||0,layer=parseLayerHead('layer',-1),head=parseLayerHead('head',-1),threshold=parseFloat(document.getElementById('threshold').value)||0;
  fetch('/extract_attended_region?token_idx='+tokenIdx+'&layer='+layer+'&head='+head+'&threshold='+threshold).then(function(r){if(!r.ok)return r.text().then(function(t){throw new Error(t||r.status);});return r.json();}).then(function(d){
    if(d.error){document.getElementById('extractPanel').style.display='none';statusEl.textContent=d.error;return;}
    document.getElementById('extractedImg').src='data:image/png;base64,'+d.image_base64;
    document.getElementById('extractPanel').style.display='block';
  }).catch(function(e){document.getElementById('extractPanel').style.display='none';statusEl.textContent='Network error: '+e.message;});
};
document.getElementById('bboxBtn').onclick=function(){
  var tokenIdx=parseInt(document.getElementById('tokenIdx').value,10)||0,layer=parseLayerHead('layer',-1),head=parseLayerHead('head',-1),threshold=parseFloat(document.getElementById('threshold').value)||0,percentile=parseFloat(document.getElementById('percentile').value)||80;
  fetch('/attention_bboxes?token_idx='+tokenIdx+'&layer='+layer+'&head='+head+'&threshold='+threshold+'&percentile='+percentile).then(function(r){return r.json();}).then(function(d){
    if(d.error){document.getElementById('bboxPanel').style.display='none';return;}
    document.getElementById('bboxImg').src='data:image/png;base64,'+d.image_with_boxes_base64;
    document.getElementById('bboxPanel').style.display='block';
  }).catch(function(){document.getElementById('bboxPanel').style.display='none';});
};
function setCurrentModel(name){var el=document.getElementById('currentModelName');if(el)el.textContent=name||'—';}
if(window.location.search.indexOf('from_8087=1')>=0){fetch('/last_attention_result').then(function(r){return r.json();}).then(function(d){setCurrentModel(d.model_name);applyLastResult(d);}).catch(function(){});}
fetch('/info').then(function(r){return r.json();}).then(function(d){setCurrentModel(d.model);}).catch(function(){});
</script>
</body>
</html>
"""


class VLMQAHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_html(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _handle_run_attention(self):
        """POST /run_attention: image (+ question for SmolVLM), run with output_attentions; store and return. Supports SmolVLM2 and DINOv3."""
        global _last_attention_run
        if self.server.model is None:
            self.send_json({"error": "Model not loaded yet"}, 503)
            return
        model_name = getattr(self.server, "model_name", "")
        content_type = self.headers.get("Content-Type", "")
        image = None
        question = ""
        try:
            if "application/json" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                data = json.loads(body)
                question = (data.get("question") or "").strip()
                b64 = data.get("image_base64") or data.get("image") or ""
                if not b64:
                    self.send_json({"error": "Missing image_base64"}, 400)
                    return
                if "," in str(b64):
                    b64 = str(b64).split(",", 1)[1]
                raw = base64.b64decode(b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                form = cgi.FieldStorage(fp=self.rfile, environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                })
                if "question" in form:
                    question = (form["question"].value or "").strip()
                if "image" in form:
                    fp = form["image"].file
                    image = Image.open(fp).convert("RGB")
                if image is None:
                    self.send_json({"error": "Missing image file"}, 400)
                    return
            if model_name == "dinov3":
                attn_list, num_layers, num_heads, num_register_tokens = run_dinov3_with_attentions(
                    self.server.model, self.server.processor, image, self.server.device
                )
                answer = "DINOv3: CLS→patch attention extracted. Use Layer/Head to view heatmap."
                tokens = ["CLS"]
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                _last_attention_run = {
                    "attn": attn_list,
                    "tokens": tokens,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "answer": answer,
                    "image_base64": img_b64,
                    "dinov3": True,
                    "num_register_tokens": num_register_tokens,
                }
                self.send_json({
                    "answer": answer,
                    "tokens": tokens,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "image_base64": img_b64,
                    "attention_available": attn_list is not None,
                })
                return
            if model_name not in ("smolvlm2-2.2b", "smolvlm2-500m"):
                self.send_json({"error": "Attention map only supported for SmolVLM2 or DINOv3. Current model: %s" % model_name}, 400)
                return
            if not question:
                self.send_json({"error": "Missing question (required for SmolVLM2)"}, 400)
                return
            answer, tokens, attn_list, num_layers, num_heads = run_with_attentions_smolvlm(
                self.server.model, self.server.processor, image, question, self.server.device
            )
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            _last_attention_run = {
                "attn": attn_list,
                "tokens": tokens,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "answer": answer,
                "image_base64": img_b64,
            }
            self.send_json({
                "answer": answer,
                "tokens": tokens,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "image_base64": img_b64,
                "attention_available": attn_list is not None,
            })
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def _handle_detect(self):
        """POST /detect: image (multipart or JSON image_base64) -> { frame_base64, detections }."""
        content_type = self.headers.get("Content-Type", "")
        image = None
        try:
            if "application/json" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                data = json.loads(body)
                b64 = data.get("image_base64") or data.get("image") or ""
                if not b64:
                    self.send_json({"error": "Missing image_base64", "frame_base64": None, "detections": []}, status=400)
                    return
                if "," in str(b64):
                    b64 = str(b64).split(",", 1)[1]
                raw = base64.b64decode(b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                form = cgi.FieldStorage(fp=self.rfile, environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                })
                if "image" in form:
                    fp = form["image"].file
                    image = Image.open(fp).convert("RGB")
                if image is None:
                    self.send_json({"error": "Missing image file", "frame_base64": None, "detections": []}, status=400)
                    return
            detections = run_object_detection(image)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            self.send_json({"frame_base64": frame_b64, "detections": detections})
        except Exception as e:
            self.send_json({"error": str(e), "frame_base64": None, "detections": []}, status=400)

    def _handle_detect_youtube(self):
        """POST /detect_youtube: JSON { url, time_sec } -> { frame_base64, detections }. No VLM, detection only."""
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            self.send_json({"error": "Missing body", "frame_base64": None, "detections": []}, status=400)
            return
        try:
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            url = (data.get("url") or "").strip()
            time_sec = float(data.get("time_sec", 0))
            if not url:
                self.send_json({"error": "Missing url", "frame_base64": None, "detections": []}, status=400)
                return
            image = extract_one_frame_youtube(url, time_sec)
            if image is None:
                self.send_json({"error": "Could not extract frame from YouTube", "frame_base64": None, "detections": []}, status=400)
                return
            detections = run_object_detection(image)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            self.send_json({"frame_base64": frame_b64, "detections": detections})
        except Exception as e:
            self.send_json({"error": str(e), "frame_base64": None, "detections": []}, status=400)

    def _handle_ask_youtube(self):
        if self.server.model is None:
            self.send_json({"error": "Model not loaded yet", "answer": None}, status=503)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            self.send_json({"error": "Missing body", "answer": None}, status=400)
            return
        try:
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            url = (data.get("url") or "").strip()
            time_sec = float(data.get("time_sec", 0))
            question = (data.get("question") or "").strip()
            if not url or not question:
                self.send_json({"error": "Missing url or question", "answer": None}, status=400)
                return
            image = extract_one_frame_youtube(url, time_sec)
            if image is None:
                self.send_json({"error": "Could not extract frame from YouTube (check URL and time)", "answer": None}, status=400)
                return
            ask_fn = self.server.ask_fn
            device = self.server.device
            answer = ask_fn(self.server.model, self.server.processor, image, question, device)
            detections = run_object_detection(image)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            self.send_json({"answer": answer, "frame_base64": frame_b64, "detections": detections})
        except json.JSONDecodeError as e:
            self.send_json({"error": "Invalid JSON: " + str(e), "answer": None}, status=400)
        except Exception as e:
            self.send_json({"error": str(e), "answer": None}, status=400)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            loaded = self.server.model is not None
            pct = getattr(self.server, "loading_progress", 0)
            self.send_json({
                "status": "ok",
                "model_loaded": loaded,
                "loading_progress": 100 if loaded else pct,
                "detector_loaded": _detector is not None,
            })
            return
        if path == "/info":
            self.send_json({
                "service": "VLM QA (SmolVLM2 or LLaVA 7B)",
                "model": getattr(self.server, "model_name", "?"),
                "endpoints": {
                    "GET /": "Web UI: upload, webcam, or YouTube — ask a question",
                    "GET /attention": "Attention map UI (same model, no extra load)",
                    "GET /health": "Health / loading progress",
                    "POST /ask": "Body: multipart (image file + question) or JSON { \"image_base64\": \"...\", \"question\": \"...\" }. Returns { \"answer\": \"...\" }.",
                    "POST /ask_youtube": "Body: JSON { \"url\": \"YouTube URL\", \"time_sec\": 0, \"question\": \"...\" }. Returns { \"answer\": \"...\" }.",
                    "POST /run_attention": "Image + (question for SmolVLM2). Runs with output_attentions. SmolVLM2 or DINOv3 (CLS→patch). Returns answer, tokens, attention data.",
                    "POST /dinov3_features": "When --model dinov3: image (multipart or JSON image_base64) -> DINOv3 feature shapes and stats.",
                    "POST /detect": "Body: multipart (image file) or JSON { \"image_base64\": \"...\" }. Returns { \"frame_base64\": \"...\", \"detections\": [...] }.",
                    "POST /detect_youtube": "Body: JSON { \"url\": \"YouTube URL\", \"time_sec\": 0 }. Returns { \"frame_base64\": \"...\", \"detections\": [...] } (no VLM).",
                },
                "port": self.server.server_port,
            })
            return
        if path == "/":
            self.send_html(HTML_PAGE)
            return
        if path == "/attention":
            self.send_html(ATTENTION_HTML_PAGE)
            return
        if path == "/last_attention_result":
            run = _last_attention_run
            if run["attn"] is None and not run.get("tokens"):
                self.send_json({"error": "No run yet"}, 404)
                return
            self.send_json({
                "answer": run.get("answer", ""),
                "tokens": run.get("tokens", []),
                "num_layers": run.get("num_layers", 0),
                "num_heads": run.get("num_heads", 0),
                "image_base64": run.get("image_base64"),
                "attention_available": run.get("attn") is not None,
                "dinov3": run.get("dinov3", False),
                "model_name": getattr(self.server, "model_name", "?"),
            })
            return
        if path.startswith("/attention_map"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            token_idx = int(qs.get("token_idx", [0])[0])
            layer = int(qs.get("layer", [-1])[0])
            head = int(qs.get("head", [-1])[0])
            threshold = float(qs.get("threshold", [0])[0])
            run = _last_attention_run
            if run["attn"] is None:
                self.send_json({"error": "No attention data. Run image + question first."}, 400)
                return
            attn_list, num_layers, num_heads = run["attn"], run["num_layers"], run["num_heads"]
            if run.get("dinov3"):
                # DINOv3: CLS→patch attention; token_idx ignored. Layer: -1 = last layer (as in reference repo); head: -1 = mean over heads.
                layer_eff = (num_layers - 1) if layer == -1 else layer
                if layer_eff < 0 or layer_eff >= num_layers or head < -1 or head >= num_heads:
                    self.send_json({"error": "Invalid layer/head (DINOv3: layer -1 or 0..N-1, head -1 or 0..N-1)"}, 400)
                    return
                try:
                    img_raw = base64.b64decode(run["image_base64"])
                    img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                    H, W = img.size[1], img.size[0]
                    heatmap = dinov3_attention_to_heatmap(
                        attn_list, layer_eff, head, run.get("num_register_tokens", 4), H, W, threshold=threshold
                    )
                    if heatmap is None:
                        self.send_json({"error": "Could not build DINOv3 heatmap"}, 400)
                        return
                    b64 = heatmap_to_base64_png(heatmap, threshold=0.0, percentile_normalize=True)
                    self.send_json({"heatmap": b64})
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
                return
            if token_idx < 0 or token_idx >= len(attn_list) or layer < -3 or layer >= num_layers or head < -3 or head >= num_heads:
                self.send_json({"error": "Invalid token/layer/head (layer/head: -3 trimmed, -2 weighted, -1 mean, >=0 single)"}, 400)
                return
            try:
                heatmap = attention_to_heatmap(attn_list, token_idx, layer, head, IMAGE_TOKEN_COUNT, threshold=threshold)
                if heatmap is None:
                    self.send_json({"error": "Could not build heatmap (token_idx 0..%d, layer %d, head %d)" % (len(attn_list) - 1, layer, head)}, 400)
                    return
                raw = attention_raw_vector(attn_list, token_idx, layer, head, IMAGE_TOKEN_COUNT)
                stats = attention_distribution_stats(raw) if raw is not None else None
                hist_b64 = attention_histogram_base64(stats) if stats else None
                b64 = heatmap_to_base64_png(heatmap, threshold=threshold, percentile_normalize=True)
                out = {"heatmap": b64}
                if stats is not None:
                    out["distribution"] = {k: v for k, v in stats.items() if k != "bin_edges"}
                if hist_b64:
                    out["distribution_histogram"] = hist_b64
                self.send_json(out)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return
        if path.startswith("/extract_attended_region"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            token_idx = int(qs.get("token_idx", [0])[0])
            layer = int(qs.get("layer", [-1])[0])
            head = int(qs.get("head", [-1])[0])
            threshold = float(qs.get("threshold", [0])[0])
            percentile = float(qs.get("percentile", [80])[0])
            run = _last_attention_run
            if run["attn"] is None or not run.get("image_base64"):
                self.send_json({"error": "No run or image. Run image + question first."}, 400)
                return
            attn_list = run["attn"]
            num_layers, num_heads = run["num_layers"], run["num_heads"]
            if run.get("dinov3"):
                layer_eff = (num_layers - 1) if layer == -1 else layer
                if layer_eff < 0 or layer_eff >= num_layers or head < -1 or head >= num_heads:
                    self.send_json({"error": "Invalid layer/head (DINOv3)"}, 400)
                    return
                img_raw = base64.b64decode(run["image_base64"])
                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                H_img, W_img = img.size[1], img.size[0]
                heatmap = dinov3_attention_to_heatmap(attn_list, layer_eff, head, run.get("num_register_tokens", 4), H_img, W_img, threshold=threshold)
            else:
                if token_idx < 0 or token_idx >= len(attn_list) or layer < -3 or layer >= num_layers or head < -3 or head >= num_heads:
                    self.send_json({"error": "Invalid token/layer/head (layer/head: -3 trimmed, -2 weighted, -1 mean)"}, 400)
                    return
                heatmap = attention_to_heatmap(attn_list, token_idx, layer, head, IMAGE_TOKEN_COUNT, threshold=threshold)
            try:
                if heatmap is None:
                    self.send_json({"error": "Could not build heatmap"}, 400)
                    return
                bbox = heatmap_to_bbox(heatmap, percentile=percentile)
                if bbox is None:
                    self.send_json({"error": "No hot region found"}, 400)
                    return
                x_hm_min, y_hm_min, x_hm_max, y_hm_max = bbox
                H_hm, W_hm = heatmap.shape[0], heatmap.shape[1]
                img_raw = base64.b64decode(run["image_base64"])
                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                W_img, H_img = img.size[0], img.size[1]
                x_min = int(x_hm_min * W_img / W_hm)
                x_max = int(x_hm_max * W_img / W_hm)
                y_min = int(y_hm_min * H_img / H_hm)
                y_max = int(y_hm_max * H_img / H_hm)
                x_min, x_max = max(0, x_min), min(W_img, x_max)
                y_min, y_max = max(0, y_min), min(H_img, y_max)
                if x_max <= x_min or y_max <= y_min:
                    self.send_json({"error": "Empty crop"}, 400)
                    return
                cropped = img.crop((x_min, y_min, x_max, y_max))
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                self.send_json({"image_base64": b64, "bbox": [x_min, y_min, x_max, y_max]})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return
        if path.startswith("/attention_bboxes"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            token_idx = int(qs.get("token_idx", [0])[0])
            layer = int(qs.get("layer", [-1])[0])
            head = int(qs.get("head", [-1])[0])
            threshold = float(qs.get("threshold", [0])[0])
            percentile = float(qs.get("percentile", [80])[0])
            run = _last_attention_run
            if run["attn"] is None or not run.get("image_base64"):
                self.send_json({"error": "No run or image. Run image + question first."}, 400)
                return
            attn_list = run["attn"]
            num_layers, num_heads = run["num_layers"], run["num_heads"]
            if run.get("dinov3"):
                layer_eff = (num_layers - 1) if layer == -1 else layer
                if layer_eff < 0 or layer_eff >= num_layers or head < -1 or head >= num_heads:
                    self.send_json({"error": "Invalid layer/head (DINOv3)"}, 400)
                    return
                img_raw = base64.b64decode(run["image_base64"])
                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                H_hm, W_hm = img.size[1], img.size[0]
                heatmap = dinov3_attention_to_heatmap(attn_list, layer_eff, head, run.get("num_register_tokens", 4), H_hm, W_hm, threshold=threshold)
            else:
                if token_idx < 0 or token_idx >= len(attn_list) or layer < -3 or layer >= num_layers or head < -3 or head >= num_heads:
                    self.send_json({"error": "Invalid token/layer/head (layer/head: -3 trimmed, -2 weighted, -1 mean)"}, 400)
                    return
                heatmap = attention_to_heatmap(attn_list, token_idx, layer, head, IMAGE_TOKEN_COUNT, threshold=threshold)
            try:
                if heatmap is None:
                    self.send_json({"error": "Could not build heatmap"}, 400)
                    return
                bboxes_hm = heatmap_to_bboxes(heatmap, percentile=percentile)
                H_hm, W_hm = heatmap.shape[0], heatmap.shape[1]
                img_raw = base64.b64decode(run["image_base64"])
                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                W_img, H_img = img.size[0], img.size[1]
                bboxes_img = []
                for (x_hm_min, y_hm_min, x_hm_max, y_hm_max) in bboxes_hm:
                    x_min = max(0, int(x_hm_min * W_img / W_hm))
                    x_max = min(W_img, int(x_hm_max * W_img / W_hm))
                    y_min = max(0, int(y_hm_min * H_img / H_hm))
                    y_max = min(H_img, int(y_hm_max * H_img / H_hm))
                    if x_max > x_min and y_max > y_min:
                        bboxes_img.append([x_min, y_min, x_max, y_max])
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                for (x_min, y_min, x_max, y_max) in bboxes_img:
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=max(2, min(W_img, H_img) // 200))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                self.send_json({"bboxes": bboxes_img, "image_with_boxes_base64": b64})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return
        self.send_response(404)
        self.end_headers()

    def _handle_dinov3_features(self):
        """POST /dinov3_features: image (multipart or JSON image_base64) -> DINOv3 feature shapes and stats."""
        if getattr(self.server, "model_name", "") != "dinov3" or self.server.model is None:
            self.send_json({"error": "DINOv3 not loaded. Start server with --model dinov3."}, 400)
            return
        content_type = self.headers.get("Content-Type", "")
        image = None
        try:
            if "application/json" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                data = json.loads(body)
                b64 = data.get("image_base64") or data.get("image") or ""
                if not b64:
                    self.send_json({"error": "Missing image_base64"}, 400)
                    return
                if "," in str(b64):
                    b64 = str(b64).split(",", 1)[1]
                raw = base64.b64decode(b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                form = cgi.FieldStorage(fp=self.rfile, environ={
                    "REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                })
                if "image" in form:
                    image = Image.open(form["image"].file).convert("RGB")
            if image is None:
                self.send_json({"error": "Missing image"}, 400)
                return
            inp = self.server.processor(images=image, return_tensors="pt")
            inp = {k: (v.to(self.server.device) if hasattr(v, "to") else v) for k, v in inp.items()}
            with torch.no_grad():
                out = self.server.model(**inp)
            pooled = getattr(out, "pooler_output", None) or (out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else None)
            hidden = getattr(out, "last_hidden_state", None)
            result = {"model": "dinov3", "model_id": DINOV3_MODEL}
            if pooled is not None:
                p = pooled.cpu().numpy()
                result["pooler_output_shape"] = list(p.shape)
                result["pooler_output_mean"] = float(p.mean())
                result["pooler_output_std"] = float(p.std())
            if hidden is not None:
                result["last_hidden_state_shape"] = list(hidden.cpu().numpy().shape)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/run_attention":
            self._handle_run_attention()
            return
        if path == "/dinov3_features":
            self._handle_dinov3_features()
            return
        if path == "/ask_youtube":
            self._handle_ask_youtube()
            return
        if path == "/detect":
            self._handle_detect()
            return
        if path == "/detect_youtube":
            self._handle_detect_youtube()
            return
        if path != "/ask":
            self.send_response(404)
            self.end_headers()
            return
        if self.server.model is None:
            self.send_json({"error": "Model not loaded yet", "answer": None}, status=503)
            return
        content_type = self.headers.get("Content-Type", "")
        image = None
        question = ""
        try:
            if "application/json" in content_type:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                data = json.loads(body)
                question = (data.get("question") or "").strip()
                b64 = data.get("image_base64") or data.get("image") or ""
                need_question = getattr(self.server, "model_name", "") != "dinov3"
                if not b64 or (need_question and not question):
                    self.send_json({"error": "Missing image_base64" + (" or question" if need_question else ""), "answer": None}, status=400)
                    return
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                # multipart/form-data
                form = cgi.FieldStorage(fp=self.rfile, environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                })
                if "question" in form:
                    question = (form["question"].value or "").strip()
                if "image" in form:
                    fp = form["image"].file
                    image = Image.open(fp).convert("RGB")
                need_question = getattr(self.server, "model_name", "") != "dinov3"
                if image is None or (need_question and not question):
                    self.send_json({"error": "Missing image file" + (" or question" if need_question else ""), "answer": None}, status=400)
                    return
            ask_fn = self.server.ask_fn
            device = self.server.device
            answer = ask_fn(self.server.model, self.server.processor, image, question, device)
            self.send_json({"answer": answer})
        except Exception as e:
            self.send_json({"error": str(e), "answer": None}, status=400)


def main():
    parser = argparse.ArgumentParser(description="VLM QA server: image + question -> answer")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default %d)" % DEFAULT_PORT)
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3",
        choices=["smolvlm2-2.2b", "smolvlm2-500m", "llava-1.5-7b", "dinov3"],
        help="Model to load: VLM (smolvlm2, llava) or DINOv3 vision backbone (default: dinov3)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    server = HTTPServer(("0.0.0.0", args.port), VLMQAHandler)
    server.model = None
    server.processor = None
    server.device = device
    server.model_name = args.model
    server.ask_fn = None
    server.loading_progress = 0

    def run_server():
        server.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=False)
    server_thread.start()

    try:
        def set_progress(pct: int, msg: str):
            server.loading_progress = min(100, max(0, pct))
            print(f"[{pct}%] {msg}")
        if args.model == "llava-1.5-7b":
            server.model, server.processor = load_llava(device, progress_callback=set_progress)
            server.ask_fn = ask_llava
        elif args.model == "dinov3":
            server.model, server.processor = load_dinov3(device, progress_callback=set_progress)
            server.ask_fn = ask_dinov3
        else:
            name = SMOLVLM2_2B if args.model == "smolvlm2-2.2b" else SMOLVLM2_500M
            server.model, server.processor = load_smolvlm(name, device, progress_callback=set_progress)
            server.ask_fn = ask_smolvlm
        print("Loading object detector (DETR)...")
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        run_object_detection(dummy)
        if _detector is not None:
            print("Object detector ready.")
        else:
            print("Object detector not available (see stderr). Detection steps will return no boxes.", file=sys.stderr)
        print(f"VLM QA server: http://0.0.0.0:{args.port}/")
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        server.loading_progress = -1
        raise

    server_thread.join()


if __name__ == "__main__":
    main()
