#!/usr/bin/env python3
"""
VLM attention map server: same image + question as VLM QA, but runs the model with
output_attentions=True and serves an UI to visualize which image regions the model
attended to for each output token. Runs on a separate port (default 8088).
Supports SmolVLM2 only (same model as serve_vlm_qa).
"""

import argparse
import base64
import io
import json
import math
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
from PIL import Image

# Model (SmolVLM2 only for attention viz)
SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_PORT = 8088
MAX_NEW_TOKENS = 80

# SmolVLM2 image token layout (from Idefics/SmolVLM: 3x4 grid of 512x512 patches, 64 tokens per patch)
PATCH_ROWS = 3
PATCH_COLS = 4
TOKENS_PER_PATCH = 64
PATCH_PIXELS = 512
IMAGE_TOKEN_COUNT = PATCH_ROWS * PATCH_COLS * TOKENS_PER_PATCH  # 768


def load_model(device, progress_callback=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if progress_callback:
        progress_callback(10, "Loading processor…")
    processor = AutoProcessor.from_pretrained(SMOLVLM2_2B, trust_remote_code=True)
    if progress_callback:
        progress_callback(30, "Loading model…")
    model = AutoModelForImageTextToText.from_pretrained(
        SMOLVLM2_2B,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=str(device) if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if progress_callback:
        progress_callback(100, "Ready")
    return model, processor


def run_with_attentions(model, processor, image: Image.Image, question: str, device):
    """Run VLM and return answer, decoded tokens, and attention weights."""
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
    prompt_length = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    gen_ids = out.sequences[:, prompt_length:]
    decoded = processor.batch_decode(
        gen_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answer = (decoded[0] if decoded else "").strip()
    # Decode token-by-token for the list of tokens
    token_list = []
    for i in range(gen_ids.shape[1]):
        tok = processor.decode(gen_ids[0, i : i + 1], skip_special_tokens=False)
        token_list.append(tok if tok else "<unk>")

    # attentions: tuple of (step0, step1, ...); each step is tuple of (layer0, layer1, ...)
    # each layer (batch, num_heads, q_len, kv_len)
    if not getattr(out, "attentions", None) or not out.attentions:
        return answer, token_list, None, prompt_length, 0, 0

    num_steps = len(out.attentions)
    num_layers = len(out.attentions[0]) if num_steps else 0
    num_heads = out.attentions[0][0].shape[1] if num_layers else 0
    # Stack: [step][layer] -> (1, heads, 1, kv_len)
    # We want [num_steps, num_layers, num_heads, kv_len]
    attn_list = []
    for step in range(num_steps):
        layer_list = []
        for layer in range(num_layers):
            a = out.attentions[step][layer]  # (1, heads, 1, kv_len)
            a = a[0].float().cpu().numpy()   # (heads, 1, kv_len)
            layer_list.append(a)
        attn_list.append(layer_list)
    # attn_list[step][layer] = (num_heads, 1, kv_len)
    return answer, token_list, attn_list, prompt_length, num_layers, num_heads


def attention_to_heatmap(attn_step_layers, token_idx, layer_idx, head_idx, image_token_count, threshold=0.0):
    """
    Build a 2D heatmap over the image grid from attention of one token to image tokens.
    attn_step_layers[layer_idx] shape (num_heads, 1, kv_len). We take head_idx and slice 0:image_token_count.
    Map to PATCH_ROWS x PATCH_COLS grid (average per patch), then upsample to pixel grid.
    """
    if token_idx >= len(attn_step_layers) or layer_idx >= len(attn_step_layers[0]):
        return None
    # attn_step_layers is list of layers; each layer (num_heads, 1, kv_len)
    layer_attn = attn_step_layers[layer_idx]  # (num_heads, 1, kv_len)
    if head_idx >= layer_attn.shape[0]:
        return None
    attn = layer_attn[head_idx, 0, :].astype(np.float32)  # (kv_len,)
    n = min(image_token_count, attn.shape[0])
    attn_img = attn[:n]
    if threshold > 0:
        attn_img = np.where(attn_img >= threshold, attn_img, 0.0)
    # Reshape to patches: (PATCH_ROWS * PATCH_COLS, TOKENS_PER_PATCH) -> average per patch
    num_patches = PATCH_ROWS * PATCH_COLS
    if attn_img.size < num_patches * TOKENS_PER_PATCH:
        pad = num_patches * TOKENS_PER_PATCH - attn_img.size
        attn_img = np.pad(attn_img, (0, pad), constant_values=0.0)
    attn_img = attn_img[: num_patches * TOKENS_PER_PATCH].reshape(num_patches, TOKENS_PER_PATCH)
    patch_attn = attn_img.mean(axis=1)  # (num_patches,)
    grid = patch_attn.reshape(PATCH_ROWS, PATCH_COLS)
    # Upsample to pixel size (PATCH_PIXELS per patch) using repeat
    scale = PATCH_PIXELS
    heatmap = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
    return heatmap


def heatmap_to_base64(heatmap, image_size, threshold=0.0):
    """Normalize heatmap, overlay-style (0-1), return PNG base64."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    h, w = heatmap.shape
    if threshold > 0:
        heatmap = np.where(heatmap >= threshold, heatmap, 0.0)
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin > 1e-6:
        heatmap = (heatmap - vmin) / (vmax - vmin)
    else:
        heatmap = np.zeros_like(heatmap)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(heatmap, cmap="hot", alpha=0.7, interpolation="bilinear")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True, dpi=100)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64


# In-memory store for last run (one session)
_last_run = {"attn": None, "prompt_length": 0, "num_layers": 0, "num_heads": 0, "tokens": [], "image_base64": None}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VLM Attention Map</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #e0e0e0; margin: 16px; }
    h1 { font-size: 1.2rem; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
    .panel { background: #2a2a2a; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
    input[type="text"], input[type="number"] { padding: 6px; background: #333; border: 1px solid #555; color: #eee; border-radius: 4px; }
    button { padding: 8px 14px; background: #0a7; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.secondary { background: #555; }
    #imageContainer { position: relative; display: inline-block; }
    #imageContainer img { max-width: 100%; height: auto; display: block; }
    #heatmapOverlay { position: absolute; left: 0; top: 0; pointer-events: none; opacity: 0.8; }
    .tokens { margin-top: 8px; }
    .tokens span { display: inline; padding: 2px 4px; margin: 1px; cursor: pointer; border-radius: 3px; }
    .tokens span:hover { background: #444; }
    .tokens span.selected { background: #0a7; color: #fff; }
    label { display: inline-block; min-width: 90px; margin-right: 8px; }
    .status { color: #888; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>VLM Attention Map</h1>
  <p class="status">Upload an image and enter a question. After running, click a word to see where the model looked.</p>
  <div class="panel">
    <div>
      <label>Image</label>
      <input type="file" id="file" accept="image/*">
    </div>
    <div style="margin-top:8px">
      <label>Question</label>
      <input type="text" id="question" placeholder="e.g. What is in this image?" style="width:320px">
    </div>
    <button id="runBtn">Run</button>
    <span id="runStatus"></span>
  </div>
    <div class="panel" id="resultPanel" style="display:none">
    <div><strong>Answer:</strong> <span id="answer"></span></div>
    <div id="noAttnMsg" style="display:none; color:#888;">Attention data not available for this run.</div>
    <div class="tokens" id="tokens"></div>
    <div style="margin-top:12px" id="mapControls">
      <label>Token index</label><input type="number" id="tokenIdx" min="0" value="0" style="width:60px">
      <label>Layer</label><input type="number" id="layer" min="0" value="0" style="width:50px">
      <label>Head</label><input type="number" id="head" min="0" value="0" style="width:50px">
      <label>Threshold</label><input type="number" id="threshold" min="0" max="1" step="0.05" value="0" style="width:60px">
      <button id="mapBtn" class="secondary">Show attention map</button>
    </div>
  </div>
  <div class="panel" id="imagePanel" style="display:none">
    <div id="imageContainer">
      <img id="preview" alt="preview">
      <img id="heatmapOverlay" alt="heatmap" style="display:none">
    </div>
  </div>

  <script>
    var file = document.getElementById('file');
    var question = document.getElementById('question');
    var runBtn = document.getElementById('runBtn');
    var runStatus = document.getElementById('runStatus');
    var resultPanel = document.getElementById('resultPanel');
    var answerEl = document.getElementById('answer');
    var tokensEl = document.getElementById('tokens');
    var tokenIdxEl = document.getElementById('tokenIdx');
    var layerEl = document.getElementById('layer');
    var headEl = document.getElementById('head');
    var thresholdEl = document.getElementById('threshold');
    var mapBtn = document.getElementById('mapBtn');
    var imagePanel = document.getElementById('imagePanel');
    var preview = document.getElementById('preview');
    var heatmapOverlay = document.getElementById('heatmapOverlay');

    var lastTokens = [];
    var lastNumLayers = 0;
    var lastNumHeads = 0;

    function applyLastResult(d) {
      if (!d || d.error) return;
      lastTokens = d.tokens || [];
      lastNumLayers = d.num_layers || 0;
      lastNumHeads = d.num_heads || 0;
      answerEl.textContent = d.answer || '';
      var attnOk = d.attention_available && lastNumLayers > 0 && lastNumHeads > 0;
      document.getElementById('mapControls').style.display = attnOk ? '' : 'none';
      document.getElementById('noAttnMsg').style.display = attnOk ? 'none' : 'block';
      tokensEl.innerHTML = '';
      lastTokens.forEach(function(t, i) {
        var s = document.createElement('span');
        s.textContent = t;
        s.dataset.index = i;
        s.addEventListener('click', function() {
          tokenIdxEl.value = this.dataset.index;
          document.querySelectorAll('.tokens span.selected').forEach(function(x) { x.classList.remove('selected'); });
          this.classList.add('selected');
        });
        tokensEl.appendChild(s);
      });
      tokenIdxEl.max = Math.max(0, lastTokens.length - 1);
      layerEl.max = Math.max(0, lastNumLayers - 1);
      headEl.max = Math.max(0, lastNumHeads - 1);
      resultPanel.style.display = 'block';
      if (d.image_base64) {
        preview.src = 'data:image/png;base64,' + d.image_base64;
        imagePanel.style.display = 'block';
      }
    }

    runBtn.addEventListener('click', function() {
      if (!file.files.length) { runStatus.textContent = 'Select an image.'; return; }
      var q = (question.value || '').trim();
      if (!q) { runStatus.textContent = 'Enter a question.'; return; }
      runStatus.textContent = 'Running…';
      runBtn.disabled = true;
      var fd = new FormData();
      fd.append('image', file.files[0]);
      fd.append('question', q);
      fetch('/run', { method: 'POST', body: fd })
        .then(function(r) { return r.json(); })
        .then(function(d) {
          runBtn.disabled = false;
          if (d.error) { runStatus.textContent = d.error; return; }
          runStatus.textContent = 'Done.';
          applyLastResult(d);
        })
        .catch(function(e) { runBtn.disabled = false; runStatus.textContent = 'Error: ' + e.message; });
    });

    mapBtn.addEventListener('click', function() {
      var tokenIdx = parseInt(tokenIdxEl.value, 10) || 0;
      var layer = parseInt(layerEl.value, 10) || 0;
      var head = parseInt(headEl.value, 10) || 0;
      var threshold = parseFloat(thresholdEl.value) || 0;
      fetch('/attention_map?token_idx=' + tokenIdx + '&layer=' + layer + '&head=' + head + '&threshold=' + threshold)
        .then(function(r) { return r.json(); })
        .then(function(d) {
          if (d.error) { heatmapOverlay.style.display = 'none'; return; }
          heatmapOverlay.src = 'data:image/png;base64,' + d.heatmap;
          heatmapOverlay.style.display = 'block';
          heatmapOverlay.style.width = preview.offsetWidth + 'px';
          heatmapOverlay.style.height = preview.offsetHeight + 'px';
        })
        .catch(function() { heatmapOverlay.style.display = 'none'; });
    });

    if (window.location.search.indexOf('from_8087=1') >= 0) {
      fetch('/last_result').then(function(r) { return r.json(); }).then(applyLastResult).catch(function() {});
    }
  </script>
</body>
</html>
"""


def _cors_headers():
    return {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type"}


class AttentionHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return
        if self.path.startswith("/last_result"):
            run = _last_run
            if run["attn"] is None and not run.get("tokens"):
                self._send_json({"error": "No run yet"}, 404)
                return
            self._send_json({
                "answer": run.get("answer", ""),
                "tokens": run.get("tokens", []),
                "num_layers": run.get("num_layers", 0),
                "num_heads": run.get("num_heads", 0),
                "image_base64": run.get("image_base64"),
                "attention_available": run.get("attn") is not None,
            })
            return
        if self.path.startswith("/attention_map"):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            token_idx = int(qs.get("token_idx", [0])[0])
            layer = int(qs.get("layer", [0])[0])
            head = int(qs.get("head", [0])[0])
            threshold = float(qs.get("threshold", [0])[0])
            run = _last_run
            if run["attn"] is None:
                self._send_json({"error": "No attention data. Run image + question first, or model may not support output_attentions."}, 400)
                return
            attn_list = run["attn"]
            num_layers = run["num_layers"]
            num_heads = run["num_heads"]
            if token_idx < 0 or token_idx >= len(attn_list):
                self._send_json({"error": "Invalid token index."}, 400)
                return
            if layer < 0 or layer >= num_layers or head < 0 or head >= num_heads:
                self._send_json({"error": "Invalid layer or head."}, 400)
                return
            try:
                heatmap = attention_to_heatmap(
                    attn_list, token_idx, layer, head, IMAGE_TOKEN_COUNT, threshold=threshold
                )
                if heatmap is None:
                    self._send_json({"error": "Could not build heatmap."}, 400)
                    return
                img_w, img_h = 512 * PATCH_COLS, 512 * PATCH_ROWS
                b64 = heatmap_to_base64(heatmap, (img_w, img_h), threshold=threshold)
                if b64 is None:
                    self._send_json({"error": "Heatmap render failed."}, 500)
                    return
                self._send_json({"heatmap": b64})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/run":
            ctype = self.headers.get("Content-Type", "")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            image = None
            q = ""
            if "application/json" in ctype:
                try:
                    data = json.loads(body.decode("utf-8"))
                    img_b64 = (data.get("image_base64") or "").strip()
                    q = (data.get("question") or "").strip()
                    if not img_b64 or not q:
                        self._send_json({"error": "Missing image_base64 or question"}, 400)
                        return
                    imgb = base64.b64decode(img_b64)
                    image = Image.open(io.BytesIO(imgb)).convert("RGB")
                except Exception as e:
                    self._send_json({"error": "Bad JSON or image: " + str(e)}, 400)
                    return
            elif "multipart/form-data" in ctype:
                try:
                    boundary = ctype.split("boundary=")[-1].strip().strip('"').strip("'")
                    parts = self._parse_multipart(body, boundary)
                    imgb = parts.get("image")
                    q = (parts.get("question") or b"").decode("utf-8", errors="replace").strip()
                    if not imgb or not q:
                        self._send_json({"error": "Missing image or question"}, 400)
                        return
                    image = Image.open(io.BytesIO(imgb)).convert("RGB")
                except Exception as e:
                    self._send_json({"error": "Bad request: " + str(e)}, 400)
                    return
            else:
                self._send_json({"error": "Expect application/json or multipart/form-data"}, 400)
                return
            server = self.server
            if server.model is None or server.processor is None:
                self._send_json({"error": "Model not loaded"}, 503)
                return
            try:
                answer, tokens, attn_list, prompt_len, num_layers, num_heads = run_with_attentions(
                    server.model, server.processor, image, q, server.device
                )
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
                return
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            _last_run["attn"] = attn_list
            _last_run["prompt_length"] = prompt_len
            _last_run["num_layers"] = num_layers
            _last_run["num_heads"] = num_heads
            _last_run["tokens"] = tokens
            _last_run["answer"] = answer
            _last_run["image_base64"] = img_b64
            self._send_json({
                "answer": answer,
                "tokens": tokens,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "image_base64": img_b64,
                "attention_available": attn_list is not None,
            })
            return
        self.send_response(404)
        self.end_headers()

    def _parse_multipart(self, body, boundary):
        parts = {}
        b = boundary.encode() if isinstance(boundary, str) else boundary
        for block in body.split(b"--" + b):
            if not block.strip():
                continue
            head, _, rest = block.partition(b"\r\n\r\n")
            if not rest:
                continue
            disp = head.decode("utf-8", errors="replace")
            name = None
            for line in disp.split("\r\n"):
                if line.lower().startswith("content-disposition:"):
                    for part in line.split(";"):
                        part = part.strip()
                        if part.lower().startswith("name="):
                            name = part.split("=", 1)[1].strip('"')
                            break
                    break
            if name:
                parts[name] = rest.rstrip(b"\r\n")
        return parts

    def _send_json(self, obj, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def log_message(self, format, *args):
        print("[attention] %s" % (args[0] if args else format % args), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="VLM attention map server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default %d)" % DEFAULT_PORT)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device, file=sys.stderr)

    server = HTTPServer(("0.0.0.0", args.port), AttentionHandler)
    server.model = None
    server.processor = None
    server.device = device

    def run_server():
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=False)
    thread.start()

    try:
        def set_progress(pct, msg):
            print("[%d%%] %s" % (pct, msg), file=sys.stderr)
        server.model, server.processor = load_model(device, progress_callback=set_progress)
        print("VLM attention server: http://0.0.0.0:%d/" % args.port, file=sys.stderr)
    except Exception as e:
        print("Failed to load model: %s" % e, file=sys.stderr)
        raise
    thread.join()


if __name__ == "__main__":
    main()
