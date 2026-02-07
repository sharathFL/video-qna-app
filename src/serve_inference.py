#!/usr/bin/env python3
"""
Inference server for trained SmolVLM2 Safe/Unsafe model.
Serves on a configurable port (default 8081). Run test set via GET /test.
"""

import argparse
import base64
import io
import json
import sys
import threading
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Paths (match train_smolvlm / evaluate)
FRAMES_ROOT = Path("/workspace/data/frames")
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/outputs")
TEST_JSONL = DATA_DIR / "test.jsonl"
SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Safe / Unsafe — Inference</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #1a1a2e; color: #eee; min-height: 100vh; }
    #loadOverlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; padding: 24px; }
    #loadOverlay.hidden { display: none; }
    #loadOverlay .msg { color: #a0a0a0; margin-bottom: 8px; font-size: 1.1rem; }
    #loadPct { font-size: 1.5rem; font-weight: bold; color: #e94560; margin: 0 0 12px 0; }
    #loadLog { width: 100%; max-width: 720px; min-height: 200px; max-height: 60vh; overflow: auto; background: #0f0f1a; color: #c0c0c0; padding: 16px; border-radius: 8px; font-size: 13px; font-family: ui-monospace, monospace; text-align: left; white-space: pre-wrap; word-break: break-word; border: 1px solid #2a2a4a; }
    #loadLog:empty::before { content: 'Waiting for server…'; color: #606080; }
    #mainContent { display: none; }
    #mainContent.ready { display: block; }
    h1 { color: #e94560; margin-top: 0; }
    .card { background: #16213e; border-radius: 12px; padding: 24px; max-width: 560px; margin-bottom: 20px; }
    label { display: block; margin-bottom: 8px; color: #a0a0a0; }
    input[type="file"] { margin-bottom: 12px; color: #eee; }
    button { background: #e94560; color: #fff; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #ff6b6b; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #previewWrap:not(.hidden) { display: block; min-height: 120px; margin-top: 12px; }
    #preview { max-width: 100%; max-height: 280px; width: auto; height: auto; border-radius: 8px; border: 2px solid #333; display: block; vertical-align: top; }
    #result { margin-top: 20px; font-size: 1.25rem; font-weight: bold; padding: 16px; border-radius: 8px; }
    #result.safe { background: #2e7d32; color: #fff; }
    #result.unsafe { background: #c62828; color: #fff; }
    #result.pending { background: #37474f; color: #cfd8dc; }
    #result.error { background: #b71c1c; color: #fff; }
    .hidden { display: none; }
    a { color: #e94560; }
  </style>
</head>
<body>
  <div id="loadOverlay">
    <p class="msg">Server log — model loading</p>
    <p id="loadPct">0% ready</p>
    <pre id="loadLog"></pre>
  </div>
  <div id="mainContent">
  <h1>Safe / Unsafe — Inference</h1>
  <p style="color:#888;">Upload a CCTV frame. The model will classify the behavior as <strong>SAFE</strong> or <strong>UNSAFE</strong>.</p>
  <div class="card">
    <label for="file">Choose an image</label>
    <input type="file" id="file" accept="image/*">
    <div id="previewWrap" class="hidden">
      <p id="previewSummary" style="margin:8px 0 4px 0;color:#b0b0b0;font-size:0.95rem;"></p>
      <p id="previewLoading" class="hidden" style="margin:4px 0;color:#888;">Loading preview…</p>
      <img id="preview" alt="Preview" style="max-width:100%;max-height:240px;display:block;margin-top:8px;">
    </div>
    <button type="button" id="run" disabled>Run inference</button>
    <div id="result" class="hidden"></div>
  </div>
  <p style="color:#666; font-size:0.9rem;"><a href="/info">API info</a> · <a href="/test-page?max=5">Test set (5 samples)</a> · <span style="color:#606080;">F12 → Console for logs</span></p>
  </div>
  <script>
    function setupUpload() {
      console.log('[Inference] setupUpload()');
      var file = document.getElementById('file');
      var preview = document.getElementById('preview');
      var previewWrap = document.getElementById('previewWrap');
      var run = document.getElementById('run');
      var result = document.getElementById('result');
      function updateButton() {
        if (file.files && file.files[0]) {
          var f = file.files[0];
          var sizeKb = (f.size / 1024).toFixed(1);
          console.log('[Inference] File selected:', f.name, f.size, 'bytes', f.type);
          run.disabled = false;
          result.classList.add('hidden');
          previewWrap.classList.remove('hidden');
          document.getElementById('previewSummary').textContent = 'Selected: ' + f.name + ' (' + sizeKb + ' KB) — click Run inference below';
          var loadingEl = document.getElementById('previewLoading');
          loadingEl.classList.remove('hidden');
          loadingEl.textContent = 'Loading preview…';
          preview.alt = f.name;
          preview.style.visibility = 'visible';
          preview.removeAttribute('src');
          if (preview._previewUrl) try { URL.revokeObjectURL(preview._previewUrl); } catch (e) {}
          preview._previewUrl = null;
          var done = false;
          function hideLoading() {
            if (done) return;
            done = true;
            loadingEl.classList.add('hidden');
          }
          setTimeout(hideLoading, 2500);
          var reader = new FileReader();
          reader.onload = function() {
            var dataUrl = reader.result;
            var img = new Image();
            img.onload = function() {
              try {
                var c = document.createElement('canvas');
                var m = 400;
                var w = img.width, h = img.height;
                if (w > m || h > m) {
                  if (w > h) { h = (h * m / w) | 0; w = m; } else { w = (w * m / h) | 0; h = m; }
                }
                c.width = w;
                c.height = h;
                var ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                preview.src = c.toDataURL('image/jpeg', 0.85);
                hideLoading();
              } catch (e) {
                console.warn('[Inference] Canvas preview failed:', e);
                preview.src = dataUrl;
                hideLoading();
              }
            };
            img.onerror = function() {
              preview.src = dataUrl;
              hideLoading();
            };
            img.src = dataUrl;
          };
          reader.onerror = function() {
            console.error('[Inference] FileReader error for preview');
            hideLoading();
          };
          reader.readAsDataURL(f);
        } else {
          console.log('[Inference] No file selected');
          run.disabled = true;
          previewWrap.classList.add('hidden');
          result.classList.add('hidden');
          document.getElementById('previewSummary').textContent = '';
          if (preview._previewUrl) try { URL.revokeObjectURL(preview._previewUrl); } catch (e) {}
          preview._previewUrl = null;
          preview.removeAttribute('src');
        }
      }
      file.addEventListener('change', updateButton);
      file.addEventListener('input', updateButton);
      run.addEventListener('click', function() {
      if (!file.files || !file.files[0]) {
        console.warn('[Inference] Run clicked but no file');
        return;
      }
      console.log('[Inference] Run inference clicked, reading file...');
      run.disabled = true;
      result.classList.remove('hidden');
      result.className = 'pending';
      result.textContent = 'Running inference… (usually 5–15 s on GPU)';
      var reader = new FileReader();
      reader.onload = function() {
        var b64 = reader.result.split(',')[1] || reader.result;
        console.log('[Inference] Sending POST /predict_image, base64 length:', b64 ? b64.length : 0);
        var controller = new AbortController();
        var timeout = setTimeout(function() { controller.abort(); }, 120000);
        fetch('/predict_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: b64 }),
          signal: controller.signal
        }).then(function(resp) {
          clearTimeout(timeout);
          console.log('[Inference] Response status:', resp.status, resp.statusText);
          return resp.text().then(function(text) {
            console.log('[Inference] Response body (first 200 chars):', text ? text.slice(0, 200) : '');
            try { return JSON.parse(text); } catch (e) { throw new Error(resp.status + ': ' + text.slice(0, 100)); }
          }).then(function(d) {
            if (d.error) throw new Error(d.error);
            return d;
          });
        }).then(function(d) {
          console.log('[Inference] Success:', d.prediction);
          result.classList.remove('pending');
          result.className = (d.prediction || '').toLowerCase();
          result.textContent = d.prediction || '—';
          run.disabled = false;
        }).catch(function(e) {
          clearTimeout(timeout);
          console.error('[Inference] Error:', e.message || e);
          result.classList.remove('pending');
          result.className = 'error';
          result.textContent = 'Error: ' + (e.message || String(e));
          run.disabled = false;
        });
      };
      reader.onerror = function() {
        console.error('[Inference] FileReader error on run');
        run.disabled = false;
        result.classList.remove('pending');
        result.className = 'error';
        result.textContent = 'Error: failed to read file';
      };
      reader.readAsDataURL(file.files[0]);
    });
    }
    (function waitForModel() {
      function pollLog() {
        fetch('/load-log').then(function(r) { return r.text(); }).then(function(text) {
          var el = document.getElementById('loadLog');
          if (el) {
            el.textContent = text || 'Waiting for server…';
            el.scrollTop = el.scrollHeight;
          }
        }).catch(function() {});
      }
      function pollHealth() {
        fetch('/health').then(function(r) { return r.json(); }).then(function(d) {
          var pctEl = document.getElementById('loadPct');
          if (pctEl) pctEl.textContent = (d.loading_progress != null ? d.loading_progress : 0) + '% ready';
          if (d && d.model_loaded) {
            clearInterval(logInterval);
            clearInterval(healthInterval);
            pollLog();
            document.getElementById('loadOverlay').classList.add('hidden');
            document.getElementById('mainContent').classList.add('ready');
            setupUpload();
          }
        }).catch(function() {});
      }
      pollLog();
      pollHealth();
      var logInterval = setInterval(pollLog, 800);
      var healthInterval = setInterval(pollHealth, 800);
    })();
  </script>
</body>
</html>
"""

TEST_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Test set — Safe / Unsafe</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #1a1a2e; color: #eee; }
    #loadOverlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; padding: 24px; }
    #loadOverlay.hidden { display: none; }
    #loadOverlay .msg { color: #a0a0a0; margin-bottom: 8px; font-size: 1.1rem; }
    #loadPct { font-size: 1.5rem; font-weight: bold; color: #e94560; margin: 0 0 12px 0; }
    #loadLog { width: 100%; max-width: 720px; min-height: 180px; max-height: 55vh; overflow: auto; background: #0f0f1a; color: #c0c0c0; padding: 16px; border-radius: 8px; font-size: 13px; font-family: ui-monospace, monospace; white-space: pre-wrap; word-break: break-word; border: 1px solid #2a2a4a; }
    #loadLog:empty::before { content: 'Waiting for server…'; color: #606080; }
    h1 { color: #e94560; margin-top: 0; }
    .loading { padding: 24px; color: #888; }
    table { border-collapse: collapse; width: 100%; max-width: 900px; margin-top: 16px; }
    th, td { padding: 12px; text-align: left; border: 1px solid #333; }
    th { background: #16213e; }
    tr.correct { background: #1b3d1b; }
    tr.incorrect { background: #4a1515; }
    td img { max-width: 160px; max-height: 120px; border-radius: 6px; display: block; }
    a { color: #e94560; }
  </style>
</head>
<body>
  <div id="loadOverlay">
    <p class="msg">Waiting for model (same server — not reloading). Server log below:</p>
    <p id="loadPct">0% ready</p>
    <pre id="loadLog"></pre>
  </div>
  <h1>Test set results</h1>
  <p><a href="/">← Upload image</a> · <a href="/test-page?max=5">5 samples</a> · <a href="/test-page?max=20">20 samples</a></p>
  <div id="out"><div class="loading">Waiting for model…</div></div>
  <script>
    function runTest() {
      var params = new URLSearchParams(window.location.search);
      var max = params.get('max') || '5';
      document.getElementById('out').innerHTML = '<div class="loading">Running inference on ' + max + ' samples…</div>';
      fetch('/test?max=' + max)
      .then(function(r) { return r.json(); })
      .then(function(d) {
        var out = document.getElementById('out');
        if (d.error || !d.results || !d.results.length) {
          out.innerHTML = '<p class="loading">' + (d.error || 'No results') + '</p>';
          return;
        }
        var acc = d.accuracy != null ? d.accuracy + '%' : '';
        var html = '<p><strong>Accuracy: ' + d.correct + '/' + d.total + ' (' + acc + ')</strong></p>';
        html += '<table><tr><th>Image</th><th>Ground truth</th><th>Prediction</th><th>Match</th></tr>';
        d.results.forEach(function(r) {
          var path = encodeURIComponent(r.image);
          var rowClass = r.correct ? 'correct' : 'incorrect';
          var match = r.correct ? 'Yes' : 'No';
          html += '<tr class="' + rowClass + '"><td><img src="/frame?path=' + path + '" alt=""></td><td>' + (r.ground_truth || '') + '</td><td>' + (r.prediction || '—') + '</td><td>' + match + '</td></tr>';
        });
        html += '</table>';
        out.innerHTML = html;
      })
      .catch(function(e) { document.getElementById('out').innerHTML = '<p class="loading" style="color:#c62828">Error: ' + e.message + '</p>'; });
    }
    (function waitForModel() {
      function pollLog() {
        fetch('/load-log').then(function(r) { return r.text(); }).then(function(text) {
          var el = document.getElementById('loadLog');
          if (el) { el.textContent = text || 'Waiting for server…'; el.scrollTop = el.scrollHeight; }
        }).catch(function() {});
      }
      function pollHealth() {
        fetch('/health').then(function(r) { return r.json(); }).then(function(d) {
          var pctEl = document.getElementById('loadPct');
          if (pctEl) pctEl.textContent = (d.loading_progress != null ? d.loading_progress : 0) + '% ready';
          if (d && d.model_loaded) {
            clearInterval(logInterval);
            clearInterval(healthInterval);
            document.getElementById('loadOverlay').classList.add('hidden');
            runTest();
          }
        }).catch(function() {});
      }
      pollLog();
      pollHealth();
      var logInterval = setInterval(pollLog, 800);
      var healthInterval = setInterval(pollHealth, 800);
    })();
  </script>
</body>
</html>
"""


def normalize_pred(raw: str) -> str:
    raw = (raw or "").strip().upper()
    if "UNSAFE" in raw:
        return "UNSAFE"
    if "SAFE" in raw:
        return "SAFE"
    return raw[:20] if raw else ""


def load_model(adapter_path: Path, base_model: str = SMOLVLM2_2B, progress_callback=None):
    def report(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[{pct}%] {msg}")

    use_cuda = torch.cuda.is_available()
    report(5, "Starting…")
    if use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    report(15, "Loading processor…")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    report(25, "Processor loaded. Loading base model…")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else None,
        trust_remote_code=True,
    )
    report(60, "Base model loaded. Loading adapter…")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    report(90, "Adapter loaded. Finalizing…")
    model.eval()
    report(100, "Ready")
    return model, processor


def predict_from_image(model, processor, img: Image.Image, device) -> str:
    """Run inference on a PIL Image. img must be RGB."""
    if img.mode != "RGB":
        img = img.convert("RGB")
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
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False)
    gen = processor.batch_decode(
        out[:, inp["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_pred(gen[0] if gen else "")


def predict_one(model, processor, image_path: Path, device) -> str:
    if not image_path.exists():
        return ""
    img = Image.open(image_path).convert("RGB")
    return predict_from_image(model, processor, img, device)




def run_test_set(model, processor, device, max_samples=None):
    if not TEST_JSONL.exists():
        return {"error": "test.jsonl not found", "correct": 0, "total": 0, "accuracy": 0.0, "results": []}
    samples = []
    with open(TEST_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]
    results = []
    correct = 0
    total = 0
    for sample in samples:
        image_path = FRAMES_ROOT / sample["image"]
        gt = (sample.get("response") or "").strip().upper()
        if gt not in ("SAFE", "UNSAFE"):
            continue
        total += 1
        try:
            pred = predict_one(model, processor, image_path, device)
            ok = pred == gt
            if ok:
                correct += 1
            results.append({
                "image": sample["image"],
                "ground_truth": gt,
                "prediction": pred,
                "correct": ok,
            })
        except Exception as e:
            results.append({
                "image": sample["image"],
                "ground_truth": gt,
                "prediction": None,
                "correct": False,
                "error": str(e),
            })
    acc = (correct / total * 100.0) if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 2),
        "results": results,
    }


class InferenceHandler(BaseHTTPRequestHandler):
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
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            loaded = self.server.model is not None
            pct = getattr(self.server, "loading_progress", 0)
            self.send_json({
                "status": "ok",
                "model_loaded": loaded,
                "loading_progress": 100 if loaded else pct,
            })
            return
        if path == "/load-log":
            log_lines = getattr(self.server, "load_log", [])
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write("\n".join(log_lines).encode("utf-8"))
            return
        if path == "/test":
            max_s = None
            qs = urllib.parse.parse_qs(parsed.query)
            if "max" in qs:
                try:
                    max_s = int(qs["max"][0])
                except ValueError:
                    pass
            out = run_test_set(self.server.model, self.server.processor, self.server.device, max_samples=max_s)
            self.send_json(out)
            return
        if path == "/":
            self.send_html(HTML_PAGE)
            return
        if path == "/test-page":
            self.send_html(TEST_PAGE_HTML)
            return
        if path == "/frame":
            qs = urllib.parse.parse_qs(parsed.query)
            rel = (qs.get("path") or [""])[0]
            if ".." in rel or rel.startswith("/"):
                self.send_response(400)
                self.end_headers()
                return
            image_path = FRAMES_ROOT / rel
            if not image_path.exists() or not image_path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            try:
                img = Image.open(image_path)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buf.getvalue())
            except Exception:
                self.send_response(500)
                self.end_headers()
            return
        if path == "/info":
            self.send_json({
                "service": "SmolVLM2 Safe/Unsafe inference",
                "endpoints": {
                    "GET /": "Webpage: upload image and run inference",
                    "GET /health": "Health check",
                    "GET /test": "Run inference on test.jsonl (query: ?max=N to limit samples)",
                    "POST /predict": "Body: {\"image\": \"test/safe/4_te1/0000.jpg\"}",
                    "POST /predict_image": "Body: {\"image_base64\": \"<base64>\"} — upload image",
                },
                "port": self.server.server_port,
            })
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        path = self.path.rstrip("/")
        if path == "/predict":
            try:
                data = json.loads(body)
                rel_path = data.get("image") or data.get("path", "")
                image_path = FRAMES_ROOT / rel_path
                pred = predict_one(
                    self.server.model,
                    self.server.processor,
                    image_path,
                    self.server.device,
                )
                self.send_json({"prediction": pred, "image": rel_path})
            except Exception as e:
                self.send_json({"error": str(e), "prediction": None}, status=400)
            return
        if path == "/predict_image":
            try:
                print("[predict_image] Request received")
                data = json.loads(body)
                b64 = data.get("image_base64") or ""
                if not b64:
                    print("[predict_image] Error: Missing image_base64")
                    self.send_json({"error": "Missing image_base64", "prediction": None}, status=400)
                    return
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                print("[predict_image] Image decoded, running inference...")
                pred = predict_from_image(
                    self.server.model,
                    self.server.processor,
                    img,
                    self.server.device,
                )
                print("[predict_image] Result:", pred)
                self.send_json({"prediction": pred})
            except Exception as e:
                print("[predict_image] Error:", e)
                self.send_json({"error": str(e), "prediction": None}, status=400)
            return
        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Inference server for Safe/Unsafe model")
    parser.add_argument("--port", type=int, default=8081, help="Port to serve on")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir (default: latest checkpoint-*)")
    parser.add_argument("--base-model", type=str, default=SMOLVLM2_2B)
    args = parser.parse_args()

    if args.checkpoint:
        adapter_path = Path(args.checkpoint)
    else:
        checkpoints = sorted(
            OUTPUT_DIR.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        adapter_path = checkpoints[-1] if checkpoints else OUTPUT_DIR
    if not (adapter_path / "adapter_model.safetensors").exists() and not (adapter_path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"No adapter in {adapter_path}")

    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    server.model = None
    server.processor = None
    server.device = None
    server.load_log = []
    server.loading_progress = 0

    class Tee:
        def __init__(self, original, log_list):
            self.original = original
            self.log_list = log_list
        def write(self, s):
            self.original.write(s)
            if s and s.strip():
                self.log_list.append(s.rstrip())
        def flush(self):
            self.original.flush()

    def run_server():
        server.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=False)
    server_thread.start()

    try:
        _stdout, _stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(_stdout, server.load_log)
        sys.stderr = Tee(_stderr, server.load_log)
        print(f"Loading model from {adapter_path}...")
        def set_progress(pct: int, _msg: str):
            server.loading_progress = min(100, max(0, pct))
        model, processor = load_model(adapter_path, args.base_model, progress_callback=set_progress)
        device = next(model.parameters()).device
        print(f"Model loaded on {device}")
        print(f"Inference server: http://0.0.0.0:{args.port}/")
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr

    server.model = model
    server.processor = processor
    server.device = device
    server_thread.join()


if __name__ == "__main__":
    main()
