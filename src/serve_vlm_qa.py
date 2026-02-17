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

DEFAULT_PORT = 8087
MAX_NEW_TOKENS = 80


def load_smolvlm(model_name: str, device, progress_callback=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if progress_callback:
        progress_callback(10, "Loading processor…")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if progress_callback:
        progress_callback(30, "Loading model…")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=str(device) if device.type == "cuda" else None,
        trust_remote_code=True,
    )
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
      <button type="button" id="askBtn" disabled>Ask</button>
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
                    "GET /health": "Health / loading progress",
                    "POST /ask": "Body: multipart (image file + question) or JSON { \"image_base64\": \"...\", \"question\": \"...\" }. Returns { \"answer\": \"...\" }.",
                    "POST /ask_youtube": "Body: JSON { \"url\": \"YouTube URL\", \"time_sec\": 0, \"question\": \"...\" }. Returns { \"answer\": \"...\" }.",
                    "POST /detect": "Body: multipart (image file) or JSON { \"image_base64\": \"...\" }. Returns { \"frame_base64\": \"...\", \"detections\": [...] }.",
                    "POST /detect_youtube": "Body: JSON { \"url\": \"YouTube URL\", \"time_sec\": 0 }. Returns { \"frame_base64\": \"...\", \"detections\": [...] } (no VLM).",
                },
                "port": self.server.server_port,
            })
            return
        if path == "/":
            self.send_html(HTML_PAGE)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = self.path.rstrip("/")
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
                if not question or not b64:
                    self.send_json({"error": "Missing question or image_base64", "answer": None}, status=400)
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
                if not question or image is None:
                    self.send_json({"error": "Missing question or image file", "answer": None}, status=400)
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
        default="smolvlm2-2.2b",
        choices=["smolvlm2-2.2b", "smolvlm2-500m", "llava-1.5-7b"],
        help="VLM to load (default: smolvlm2-2.2b)",
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
