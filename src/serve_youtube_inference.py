#!/usr/bin/env python3
"""
YouTube inference: run Safe/Unsafe model on YouTube videos without downloading.
- User pastes a YouTube URL; video is embedded in the page.
- Backend streams the video (yt-dlp -> ffmpeg pipe), extracts frames in memory at 1 fps,
  runs inference with checkpoint 2056, returns prediction and frame-wise distribution.
No files are written to disk.
"""

import argparse
import io
import json
import re
import subprocess
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

import psutil
from PIL import Image

# Reuse model and prediction from serve_inference
from serve_inference import (
    OUTPUT_DIR,
    load_model,
    predict_from_image_with_reason,
)


def gather_system_stats():
    """Return dict with GPU and CPU/memory stats for load-status."""
    out = {
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_temp_c": None,
        "cpu_pct": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
    }
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split("\n")[0].split(",")
            if len(parts) >= 4:
                out["gpu_util_pct"] = int(parts[0].strip() or 0)
                out["gpu_mem_used_mb"] = int(parts[1].strip().split()[0] or 0)
                out["gpu_mem_total_mb"] = int(parts[2].strip().split()[0] or 0)
                out["gpu_temp_c"] = int(parts[3].strip() or 0)
    except Exception:
        pass
    try:
        out["cpu_pct"] = round(psutil.cpu_percent(interval=0.05), 1)
        v = psutil.virtual_memory()
        out["ram_used_mb"] = int(v.used / (1024**2))
        out["ram_total_mb"] = int(v.total / (1024**2))
    except Exception:
        pass
    return out

DEFAULT_PORT = 8083
DEFAULT_CHECKPOINT = "checkpoint-2056"
MAX_FRAMES = 120  # cap at ~2 min at 1 fps to keep latency reasonable
FPS_SAMPLE = 1

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIME_TYPES = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}

# Shared load state: updated by background loader and read by /load-status
_load_status = {
    "loaded": False,
    "progress_pct": 0,
    "message": "Starting…",
    "error": None,
}
_load_status_lock = threading.Lock()

# PNG end marker (IEND chunk footer)
PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def extract_youtube_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    if not url or not url.strip():
        return None
    url = url.strip()
    # youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID
    m = re.search(r"(?:v=|/embed/|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None


def _read_pipe_to_list(pipe, out_list, prefix=""):
    """Read lines from pipe into out_list; decode as utf-8."""
    try:
        for line in iter(pipe.readline, b""):
            out_list.append((prefix, line.decode("utf-8", errors="replace").strip()))
    except Exception:
        pass


def extract_frames_from_youtube_stream(url: str, fps: int = 1, max_frames: int = MAX_FRAMES):
    """
    Stream YouTube video via yt-dlp -> ffmpeg, extract frames in memory.
    Returns (list of (frame_index, time_sec, PIL.Image), stderr_ydl, stderr_ff).
    No disk write.
    """
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
        "-hide_banner", "-loglevel", "warning",
        "-i", "pipe:0",
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-f", "image2pipe",
        "-c:v", "png",
        "pipe:1",
    ]
    stderr_ydl, stderr_ff = [], []
    try:
        p_ydl = subprocess.Popen(
            cmd_ydl,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        t_ydl = threading.Thread(target=_read_pipe_to_list, args=(p_ydl.stderr, stderr_ydl, "yt-dlp"))
        t_ydl.daemon = True
        t_ydl.start()
    except FileNotFoundError:
        return [], [("yt-dlp", "yt-dlp not found in PATH")], []
    try:
        p_ff = subprocess.Popen(
            cmd_ffmpeg,
            stdin=p_ydl.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        p_ydl.stdout.close()
        t_ff = threading.Thread(target=_read_pipe_to_list, args=(p_ff.stderr, stderr_ff, "ffmpeg"))
        t_ff.daemon = True
        t_ff.start()
    except FileNotFoundError:
        p_ydl.terminate()
        return [], stderr_ydl, [("ffmpeg", "ffmpeg not found in PATH")]

    frames = []
    buffer = b""
    frame_index = 0
    try:
        while frame_index < max_frames:
            chunk = p_ff.stdout.read(8192)
            if not chunk:
                break
            buffer += chunk
            while PNG_END in buffer:
                i = buffer.index(PNG_END)
                png_bytes = buffer[: i + len(PNG_END)]
                buffer = buffer[i + len(PNG_END) :]
                try:
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    frames.append((frame_index, frame_index, img))
                    frame_index += 1
                except Exception:
                    pass
    finally:
        p_ff.terminate()
        p_ydl.terminate()
        try:
            p_ff.wait(timeout=3)
            p_ydl.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p_ff.kill()
            p_ydl.kill()
        t_ff.join(timeout=1)
        t_ydl.join(timeout=1)
    return frames, stderr_ydl, stderr_ff


def infer_youtube_url(model, processor, device, url: str, max_frames: int = MAX_FRAMES):
    """
    Stream YouTube URL, extract frames in memory, run per-frame inference, max-vote.
    Returns dict with prediction, frame_votes, frame_distribution, total_frames.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return {"error": "Invalid YouTube URL", "prediction": None, "frame_votes": {"SAFE": 0, "UNSAFE": 0}, "frame_distribution": [], "total_frames": 0}
    frames, stderr_ydl, stderr_ff = extract_frames_from_youtube_stream(url, fps=FPS_SAMPLE, max_frames=max_frames)
    if not frames:
        err_parts = ["Could not extract frames."]
        if stderr_ydl:
            err_parts.append("yt-dlp: " + " ".join(line for _, line in stderr_ydl[-5:]))
        if stderr_ff:
            err_parts.append("ffmpeg: " + " ".join(line for _, line in stderr_ff[-5:]))
        return {"error": " ".join(err_parts), "prediction": None, "frame_votes": {"SAFE": 0, "UNSAFE": 0}, "frame_distribution": [], "total_frames": 0, "video_id": video_id}
    votes = {"SAFE": 0, "UNSAFE": 0}
    frame_distribution = []
    for frame_index, time_sec, img in frames:
        pred, reason = predict_from_image_with_reason(model, processor, img, device)
        if pred in ("SAFE", "UNSAFE"):
            votes[pred] = votes.get(pred, 0) + 1
        frame_distribution.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "prediction": pred,
            "reason": reason or "",
        })
    prediction = "UNSAFE" if votes["UNSAFE"] >= votes["SAFE"] else "SAFE"
    return {
        "prediction": prediction,
        "frame_votes": votes,
        "frame_distribution": frame_distribution,
        "total_frames": len(frames),
        "video_id": video_id,
        "embed_url": f"https://www.youtube.com/embed/{video_id}",
    }


class YouTubeInferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_static_file(self, file_path: Path):
        if not file_path.exists() or not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        suffix = file_path.suffix.lower()
        content_type = MIME_TYPES.get(suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self.send_json({"status": "ok", "model_loaded": self.server.model is not None, "service": "youtube-inference"})
            return
        if path == "/load-status":
            with _load_status_lock:
                status = dict(_load_status)
            status.update(gather_system_stats())
            self.send_json(status)
            return
        if path == "/infer-youtube":
            if self.server.model is None:
                self.send_json({"error": "Model not loaded"}, status=503)
                return
            qs = urllib.parse.parse_qs(parsed.query)
            url = (qs.get("url") or [None])[0]
            if not url:
                self.send_json({"error": "Missing query parameter: url (YouTube URL)"}, status=400)
                return
            url = urllib.parse.unquote(url)
            max_frames = MAX_FRAMES
            if "max_frames" in qs:
                try:
                    max_frames = min(MAX_FRAMES, max(1, int(qs["max_frames"][0])))
                except ValueError:
                    pass
            out = infer_youtube_url(
                self.server.model,
                self.server.processor,
                self.server.device,
                url,
                max_frames=max_frames,
            )
            self.send_json(out)
            return
        if path == "/info":
            self.send_json({
                "service": "SmolVLM2 Safe/Unsafe — YouTube inference (no download)",
                "port": self.server.server_port,
                "checkpoint": str(self.server.checkpoint),
                "endpoints": {
                    "GET /": "UI: paste YouTube URL, embed video, run inference",
                    "GET /health": "Health check",
                    "GET /infer-youtube?url=...": "Run inference on a YouTube URL (stream in memory). Optional: &max_frames=N (default 120).",
                },
            })
            return
        if path == "/":
            self.send_static_file(STATIC_DIR / "youtube.html")
            return
        if path.startswith("/static/"):
            subpath = path[8:].lstrip("/")
            if ".." in subpath or subpath.startswith("/"):
                self.send_response(404)
                self.end_headers()
                return
            self.send_static_file(STATIC_DIR / subpath)
            return
        self.send_response(404)
        self.end_headers()


def _load_model_into_server(server, adapter_path: Path, base_model: str):
    """Run in background: load model and attach to server; update _load_status."""
    global _load_status
    try:
        def progress_cb(pct: int, msg: str):
            with _load_status_lock:
                _load_status["progress_pct"] = pct
                _load_status["message"] = msg
                _load_status.update(gather_system_stats())

        with _load_status_lock:
            _load_status["message"] = "Loading model…"
            _load_status["progress_pct"] = 0
        model, processor = load_model(adapter_path, base_model, progress_callback=progress_cb)
        device = next(model.parameters()).device
        server.model = model
        server.processor = processor
        server.device = device
        with _load_status_lock:
            _load_status["loaded"] = True
            _load_status["progress_pct"] = 100
            _load_status["message"] = "Ready"
            _load_status["error"] = None
        print(f"Model loaded on {device}")
    except Exception as e:
        with _load_status_lock:
            _load_status["error"] = str(e)
            _load_status["message"] = "Load failed"
        print(f"Model load failed: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="YouTube inference (no download), checkpoint 2056")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default %d)" % DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir (default: %s)" % DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    args = parser.parse_args()

    adapter_path = Path(args.checkpoint) if args.checkpoint else (OUTPUT_DIR / DEFAULT_CHECKPOINT)
    if not adapter_path.is_absolute():
        adapter_path = (OUTPUT_DIR / adapter_path).resolve()
    if not (adapter_path / "adapter_model.safetensors").exists() and not (adapter_path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"No adapter in {adapter_path}")

    server = HTTPServer(("0.0.0.0", args.port), YouTubeInferenceHandler)
    server.model = None
    server.processor = None
    server.device = None
    server.checkpoint = adapter_path

    t = threading.Thread(target=_load_model_into_server, args=(server, adapter_path, args.base_model), daemon=True)
    t.start()

    print(f"YouTube inference server: http://0.0.0.0:{args.port}/ (model loading in background)")
    server.serve_forever()


if __name__ == "__main__":
    main()
