#!/usr/bin/env python3
"""
Video-level inference: run model on .mp4 test videos.
- List test videos from data/Video_Dataset.../test/
- Load a video -> extract frames (1 fps) -> per-frame predictions -> max vote.
- Show frame-wise SAFE/UNSAFE distribution.
Uses checkpoint-990 by default. Serves on port 8082.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

import cv2
import torch
from PIL import Image

# Reuse model loading and prediction from frame inference
from serve_inference import (
    FRAMES_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    TEST_JSONL,
    load_model,
    predict_one,
    predict_from_image,
    predict_from_image_with_reason,
)

# Actual test .mp4 location (allow spaces in path)
TEST_VIDEOS_ROOT = DATA_DIR / "Video_Dataset _for_Safe_and_Unsafe_Behaviours" / "Safe_and _Unsafe_Behaviours_Dataset" / "test"
if not TEST_VIDEOS_ROOT.exists():
    TEST_VIDEOS_ROOT = DATA_DIR / "Safe and Unsafe Behaviours Dataset" / "test"

# Folder prefix -> SAFE/UNSAFE (0-3 UNSAFE, 4-7 SAFE)
UNSAFE_PREFIXES = ("0_", "1_", "2_", "3_")

DEFAULT_PORT = 8082
DEFAULT_CHECKPOINT = "checkpoint-990"

# Static files for the UI (edit look/feel in src/static/)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MIME_TYPES = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}


def folder_to_label(folder_name: str) -> str:
    """Map folder name (e.g. 4_safe_walkway) to SAFE or UNSAFE."""
    for p in UNSAFE_PREFIXES:
        if folder_name.startswith(p):
            return "UNSAFE"
    return "SAFE"


def list_test_videos():
    """List all .mp4 files under TEST_VIDEOS_ROOT. Returns list of { path, label, folder }."""
    if not TEST_VIDEOS_ROOT.exists():
        return []
    out = []
    for folder in sorted(TEST_VIDEOS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label = folder_to_label(folder.name)
        for f in sorted(folder.glob("*.mp4")):
            rel = f.relative_to(TEST_VIDEOS_ROOT)
            out.append({
                "path": str(rel).replace("\\", "/"),
                "label": label,
                "folder": folder.name,
            })
    return out


def extract_frames_from_video(video_path: Path, fps_sample=1):
    """
    Extract frames at fps_sample per second. Returns list of (frame_index, time_sec, PIL.Image).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        cap.release()
        return []
    frame_interval = max(1, int(video_fps) // max(1, fps_sample))
    result = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            time_sec = frame_count // frame_interval
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            result.append((len(result), time_sec, img))
        frame_count += 1
    cap.release()
    return result


def infer_one_video_from_frames(model, processor, device, video_path: Path, ground_truth: str):
    """
    Extract frames from .mp4, run per-frame inference, max-vote, return result with frame-wise distribution.
    """
    frames = extract_frames_from_video(video_path)
    if not frames:
        return {
            "error": "No frames extracted",
            "prediction": None,
            "frame_votes": {"SAFE": 0, "UNSAFE": 0},
            "frame_distribution": [],
            "ground_truth": ground_truth,
        }
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
    if votes["UNSAFE"] >= votes["SAFE"]:
        prediction = "UNSAFE"
    else:
        prediction = "SAFE"
    return {
        "prediction": prediction,
        "frame_votes": votes,
        "frame_distribution": frame_distribution,
        "ground_truth": ground_truth,
        "correct": prediction == ground_truth,
        "total_frames": len(frames),
    }


def group_test_jsonl_by_video(test_jsonl_path: Path):
    """Group test.jsonl by video (dirname of image). Returns list of { video_id, frames, ground_truth }."""
    if not test_jsonl_path.exists():
        return []
    groups = defaultdict(list)
    with open(test_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row.get("image", "")
            response = (row.get("response") or "").strip().upper()
            if not img_path:
                continue
            # video_id = directory containing the frame (e.g. test/safe/4_te1)
            parts = img_path.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            video_id = "/".join(parts[:-1])
            groups[video_id].append({
                "image": img_path,
                "response": "UNSAFE" if "UNSAFE" in response else "SAFE",
            })
    out = []
    for video_id, rows in sorted(groups.items()):
        if not rows:
            continue
        # ground truth from first frame's response
        gt = rows[0]["response"]
        out.append({
            "video_id": video_id,
            "frames": [r["image"] for r in rows],
            "ground_truth": gt,
        })
    return out


def run_video_inference(model, processor, device, video_list, frames_root: Path):
    """Run inference on each frame of each video; return prediction per video by max-vote."""
    results = []
    for v in video_list:
        video_id = v["video_id"]
        frame_paths = v["frames"]
        ground_truth = v["ground_truth"]
        votes = {"SAFE": 0, "UNSAFE": 0}
        frame_predictions = []
        for rel_path in frame_paths:
            image_path = frames_root / rel_path
            pred = predict_one(model, processor, image_path, device)
            if pred in ("SAFE", "UNSAFE"):
                votes[pred] = votes.get(pred, 0) + 1
            frame_predictions.append({"frame": rel_path, "prediction": pred})
        # Max-vote: majority wins; tie -> UNSAFE
        if votes["UNSAFE"] >= votes["SAFE"]:
            prediction = "UNSAFE"
        else:
            prediction = "SAFE"
        results.append({
            "video_id": video_id,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "frame_votes": votes,
            "frame_predictions": frame_predictions,
            "correct": prediction == ground_truth,
        })
    return results


def get_video_results(model, processor, device, max_videos=None):
    """Load test videos from test.jsonl, run video inference, return results."""
    videos = group_test_jsonl_by_video(TEST_JSONL)
    if max_videos is not None:
        videos = videos[:max_videos]
    if not videos:
        return {"error": "No test videos", "results": [], "correct": 0, "total": 0, "accuracy": 0.0}
    results = run_video_inference(model, processor, device, videos, FRAMES_ROOT)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = (correct / total * 100.0) if total else 0.0
    return {
        "results": results,
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 2),
    }


class VideoInferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_html(self, body, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def send_static_file(self, file_path: Path):
        """Serve a file from STATIC_DIR with appropriate Content-Type."""
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
            self.send_json({
                "status": "ok",
                "model_loaded": self.server.model is not None,
                "service": "video-inference (max-vote per video)",
            })
            return
        if path == "/video":
            qs = urllib.parse.parse_qs(parsed.query)
            rel = (qs.get("path") or [None])[0]
            if not rel:
                self.send_response(400)
                self.end_headers()
                return
            rel = rel.lstrip("/").replace("\\", "/")
            video_path = (TEST_VIDEOS_ROOT / rel).resolve()
            try:
                video_path.relative_to(TEST_VIDEOS_ROOT.resolve())
            except ValueError:
                self.send_response(404)
                self.end_headers()
                return
            if not video_path.exists() or not video_path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            size = video_path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    part = range_header[6:].strip().split("-")
                    start = int(part[0]) if part[0] else 0
                    end = int(part[1]) if len(part) > 1 and part[1] else size - 1
                    end = min(end, size - 1)
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(video_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                except (ValueError, OSError):
                    self.send_response(416)
                    self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(video_path, "rb") as f:
                    self.wfile.write(f.read())
            return
        if path == "/videos":
            videos = list_test_videos()
            self.send_json({"videos": videos, "root": str(TEST_VIDEOS_ROOT)})
            return
        if path == "/infer-video":
            if self.server.model is None:
                self.send_json({"error": "Model not loaded"}, status=503)
                return
            qs = urllib.parse.parse_qs(parsed.query)
            rel = (qs.get("path") or [None])[0]
            if not rel:
                self.send_json({"error": "Missing query parameter: path (e.g. path=4_safe_walkway/4_te1.mp4)"}, status=400)
                return
            # Normalize: no leading slash, forward slashes
            rel = rel.lstrip("/").replace("\\", "/")
            video_path = TEST_VIDEOS_ROOT / rel
            if not video_path.exists() or not video_path.is_file():
                self.send_json({"error": f"Video not found: {rel}"}, status=404)
                return
            folder_name = video_path.parent.name
            ground_truth = folder_to_label(folder_name)
            out = infer_one_video_from_frames(
                self.server.model,
                self.server.processor,
                self.server.device,
                video_path,
                ground_truth,
            )
            out["video_path"] = rel
            self.send_json(out)
            return
        if path == "/test-videos":
            if self.server.model is None:
                self.send_json({"error": "Model not loaded", "results": []}, status=503)
                return
            qs = urllib.parse.parse_qs(parsed.query)
            max_v = None
            if "max" in qs:
                try:
                    max_v = int(qs["max"][0])
                except ValueError:
                    pass
            out = get_video_results(
                self.server.model,
                self.server.processor,
                self.server.device,
                max_videos=max_v,
            )
            self.send_json(out)
            return
        if path == "/info":
            self.send_json({
                "service": "SmolVLM2 Safe/Unsafe — Video inference (max-vote)",
                "port": self.server.server_port,
                "endpoints": {
                    "GET /": "UI: load test video, run inference, see frame-wise distribution",
                    "GET /health": "Health check",
                    "GET /video?path=...": "Stream .mp4 file (for playback). path = e.g. 4_safe_walkway/4_te1.mp4",
                    "GET /videos": "List .mp4 test videos (path, label, folder).",
                    "GET /infer-video?path=...": "Infer one .mp4: extract frames, per-frame prediction, max-vote. path = e.g. 4_safe_walkway/4_te1.mp4",
                    "GET /test-videos": "Run on pre-extracted test.jsonl videos. Query: ?max=N.",
                },
            })
            return
        if path == "/":
            self.send_static_file(STATIC_DIR / "index.html")
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


def main():
    parser = argparse.ArgumentParser(description="Video-level inference (max-vote per video)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default %d)" % DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir (default: %s)" % DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    args = parser.parse_args()

    if args.checkpoint:
        adapter_path = Path(args.checkpoint)
        if not adapter_path.is_absolute():
            adapter_path = (OUTPUT_DIR / adapter_path).resolve()
    else:
        # Prefer checkpoint-990 if present
        ckpt_990 = OUTPUT_DIR / DEFAULT_CHECKPOINT
        if (ckpt_990 / "adapter_model.safetensors").exists() or (ckpt_990 / "adapter_model.bin").exists():
            adapter_path = ckpt_990
        else:
            checkpoints = sorted(
                OUTPUT_DIR.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else -1,
            )
            adapter_path = checkpoints[-1] if checkpoints else OUTPUT_DIR
    if not (adapter_path / "adapter_model.safetensors").exists() and not (adapter_path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"No adapter in {adapter_path}")

    print(f"Loading model from {adapter_path}...")
    model, processor = load_model(adapter_path, args.base_model)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    server = HTTPServer(("0.0.0.0", args.port), VideoInferenceHandler)
    server.model = model
    server.processor = processor
    server.device = device
    print(f"Video inference server: http://0.0.0.0:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
