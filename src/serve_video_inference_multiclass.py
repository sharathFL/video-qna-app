#!/usr/bin/env python3
"""
Video-level inference for multiclass (8 classes). Same flow as serve_video_inference
but loads multiclass checkpoint and does max-vote over 8 class names.
Serves on port 8082. Use checkpoint-1266 by default.
"""

import argparse
import json
import threading
from pathlib import Path
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Multiclass paths
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR_MULTICLASS = Path("/workspace/outputs_multiclass")
TEST_JSONL_MULTICLASS = DATA_DIR / "test_multiclass.jsonl"
TEST_VIDEOS_ROOT = DATA_DIR / "Video_Dataset _for_Safe_and_Unsafe_Behaviours" / "Safe_and _Unsafe_Behaviours_Dataset" / "test"
if not TEST_VIDEOS_ROOT.exists():
    TEST_VIDEOS_ROOT = DATA_DIR / "Safe and Unsafe Behaviours Dataset" / "test"

SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_PORT = 8082
DEFAULT_CHECKPOINT = "checkpoint-1266"

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIME_TYPES = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}


def get_multiclass_names():
    """Discover 8 class folder names from test root."""
    if not TEST_VIDEOS_ROOT.exists():
        return []
    return sorted(d.name for d in TEST_VIDEOS_ROOT.iterdir() if d.is_dir())


def make_multiclass_prompt(class_names):
    return (
        "You are a workplace safety inspector reviewing CCTV footage. "
        "Classify the behavior into exactly one of: "
        + ", ".join(class_names)
        + ". Answer with only that exact class name."
    )


def _normalize_multiclass_pred(pred, valid_responses):
    """Map raw model output to one of valid_responses."""
    if not pred or not valid_responses:
        return (pred[:32] if pred else "") or list(valid_responses)[0]
    pred_clean = pred.strip()
    if pred_clean in valid_responses:
        return pred_clean
    for c in sorted(valid_responses, key=len, reverse=True):
        if c in pred:
            return c
    return pred_clean[:32] if pred_clean else list(valid_responses)[0]


def load_model(adapter_path: Path, base_model: str = SMOLVLM2_2B, progress_callback=None):
    def report(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[{pct}%] {msg}")

    use_cuda = torch.cuda.is_available()
    report(5, "Loading processor…")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    report(25, "Loading base model…")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else None,
        trust_remote_code=True,
    )
    report(60, "Loading adapter…")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    report(90, "Finalizing…")
    model.eval()
    report(100, "Ready")
    return model, processor


def predict_multiclass_image(model, processor, img: Image.Image, device, prompt: str, class_names: set):
    """Run multiclass inference on a PIL Image. Returns one of class_names."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
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
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    gen = processor.batch_decode(
        out[:, inp["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw = (gen[0].strip() if gen else "")
    return _normalize_multiclass_pred(raw, class_names)


def list_test_videos():
    """List all .mp4 under TEST_VIDEOS_ROOT. label = folder name (class)."""
    if not TEST_VIDEOS_ROOT.exists():
        return []
    out = []
    for folder in sorted(TEST_VIDEOS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.mp4")):
            rel = f.relative_to(TEST_VIDEOS_ROOT)
            out.append({
                "path": str(rel).replace("\\", "/"),
                "label": folder.name,
                "folder": folder.name,
            })
    return out


def extract_frames_from_video(video_path: Path, fps_sample=1):
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


def infer_one_video_from_frames(model, processor, device, video_path: Path, ground_truth: str, prompt: str, class_names: set):
    frames = extract_frames_from_video(video_path)
    if not frames:
        return {
            "error": "No frames extracted",
            "prediction": None,
            "frame_votes": {c: 0 for c in class_names},
            "frame_distribution": [],
            "ground_truth": ground_truth,
        }
    votes = {c: 0 for c in class_names}
    frame_distribution = []
    for frame_index, time_sec, img in frames:
        pred = predict_multiclass_image(model, processor, img, device, prompt, class_names)
        votes[pred] = votes.get(pred, 0) + 1
        frame_distribution.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "prediction": pred,
            "reason": "",
        })
    prediction = max(votes, key=votes.get)
    return {
        "prediction": prediction,
        "frame_votes": votes,
        "frame_distribution": frame_distribution,
        "ground_truth": ground_truth,
        "correct": prediction == ground_truth,
        "total_frames": len(frames),
        "multiclass": True,
    }


def group_test_jsonl_by_video(test_jsonl_path: Path):
    """Group test_multiclass.jsonl by video. ground_truth = response (class name)."""
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
            response = (row.get("response") or "").strip()
            if not img_path:
                continue
            parts = img_path.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            video_id = "/".join(parts[:-1])
            groups[video_id].append({"image": img_path, "response": response})
    out = []
    for video_id, rows in sorted(groups.items()):
        if not rows:
            continue
        gt = rows[0]["response"]
        out.append({
            "video_id": video_id,
            "frames": [r["image"] for r in rows],
            "ground_truth": gt,
        })
    return out


def run_video_inference(model, processor, device, video_list, frames_root: Path, prompt: str, class_names: set, progress_callback=None):
    results = []
    total = len(video_list)
    for i, v in enumerate(video_list):
        if progress_callback:
            progress_callback(i + 1, total)
        video_id = v["video_id"]
        frame_paths = v["frames"]
        ground_truth = v["ground_truth"]
        votes = {c: 0 for c in class_names}
        frame_predictions = []
        for rel_path in frame_paths:
            image_path = frames_root / rel_path
            if not image_path.exists():
                continue
            img = Image.open(image_path).convert("RGB")
            pred = predict_multiclass_image(model, processor, img, device, prompt, class_names)
            votes[pred] = votes.get(pred, 0) + 1
            frame_predictions.append({"frame": rel_path, "prediction": pred})
        prediction = max(votes, key=votes.get)
        results.append({
            "video_id": video_id,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "frame_votes": votes,
            "frame_predictions": frame_predictions,
            "correct": prediction == ground_truth,
            "multiclass": True,
        })
    return results


def get_video_results(model, processor, device, prompt: str, class_names: set, max_videos=None, progress_callback=None):
    videos = group_test_jsonl_by_video(TEST_JSONL_MULTICLASS)
    if max_videos is not None:
        videos = videos[:max_videos]
    if not videos:
        return {"error": "No test videos", "results": [], "correct": 0, "total": 0, "accuracy": 0.0}
    frames_root = DATA_DIR / "frames_multiclass"
    results = run_video_inference(model, processor, device, videos, frames_root, prompt, class_names, progress_callback=progress_callback)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = (correct / total * 100.0) if total else 0.0
    return {
        "results": results,
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 2),
    }


_test_state = {"in_progress": False, "current": 0, "total": 0, "results": None, "error": None}
_test_state_lock = threading.Lock()


class VideoInferenceMulticlassHandler(BaseHTTPRequestHandler):
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
            self.send_json({
                "status": "ok",
                "model_loaded": self.server.model is not None,
                "service": "video-inference-multiclass (8 classes)",
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
                self.send_json({"error": "Missing query parameter: path"}, status=400)
                return
            rel = rel.lstrip("/").replace("\\", "/")
            video_path = TEST_VIDEOS_ROOT / rel
            if not video_path.exists() or not video_path.is_file():
                self.send_json({"error": f"Video not found: {rel}"}, status=404)
                return
            ground_truth = video_path.parent.name
            out = infer_one_video_from_frames(
                self.server.model,
                self.server.processor,
                self.server.device,
                video_path,
                ground_truth,
                self.server.prompt,
                self.server.class_names,
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
                self.server.prompt,
                self.server.class_names,
                max_videos=max_v,
            )
            self.send_json(out)
            return
        if path == "/run-test":
            if self.server.model is None:
                self.send_json({"error": "Model not loaded"}, status=503)
                return
            with _test_state_lock:
                if _test_state["in_progress"]:
                    self.send_json({"error": "Test already in progress", "current": _test_state["current"], "total": _test_state["total"]}, status=409)
                    return
                _test_state["in_progress"] = True
                _test_state["current"] = 0
                _test_state["total"] = 0
                _test_state["results"] = None
                _test_state["error"] = None

            def run_full_test():
                try:
                    def progress_cb(current, total):
                        with _test_state_lock:
                            _test_state["current"] = current
                            _test_state["total"] = total
                    out = get_video_results(
                        self.server.model,
                        self.server.processor,
                        self.server.device,
                        self.server.prompt,
                        self.server.class_names,
                        max_videos=None,
                        progress_callback=progress_cb,
                    )
                    with _test_state_lock:
                        _test_state["results"] = out
                        _test_state["in_progress"] = False
                except Exception as e:
                    with _test_state_lock:
                        _test_state["error"] = str(e)
                        _test_state["in_progress"] = False

            t = threading.Thread(target=run_full_test, daemon=True)
            t.start()
            self.send_json({"status": "started", "message": "Full test running. Poll /test-status for progress."})
            return
        if path == "/test-status":
            with _test_state_lock:
                state = {
                    "in_progress": _test_state["in_progress"],
                    "current": _test_state["current"],
                    "total": _test_state["total"],
                }
                if _test_state["error"]:
                    state["error"] = _test_state["error"]
                if _test_state["results"] is not None:
                    r = _test_state["results"]
                    state["done"] = True
                    state["correct"] = r.get("correct", 0)
                    state["total"] = r.get("total", 0)
                    state["accuracy"] = r.get("accuracy")
                    state["results"] = r.get("results", [])
                else:
                    state["done"] = False
            self.send_json(state)
            return
        if path == "/info":
            self.send_json({
                "service": "SmolVLM2 — Multiclass video inference (8 classes)",
                "port": self.server.server_port,
                "checkpoint": str(self.server.checkpoint),
                "multiclass": True,
                "endpoints": {
                    "GET /": "UI: select video, run inference (max-vote over 8 classes)",
                    "GET /videos": "List test videos",
                    "GET /infer-video?path=...": "Infer one .mp4",
                    "GET /test-videos?max=N": "Run on N test videos (from test_multiclass.jsonl)",
                    "GET /run-test": "Run full test set",
                    "GET /test-status": "Progress and metrics",
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
    parser = argparse.ArgumentParser(description="Multiclass video inference (8 classes, max-vote per video)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default %d)" % DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name or path (default: %s)" % DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=str, default=SMOLVLM2_2B)
    args = parser.parse_args()

    class_names = get_multiclass_names()
    if not class_names:
        raise RuntimeError("No class folders found under %s" % TEST_VIDEOS_ROOT)
    class_names_set = set(class_names)
    prompt = make_multiclass_prompt(class_names)

    if args.checkpoint:
        adapter_path = Path(args.checkpoint)
        if not adapter_path.is_absolute():
            adapter_path = (OUTPUT_DIR_MULTICLASS / adapter_path).resolve()
    else:
        adapter_path = OUTPUT_DIR_MULTICLASS / DEFAULT_CHECKPOINT
    if not (adapter_path / "adapter_model.safetensors").exists() and not (adapter_path / "adapter_model.bin").exists():
        raise FileNotFoundError("No adapter in %s" % adapter_path)

    print("Loading multiclass model from %s..." % adapter_path)
    model, processor = load_model(adapter_path, args.base_model)
    device = next(model.parameters()).device
    print("Model loaded on %s" % device)

    server = HTTPServer(("0.0.0.0", args.port), VideoInferenceMulticlassHandler)
    server.model = model
    server.processor = processor
    server.device = device
    server.prompt = prompt
    server.class_names = class_names_set
    server.checkpoint = adapter_path
    print("Multiclass video inference server: http://0.0.0.0:%d/" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
