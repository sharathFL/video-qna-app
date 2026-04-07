"""
serve_adminFeature.py  —  Safety Logic Engine
Port 8089 | Fully local | No external API calls

Architecture:
  - YOLO (best.pt) for fast detection: persons, helmets, vests, masks,
    machinery, vehicles, safety cones — answers 80% of criteria in <100ms
  - Qwen2.5-VL 3B via Ollama for reasoning questions — fully local, ~2-4s
  - Question router automatically picks YOLO or Ollama per criterion
  - Face blur applied before sending any frame to Ollama (privacy)
  - Webcam / file upload, frame-change detection, history log — all retained

Drop into ./src/
Usage:
    python src/serve_adminFeature.py \
        --checkpoint /workspace/outputs/best.pt \
        --port 8089
"""

import argparse
import base64
import json
import math
import os
import re
import time
import threading
from io import BytesIO

import cv2
import numpy as np
import psutil
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="/workspace/outputs/best.pt")
parser.add_argument("--person-checkpoint", type=str,
                    default="/workspace/outputs/yolov8n.pt")
parser.add_argument("--port",       type=int, default=8089)
parser.add_argument("--host",       type=str, default="0.0.0.0")
parser.add_argument("--smolvlm-model", type=str,
                    default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    help="HuggingFace model ID or local path for SmolVLM2")
args = parser.parse_args()

app = FastAPI(title="Safety Logic Engine")

# ── Globals ───────────────────────────────────────────────────────────────────
yolo_model        = None
yolo_person_model = None
smolvlm_ready     = False
smolvlm_model     = None
smolvlm_processor = None

# -- Routing Memory Agent --
MEMORY_PATH = "/workspace/outputs/routing_memory.json"
_memory_cache = None  # in-RAM cache, avoids disk read on every frame

def load_memory():
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache
    if os.path.exists(MEMORY_PATH):
        try:
            _memory_cache = json.load(open(MEMORY_PATH))
            return _memory_cache
        except Exception:
            pass
    _memory_cache = {"patterns": []}
    return _memory_cache

def save_memory(mem):
    global _memory_cache
    _memory_cache = mem
    try:
        json.dump(mem, open(MEMORY_PATH, "w"), indent=2)
    except Exception as e:
        print("[Memory] Save failed:", e)


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def cosine_similarity(a, b):
    def tf(tokens):
        d = {}
        for t in tokens:
            d[t] = d.get(t, 0) + 1
        return d
    ta, tb = tf(a), tf(b)
    vocab  = set(ta) | set(tb)
    dot    = sum(ta.get(v, 0) * tb.get(v, 0) for v in vocab)
    na     = math.sqrt(sum(v**2 for v in ta.values()))
    nb     = math.sqrt(sum(v**2 for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0

def find_learned_route(question, memory, threshold=0.82):
    tokens = tokenize(question)
    best_sim, best_pat = 0.0, None
    for pat in memory["patterns"]:
        sim = cosine_similarity(tokens, tokenize(pat["question"]))
        if sim > best_sim:
            best_sim, best_pat = sim, pat
    if best_sim >= threshold and best_pat:
        return best_pat["route"], round(best_sim, 3), best_pat["id"]
    return None, 0.0, None

def upsert_pattern(memory, question, route, quality_good=None):
    tokens = tokenize(question)
    for pat in memory["patterns"]:
        sim = cosine_similarity(tokens, tokenize(pat["question"]))
        if sim >= 0.92:
            pat["route"]          = route
            pat["feedback_count"] = pat.get("feedback_count", 0) + 1
            if quality_good is not None:
                pat["quality_good"]  = (pat.get("quality_good") or 0) + (1 if quality_good else 0)
                pat["quality_total"] = (pat.get("quality_total") or 0) + 1
            save_memory(memory)
            return pat["id"]
    pat_id = "p" + str(len(memory["patterns"]) + 1) + "_" + str(int(time.time()))
    entry = {
        "id":             pat_id,
        "question":       question.lower().strip(),
        "route":          route,
        "feedback_count": 1,
        "quality_good":   (1 if quality_good else 0) if quality_good is not None else None,
        "quality_total":  1 if quality_good is not None else 0,
    }
    memory["patterns"].append(entry)
    save_memory(memory)
    return pat_id

# -- YOLO class names from best.pt ─────────────────────────────────────────────
# {0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask, 4: NO-Safety Vest,
#  5: Person,  6: Safety Cone, 7: Safety Vest, 8: machinery, 9: vehicle}
YOLO_CLASSES = {}

# ── Question router ───────────────────────────────────────────────────────────
# YOLO can only answer questions about:
#   1. Counting/presence of specific detectable objects
#   2. PPE compliance (wearing/not wearing a specific item)
#
# Routing logic: question must match BOTH a detectable object AND a
# count/presence/compliance intent. Pure reasoning questions go to Ollama
# even if they mention a detectable object.

YOLO_OBJECTS = [
    "helmet", "hardhat", "hard hat",
    "vest", "safety vest", "high-vis",
    "mask",
    "machinery", "vehicle", "forklift", "truck",
    "cone", "safety cone",
    "person", "people", "worker", "workers",
    "ppe",
]

# These intents mean the question is about counting or binary presence
YOLO_INTENTS = [
    "how many", "count", "number of",
    "wearing", "not wearing", "without",
    "present", "visible", "detected", "any ",
    "all ", "is there", "are there",
    "have ", "has ",
]

# These words signal reasoning is needed — always send to VLM.
# IMPORTANT: keep these specific — do NOT add words that overlap with
# counting/presence intent (e.g. "visible", "have", "all", "any", "look").
REASONING_SIGNALS = [
    "correctly", "properly", "safely", "appropriately",
    "appear to be", "seems to be", "behav",
    "posture", "organized", "clean",
    "right way", "correct way",
    "risk", "danger", "hazard",
    "idle", "distract", "fatigue", "tired",
    # Opinion / sentiment / open-ended — needs VLM
    "sentiment", "emotion", "mood",
    "describe", "explain", "what is happening",
    "overall", "assess", "evaluate",
]

# These intents confirm YOLO can handle it — checked AFTER reasoning signals.
YOLO_INTENTS = [
    "how many", "count", "number of",
    "wearing", "not wearing", "without",
    "present", "visible", "detected",
    "any ", "all ", "is there", "are there",
    "have ", "has ",
]

def is_yolo_question(q: str) -> bool:
    """
    Route to YOLO if the question has a clear detection/counting intent
    AND no reasoning signal. Reasoning signals take priority.
    Fallback (no match either way) also goes to YOLO since YOLO runs
    unconditionally and can always return a generic detection summary.
    """
    lower = q.lower()
    # Reasoning signals take absolute priority
    if any(sig in lower for sig in REASONING_SIGNALS):
        return False
    # Explicit YOLO-compatible intent found → YOLO
    if any(intent in lower for intent in YOLO_INTENTS):
        return True
    # Question mentions a detectable object → YOLO can at least count it
    if any(obj in lower for obj in YOLO_OBJECTS):
        return True
    # Truly open-ended with no detectable object → VLM
    return False


# ── YOLO answer builder ───────────────────────────────────────────────────────
def answer_with_yolo(question: str, detections: dict) -> str:
    """
    Map YOLO detection counts to a natural language answer.
    detections = {class_name_lower: count}
    """
    q = question.lower()

    persons   = detections.get("person", 0)
    helmets   = detections.get("hardhat", 0)
    no_helmet = detections.get("no-hardhat", 0)
    vests     = detections.get("safety vest", 0)
    no_vest   = detections.get("no-safety vest", 0)
    masks     = detections.get("mask", 0)
    no_mask   = detections.get("no-mask", 0)
    machinery = detections.get("machinery", 0)
    vehicles  = detections.get("vehicle", 0)
    cones     = detections.get("safety cone", 0)

    # ── Person / count questions ──
    if any(kw in q for kw in ["how many", "count", "number of"]):
        if any(kw in q for kw in ["person", "people", "worker"]):
            return f"{persons} person(s) detected in the frame."
        if any(kw in q for kw in ["helmet", "hardhat"]):
            return f"{helmets} helmet(s) detected, {no_helmet} without helmet."
        if "vest" in q:
            return f"{vests} safety vest(s) detected, {no_vest} without vest."
        if "vehicle" in q or "forklift" in q:
            return f"{vehicles} vehicle(s) detected."
        if "machine" in q or "equipment" in q:
            return f"{machinery} piece(s) of machinery detected."

    # ── Helmet compliance ──
    if any(kw in q for kw in ["helmet", "hardhat", "hard hat"]):
        if "wearing" in q or "have" in q or "all" in q or "is" in q or "are" in q:
            if persons == 0:
                return "No persons detected in the frame."
            if no_helmet == 0 and helmets > 0:
                return f"Yes — all {helmets} worker(s) are wearing helmets."
            elif no_helmet > 0:
                return f"No — {no_helmet} worker(s) detected without a helmet."
            else:
                return f"{helmets} helmet(s) detected among {persons} person(s)."

    # ── Vest compliance ──
    if "vest" in q:
        if persons == 0:
            return "No persons detected in the frame."
        if no_vest == 0 and vests > 0:
            return f"Yes — all {vests} worker(s) are wearing safety vests."
        elif no_vest > 0:
            return f"No — {no_vest} worker(s) detected without a safety vest."
        else:
            return f"{vests} safety vest(s) detected among {persons} person(s)."

    # ── Mask compliance ──
    if "mask" in q:
        if persons == 0:
            return "No persons detected in the frame."
        if no_mask == 0 and masks > 0:
            return f"Yes — all {masks} worker(s) are wearing masks."
        elif no_mask > 0:
            return f"No — {no_mask} worker(s) detected without a mask."
        else:
            return f"{masks} mask(s) detected among {persons} person(s)."

    # ── Machinery / vehicle presence ──
    if any(kw in q for kw in ["machine", "equipment", "machinery"]):
        if machinery > 0:
            return f"Yes — {machinery} piece(s) of machinery detected."
        return "No machinery detected in the frame."

    if any(kw in q for kw in ["vehicle", "forklift", "truck"]):
        if vehicles > 0:
            return f"Yes — {vehicles} vehicle(s) detected."
        return "No vehicles detected in the frame."

    # ── Safety cone / restricted zone ──
    if "cone" in q:
        if cones > 0:
            return f"Yes — {cones} safety cone(s) visible."
        return "No safety cones detected in the frame."

    # ── Generic PPE fallback ──
    if any(kw in q for kw in ["ppe", "protective", "equipment"]):
        issues = []
        if no_helmet > 0: issues.append(f"{no_helmet} without helmet")
        if no_vest   > 0: issues.append(f"{no_vest} without vest")
        if no_mask   > 0: issues.append(f"{no_mask} without mask")
        if issues:
            return "PPE violations detected: " + ", ".join(issues) + "."
        return f"All {persons} worker(s) appear to have required PPE."

    # Fallback
    return f"Detected — persons: {persons}, helmets: {helmets}, vests: {vests}, machinery: {machinery}, vehicles: {vehicles}."


# ── Face blur ─────────────────────────────────────────────────────────────────
def blur_faces(pil_image: Image.Image) -> Image.Image:
    """Detect and blur faces before sending to Ollama."""
    img_cv  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi = img_cv[y:y+h, x:x+w]
        img_cv[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51, 51), 0)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ── Ollama vision call ────────────────────────────────────────────────────────
def ask_smolvlm(image: Image.Image, questions: list) -> tuple:
    """
    Send all questions to SmolVLM2-500M in a single forward pass.
    Mirrors the working inference pattern from serve_vlm_qa.py.
    Returns (answer_map {index: answer}, raw_string)
    """
    # Blur faces before sending to VLM
    image = blur_faces(image)

    # Resize to nearest 384 multiple — SmolVLM2 patch requirement
    w = max(round(image.width  / 384) * 384, 384)
    h = max(round(image.height / 384) * 384, 384)
    image = image.resize((w, h))

    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = (
        "You are a safety inspector analyzing a CCTV frame from an industrial site.\n"
        "Answer each question based strictly on what you see.\n"
        "For counting questions: count carefully and give the exact number.\n"
        "For PPE questions: check each visible person individually.\n"
        "Keep each answer to one concise sentence.\n\n"
        f"Questions:\n{questions_block}\n\n"
        "Answers (number each answer to match the question number):"
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]

    text   = smolvlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = smolvlm_processor(text=text, images=[image], return_tensors="pt")

    with torch.no_grad():
        generated_ids = smolvlm_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,   # greedy — fastest + most deterministic
        )

    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    raw = smolvlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # Parse numbered answers
    answer_map = {}
    for match in re.finditer(r'(?m)^\s*(\d+)[.)]\s*(.+)', raw):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(questions):
            answer_map[idx] = match.group(2).strip()

    # Fallback: split by newline
    if not answer_map:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        for i, line in enumerate(lines[:len(questions)]):
            answer_map[i] = line

    return answer_map, raw


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global yolo_model, YOLO_CLASSES, yolo_person_model
    global smolvlm_model, smolvlm_processor, smolvlm_ready

    # Load YOLO
    print(f"[SafetyEngine] Loading YOLO checkpoint: {args.checkpoint}")
    yolo_model   = YOLO(args.checkpoint)
    YOLO_CLASSES = {v.lower(): k for k, v in yolo_model.names.items()}
    print(f"[SafetyEngine] YOLO PPE ready. Classes: {list(yolo_model.names.values())}")

    try:
        yolo_person_model = YOLO(args.person_checkpoint)
        print(f"[SafetyEngine] YOLO Person model ready: {args.person_checkpoint}")
    except Exception as e:
        print(f"[SafetyEngine] Person model not found, using best.pt for persons: {e}")
        yolo_person_model = yolo_model

    # Load SmolVLM2
    print(f"[SafetyEngine] Loading SmolVLM2: {args.smolvlm_model}")
    try:
        smolvlm_processor = AutoProcessor.from_pretrained(args.smolvlm_model)
        smolvlm_model     = AutoModelForImageTextToText.from_pretrained(
            args.smolvlm_model,
            torch_dtype=torch.float32,
        )
        smolvlm_model.eval()
        torch.set_grad_enabled(False)
        smolvlm_ready = True
        print("[SafetyEngine] ✅ SmolVLM2 ready.")
    except Exception as e:
        print(f"[SafetyEngine] ❌ SmolVLM2 load failed: {e}")
        smolvlm_ready = False

    print("[SafetyEngine] Ready.")



HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Safety Logic Engine</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --bg:      #0d1117;
  --surf:    #161b22;
  --surf2:   #21262d;
  --border:  #30363d;
  --accent:  #f0b429;
  --green:   #238636;
  --green-t: #3fb950;
  --red:     #da3633;
  --red-t:   #ff7b72;
  --yel:     #9e6a03;
  --yel-t:   #e3b341;
  --text:    #e6edf3;
  --muted:   #8b949e;
  --r:       10px;
  --mono:    'IBM Plex Mono', monospace;
  --sans:    'DM Sans', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}

header{background:var(--surf);border-bottom:1px solid var(--border);height:54px;padding:0 22px;display:flex;align-items:center;gap:11px;flex-shrink:0}
.logo{width:32px;height:32px;background:linear-gradient(135deg,#f0b429,#e05c00);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px}
header h1{font-size:1rem;font-weight:700}
.pill{border-radius:20px;padding:3px 10px;font-size:.7rem;font-weight:600;border:1px solid}
.pill-green{background:#1a2e1a;color:var(--green-t);border-color:var(--green)}
.pill-gray{background:var(--surf2);color:var(--muted);border-color:var(--border);font-family:var(--mono);font-weight:400}
header .right{margin-left:auto;display:flex;gap:8px;align-items:center}

.layout{display:grid;grid-template-columns:310px 1fr;flex:1;overflow:hidden}

aside{background:var(--surf);border-right:1px solid var(--border);overflow-y:auto;padding:16px 14px;display:flex;flex-direction:column;gap:16px}
.sec{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted);margin-bottom:7px}

select.zone-sel{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:var(--r);padding:8px 10px;font-size:.85rem}
select.zone-sel:focus{outline:none;border-color:var(--accent)}

#crit-list{display:flex;flex-direction:column;gap:6px}
.crit-item{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:8px 10px;display:flex;align-items:flex-start;gap:7px}
.cnum{min-width:19px;height:19px;background:var(--surf2);border-radius:50%;font-size:.62rem;font-weight:700;color:var(--muted);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.crit-item textarea{flex:1;background:transparent;border:none;color:var(--text);font-size:.8rem;resize:none;outline:none;line-height:1.4;font-family:var(--sans)}
.del{background:none;border:none;color:var(--muted);cursor:pointer;font-size:.85rem;flex-shrink:0;transition:color .15s}
.del:hover{color:var(--red-t)}
.add-crit{width:100%;background:none;border:1px dashed var(--border);color:var(--muted);border-radius:var(--r);padding:7px 10px;font-size:.8rem;cursor:pointer;transition:all .15s;margin-top:5px}
.add-crit:hover{border-color:var(--accent);color:var(--accent)}

.sev-rules{display:flex;flex-direction:column;gap:5px}
.sev-row{display:flex;align-items:center;gap:7px;background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:7px 10px;font-size:.78rem}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.sev-row select{margin-left:auto;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:2px 6px;font-size:.75rem}

#go-btn{width:100%;padding:11px;background:linear-gradient(135deg,#f0b429,#e05c00);color:#000;border:none;border-radius:var(--r);font-size:.9rem;font-weight:700;cursor:pointer;transition:opacity .15s,transform .1s;letter-spacing:.3px;font-family:var(--sans)}
#go-btn:hover{opacity:.88;transform:translateY(-1px)}
#go-btn:active{transform:translateY(0)}
#go-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

.main{display:flex;flex-direction:column;overflow:hidden}
.main-body{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:18px}

/* ── Source toggle ── */
.source-toggle{display:flex;gap:0;border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:12px}
.src-btn{flex:1;padding:8px 12px;background:var(--bg);border:none;color:var(--muted);font-size:.82rem;font-family:var(--sans);font-weight:600;cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:6px}
.src-btn.active{background:var(--surf2);color:var(--text)}
.src-btn:first-child{border-right:1px solid var(--border)}

/* ── Upload zone ── */
.upload-zone{border:2px dashed var(--border);border-radius:14px;min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;cursor:pointer;transition:border-color .2s,background .2s;position:relative;overflow:hidden}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(240,180,41,.04)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone .icon{font-size:2.2rem}
.upload-zone p{color:var(--muted);font-size:.85rem}
.upload-zone strong{color:var(--accent)}
#prev-wrap{text-align:center;display:none}
#prev-img{max-height:300px;border-radius:10px;object-fit:contain;display:none}
.img-meta{font-size:.72rem;color:var(--muted);margin-top:5px}

/* ── Webcam panel ── */
#webcam-panel{display:none;flex-direction:column;gap:10px}
.webcam-container{position:relative;border-radius:14px;overflow:hidden;background:#000;border:2px solid var(--border)}
#webcam-video{width:100%;max-height:300px;object-fit:cover;display:block}
#webcam-canvas{display:none}
.webcam-overlay{position:absolute;top:10px;left:10px;display:flex;gap:6px;align-items:center}
.rec-dot{width:10px;height:10px;border-radius:50%;background:var(--red-t);animation:blink 1s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.rec-label{font-size:.7rem;font-weight:700;color:#fff;font-family:var(--mono);text-shadow:0 1px 3px rgba(0,0,0,.8)}
.webcam-badge{position:absolute;top:10px;right:10px;background:rgba(13,17,23,.8);border:1px solid var(--border);border-radius:8px;padding:4px 10px;font-size:.7rem;font-family:var(--mono);color:var(--muted)}

.webcam-controls{display:flex;gap:8px;align-items:center}
#cam-start-btn{flex:1;padding:9px 12px;background:var(--green);color:#fff;border:none;border-radius:var(--r);font-size:.85rem;font-weight:700;cursor:pointer;transition:all .15s;font-family:var(--sans)}
#cam-start-btn:hover{opacity:.85}
#cam-start-btn.streaming{background:var(--red)}
#cam-snap-btn{padding:9px 13px;background:var(--surf2);border:1px solid var(--border);color:var(--text);border-radius:var(--r);font-size:.82rem;cursor:pointer;transition:all .15s;font-family:var(--sans)}
#cam-snap-btn:hover{border-color:var(--accent);color:var(--accent)}
#cam-snap-btn:disabled{opacity:.35;cursor:not-allowed}

/* Auto-inference row */
.auto-row{display:flex;align-items:center;gap:10px;background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:8px 12px}
.auto-label{font-size:.78rem;color:var(--muted);white-space:nowrap}
.toggle-switch{position:relative;width:36px;height:20px;flex-shrink:0}
.toggle-switch input{opacity:0;width:0;height:0;position:absolute}
.toggle-slider{position:absolute;inset:0;background:var(--border);border-radius:20px;cursor:pointer;transition:.2s}
.toggle-slider:before{content:'';position:absolute;width:14px;height:14px;left:3px;top:3px;background:var(--muted);border-radius:50%;transition:.2s}
.toggle-switch input:checked+.toggle-slider{background:var(--accent)}
.toggle-switch input:checked+.toggle-slider:before{transform:translateX(16px);background:#000}
#interval-wrap{display:flex;align-items:center;gap:7px;margin-left:auto}
#interval-slider{-webkit-appearance:none;width:90px;height:4px;border-radius:2px;background:var(--border);outline:none;cursor:pointer}
#interval-slider::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--accent);cursor:pointer}
#interval-val{font-size:.72rem;font-family:var(--mono);color:var(--accent);min-width:32px}

/* Change detection indicator */
.change-row{display:flex;align-items:center;gap:8px;font-size:.75rem;color:var(--muted);padding:4px 2px}
#change-indicator{width:8px;height:8px;border-radius:50%;background:var(--border);transition:background .3s;flex-shrink:0}
#change-indicator.changed{background:var(--accent)}
#change-indicator.same{background:var(--green-t)}
#hash-display{font-family:var(--mono);font-size:.65rem;color:var(--border);margin-left:auto}

/* Results */
.res-header{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.verdict{padding:5px 16px;border-radius:20px;font-weight:700;font-size:.95rem;letter-spacing:.4px;border:1px solid;font-family:var(--mono)}
.v-safe{background:#1a2e1a;color:var(--green-t);border-color:var(--green)}
.v-warn{background:#2e2a14;color:var(--yel-t);border-color:var(--yel)}
.v-crit{background:#2e1414;color:var(--red-t);border-color:var(--red)}

.crit-results{display:flex;flex-direction:column;gap:9px}
.res-card{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:13px 15px;display:grid;grid-template-columns:auto 1fr auto;gap:9px;align-items:start}
.res-card.pass{border-left:3px solid var(--green)}
.res-card.fail{border-left:3px solid var(--red)}
.res-card .ri{font-size:1.2rem;margin-top:1px}
.q-text{font-weight:600;font-size:.84rem;margin-bottom:3px}
.obs-text{font-size:.77rem;color:var(--muted);line-height:1.5}
.ans-pill{padding:3px 9px;border-radius:20px;font-size:.72rem;font-weight:700;white-space:nowrap;border:1px solid;font-family:var(--mono)}
.ans-yes{background:#1a2e1a;color:var(--green-t);border-color:var(--green)}
.ans-no{background:#2e1414;color:var(--red-t);border-color:var(--red)}

.res-panel{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px}
.res-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:10px}
.res-tile{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center}
.res-tile .val{font-size:1.1rem;font-weight:700;color:var(--accent);font-family:var(--mono)}
.res-tile .lbl{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}
.ram-bar-bg{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:8px}
.ram-bar{height:100%;border-radius:3px;transition:width .6s ease}

.raw-box{background:#0a0f15;border:1px solid var(--border);border-radius:var(--r);padding:13px 15px}
.raw-box pre{font-family:var(--mono);font-size:.75rem;color:#7ee787;white-space:pre-wrap;word-break:break-word;line-height:1.6}

.empty{text-align:center;padding:50px 20px;color:var(--muted)}
.empty .big{font-size:2.8rem;margin-bottom:12px}

.statusbar{background:var(--surf);border-top:1px solid var(--border);padding:6px 18px;font-size:.72rem;color:var(--muted);display:flex;align-items:center;gap:12px;flex-shrink:0}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--green-t)}

/* inline loading spinner in statusbar */
.spin-inline{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;display:none;flex-shrink:0}
.spin-inline.on{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
/* stop inference button */
#stop-btn{padding:3px 10px;background:var(--red);color:#fff;border:none;border-radius:6px;font-size:.72rem;font-weight:700;cursor:pointer;display:none;font-family:var(--sans)}
#stop-btn.on{display:inline-block}

#toast{position:fixed;bottom:44px;right:18px;background:var(--surf2);border:1px solid var(--border);border-radius:10px;padding:11px 16px;font-size:.82rem;transform:translateY(70px);opacity:0;transition:all .3s ease;z-index:1000;max-width:300px}
#toast.show{transform:translateY(0);opacity:1}

#analyzed-thumb-wrap{display:none;margin-bottom:12px}
#analyzed-thumb{max-height:110px;border-radius:8px;border:1px solid var(--border);object-fit:contain}
.analyzed-lbl{font-size:.65rem;color:var(--muted);margin-top:4px;font-family:var(--mono)}
</style>
</head>
<body>

<header>
  <div class="logo">&#x1F6E1;&#xFE0F;</div>
  <h1>Safety Logic Engine</h1>
  <div class="right">
    <span class="pill pill-gray" id="ckpt-label">loading...</span>
    <span class="pill" id="model-pill" style="background:#2e2a14;color:var(--yel-t);border-color:var(--yel)">&#x25CF; Loading...</span>
  </div>
</header>

<div class="layout">
  <aside>
    <div>
      <div class="sec">&#x1F4CD; Camera Zone</div>
      <select class="zone-sel" id="zone" onchange="loadPreset()">
        <option value="walkway">Walkway / Corridor</option>
        <option value="loading">Loading Dock</option>
        <option value="production">Production Line</option>
        <option value="heights">Working at Heights</option>
        <option value="hazmat">Hazardous Materials</option>
        <option value="custom">Custom Zone</option>
      </select>
    </div>

    <div>
      <div class="sec">&#x2705; Safety Criteria</div>
      <div id="crit-list"></div>
      <button class="add-crit" onclick="addCrit()">&#xFF0B; Add Criterion</button>
    </div>

    <div>
      <div class="sec">&#x26A0;&#xFE0F; Severity Rules</div>
      <div class="sev-rules">
        <div class="sev-row">
          <span class="dot" style="background:var(--green-t)"></span>
          <span>All pass</span>
          <select id="sev0"><option>SAFE</option></select>
        </div>
        <div class="sev-row">
          <span class="dot" style="background:var(--yel-t)"></span>
          <span>1 fails</span>
          <select id="sev1">
            <option>WARNING</option>
            <option>CRITICAL</option>
          </select>
        </div>
        <div class="sev-row">
          <span class="dot" style="background:var(--red-t)"></span>
          <span>2+ fail</span>
          <select id="sev2">
            <option>CRITICAL</option>
            <option>WARNING</option>
          </select>
        </div>
      </div>
    </div>

    <button id="go-btn" onclick="runAnalysis()">&#x1F50D; Analyze Frame</button>
  </aside>

  <div class="main">
    <div class="main-body">

      <div>
        <div class="sec">&#x1F4F7; Input Source</div>

        <!-- Source toggle -->
        <div class="source-toggle" style="display:flex;gap:0;border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:12px">
          <button class="src-btn active" id="btn-upload" onclick="setSource('upload')" style="flex:1;padding:8px 6px;font-size:.78rem">
            &#x1F4C2; File
          </button>
          <button class="src-btn" id="btn-webcam" onclick="setSource('webcam')" style="flex:1;padding:8px 6px;font-size:.78rem;border-left:1px solid var(--border)">
            &#x1F4F9; Camera
          </button>
          <button class="src-btn" id="btn-screen" onclick="setSource('screen')" style="flex:1;padding:8px 6px;font-size:.78rem;border-left:1px solid var(--border)">
            &#x1F5A5; Screen
          </button>
        </div>

        <!-- File upload panel -->
        <div id="upload-panel">
          <div class="upload-zone" id="drop-zone"
               ondragover="onDragOver(event)" ondragleave="onDragLeave()" ondrop="onDrop(event)">
            <input type="file" id="file-in" accept="image/*" onchange="onFileChange(event)"/>
            <div class="icon">&#x1F4C2;</div>
            <p><strong>Click or drag</strong> a CCTV frame here</p>
            <p>PNG &middot; JPG &middot; JPEG</p>
          </div>
          <div id="prev-wrap">
            <img id="prev-img" alt="preview" style="display:none"/>
            <div class="img-meta" id="img-meta"></div>
          </div>
        </div>

        <!-- Webcam panel -->
        <div id="webcam-panel" style="display:none;flex-direction:column;gap:10px">
          <div class="webcam-container">
            <video id="webcam-video" autoplay muted playsinline></video>
            <canvas id="webcam-canvas" style="display:none"></canvas>
            <div class="webcam-overlay">
              <div class="rec-dot" id="rec-dot" style="display:none"></div>
              <span class="rec-label" id="rec-label" style="display:none">LIVE</span>
            </div>
            <div class="webcam-badge" id="cam-res-badge">No camera</div>
          </div>

          <div class="webcam-controls" style="display:flex;flex-direction:column;gap:6px">
            <button id="cam-snap-start-btn" onclick="handleStartBtn()" style="width:100%">&#x25B6; Start Camera</button>
            <button id="cam-snap-btn" onclick="snapAndAnalyze()" disabled style="width:100%">&#x1F4F8; Analyze Now</button>
          </div>

          <div class="auto-row">
            <span class="auto-label">Auto-infer on change</span>
            <label class="toggle-switch">
              <input type="checkbox" id="auto-toggle" onchange="onAutoToggle()">
              <span class="toggle-slider"></span>
            </label>
            <div id="interval-wrap">
              <span class="auto-label">every</span>
              <input type="range" id="interval-slider" min="3" max="60" value="10"
                     oninput="document.getElementById('interval-val').textContent=this.value+'s'"/>
              <span id="interval-val">10s</span>
            </div>
          </div>

          <div class="change-row">
            <div id="change-indicator"></div>
            <span id="change-status">Waiting for camera...</span>
            <span id="hash-display"></span>
          </div>
        </div>
      </div>

      <!-- History log — one card per inference, newest on top -->
      <div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
          <div class="sec" style="margin:0">&#x1F4CA; Inference History</div>
          <button onclick="document.getElementById('history-list').innerHTML='';document.getElementById('empty').style.display='block'"
            style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:3px 10px;font-size:.72rem;cursor:pointer;font-family:var(--sans)">
            Clear
          </button>
        </div>
        <div id="history-list" style="display:flex;flex-direction:column;gap:14px"></div>
      </div>

      <div id="empty" class="empty">
        <div class="big">&#x1F3AF;</div>
        <p>Upload a frame or start your webcam,<br>configure criteria, then click <strong>Analyze Frame</strong>.</p>
      </div>
    </div>

    <div class="statusbar">
      <span class="sdot"></span>
      <span>SmolVLM2 &middot; CPU</span>
      <span>|</span>
      <div class="spin-inline" id="inline-spinner"></div>
      <span id="status-msg">Idle</span>
      <button id="stop-btn" onclick="stopInference()">&#x25A0; Stop</button>
    </div>
  </div>
</div>

<!-- Lightbox for frame preview -->
<div id="lightbox" onclick="this.style.display='none'"
  style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;align-items:center;justify-content:center;cursor:zoom-out">
  <img id="lightbox-img" style="max-width:90vw;max-height:90vh;border-radius:10px;border:1px solid var(--border);box-shadow:0 8px 40px rgba(0,0,0,.6)"/>
</div>
<div id="toast"></div>

<script>
// ── Presets ───────────────────────────────────────────────────────────────────
const PRESETS = {
  walkway:    ["Is the worker staying within the yellow floor safety markings?",
               "Is the worker wearing a high-visibility safety vest?",
               "Is the walkway free of obstructions or trip hazards?"],
  loading:    ["Is the worker maintaining a safe distance from any active forklift?",
               "Is the worker wearing a hard hat?",
               "Are the loading bay barriers correctly positioned?"],
  production: ["Are all machine guards visibly in place and not bypassed?",
               "Is the worker keeping hands and clothing away from moving parts?",
               "Is the worker wearing appropriate PPE such as gloves and goggles?"],
  heights:    ["Is the worker wearing a safety harness?",
               "Is the harness anchor point correctly secured to a fixed structure?",
               "Is a guardrail or safety net visible below the work area?"],
  hazmat:     ["Is the worker wearing a full-face respirator or appropriate mask?",
               "Are chemical storage containers properly sealed and labeled?",
               "Is the worker wearing chemical-resistant gloves and protective clothing?"],
  custom:     ["Is the worker wearing the required PPE for this zone?"]
};

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile         = null;
let currentSource       = 'upload';
let camStream           = null;
let camRunning          = false;
let autoInterval        = null;
let inferBusy           = false;
let abortController     = null;
let lastHash            = null;
let lastAnalyzedDataUrl = null;

// Hamming distance threshold — frames closer than this are "same scene", skip inference
const HASH_THRESHOLD = 10;

// ── Init ──────────────────────────────────────────────────────────────────────
window.onload = () => { loadPreset(); pollHealth(); };

// ── Health poll ───────────────────────────────────────────────────────────────
function pollHealth() {
  fetch('/health').then(r=>r.json()).then(d=>{
    const pill = document.getElementById('model-pill');
    document.getElementById('ckpt-label').textContent = (d.checkpoint||'').split('/').pop()||'checkpoint';
    if (d.model_loaded) {
      pill.textContent = '\u25CF Model Ready';
      pill.style.cssText = 'background:#1a2e1a;color:var(--green-t);border-color:var(--green)';
      setStatus('RAM ' + d.system_ram_used_mb + 'MB / ' + d.system_ram_total_mb + 'MB  \u00b7  CPU ' + d.cpu_pct + '%');
    } else {
      pill.textContent = '\u25CF Loading model...';
      pill.style.cssText = 'background:#2e2a14;color:var(--yel-t);border-color:var(--yel)';
      setTimeout(pollHealth, 2000);
    }
  }).catch(()=>setTimeout(pollHealth, 3000));
}

// ── Source toggle ─────────────────────────────────────────────────────────────
function setSource(src) {
  currentSource = src;
  document.getElementById('btn-upload').classList.toggle('active', src==='upload');
  document.getElementById('btn-webcam').classList.toggle('active', src==='webcam');
  document.getElementById('btn-screen').classList.toggle('active', src==='screen');
  const up = document.getElementById('upload-panel');
  const wp = document.getElementById('webcam-panel');
  up.style.display = src==='upload' ? 'block' : 'none';
  wp.style.display = (src==='webcam'||src==='screen') ? 'flex' : 'none';
  if (src === 'upload') { stopCamera(); return; }
  // Update the single start button label based on selected source
  const startBtn = document.getElementById('cam-snap-start-btn');
  if (startBtn) {
    startBtn.textContent = src === 'screen' ? '\uD83D\uDDA5 Start Screen Share' : '\u25B6 Start Camera';
  }
}


// -- Camera / Screen share --
let camSource = 'camera';

async function toggleCamera(source) {
  if (camRunning) { stopCamera(); return; }
  camSource = source || 'camera';
  await startCamera();
}

function handleStartBtn() {
  if (camRunning) { stopCamera(); return; }
  // Direct user gesture — safe to call getDisplayMedia here
  camSource = currentSource === 'screen' ? 'screen' : 'camera';
  startCamera();
}

async function startCamera() {
  try {
    if (camSource === 'screen') {
      camStream = await navigator.mediaDevices.getDisplayMedia({
        video: { width:{ideal:1920}, height:{ideal:1080} },
        audio: false
      });
    } else {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:1280}, height:{ideal:720}, facingMode:'environment' },
        audio: false
      });
    }
    const video = document.getElementById('webcam-video');
    video.srcObject = camStream;
    await video.play();
    camRunning = true;

    // update start button to show stop
    const startBtn = document.getElementById('cam-snap-start-btn');
    if (startBtn) {
      startBtn.textContent = '\u25A0 Stop ' + (camSource === 'screen' ? 'Screen' : 'Camera');
    }

    document.getElementById('cam-snap-btn').disabled = false;
    document.getElementById('rec-dot').style.display  = 'block';
    document.getElementById('rec-label').style.display = 'block';

    video.addEventListener('loadedmetadata', () => {
      document.getElementById('cam-res-badge').textContent =
        video.videoWidth + '\u00d7' + video.videoHeight;
    }, {once:true});

    // Auto-stop when user ends screen share from browser UI
    camStream.getVideoTracks()[0].addEventListener('ended', () => stopCamera());

    const label = camSource === 'screen' ? 'Screen share active' : 'Camera running';
    setStatus(label + ' \u2014 configure criteria and analyze');
    toast(camSource === 'screen' ? '\uD83D\uDDA5 Screen share started' : '\uD83D\uDCF9 Camera started');
  } catch(err) {
    toast('\u274c ' + err.message);
    console.error(err);
  }
}

function stopCamera() {
  if (camStream) { camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  camRunning = false;
  stopAutoInfer();
  document.getElementById('auto-toggle').checked = false;
  // reset start button
  const startBtn = document.getElementById('cam-snap-start-btn');
  if (startBtn) {
    startBtn.textContent = currentSource === 'screen' ? '\uD83D\uDDA5 Start Screen Share' : '\u25B6 Start Camera';
  }
  document.getElementById('cam-snap-btn').disabled = true;
  document.getElementById('rec-dot').style.display  = 'none';
  document.getElementById('rec-label').style.display = 'none';
  document.getElementById('cam-res-badge').textContent = 'No camera';
  document.getElementById('change-status').textContent = 'Waiting for camera...';
  document.getElementById('change-indicator').className = '';
  document.getElementById('hash-display').textContent = '';
  lastHash = null;
  setStatus('Stopped');
}


// ── Capture helpers ───────────────────────────────────────────────────────────
function captureFrame() {
  const video  = document.getElementById('webcam-video');
  const canvas = document.getElementById('webcam-canvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function canvasToBlob(canvas) {
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

// ── Perceptual hash (average hash, 16×16 grayscale) ──────────────────────────
function computeHash(canvas, size=16) {
  const tmp = document.createElement('canvas');
  tmp.width = size; tmp.height = size;
  const ctx = tmp.getContext('2d');
  ctx.drawImage(canvas, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  const gray = [];
  for (let i=0; i<data.length; i+=4)
    gray.push(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]);
  const avg = gray.reduce((a,b)=>a+b,0)/gray.length;
  let bits = gray.map(v => v>=avg ? '1' : '0').join('');
  let hex = '';
  for (let i=0; i<bits.length; i+=4)
    hex += parseInt(bits.slice(i,i+4),2).toString(16);
  return hex;
}

function hammingDist(h1, h2) {
  let dist = 0;
  for (let i=0; i<Math.min(h1.length,h2.length); i++) {
    const b1 = parseInt(h1[i],16).toString(2).padStart(4,'0');
    const b2 = parseInt(h2[i],16).toString(2).padStart(4,'0');
    for (let j=0; j<4; j++) if (b1[j]!==b2[j]) dist++;
  }
  return dist;
}

// ── Auto-inference ────────────────────────────────────────────────────────────
function onAutoToggle() {
  const on = document.getElementById('auto-toggle').checked;
  if (on) {
    if (!camRunning) {
      toast('\u26a0\ufe0f Start the camera first');
      document.getElementById('auto-toggle').checked = false;
      return;
    }
    startAutoInfer();
  } else {
    stopAutoInfer();
  }
}

function startAutoInfer() {
  const secs = parseInt(document.getElementById('interval-slider').value);
  autoInterval = setInterval(tryAutoInfer, secs * 1000);
  setStatus('Auto-inference every ' + secs + 's \u2014 watching for scene changes...');
  toast('\u23F1 Auto-inference started (' + secs + 's interval)');
}

function stopAutoInfer() {
  if (autoInterval) { clearInterval(autoInterval); autoInterval = null; }
}

async function tryAutoInfer() {
  if (!camRunning || inferBusy || stopRequested) return;

  const canvas  = captureFrame();
  const newHash = computeHash(canvas);

  if (lastHash !== null) {
    const dist    = hammingDist(lastHash, newHash);
    const changed = dist >= HASH_THRESHOLD;
    document.getElementById('change-indicator').className = changed ? 'changed' : 'same';
    document.getElementById('hash-display').textContent   = '\u0394' + dist;
    document.getElementById('change-status').textContent  =
      changed
        ? 'Scene changed (\u0394' + dist + ') \u2014 running inference...'
        : 'Scene unchanged (\u0394' + dist + ') \u2014 skipping';
    if (!changed) return;
  }

  lastHash = newHash;
  lastAnalyzedDataUrl = canvas.toDataURL('image/jpeg', 0.92);
  const blob = await canvasToBlob(canvas);
  await runAnalysisWithBlob(blob, 'auto @ ' + new Date().toLocaleTimeString());
}

// ── Manual snap (webcam) ──────────────────────────────────────────────────────
async function snapAndAnalyze() {
  if (!camRunning) { toast('\u26a0\ufe0f Camera not running'); return; }
  const canvas  = captureFrame();
  const newHash = computeHash(canvas);
  const dist    = lastHash ? hammingDist(lastHash, newHash) : 999;

  document.getElementById('change-indicator').className = dist >= HASH_THRESHOLD ? 'changed' : 'same';
  document.getElementById('hash-display').textContent   = lastHash ? '\u0394' + dist : '';
  document.getElementById('change-status').textContent  = 'Manual snapshot \u2014 running inference...';

  lastHash = newHash;
  lastAnalyzedDataUrl = canvas.toDataURL('image/jpeg', 0.92);
  const blob = await canvasToBlob(canvas);
  await runAnalysisWithBlob(blob, 'manual @ ' + new Date().toLocaleTimeString());
}

// ── Main dispatch (sidebar button) ───────────────────────────────────────────
async function runAnalysis() {
  if (currentSource === 'webcam') await snapAndAnalyze();
  else await runAnalysisWithFile();
}

// ── File-upload analysis ──────────────────────────────────────────────────────
async function runAnalysisWithFile() {
  const criteria = getCriteria();
  if (!currentFile)     { toast('\u26a0\ufe0f Upload a CCTV frame first'); return; }
  if (!criteria.length) { toast('\u26a0\ufe0f Add at least one criterion'); return; }
  lastAnalyzedDataUrl = document.getElementById('prev-img').src;
  setLoading(true, criteria.length);
  setStatus('Analyzing ' + criteria.length + ' criteria...');
  const t0 = Date.now();
  try {
    const fd = new FormData();
    fd.append('image',     currentFile);
    fd.append('criteria',  JSON.stringify(criteria));
    fd.append('sev_one',   document.getElementById('sev1').value);
    fd.append('sev_multi', document.getElementById('sev2').value);
    abortController = new AbortController();
    const res = await fetch('/analyze', {method:'POST', body:fd, signal:abortController.signal});
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const elapsed = ((Date.now()-t0)/1000).toFixed(1);
    renderResults(data, elapsed, 'uploaded file');
    setStatus('Done in ' + elapsed + 's');
  } catch(err) {
    toast('\u274c ' + err.message); setStatus('Error'); console.error(err);
  } finally { setLoading(false); }
}

// ── Webcam blob analysis ──────────────────────────────────────────────────────
async function runAnalysisWithBlob(blob, label) {
  const criteria = getCriteria();
  if (!criteria.length) { toast('\u26a0\ufe0f Add at least one criterion'); return; }
  if (inferBusy) return;
  if (stopRequested) { stopRequested = false; return; }
  inferBusy = true;
  stopRequested = false;
  setLoading(true, criteria.length);
  setStatus('Analyzing ' + criteria.length + ' criteria...');
  const t0 = Date.now();
  try {
    const fd = new FormData();
    fd.append('image',     blob, 'frame.jpg');
    fd.append('criteria',  JSON.stringify(criteria));
    fd.append('sev_one',   document.getElementById('sev1').value);
    fd.append('sev_multi', document.getElementById('sev2').value);
    const res = await fetch('/analyze', {method:'POST', body:fd});
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const elapsed = ((Date.now()-t0)/1000).toFixed(1);
    renderResults(data, elapsed, label);
    setStatus('Done in ' + elapsed + 's  \u00b7  ' + label);
  } catch(err) {
    if (err.name === 'AbortError') {
      setStatus('Analysis stopped.');
    } else {
      toast('\u274c ' + err.message); setStatus('Error'); console.error(err);
    }
  } finally { setLoading(false); inferBusy = false; abortController = null; }
}

// ── Render results — appends a new card to history, newest on top ─────────────
function renderResults(data, elapsed, label) {
  document.getElementById('empty').style.display = 'none';

  const ts = label || new Date().toLocaleTimeString();
  const thumbSrc = data.annotated_image || lastAnalyzedDataUrl || '';
  const thumb = thumbSrc
    ? `<img src="${thumbSrc}" onclick="openLightbox('${thumbSrc}')"
        style="height:64px;width:96px;object-fit:cover;border-radius:6px;border:1px solid var(--border);flex-shrink:0;cursor:zoom-in"
        title="Click to enlarge — bounding boxes shown"/>`
    : '';


  const rows = data.results.map((r,i) => {
    const src      = r.source || 'unknown';
    const rsrc     = r.route_source || '';
    const srcColor = src === 'YOLO' ? 'var(--green-t)' : src === 'Ollama' ? '#79c0ff' : 'var(--red-t)';
    const altRoute = src === 'YOLO' ? 'smolvlm' : 'yolo';
    const altLabel = src === 'YOLO' ? 'Switch to SmolVLM' : 'Switch to YOLO';
    const rawBlock = (src === 'Ollama' && r.raw)
      ? `<div id="raw-${i}-${Date.now()}" style="display:none;margin-top:8px;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;font-size:.72rem;font-family:var(--mono);color:var(--muted);white-space:pre-wrap;max-height:120px;overflow-y:auto">${esc(r.raw)}</div>`
      : '';
    const rawToggleId = `raw-${i}-${Date.now()}`;
    const feedbackBtns = `
      <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center">
        <span style="font-size:.65rem;color:${srcColor};font-family:var(--mono);background:${srcColor}22;padding:2px 7px;border-radius:10px;border:1px solid ${srcColor}44">${src}${rsrc.startsWith('memory') ? ' \u{1F9E0}' : ''}</span>
        <button onclick="saveRouteCorrection('${esc(r.question)}','${altRoute}',this)"
          style="font-size:.68rem;padding:2px 8px;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;cursor:pointer"
          title="Save correction for next time">\u{1F504} ${altLabel}</button>
        ${src === 'Ollama' ? `
        <button onclick="rateQuality('${esc(r.question)}',true,this)"
          style="font-size:.68rem;padding:2px 8px;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;cursor:pointer">\uD83D\uDC4D Good answer</button>
        <button onclick="showRaw(this.nextElementSibling)"
          style="font-size:.68rem;padding:2px 8px;background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;cursor:pointer">\uD83D\uDC4E Show raw</button>
        <div style="display:none;margin-top:6px;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;font-size:.72rem;font-family:var(--mono);color:var(--muted);white-space:pre-wrap;max-height:120px;overflow-y:auto;width:100%">${esc(r.raw||'')}<br><button onclick="rateQuality('${esc(r.question)}',false,this.parentElement.previousElementSibling)" style="margin-top:6px;font-size:.68rem;padding:2px 8px;background:none;border:1px solid var(--red);color:var(--red-t);border-radius:8px;cursor:pointer">Mark as bad answer \u2192 learn</button></div>
        ` : ''}
      </div>`;
    return `
    <div style="padding:10px 14px;border-bottom:1px solid var(--border)">
      <div style="display:grid;grid-template-columns:auto 1fr;gap:9px;align-items:start">
        <span style="font-size:.7rem;font-family:var(--mono);color:var(--accent);padding-top:2px;white-space:nowrap">Q${i+1}</span>
        <div>
          <div style="font-weight:600;font-size:.82rem;margin-bottom:3px">${esc(r.question)}</div>
          <div style="font-size:.8rem;color:var(--text);line-height:1.5">${esc(r.answer)}</div>
          ${feedbackBtns}
        </div>
      </div>
    </div>`;
  }).join('');


  const s = data.resource_stats;
  const stats = s
    ? `<div style="padding:8px 14px;font-size:.7rem;color:var(--muted);font-family:var(--mono);display:flex;gap:14px;flex-wrap:wrap">
        <span>&#x23f1; ${s.total_time_s}s total</span>
        <span>&#x26a1; ${s.cpu_pct}% CPU</span>
        <span>&#x1F4BE; ${s.process_ram_mb} MB RAM</span>
        <span>avg ${s.avg_time_per_criteria}s/criterion</span>
       </div>`
    : '';

  const card = document.createElement('div');
  card.style.cssText = 'background:var(--surf);border:1px solid var(--border);border-radius:12px;overflow:hidden';
  card.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--border);background:var(--surf2)">
      ${thumb}
      <div style="flex:1;min-width:0">
        <div style="font-size:.78rem;font-weight:700;color:var(--text)">${esc(ts)}</div>
        <div style="font-size:.68rem;color:var(--muted);margin-top:2px">${data.results.length} question(s) &middot; ${elapsed}s</div>
      </div>
    </div>
    ${rows}
    ${stats}`;

  const list = document.getElementById('history-list');
  list.insertBefore(card, list.firstChild);
  setStatus('Done in ' + elapsed + 's · ' + ts);
}

// -- Routing agent feedback helpers --
function showRaw(el) {
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function saveRouteCorrection(question, correctRoute, btn) {
  try {
    await fetch('/feedback/route', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question, correct_route: correctRoute})
    });
    btn.textContent = '\u2713 Saved for next time';
    btn.style.color = 'var(--green-t)';
    btn.style.borderColor = 'var(--green)';
    btn.disabled = true;
    toast('\uD83E\uDDE0 Learned: ' + question.slice(0,30) + '... \u2192 ' + correctRoute.toUpperCase());
  } catch(e) { toast('\u274c Could not save: ' + e.message); }
}

async function rateQuality(question, good, btn) {
  try {
    await fetch('/feedback/quality', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question, good})
    });
    btn.textContent = good ? '\u2713 Marked good' : '\u2713 Marked bad';
    btn.style.color = good ? 'var(--green-t)' : 'var(--red-t)';
    btn.disabled = true;
    toast('\uD83E\uDDE0 Quality feedback saved');
  } catch(e) { toast('\u274c ' + e.message); }
}


// ── Criteria helpers ──────────────────────────────────────────────────────────
function loadPreset() {
  document.getElementById('crit-list').innerHTML = '';
  PRESETS[document.getElementById('zone').value].forEach(q => addCrit(q));
}
function addCrit(text='') {
  const id = 'c'+Date.now()+Math.random().toString(36).slice(2,5);
  const el = document.createElement('div');
  el.className='crit-item'; el.id=id;
  el.innerHTML=`<span class="cnum">?</span>
    <textarea rows="2" placeholder="e.g. Is the worker wearing a hard hat?">${text}</textarea>
    <button class="del" onclick="removeCrit('${id}')">&#x2715;</button>`;
  document.getElementById('crit-list').appendChild(el);
  renumber();
}
function removeCrit(id){ const el=document.getElementById(id); if(el){el.remove();renumber();} }
function renumber(){ document.querySelectorAll('.crit-item .cnum').forEach((el,i)=>el.textContent=i+1); }
function getCriteria(){ return [...document.querySelectorAll('.crit-item textarea')].map(t=>t.value.trim()).filter(Boolean); }

// ── File upload helpers ───────────────────────────────────────────────────────
function onFileChange(e){ if(e.target.files[0]) loadFile(e.target.files[0]); }
function onDragOver(e){ e.preventDefault(); document.getElementById('drop-zone').classList.add('drag'); }
function onDragLeave(){ document.getElementById('drop-zone').classList.remove('drag'); }
function onDrop(e){
  e.preventDefault(); onDragLeave();
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')) loadFile(f);
}
function loadFile(file){
  currentFile=file;
  const reader=new FileReader();
  reader.onload=e=>{
    const img=document.getElementById('prev-img');
    img.src=e.target.result; img.style.display='block';
    document.getElementById('prev-wrap').style.display='block';
    document.getElementById('drop-zone').style.display='none';
    document.getElementById('img-meta').textContent=file.name+' \u00b7 '+(file.size/1024).toFixed(1)+' KB';
    document.getElementById('empty').style.display='none';
  };
  reader.readAsDataURL(file);
  toast('Image loaded \u2014 click Analyze Frame to begin');
}

// ── Stop inference ────────────────────────────────────────────────────────────
let stopRequested = false;

function stopInference() {
  stopRequested = true;
  stopAutoInfer();
  if (abortController) { abortController.abort(); abortController = null; }
  inferBusy = false;
  document.getElementById('auto-toggle').checked = false;
  setStatus('Stopped.');
  setLoading(false);
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on, count){
  document.getElementById('inline-spinner').classList.toggle('on', on);
  document.getElementById('stop-btn').classList.toggle('on', on);
  document.getElementById('go-btn').disabled = on;
}
function setStatus(msg){ document.getElementById('status-msg').textContent=msg; }

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.style.display = 'flex';
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function toast(msg){
  const el=document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),3000);
}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    return {
        "status":         "ready" if yolo_model is not None else "loading",
        "model_loaded":   yolo_model is not None,
        "checkpoint":     args.checkpoint,
        "smolvlm_ready":  smolvlm_ready,
        "smolvlm_model":  args.smolvlm_model,
        "yolo_classes":   list(yolo_model.names.values()) if yolo_model else [],
        "system_ram_used_mb":  round(mem.used  / 1024 / 1024, 1),
        "system_ram_total_mb": round(mem.total / 1024 / 1024, 1),
        "system_ram_pct":      round(mem.percent, 1),
        "cpu_pct":             round(psutil.cpu_percent(interval=None), 1),
    }


@app.post("/analyze")
async def analyze(
    image:     UploadFile = File(...),
    criteria:  str        = Form(...),
    sev_one:   str        = Form("WARNING"),
    sev_multi: str        = Form("CRITICAL"),
):
    if yolo_model is None:
        return JSONResponse({"error": "Model not ready yet"}, status_code=503)

    questions = json.loads(criteria)
    raw_bytes  = await image.read()
    pil_image  = Image.open(BytesIO(raw_bytes)).convert("RGB")
    # Resize to max 640px wide for fast YOLO inference
    # YOLO internally uses 640px anyway — sending larger just wastes time
    if pil_image.width > 640:
        ratio     = 640 / pil_image.width
        new_size  = (640, int(pil_image.height * ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    proc          = psutil.Process(os.getpid())
    ram_before    = proc.memory_info().rss / 1024 / 1024
    t_total       = time.time()

    # ── YOLO inference (always runs) ──────────────────────────────────────────
    # PPE detections (best.pt)
    yolo_results  = yolo_model.predict(pil_image, conf=0.25, iou=0.4, agnostic_nms=True, verbose=False)[0]
    # Person detections (yolov8n — better at crowds)
    person_results = yolo_person_model.predict(pil_image, conf=0.25, iou=0.35, agnostic_nms=True, verbose=False)[0]
    detections    = {}
    for box in yolo_results.boxes:
        cls_name = yolo_model.names[int(box.cls)].lower()
        if cls_name != "person":  # use dedicated person model for counting
            detections[cls_name] = detections.get(cls_name, 0) + 1
    # Count persons from dedicated person model
    for box in person_results.boxes:
        cls_id = int(box.cls)
        cls_name = yolo_person_model.names[cls_id].lower()
        if cls_name == "person":
            detections["person"] = detections.get("person", 0) + 1

    # -- Determine which classes to draw based on questions asked --
    CLASS_COLORS = {
        "hardhat":        (0, 200, 0),
        "no-hardhat":     (0, 0, 220),
        "safety vest":    (0, 180, 0),
        "no-safety vest": (0, 0, 220),
        "mask":           (180, 180, 0),
        "no-mask":        (0, 0, 220),
        "person":         (200, 130, 0),
        "machinery":      (180, 0, 180),
        "vehicle":        (180, 0, 180),
        "safety cone":    (0, 180, 180),
    }
    # Map question keywords to YOLO class names
    QUESTION_CLASS_MAP = [
        (["helmet", "hardhat", "hard hat"],         ["hardhat", "no-hardhat"]),
        (["vest", "safety vest", "high-vis"],        ["safety vest", "no-safety vest"]),
        (["mask"],                                   ["mask", "no-mask"]),
        (["person", "people", "worker", "workers",
          "how many", "count", "anyone", "someone"], ["person"]),
        (["machine", "machinery", "equipment"],      ["machinery"]),
        (["vehicle", "forklift", "truck"],           ["vehicle"]),
        (["cone", "safety cone"],                    ["safety cone"]),
        (["ppe", "protective"],                      ["hardhat", "no-hardhat",
                                                      "safety vest", "no-safety vest",
                                                      "mask", "no-mask"]),
    ]
    relevant_classes = set()
    all_questions_lower = " ".join(questions).lower()
    for keywords, classes in QUESTION_CLASS_MAP:
        if any(kw in all_questions_lower for kw in keywords):
            relevant_classes.update(classes)
    # If no specific class matched, draw all detected classes
    if not relevant_classes:
        relevant_classes = set(CLASS_COLORS.keys())

    # -- Draw bounding boxes only for relevant classes --
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def draw_box(img, box, cls_name, color):
        conf_val = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = cls_name + " " + str(round(conf_val, 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw PPE boxes from best.pt
    for box in yolo_results.boxes:
        cls_name = yolo_model.names[int(box.cls)].lower()
        if cls_name not in relevant_classes or cls_name == "person":
            continue
        draw_box(img_cv, box, cls_name, CLASS_COLORS.get(cls_name, (180, 180, 180)))

    # Draw person boxes from dedicated person model
    if "person" in relevant_classes:
        for box in person_results.boxes:
            cls_name = yolo_person_model.names[int(box.cls)].lower()
            if cls_name == "person":
                draw_box(img_cv, box, cls_name, CLASS_COLORS["person"])
    annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=85)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # -- Route each question (memory-first, then keyword router) --
    routing_memory  = load_memory()
    yolo_questions  = []
    vlm_questions   = []

    for i, q in enumerate(questions):
        # 1. Check routing memory first (learned corrections take priority)
        learned_route, sim, pat_id = find_learned_route(q, routing_memory)
        if learned_route == "yolo":
            yolo_questions.append((i, q, f"memory:{pat_id}(sim={sim})"))
        elif learned_route == "smolvlm":
            vlm_questions.append((i, q, f"memory:{pat_id}(sim={sim})"))
        # 2. Fall back to keyword router
        elif is_yolo_question(q):
            yolo_questions.append((i, q, "keyword-router"))
        else:
            vlm_questions.append((i, q, "keyword-router"))
    print(f"[ROUTE] YOLO: {[q for _,q,_ in yolo_questions]}")
    print(f"[ROUTE] VLM:  {[q for _,q,_ in vlm_questions]}")

    results   = [None] * len(questions)
    raw_parts = ["[YOLO detections] " + str(detections)]

    # -- Answer YOLO questions --
    for i, q, rsrc in yolo_questions:
        answer = answer_with_yolo(q, detections)
        results[i] = {"question": q, "answer": answer, "source": "YOLO", "route_source": rsrc}
        raw_parts.append("[YOLO] Q" + str(i+1) + ": " + q + "\n-> " + answer)

    # -- Answer SmolVLM2 questions --
    if vlm_questions:
        if smolvlm_ready:
            try:
                vlm_qs = [q for _, q, _ in vlm_questions]
                answer_map, raw_vlm = ask_smolvlm(pil_image, vlm_qs)

                raw_parts.append("[SmolVLM2 raw]\n" + raw_vlm)
                for local_idx, (orig_idx, q, rsrc) in enumerate(vlm_questions):
                    answer = answer_map.get(local_idx, "(no answer)")
                    results[orig_idx] = {
                        "question":     q,
                        "answer":       answer,
                        "source":       "SmolVLM2",
                        "route_source": rsrc,
                        "raw":          raw_vlm,
                    }
            except Exception as e:
                for orig_idx, q, rsrc in vlm_questions:
                    results[orig_idx] = {"question": q, "answer": "SmolVLM2 error: " + str(e), "source": "error", "route_source": rsrc}
        else:
            for orig_idx, q, rsrc in vlm_questions:
                results[orig_idx] = {
                    "question":     q,
                    "answer":       "SmolVLM2 not ready — check model path",
                    "source":       "error",
                    "route_source": rsrc,
                }

    total_time    = round(time.time() - t_total, 2)
    ram_after     = proc.memory_info().rss / 1024 / 1024
    mem           = psutil.virtual_memory()

    return JSONResponse({
        "results":        results,
        "raw_output":     "\n\n".join(raw_parts),
        "detections":     detections,
        "annotated_image": annotated_b64,
        "resource_stats": {
            "total_time_s":          total_time,
            "criteria_count":        len(questions),
            "yolo_count":            len(yolo_questions),
            "vlm_count":          len(vlm_questions),
            "avg_time_per_criteria": round(total_time / max(len(questions), 1), 2),
            "cpu_pct":               round(psutil.cpu_percent(interval=None), 1),
            "process_ram_mb":        round(ram_after, 1),
            "ram_delta_mb":          round(ram_after - ram_before, 1),
            "system_ram_used_mb":    round(mem.used  / 1024 / 1024, 1),
            "system_ram_total_mb":   round(mem.total / 1024 / 1024, 1),
            "system_ram_pct":        round(mem.percent, 1),
        },
    })


@app.post("/feedback/route")
async def feedback_route(payload: dict):
    """
    User corrects the route for a question.
    payload: {question, correct_route}  -- correct_route: "yolo" | "ollama"
    """
    question = payload.get("question", "").strip()
    correct  = payload.get("correct_route", "").strip()
    if not question or correct not in ("yolo", "smolvlm"):
        return JSONResponse({"error": "invalid payload"}, status_code=400)
    mem    = load_memory()
    pat_id = upsert_pattern(mem, question, correct)
    print("[Memory] Route correction saved:", question, "->", correct)
    return {"status": "saved", "pattern_id": pat_id}


@app.post("/feedback/quality")
async def feedback_quality(payload: dict):
    """
    User rates Ollama answer quality.
    payload: {question, good}  -- good: true | false
    """
    question = payload.get("question", "").strip()
    good     = payload.get("good", None)
    if not question or good is None:
        return JSONResponse({"error": "invalid payload"}, status_code=400)
    mem    = load_memory()
    pat_id = upsert_pattern(mem, question, "smolvlm", quality_good=bool(good))
    print("[Memory] Quality feedback saved:", question, "good=" + str(good))
    return {"status": "saved", "pattern_id": pat_id}


@app.get("/memory")
async def get_memory():
    """Return the current routing memory for inspection."""
    mem = load_memory()
    return {
        "total_patterns": len(mem["patterns"]),
        "patterns": mem["patterns"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)