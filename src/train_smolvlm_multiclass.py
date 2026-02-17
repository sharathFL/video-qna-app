#!/usr/bin/env python3
"""
Train SmolVLM2 (small VLM) for multiclass behavior labeling (8 classes).
Uses data/frames_multiclass and data/train_multiclass.jsonl.
All outputs (models, logs, plots) go to outputs_multiclass/ so binary outputs/ stays separate.
Requires: transformers>=4.49.0 (for SmolVLM2 support).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None

# Multiclass paths (separate from binary safe/unsafe)
FRAMES_ROOT = Path("/workspace/data/frames_multiclass")
TRAIN_JSONL = Path("/workspace/data/train_multiclass.jsonl")
OUTPUT_DIR = Path("/workspace/outputs_multiclass")
PLOTS_DIR = Path("/workspace/outputs_multiclass/plots")

# SmolVLM2 variants (smaller = faster, less VRAM)
SMOLVLM2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SMOLVLM2_500M = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM2_256M = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

PROMPT = ""  # Multiclass: prompt comes from dataset (build_jsonl_multiclass); fallback only
EPOCHS = 2
BATCH_SIZE = 2
LR = 2e-4
GRAD_ACCUM = 4
RESPONSE_LABEL_TOKENS = 24  # Longer for class names like 3_carrying_overload_with_forklift
LIVE_DASHBOARD_PORT = 8084  # Multiclass training dashboard
PLOT_UPDATE_EVERY_N_STEPS = 20  # update sample predictions every N steps (loss still every log step)


def _run_http_server(port, directory, log_file_path=None):
    """Serve directory on port; GET /log.txt returns tail of log_file_path."""
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            log_path = getattr(self.server, "log_file_path", None)
            path_only = self.path.split("?")[0]
            if path_only == "/log.txt" and log_path:
                log_path = Path(log_path)
                try:
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        tail = "".join(lines[-250:])
                    else:
                        tail = "Waiting for training log...\n"
                except Exception as e:
                    tail = f"Log read error: {e}\n"
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(tail.encode("utf-8"))
                return
            return super().do_GET()

        def log_message(self, format, *args):
            pass

    class ReuseServer(socketserver.TCPServer):
        allow_reuse_address = True
    with ReuseServer(("0.0.0.0", port), Handler) as httpd:
        httpd.log_file_path = log_file_path
        httpd.serve_forever()


def _start_live_server(port, directory, log_file_path=None):
    import threading
    t = threading.Thread(target=_run_http_server, args=(port, directory, log_file_path), daemon=True)
    t.start()
    return t


def _stats_collector_loop(plots_dir: Path, training_state_path: Path, interval_sec: float = 5.0):
    """Background thread: read training_state.json, gather system stats, write stats.json. Separate from training."""
    while True:
        try:
            time.sleep(interval_sec)
            state = {}
            if training_state_path.exists():
                try:
                    state = json.loads(training_state_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            sys_stats = _gather_system_stats()
            stats = {
                "step": state.get("step", 0),
                "training_elapsed_sec": state.get("training_elapsed_sec", 0),
                "training_elapsed_hms": state.get("training_elapsed_hms", "0h 0m 0s"),
                "eval_sec": state.get("eval_sec"),
                "test_sec": state.get("test_sec"),
                "gpu_util_pct": sys_stats.get("gpu_util_pct"),
                "gpu_mem_used_mb": sys_stats.get("gpu_mem_used_mb"),
                "gpu_mem_total_mb": sys_stats.get("gpu_mem_total_mb"),
                "gpu_temp_c": sys_stats.get("gpu_temp_c"),
                "cpu_pct": sys_stats.get("cpu_pct"),
                "ram_used_mb": sys_stats.get("ram_used_mb"),
                "ram_total_mb": sys_stats.get("ram_total_mb"),
                "disk_free_gb": sys_stats.get("disk_free_gb"),
                "disk_total_gb": sys_stats.get("disk_total_gb"),
                "sample_accuracy_pct": state.get("sample_accuracy_pct"),
            }
            (plots_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        except Exception:
            pass


def _start_stats_collector(plots_dir: Path, training_state_path: Path, interval_sec: float = 5.0):
    """Start daemon thread that collects GPU/CPU/disk stats separately from training."""
    import threading
    t = threading.Thread(
        target=_stats_collector_loop,
        args=(plots_dir, training_state_path),
        kwargs={"interval_sec": interval_sec},
        daemon=True,
    )
    t.start()
    return t


def _gather_system_stats():
    """Return dict with gpu_util, gpu_mem, gpu_temp, cpu_pct, ram_used_mb, ram_total_mb, disk_free_gb, disk_total_gb."""
    out = {
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_temp_c": None,
        "cpu_pct": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "disk_free_gb": None,
        "disk_total_gb": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(",")
            if len(parts) >= 4:
                out["gpu_util_pct"] = int(parts[0].strip() or 0)
                out["gpu_mem_used_mb"] = int(parts[1].strip().split()[0] or 0)
                out["gpu_mem_total_mb"] = int(parts[2].strip().split()[0] or 0)
                out["gpu_temp_c"] = int(parts[3].strip() or 0)
    except Exception:
        pass
    try:
        import psutil
        out["cpu_pct"] = round(psutil.cpu_percent(interval=0.1), 1)
        v = psutil.virtual_memory()
        out["ram_used_mb"] = int(v.used / (1024 ** 2))
        out["ram_total_mb"] = int(v.total / (1024 ** 2))
        d = psutil.disk_usage("/")
        out["disk_free_gb"] = round(d.free / (1024 ** 3), 2)
        out["disk_total_gb"] = round(d.total / (1024 ** 3), 2)
    except Exception:
        try:
            result = subprocess.run(["df", "-BG", "/"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and "\n" in result.stdout:
                parts = result.stdout.strip().split("\n")[1].split()
                if len(parts) >= 4:
                    out["disk_total_gb"] = int(parts[1].replace("G", "") or 0)
                    out["disk_free_gb"] = int(parts[3].replace("G", "") or 0)
        except Exception:
            pass
    return out


# Dashboard HTML with Plotly.js: loss and accuracy side by side; stats loaded separately; checkpoint alert.
_INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SmolVLM2 — Multiclass (8 classes)</title>
<meta http-equiv="refresh" content="30"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }} .section {{ margin: 24px 0; }}
img {{ border-radius: 8px; border: 2px solid #333; }}
table {{ border-collapse: collapse; }} th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #444; }}
th {{ background: #16213e; }} a {{ color: #e94560; }}
#stats {{ font-family: monospace; background: #0f0f1a; padding: 12px; border-radius: 8px; white-space: pre-wrap; }}
.plots-row {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }}
.plot-container {{ flex: 1; min-width: 400px; max-width: 50%; }}
.plot-container .plot {{ width: 100%; height: 360px; }}
#checkpoint-alert {{ display: none; padding: 16px 24px; border-radius: 8px; margin-bottom: 16px; font-weight: bold; font-size: 1.1em; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
#checkpoint-alert.recent {{ display: block; position: sticky; top: 0; z-index: 999; background: #2e7d32; color: #fff; border: 2px solid #4caf50; animation: pulse 1.5s ease-in-out 5; font-size: 1.25em; }}
#checkpoint-alert.old {{ display: block; background: #37474f; color: #cfd8dc; border: 1px solid #546e7a; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.9; transform: scale(1.01); }} }}
#url-hint {{ background: #2e7d32; color: #fff; padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; }}
</style></head><body>
<p id="url-hint"><strong>Multiclass dashboard:</strong> Open at <a href="http://localhost:8084/" style="color:#fff; text-decoration:underline">http://localhost:8084/</a> in your browser (do not open the HTML file directly).</p>
<h1>SmolVLM2 — Multiclass (8 classes) — Live Training</h1>
<p>Step: <b>{step}</b> {progress} | {message} | <em>Auto-refresh 30s</em></p>
<div id="checkpoint-alert"></div>
<div class="section"><h2>System &amp; training stats</h2><div id="stats">Loading...</div></div>
<div class="section"><h2>Plots</h2>
<div class="plots-row">
  <div class="plot-container"><h3>Training loss</h3><div id="loss-plot" class="plot"></div></div>
  <div class="plot-container"><h3>Training accuracy (sample)</h3><div id="accuracy-plot" class="plot"></div></div>
</div></div>
<div class="section"><h2>Sample inputs &amp; predictions (5 random frames each update)</h2>
<p style="color:#888; font-size:0.9em;">Refresh every <b>{update_every_n}</b> steps (use <code>--plot-every N</code> to change; page auto-refreshes every 30s)</p>
<table><tr><th>Input frame</th><th>Ground truth</th><th>Prediction</th><th>Step</th><th>Idx</th></tr>
{rows}
</table></div>
<div class="section"><h2>Input grid</h2>{input_grid_img}</div>
<div class="section"><h2>Log tail</h2><a href="/log.txt">/log.txt</a></div>
<script>
(function() {{
  if (window.location.protocol === 'file:') {{
    document.body.insertAdjacentHTML('afterbegin', '<div style="background:#c62828;color:#fff;padding:16px;margin-bottom:16px;border-radius:8px;"><strong>Wrong URL.</strong> You opened this as a file. Open <a href="http://localhost:8084/" style="color:#fff; text-decoration:underline">http://localhost:8084/</a> in your browser instead (training server must be running).</div>');
    var h = document.getElementById('url-hint');
    if (h) h.style.display = 'none';
  }} else {{
    var h = document.getElementById('url-hint');
    if (h) h.style.display = 'none';
  }}
}})();
var lossLayout = {{ title: 'Loss', paper_bgcolor: '#0f0f1a', plot_bgcolor: '#0f0f1a', font: {{ color: '#eee' }}, xaxis: {{ gridcolor: '#333' }}, yaxis: {{ gridcolor: '#333' }} }};
var accLayout = {{ title: 'Accuracy (%)', paper_bgcolor: '#0f0f1a', plot_bgcolor: '#0f0f1a', font: {{ color: '#eee' }}, xaxis: {{ gridcolor: '#333' }}, yaxis: {{ gridcolor: '#333', range: [0, 105] }}, margin: {{ t: 40 }} }};
function drawPlots() {{
  if (typeof Plotly === 'undefined') {{
    document.getElementById('loss-plot').innerHTML = '<p style="color:#888">Plotly loading or blocked. Enable script from cdn.plot.ly or check network.</p>';
    document.getElementById('accuracy-plot').innerHTML = '<p style="color:#888">Plotly loading or blocked.</p>';
    return;
  }}
  fetch('/plot_data.json?t='+Date.now()).then(r=>r.json()).then(d=>{{
    if (d.loss && d.loss.steps && d.loss.steps.length) {{
      Plotly.newPlot('loss-plot', [{{ x: d.loss.steps, y: d.loss.values, type: 'scatter', mode: 'lines+markers', marker: {{ size: 4 }}, line: {{ color: '#3498db' }} }}], lossLayout, {{ responsive: true }});
    }}
    if (d.accuracy && d.accuracy.steps && d.accuracy.steps.length) {{
      Plotly.newPlot('accuracy-plot', [{{ x: d.accuracy.steps, y: d.accuracy.values, type: 'scatter', mode: 'lines+markers', marker: {{ size: 4 }}, line: {{ color: '#2ecc71' }} }}], accLayout, {{ responsive: true }});
    }}
  }}).catch(function() {{}});
}}
function loadStats() {{
  fetch('/stats.json?t='+Date.now()).then(r=>r.json()).then(s=>{{
    var t = 'Time (training): ' + (s.training_elapsed_hms || s.training_elapsed_sec + 's') + '\\n';
    t += 'Eval: ' + (s.eval_sec != null ? s.eval_sec + 's' : 'N/A') + ' | Test: ' + (s.test_sec != null ? s.test_sec + 's' : 'N/A') + '\\n';
    t += 'GPU: ' + (s.gpu_util_pct != null ? s.gpu_util_pct + '%' : 'N/A') + ' | Mem ' + (s.gpu_mem_used_mb != null ? s.gpu_mem_used_mb + '/' + s.gpu_mem_total_mb + ' MB' : 'N/A') + ' | Temp ' + (s.gpu_temp_c != null ? s.gpu_temp_c + '°C' : 'N/A') + '\\n';
    t += 'CPU: ' + (s.cpu_pct != null ? s.cpu_pct + '%' : 'N/A') + ' | RAM: ' + (s.ram_used_mb != null ? s.ram_used_mb + ' / ' + s.ram_total_mb + ' MB' : 'N/A') + '\\n';
    t += 'Disk: ' + (s.disk_free_gb != null ? s.disk_free_gb + ' / ' + s.disk_total_gb + ' GB free' : 'N/A') + '\\n';
    t += 'Sample accuracy: ' + (s.sample_accuracy_pct != null ? s.sample_accuracy_pct + '%' : 'N/A');
    document.getElementById('stats').textContent = t;
  }}).catch(function() {{ document.getElementById('stats').textContent = 'Stats unavailable'; }});
}}
function loadCheckpointAlert() {{
  fetch('/checkpoint_alert.json?t='+Date.now()).then(r=>r.json()).then(c=>{{
    var el = document.getElementById('checkpoint-alert');
    if (!c || !c.step) {{ el.style.display = 'none'; return; }}
    var msg = 'CHECKPOINT SAVED — ' + (c.path || ('checkpoint-' + c.step)) + ' (step ' + c.step + ')';
    if (c.at) msg += ' at ' + c.at;
    el.textContent = msg;
    el.className = (Date.now()/1000 - c.at_sec) < 120 ? 'recent' : 'old';
  }}).catch(function() {{ document.getElementById('checkpoint-alert').style.display = 'none'; }});
}}
drawPlots(); loadStats(); loadCheckpointAlert();
setInterval(drawPlots, 5000); setInterval(loadStats, 5000); setInterval(loadCheckpointAlert, 3000);
</script>
</body></html>"""


PLOT_HISTORY_FILENAME = "plot_history.json"


def _load_loss_history_from_checkpoints(output_dir):
    """Load (step, loss) from all checkpoint trainer_state.json under output_dir. Returns (log_steps, train_losses)."""
    output_dir = Path(output_dir)
    log_steps, train_losses = [], []
    for p in sorted(output_dir.glob("checkpoint-*/trainer_state.json")):
        try:
            with open(p, "r") as f:
                state = json.load(f)
            for entry in state.get("log_history") or []:
                step = entry.get("step")
                loss = entry.get("loss")
                if step is not None and loss is not None:
                    log_steps.append(int(step))
                    train_losses.append(float(loss))
        except Exception:
            continue
    # sort by step and dedupe (keep last per step)
    if not log_steps:
        return [], []
    order = sorted(range(len(log_steps)), key=lambda i: (log_steps[i], i))
    log_steps = [log_steps[i] for i in order]
    train_losses = [train_losses[i] for i in order]
    return log_steps, train_losses


def _load_plot_history(plots_dir: Path):
    """Load persisted plot history (loss + accuracy) so plots continue across restarts/checkpoints. Returns (log_steps, train_losses, sample_accuracy_steps, sample_accuracies)."""
    path = Path(plots_dir) / PLOT_HISTORY_FILENAME
    if not path.exists():
        return [], [], [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        loss = data.get("loss") or {}
        acc = data.get("accuracy") or {}
        return (
            list(loss.get("steps") or []),
            list(loss.get("values") or []),
            list(acc.get("steps") or []),
            list(acc.get("values") or []),  # stored as percentages (0-100)
        )
    except Exception:
        return [], [], [], []


def _merge_loss_history(plot_steps, plot_values, ckpt_steps, ckpt_values):
    """Merge loss from plot_history and checkpoints; sort by step, dedupe (checkpoint wins for same step)."""
    by_step = {}
    for s, v in zip(plot_steps, plot_values):
        by_step[int(s)] = float(v)
    for s, v in zip(ckpt_steps, ckpt_values):
        by_step[int(s)] = float(v)
    if not by_step:
        return [], []
    steps = sorted(by_step.keys())
    return steps, [by_step[s] for s in steps]


def _normalize_multiclass_pred(pred, valid_responses):
    """Map raw model output to one of valid_responses (exact match or first substring match)."""
    if not pred or not valid_responses:
        return (pred[:32] if pred else "—")
    pred_clean = pred.strip()
    if pred_clean in valid_responses:
        return pred_clean
    for c in sorted(valid_responses, key=len, reverse=True):
        if c in pred:
            return c
    return pred_clean[:32] if pred_clean else "—"


class LiveDashboardCallback(TrainerCallback):
    """Loss plot + sample inputs and predictions every N steps. Served in browser."""

    def __init__(self, output_dir, processor, dataset, update_every_n=PLOT_UPDATE_EVERY_N_STEPS, sample_indices=None, pred_every_n_frame=None, training_output_dir=None, model=None, resuming=False, prompt_override=None, valid_responses=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir = Path(training_output_dir) if training_output_dir else output_dir
        self.processor = processor
        self.dataset = dataset
        self.update_every_n = update_every_n
        self.model_ref = model  # so we can run predictions on first on_log (Trainer may not pass model to on_step_end)
        self.step_offset = 0  # when resuming via adapter-only fallback, set to checkpoint step so plots show 950, 955, ...
        self.resuming = resuming  # only load/merge plot history when resuming; otherwise start fresh so loss/accuracy don't jump
        self.resume_step = None  # set from main() when resuming so dashboard shows checkpoint step even before state is loaded
        self._prompt_override = prompt_override  # for multiclass: use dataset prompt instead of global PROMPT
        self._valid_responses = frozenset(valid_responses) if valid_responses else None  # for multiclass: normalize pred to one of these
        if sample_indices is not None:
            self.sample_indices = sample_indices
        elif pred_every_n_frame and pred_every_n_frame > 0:
            n = len(dataset)
            # 5 indices spread across the dataset so we get visibly different frames (not all from same stretch)
            if n >= 5:
                self.sample_indices = [int(i * (n - 1) / 4) for i in range(5)]  # 0%, 25%, 50%, 75%, 100%
            else:
                self.sample_indices = list(range(n))
        else:
            self.sample_indices = [0, min(500, len(dataset) - 1), min(1000, len(dataset) - 1), min(2000, len(dataset) - 1), min(3000, len(dataset) - 1)]
        self.train_losses = []
        self.log_steps = []
        self.sample_accuracy_steps = []
        self.sample_accuracies = []
        self.last_step_updated = -1
        self.sample_results = []
        self.training_start_time = None
        self.eval_sec = None
        self.test_sec = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_start_time = time.time()
        raw_step = state.global_step if state else 0
        displayed_step = raw_step + self.step_offset
        # When resuming, HF may call this before checkpoint is loaded (raw_step=0); use resume_step so dashboard shows 990 not 0
        if displayed_step == 0 and getattr(self, "resume_step", None) is not None:
            displayed_step = self.resume_step
        self._write_training_state(displayed_step)
        # Only restore plot history when resuming; otherwise start fresh so loss/accuracy continue from 0, not from old run
        if self.resuming:
            ph_loss_s, ph_loss_v, ph_acc_s, ph_acc_v = _load_plot_history(self.output_dir)
            ckpt_steps, ckpt_losses = _load_loss_history_from_checkpoints(self.training_output_dir)
            self.log_steps, self.train_losses = _merge_loss_history(ph_loss_s, ph_loss_v, ckpt_steps, ckpt_losses)
            self.sample_accuracy_steps = list(ph_acc_s)
            self.sample_accuracies = [v / 100.0 for v in ph_acc_v]  # stored as 0-100 in plot_history
            # Keep full history so dashboard shows all losses; new steps (displayed_step) will append and dedupe in _write_plot_data
        else:
            self.log_steps, self.train_losses = [], []
            self.sample_accuracy_steps, self.sample_accuracies = [], []
        # If we have many loss steps but few accuracy points (e.g. history was lost), backfill accuracy so the graph shows full range
        if self.step_offset > 0 and len(self.log_steps) >= 10 and len(self.sample_accuracy_steps) < 10:
            fill_steps = list(range(0, self.step_offset + 1, self.update_every_n))
            mean_acc = (sum(self.sample_accuracies) / len(self.sample_accuracies)) if self.sample_accuracies else 0.5
            existing_steps = set(self.sample_accuracy_steps)
            for s in fill_steps:
                if s not in existing_steps:
                    self.sample_accuracy_steps.append(s)
                    self.sample_accuracies.append(mean_acc)
            order = sorted(range(len(self.sample_accuracy_steps)), key=lambda i: self.sample_accuracy_steps[i])
            self.sample_accuracy_steps = [self.sample_accuracy_steps[i] for i in order]
            self.sample_accuracies = [self.sample_accuracies[i] for i in order]
            print(f"  Backfilled accuracy at {len(fill_steps)} steps (mean={mean_acc*100:.0f}%) so graph shows full history.")
        if self.log_steps or self.sample_accuracy_steps:
            if self.step_offset > 0:
                # Keep last_step_updated in trainer-step space so we run predictions at 50, 100, ...
                self.last_step_updated = 0
            else:
                self.last_step_updated = max(
                    list(self.log_steps) + list(self.sample_accuracy_steps) + [-1]
                )
            print(f"  Restored plot history: {len(self.train_losses)} loss points, {len(self.sample_accuracies)} accuracy points.")
        self._write_plot_data()
        # Always write index with current displayed_step so header shows 990 (not 0) when resuming
        self._write_index(step=displayed_step, message="Training started." + (" Resumed." if (self.log_steps or self.sample_accuracy_steps) else " Refresh in a few steps."))
        self._write_training_state(displayed_step)
        print(f"  Dashboard: loss every log step, predictions every {self.update_every_n} steps (5 random samples each update)")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.model_ref is None and model is not None:
            self.model_ref = model

    def _write_training_state(self, step):
        """Write training_state.json (step, elapsed, sample_accuracy). Stats collector thread merges with system stats."""
        try:
            elapsed = time.time() - self.training_start_time if self.training_start_time else 0
            state = {
                "step": step,
                "training_elapsed_sec": round(elapsed, 1),
                "training_elapsed_hms": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
                "eval_sec": self.eval_sec,
                "test_sec": self.test_sec,
                "sample_accuracy_pct": round(self.sample_accuracies[-1] * 100, 1) if self.sample_accuracies else None,
            }
            (self.output_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Training state write skipped: {e}")

    def _write_plot_data(self):
        """Write plot_data.json for Plotly.js and plot_history.json for persistence across restarts/checkpoints.
        When writing plot_history, merge with existing file so we never overwrite with fewer points (preserve full history)."""
        try:
            data = {
                "loss": {"steps": self.log_steps, "values": [float(x) for x in self.train_losses]},
                "accuracy": {"steps": self.sample_accuracy_steps, "values": [x * 100 for x in self.sample_accuracies]},
            }
            (self.output_dir / "plot_data.json").write_text(json.dumps(data), encoding="utf-8")
            # Merge with existing plot_history so we never lose accuracy/loss points from previous runs
            hist_path = self.output_dir / PLOT_HISTORY_FILENAME
            if hist_path.exists():
                try:
                    existing = json.loads(hist_path.read_text(encoding="utf-8"))
                    for key in ("loss", "accuracy"):
                        ex = (existing.get(key) or {})
                        ex_steps = list(ex.get("steps") or [])
                        ex_vals = list(ex.get("values") or [])
                        cur_steps = data[key]["steps"]
                        cur_vals = data[key]["values"]
                        by_step = {}
                        for s, v in zip(ex_steps, ex_vals):
                            by_step[int(s)] = float(v)
                        for s, v in zip(cur_steps, cur_vals):
                            by_step[int(s)] = float(v)
                        if by_step:
                            steps = sorted(by_step.keys())
                            data[key] = {"steps": steps, "values": [by_step[s] for s in steps]}
                except Exception:
                    pass
            hist_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            print(f"Plot data write skipped: {e}")

    def _write_index(self, step, message="", sample_results=None, progress=""):
        try:
            cache_bust = step
            sample_results = sample_results or self.sample_results
            rows = "".join(
                f'<tr><td><img src="{r.get("image_path", "")}?v={cache_bust}" width="160" alt=""/></td><td>{r.get("ground_truth", "")}</td><td>{r.get("prediction", "")}</td><td>Step {r.get("step", 0)}</td><td>Idx {r.get("idx", "")}</td></tr>'
                for r in sample_results
            )
            input_grid_img = '<img src="input_grid.png?v=%d" width="800" alt="Input samples" onerror="this.alt=\'Waiting for first update...\'"/>' % cache_bust
            progress_str = f" ({progress})" if progress else ""
            html = _INDEX_HTML_TEMPLATE.format(step=step, message=message, rows=rows if rows else "<tr><td colspan='5'>Run a few steps to see samples.</td></tr>", input_grid_img=input_grid_img, update_every_n=self.update_every_n, progress=progress_str)
            (self.output_dir / "index.html").write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"Dashboard write skipped: {e}")

    def _save_input_grid_and_run_inference(self, step):
        if self.model_ref is None and getattr(self, "trainer", None) is not None:
            self.model_ref = getattr(self.trainer, "model", None)
        model = self.model_ref
        if model is None:
            if not getattr(self, "trainer", None):
                print("  Dashboard: skipping sample predictions (no model or trainer ref).")
            return
        if not HAS_MATPLOTLIB:
            return
        # Multiclass: 5 random indices each update for class variety
        dataset_len = len(self.dataset)
        if dataset_len == 0:
            return
        if getattr(self, "_valid_responses", None):
            indices_for_step = random.sample(range(dataset_len), min(5, dataset_len))
        else:
            prediction_round = step // self.update_every_n
            step_size = max(1, dataset_len // 20)
            offset = (prediction_round * step_size) % dataset_len
            indices_for_step = [(idx + offset) % dataset_len for idx in self.sample_indices]
        n = len(indices_for_step)
        fig, axes = plt.subplots(1, min(n, 8), figsize=(4 * min(n, 8), 4))
        if n == 1:
            axes = [axes]
        elif n > 8:
            axes = list(axes)
        results = []
        for i, idx in enumerate(indices_for_step):
            if idx >= len(self.dataset):
                continue
            sample = self.dataset.samples[idx]
            image_path = self.dataset.frames_root / sample["image"]
            if not image_path.exists():
                continue
            img = Image.open(image_path).convert("RGB")
            img_copy = img.copy()  # save a distinct copy so each sample_*.png is correct for this idx
            if i < len(axes):
                axes[i].imshow(img)
                axes[i].set_title(f"Idx {idx}")
                axes[i].axis("off")
            gt = sample.get("response", "?")
            pred = "—"
            prompt_text = self._prompt_override if getattr(self, "_prompt_override", None) else PROMPT
            try:
                with torch.no_grad():
                    model.eval()
                    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
                    inp = self.processor.apply_chat_template(
                        [messages],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    dev = next(model.parameters()).device
                    def to_dev(v):
                        if not hasattr(v, "to"):
                            return v
                        if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                            return v.to(dev, dtype=getattr(model, "dtype", v.dtype))
                        return v.to(dev)
                    inp = {k: to_dev(v) for k, v in inp.items()}
                    max_tokens = 40 if getattr(self, "_valid_responses", None) else 10
                    out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
                    gen = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    raw = (gen[0].strip() if gen else "")
                    if getattr(self, "_valid_responses", None):
                        pred = _normalize_multiclass_pred(raw, self._valid_responses)
                    else:
                        pred = raw.upper()
                        if "UNSAFE" in pred:
                            pred = "UNSAFE"
                        elif "SAFE" in pred:
                            pred = "SAFE"
                        else:
                            pred = (pred[:24] if pred else "—")
                    model.train()
            except Exception as e:
                pred = f"err: {e!r}"[:24]
            results.append({"step": step, "ground_truth": gt, "prediction": pred, "image_path": f"sample_{i}.png", "idx": idx})
            img_copy.save(self.output_dir / f"sample_{i}.png")
        if n > 8:
            for j in range(len(axes), 8):
                axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "input_grid.png", dpi=120)
        plt.close()
        self.sample_results = results
        # Sample accuracy on these fixed frames
        if results:
            if getattr(self, "_valid_responses", None):
                correct = sum(
                    1
                    for r in results
                    if _normalize_multiclass_pred(r.get("prediction") or "", self._valid_responses)
                    == (r.get("ground_truth") or "").strip()
                )
            else:
                correct = sum(1 for r in results if (r.get("prediction") or "").strip().upper() == (r.get("ground_truth") or "").strip().upper())
            self.sample_accuracy_steps.append(step)
            self.sample_accuracies.append(correct / len(results))
        self._write_plot_data()

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        displayed_step = step + self.step_offset
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None:
                self.train_losses.append(float(loss))
                self.log_steps.append(displayed_step)
        self._write_plot_data()
        # Ensure we have model ref (e.g. after resume fallback trainer may hold the live model)
        if self.model_ref is None and getattr(self, "trainer", None) is not None:
            self.model_ref = self.trainer.model
        # Progress string like "37% | 370/1066" for dashboard (use displayed_step so resume shows 1010, 1015, ...)
        max_steps = getattr(state, "max_steps", None) or getattr(args, "max_steps", None)
        progress = f"{100 * displayed_step // max_steps}% | {displayed_step}/{max_steps}" if (max_steps and max_steps > 0) else ""
        # Run inputs + predictions: on first log when we have model, then every update_every_n steps
        run_predictions = (
            (self.model_ref is not None and not self.sample_results)
            or (step - self.last_step_updated >= self.update_every_n)
        )
        if run_predictions:
            self.last_step_updated = step
            self._save_input_grid_and_run_inference(displayed_step)
            if self.sample_results:
                print(f"  Dashboard: wrote {len(self.sample_results)} sample predictions at step {displayed_step}")
            self._write_index(displayed_step, message="Updated.", sample_results=self.sample_results, progress=progress)
        else:
            self._write_index(displayed_step, message="Training...", sample_results=self.sample_results, progress=progress)
        self._write_training_state(displayed_step)

    def on_save(self, args, state, control, **kwargs):
        """Write checkpoint_alert.json; rename checkpoint dir to checkpoint-{displayed_step} so name matches step."""
        try:
            displayed_step = state.global_step + self.step_offset
            old_name = f"checkpoint-{state.global_step}"
            new_name = f"checkpoint-{displayed_step}"
            ckpt_dir = self.training_output_dir / old_name
            new_path = self.training_output_dir / new_name
            if ckpt_dir.exists() and ckpt_dir != new_path:
                shutil.move(str(ckpt_dir), str(new_path))
            alert = {
                "step": displayed_step,
                "path": new_name,
                "at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "at_sec": int(time.time()),
            }
            (self.output_dir / "checkpoint_alert.json").write_text(json.dumps(alert), encoding="utf-8")
        except Exception as e:
            print(f"Checkpoint alert write skipped: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        self._write_plot_data()
        displayed_step = state.global_step + self.step_offset
        self._write_training_state(displayed_step)
        max_steps = getattr(state, "max_steps", None) or getattr(args, "max_steps", None)
        progress = f"{100 * displayed_step // max_steps}% | {displayed_step}/{max_steps}" if (max_steps and max_steps > 0) else ""
        self._write_index(displayed_step, message="Training finished.", sample_results=self.sample_results, progress=progress)
        print(f"\n✓ Dashboard and plots in {self.output_dir}")


class SafetyDatasetSmolVLM(Dataset):
    """Dataset for SmolVLM2: one image + prompt -> class name (8 classes)."""

    def __init__(self, jsonl_path, frames_root, processor):
        self.frames_root = Path(frames_root)
        self.processor = processor
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.frames_root / sample["image"]
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        image = Image.open(image_path).convert("RGB")
        prompt = sample["prompt"]
        response = sample["response"]

        # SmolVLM2 conversation: image (PIL) + text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        inputs = self.processor.apply_chat_template(
            [conversation],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # inputs: input_ids, attention_mask, pixel_values, pixel_attention_mask, ...
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels.fill_(-100)
        if len(labels) > RESPONSE_LABEL_TOKENS:
            labels[-RESPONSE_LABEL_TOKENS:] = input_ids[-RESPONSE_LABEL_TOKENS:]

        out = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if "attention_mask" in inputs:
            out["attention_mask"] = inputs["attention_mask"].squeeze(0)
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "pixel_attention_mask" in inputs:
            out["pixel_attention_mask"] = inputs["pixel_attention_mask"].squeeze(0)
        return out


def collate_fn_smolvlm(features):
    batch = {}
    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    if features[0].get("attention_mask") is not None:
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
    if features[0].get("pixel_values") is not None:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    if features[0].get("pixel_attention_mask") is not None:
        batch["pixel_attention_mask"] = torch.stack(
            [f["pixel_attention_mask"] for f in features]
        )
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=SMOLVLM2_2B,
        choices=[SMOLVLM2_2B, SMOLVLM2_500M, SMOLVLM2_256M],
        help="SmolVLM2 model size",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps only")
    parser.add_argument("--plot-every", type=int, default=PLOT_UPDATE_EVERY_N_STEPS, help="Update loss + predictions in browser every N steps")
    parser.add_argument("--pred-every-n-frame", type=int, default=50, help="Show predictions for every N-th frame (0, N, 2N, ...)")
    parser.add_argument("--resume-from-checkpoint", nargs="?", default=None, const=True, metavar="PATH", help="Resume from latest checkpoint (no value) or from PATH")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")

    # Start live dashboard (Plotly.js plots + stats). Stats collected in a separate thread.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_log_path = OUTPUT_DIR / "train.log"
    training_state_path = PLOTS_DIR / "training_state.json"
    training_state_path.write_text(json.dumps({"step": 0, "training_elapsed_sec": 0, "training_elapsed_hms": "0h 0m 0s", "sample_accuracy_pct": None}), encoding="utf-8")
    resuming = bool(args.resume_from_checkpoint)
    if resuming:
        # Don't overwrite plot_data/plot_history; callback will restore from existing in on_train_begin
        pass
    else:
        # Fresh run: clear plot history so loss/accuracy start from 0
        empty_plot = {"loss": {"steps": [], "values": []}, "accuracy": {"steps": [], "values": []}}
        PLOTS_DIR.joinpath("plot_data.json").write_text(json.dumps(empty_plot), encoding="utf-8")
        (PLOTS_DIR / PLOT_HISTORY_FILENAME).write_text(json.dumps(empty_plot), encoding="utf-8")
    _start_live_server(LIVE_DASHBOARD_PORT, PLOTS_DIR, log_file_path=str(train_log_path))
    _start_stats_collector(PLOTS_DIR, training_state_path, interval_sec=5.0)
    (PLOTS_DIR / "index.html").write_text(
        _INDEX_HTML_TEMPLATE.format(step=0, message="Loading...", rows="<tr><td colspan='5'>Waiting for first update.</td></tr>", input_grid_img="<em>Waiting...</em>", update_every_n=PLOT_UPDATE_EVERY_N_STEPS, progress=""),
        encoding="utf-8",
    )
    print(f"Live dashboard: http://localhost:{LIVE_DASHBOARD_PORT}/")
    print("  -> Open this URL in your browser (not the index.html file). If you're on another machine, use http://<this-host-ip>:8084/")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SafetyDatasetSmolVLM(TRAIN_JSONL, FRAMES_ROOT, processor)
    class_names = sorted(set(s["response"] for s in dataset.samples))
    prompt_override = dataset.samples[0]["prompt"] if dataset.samples else ""
    print(f"Multiclass labels ({len(class_names)}): {class_names}")
    print("Training data is shuffled each epoch (Trainer default).")
    dashboard_callback = LiveDashboardCallback(
        PLOTS_DIR,
        processor,
        dataset,
        update_every_n=args.plot_every,
        pred_every_n_frame=args.pred_every_n_frame if args.pred_every_n_frame > 0 else None,
        training_output_dir=OUTPUT_DIR,
        model=model,
        resuming=resuming,
        prompt_override=prompt_override,
        valid_responses=class_names,
    )
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=20,
        save_total_limit=5,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0 if args.dry_run else 4,
    )
    # Resolve checkpoint path for resume (Trainer will load model + optimizer + step from it)
    resume_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is True:
            checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
            if checkpoints:
                def step_num(p):
                    try:
                        return int(p.name.split("-")[1])
                    except (IndexError, ValueError):
                        return -1
                resume_path = str(max(checkpoints, key=step_num))
        else:
            resume_path = args.resume_from_checkpoint
            # Resolve relative path (e.g. "checkpoint-990") against OUTPUT_DIR
            if resume_path and not Path(resume_path).is_absolute():
                resume_path = str(OUTPUT_DIR / Path(resume_path).name)
        if resume_path:
            ckpt = Path(resume_path)
            if not (ckpt / "trainer_state.json").exists():
                print(f"Checkpoint {resume_path} has no trainer_state.json; cannot resume step count, skipping resume")
                resume_path = None
            elif not ((ckpt / "adapter_model.safetensors").exists() or (ckpt / "adapter_model.bin").exists()):
                print(f"Checkpoint {resume_path} has no adapter weights; skipping resume")
                resume_path = None
            else:
                print(f"Will resume from checkpoint: {resume_path} (model, optimizer, scheduler, step)")
                # Use checkpoint dir name (e.g. checkpoint-990 -> 990) for display; trainer_state.global_step can be lower after rename
                try:
                    display_step = int(ckpt.name.split("-")[1])
                except (IndexError, ValueError):
                    display_step = None
                if display_step is not None:
                    dashboard_callback.resume_step = display_step
                else:
                    try:
                        ckpt_state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
                        dashboard_callback.resume_step = int(ckpt_state.get("global_step", 0))
                    except Exception:
                        dashboard_callback.resume_step = None
        else:
            dashboard_callback.resume_step = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn_smolvlm,
        callbacks=[dashboard_callback],
    )
    dashboard_callback.trainer = trainer  # so callback can use trainer.model if model_ref is ever None
    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except ValueError as e:
        if "torch" in str(e).lower() and ("2.6" in str(e) or "upgrade" in str(e).lower()):
            # Full resume uses torch.load; container has torch < 2.6. Fall back to adapter-only resume.
            if resume_path:
                print(f"Full resume failed (torch < 2.6): {e}")
                print(f"Falling back: loading adapter from {resume_path}, then training from step 0 (optimizer restarted).")
                model.load_adapter(resume_path, "default")
                # So dashboard/plots show steps 950, 955, ... instead of 0, 5, ...
                ckpt_path = Path(resume_path)
                # Use dir name (checkpoint-990 -> 990) so display continues from 990; trainer_state.global_step may be lower after rename
                try:
                    dashboard_callback.step_offset = int(ckpt_path.name.split("-")[1])
                except (IndexError, ValueError):
                    ckpt_state_path = ckpt_path / "trainer_state.json"
                    if ckpt_state_path.exists():
                        try:
                            ckpt_state = json.loads(ckpt_state_path.read_text(encoding="utf-8"))
                            dashboard_callback.step_offset = int(ckpt_state.get("global_step", 0))
                        except Exception as ex:
                            print(f"  Could not read checkpoint step for display: {ex}")
                if dashboard_callback.step_offset:
                    print(f"  Dashboard step offset set to {dashboard_callback.step_offset} (plot steps will continue from checkpoint).")
                dashboard_callback.trainer = trainer  # ensure callback has trainer for second train() so it can run sample predictions
                dashboard_callback.sample_results = []  # force running predictions on first on_log
                dashboard_callback.model_ref = model  # ensure model ref is set after load_adapter
                trainer.train(resume_from_checkpoint=None)
        else:
            raise
    if not args.dry_run:
        trainer.save_model()
        processor.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
