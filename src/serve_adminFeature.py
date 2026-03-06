"""
serve_adminFeature.py  —  Safety Logic Engine
Port 8089 | CPU inference | Uses EXACT same load_smolvlm + ask_smolvlm
from serve_vlm_qa.py — no new inference code.

Drop into ./src/
Usage:
    python src/serve_adminFeature.py \
        --checkpoint /workspace/outputs/checkpoint-2056 \
        --port 8089
"""

import argparse
import inspect
import json
import os
import time
from io import BytesIO

import psutil
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="/workspace/outputs/checkpoint-2056")
parser.add_argument("--port", type=int, default=8089)
parser.add_argument("--host", type=str, default="0.0.0.0")
args = parser.parse_args()

app = FastAPI(title="Safety Logic Engine")

# ── Globals ───────────────────────────────────────────────────────────────────
processor = None
model = None
device = torch.device("cpu")

BASE_MODEL   = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MAX_NEW_TOKENS = 80


# ── Exact copy of load_smolvlm from serve_vlm_qa.py ──────────────────────────
def load_smolvlm(model_name: str, dev):
    global processor, model
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {
        "torch_dtype": torch.bfloat16 if dev.type == "cuda" else torch.float32,
        "device_map": str(dev) if dev.type == "cuda" else None,
        "trust_remote_code": True,
    }
    try:
        sig = inspect.signature(AutoModelForImageTextToText.from_pretrained)
        if "attn_implementation" in sig.parameters:
            kwargs["attn_implementation"] = "eager"
    except Exception:
        pass
    try:
        mdl = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    except TypeError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mdl = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **{k: v for k, v in kwargs.items() if k != "attn_implementation"}
        )
    # Apply int8 quantization on CPU — ~50% less RAM, ~2x faster inference
    if dev.type == "cpu":
        print("[SafetyEngine] Applying int8 quantization for CPU...")
        mdl = torch.quantization.quantize_dynamic(
            mdl, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("[SafetyEngine] Quantization done.")
    processor = proc
    model = mdl
    model.eval()


# ── Exact copy of ask_smolvlm from serve_vlm_qa.py ───────────────────────────
def ask_smolvlm(image: Image.Image, question: str) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  question},
            ],
        }
    ]
    inp = processor.apply_chat_template(
        messages,
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


# ── HTML / CSS / JS ───────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Safety Logic Engine</title>
<style>
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
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}

header{background:var(--surf);border-bottom:1px solid var(--border);height:54px;padding:0 22px;display:flex;align-items:center;gap:11px;flex-shrink:0}
.logo{width:32px;height:32px;background:linear-gradient(135deg,#f0b429,#e05c00);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px}
header h1{font-size:1rem;font-weight:700}
.pill{border-radius:20px;padding:3px 10px;font-size:.7rem;font-weight:600;border:1px solid}
.pill-green{background:#1a2e1a;color:var(--green-t);border-color:var(--green)}
.pill-gray{background:var(--surf2);color:var(--muted);border-color:var(--border);font-family:monospace;font-weight:400}
header .right{margin-left:auto;display:flex;gap:8px;align-items:center}

.layout{display:grid;grid-template-columns:310px 1fr;flex:1;overflow:hidden}

aside{background:var(--surf);border-right:1px solid var(--border);overflow-y:auto;padding:16px 14px;display:flex;flex-direction:column;gap:16px}
.sec{font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted);margin-bottom:7px}

select.zone-sel{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:var(--r);padding:8px 10px;font-size:.85rem}
select.zone-sel:focus{outline:none;border-color:var(--accent)}

#crit-list{display:flex;flex-direction:column;gap:6px}
.crit-item{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:8px 10px;display:flex;align-items:flex-start;gap:7px}
.cnum{min-width:19px;height:19px;background:var(--surf2);border-radius:50%;font-size:.62rem;font-weight:700;color:var(--muted);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.crit-item textarea{flex:1;background:transparent;border:none;color:var(--text);font-size:.8rem;resize:none;outline:none;line-height:1.4;font-family:inherit}
.del{background:none;border:none;color:var(--muted);cursor:pointer;font-size:.85rem;flex-shrink:0;transition:color .15s}
.del:hover{color:var(--red-t)}
.add-crit{width:100%;background:none;border:1px dashed var(--border);color:var(--muted);border-radius:var(--r);padding:7px 10px;font-size:.8rem;cursor:pointer;transition:all .15s;margin-top:5px}
.add-crit:hover{border-color:var(--accent);color:var(--accent)}

.sev-rules{display:flex;flex-direction:column;gap:5px}
.sev-row{display:flex;align-items:center;gap:7px;background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:7px 10px;font-size:.78rem}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.sev-row select{margin-left:auto;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:2px 6px;font-size:.75rem}

#go-btn{width:100%;padding:11px;background:linear-gradient(135deg,#f0b429,#e05c00);color:#000;border:none;border-radius:var(--r);font-size:.9rem;font-weight:700;cursor:pointer;transition:opacity .15s,transform .1s;letter-spacing:.3px}
#go-btn:hover{opacity:.88;transform:translateY(-1px)}
#go-btn:active{transform:translateY(0)}
#go-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

.main{display:flex;flex-direction:column;overflow:hidden}
.main-body{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:18px}

.upload-zone{border:2px dashed var(--border);border-radius:14px;min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;cursor:pointer;transition:border-color .2s,background .2s;position:relative;overflow:hidden}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(240,180,41,.04)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-zone .icon{font-size:2.2rem}
.upload-zone p{color:var(--muted);font-size:.85rem}
.upload-zone strong{color:var(--accent)}
#prev-wrap{text-align:center;display:none}
#prev-img{max-height:300px;border-radius:10px;object-fit:contain;display:none}
.img-meta{font-size:.72rem;color:var(--muted);margin-top:5px}

.res-header{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.verdict{padding:5px 16px;border-radius:20px;font-weight:700;font-size:.95rem;letter-spacing:.4px;border:1px solid}
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
.ans-pill{padding:3px 9px;border-radius:20px;font-size:.72rem;font-weight:700;white-space:nowrap;border:1px solid}
.ans-yes{background:#1a2e1a;color:var(--green-t);border-color:var(--green)}
.ans-no{background:#2e1414;color:var(--red-t);border-color:var(--red)}

/* Resource panel */
.res-panel{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px}
.res-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:10px}
.res-tile{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;text-align:center}
.res-tile .val{font-size:1.1rem;font-weight:700;color:var(--accent);font-family:monospace}
.res-tile .lbl{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}
.ram-bar-bg{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:8px}
.ram-bar{height:100%;border-radius:3px;transition:width .6s ease}

.raw-box{background:#0a0f15;border:1px solid var(--border);border-radius:var(--r);padding:13px 15px}
.raw-box pre{font-family:'Cascadia Code','Fira Code',monospace;font-size:.75rem;color:#7ee787;white-space:pre-wrap;word-break:break-word;line-height:1.6}

.empty{text-align:center;padding:50px 20px;color:var(--muted)}
.empty .big{font-size:2.8rem;margin-bottom:12px}

.statusbar{background:var(--surf);border-top:1px solid var(--border);padding:6px 18px;font-size:.72rem;color:var(--muted);display:flex;align-items:center;gap:12px;flex-shrink:0}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--green-t)}

.spinner-wrap{position:fixed;inset:0;background:rgba(13,17,23,.88);display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:999}
.spinner-wrap.on{display:flex}
.spinner{width:44px;height:44px;border:4px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner-wrap p{font-size:.95rem}
.spinner-wrap small{color:var(--muted);font-size:.8rem}

#toast{position:fixed;bottom:44px;right:18px;background:var(--surf2);border:1px solid var(--border);border-radius:10px;padding:11px 16px;font-size:.82rem;transform:translateY(70px);opacity:0;transition:all .3s ease;z-index:1000;max-width:300px}
#toast.show{transform:translateY(0);opacity:1}
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
        <div class="sec">&#x1F4F7; CCTV Frame</div>
        <div class="upload-zone" id="drop-zone"
             ondragover="onDragOver(event)" ondragleave="onDragLeave()" ondrop="onDrop(event)">
          <input type="file" id="file-in" accept="image/*" onchange="onFileChange(event)"/>
          <div class="icon">&#x1F4C2;</div>
          <p><strong>Click or drag</strong> a CCTV frame here</p>
          <p>PNG &middot; JPG &middot; JPEG</p>
        </div>
        <div id="prev-wrap">
          <img id="prev-img" alt="preview"/>
          <div class="img-meta" id="img-meta"></div>
        </div>
      </div>

      <div id="results" style="display:none">
        <div class="res-header">
          <div class="sec" style="margin:0">&#x1F4CA; Results</div>
          <div id="verdict-badge" class="verdict"></div>
          <span id="elapsed" style="color:var(--muted);font-size:.75rem;margin-left:auto"></span>
        </div>
        <div id="crit-results" class="crit-results"></div>

        <div style="margin-top:14px" id="resource-panel" style="display:none">
          <div class="sec">&#x26A1; Resource Usage</div>
          <div class="res-panel">
            <div class="res-grid" id="res-grid"></div>
            <div style="margin-top:10px">
              <div style="display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);margin-bottom:3px">
                <span>System RAM</span><span id="ram-pct-lbl"></span>
              </div>
              <div class="ram-bar-bg"><div class="ram-bar" id="ram-bar"></div></div>
            </div>
          </div>
        </div>

        <div style="margin-top:14px">
          <div class="sec">&#x1F9E0; Raw Model Output</div>
          <div class="raw-box"><pre id="raw-out"></pre></div>
        </div>
      </div>

      <div id="empty" class="empty">
        <div class="big">&#x1F3AF;</div>
        <p>Upload a frame and configure criteria,<br>then click <strong>Analyze Frame</strong>.</p>
      </div>
    </div>

    <div class="statusbar">
      <span class="sdot"></span>
      <span>SmolVLM2 &middot; CPU</span>
      <span>|</span>
      <span id="status-msg">Idle</span>
    </div>
  </div>
</div>

<div class="spinner-wrap" id="spinner">
  <div class="spinner"></div>
  <p id="spinner-msg">Analyzing frame...</p>
  <small id="spinner-sub">Running criterion 1 of 1...</small>
</div>
<div id="toast"></div>

<script>
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

let currentFile = null;

window.onload = () => { loadPreset(); pollHealth(); };

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

async function runAnalysis(){
  const criteria=getCriteria();
  if(!currentFile){ toast('\u26a0\ufe0f Upload a CCTV frame first'); return; }
  if(!criteria.length){ toast('\u26a0\ufe0f Add at least one criterion'); return; }
  setLoading(true, criteria.length);
  setStatus('Analyzing '+criteria.length+' criteria...');
  const t0=Date.now();
  try {
    const fd=new FormData();
    fd.append('image', currentFile);
    fd.append('criteria', JSON.stringify(criteria));
    fd.append('sev_one',   document.getElementById('sev1').value);
    fd.append('sev_multi', document.getElementById('sev2').value);
    const res=await fetch('/analyze',{method:'POST',body:fd});
    if(!res.ok) throw new Error(await res.text());
    const data=await res.json();
    const elapsed=((Date.now()-t0)/1000).toFixed(1);
    renderResults(data, elapsed);
    setStatus('Done in '+elapsed+'s \u00b7 '+data.failures+' failure(s)');
  } catch(err){
    toast('\u274c '+err.message); setStatus('Error'); console.error(err);
  } finally { setLoading(false); }
}

function renderResults(data, elapsed){
  document.getElementById('results').style.display='block';
  document.getElementById('empty').style.display='none';
  const vb=document.getElementById('verdict-badge');
  vb.textContent=data.verdict;
  vb.className='verdict '+({SAFE:'v-safe',WARNING:'v-warn',CRITICAL:'v-crit'}[data.verdict]||'v-crit');
  document.getElementById('elapsed').textContent='\u23f1 '+elapsed+'s';

  const container=document.getElementById('crit-results');
  container.innerHTML='';
  data.results.forEach((r,i)=>{
    const pass=r.answer.toLowerCase().startsWith('yes');
    container.innerHTML+=`
      <div class="res-card ${pass?'pass':'fail'}">
        <span class="ri">${pass?'\u2705':'\u274c'}</span>
        <div>
          <div class="q-text">Q${i+1}: ${esc(r.question)}</div>
          <div class="obs-text">${esc(r.answer)}</div>
        </div>
        <span class="ans-pill ${pass?'ans-yes':'ans-no'}">${pass?'YES':'NO'}</span>
      </div>`;
  });

  // Resource panel
  const s=data.resource_stats;
  if(s){
    document.getElementById('resource-panel').style.display='block';
    const tiles=[
      {val:s.total_time_s+'s',       lbl:'Total Time'},
      {val:s.criteria_count,         lbl:'Criteria Run'},
      {val:s.avg_time_per_criteria+'s', lbl:'Avg / Criterion'},
      {val:s.cpu_pct+'%',            lbl:'CPU %'},
      {val:s.process_ram_mb+' MB',   lbl:'Process RAM'},
      {val:(s.ram_delta_mb>=0?'+':'')+s.ram_delta_mb+' MB', lbl:'RAM Delta'},
    ];
    document.getElementById('res-grid').innerHTML=tiles.map(t=>
      `<div class="res-tile"><div class="val">${t.val}</div><div class="lbl">${t.lbl}</div></div>`
    ).join('');
    const pct=s.system_ram_pct||0;
    const col=pct>85?'var(--red-t)':pct>65?'var(--yel-t)':'var(--green-t)';
    document.getElementById('ram-bar').style.cssText=`width:${pct}%;background:${col}`;
    document.getElementById('ram-pct-lbl').textContent=s.system_ram_used_mb+' MB / '+s.system_ram_total_mb+' MB ('+pct+'%)';
  }

  document.getElementById('raw-out').textContent=data.raw_output||'(empty)';
}

function setLoading(on, count){
  document.getElementById('spinner').classList.toggle('on',on);
  document.getElementById('go-btn').disabled=on;
  if(on && count){
    document.getElementById('spinner-sub').textContent=
      'Running '+count+' criteria \u2014 may take '+(count*20)+'s on CPU';
  }
}
function setStatus(msg){ document.getElementById('status-msg').textContent=msg; }
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
@app.on_event("startup")
async def startup():
    torch.set_grad_enabled(False)
    print(f"[SafetyEngine] Loading base model: {BASE_MODEL}")
    load_smolvlm(BASE_MODEL, device)
    print("[SafetyEngine] Ready.")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    return {
        "status":              "ready" if model is not None else "loading",
        "model_loaded":        model is not None,
        "checkpoint":          args.checkpoint,
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
    if model is None:
        return JSONResponse({"error": "Model not ready yet"}, status_code=503)

    questions  = json.loads(criteria)
    pil_image  = Image.open(BytesIO(await image.read())).convert("RGB")
    # Force 384x384 — minimum patch size for SmolVLM2.
    # Larger images create exponentially more image tokens and dominate inference time.
    # 384x384 gives correct answers and is ~4x faster than a typical 1080p CCTV frame.
    pil_image = pil_image.resize((384, 384), Image.LANCZOS)

    results   = []
    raw_parts = []
    proc      = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024 / 1024
    ram_sys_before = psutil.virtual_memory().used / 1024 / 1024
    t_total   = time.time()

    # ── One ask_smolvlm call per criterion — exact same path as serve_vlm_qa ──
    for i, q in enumerate(questions):
        # Prompt: ask for Yes/No + one-line reason, same style as vlm_qa
        prompt = (
            f"{q}\n"
            "Answer with Yes or No, then explain in one sentence why."
        )
        t0  = time.time()
        raw = ask_smolvlm(pil_image, prompt)
        elapsed = round(time.time() - t0, 2)

        raw_parts.append(f"[Q{i+1}] {q}\n-> {raw}  ({elapsed}s)")

        # Parse Yes/No from the raw answer
        lower = raw.lower().strip()
        if lower.startswith("yes"):
            answer = "Yes"
        elif lower.startswith("no"):
            answer = "No"
        else:
            # fallback — look for yes/no anywhere in first 10 chars
            answer = "Yes" if "yes" in lower[:10] else "No"

        results.append({
            "question": q,
            "answer":   answer,
            "raw":      raw,
        })

    total_time = round(time.time() - t_total, 2)
    ram_after  = proc.memory_info().rss / 1024 / 1024
    ram_sys_after = psutil.virtual_memory().used / 1024 / 1024
    mem        = psutil.virtual_memory()

    failures = sum(1 for r in results if r["answer"] != "Yes")
    verdict  = "SAFE" if failures == 0 else (sev_one if failures == 1 else sev_multi)

    return JSONResponse({
        "verdict":    verdict,
        "results":    results,
        "raw_output": "\n\n".join(raw_parts),
        "failures":   failures,
        "resource_stats": {
            "total_time_s":          total_time,
            "criteria_count":        len(questions),
            "avg_time_per_criteria": round(total_time / max(len(questions), 1), 2),
            "cpu_pct":               round(psutil.cpu_percent(interval=None), 1),
            "process_ram_mb":        round(ram_after, 1),
            "ram_delta_mb":          round(ram_after - ram_before, 1),
            "system_ram_used_mb":    round(ram_sys_after, 1),
            "system_ram_total_mb":   round(mem.total / 1024 / 1024, 1),
            "system_ram_pct":        round(mem.percent, 1),
        },
    })


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)