#!/usr/bin/env python3
"""
Port map hub: serves a single page on port 8080 listing all services and links.
Run with docker-compose so you have one place to see what runs where.
"""

import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

HUB_PORT = 8080

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safe/Unsafe demo — Port map</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 640px; background: #0f0f1a; color: #e0e0e0; }
    h1 { color: #e94560; font-size: 1.5rem; }
    p { color: #aaa; font-size: 0.95rem; }
    .card { background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 1.25rem; margin: 1rem 0; }
    .card h2 { margin: 0 0 0.25rem 0; font-size: 1.1rem; }
    .card a { color: #e94560; text-decoration: none; font-weight: 600; }
    .card a:hover { text-decoration: underline; }
    .port { color: #888; font-family: monospace; font-size: 0.9rem; }
    .badge { display: inline-block; background: #2e7d32; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 0.5rem; }
    ul { margin: 0.5rem 0; padding-left: 1.25rem; color: #aaa; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Safe/Unsafe demo — what runs where</h1>
  <p>Open this page (localhost:8080) anytime to see all services. Click a link to go there.</p>

  <div class="card">
    <h2><span class="port">8080</span> Port map <span class="badge">you are here</span></h2>
    <p>This page. Bookmark it to avoid port confusion.</p>
  </div>

  <div class="card">
    <h2><a href="http://localhost:8082/" target="_blank" rel="noopener">8082 — Video inference</a></h2>
    <p>CCTV test videos: upload or pick from dataset, run Safe/Unsafe per video (max-vote over frames).</p>
  </div>

  <div class="card">
    <h2><a href="http://localhost:8083/" target="_blank" rel="noopener">8083 — YouTube inference</a></h2>
    <p>Paste a YouTube URL; video is embedded and inference runs in the cloud (no download).</p>
    <p style="margin-top:0.5rem; font-size:0.85rem;">If 8083 doesn’t load, run <code>docker compose ps</code> and ensure the <code>youtube</code> service is up.</p>
  </div>

  <div class="card">
    <h2><a href="http://localhost:8086/" target="_blank" rel="noopener">8086 — Multiclass video inference</a></h2>
    <p>8-class behavior classification on test videos (separate compose: <code>docker-compose.multiclass-inference.yml</code>).</p>
  </div>

  <div class="card">
    <h2><a href="http://localhost:8087/" target="_blank" rel="noopener">8087 — VLM QA</a></h2>
    <p>Ask questions about an image (e.g. &quot;Is the person wearing a hardhat? yes/no&quot;). SmolVLM2 or LLaVA 7B on GPU. For YouTube/webcam streams: send frames + questions, build answers from replies.</p>
  </div>

  <p style="margin-top: 2rem; color: #666; font-size: 0.85rem;">
    When running training manually (not via compose):<br>
    Binary training dashboard → 8080 (or 8081 if you changed it).<br>
    Multiclass training dashboard → 8081.
  </p>
</body>
</html>
"""


class HubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = (self.path.split("?")[0] or "/").rstrip("/") or "/"
        if path != "/" and path != "/index.html":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(INDEX_HTML.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Port map hub — serves a single page listing all services")
    parser.add_argument("--port", type=int, default=HUB_PORT, help="Port (default %d)" % HUB_PORT)
    args = parser.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), HubHandler)
    print("Port hub: http://0.0.0.0:%d/ (open in browser to see what runs where)" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
