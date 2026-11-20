"""Serve the local GPU Playground using only the Python standard library."""
import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments import run_experiment
from puzzles import map_double

ROOT = Path(__file__).resolve().parent
ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css")}


def map_result():
    values = [-3, 0, 4, 7, 10]
    try:
        run = subprocess.run([sys.executable, "-B", str(ROOT / "gpu_map.py")],
                             capture_output=True, text=True, timeout=20, check=True)
        log = run.stdout.strip()
        output = json.loads(log.split(" -> ", 1)[1].split(" | ", 1)[0])
        return dict(source="gpu", input=values, output=output, log=log)
    except subprocess.CalledProcessError as error:
        log = error.stderr.strip().splitlines()
        reason = log[-1] if log else f"退出码 {error.returncode}"
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError) as error:
        reason = str(error)
    return dict(source="cpu-reference", input=values, output=map_double(values), log=reason)


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self, status, payload):
        self.reply(status, "application/json", json.dumps(payload, ensure_ascii=False).encode())

    def do_GET(self):
        asset = ASSETS.get(self.path)
        if asset is None:
            self.send_error(404)
            return
        name, content_type = asset
        self.reply(200, content_type, (ROOT / "web" / name).read_bytes())

    def do_POST(self):
        if self.path not in ("/api/experiment", "/api/map"):
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.json(415, {"error": "请提交 JSON。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("请求大小无效。")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("参数须为 JSON 对象。")
            result = map_result() if self.path == "/api/map" else run_experiment(payload)
        except ValueError as error:
            self.json(400, {"error": str(error)})
            return
        self.json(200, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"GPU Playground: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

