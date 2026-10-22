"""Stand-in for `vllm serve` in tests: same CLI shape, OpenAI-compatible /v1/chat/completions,
answers with MockBackend. Usage: python tests/fake_vllm.py serve MODEL --port N [...ignored]"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifelike_bench.backends import MockBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd")
ap.add_argument("model")
ap.add_argument("--port", type=int, required=True)
args, _ = ap.parse_known_args()
mock = MockBackend()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200 if self.path == "/health" else 404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        rest = [m for m in msgs if m["role"] != "system"]
        text = mock.chat(system, rest, seed=req.get("seed"))
        if "Character reply" not in rest[-1]["content"] and "JSON" not in system:
            text = "<think>hidden reasoning</think>" + text   # exercise think-stripping
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
