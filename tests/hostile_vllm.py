"""A fake `vllm serve` that misbehaves on request, for the sandbox relay tests.

POST /emoji   6 MB of 4-byte characters (a frame that ASCII escaping would triple)
POST /deep    100k nested lists (RecursionError in json.loads)
POST /error   a 500
POST /hang    never answers
POST /spoof   kill the reader (its pid file) and serve /v1/systemone on its port
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _raw(self, code: int, raw: bytes) -> None:
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._raw(200, b'{"status":"ok"}')

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path == "/emoji":
            self._raw(200, json.dumps({"x": "\U0001f600" * 1_400_000}, ensure_ascii=False).encode())
        elif self.path == "/deep":
            self._raw(200, b"[" * 100_000 + b"]" * 100_000)
        elif self.path == "/error":
            self._raw(500, b'{"error":"boom"}')
        elif self.path == "/hang":
            time.sleep(60)
        elif self.path == "/spoof":
            pid = int(Path(PID_FILE).read_text())
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            port = READER_PORT
            spoof = ThreadingHTTPServer(("127.0.0.1", port), Spoof)
            threading.Thread(target=spoof.serve_forever, daemon=True).start()
            self._raw(200, b'{"spoofing":true}')
        else:
            self._raw(404, b'{"error":"unknown"}')


class Spoof(Handler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self._raw(200, b'{"answers":{"forged":true}}')


PID_FILE, READER_PORT = "", 0


def main(argv: list[str]) -> None:
    """argv: <reader pid file> <reader port> serve <model> ... --port N (as vllm is called)."""
    global PID_FILE, READER_PORT
    PID_FILE, READER_PORT = argv[0], int(argv[1])
    port = int(argv[argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
