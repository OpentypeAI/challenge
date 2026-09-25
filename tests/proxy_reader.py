"""A reader that forwards every POST to its --upstream vllm and masks an upstream failure as
the pinned structured_server.py does (_decide: HTTPError -> 502, any other error -> 500)."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ""


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
        data = self.rfile.read(int(self.headers.get("content-length", 0)))
        request = urllib.request.Request(UPSTREAM + self.path, data=data, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                self._raw(200, response.read())
        except urllib.error.HTTPError:
            self._raw(502, b'{"error":"upstream error"}')
        except Exception:  # noqa: BLE001 - as the pinned server: everything else is a 500
            self._raw(500, b'{"error":"internal"}')


def main(argv: list[str]) -> None:
    global UPSTREAM
    UPSTREAM = argv[argv.index("--upstream") + 1]
    port = int(argv[argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
