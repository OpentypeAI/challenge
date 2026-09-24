"""A stand-in for `vllm serve` and structured_server.py (stdlib + the generator only).

As `vllm serve <model> ... --port N` it only answers GET /health. As the structured server
(`--upstream ... --tokenizer <model dir> --port N`) it answers POST /v1/systemone in the
pinned server's shapes. Its skill comes from <model dir>/model.safetensors:
  exact  -> the exact posterior (a perfect model)
  base   -> gold blurred toward uniform, and a quarter of the items pushed to a wrong option
  broken -> uniform answers on every item
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from opentype_challenge.generator import solve

SKILL = "exact"


def blur(skill: str, qid: str, seed: int, gold: list[float]) -> list[float]:
    k = len(gold)
    if skill == "broken":
        return [1 / k] * k
    if skill == "exact":
        return gold
    probs = [0.7 * g + 0.3 / k for g in gold]
    if hashlib.sha256(f"{seed}|{qid}".encode()).digest()[0] < 64:
        top = max(range(k), key=gold.__getitem__)
        wrong = (top + 1) % k
        probs = [0.1 / k] * k
        probs[wrong] += 0.9
    return probs


def answer(question: dict, probs: list[float]) -> dict:
    top = max(range(len(probs)), key=probs.__getitem__)
    if question["type"] == "noul":
        return {"noul": probs[0]}
    if question["type"] == "choice":
        names = list(question["criteria"])
        return {
            "choice": names[top],
            "probabilities": dict(zip(names, probs, strict=True)),
            "confidence": probs[top],
        }
    levels = question["criteria"]
    return {
        "score": sum(i * p for i, p in enumerate(probs)),
        "legend": {str(i): name for i, name in enumerate(levels)},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "confidence": probs[top],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._json(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/v1/systemone":
            return self._json(404, {"error": {"message": "unknown route"}})
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        questions = body.get("questions")
        if not isinstance(questions, dict) or body.get("samples") != "auto":
            return self._json(422, {"error": {"message": "bad schema"}})
        gold = solve(body)
        answers = {
            qid: answer(q, blur(SKILL, qid, int(body["seed"]), gold[qid]))
            for qid, q in questions.items()
        }
        reads = {qid: {"label_mass": 0.97, "argmax_is_label": True} for qid in questions}
        self._json(200, {"model": "fake", "answers": answers, "diagnostics": {"questions": reads}})


def main(argv: list[str]) -> None:
    global SKILL
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tokenizer")
    args, _ = parser.parse_known_args(argv[1:] if argv and argv[0] == "serve" else argv)
    if args.tokenizer:
        SKILL = (Path(args.tokenizer) / "model.safetensors").read_text().split()[0]
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
