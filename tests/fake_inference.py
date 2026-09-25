"""A stand-in for `vllm serve` and structured_server.py (stdlib + the challenge package only).

As `vllm serve <model dir> ... --port N` it answers GET /health and POST /v1/chat/completions.
As the structured server (`--upstream ... --tokenizer <model dir> --port N`) it answers
POST /v1/systemone in the pinned server's shapes. Its skill comes from
<model dir>/model.safetensors:
  exact  -> reads: the exact posterior; chat: the env's reference oracle (a perfect model)
  base   -> reads: gold blurred toward uniform, a quarter of the items pushed to a wrong
            option; chat: the oracle, but a deliberately wrong final action on about a
            quarter of the conversations (by a hash of the first user message)
  broken -> reads: uniform answers; chat: always an invalid reply
Chat dispatches on the first system line "OpenType harness: <env>". Where the oracle has none
(teacher prose, paint depict) every skill plays the same fallback: a depict painter draws one
shape and calls done, any other env gets an invalid reply.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from opentype_challenge.tracks import solve_body

SKILL = "exact"
HEADER = "OpenType harness: "
MODULES = {"ops": "ops", "sql": "sqltask", "paint": "paint"}
INVALID_REPLY = "I would rather not use a tool right now."
DEPICT_DRAW = json.dumps(
    {
        "tool": "draw",
        "args": {"commands": [{"op": "circle", "cx": 128, "cy": 128, "r": 60, "fill": "#cc2222"}]},
    }
)
DONE = json.dumps({"tool": "done", "args": {}})
# A final action that is wrong whatever the task: finishing an ops conversation as escalated
# without escalating, answering an sql question with a value no table holds, or ending a
# paint episode on a blank canvas.
WRONG = {
    "ops": json.dumps({"tool": "finish", "args": {"outcome": "escalated"}}),
    "sql": json.dumps({"tool": "answer", "args": {"value": "no such value 0x7f3a"}}),
    "paint": DONE,
}


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


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def wrong_conversation(messages: list[dict]) -> bool:
    """About a quarter of the conversations, by a hash of the first user message."""
    first = json.dumps(messages[1].get("content"), sort_keys=True)
    return hashlib.sha256(first.encode()).digest()[0] < 64


def chat_reply(skill: str, messages: list[dict]) -> str:
    """The next raw output of a model of this skill in one harness conversation."""
    env = _text(messages[0].get("content")).split("\n", 1)[0].removeprefix(HEADER)
    if skill == "broken" or env not in MODULES:
        return INVALID_REPLY
    policy = importlib.import_module(f"opentype_challenge.{MODULES[env]}").reference_policy
    if skill == "base" and wrong_conversation(messages):
        return WRONG[env]
    try:
        return str(policy(messages))
    except ValueError:  # no oracle (teacher prose, depict): the same fallback on both sides
        if env != "paint":
            return INVALID_REPLY
        return DONE if any(m.get("role") == "assistant" for m in messages) else DEPICT_DRAW


def chat_problem(body: Any) -> str | None:
    """The worker's §7 request shape; anything else is a 400."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return "messages must be a list"
    if len(body["messages"]) < 2 or body["messages"][0].get("role") != "system":
        return "expected a system message then a user message"
    if not isinstance(body.get("model"), str) or type(body.get("max_tokens")) is not int:
        return "model and max_tokens are required"
    if body.get("temperature") != 0.0 or type(body.get("seed")) is not int:
        return "temperature 0.0 and an int seed are required"
    return None


def chat(skill: str, body: Any) -> tuple[int, dict]:
    """(status, payload) of POST /v1/chat/completions."""
    problem = chat_problem(body)
    if problem:
        return 400, {"error": {"message": problem}}
    message = {"role": "assistant", "content": chat_reply(skill, body["messages"])}
    return 200, {"choices": [{"index": 0, "message": message}]}


def systemone(skill: str, body: Any) -> tuple[int, dict]:
    """(status, payload) of POST /v1/systemone."""
    questions = body.get("questions") if isinstance(body, dict) else None
    if not isinstance(questions, dict) or body.get("samples") != "auto":
        return 422, {"error": {"message": "bad schema"}}
    gold = solve_body(body)
    answers = {
        qid: answer(q, blur(skill, qid, int(body["seed"]), gold[qid]))
        for qid, q in questions.items()
    }
    reads = {qid: {"label_mass": 0.97, "argmax_is_label": True} for qid in questions}
    return 200, {"model": "fake", "answers": answers, "diagnostics": {"questions": reads}}


ROUTES = {"/v1/chat/completions": chat, "/v1/systemone": systemone}


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
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        route = ROUTES.get(self.path)
        if route is None:
            return self._json(404, {"error": {"message": "unknown route"}})
        self._json(*route(SKILL, body))


def main(argv: list[str]) -> None:
    global SKILL
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tokenizer")
    serve = bool(argv) and argv[0] == "serve"
    args, _ = parser.parse_known_args(argv[2:] if serve else argv)
    model = argv[1] if serve else args.tokenizer
    if model:
        SKILL = (Path(model) / "model.safetensors").read_text().split()[0]
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
