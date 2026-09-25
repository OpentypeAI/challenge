"""Multi-turn tool harness shared by the ops, sql and paint tracks (docs/tracks.md §5).

The worker runs episodes against the served model; the container scores only by replaying
the raw outputs through the same environment, so nothing the worker reports but the
model's own text is trusted. Model output is data: it is parsed as JSON, never executed.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

Content = list[dict[str, Any]]
MAX_OUTPUT_CHARS = 8192
INVALID = 'Invalid action. Reply with one JSON object: {"tool": "<name>", "args": {...}}'
OMITTED = "[earlier canvas omitted]"


@dataclass(frozen=True)
class Env:
    name: str
    system: Callable[[dict[str, Any]], str]
    reset: Callable[[dict[str, Any]], Any]
    observe: Callable[[dict[str, Any], Any], Content]
    step: Callable[[dict[str, Any], Any, dict[str, Any] | None], tuple[Content, bool]]
    loss: Callable[[dict[str, Any], Any], float | None]


def text(value: str) -> Content:
    return [{"type": "text", "text": value}]


def parse_action(output: str) -> dict[str, Any] | None:
    """The first balanced top-level JSON object with a string `tool` and an object `args`."""
    decoder = json.JSONDecoder()
    index = output.find("{")
    while index != -1:
        try:
            value, _ = decoder.raw_decode(output, index)
        except ValueError:
            value = None
        if (
            isinstance(value, dict)
            and isinstance(value.get("tool"), str)
            and isinstance(value.get("args", {}), dict)
        ):
            return {"tool": value["tool"], "args": value.get("args", {})}
        index = output.find("{", index + 1)
    return None


def _strip_images(content: Content) -> Content:
    return [
        part if part.get("type") != "image_url" else {"type": "text", "text": OMITTED}
        for part in content
    ]


def chat_messages(
    env: Env, task: dict[str, Any], history: Sequence[tuple[str, Content]], first: Content
) -> list[dict[str, Any]]:
    """system, user(first observation), then (assistant output, user observation) per turn.
    Only the latest image part survives; earlier ones become OMITTED."""
    observations = [first, *(obs for _, obs in history)]
    last_image = max(
        (i for i, obs in enumerate(observations) if any(p.get("type") == "image_url" for p in obs)),
        default=-1,
    )
    shown = [obs if i == last_image else _strip_images(obs) for i, obs in enumerate(observations)]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": env.system(task)},
        {"role": "user", "content": shown[0]},
    ]
    for (output, _), observation in zip(history, shown[1:], strict=True):
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": observation})
    return messages


Generate = Callable[[list[dict[str, Any]], int], Awaitable[str]]


async def run_episode(env: Env, body: dict[str, Any], generate: Generate) -> list[str]:
    """Worker side: play up to limits.turns turns; the raw outputs are the transcript."""
    task, turns = body["task"], int(body["limits"]["turns"])
    state = env.reset(task)
    first = env.observe(task, state)
    history: list[tuple[str, Content]] = []
    for turn in range(turns):
        output = (await generate(chat_messages(env, task, history, first), body["seed"] + turn))[
            :MAX_OUTPUT_CHARS
        ]
        observation, done = env.step(task, state, parse_action(output))
        history.append((output, observation))
        if done:
            break
    return [output for output, _ in history]


def replay(env: Env, body: dict[str, Any], outputs: Sequence[str]) -> tuple[Any, float | None]:
    """Container side: rebuild the final state from the outputs alone and score it."""
    task, turns = body["task"], int(body["limits"]["turns"])
    state = env.reset(task)
    for output in list(outputs)[:turns]:
        if not isinstance(output, str):
            break
        _, done = env.step(task, state, parse_action(output[:MAX_OUTPUT_CHARS]))
        if done:
            break
    return state, env.loss(task, state)
