"""Teacher gateway client and per-window bank builder (docs/tracks.md §2, §3, §11).

The teacher LLM writes private content (sealed families, prose records, customer stories,
drawing briefs); it never labels an answer. A text is kept only when every extractor model
recovers its exact structured spec (round trip), and a drawing rubric only when the judge
fails every item on a blank canvas (negative control).

The token is re-read from its file on every call and never logged or put in an exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import __version__, ops, paint
from . import generator as g
from .bank import BANK_KINDS, BankItem

__all__ = [
    "BANK_KINDS",
    "DEFAULT_TARGETS",
    "Gateway",
    "GatewayError",
    "TeacherConfig",
    "build_bank",
    "judge_png",
    "round_trip",
]

log = logging.getLogger(__name__)

DEFAULT_TARGETS: dict[str, int] = {"family": 4, "prose": 600, "ops_story": 200, "depict": 80}
DEFAULT_TOKEN_FILE = Path("/run/secrets/teacher.token")
TIMEOUT = 180.0
RETRIES = 6  # extra attempts on transport errors, 408, 409, 429 and 5xx
BACKOFF = 1.0  # seconds, doubled per retry
MAX_BACKOFF = 60.0
VALID_ATTEMPTS = 3  # calls per json() while the result fails the schema
JUDGE_ATTEMPTS = 5
RETRY_STATUS = frozenset({408, 409, 429})
# ponytail: fixed overdraw and round count; a kind whose discard rate beats 1/OVERDRAW over
# MAX_ROUNDS ends short of its target. Size rounds from the observed keep rate if it bites.
OVERDRAW = 1.5
MAX_ROUNDS = 3
STYLES = ("email", "chat", "form", "log", "note")
NOT_STATED = "not_stated"
DOMAINS = (
    "logistics and parcel delivery",
    "hospital emergency triage",
    "insurance claims",
    "HR and hiring",
    "cloud infrastructure incidents",
    "payment fraud review",
    "retail returns",
    "loan underwriting",
    "airline disruption handling",
    "manufacturing quality control",
    "real estate rental applications",
    "IT access requests",
    "clinical trial screening",
    "content moderation",
    "energy grid maintenance",
    "university admissions",
)
THEMES = (
    "an animal",
    "a vehicle",
    "a building",
    "a landscape",
    "food on a table",
    "a household object",
    "a tool",
    "weather",
    "a plant",
    "a musical instrument",
    "a toy",
    "a piece of furniture",
    "a sports scene",
    "a night sky",
    "a boat on water",
    "a simple machine",
)
_INT = re.compile(r"-?\d+")


class GatewayError(Exception):
    """A gateway call failed; `retry` tells whether trying again later may help."""

    def __init__(self, message: str, *, retry: bool) -> None:
        super().__init__(message)
        self.retry = retry


class InvalidReply(GatewayError):
    """The gateway answered, but no reply validated: a content failure, not an outage."""


def _models(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _targets(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or set(raw) - set(BANK_KINDS):
        raise ValueError(f"bank targets must be an object over {list(BANK_KINDS)}")
    out = dict(DEFAULT_TARGETS)
    for kind in sorted(raw):
        if type(raw[kind]) is not int or raw[kind] < 0:
            raise ValueError(f"bank target {kind} must be a non-negative integer")
        out[kind] = raw[kind]
    return out


@dataclass(frozen=True)
class TeacherConfig:
    url: str
    token_file: Path = DEFAULT_TOKEN_FILE
    teacher: str = "cx/gpt-6-sol"
    extractors: tuple[str, ...] = ("cx/gpt-5.6-luna", "cc/claude-sonnet-5")
    judges: tuple[str, ...] = ("cx/gpt-6-sol",)
    concurrency: int = 8
    targets: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_TARGETS))

    @classmethod
    def from_env(cls) -> TeacherConfig | None:
        """None when OPENTYPE_TEACHER_URL is unset; ValueError on a malformed setting."""
        url = os.environ.get("OPENTYPE_TEACHER_URL", "").strip().rstrip("/")
        if not url:
            return None
        env = os.environ.get
        config = cls(
            url=url,
            token_file=Path(env("OPENTYPE_TEACHER_TOKEN_FILE", str(DEFAULT_TOKEN_FILE))),
            teacher=env("OPENTYPE_TEACHER_MODEL", "").strip() or cls.teacher,
            extractors=_models(env("OPENTYPE_EXTRACTOR_MODELS", "")) or cls.extractors,
            judges=_models(env("OPENTYPE_JUDGE_MODELS", "")) or cls.judges,
            concurrency=int(env("OPENTYPE_TEACHER_CONCURRENCY", "8")),
            targets=_targets(json.loads(env("OPENTYPE_BANK_TARGETS", "{}"))),
        )
        if config.concurrency < 1:
            raise ValueError("OPENTYPE_TEACHER_CONCURRENCY must be at least 1")
        return config


# ---------------------------------------------------------------------------
# Schema subset validator (§3): type, properties, required, additionalProperties: false,
# enum, items, minItems, maxItems, minimum, maximum. Messages name paths, never values.

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: type(v) is int,
    "number": lambda v: type(v) in (int, float),
    "boolean": lambda v: type(v) is bool,
    "null": lambda v: v is None,
}


def validate(schema: Mapping[str, Any], value: Any, path: str = "$") -> None:
    """ValueError naming the first path where `value` breaks `schema`."""
    kind = schema.get("type")
    if kind is not None and not _TYPES[kind](value):
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and not any(
        type(value) is type(option) and value == option for option in schema["enum"]
    ):
        raise ValueError(f"{path}: not an allowed value")
    if type(value) in (int, float) and not (
        schema.get("minimum", value) <= value <= schema.get("maximum", value)
    ):
        raise ValueError(f"{path}: out of range")
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise ValueError(f"{path}: wrong number of items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(schema["items"], item, f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", ()):
            if name not in value:
                raise ValueError(f"{path}.{name}: missing")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: unexpected property")
        for name in sorted(value):
            if name in properties:
                validate(properties[name], value[name], f"{path}.{name}")


# ---------------------------------------------------------------------------
# Response parsing: one JSON body (maybe followed by "data: [DONE]") or an SSE stream of
# chat.completion.chunk events, whatever `stream` said.


def _completion(raw: str) -> tuple[str, list[str]]:
    """(content, tool call arguments in order) of a chat completion response."""
    stripped = raw.lstrip()
    if not stripped.startswith(("data:", "event:", ":")):
        body, _ = json.JSONDecoder().raw_decode(stripped)
        message = body["choices"][0]["message"]
        content = message.get("content")
        calls = message.get("tool_calls") or []
        args = [str((call.get("function") or {}).get("arguments") or "") for call in calls]
        return content if isinstance(content, str) else "", args
    parts: list[str] = []
    tools: dict[int, list[str]] = {}
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            for position, call in enumerate(delta.get("tool_calls") or []):
                index = call.get("index", position)
                arguments = (call.get("function") or {}).get("arguments") or ""
                tools.setdefault(index, []).append(str(arguments))
    return "".join(parts), ["".join(tools[i]) for i in sorted(tools)]


def _first_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the reply")


def _result(raw: str) -> dict[str, Any]:
    content, tool_args = _completion(raw)
    for args in tool_args[:1]:
        try:
            value = json.loads(args)
        except ValueError:
            break
        if isinstance(value, dict):
            return value
    return _first_object(content)


class Gateway:
    """OpenAI-compatible chat completions client returning schema-valid JSON objects."""

    def __init__(
        self,
        config: TeacherConfig,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._client = httpx.AsyncClient(transport=transport, timeout=TIMEOUT)

    def _headers(self) -> dict[str, str]:
        try:
            token = self.config.token_file.read_text().strip()
        except OSError:
            raise GatewayError("teacher token file is unreadable", retry=False) from None
        if not token:
            raise GatewayError("teacher token file is empty", retry=False)
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": f"opentype-challenge/{__version__}",
        }

    async def _post(self, body: dict[str, Any]) -> str:
        url = f"{self.config.url.rstrip('/')}/v1/chat/completions"
        last = ""
        for attempt in range(RETRIES + 1):
            if attempt:
                await self._sleep(min(BACKOFF * 2 ** (attempt - 1), MAX_BACKOFF))
            try:
                response = await self._client.post(url, json=body, headers=self._headers())
            except httpx.TransportError as exc:
                last = type(exc).__name__
                continue
            status = response.status_code
            if 200 <= status < 300:
                return response.text
            if status in RETRY_STATUS or status >= 500:
                last = f"HTTP {status}"
                continue
            raise GatewayError(f"{body['model']}: HTTP {status}", retry=False)
        raise GatewayError(f"{body['model']}: {last} after {RETRIES + 1} attempts", retry=True)

    async def json(
        self,
        model: str,
        system: str,
        user: str | list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model.startswith("cx/"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "submit", "strict": True, "schema": schema},
            }
            # the gateway ignores response_format for cx/* (seen 2026-09): restate the schema
            system += (
                "\nReply with exactly one JSON object that validates against this JSON "
                "schema, and nothing else:\n" + json.dumps(schema, sort_keys=True)
            )
        else:
            body["tools"] = [
                {"type": "function", "function": {"name": "submit", "parameters": schema}}
            ]
            body["tool_choice"] = "auto"
            system += "\nCall the submit tool exactly once."
        body["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        reason = ""
        for _ in range(VALID_ATTEMPTS):
            raw = await self._post(body)
            try:
                value = _result(raw)
                validate(schema, value)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else "malformed response"
                continue
            return value
        raise InvalidReply(f"{model}: no valid result ({reason})", retry=True)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Round trip and judge.

EXTRACT_SYSTEM = {
    "prose": (
        "You read one record and fill in a form about it. For every field, return the value "
        'the record states, as one of the allowed values exactly, or "not_stated" when the '
        "record does not state it. Never guess or infer a value the record does not state."
    ),
    "ops_story": (
        "You read one customer message sent to an online store and fill in the request form. "
        'Copy the email address and order number exactly as written ("" when no order number '
        "is given). items: the product names the customer names, exactly as in the allowed "
        'list, sorted alphabetically. variant: the variant wanted in an exchange, "" '
        "otherwise. Reason codes: "
        + "; ".join(f"{code} = {text}" for code, text in sorted(ops.REASON_TEXT.items()))
        + ". Claim codes (what the customer asserts or how they press): "
        + "; ".join(f"{code} = {text}" for code, text in sorted(ops.CLAIM_FACT.items()))
        + "."
    ),
}


async def round_trip(
    gateway: Gateway,
    config: TeacherConfig,
    kind: str,
    text: str,
    expected: Mapping[str, Any],
    schema: dict[str, Any],
) -> bool:
    """True only when every extractor returns exactly `expected` from the text alone."""
    system = EXTRACT_SYSTEM[kind]
    user = f"Text:\n<<<\n{text}\n>>>"
    results = await asyncio.gather(
        *(gateway.json(model, system, user, schema) for model in config.extractors)
    )
    want = json.loads(json.dumps(expected, sort_keys=True))
    return all(result == want for result in results)


async def _verdicts(
    gateway: Gateway, config: TeacherConfig, brief: str, rubric: Sequence[str], png: bytes
) -> list[dict[str, Any]] | None:
    system, user, schema = paint.judge_request(brief, rubric, png)
    items = len(rubric) + 1

    async def one(model: str) -> dict[str, Any] | None:
        for _ in range(JUDGE_ATTEMPTS):
            try:
                verdict = await gateway.json(model, system, user, schema)
                paint.judge_loss([verdict], items)  # ids exactly 1..items
            except (InvalidReply, ValueError):  # an outage (other GatewayError) propagates
                continue
            return verdict
        return None

    results = await asyncio.gather(*(one(model) for model in config.judges))
    return None if any(r is None for r in results) else [r for r in results if r is not None]


async def judge_png(
    gateway: Gateway, config: TeacherConfig, brief: str, rubric: Sequence[str], png: bytes
) -> float | None:
    """§6 judge loss averaged over config.judges; None when any judge's replies cannot be
    read. A gateway outage raises GatewayError: the side stays pending."""
    verdicts = await _verdicts(gateway, config, brief, rubric, png)
    return None if verdicts is None else paint.judge_loss(verdicts, len(rubric) + 1)


# ---------------------------------------------------------------------------
# Teacher prompts and schemas.


def _strings(low: int, high: int) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "minItems": low, "maxItems": high}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


TEXT_SCHEMA = _object({"text": {"type": "string"}})
FAMILY_SCHEMA = _object(
    {
        "title": {"type": "string"},
        "subject": {"type": "string"},
        "facts": {
            "type": "array",
            "minItems": 6,
            "maxItems": 12,
            "items": _object(
                {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "domain": _strings(2, 8),
                    "unit": {"type": "string"},
                }
            ),
        },
        "derived": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": _object({"name": {"type": "string"}, "domain": _strings(2, 4)}),
        },
        "questions": {
            "type": "array",
            "minItems": 6,
            "maxItems": 12,
            "items": _object(
                {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": list(g.KINDS)},
                    "prompt": {"type": "string"},
                    "options": _strings(2, 26),
                }
            ),
        },
    }
)
DEPICT_SCHEMA = _object(
    {"subject": {"type": "string"}, "brief": {"type": "string"}, "rubric": _strings(4, 8)}
)

FAMILY_SYSTEM = """Task: family.
You design a record type for a decision benchmark: a realistic kind of business record, the \
facts it carries, a few derived facts and the decisions a reviewer makes about it.
Rules:
- title and subject: lower case words (for example "parcel exception" and "shipment").
- 6 to 12 facts. name: snake_case. label: lower case words, unique, not containing another \
label. domain: 2 to 8 distinct values, either all snake_case words or all integers written \
as digit strings in ascending order (for example ["0", "5", "10"]). unit: "" or a plain \
unit word such as "USD", "days" or "percent" (integers only).
- 1 to 3 derived facts: snake_case name and 2 to 4 snake_case values (e.g. low, medium, high).
- 6 to 12 questions with every kind present: "choice" (4 to 26 snake_case options), "noul" \
(options exactly ["yes", "no"]) and "score" (3 to 6 snake_case levels, lowest first). \
prompt: one line ending with "?".
- Names of facts, derived facts and questions are all distinct.
- Make it realistic for the domain, with values a real record would carry."""

PROSE_SYSTEM = """Task: prose.
You write one realistic business record in the requested style. It must state every listed \
fact so that a careful reader recovers its exact value without doubt: numbers exactly as \
given with their unit, category values in plain words that clearly mean that value. Never \
mention the facts you are told to leave out, not even as unknown, pending or blank. Do not \
state any other value of the listed kind. Names, dates and small talk that are not listed \
facts are welcome. No headings about the task, no commentary: only the record."""

OPS_SYSTEM = """Task: ops_story.
You write the message one customer sends to an online store's support, in the customer's \
own voice. You may add tone and irrelevant personal detail. It must: contain the email \
address exactly; contain the order number exactly when one is given and none otherwise; \
name each product exactly as written; state the reason clearly; for an exchange, state the \
wanted variant exactly; make the stated claim or pressure, and no other claim or pressure. \
Mention no other product. Only the message."""

DEPICT_SYSTEM = """Task: depict.
You propose one picture for a painter that draws flat shapes (rectangles, circles, ellipses, \
lines, polygons) in solid colours on a 256x256 canvas. subject: a short noun phrase. brief: \
one to three sentences describing the picture, with no text or lettering in it. rubric: 4 to \
8 concrete statements a viewer can check on the picture (shapes, colours, positions, counts, \
relations). A blank white canvas must fail every rubric item, and no item may be about text."""


def _family_user(domain: str, name: str) -> str:
    example = g.family_to_json(g.FAMILIES[1])
    del example["name"]
    for fact in example["facts"]:
        fact["domain"] = [str(v) for v in fact["domain"]]  # the schema carries strings
    return (
        f"Domain: {domain}.\nThe family will be named {name}.\n"
        f"Example of the shape (another domain):\n{json.dumps(example, sort_keys=True)}"
    )


def _family_payload(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(raw))
    payload["name"] = name
    for fact in payload["facts"]:
        domain = fact["domain"]
        if all(_INT.fullmatch(v) for v in domain):
            fact["domain"] = [int(v) for v in domain]
    return g.family_to_json(g.family_from_json(payload))  # GeneratorError when invalid


def _prose_user(family: g.Family, known: Mapping[str, g.Value], hidden: Sequence[str]) -> str:
    lines = [f"Record type: {family.title} (about one {family.subject})"]
    lines.append("State every one of these facts:")
    lines += [f"- {family.fact(n).label}: {family.fact(n).show(v)}" for n, v in known.items()]
    if hidden:
        lines.append("Never mention these facts, not even as unknown:")
        lines += [f"- {family.fact(name).label}" for name in hidden]
    return "\n".join(lines)


def prose_schema(family: g.Family) -> dict[str, Any]:
    """The extractors' form: every fact as a domain value (a string) or not_stated."""
    return _object(
        {
            fact.name: {
                "type": "string",
                "enum": [str(v) for v in fact.domain] + [NOT_STATED],
                "description": fact.label + (f", in {fact.unit}" if fact.unit else ""),
            }
            for fact in family.facts
        }
    )


# ---------------------------------------------------------------------------
# Builder.


class _Discard(Exception):
    """An item that failed a check; the message is a fixed reason, never teacher text."""


Job = Callable[[], Awaitable[dict[str, Any]]]


async def _kind(
    kind: str, target: int, plan: Callable[[], Job], limit: asyncio.Semaphore
) -> list[BankItem]:
    kept: dict[str, BankItem] = {}
    reasons: Counter[str] = Counter()

    async def guarded(job: Job) -> dict[str, Any] | str:
        async with limit:
            try:
                return await job()
            except _Discard as exc:
                return str(exc)
            except GatewayError:
                return "gateway error"
            except g.GeneratorError:
                return "invalid payload"

    for _ in range(MAX_ROUNDS):
        missing = target - len(kept)
        if missing <= 0:
            break
        jobs = [plan() for _ in range(math.ceil(missing * OVERDRAW))]  # rng draws in order
        for result in await asyncio.gather(*(guarded(job) for job in jobs)):
            if isinstance(result, str):
                reasons[result] += 1
                continue
            item = BankItem.make(kind, result)
            if item.key in kept:
                reasons["duplicate"] += 1
            else:
                kept[item.key] = item
    items = list(kept.values())[:target]
    log.info(
        "bank %s: kept %d of target %d, discarded %s",
        kind,
        len(items),
        target,
        dict(sorted(reasons.items())),
    )
    return items


async def build_bank(
    gateway: Gateway,
    config: TeacherConfig,
    rng: random.Random,
    targets: Mapping[str, int],
    *,
    families: Sequence[g.Family],
) -> list[BankItem]:
    """The window's bank: sealed families first, then prose (public and sealed families),
    ops stories and depict briefs. A failed item is skipped; GatewayError only when a kind
    with a non-zero target yields nothing."""
    limit = asyncio.Semaphore(config.concurrency)
    want = {kind: int(targets.get(kind, 0)) for kind in BANK_KINDS}
    # one sub-stream per kind, drawn up front: kinds run concurrently
    streams = {kind: random.Random(rng.getrandbits(64)) for kind in BANK_KINDS}

    def family_job() -> Job:
        r = streams["family"]
        name = f"sealed_{r.getrandbits(32):08x}"
        domain = r.choice(DOMAINS)

        async def job() -> dict[str, Any]:
            raw = await gateway.json(
                config.teacher,
                FAMILY_SYSTEM,
                _family_user(domain, name),
                FAMILY_SCHEMA,
                max_tokens=8000,
                temperature=1.0,
            )
            return _family_payload(raw, name)

        return job

    sealed = await _kind("family", want["family"], family_job, limit)
    pool = [*families, *(g.family_from_json(item.payload) for item in sealed)]

    def prose_job() -> Job:
        r = streams["prose"]
        family = r.choice(pool)
        known, hidden = g.sample_known(r, family, g.MAX_HIDDEN)
        style = r.choice(STYLES)

        async def job() -> dict[str, Any]:
            user = f"Style: {style}\n{_prose_user(family, known, hidden)}"
            written = await gateway.json(
                config.teacher, PROSE_SYSTEM, user, TEXT_SCHEMA, temperature=1.0
            )
            text = written["text"].strip()
            expected = {
                f.name: str(known[f.name]) if f.name in known else NOT_STATED for f in family.facts
            }
            if not text or not await round_trip(
                gateway, config, "prose", text, expected, prose_schema(family)
            ):
                raise _Discard("round trip disagreement")
            return {
                "family": family.name,
                "known": dict(known),
                "hidden": list(hidden),
                "text": text,
                "style": style,
            }

        return job

    def ops_job() -> Job:
        r = streams["ops_story"]
        intent = ops.sample_intent(r, r.randint(1, max(ops.LEVELS)))

        async def job() -> dict[str, Any]:
            user = f"Facts about the request:\n{ops.describe_intent(intent)}"
            written = await gateway.json(
                config.teacher, OPS_SYSTEM, user, TEXT_SCHEMA, temperature=1.0
            )
            text = written["text"].strip()
            if not text or not await round_trip(
                gateway, config, "ops_story", text, intent, ops.INTENT_SCHEMA
            ):
                raise _Discard("round trip disagreement")
            return {"intent": intent, "text": text}

        return job

    def depict_job() -> Job:
        theme = streams["depict"].choice(THEMES)

        async def job() -> dict[str, Any]:
            raw = await gateway.json(
                config.teacher,
                DEPICT_SYSTEM,
                f"Theme: {theme}.",
                DEPICT_SCHEMA,
                temperature=1.0,
            )
            payload = {
                "subject": raw["subject"].strip(),
                "brief": raw["brief"].strip(),
                "rubric": [item.strip() for item in raw["rubric"]],
            }
            if not payload["subject"] or not payload["brief"] or not all(payload["rubric"]):
                raise _Discard("empty field")
            verdicts = await _verdicts(
                gateway, config, payload["brief"], payload["rubric"], paint.blank_png()
            )
            if verdicts is None:
                raise _Discard("judge unreadable")
            size = len(payload["rubric"])
            for verdict in verdicts:
                if any(e["pass"] for e in verdict["items"] if e["id"] <= size):
                    raise _Discard("blank canvas passes an item")
            return payload

        return job

    rest = await asyncio.gather(
        _kind("prose", want["prose"], prose_job, limit),
        _kind("ops_story", want["ops_story"], ops_job, limit),
        _kind("depict", want["depict"], depict_job, limit),
    )
    built = dict(zip(BANK_KINDS, (sealed, *rest), strict=True))
    for kind in BANK_KINDS:
        if want[kind] > 0 and not built[kind]:
            raise GatewayError(f"bank: no {kind} item could be built", retry=True)
    return [item for kind in BANK_KINDS for item in built[kind]]
