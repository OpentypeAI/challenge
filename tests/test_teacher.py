"""Teacher gateway client and bank builder, against a fake gateway (docs/tracks.md §2, §3)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from opentype_challenge import generator as g
from opentype_challenge import ops, teacher
from opentype_challenge.teacher import Gateway, GatewayError, TeacherConfig

TOKEN = "sk-secret-token-4242"
SCHEMA = teacher._object({"answer": {"type": "string", "enum": ["a", "b"]}})


async def no_sleep(_: float) -> None:
    return None


def config(tmp_path: Path, **kw: Any) -> TeacherConfig:
    token = tmp_path / "teacher.token"
    token.write_text(TOKEN + "\n")
    return TeacherConfig(url="https://gw.test", token_file=token, **kw)


def gateway(cfg: TeacherConfig, handler) -> Gateway:
    return Gateway(cfg, httpx.MockTransport(handler), sleep=no_sleep)


def body_json(value: dict[str, Any]) -> httpx.Response:
    reply = {"choices": [{"message": {"role": "assistant", "content": json.dumps(value)}}]}
    return httpx.Response(200, text=json.dumps(reply) + "\n\ndata: [DONE]\n\n")


def sse(deltas: list[dict[str, Any]]) -> httpx.Response:
    events = [
        {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": d}]} for d in deltas
    ]
    text = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


def tool_sse(value: dict[str, Any]) -> httpx.Response:
    args = json.dumps(value)
    half = len(args) // 2
    call = {"index": 0, "id": "t1", "type": "function", "function": {"name": "submit"}}
    return sse(
        [
            {"role": "assistant"},
            {"tool_calls": [{**call, "function": {"name": "submit", "arguments": ""}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": args[:half]}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]},
        ]
    )


def run(coro):
    return asyncio.run(coro)


async def ask(gw: Gateway, model: str = "cx/m") -> dict[str, Any]:
    try:
        return await gw.json(model, "sys", "user", SCHEMA)
    finally:
        await gw.aclose()


def test_plain_json_body_and_request_shape(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return body_json({"answer": "b"})

    assert run(ask(gateway(config(tmp_path), handler))) == {"answer": "b"}
    request = seen[0]
    assert str(request.url) == "https://gw.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert re.fullmatch(r"opentype-challenge/\S+", request.headers["user-agent"])
    sent = json.loads(request.content)
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "tools" not in sent


def test_sse_content_deltas(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return sse([{"role": "assistant"}, {"content": '{"ans'}, {"content": 'wer": "a"}'}])

    assert run(ask(gateway(config(tmp_path), handler))) == {"answer": "a"}


def test_sse_tool_call_deltas_for_non_cx_models(tmp_path):
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return tool_sse({"answer": "b"})

    assert run(ask(gateway(config(tmp_path), handler), "cc/claude")) == {"answer": "b"}
    sent = seen[0]
    assert sent["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in sent["tools"]] == ["submit"]
    assert sent["messages"][0]["content"].endswith("Call the submit tool exactly once.")
    assert "response_format" not in sent


def test_retry_on_503_then_success(tmp_path):
    calls: list[int] = []
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else body_json({"answer": "a"})

    gw = Gateway(config(tmp_path), httpx.MockTransport(handler), sleep=sleep)
    assert run(ask(gw)) == {"answer": "a"}
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_transport_errors_exhaust_retries(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("boom")

    with pytest.raises(GatewayError) as info:
        run(ask(gateway(config(tmp_path), handler)))
    assert info.value.retry is True and len(calls) == teacher.RETRIES + 1


def test_400_is_not_retried(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(GatewayError) as info:
        run(ask(gateway(config(tmp_path), handler)))
    assert info.value.retry is False and len(calls) == 1


def test_invalid_result_is_retried_up_to_three_times(tmp_path):
    replies: list[dict[str, Any]] = [{"answer": "c"}, {"answer": "a", "extra": 1}, {"answer": "b"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return body_json(replies.pop(0))

    assert run(ask(gateway(config(tmp_path), handler))) == {"answer": "b"}

    def always_bad(request: httpx.Request) -> httpx.Response:
        return body_json({"answer": 3})

    with pytest.raises(GatewayError):
        run(ask(gateway(config(tmp_path), always_bad)))


def test_validator_subset():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 3},
            "xs": {"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2},
        },
        "required": ["n", "xs"],
        "additionalProperties": False,
    }
    teacher.validate(schema, {"n": 2, "xs": [True]})
    for bad in (
        {"n": 0, "xs": [True]},
        {"n": 2.0, "xs": [True]},
        {"n": True, "xs": [True]},
        {"n": 2, "xs": []},
        {"n": 2, "xs": [1]},
        {"n": 2},
        {"n": 2, "xs": [True], "z": 1},
    ):
        with pytest.raises(ValueError):
            teacher.validate(schema, bad)


def test_token_never_logged_or_raised(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    errors: list[BaseException] = []
    handlers = [
        lambda r: httpx.Response(401, text=r.headers["authorization"]),
        lambda r: httpx.Response(500, text=r.headers["authorization"]),
        lambda r: body_json({"answer": r.headers["authorization"]}),
    ]
    for handler in handlers:
        try:
            run(ask(gateway(config(tmp_path), handler)))
        except GatewayError as exc:
            errors.append(exc)
    missing = TeacherConfig(url="https://gw.test", token_file=tmp_path / "absent")
    try:
        run(ask(gateway(missing, handlers[0])))
    except GatewayError as exc:
        errors.append(exc)
    assert len(errors) == 4
    for error in errors:
        assert TOKEN not in repr(error) and TOKEN not in str(error.__cause__)
    assert all(TOKEN not in r.getMessage() for r in caplog.records)


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENTYPE_TEACHER_URL", raising=False)
    assert TeacherConfig.from_env() is None
    monkeypatch.setenv("OPENTYPE_TEACHER_URL", "https://gw.test/")
    monkeypatch.setenv("OPENTYPE_EXTRACTOR_MODELS", "a/x, b/y")
    monkeypatch.setenv("OPENTYPE_BANK_TARGETS", '{"prose": 3}')
    cfg = TeacherConfig.from_env()
    assert cfg is not None and cfg.url == "https://gw.test"
    assert cfg.extractors == ("a/x", "b/y") and cfg.judges == ("cx/gpt-6-sol",)
    assert cfg.targets == {"family": 4, "prose": 3, "ops_story": 200, "depict": 80}


# ---------------------------------------------------------------------------
# build_bank end to end against a fake teacher, extractors and judge.

SAMPLE_INTENT = ops.sample_intent  # unpatched
SEALED_SOURCE = g.family_to_json(g.FAMILIES[0])


def fake_family(user: str) -> dict[str, Any]:
    raw = json.loads(json.dumps(SEALED_SOURCE))
    del raw["name"]
    raw["title"] = "courier exception"
    raw["subject"] = "parcel"
    for fact in raw["facts"]:
        fact["domain"] = [str(v) for v in fact["domain"]]
        fact["label"] = fact["label"].lower()  # sealed labels are lower case words
    return raw


def prose_text(user: str) -> str:
    return user  # the fake teacher echoes the brief: it carries the facts


def parse_prose(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    stated = {}
    for line in text.splitlines():
        if line.startswith("- ") and ": " in line:
            label, value = line[2:].split(": ", 1)
            stated[label] = value
    out: dict[str, str] = {}
    for name, prop in schema["properties"].items():
        label, _, unit = prop["description"].partition(", in ")
        shown = {v.replace("_", " ") + (f" {unit}" if unit else ""): v for v in prop["enum"]}
        out[name] = shown.get(stated.get(label, ""), "not_stated")
    return out


class Fake:
    """Routes on model and system prompt; `lying` extractors corrupt some answers."""

    def __init__(self, *, lie_every: int = 0, lax_judge: bool = False) -> None:
        self.lie_every = lie_every
        self.lax_judge = lax_judge
        self.extractions = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        system = sent["messages"][0]["content"]
        user = sent["messages"][1]["content"]
        model = sent["model"]
        if model == "t/teacher":
            if system.startswith("Task: family."):
                return body_json(fake_family(user))
            if system.startswith("Task: prose."):
                return body_json({"text": prose_text(user)})
            if system.startswith("Task: ops_story."):
                return body_json({"text": user})
            if system.startswith("Task: depict."):
                theme = user
                rubric = (
                    ["The canvas is white."]
                    if self.lax_judge and theme.startswith("Theme: a ")
                    else []
                )
                rubric += ["A red circle is drawn.", "A blue square is drawn.", "A line.", "A dot."]
                return body_json({"subject": theme, "brief": f"Draw {theme}", "rubric": rubric})
        if model == "j/judge":
            n = sent["tools"][0]["function"]["parameters"]["properties"]["items"]["minItems"]
            text = user[0]["text"]
            items = [
                {"id": i, "pass": i == n or (f"{i}. The canvas is white." in text)}
                for i in range(1, n + 1)
            ]
            return tool_sse({"items": items})
        # extractors
        schema = (
            sent["tools"][0]["function"]["parameters"]
            if "tools" in sent
            else (sent["response_format"]["json_schema"]["schema"])
        )
        text = user.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        if "email" in schema["properties"]:
            intent = self.intent(text)
            result = intent
        else:
            result = parse_prose(text, schema)
        self.extractions += 1
        if self.lie_every and model == "e/two" and self.extractions % self.lie_every == 0:
            result = dict(result)
            key = sorted(result)[0]
            result[key] = "not_stated" if result[key] != "not_stated" else result[key]
            if "email" in result:
                result["email"] = "wrong@example.com"
        return tool_sse(result) if "tools" in sent else body_json(result)

    @staticmethod
    def intent(text: str) -> dict[str, Any]:
        # the fake teacher echoes describe_intent; look it up in the small table
        for described, (seed, level) in sorted(INTENTS.items()):
            if described in text:
                return SAMPLE_INTENT(random.Random(seed), level)
        raise AssertionError("unknown intent")


def fake_config(tmp_path: Path) -> TeacherConfig:
    return config(
        tmp_path,
        teacher="t/teacher",
        extractors=("e/one", "e/two"),
        judges=("j/judge",),
        concurrency=4,
    )


INTENTS = {
    ops.describe_intent(ops.sample_intent(random.Random(seed), level)): (seed, level)
    for level in range(1, 5)
    for seed in range(50)
}


def build(tmp_path, fake: Fake, targets: dict[str, int], seed: int = 7):
    async def go():
        gw = Gateway(fake_config(tmp_path), httpx.MockTransport(fake.handler), sleep=no_sleep)
        try:
            return await teacher.build_bank(
                gw, fake_config(tmp_path), random.Random(seed), targets, families=g.FAMILIES[:1]
            )
        finally:
            await gw.aclose()

    return run(go())


def patch_ops_rng(monkeypatch):
    """Make ops_job draw intents from the small INTENTS table."""
    real = SAMPLE_INTENT
    table = sorted(INTENTS.values())

    def sample(rng: random.Random, level: int) -> dict[str, Any]:
        seed, lvl = table[rng.randrange(len(table))]
        return real(random.Random(seed), lvl)

    monkeypatch.setattr(teacher.ops, "sample_intent", sample)


def test_build_bank_keeps_agreeing_items(tmp_path, monkeypatch):
    patch_ops_rng(monkeypatch)
    targets = {"family": 1, "prose": 5, "ops_story": 3, "depict": 2}
    items = build(tmp_path, Fake(), targets)
    kinds = [item.kind for item in items]
    assert {k: kinds.count(k) for k in teacher.BANK_KINDS} == targets
    family = next(i.payload for i in items if i.kind == "family")
    assert re.fullmatch(r"sealed_[0-9a-f]{8}", family["name"])
    parsed = g.family_from_json(family)
    assert any(f.numeric for f in parsed.facts)  # digit strings came back as integers
    for item in items:
        if item.kind == "prose":
            fam = parsed if item.payload["family"] == family["name"] else g.FAMILIES[0]
            known, text = g._prose_known(fam, item.payload)
            assert text and item.payload["style"] in teacher.STYLES
        if item.kind == "ops_story":
            assert ops._valid(item.payload["intent"])
        if item.kind == "depict":
            assert 4 <= len(item.payload["rubric"]) <= 8
        json.dumps(item.payload, sort_keys=True)


def test_build_bank_discards_disagreeing_items(tmp_path, monkeypatch, caplog):
    patch_ops_rng(monkeypatch)
    caplog.set_level(logging.INFO, logger="opentype_challenge.teacher")
    fake = Fake(lie_every=2)
    items = build(tmp_path, fake, {"family": 0, "prose": 4, "ops_story": 2, "depict": 0})
    assert [i.kind for i in items].count("prose") <= 4
    assert "round trip disagreement" in caplog.text
    # every kept prose really agrees with an honest extraction
    for item in items:
        if item.kind == "prose":
            fam = g.FAMILIES[0]
            schema = teacher.prose_schema(fam)
            got = parse_prose(item.payload["text"], schema)
            want = {
                f.name: str(item.payload["known"][f.name])
                if f.name in item.payload["known"]
                else "not_stated"
                for f in fam.facts
            }
            assert got == want
    assert all(
        item.payload.get("text", "") not in caplog.text for item in items if "text" in item.payload
    )


def test_depict_negative_control_rejects_rubric_a_blank_canvas_passes(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="opentype_challenge.teacher")
    items = build(
        tmp_path, Fake(lax_judge=True), {"family": 0, "prose": 0, "ops_story": 0, "depict": 6}
    )
    assert items
    assert all("The canvas is white." not in i.payload["rubric"] for i in items)
    assert "blank canvas passes an item" in caplog.text


def test_build_bank_raises_when_a_kind_yields_nothing(tmp_path):
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    async def go():
        gw = Gateway(fake_config(tmp_path), httpx.MockTransport(broken), sleep=no_sleep)
        try:
            await teacher.build_bank(
                gw, fake_config(tmp_path), random.Random(1), {"depict": 1}, families=g.FAMILIES
            )
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError):
        run(go())


def test_build_bank_keys_are_deterministic(tmp_path, monkeypatch):
    patch_ops_rng(monkeypatch)
    targets = {"family": 1, "prose": 4, "ops_story": 2, "depict": 2}
    first = build(tmp_path, Fake(), targets, seed=11)
    second = build(tmp_path, Fake(), targets, seed=11)
    other = build(tmp_path, Fake(), targets, seed=12)
    assert [(i.kind, i.key) for i in first] == [(i.kind, i.key) for i in second]
    assert {i.key for i in first} != {i.key for i in other}


def test_judge_png_averages_and_gives_up(tmp_path):
    from opentype_challenge import paint

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        n = sent["response_format"]["json_schema"]["schema"]["properties"]["items"]["minItems"]
        passes = 1 if sent["model"] == "cx/a" else n
        return body_json({"items": [{"id": i, "pass": i <= passes} for i in range(1, n + 1)]})

    cfg = config(tmp_path, judges=("cx/a", "cx/b"))

    async def go(h):
        gw = gateway(cfg, h)
        try:
            return await teacher.judge_png(gw, cfg, "b", ["x", "y", "z"], paint.blank_png())
        finally:
            await gw.aclose()

    assert run(go(handler)) == pytest.approx(1 - (1 / 4 + 1) / 2)
    calls: list[int] = []

    def unreadable(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return body_json({"items": [{"id": 1, "pass": True}]})

    assert run(go(unreadable)) is None
    assert len(calls) == 2 * teacher.JUDGE_ATTEMPTS * teacher.VALID_ATTEMPTS


# ---------------------------------------------------------------------------
# Live smoke test against the operator gateway.


@pytest.mark.skipif(os.environ.get("OPENTYPE_LIVE_TEACHER") != "1", reason="live gateway")
def test_live_tiny_bank(caplog):
    env = Path("/root/.opentype-teacher.env").read_text()
    url = re.search(r"OPENTYPE_TEACHER_URL=(\S+)", env)
    assert url is not None
    cfg = TeacherConfig(
        url=url.group(1).strip("\"'"),
        token_file=Path("/root/.opentype-teacher.token"),
        teacher="cx/gpt-6-sol",
        extractors=("cx/gpt-5.6-luna", "cc/claude-sonnet-5"),
        judges=("cx/gpt-6-sol",),
    )
    targets = {"family": 1, "prose": 4, "ops_story": 3, "depict": 2}
    caplog.set_level(logging.INFO, logger="opentype_challenge.teacher")

    async def go():
        gw = Gateway(cfg)
        try:
            return await teacher.build_bank(
                gw, cfg, random.Random(2026), targets, families=g.FAMILIES
            )
        finally:
            await gw.aclose()

    start = time.monotonic()
    items = run(go())
    wall = time.monotonic() - start
    counts = {k: sum(i.kind == k for i in items) for k in teacher.BANK_KINDS}
    print(f"live bank: {counts} in {wall:.0f} s")
    for record in caplog.records:
        print(record.getMessage())
    assert all(counts[k] >= 1 for k in targets)
    assert Path(cfg.token_file).read_text().strip() not in caplog.text
