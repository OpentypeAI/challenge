"""The worker's case loop against a fake worker API and fake inference (no store, no processes):
paging, read and harness items, 4xx/5xx classification and answer batching (docs/tracks.md §7)."""

import asyncio
import json
import random
from typing import Any

import httpx
import pytest

from opentype_challenge import generator as g
from opentype_challenge import harness, longctx, ops, paint, sqltask, tracks
from opentype_challenge.bank import EMPTY_BANK, cases_digest
from opentype_challenge.worker import (
    BATCH_BYTES,
    BATCH_HARNESS,
    Api,
    JobFailed,
    VllmLauncher,
    Worker,
    _batches,
)

from . import fake_inference as fake

URLS = {
    side: {"reader": f"http://{side}-reader.test", "chat": f"http://{side}-chat.test"}
    for side in ("champion", "challenger")
}


def served_cases() -> list[dict[str, Any]]:
    """Two cases of each track, in a fixed order."""
    built = []
    for i in range(2):
        rng = random.Random(f"worker|{i}")
        built.append(g.make_case(rng, rng.choice(g.FAMILIES), 1 + i))
        built.append(longctx.make_case(random.Random(f"worker-lc|{i}"), 1, EMPTY_BANK))
        built.append(ops.make_case(random.Random(f"worker-ops|{i}"), 1 + i, EMPTY_BANK))
        built.append(sqltask.make_case(random.Random(f"worker-sql|{i}"), 1 + i, EMPTY_BANK))
        built.append(paint.make_case(random.Random(f"worker-paint|{i}"), 1 + i, EMPTY_BANK))
    return [{"index": i, "track": c.track, "body": c.body} for i, c in enumerate(built)]


class FakeApi:
    """The worker routes of the container: pages of at most `page` cases, answer batches."""

    def __init__(self, cases: list[dict[str, Any]], page: int = 3, stop_after: int | None = None):
        self.cases, self.page, self.stop_after = cases, page, stop_after
        self.offsets: list[int] = []
        self.items: list[dict[str, Any]] = []
        self.batch_bytes: list[int] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer worker"
        if request.url.path.endswith("/cases"):
            offset = int(request.url.params["offset"])
            limit = min(int(request.url.params["limit"]), self.page)
            self.offsets.append(offset)
            return httpx.Response(200, json={"cases": self.cases[offset : offset + limit]})
        assert request.url.path.endswith("/answers")
        self.batch_bytes.append(len(request.content))
        self.items += json.loads(request.content)["items"]
        done = self.stop_after is not None and len(self.items) >= self.stop_after
        return httpx.Response(200, json={"continue": not done})


class FakeInference:
    """Both sides' reader and chat servers in-process; `status` overrides replies per side."""

    def __init__(self, skills: dict[str, str], status: dict[str, int] | None = None):
        self.skills, self.status = skills, status or {}
        self.chats: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        side, kind = request.url.host.split(".")[0].split("-")
        body = json.loads(request.content)
        if side in self.status:
            return httpx.Response(self.status[side], json={"error": {"message": "nope"}})
        if kind == "chat":
            assert request.url.path == "/v1/chat/completions"
            self.chats.append(body)
            code, payload = fake.chat(self.skills[side], body)
        else:
            assert request.url.path == "/v1/systemone"
            code, payload = fake.systemone(self.skills[side], body)
        return httpx.Response(code, json=payload)


async def run_read(api: FakeApi, inference: FakeInference | httpx.MockTransport, cases: int):
    if isinstance(inference, httpx.MockTransport):
        transport = inference
    else:
        transport = httpx.MockTransport(inference.handler)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(api.handler)) as api_client,
        httpx.AsyncClient(transport=transport) as infer_client,
    ):
        worker = Worker(
            Api("http://api.test", "worker", api_client),
            workdir=None,  # type: ignore[arg-type]
            launcher=None,  # type: ignore[arg-type]
            inference=infer_client,
            concurrency=4,
        )
        job = {"job": "j1", "lease": "l1", "cases": cases}
        return await worker._read(job, URLS)


def test_pages_to_completion_and_replays_to_the_oracle():
    cases = served_cases()
    api = FakeApi(cases, page=3)
    inference = FakeInference({"champion": "base", "challenger": "exact"})
    counts = asyncio.run(run_read(api, inference, len(cases)))

    assert api.offsets == list(range(0, len(cases), 3))
    assert counts == {
        "cases_fetched": len(cases),
        "cases_sha256": cases_digest(c["body"] for c in cases),
        "errors": 0,
    }
    by_key = {(i["case_index"], i["side"]): i for i in api.items}
    assert set(by_key) == {(c["index"], s) for c in cases for s in ("champion", "challenger")}
    for case in cases:
        exact = by_key[(case["index"], "challenger")]
        if case["track"] in tracks.ENVS:
            assert set(exact) == {"case_index", "side", "transcript"}
            _, loss = harness.replay(tracks.ENVS[case["track"]], case["body"], exact["transcript"])
            assert loss == 0.0
        else:
            assert set(exact) == {"case_index", "side", "answers", "reads"}
            assert set(exact["answers"]) == set(case["body"]["questions"])
    # the chat requests are exactly §7's shape
    for request in inference.chats:
        assert set(request) == {"model", "messages", "max_tokens", "temperature", "seed"}
        assert request["temperature"] == 0.0 and request["model"] in URLS
        assert request["messages"][0]["content"].startswith("OpenType harness: ")


def test_seeds_follow_the_body_seed_per_turn():
    case = served_cases()[2]
    assert case["track"] == "ops"
    inference = FakeInference({"champion": "exact", "challenger": "exact"})
    asyncio.run(run_read(FakeApi([case]), inference, 1))
    seeds = sorted({r["seed"] for r in inference.chats})
    turns = len(seeds)
    assert seeds == [case["body"]["seed"] + t for t in range(turns)]
    assert all(r["max_tokens"] == case["body"]["limits"]["max_tokens"] for r in inference.chats)


def test_4xx_is_an_item_error_and_the_duel_goes_on():
    cases = served_cases()
    api = FakeApi(cases)
    inference = FakeInference({"champion": "exact", "challenger": "exact"}, {"champion": 400})
    counts = asyncio.run(run_read(api, inference, len(cases)))
    assert counts["errors"] == len(cases)
    for item in api.items:
        if item["side"] == "champion":
            assert set(item) == {"case_index", "side", "error"}
            assert item["error"].startswith("400: ")
        else:
            assert "error" not in item


@pytest.mark.parametrize("status", [500, 503])
def test_5xx_is_infrastructure(status):
    cases = served_cases()
    for track in ("decisions", "ops"):
        chosen = [c for c in cases if c["track"] == track][:1]
        inference = FakeInference({"challenger": "exact"}, {"champion": status})
        with pytest.raises(JobFailed) as error:
            asyncio.run(run_read(FakeApi(chosen), inference, 1))
        assert error.value.retry is True


def test_transport_error_is_infrastructure():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    chosen = [c for c in served_cases() if c["track"] == "paint"][:1]
    with pytest.raises(JobFailed) as error:
        asyncio.run(run_read(FakeApi(chosen), httpx.MockTransport(down), 1))
    assert error.value.retry is True


def test_broken_model_forfeits_by_its_transcript_not_an_error():
    cases = [c for c in served_cases() if c["track"] in tracks.ENVS]
    api = FakeApi(cases)
    inference = FakeInference({"champion": "broken", "challenger": "exact"})
    asyncio.run(run_read(api, inference, len(cases)))
    for item in api.items:
        if item["side"] == "champion":
            body = cases[[c["index"] for c in cases].index(item["case_index"])]["body"]
            assert item["transcript"] == [fake.INVALID_REPLY] * body["limits"]["turns"]


def test_continue_false_stops_paging():
    cases = served_cases()
    api = FakeApi(cases, page=2, stop_after=2)
    inference = FakeInference({"champion": "exact", "challenger": "exact"})
    counts = asyncio.run(run_read(api, inference, len(cases)))
    assert api.offsets == [0] and counts["cases_fetched"] == 2


def test_batches_stay_under_the_request_cap():
    item = {"case_index": 0, "side": "champion", "transcript": ["x" * 8192] * 12}
    items = [{**item, "case_index": i} for i in range(60)]
    batches = _batches(items)
    assert [i for b in batches for i in b] == items and len(batches) > 1
    for batch in batches:
        body = json.dumps({"lease": "l" * 64, "items": batch}, separators=(",", ":"))
        assert len(body) < BATCH_BYTES + 1024 < 1 << 20


def test_batches_cap_the_replays_of_one_request():
    reads = [{"case_index": i, "side": "champion", "answers": {}} for i in range(100)]
    plays = [{"case_index": i, "side": "champion", "transcript": ["x"]} for i in range(100, 120)]
    batches = _batches(reads + plays)
    assert [i for b in batches for i in b] == reads + plays
    assert all(sum("transcript" in i for i in b) <= BATCH_HARNESS for b in batches)
    assert len(batches) == 3


def test_vllm_command_and_urls():
    launcher = VllmLauncher(port_base=9000)
    serve, reader = launcher.commands("challenger", launcher.log_dir)
    assert serve[serve.index("--max-model-len") + 1] == "131072"
    assert json.loads(serve[serve.index("--limit-mm-per-prompt") + 1]) == {"image": 1}
    assert reader[reader.index("--port") + 1] == "9011"
    assert VllmLauncher(max_model_len=65536).commands("champion", launcher.log_dir)[0][-3] == (
        "65536"
    )


def test_until_empty_runs_jobs_until_the_queue_is_empty(monkeypatch, tmp_path):
    from opentype_challenge import cli, pins, worker

    runs = iter([True, True, False])
    calls: list[int] = []

    async def run_once(self):
        calls.append(1)
        result = next(runs)
        if isinstance(result, Exception):
            raise result
        return result

    async def no_sleep(seconds):
        pass

    token = tmp_path / "token"
    token.write_text("t")
    monkeypatch.setattr(worker.Worker, "run_once", run_once)
    monkeypatch.setattr(worker.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(worker, "sha256_file", lambda path: pins.STRUCTURED_SERVER_SHA256)
    cli.main(["worker", "--api", "http://x", "--token-file", str(token), "--workdir",
              str(tmp_path), "--until-empty"])  # fmt: skip
    assert len(calls) == 3  # the empty queue ends the run
    with pytest.raises(SystemExit):
        cli.main(["worker", "--api", "http://x", "--token-file", str(token), "--workdir",
                  str(tmp_path), "--once", "--until-empty"])  # fmt: skip


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (503, 5)])
def test_until_empty_api_failure_is_bounded(monkeypatch, tmp_path, status, expected_calls):
    from opentype_challenge import pins, worker

    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(status, text="unavailable")

    async def no_sleep(seconds):
        pass

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            instance = Worker(Api("http://x", "t", client), tmp_path, VllmLauncher())
            with pytest.raises(RuntimeError):
                await instance.run_forever(until_empty=True)

    monkeypatch.setattr(worker.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(worker, "sha256_file", lambda path: pins.STRUCTURED_SERVER_SHA256)
    asyncio.run(run())
    assert len(calls) == expected_calls


def test_polling_worker_retries_transient_outage(monkeypatch, tmp_path):
    from opentype_challenge import pins, worker

    runs = iter([worker.ApiUnavailable("master down"), RuntimeError("fatal")])
    sleeps = []

    async def run_once(self):
        raise next(runs)

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(worker.Worker, "run_once", run_once)
    monkeypatch.setattr(worker.asyncio, "sleep", sleep)
    monkeypatch.setattr(worker, "sha256_file", lambda path: pins.STRUCTURED_SERVER_SHA256)
    instance = Worker(None, tmp_path, VllmLauncher())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="fatal"):
        asyncio.run(instance.run_forever())
    assert sleeps == [30.0]
