"""Full duels: intake -> worker (real processes, fake inference) -> crown -> ledger -> weights.

The multi-track duel runs every track against a fake teacher bank (one sealed family, prose,
ops stories, depict briefs) and a fake judge; decisions-only duels keep v1's settings.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
import secrets
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from opentype_challenge import generator as g
from opentype_challenge import ops, paint, pins, tracks
from opentype_challenge.app import Config, create_app
from opentype_challenge.bank import Bank, BankItem, bank_digest, cases_digest, job_seed
from opentype_challenge.store import Settings
from opentype_challenge.tracks import TrackPlan
from opentype_challenge.worker import Api, JobFailed, VllmLauncher, Worker, assemble

from .conftest import ADMIN, INTERNAL, SLUG, WORKER, Miner, bearer, submit
from .test_generator import sealed_payload
from .test_ops import STORY_INTENT
from .test_paint import DEPICT

FAKE = Path(__file__).with_name("fake_inference.py")
BASE_SUPPORT = {name: f"base support {name}".encode() for name in pins.BASE_SUPPORT_FILES}
BASE_CONFIG = b"base config"


class Hub:
    """A fake Hugging Face hub: repo -> {filename: bytes}, with sha256 pins patched in."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.repos: dict[tuple[str, str], dict[str, bytes]] = {}
        support = {n: hashlib.sha256(b).hexdigest() for n, b in BASE_SUPPORT.items()}
        monkeypatch.setattr(pins, "BASE_SUPPORT_FILES", support)
        base = self.model("base")
        self.repos[(pins.BASE_REPO, pins.BASE_REVISION)] = {**base, **BASE_SUPPORT}
        monkeypatch.setattr(pins, "BASE_FILES", _digests(base))
        self.downloads: list[tuple[str, str]] = []

    @staticmethod
    def model(skill: str, salt: str = "") -> dict[str, bytes]:
        return {
            "config.json": BASE_CONFIG,
            "model.safetensors": f"{skill} {salt}".encode(),
        }

    def publish(self, repo: str, skill: str, salt: str = "") -> dict:
        revision = hashlib.sha1(f"{repo}{skill}{salt}".encode()).hexdigest()
        files = self.model(skill, salt)
        self.repos[(repo, revision)] = files
        return {"repo": repo, "revision": revision, "files": _digests(files)}

    def fetch(self, repo: str, revision: str, filename: str, directory: Path) -> Path:
        self.downloads.append((repo, filename))
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.repos[(repo, revision)][filename])
        return target


def _digests(files: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def hub(monkeypatch):
    return Hub(monkeypatch)


@pytest.fixture
def launcher(tmp_path):
    reader = tmp_path / "structured_server.py"
    reader.write_text(FAKE.read_text())
    base = _free_port()
    while base + 11 > 65535:
        base = _free_port()
    return VllmLauncher(
        canvas=64,
        port_base=base,
        log_dir=tmp_path,
        vllm=(sys.executable, str(FAKE)),
        reader=reader,
    )


PLAN = {
    "decisions": TrackPlan(0.35, 160),
    "longctx": TrackPlan(0.25, 16),
    "ops": TrackPlan(0.15, 32),
    "sql": TrackPlan(0.10, 32),
    "paint": TrackPlan(0.15, 40),
}
BEACON = {"round": 4242, "randomness": "ab" * 32}


def template_prose(rng: random.Random, family: g.Family) -> dict[str, Any]:
    """A prose payload whose text is template lines (no record header), so the fake reader's
    text-only solver stays exact on it, as the teacher's round trip guarantees for real prose."""
    known, hidden = g.sample_known(rng, family, g.LEVELS[2].hidden)
    for _ in range(g.MAX_ATTEMPTS):
        text = "\n".join(g.render_state(rng, family, known, g.LEVELS[2]).split("\n")[1:])
        if g.extract(family, text) == known and "#" not in text:
            break
    else:
        raise AssertionError("no template prose round-trips")
    return {"family": family.name, "known": known, "hidden": hidden, "text": text, "style": "log"}


def fake_bank() -> list[BankItem]:
    """One sealed family, prose for it and two public families, an ops story (in the template
    wording, so the oracle can play it), a depict brief."""
    sealed = g.family_from_json(sealed_payload("e2e0cafe"))
    story = {"intent": STORY_INTENT, "text": ops.render_intent(STORY_INTENT)}
    items = [
        BankItem.make("family", sealed_payload("e2e0cafe")),
        BankItem.make("ops_story", story),
        BankItem.make("depict", DEPICT),
    ]
    for i, family in enumerate([sealed, *g.FAMILIES[:2]] * 2):
        items.append(BankItem.make("prose", template_prose(random.Random(f"e2e|{i}"), family)))
    return items


class Teacher:
    """The injected bank builder and judge: records every call, never touches a gateway."""

    def __init__(self) -> None:
        self.items = fake_bank()
        self.builds = 0
        self.judged: list[tuple[str, list[str], bytes]] = []

    async def bank_builder(self, *_args: Any, **_kwargs: Any) -> list[BankItem]:
        self.builds += 1
        return list(self.items)

    async def judge(self, *args: Any, **kwargs: Any) -> float:
        """0 for any drawing, 1 for a blank canvas (the negative control fails everything).
        ponytail: finds (brief, rubric, png) by type because §8 fixes only judge=; pin the
        signature once store.py settles it."""
        values = [*args, *kwargs.values()]
        png = next(v for v in values if isinstance(v, bytes))
        brief = next(v for v in values if isinstance(v, str))
        rubric = next(list(v) for v in values if isinstance(v, list | tuple))
        self.judged.append((brief, rubric, png))
        with Image.open(io.BytesIO(png)) as image:
            blank = image.convert("RGB").getextrema() == ((255, 255), (255, 255), (255, 255))
        return 1.0 if blank else 0.0


@pytest.fixture
def teacher() -> Teacher:
    return Teacher()


@pytest.fixture
def duel_client(tmp_path, secrets_dir, clock, master, teacher, hub) -> Iterator[TestClient]:
    """The container with the multi-track plan, the fake teacher and a fixed drand beacon."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.drand.sh":
            return httpx.Response(200, json=BEACON)
        return master.handler(request)

    config = Config(
        slug=SLUG,
        state_dir=tmp_path / "duel-data",
        master_url="http://master.test",
        internal_token_file=secrets_dir / "internal.token",
        admin_token_file=secrets_dir / "admin.token",
        worker_token_file=secrets_dir / "worker.token",
        settings=Settings(plan=PLAN),
    )
    app = create_app(
        config,
        clock,
        httpx.MockTransport(handler),
        judge=teacher.judge,
        bank_builder=teacher.bank_builder,
    )
    with TestClient(app) as client:
        yield client


def _open_bank_window(client: TestClient, expected: str) -> dict[str, Any]:
    """Rotate until the open window carries the fake bank (the builder runs in background)."""
    for _ in range(100):
        windows = client.get("/v1/windows").json()["windows"]
        current = [w for w in windows if w["closed_at"] is None][-1]
        if current.get("bank_digest") == expected:
            return current
        client.post("/v1/admin/window/rotate", headers=bearer(ADMIN))
        time.sleep(0.05)
    raise AssertionError(f"no window opened with bank {expected}: {windows}")


def _published_bank(client: TestClient, window_id: int) -> Bank:
    rows: list[Any] = []
    while True:
        page = client.get(
            f"/v1/windows/{window_id}/bank", params={"offset": len(rows), "limit": 100}
        ).json()
        rows += page["items"]
        if not page["items"] or len(rows) >= page.get("total", len(rows)):
            return Bank.from_json(rows)


def _worker(client, tmp_path, hub, launcher) -> tuple[Worker, httpx.AsyncClient]:
    api_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app))
    worker = Worker(
        Api("http://challenge.test", WORKER, api_client),
        tmp_path / "work",
        launcher,
        fetch=hub.fetch,
        concurrency=8,
    )
    return worker, api_client


async def _run(client, tmp_path, hub, launcher) -> bool:
    worker, api_client = _worker(client, tmp_path, hub, launcher)
    try:
        return await worker.run_once()
    finally:
        await api_client.aclose()


def weights(client, epoch):
    return client.get(
        f"/internal/v1/get_weights?epoch={epoch}",
        headers={**bearer(INTERNAL), "x-platform-challenge-slug": SLUG},
    )


def test_duel_crowns_pays_and_audits(duel_client, teacher, miner, clock, hub, launcher, tmp_path):
    client = duel_client
    expected_digest = bank_digest(teacher.items)
    window = _open_bank_window(client, expected_digest)
    manifest = hub.publish("miner/exact", "exact")
    sid = submit(client, miner, manifest, clock).json()["id"]

    assert asyncio.run(_run(client, tmp_path, hub, launcher)) is True
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "crowned", result
    verdict = result["job"]["verdict"]
    assert verdict["crown"] and verdict["g_lcb"] >= verdict["g_min"]
    challenger = verdict["levels"]["1"]["challenger"]
    assert challenger["accuracy"] == 1.0  # exact gold: 100 % is reachable

    # Per-track metrics: every buildable track is present, and the container's replay of the
    # harness transcripts gives the oracle (exact) zero loss and the base champion some loss.
    bank = Bank(tuple(teacher.items))
    plan = tracks.effective_plan(PLAN, bank, judge=True)
    assert set(verdict["tracks"]) == set(plan)
    for track, metrics in verdict["tracks"].items():
        assert metrics["pairs"] == plan[track].cases, track
        assert metrics["accuracy"]["challenger"] == 1.0, track
        if track in tracks.ENVS:
            assert metrics["challenger_loss"] == 0.0, track
    assert sum(verdict["tracks"][t]["champion_loss"] for t in tracks.ENVS if t in plan) > 0
    assert verdict.get("unjudged", 0) == 0

    evidence = result["job"]["evidence"]
    assert (
        evidence["challenger_files"]["model.safetensors"] == manifest["files"]["model.safetensors"]
    )
    assert (
        evidence["challenger_files"]["tokenizer.json"] == pins.BASE_SUPPORT_FILES["tokenizer.json"]
    )
    assert evidence["structured_server_sha256"] == hashlib.sha256(FAKE.read_bytes()).hexdigest()
    assert evidence["max_model_len"] == 131072
    total = sum(p.cases for p in plan.values())
    assert evidence["cases_fetched"] == total and evidence["errors"] == 0
    assert not (tmp_path / "work" / result["job"]["id"]).exists()  # weights deleted

    status = client.get("/v1/status").json()
    assert status["champion"]["hotkey"] == miner.hotkey
    entitlement = client.get("/v1/leaderboard").json()["hotkeys"][miner.hotkey]["entitlement"]
    assert entitlement == pytest.approx(verdict["g_lcb"] / verdict["g_min"], rel=1e-6)

    paid = 0.0
    for epoch in range(100, 100 + int(entitlement) + 2):
        body = weights(client, epoch).json()
        paid += body["weights"].get(miner.hotkey, 0.0)
        assert sum(body["weights"].values()) + body["metadata"]["burned"] == pytest.approx(1)
    assert paid == pytest.approx(entitlement, abs=1e-6)
    assert weights(client, 100).json()["weights"] == {miner.hotkey: 1.0}

    # After rotation the secret and the bank are revealed and anyone regenerates the exact
    # served cases from the secret, the job, the drand beacon and the bank.
    client.post("/v1/admin/window/rotate", headers=bearer(ADMIN))
    revealed = client.get(f"/v1/windows/{window['id']}").json()
    assert revealed["bank_digest"] == expected_digest
    published = _published_bank(client, window["id"])
    assert published.digest == expected_digest
    job = revealed["jobs"][0]
    assert job["beacon"] == BEACON
    seed = job_seed(bytes.fromhex(revealed["secret"]), job["id"], job["digest"], job["beacon"])
    cases = [
        tracks.job_case(seed, PLAN, job["mix"], i, published, judge=True)
        for i in range(job["cases"])
    ]
    regenerated = cases_digest(case.body for case in cases)
    assert regenerated == job["cases_sha256"] == evidence["cases_sha256"]

    # Every depict case was judged once per side, on the container's render with the rubric.
    depicts = [c for c in cases if c.track == "paint" and paint.LEVELS[c.level] == "depict"]
    assert len(teacher.judged) == 2 * len(depicts)
    for brief, rubric, png in teacher.judged:
        assert brief == DEPICT["brief"] and DEPICT["rubric"][0] in rubric
        assert png.startswith(b"\x89PNG")


def test_worse_challenger_is_rejected_and_early_stopped(
    make_client, miner, master, clock, hub, launcher, tmp_path
):
    client = make_client(duel_cases=1200)
    other = Miner(secrets.token_bytes(32))
    master.hotkeys[other.hotkey] = 2
    champion = hub.publish("miner/champion", "exact")
    assert submit(client, miner, champion, clock).status_code == 201
    asyncio.run(_run(client, tmp_path, hub, launcher))
    assert client.get("/v1/status").json()["champion"]["hotkey"] == miner.hotkey

    sid = submit(client, other, hub.publish("miner/base", "base"), clock).json()["id"]
    asyncio.run(_run(client, tmp_path, hub, launcher))
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "rejected" and result["reason"] == "early stop"
    assert 5000 / 7 <= result["job"]["paired"] < 1200  # stopped after >= 5 000 decisions
    assert client.get("/v1/status").json()["champion"]["hotkey"] == miner.hotkey


def test_tampered_weights_are_rejected_without_retry(
    make_client, miner, clock, hub, launcher, tmp_path
):
    client = make_client(duel_cases=12)
    manifest = hub.publish("miner/tampered", "exact")
    hub.repos[(manifest["repo"], manifest["revision"])]["model.safetensors"] = b"swapped"
    sid = submit(client, miner, manifest, clock).json()["id"]
    asyncio.run(_run(client, tmp_path, hub, launcher))
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "rejected" and "sha256" in result["reason"]


def test_assemble_takes_support_files_from_the_base(hub, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name, data in BASE_SUPPORT.items():
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_bytes(data)
    manifest = hub.publish("miner/x", "exact")
    resolved = assemble(manifest, base, tmp_path / "model", hub.fetch)
    assert set(resolved) == set(manifest["files"]) | set(pins.BASE_SUPPORT_FILES)
    assert ("miner/x", "tokenizer.json") not in hub.downloads
    bad = {**manifest, "files": {**manifest["files"], "config.json": "0" * 64}}
    with pytest.raises(JobFailed) as error:
        assemble(bad, base, tmp_path / "bad", hub.fetch)
    assert error.value.retry is False


def test_infrastructure_failure_requeues_then_fails(
    make_client, miner, clock, hub, launcher, tmp_path
):
    client = make_client(duel_cases=12)
    sid = submit(client, miner, hub.publish("miner/y", "exact"), clock).json()["id"]

    def broken(*_args):
        raise OSError("hub unreachable")

    hub.fetch = broken
    for attempt in range(1, 4):
        asyncio.run(_run(client, tmp_path, hub, launcher))
        state = client.get(f"/v1/submissions/{sid}").json()
        expected = "queued" if attempt < 3 else "failed"
        assert state["state"] == expected, state
    assert asyncio.run(_run(client, tmp_path, hub, launcher)) is False  # queue empty


def test_stale_champion_requeues_and_earliest_intake_wins(
    make_client, master, clock, hub, launcher, tmp_path
):
    """Two winners against the same champion: the earlier intake crowns, the later one duels
    the new champion (and loses as a copy of equal skill)."""
    client = make_client(duel_cases=240)
    first, second = Miner(secrets.token_bytes(32)), Miner(secrets.token_bytes(32))
    master.hotkeys.update({first.hotkey: 1, second.hotkey: 2})
    a = submit(client, first, hub.publish("miner/a", "exact", "a"), clock).json()["id"]
    b = submit(client, second, hub.publish("miner/b", "exact", "b"), clock).json()["id"]

    store = client.app.state.store
    lease_a = store.lease()
    lease_b = store.lease()
    assert (lease_a["job"], lease_b["job"]) == (
        store.submission(a)["job"]["id"],
        store.submission(b)["job"]["id"],
    )
    # Run b's duel first: its crown must wait for a, which is earlier and still open.
    asyncio.run(_answer_perfectly(client, lease_b))
    assert store.submission(b)["state"] == "queued"
    assert store.submission(b)["job"]["state"] == "scored"
    asyncio.run(_answer_perfectly(client, lease_a))
    assert store.submission(a)["state"] == "crowned"
    # the duel ran on the empty bank (public templates only): the crown pays nothing
    board = client.get("/v1/leaderboard").json()["hotkeys"][first.hotkey]
    assert store.submission(a)["job"]["verdict"]["g_lcb"] > 0 and board["entitlement"] == 0
    job_b = store.submission(b)["job"]
    assert store.submission(b)["state"] == "queued" and job_b["id"] != lease_b["job"]
    assert job_b["champion"] == 2  # re-targeted at the new champion

    asyncio.run(_run(client, tmp_path, hub, launcher))
    final = store.submission(b)
    assert final["state"] == "rejected" and final["reason"] == "no certified gain"
    assert client.get("/v1/status").json()["champion"]["hotkey"] == first.hotkey


async def _answer_perfectly(client, lease) -> None:
    """Answer a leased job with the exact posterior for the challenger and a blurred
    champion, straight through the worker API."""
    from .fake_inference import answer, blur

    headers = bearer(WORKER)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app), base_url="http://t"
    ) as api:
        cases = (
            await api.get(
                f"/v1/worker/jobs/{lease['job']}/cases",
                params={"lease": lease["lease"], "offset": 0, "limit": 200},
                headers=headers,
            )
        ).json()["cases"]
        offset = len(cases)
        while offset < lease["cases"]:
            page = (
                await api.get(
                    f"/v1/worker/jobs/{lease['job']}/cases",
                    params={"lease": lease["lease"], "offset": offset, "limit": 200},
                    headers=headers,
                )
            ).json()["cases"]
            cases += page
            offset += len(page)
        items = []
        for case in cases:
            body, gold = case["body"], tracks.solve_body(case["body"])
            for side, skill in (("champion", "base"), ("challenger", "exact")):
                items.append(
                    {
                        "case_index": case["index"],
                        "side": side,
                        "answers": {
                            qid: answer(q, blur(skill, qid, body["seed"], gold[qid]))
                            for qid, q in body["questions"].items()
                        },
                        "reads": {
                            qid: {"label_mass": 1.0, "argmax_is_label": True}
                            for qid in body["questions"]
                        },
                    }
                )
        for start in range(0, len(items), 200):
            response = await api.post(
                f"/v1/worker/jobs/{lease['job']}/answers",
                json={"lease": lease["lease"], "items": items[start : start + 200]},
                headers=headers,
            )
            assert response.status_code == 200, response.text
        response = await api.post(
            f"/v1/worker/jobs/{lease['job']}/complete",
            json={"lease": lease["lease"], "evidence": {"test": True}},
            headers=headers,
        )
        assert response.status_code == 200, response.text


def test_cases_never_expose_gold(make_client, miner, clock, hub):
    client = make_client(duel_cases=5)
    submit(client, miner, hub.publish("miner/z", "exact"), clock)
    lease = client.post("/v1/worker/lease", headers=bearer(WORKER)).json()
    page = client.get(
        f"/v1/worker/jobs/{lease['job']}/cases",
        params={"lease": lease["lease"], "limit": 5},
        headers=bearer(WORKER),
    )
    text = page.text
    assert "gold" not in text and "realized" not in text
    assert json.loads(text)["cases"][0]["body"]["samples"] == "auto"
    wrong = client.get(
        f"/v1/worker/jobs/{lease['job']}/cases",
        params={"lease": "nope"},
        headers=bearer(WORKER),
    )
    assert wrong.status_code == 409


def test_champion_weights_are_cached_and_their_failures_retried(
    make_client, master, clock, hub, launcher, tmp_path
):
    client = make_client(duel_cases=12)
    first, second = Miner(secrets.token_bytes(32)), Miner(secrets.token_bytes(32))
    master.hotkeys.update({first.hotkey: 1, second.hotkey: 2})
    for who, repo in ((first, "miner/p"), (second, "miner/q")):
        sid = submit(client, who, hub.publish(repo, "base", repo), clock).json()["id"]
        asyncio.run(_run(client, tmp_path, hub, launcher))
        assert client.get(f"/v1/submissions/{sid}").json()["state"] == "rejected"
    champion_weights = [d for d in hub.downloads if d == (pins.BASE_REPO, "model.safetensors")]
    assert len(champion_weights) == 1  # the second job reused the verified champion

    # A champion that no longer verifies is infrastructure: re-queued, never rejected.
    for path in (tmp_path / "work" / "champion").rglob("*"):
        if path.is_file():
            path.unlink()
    hub.repos[(pins.BASE_REPO, pins.BASE_REVISION)]["model.safetensors"] = b"gone"
    sid = submit(client, first, hub.publish("miner/r", "exact"), clock).json()["id"]
    asyncio.run(_run(client, tmp_path, hub, launcher))
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "queued" and result["job"]["reason"].startswith("champion:")


def _sandbox_launcher(tmp_path: Path, starts: list[str], build: str = "") -> Any:
    """SandboxLauncher over local bootstrap processes (sandbox.serve, no demotion) serving the
    fake vllm and reader: the production controller path with the sandbox stood in for. The
    bootstrap reports the pinned B300 identity (no GPU here); `build` is the build bootstrap."""
    from opentype_challenge import runtime, sandbox

    from .test_runtime import PROFILE
    from .test_sandbox import _port_base

    reader = tmp_path / "structured_server.py"
    reader.write_text(FAKE.read_text())
    base = _port_base()
    measured = {k: PROFILE[k] for k in (*runtime.MEASURED, "vllm_image")}
    measured["structured_server_sha256"] = pins.STRUCTURED_SERVER_SHA256
    code = (
        "import sys; from pathlib import Path; from opentype_challenge import sandbox; "
        f"sandbox.identity = lambda reader: {{**{measured!r}, 'gpu_uuids': ['GPU-x']}}; "
        "sys.exit(sandbox.serve(sys.stdin.buffer, sys.stdout.buffer, "
        f"vllm={(sys.executable, str(FAKE))!r}, reader=Path({str(reader)!r}), "
        f"demote=False, kernel_dir=Path({str(tmp_path / 'k')!r}), "
        f"log_dir=Path({str(tmp_path / 'logs')!r}), health_timeout=30, port_base={base}))"
    )

    class Backend(sandbox.ProcessBackend):
        async def start(self, mode: str, model: Path | None) -> Any:
            starts.append(model.name if model else mode)
            return await super().start(mode, model)

    build_command = [sys.executable, "-c", build] if build else []
    return sandbox.SandboxLauncher(
        Backend([sys.executable, "-c", code], build_command, gpu=runtime.GPU_TYPE)
    )


def _pin_nvfp4(monkeypatch, hub, config: bytes = b"nvfp4 config") -> dict[str, Any]:
    """A fake official NVFP4 export on the hub, with every pin (and the profile's copies of
    them, fixed at import) patched to it."""
    from opentype_challenge import runtime, sandbox

    files = {"config.json": config, "model.safetensors": b"base",
             "model.safetensors.index.json": b'{"weight_map": {}}'}  # fmt: skip
    revision = hashlib.sha1(b"nvidia/export").hexdigest()
    hub.repos[("nvidia/export", revision)] = files
    config_sha, schema = hashlib.sha256(config).hexdigest(), "5" * 64
    for name, value in (
        ("NVFP4_CONFIG_SHA256", config_sha),
        ("NVFP4_REPO", "nvidia/export"),
        ("NVFP4_REVISION", revision),
        ("NVFP4_FILES", _digests(files)),
        ("NVFP4_SCHEMA_SHA256", schema),
    ):
        monkeypatch.setattr(pins, name, value)
    monkeypatch.setitem(runtime.PROFILE_FIXED, "weights_config_sha256", config_sha)
    monkeypatch.setitem(runtime.PROFILE_FIXED, "weights_schema_sha256", schema)
    monkeypatch.setattr(sandbox, "tensor_schema", lambda model: schema)
    return files


def test_an_nvfp4_duel_runs_one_fresh_sandbox_per_side(
    make_client, miner, clock, hub, tmp_path, monkeypatch
):
    """After the migration a quality duel runs on the B300 path: the worker declares nvfp4,
    serves the champion alone in a fresh sandbox, then the challenger alone in another, both
    at the pinned quality serving config and verified NVFP4, reads through the relay (the
    sandboxes hold no token and are not routable), and the container crowns on its own
    scores. A BF16 worker leases nothing meanwhile."""
    from opentype_challenge import runtime

    nvfp4 = _pin_nvfp4(monkeypatch, hub)

    def publish(repo: str, skill: str) -> dict[str, Any]:
        files = {**nvfp4, "model.safetensors": skill.encode()}
        revision = hashlib.sha1(f"{repo}{skill}".encode()).hexdigest()
        hub.repos[(repo, revision)] = files
        return {"repo": repo, "revision": revision, "files": _digests(files)}

    client = make_client(duel_cases=200)
    assert client.post("/v1/admin/champion/nvfp4", headers=bearer(ADMIN)).status_code == 200

    manifest = publish("miner/exact", "exact")
    sid = submit(client, miner, manifest, clock).json()["id"]
    assert client.post("/v1/worker/lease", headers=bearer(WORKER)).status_code == 204  # BF16

    starts: list[str] = []
    launcher = _sandbox_launcher(tmp_path, starts)
    assert asyncio.run(_run(client, tmp_path, hub, launcher)) is True
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "crowned", result
    assert len(starts) == 2 and starts[1] == "challenger"  # two fresh sandboxes, in order
    assert launcher.quiescent()
    assert [p["side"] for p in launcher.placements] == ["champion", "challenger"]
    _, share = runtime.quality_serving()
    profile = launcher.profile()
    assert profile["weights"] == "modelopt-nvfp4" and profile["gpu_memory_utilization"] == share
    assert profile["moe_backend"] == runtime.QUALITY_SERVING["moe_backend"]
    evidence = result["job"]["evidence"]
    assert evidence["quality_serving"] == runtime.QUALITY_SERVING["version"]
    assert evidence["errors"] == 0 and evidence["executor"] == "modal-sandbox"
    assert client.get("/v1/status").json()["champion"]["hotkey"] == miner.hotkey


def test_a_bf16_champion_never_goes_to_the_sandbox_path(make_client, miner, clock, hub):
    """Before the migration a B300 sandbox worker leases nothing: formats are never mixed."""
    client = make_client(duel_cases=12)
    submit(client, miner, hub.publish("miner/exact", "exact"), clock)
    nvfp4 = client.post("/v1/worker/lease?nvfp4=true", headers=bearer(WORKER))
    assert nvfp4.status_code == 204
    assert client.post("/v1/worker/lease", headers=bearer(WORKER)).status_code == 200


def test_a_runtime_kernel_job_runs_through_the_sandbox_controller(
    make_client, miner, clock, hub, tmp_path, monkeypatch
):
    """The runtime lane end to end on the controller path: an NVFP4 champion, a calibration,
    a signed kernel submission, then Worker(lane="runtime") over SandboxLauncher: one build
    sandbox compiles the kernel, stock and candidate fidelity each serve alone, then B/C/B'
    blocks, each a fresh bootstrap whose measured profile matches the calibration; the kernel
    reaches only C's servers and the container settles the job on its own scores. The local
    bootstrap stands in for the B300 sandbox (identity and the compile child are faked in
    the child process: no GPU, no triton here)."""
    from opentype_challenge import runtime
    from opentype_challenge.miner import signed_runtime_submission

    from .test_runtime import PROFILE, calibration_json
    from .test_sandbox import KERNEL

    _pin_nvfp4(monkeypatch, hub)
    profile = {**PROFILE, **runtime.PROFILE_FIXED}
    client = make_client()
    assert client.post("/v1/admin/champion/nvfp4", headers=bearer(ADMIN)).status_code == 200
    cal = calibration_json(profile=profile, blocks=3, max_drift=50.0)  # CPU timing noise
    assert (
        client.put("/v1/admin/runtime/calibration", json=cal, headers=bearer(ADMIN)).status_code
        == 200
    )
    assert (
        client.put("/v1/admin/lanes", json={"epoch": 1}, headers=bearer(ADMIN)).status_code == 200
    )
    state = client.get("/v1/runtime").json()
    assert state["open"], state
    kernel = {"slot": "rms_norm", "source": KERNEL}
    body = signed_runtime_submission(
        SLUG, state["target"], state["calibration"]["profile_digest"], {}, miner.signer,
        clock.now, kernel=kernel,
    )  # fmt: skip
    posted = client.post("/v1/runtime/submissions", json=body)
    assert posted.status_code == 201, posted.text

    compiled = "print('compiled rms_norm for sm_103')"
    build = (
        "import sys; from opentype_challenge import sandbox; real = sandbox._spawn; "
        f"sandbox._spawn = lambda c, *a, **k: real([sys.executable, '-c', {compiled!r}], *a, **k); "
        "from pathlib import Path; sys.exit(sandbox.build(sys.stdin.buffer, sys.stdout.buffer, "
        f"demote=False, kernel_dir=Path({str(tmp_path / 'b')!r})))"
    )
    starts: list[str] = []
    launcher = _sandbox_launcher(tmp_path, starts, build)

    async def go() -> bool:
        api_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app))
        try:
            instance = Worker(
                Api("http://challenge.test", WORKER, api_client), tmp_path / "work", launcher,
                fetch=hub.fetch, lane="runtime",
            )  # fmt: skip
            return await instance.run_once()
        finally:
            await api_client.aclose()

    assert asyncio.run(go()) is True
    result = client.get(f"/v1/submissions/{posted.json()['id']}").json()
    job = result["job"]
    # CPU timings are noise: the container, not the worker, decides; any verdict it reached
    # on complete evidence (crown, reject or no decision) proves the path
    assert "verdict" in job or "reason" in job, result
    assert starts[0] == "build" and len(starts) == 1 + 2 + 3 * 3  # a fresh sandbox per run
    assert launcher.quiescent()
    sides = [(p["side"], p["kernel"] is not None) for p in launcher.placements]
    assert sides[:2] == [("champion", False), ("challenger", True)]  # stock, then candidate
    assert sides[2:] == [("champion", False), ("champion", True), ("champion", False)] * 3
    evidence = job.get("evidence") or {}
    if not evidence:
        return  # a NO_DECISION requeue keeps no evidence; the placements above are the proof
    assert evidence["kernel_build"] == "compiled rms_norm for sm_103"
    assert len(evidence["runtime"]["blocks"]) == 3 and evidence["errors"] == 0
    assert evidence["runtime"]["profile"] == profile
