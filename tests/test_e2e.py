"""Full duels: intake -> worker (real processes, fake inference) -> crown -> ledger -> weights."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import socket
import sys
from pathlib import Path

import httpx
import pytest

from opentype_challenge import pins
from opentype_challenge.bank import cases_digest, job_case, job_seed
from opentype_challenge.worker import Api, JobFailed, VllmLauncher, Worker, assemble

from .conftest import ADMIN, INTERNAL, SLUG, WORKER, Miner, bearer, submit

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


def test_duel_crowns_pays_and_audits(make_client, miner, clock, hub, launcher, tmp_path):
    client = make_client(duel_cases=240)
    manifest = hub.publish("miner/exact", "exact")
    sid = submit(client, miner, manifest, clock).json()["id"]

    assert asyncio.run(_run(client, tmp_path, hub, launcher)) is True
    result = client.get(f"/v1/submissions/{sid}").json()
    assert result["state"] == "crowned", result
    verdict = result["job"]["verdict"]
    assert verdict["crown"] and verdict["g_lcb"] >= verdict["g_min"]
    challenger = verdict["levels"]["1"]["challenger"]
    assert challenger["accuracy"] == 1.0  # exact gold: 100 % is reachable
    evidence = result["job"]["evidence"]
    assert (
        evidence["challenger_files"]["model.safetensors"] == manifest["files"]["model.safetensors"]
    )
    assert (
        evidence["challenger_files"]["tokenizer.json"] == pins.BASE_SUPPORT_FILES["tokenizer.json"]
    )
    assert evidence["structured_server_sha256"] == hashlib.sha256(FAKE.read_bytes()).hexdigest()
    assert evidence["cases_fetched"] == 240
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

    # After rotation the secret is revealed and anyone regenerates the exact served cases.
    client.post("/v1/admin/window/rotate", headers=bearer(ADMIN))
    window = client.get("/v1/windows/1").json()
    job = window["jobs"][0]
    seed = job_seed(bytes.fromhex(window["secret"]), job["id"], job["digest"])
    regenerated = cases_digest(job_case(seed, job["mix"], i).body for i in range(job["cases"]))
    assert regenerated == job["cases_sha256"] == evidence["cases_sha256"]


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
    from opentype_challenge.generator import solve

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
            body, gold = case["body"], solve(case["body"])
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
