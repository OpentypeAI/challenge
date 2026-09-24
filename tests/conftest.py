from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from opentype_challenge import pins
from opentype_challenge.app import Config, create_app
from opentype_challenge.crypto import encode_hotkey, sign_with_seed
from opentype_challenge.miner import Signer, signed_submission
from opentype_challenge.store import Settings

SLUG = "opentype"
INTERNAL, ADMIN, WORKER = "internal-token", "admin-token", "worker-token"


class Clock:
    def __init__(self, now: float = 1_790_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Master:
    """A fake Cortex master: GET /v1/metagraph/latest."""

    def __init__(self) -> None:
        self.hotkeys: dict[str, int] = {}
        self.up = True
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if not self.up:
            return httpx.Response(503, json={"detail": "no seal"})
        assert request.url.path == "/v1/metagraph/latest"
        return httpx.Response(
            200, json={"epoch": 7, "block": 100, "netuid": 1, "hotkeys": self.hotkeys}
        )


class Miner:
    def __init__(self, seed: bytes):
        self.seed = seed
        public = sign_with_seed(seed, b"")[0]
        self.signer = Signer(public, lambda message: sign_with_seed(seed, message)[1])
        self.hotkey = encode_hotkey(public)


def weights_manifest(tag: str, repo: str = "miner/model") -> dict[str, Any]:
    """A challenger manifest whose weight digests are derived from tag."""
    files = dict(pins.BASE_FILES)
    for name in files:
        if name.endswith(".safetensors"):
            files[name] = hashlib.sha256(f"{tag}|{name}".encode()).hexdigest()
    return {"repo": repo, "revision": hashlib.sha1(tag.encode()).hexdigest(), "files": files}


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def master() -> Master:
    return Master()


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "secrets"
    directory.mkdir()
    for name, token in (("internal", INTERNAL), ("admin", ADMIN), ("worker", WORKER)):
        (directory / f"{name}.token").write_text(token + "\n")
    return directory


@pytest.fixture(autouse=True)
def no_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never pick up an operator gateway from the environment."""
    monkeypatch.delenv("OPENTYPE_TEACHER_URL", raising=False)


@pytest.fixture
def make_client(
    tmp_path: Path, secrets_dir: Path, clock: Clock, master: Master
) -> Iterator[Callable[..., TestClient]]:
    """make(judge=, bank_builder=, beacon=, **Settings): judge and bank_builder are the
    app's async fakes; beacon defaults to "drand unreachable"."""
    clients: list[TestClient] = []

    def make(
        judge: Any = None, bank_builder: Any = None, beacon: Any = None, **settings: Any
    ) -> TestClient:
        config = Config(
            slug=SLUG,
            state_dir=tmp_path / "data",
            master_url="http://master.test",
            internal_token_file=secrets_dir / "internal.token",
            admin_token_file=secrets_dir / "admin.token",
            worker_token_file=secrets_dir / "worker.token",
            settings=Settings(**{"duel_cases": 40, **settings}),
        )
        app = create_app(
            config,
            clock,
            httpx.MockTransport(master.handler),
            judge=judge,
            bank_builder=bank_builder,
            beacon=beacon or (lambda: None),  # never reach drand from tests
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture
def miner(master: Master) -> Miner:
    registered = Miner(secrets.token_bytes(32))
    master.hotkeys[registered.hotkey] = 1
    return registered


def submit(client: TestClient, who: Miner, manifest: dict[str, Any], clock: Clock) -> Any:
    return client.post("/v1/submissions", json=signed_submission(manifest, who.signer, clock.now))


def bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}
