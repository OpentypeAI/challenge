"""The challenge container: contract routes, public routes, worker API and admin API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__, pins
from .crypto import (
    CryptoError,
    decode_hotkey,
    encode_hotkey,
    manifest_digest,
    manifest_problem,
    submit_message,
    verify,
)
from .store import Settings, Store, StoreError

SUBMIT_BODY_MAX = 64 * 1024
WORKER_BODY_MAX = 1024 * 1024
RESPONSE_MAX = 8 * 1024 * 1024
MAX_EXP_SECONDS = 300
CASES_PAGE_MAX = 200
METAGRAPH_TTL = 30.0
REPO = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Manifest(Strict):
    repo: str = Field(pattern=REPO)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: dict[str, str] = Field(min_length=2, max_length=128)


class Submission(Strict):
    manifest: Manifest
    hotkey: str = Field(max_length=66)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    exp: int
    signature: str = Field(pattern=r"^(0x)?[0-9a-fA-F]{128}$")


class Answer(Strict):
    case_index: int = Field(ge=0)
    side: Literal["champion", "challenger"]
    answers: dict[str, Any] | None = None
    reads: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=500)


class Lease(Strict):
    lease: str = Field(max_length=64)


class Answers(Lease):
    items: list[Answer] = Field(min_length=1, max_length=2000)


class Finish(Lease):
    evidence: dict[str, Any] = Field(default_factory=dict)


class Failure(Finish):
    reason: str = Field(max_length=500)
    retry: bool = True


class Ladder(Strict):
    order: list[int] = Field(min_length=1)
    width: int


class Pause(Strict):
    paused: bool


@dataclass
class Config:
    slug: str
    state_dir: Path
    master_url: str
    internal_token_file: Path | None
    admin_token_file: Path | None
    worker_token_file: Path | None
    settings: Settings = field(default_factory=Settings)

    @classmethod
    def from_env(cls) -> Config:
        def path(name: str) -> Path | None:
            value = os.environ.get(name)
            return Path(value) if value else None

        cap = os.environ.get("OPENTYPE_WINDOW_ENTITLEMENT_CAP")
        return cls(
            slug=os.environ.get("CHALLENGE_SLUG", "opentype"),
            state_dir=Path(os.environ.get("CHALLENGE_STATE_DIR", "/data")),
            master_url=os.environ.get("CHALLENGE_MASTER_URL", "http://cortex-master:8080"),
            internal_token_file=path("CHALLENGE_INTERNAL_TOKEN_FILE"),
            admin_token_file=path("CHALLENGE_ADMIN_TOKEN_FILE"),
            worker_token_file=path("CHALLENGE_WORKER_TOKEN_FILE"),
            settings=Settings(
                duel_cases=int(os.environ.get("OPENTYPE_DUEL_CASES", "40000")),
                max_pending=int(os.environ.get("OPENTYPE_MAX_PENDING", "4")),
                window_cap=float(cap) if cap else None,
            ),
        )


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _token(path: Path | None) -> str | None:
    """Read lazily per request: the supervisor canary runs with no secret files at all."""
    if path is None:
        return None
    try:
        token = path.read_text().strip()
    except OSError:
        return None
    return token or None


def _require(path: Path | None, authorization: str | None) -> None:
    expected = _token(path)
    if expected is None:
        raise StoreError(503, "this route is not configured")
    presented = (authorization or "").removeprefix("Bearer ").strip()
    if (
        not authorization
        or not authorization.startswith("Bearer ")
        or not hmac.compare_digest(
            hashlib.sha256(presented.encode()).digest(), hashlib.sha256(expected.encode()).digest()
        )
    ):
        raise StoreError(401, "unauthorized")


class Metagraph:
    """GET {master}/v1/metagraph/latest, cached briefly."""

    def __init__(self, url: str, client: httpx.AsyncClient):
        self.url, self.client = url.rstrip("/") + "/v1/metagraph/latest", client
        self._cache: tuple[float, dict[str, int]] | None = None
        self._lock = asyncio.Lock()

    async def hotkeys(self) -> dict[str, int]:
        async with self._lock:
            if self._cache and time.monotonic() - self._cache[0] < METAGRAPH_TTL:
                return self._cache[1]
            try:
                response = await self.client.get(self.url, timeout=10)
                response.raise_for_status()
                hotkeys = response.json()["hotkeys"]
                if not isinstance(hotkeys, dict):
                    raise TypeError("hotkeys must be an object")
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                raise StoreError(503, "the metagraph is unavailable, retry later") from error
            self._cache = (time.monotonic(), hotkeys)
            return hotkeys


def create_app(
    config: Config,
    clock: Callable[[], float] = time.time,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    store = Store(config.state_dir, config.settings, clock)
    client = httpx.AsyncClient(transport=transport)
    metagraph = Metagraph(config.master_url, client)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(
        title="opentype challenge",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.store = store

    @app.exception_handler(StoreError)
    async def store_error(_: Request, error: StoreError) -> JSONResponse:
        return _error(error.status, error.detail)

    async def body(request: Request, limit: int, model: type[Any]) -> Any:
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > limit):
            raise StoreError(413, f"the body exceeds {limit} bytes")
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > limit:
                raise StoreError(413, f"the body exceeds {limit} bytes")
        try:
            return model.model_validate_json(bytes(raw))
        except ValidationError as error:
            raise StoreError(422, _first_error(error)) from None

    def run(function: Callable[..., Any], *args: Any) -> Awaitable[Any]:
        return asyncio.to_thread(function, *args)

    # -- contract -------------------------------------------------------------

    @app.get("/health")
    async def health() -> JSONResponse:
        """Readiness: the state volume is writable and the master bearer is readable."""
        ok = await run(store.healthy) and _token(config.internal_token_file) is not None
        return JSONResponse({"ok": ok}, status_code=200 if ok else 503)

    @app.get("/version")
    async def version() -> dict[str, Any]:
        return {
            "slug": config.slug,
            "version": __version__,
            "contract": 1,
            "capabilities": ["get_weights", "proxy_routes"],
        }

    @app.get("/internal/v1/get_weights")
    async def get_weights(
        epoch: Annotated[int, Query(ge=0, lt=2**64)],
        authorization: Annotated[str | None, Header()] = None,
        x_platform_challenge_slug: Annotated[str | None, Header()] = None,
    ) -> Response:
        _require(config.internal_token_file, authorization)
        if x_platform_challenge_slug != config.slug:
            raise StoreError(403, "challenge slug mismatch")
        text = await run(store.weights, epoch, config.slug)
        return Response(text, media_type="application/json")

    # -- public -----------------------------------------------------------------

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        result: dict[str, Any] = await run(store.status)
        return result

    @app.get("/v1/leaderboard")
    async def leaderboard() -> dict[str, Any]:
        result: dict[str, Any] = await run(store.leaderboard)
        return result

    @app.get("/v1/windows")
    async def windows() -> dict[str, Any]:
        return {"windows": await run(store.windows)}

    @app.get("/v1/windows/{window_id}")
    async def window(window_id: int) -> dict[str, Any]:
        result: dict[str, Any] = await run(store.window, window_id)
        return result

    @app.post("/v1/submissions", status_code=201)
    async def submit(request: Request) -> dict[str, Any]:
        item: Submission = await body(request, SUBMIT_BODY_MAX, Submission)
        manifest = item.manifest
        files = manifest.files
        problem = manifest_problem(files)
        if problem:
            raise StoreError(422, problem)
        if files["config.json"] != pins.BASE_FILES["config.json"]:
            raise StoreError(422, "config.json must be byte-equal to the base revision's")
        now = int(clock())
        if not now < item.exp <= now + MAX_EXP_SECONDS:
            raise StoreError(400, f"exp must be in the future and within {MAX_EXP_SECONDS} s")
        try:
            public = decode_hotkey(item.hotkey)
        except CryptoError:
            raise StoreError(400, "invalid hotkey") from None
        digest = manifest_digest(manifest.repo, manifest.revision, files)
        signature = bytes.fromhex(item.signature.removeprefix("0x"))
        if not verify(public, submit_message(public, digest, item.nonce, item.exp), signature):
            raise StoreError(401, "signature verification failed")
        ss58 = encode_hotkey(public)
        registered = await metagraph.hotkeys()
        if ss58 not in registered:
            raise StoreError(403, "the hotkey is not registered on the subnet")
        result: dict[str, Any] = await run(
            store.submit,
            ss58,
            manifest.repo,
            manifest.revision,
            files,
            digest,
            item.nonce,
            item.exp,
        )
        return result

    @app.get("/v1/submissions/{submission_id}")
    async def submission(submission_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await run(store.submission, submission_id)
        return result

    # -- worker -------------------------------------------------------------------

    def worker(authorization: str | None) -> None:
        _require(config.worker_token_file, authorization)

    @app.post("/v1/worker/lease")
    async def lease(authorization: Annotated[str | None, Header()] = None) -> Response:
        worker(authorization)
        job = await run(store.lease)
        if job is None:
            return Response(status_code=204)
        return JSONResponse(job)

    @app.post("/v1/worker/jobs/{job_id}/heartbeat")
    async def heartbeat(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Lease = await body(request, WORKER_BODY_MAX, Lease)
        result: dict[str, Any] = await run(store.heartbeat, job_id, item.lease)
        return result

    @app.get("/v1/worker/jobs/{job_id}/cases")
    async def cases(
        job_id: str,
        lease: Annotated[str, Query()],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=CASES_PAGE_MAX)] = 100,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        worker(authorization)
        items = await run(store.cases, job_id, lease, offset, limit)
        text = json.dumps({"cases": items}, separators=(",", ":"))
        if len(text) > RESPONSE_MAX:
            raise StoreError(413, "lower the page limit")
        return Response(text, media_type="application/json")

    @app.post("/v1/worker/jobs/{job_id}/answers")
    async def answers(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Answers = await body(request, WORKER_BODY_MAX, Answers)
        rows = [a.model_dump() for a in item.items]
        result: dict[str, Any] = await run(store.record_answers, job_id, item.lease, rows)
        return result

    @app.post("/v1/worker/jobs/{job_id}/complete")
    async def complete(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Finish = await body(request, WORKER_BODY_MAX, Finish)
        result: dict[str, Any] = await run(store.complete, job_id, item.lease, item.evidence)
        return result

    @app.post("/v1/worker/jobs/{job_id}/fail")
    async def fail(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Failure = await body(request, WORKER_BODY_MAX, Failure)
        result: dict[str, Any] = await run(
            store.fail, job_id, item.lease, item.reason, item.retry, item.evidence
        )
        return result

    # -- admin ------------------------------------------------------------------

    def admin(authorization: str | None) -> None:
        _require(config.admin_token_file, authorization)

    @app.post("/v1/admin/window/rotate")
    async def rotate(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        admin(authorization)
        result: dict[str, Any] = await run(store.rotate_window)
        return result

    @app.put("/v1/admin/ladder")
    async def ladder(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        admin(authorization)
        item: Ladder = await body(request, SUBMIT_BODY_MAX, Ladder)
        result: dict[str, Any] = await run(store.set_ladder, item.order, item.width)
        return result

    @app.put("/v1/admin/crowns")
    async def crowns(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        admin(authorization)
        item: Pause = await body(request, SUBMIT_BODY_MAX, Pause)
        await run(store.set_crowns_paused, item.paused)
        return {"crowns_paused": item.paused}

    @app.post("/v1/admin/jobs/{job_id}/requeue")
    async def requeue(
        job_id: str, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        admin(authorization)
        result: dict[str, Any] = await run(store.requeue, job_id)
        return result

    return app


def _first_error(error: ValidationError) -> str:
    first = error.errors()[0]
    where = ".".join(str(part) for part in first.get("loc", ()))
    return f"{where}: {first.get('msg', 'invalid')}" if where else str(first.get("msg"))
