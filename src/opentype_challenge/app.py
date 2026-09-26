"""The challenge container: contract routes, public routes, worker API and admin API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import secrets
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictInt,
    ValidationError,
)

from . import __version__, bank, generator, runtime, teacher, tracks
from .crypto import (
    CryptoError,
    decode_hotkey,
    encode_hotkey,
    manifest_digest,
    manifest_problem,
    runtime_digest,
    runtime_message,
    submit_message,
    verify,
)
from .harness import MAX_OUTPUT_CHARS
from .store import Settings, Store, StoreError, plan_from_json

log = logging.getLogger(__name__)

Judge = Callable[[str, list[str], bytes], Awaitable[float | None]]
BankBuilder = Callable[[random.Random], Awaitable[list[bank.BankItem]]]

SUBMIT_BODY_MAX = 64 * 1024
WORKER_BODY_MAX = 1024 * 1024
RESPONSE_MAX = 8 * 1024 * 1024
MAX_EXP_SECONDS = 300
CASES_PAGE_MAX = 200
BANK_PAGE_MAX = 10_000
TRANSCRIPT_ITEMS = 64
BACKGROUND_SECONDS = 30.0  # judging and window-rotation poll period
BUILDER_POLL_SECONDS = 1.0  # how soon a promoted bank's successor starts building
BANK_RETRY_SECONDS = 600.0
COMPLETE_WAIT_SECONDS = 20.0  # complete waits this long for judging before answering
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


class RuntimeTarget(Strict):
    champion: int = Field(ge=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class Kernel(Strict):
    slot: str = Field(max_length=32)
    source: str = Field(min_length=1, max_length=runtime.KERNEL_MAX_BYTES)


class RuntimeSubmission(Strict):
    """Allowlisted vLLM options and/or one Triton kernel for a registered slot; never argv,
    env, image, plugin or reader."""

    target: RuntimeTarget
    profile: str = Field(pattern=r"^[0-9a-f]{64}$")
    options: dict[str, StrictBool | StrictInt] = Field(
        default_factory=dict, max_length=len(runtime.OPTIONS)
    )
    kernel: Kernel | None = None
    hotkey: str = Field(max_length=66)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    exp: int
    signature: str = Field(pattern=r"^(0x)?[0-9a-fA-F]{128}$")


class LanesFrom(Strict):
    epoch: int = Field(ge=0, lt=2**64)


class Answer(Strict):
    """A read item (answers, reads), a harness item (transcript) or a failure (error)."""

    case_index: int = Field(ge=0)
    side: Literal["champion", "challenger"]
    answers: dict[str, Any] | None = None
    reads: dict[str, Any] | None = None
    transcript: list[Annotated[str, Field(max_length=MAX_OUTPUT_CHARS)]] | None = Field(
        default=None, max_length=TRANSCRIPT_ITEMS
    )
    error: str | None = Field(default=None, max_length=500)


class Lease(Strict):
    lease: str = Field(max_length=64)


class Timing(Answer):
    """One timed runtime task: the raw output (scored by the container) and its latency."""

    side: Literal["B", "C", "B2"]  # type: ignore[assignment]
    block: int = Field(ge=0, le=runtime.MAX_BLOCKS)
    cell: str = Field(max_length=runtime.MAX_CELL_NAME)
    ms: float = Field(ge=0, allow_inf_nan=False)


class Timings(Lease):
    items: list[Timing] = Field(min_length=1, max_length=2000)


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
    window_hours: float = 24.0

    @classmethod
    def from_env(cls) -> Config:
        def path(name: str) -> Path | None:
            value = os.environ.get(name)
            return Path(value) if value else None

        cap = os.environ.get("OPENTYPE_WINDOW_ENTITLEMENT_CAP")
        empty_cap = os.environ.get("OPENTYPE_EMPTY_BANK_CAP")
        plan = os.environ.get("OPENTYPE_PLAN")
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
                empty_bank_cap=float(empty_cap) if empty_cap else 0.0,
                plan=plan_from_json(json.loads(plan)) if plan else None,
            ),
            window_hours=float(os.environ.get("OPENTYPE_WINDOW_HOURS", "24")),
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


def _teacher() -> tuple[teacher.Gateway, Judge, BankBuilder] | None:
    """The gateway judge and bank builder when the teacher is configured and its token file
    is readable; None otherwise (the teacher is off, everything else works)."""
    try:
        cfg = teacher.TeacherConfig.from_env()
    except ValueError as error:
        log.warning("teacher off: %s", error)
        return None
    if cfg is None:
        return None
    try:
        if not cfg.token_file.read_text().strip():
            return None
    except OSError:
        log.warning("teacher off: the token file is unreadable")
        return None
    gateway = teacher.Gateway(cfg)

    async def judge(brief: str, rubric: list[str], png: bytes) -> float | None:
        return await teacher.judge_png(gateway, cfg, brief, rubric, png)

    async def build(rng: random.Random) -> list[bank.BankItem]:
        return await teacher.build_bank(gateway, cfg, rng, cfg.targets, families=generator.FAMILIES)

    return gateway, judge, build


def create_app(
    config: Config,
    clock: Callable[[], float] = time.time,
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    judge: Judge | None = None,
    bank_builder: BankBuilder | None = None,
    beacon: Callable[[], dict[str, Any] | None] | None = None,
) -> FastAPI:
    """judge(brief, rubric, png) -> loss | None and bank_builder(rng) -> items are the teacher
    hooks; both None means "from teacher.TeacherConfig.from_env() when its token is readable".
    beacon() -> {round, randomness} | None defaults to drand through `transport` when it can
    serve sync requests."""
    # a plan that builds nothing must fail the canary, not 500 every submission later
    tracks.effective_plan(config.settings.track_plan(), bank.EMPTY_BANK, judge=False)
    gateway: teacher.Gateway | None = None
    if judge is None and bank_builder is None:
        found = _teacher()
        if found is not None:
            gateway, judge, bank_builder = found
    sync_client: httpx.Client | None = None
    if beacon is None:
        if isinstance(transport, httpx.BaseTransport):
            sync_client = httpx.Client(transport=transport)
        drand = sync_client

        def beacon() -> dict[str, Any] | None:
            return bank.drand_beacon(drand)

    store = Store(config.state_dir, config.settings, clock, judge=judge is not None, beacon=beacon)
    store.teacher_configured = judge is not None or bank_builder is not None
    client = httpx.AsyncClient(transport=transport)
    metagraph = Metagraph(config.master_url, client)
    # sides being judged; the background loop and a waiting complete never judge one twice
    claims: set[tuple[str, int, str]] = set()
    claims_lock = threading.Lock()
    running: set[asyncio.Future[int]] = set()

    def run(function: Callable[..., Any], *args: Any) -> Awaitable[Any]:
        return asyncio.to_thread(function, *args)

    async def judge_pending() -> int:
        """Judge the pending sides (each case's sides in its seed order), then settle the jobs
        with nothing left to judge. Returns the number of sides judged here. A judge that
        raises (gateway outage, closed at shutdown) leaves its side pending for a later pass;
        only None (the judge's replies about this render were unreadable) excludes the pair.
        Without a judge (teacher off at startup) nothing is judged: the sides stay pending
        until a restart with the teacher or the judging deadline, never forfeited.
        A job still judging after JUDGE_DEADLINE_SECONDS settles without its missing sides.
        """
        done = 0
        down = False
        while judge is not None and not down:
            batch = await run(store.pending_judgments)
            with claims_lock:
                mine = [
                    item
                    for item in batch
                    if (item["job"], item["case_index"], item["side"]) not in claims
                ]
                keys = {(item["job"], item["case_index"], item["side"]) for item in mine}
                claims.update(keys)
            if not mine:
                break
            try:
                for item in mine:
                    try:
                        loss = await judge(item["brief"], item["rubric"], item["png"])
                    except Exception:
                        log.exception("judge call failed; the side stays pending")
                        down = True
                        break
                    await run(
                        store.record_judgment, item["job"], item["case_index"], item["side"], loss
                    )
                    done += 1
            finally:
                with claims_lock:
                    claims.difference_update(keys)
        await run(store.settle_judged)
        return done

    async def rotate_due() -> None:
        await run(store.auto_rotate, config.window_hours * 3600)

    async def tick() -> None:
        """One pass of background work: window auto-rotation, judging and settling."""
        await rotate_due()
        await judge_pending()

    async def build_next() -> bool:
        """Build the next window's bank when none is ready; True when one was built."""
        if bank_builder is None or await run(store.next_bank_ready):
            return False
        store.teacher_building = True
        try:
            items = await bank_builder(random.Random(secrets.token_bytes(32)))
        finally:
            store.teacher_building = False
        digest = await run(store.set_next_bank, items)
        log.info("next window bank ready: %d items, digest %s", len(items), digest)
        return True

    async def background(work: Callable[[], Awaitable[Any]]) -> None:
        # rotation has its own loop: a judging backlog (hours of calls) never delays it
        while True:
            try:
                await work()
            except Exception:
                log.exception("background pass failed")
            await asyncio.sleep(BACKGROUND_SECONDS)

    async def builder() -> None:
        while bank_builder is not None:
            try:
                await build_next()
            except Exception:
                log.exception("bank build failed")
                await asyncio.sleep(BANK_RETRY_SECONDS)
            await asyncio.sleep(BUILDER_POLL_SECONDS)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        tasks = [
            asyncio.create_task(background(rotate_due)),
            asyncio.create_task(background(judge_pending)),
            asyncio.create_task(builder()),
        ]
        yield
        # judge calls started by complete too: they must not outlive the gateway
        futures: list[asyncio.Future[Any]] = [*tasks, *running]
        for future in futures:
            future.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        await client.aclose()
        if sync_client is not None:
            sync_client.close()
        if gateway is not None:
            await gateway.aclose()

    app = FastAPI(
        title="opentype challenge",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.tick = tick
    app.state.build_next = build_next

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

    @app.get("/v1/windows/{window_id}/bank")
    async def window_bank(
        window_id: int,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=BANK_PAGE_MAX)] = 1000,
    ) -> Response:
        page = await run(store.window_bank, window_id, offset, limit)
        return Response(json.dumps(page, separators=(",", ":")), media_type="application/json")

    @app.post("/v1/submissions", status_code=201)
    async def submit(request: Request) -> dict[str, Any]:
        item: Submission = await body(request, SUBMIT_BODY_MAX, Submission)
        manifest = item.manifest
        files = manifest.files
        problem = manifest_problem(files)
        if problem:
            raise StoreError(422, problem)
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

    @app.post("/v1/runtime/submissions", status_code=201)
    async def submit_runtime(request: Request) -> dict[str, Any]:
        item: RuntimeSubmission = await body(request, SUBMIT_BODY_MAX, RuntimeSubmission)
        try:
            options, kernel = runtime.normalize_candidate(
                item.options, item.kernel.model_dump() if item.kernel else None
            )
        except runtime.RuntimeError_ as error:
            raise StoreError(422, str(error)) from None
        now = int(clock())
        if not now < item.exp <= now + MAX_EXP_SECONDS:
            raise StoreError(400, f"exp must be in the future and within {MAX_EXP_SECONDS} s")
        try:
            public = decode_hotkey(item.hotkey)
        except CryptoError:
            raise StoreError(400, "invalid hotkey") from None
        target = item.target.model_dump()
        digest = runtime_digest(config.slug, target, item.profile, options, kernel)
        signature = bytes.fromhex(item.signature.removeprefix("0x"))
        if not verify(public, runtime_message(public, digest, item.nonce, item.exp), signature):
            raise StoreError(401, "signature verification failed")
        ss58 = encode_hotkey(public)
        if ss58 not in await metagraph.hotkeys():
            raise StoreError(403, "the hotkey is not registered on the subnet")
        result: dict[str, Any] = await run(
            store.submit_runtime,
            ss58,
            target,
            item.profile,
            options,
            digest,
            item.nonce,
            item.exp,
            kernel,
        )
        return result

    @app.get("/v1/runtime")
    async def runtime_view() -> dict[str, Any]:
        result: dict[str, Any] = await run(store.runtime_status)
        return result

    @app.get("/v1/submissions/{submission_id}")
    async def submission(submission_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await run(store.submission, submission_id)
        return result

    # -- worker -------------------------------------------------------------------

    def worker(authorization: str | None) -> None:
        _require(config.worker_token_file, authorization)

    @app.post("/v1/worker/lease")
    async def lease(
        lane: Annotated[Literal["quality", "runtime"], Query()] = "quality",
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Workers ask for their lane; a worker that does not ask gets quality jobs only."""
        worker(authorization)
        job = await run(store.lease, lane)
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
        if len(text) > RESPONSE_MAX:  # the store keeps pages under PAGE_BYTES already
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

    @app.post("/v1/worker/jobs/{job_id}/timings")
    async def timings(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Timings = await body(request, WORKER_BODY_MAX, Timings)
        rows = [t.model_dump() for t in item.items]
        result: dict[str, Any] = await run(store.record_timings, job_id, item.lease, rows)
        return result

    @app.post("/v1/worker/jobs/{job_id}/complete")
    async def complete(
        job_id: str, request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        worker(authorization)
        item: Finish = await body(request, WORKER_BODY_MAX, Finish)
        result: dict[str, Any] = await run(store.complete, job_id, item.lease, item.evidence)
        if result.get("job", {}).get("state") == "judging":
            # ponytail: judges inline for up to COMPLETE_WAIT_SECONDS so a fast judge settles
            # before the worker moves on; the background loop finishes anything slower.
            judging = asyncio.ensure_future(judge_pending())
            running.add(judging)  # keep a reference until it finishes
            judging.add_done_callback(running.discard)
            try:
                await asyncio.wait_for(asyncio.shield(judging), COMPLETE_WAIT_SECONDS)
            except TimeoutError:
                pass
            result = await run(store.submission, result["id"])
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

    @app.put("/v1/admin/runtime/calibration")
    async def calibration(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        """The operator's pilot result; null withdraws it and closes the runtime lane."""
        admin(authorization)
        raw: RootModel[Any] = await body(request, SUBMIT_BODY_MAX, RootModel[Any])
        try:
            published = await run(store.set_calibration, raw.root)
        except runtime.RuntimeError_ as error:
            raise StoreError(422, str(error)) from None
        return {"calibration": published}

    @app.post("/v1/admin/champion/nvfp4")
    async def migrate_nvfp4(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        """One-way: the champion becomes this NVFP4 checkpoint of its weights (prospective)."""
        admin(authorization)
        item: Manifest = await body(request, SUBMIT_BODY_MAX, Manifest)
        result: dict[str, Any] = await run(
            store.migrate_nvfp4, item.repo, item.revision, item.files
        )
        return result

    @app.put("/v1/admin/lanes")
    async def lanes(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        """Schedule the 75/25 split from a future epoch (once, never retroactive)."""
        admin(authorization)
        item: LanesFrom = await body(request, SUBMIT_BODY_MAX, LanesFrom)
        return {"lanes_from_epoch": await run(store.set_lanes_from, item.epoch)}

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
