"""The B300 duel worker: lease, fetch and verify weights, serve both models, run every case,
report (docs/tracks.md §7).

Read tracks (decisions, longctx) post the body to the structured server; harness tracks (ops,
sql, paint) play an episode against vllm's chat endpoint and report the raw outputs, which
the container replays. The worker reaches the container only through the master proxy, so
every call is a public path with the worker bearer, each request body stays under 1 MiB and
each page of cases under 8 MiB.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import __version__, harness, pins, tracks
from .bank import case_line
from .crypto import manifest_problem

SIDES = ("champion", "challenger")
BATCH_BYTES = 900 * 1024
PAGE = 100
CONCURRENCY = 64
READ_TIMEOUT = 300.0
HEALTH_TIMEOUT = 1800.0
HEARTBEAT_SECONDS = 300.0
RESOLVED = ".opentype-resolved.json"  # written only after every sha256 matched
CANVAS = 256
MAX_MODEL_LEN = 131072  # longctx level 5 (~100k tokens) fits with headroom
STRUCTURED_SERVER = Path(
    os.environ.get("OPENTYPE_STRUCTURED_SERVER", "/opt/opentype/structured_server.py")
)


class JobFailed(Exception):
    """An error attributable to the challenger (no retry) or to infrastructure (retry)."""

    def __init__(self, reason: str, retry: bool):
        super().__init__(reason)
        self.reason, self.retry = reason, retry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Weights.


Fetch = Callable[[str, str, str, Path], Path]


def hf_fetch(repo: str, revision: str, filename: str, directory: Path) -> Path:
    """One file of a public repo at a pinned commit, anonymously (no BYOK)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        path = hf_hub_download(repo, filename, revision=revision, local_dir=directory, token=False)
    except (EntryNotFoundError, GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
        raise JobFailed(
            f"{repo}@{revision}/{filename} is not publicly downloadable", False
        ) from None
    return Path(path)


def assemble(
    manifest: Mapping[str, Any], base_dir: Path, directory: Path, fetch: Fetch
) -> dict[str, str]:
    """Download a manifest into directory, verify every sha256, add the base support files.

    Only weights, the weight index and config.json come from the miner; config.json must be
    byte-equal to the base revision's, and tokenizer/chat template/processor files are
    copied from the verified base snapshot. Returns the resolved sha256 of every file.
    """
    files: dict[str, str] = dict(manifest["files"])
    problem = manifest_problem(files)
    if problem:
        raise JobFailed(f"{manifest['repo']}: {problem}", retry=False)
    if files["config.json"] != pins.BASE_FILES["config.json"]:
        raise JobFailed("config.json differs from the base revision", retry=False)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for name, expected in sorted(files.items()):
        try:
            path = fetch(manifest["repo"], manifest["revision"], name, directory)
        except JobFailed:
            raise
        except Exception as error:  # noqa: BLE001 - any other hub error is infrastructure
            raise JobFailed(f"download {manifest['repo']}/{name}: {error}", retry=True) from None
        actual = sha256_file(path)
        if actual != expected:
            raise JobFailed(f"{manifest['repo']}/{name}: sha256 {actual} != {expected}", False)
        resolved[name] = actual
    for name in pins.BASE_SUPPORT_FILES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base_dir / name, target)
        resolved[name] = sha256_file(target)
    return resolved


def base_snapshot(directory: Path, fetch: Fetch) -> Path:
    """The pinned base tokenizer/template/processor files, verified once and kept."""
    for name, expected in pins.BASE_SUPPORT_FILES.items():
        path = directory / name
        if not path.exists() or sha256_file(path) != expected:
            path = fetch(pins.BASE_REPO, pins.BASE_REVISION, name, directory)
            if sha256_file(path) != expected:
                raise JobFailed(f"base {name} does not match its pinned sha256", retry=True)
    return directory


# ---------------------------------------------------------------------------
# Serving. The launcher is injected; tests use a fake systemone server.


Urls = Mapping[str, Mapping[str, str]]  # {side: {"reader": url, "chat": url}}


class Launcher(Protocol):
    def __call__(
        self, models: Mapping[str, Path]
    ) -> AbstractAsyncContextManager[dict[str, dict[str, str]]]:
        """Serve each side's model; yield {side: {"reader": systemone URL, "chat": vllm URL}}."""
        ...

    def evidence(self) -> dict[str, Any]:
        """What serves the reads: versions and file digests."""
        ...


@dataclass
class VllmLauncher:
    """Two `vllm serve` processes (BF16, split memory) and one structured_server.py each."""

    canvas: int = CANVAS
    max_model_len: int = MAX_MODEL_LEN
    memory_share: float = 0.45
    port_base: int = 8100
    log_dir: Path = field(default_factory=lambda: Path("/tmp"))  # noqa: S108
    vllm: tuple[str, ...] = ("vllm",)
    reader: Path = STRUCTURED_SERVER

    def evidence(self) -> dict[str, Any]:
        return {
            "vllm_image": pins.VLLM_IMAGE,
            "vllm_version": _package_version("vllm"),
            "structured_server_sha256": sha256_file(self.reader),
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
        }

    def ports(self, side: str) -> tuple[int, int]:
        """(vllm port, structured server port) of one side."""
        offset = SIDES.index(side)
        return self.port_base + offset, self.port_base + 10 + offset

    def commands(self, side: str, model: Path) -> list[list[str]]:
        vllm_port, reader_port = self.ports(side)
        return [
            [
                *self.vllm,
                "serve",
                str(model),
                "--served-model-name",
                side,
                "--port",
                str(vllm_port),
                "--host",
                "127.0.0.1",
                "--dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                str(self.memory_share),
                "--diffusion-config",
                json.dumps({"canvas_length": self.canvas}),
                "--max-logprobs",
                "32",
                "--enable-prefix-caching",
                "--max-model-len",
                str(self.max_model_len),
                "--limit-mm-per-prompt",
                json.dumps({"image": 1}),
            ],
            [
                sys.executable,
                str(self.reader),
                "--upstream",
                f"http://127.0.0.1:{vllm_port}",
                "--model",
                side,
                "--tokenizer",
                str(model),
                "--canvas",
                str(self.canvas),
                "--host",
                "127.0.0.1",
                "--port",
                str(reader_port),
            ],
        ]

    @asynccontextmanager
    async def __call__(
        self, models: Mapping[str, Path]
    ) -> AsyncIterator[dict[str, dict[str, str]]]:
        processes: list[subprocess.Popen[bytes]] = []
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        try:
            async with httpx.AsyncClient() as client:
                for side, model in models.items():
                    serve, reader = self.commands(side, model)
                    vllm_port, reader_port = self.ports(side)
                    processes.append(self._spawn(serve, env, f"vllm-{side}"))
                    await _wait_health(client, f"http://127.0.0.1:{vllm_port}/health", processes)
                    processes.append(self._spawn(reader, env, f"reader-{side}"))
                    await _wait_health(client, f"http://127.0.0.1:{reader_port}/health", processes)
            yield {
                side: {
                    "reader": f"http://127.0.0.1:{self.ports(side)[1]}",
                    "chat": f"http://127.0.0.1:{self.ports(side)[0]}",
                }
                for side in models
            }
        finally:
            for process in reversed(processes):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)

    def _spawn(self, command: list[str], env: dict[str, str], name: str) -> subprocess.Popen[bytes]:
        log = (self.log_dir / f"{name}.log").open("wb")
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


async def _wait_health(
    client: httpx.AsyncClient, url: str, processes: list[subprocess.Popen[bytes]]
) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise JobFailed(f"a serving process exited before {url} was healthy", retry=True)
        with contextlib.suppress(httpx.HTTPError):
            if (await client.get(url, timeout=5)).status_code == 200:
                return
        await asyncio.sleep(0.5)
    raise JobFailed(f"{url} never became healthy", retry=True)


# ---------------------------------------------------------------------------
# The duel.


def slim(answer: Any) -> Any:
    """Drop per-read diagnostics; keep what scoring reads."""
    if not isinstance(answer, dict):
        return None
    keys = ("noul", "choice", "score", "legend", "probabilities", "confidence")
    return {k: answer[k] for k in keys if k in answer}


def reads_of(body: Mapping[str, Any]) -> dict[str, Any]:
    questions = (body.get("diagnostics") or {}).get("questions") or {}
    return {
        qid: {"label_mass": q.get("label_mass"), "argmax_is_label": q.get("argmax_is_label")}
        for qid, q in questions.items()
        if isinstance(q, dict)
    }


class _ItemError(Exception):
    """A 4xx or malformed model reply: that side forfeits the case, the job goes on."""


class Api:
    def __init__(self, base: str, token: str, client: httpx.AsyncClient):
        self.base, self.client = base.rstrip("/"), client
        self.headers = {"authorization": f"Bearer {token}"}

    async def call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(5):
            try:
                response = await self.client.request(
                    method, self.base + path, headers=self.headers, timeout=60, **kwargs
                )
            except httpx.TransportError:
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code in (502, 503, 504):
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {path}: {response.status_code} {response.text[:300]}")
            return response
        raise RuntimeError(f"{method} {path}: the API stayed unavailable")


@dataclass
class Worker:
    api: Api
    workdir: Path
    launcher: Launcher
    fetch: Fetch = hf_fetch
    concurrency: int = CONCURRENCY
    inference: httpx.AsyncClient | None = None

    async def run_once(self) -> bool:
        """Lease and run one job. False when the queue is empty."""
        response = await self.api.call("POST", "/v1/worker/lease")
        if response.status_code == 204:
            return False
        job = response.json()
        started = time.time()
        evidence: dict[str, Any] = {
            "worker_version": __version__,
            "image": os.environ.get("OPENTYPE_WORKER_IMAGE", "unknown"),
            **self.launcher.evidence(),
            "base": {"repo": pins.BASE_REPO, "revision": pins.BASE_REVISION},
        }
        job_dir = self.workdir / job["job"]
        beat = asyncio.create_task(self._heartbeat(job))
        try:
            evidence.update(await self._duel(job, job_dir, evidence))
            evidence["seconds"] = round(time.time() - started, 1)
            await self.api.call(
                "POST",
                f"/v1/worker/jobs/{job['job']}/complete",
                json={"lease": job["lease"], "evidence": evidence},
            )
        except Exception as error:  # noqa: BLE001 - every failure is reported, never swallowed
            failure = error if isinstance(error, JobFailed) else JobFailed(repr(error), True)
            await self.api.call(
                "POST",
                f"/v1/worker/jobs/{job['job']}/fail",
                json={
                    "lease": job["lease"],
                    "reason": failure.reason[:500],
                    "retry": failure.retry,
                    "evidence": evidence,
                },
            )
        finally:
            beat.cancel()
            shutil.rmtree(job_dir, ignore_errors=True)
        return True

    async def _heartbeat(self, job: dict[str, Any]) -> None:
        """Keep the lease alive during downloads and server start-up."""
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            with contextlib.suppress(RuntimeError):
                await self.api.call(
                    "POST", f"/v1/worker/jobs/{job['job']}/heartbeat", json={"lease": job["lease"]}
                )

    async def _duel(self, job: dict[str, Any], job_dir: Path, evidence: dict[str, Any]) -> dict:
        base = await asyncio.to_thread(base_snapshot, self.workdir / "base", self.fetch)
        try:
            champion, evidence["champion_files"] = await asyncio.to_thread(
                self._champion, job["champion"], base
            )
        except JobFailed as error:  # never the challenger's fault
            raise JobFailed(f"champion: {error.reason}", retry=True) from None
        evidence["challenger_files"] = await asyncio.to_thread(
            assemble, job["challenger"], base, job_dir / "challenger", self.fetch
        )
        models = {"champion": champion, "challenger": job_dir / "challenger"}
        timings: dict[str, Any] = {}
        t0 = time.time()
        async with self.launcher(models) as urls:
            timings["serve_seconds"] = round(time.time() - t0, 1)
            t1 = time.time()
            counts = await self._read(job, urls)
            timings["read_seconds"] = round(time.time() - t1, 1)
        return {"timings": timings, **counts}

    def _champion(self, manifest: Mapping[str, Any], base: Path) -> tuple[Path, dict[str, str]]:
        """The champion's verified weights, kept across jobs by manifest digest (it is public
        and duels every challenger); challenger weights are deleted after each job."""
        root = self.workdir / "champion"
        directory = root / manifest["digest"]
        record = directory / RESOLVED
        if record.exists():
            resolved: dict[str, str] = json.loads(record.read_text())
            return directory, resolved
        shutil.rmtree(root, ignore_errors=True)  # only the current champion is kept
        resolved = assemble(manifest, base, directory, self.fetch)
        record.write_text(json.dumps(resolved, sort_keys=True))
        return directory, resolved

    async def _read(self, job: dict[str, Any], urls: Urls) -> dict[str, Any]:
        """Page through every case, run both sides and post the answer items in batches."""
        client = self.inference or httpx.AsyncClient()
        limit = asyncio.Semaphore(self.concurrency)
        cases_hash = hashlib.sha256()
        fetched = errors = 0
        keep_going = True

        async def post(side: str, url: str, body: dict[str, Any]) -> httpx.Response:
            """One inference call; 5xx and transport errors are infrastructure (retry)."""
            async with limit:
                try:
                    response = await client.post(url, json=body, timeout=READ_TIMEOUT)
                except httpx.HTTPError as error:
                    raise JobFailed(f"{side} {url}: {error!r}", retry=True) from None
            if response.status_code >= 500:
                raise JobFailed(f"{side} {url} returned {response.status_code}", retry=True)
            if response.status_code != 200:
                raise _ItemError(f"{response.status_code}: {response.text[:200]}")
            return response

        async def read(side: str, body: dict[str, Any]) -> dict[str, Any]:
            data = (await post(side, urls[side]["reader"] + "/v1/systemone", body)).json()
            answers = data.get("answers") or {}
            return {
                "answers": {qid: slim(a) for qid, a in answers.items()},
                "reads": reads_of(data),
            }

        async def play(side: str, track: str, body: dict[str, Any]) -> dict[str, Any]:
            url = urls[side]["chat"] + "/v1/chat/completions"
            max_tokens = int(body["limits"]["max_tokens"])

            async def generate(messages: list[dict[str, Any]], seed: int) -> str:
                request = {
                    "model": side,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                }
                data = (await post(side, url, request)).json()
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise _ItemError("chat reply without choices[0].message.content") from None
                return content if isinstance(content, str) else ""  # null content: no action

            return {"transcript": await harness.run_episode(tracks.ENVS[track], body, generate)}

        async def run(side: str, case: dict[str, Any]) -> dict[str, Any]:
            item: dict[str, Any] = {"case_index": case["index"], "side": side}
            track = case.get("track", "decisions")
            try:
                if track in tracks.ENVS:
                    item.update(await play(side, track, case["body"]))
                else:
                    item.update(await read(side, case["body"]))
            except _ItemError as error:  # the model's side forfeits the case
                item["error"] = error.args[0]
            return item

        try:
            offset = 0
            while keep_going and offset < job["cases"]:
                page = await self.api.call(
                    "GET",
                    f"/v1/worker/jobs/{job['job']}/cases",
                    params={"lease": job["lease"], "offset": offset, "limit": PAGE},
                )
                cases = page.json()["cases"]
                if not cases:
                    break
                for case in cases:
                    cases_hash.update(case_line(case["body"]))
                fetched += len(cases)
                offset += len(cases)
                items = await asyncio.gather(*(run(side, c) for c in cases for side in SIDES))
                errors += sum("error" in item for item in items)
                for batch in _batches(items):
                    result = (
                        await self.api.call(
                            "POST",
                            f"/v1/worker/jobs/{job['job']}/answers",
                            json={"lease": job["lease"], "items": batch},
                        )
                    ).json()
                    if not result["continue"]:
                        keep_going = False
                        break
        finally:
            if self.inference is None:
                await client.aclose()
        return {"cases_fetched": fetched, "cases_sha256": cases_hash.hexdigest(), "errors": errors}

    async def run_forever(self, idle: float = 30.0) -> None:
        while True:
            try:
                ran = await self.run_once()
            except RuntimeError as error:
                print(f"worker: {error}", file=sys.stderr, flush=True)
                ran = False
            if not ran:
                await asyncio.sleep(idle)


def _batches(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split answers into request bodies under BATCH_BYTES."""
    batches: list[list[dict[str, Any]]] = [[]]
    size = 0
    for item in items:
        length = len(json.dumps(item, separators=(",", ":"))) + 1
        if batches[-1] and (size + length > BATCH_BYTES or len(batches[-1]) >= 2000):
            batches.append([])
            size = 0
        batches[-1].append(item)
        size += length
    return [batch for batch in batches if batch]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
