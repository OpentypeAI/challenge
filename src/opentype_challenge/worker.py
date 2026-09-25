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
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import __version__, harness, pins, runtime, tracks
from .bank import case_line
from .crypto import manifest_problem
from .generator import Case

SIDES = ("champion", "challenger")
BATCH_BYTES = 900 * 1024
# The container replays every harness transcript inline before it answers: 8 items x 12
# turns x <= 0.06 s per query (sqltask.MAX_STEPS) keeps one POST far below the 30 s proxy.
BATCH_HARNESS = 8
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
# Written by the Dockerfile's worker stage from its VLLM_IMAGE build argument. Self-reported
# by an operator-owned worker, not an attestation (docs/operator.md, runtime lane).
BUILD_MANIFEST = Path("/opt/opentype/build.json")


class JobFailed(Exception):
    """An error attributable to the challenger (no retry) or to infrastructure (retry)."""

    def __init__(self, reason: str, retry: bool):
        super().__init__(reason)
        self.reason, self.retry = reason, retry


class ServeFailed(JobFailed):
    """A server of one side never became healthy."""

    def __init__(self, reason: str, side: str):
        super().__init__(reason, retry=True)
        self.side = side


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
    manifest: Mapping[str, Any],
    base_dir: Path,
    directory: Path,
    fetch: Fetch,
    config_sha256: str | None = None,
) -> dict[str, str]:
    """Download a manifest into directory, verify every sha256, add the base support files.

    Only weights, the weight index and config.json come from the miner; config.json must be
    byte-equal to config_sha256 (the base revision's for a challenger; the champion's own,
    which the container accepted, for the champion), and tokenizer/chat template/processor
    files are copied from the verified base snapshot. Returns the resolved sha256 of every
    file.
    """
    files: dict[str, str] = dict(manifest["files"])
    problem = manifest_problem(files)
    if problem:
        raise JobFailed(f"{manifest['repo']}: {problem}", retry=False)
    if files["config.json"] != (config_sha256 or pins.BASE_FILES["config.json"]):
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
        self,
        models: Mapping[str, Path],
        extra: Mapping[str, Sequence[str]] | None = None,
        share: float | None = None,
    ) -> AbstractAsyncContextManager[dict[str, dict[str, str]]]:
        """Serve each side's model; yield {side: {"reader": systemone URL, "chat": vllm URL}}.
        extra: allowlisted vllm flags per side; share: GPU memory share of each server."""
        ...

    def evidence(self) -> dict[str, Any]:
        """What serves the reads: versions and file digests."""
        ...

    def profile(self) -> dict[str, Any]:
        """The serving profile a runtime measurement runs under, gpu and driver included."""
        ...

    def quiescent(self) -> bool:
        """Every serving process is gone and no process holds the GPU."""
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
    dtype: str = "bfloat16"
    _live: list[subprocess.Popen[bytes]] = field(default_factory=list, repr=False)

    def evidence(self) -> dict[str, Any]:
        return {
            "vllm_image": pins.VLLM_IMAGE,
            "vllm_version": _package_version("vllm"),
            "structured_server_sha256": sha256_file(self.reader),
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
        }

    def profile(self) -> dict[str, Any]:
        """What this host actually runs: the image named by the baked build manifest, the
        installed vllm, the reader's hash, the launcher's own flags and the GPU. Refuses the
        job (infrastructure) when any of it cannot be read."""
        gpu, driver = _gpu_identity()
        try:
            image = json.loads(BUILD_MANIFEST.read_text())["vllm_image"]
        except (OSError, ValueError, KeyError, TypeError):
            image = None
        profile = {
            "vllm_image": image,
            "vllm_version": _package_version("vllm"),
            "structured_server_sha256": sha256_file(self.reader) if self.reader.exists() else None,
            "base": f"{pins.BASE_REPO}@{pins.BASE_REVISION}",
            "dtype": self.dtype,
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": runtime.PROFILE_FIXED["gpu_memory_utilization"],
            "gpu": gpu,
            "driver": driver,
            # never the calibrated "modal-sandbox": miner code does not run on this host
            "executor": "local-process",
        }
        missing = sorted(k for k, v in profile.items() if not v)
        if missing:
            raise JobFailed(f"cannot read this worker's serving profile: {missing}", retry=True)
        return profile

    def quiescent(self) -> bool:
        return not self._live and _gpu_idle()

    def ports(self, side: str) -> tuple[int, int]:
        """(vllm port, structured server port) of one side."""
        offset = SIDES.index(side)
        return self.port_base + offset, self.port_base + 10 + offset

    def commands(
        self, side: str, model: Path, extra: Sequence[str] = (), share: float | None = None
    ) -> list[list[str]]:
        """extra: allowlisted flags from runtime.options_argv, after the fixed ones."""
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
                self.dtype,
                "--gpu-memory-utilization",
                str(self.memory_share if share is None else share),
                "--diffusion-config",
                json.dumps({"canvas_length": self.canvas}),
                "--max-logprobs",
                "32",
                "--enable-prefix-caching",
                "--max-model-len",
                str(self.max_model_len),
                "--limit-mm-per-prompt",
                json.dumps({"image": 1}),
                *extra,
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
        self,
        models: Mapping[str, Path],
        extra: Mapping[str, Sequence[str]] | None = None,
        share: float | None = None,
        kernel: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, dict[str, str]]]:
        if kernel:  # this host holds the token and a writable workdir: miner code never runs
            raise JobFailed("a local launcher never runs a miner kernel", retry=True)
        processes: list[subprocess.Popen[bytes]] = []
        env = {**scrubbed_env(), "HF_HUB_OFFLINE": "1"}
        self._live = processes
        try:
            async with httpx.AsyncClient() as client:
                for side, model in models.items():
                    serve, reader = self.commands(side, model, (extra or {}).get(side, ()), share)
                    vllm_port, reader_port = self.ports(side)
                    try:
                        processes.append(self._spawn(serve, env, f"vllm-{side}"))
                        await _wait_health(
                            client, f"http://127.0.0.1:{vllm_port}/health", processes
                        )
                        processes.append(self._spawn(reader, env, f"reader-{side}"))
                        await _wait_health(
                            client, f"http://127.0.0.1:{reader_port}/health", processes
                        )
                    except JobFailed as error:
                        raise ServeFailed(error.reason, side) from None
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
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=30)
            # quiescent() reports a process that outlived SIGKILL
            self._live = [p for p in processes if p.poll() is None]

    def _spawn(self, command: list[str], env: dict[str, str], name: str) -> subprocess.Popen[bytes]:
        log = (self.log_dir / f"{name}.log").open("wb")
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")


def scrubbed_env() -> dict[str, str]:
    """The worker's environment without anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in SECRET_WORDS)}


def _nvidia_smi(*query: str) -> list[str] | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", *query, "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def _gpu_identity() -> tuple[str, str]:
    rows = _nvidia_smi("--query-gpu=name,driver_version")
    if not rows:
        return "", ""  # unreadable: profile() refuses the job
    names = {row.rsplit(",", 1)[0].strip() for row in rows}
    drivers = {row.rsplit(",", 1)[-1].strip() for row in rows}
    return ",".join(sorted(names)), ",".join(sorted(drivers))


def _gpu_idle() -> bool:
    """No compute process on any visible GPU; unknown (no nvidia-smi) is not idle."""
    rows = _nvidia_smi("--query-compute-apps=pid")
    return rows is not None and not rows


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


class ApiUnavailable(RuntimeError):
    """The API exhausted its bounded transport/service retries."""


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
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {path}: {response.status_code} {response.text[:300]}")
            return response
        raise ApiUnavailable(f"{method} {path}: the API stayed unavailable")


@dataclass
class Worker:
    api: Api
    workdir: Path
    launcher: Launcher
    fetch: Fetch = hf_fetch
    concurrency: int = CONCURRENCY
    inference: httpx.AsyncClient | None = None
    lane: str = "quality"

    async def run_once(self) -> bool:
        """Lease and run one job. False when the queue is empty."""
        params = {"lane": self.lane} if self.lane != "quality" else None
        response = await self.api.call("POST", "/v1/worker/lease", params=params)
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
            if job.get("lane", "quality") != self.lane:
                raise JobFailed(f"leased a {job.get('lane')} job on a {self.lane} worker", True)
            run = self._bench if self.lane == "runtime" else self._duel
            evidence.update(await run(job, job_dir, evidence))
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

    async def _bench(self, job: dict[str, Any], job_dir: Path, evidence: dict[str, Any]) -> dict:
        """A runtime job on the champion's weights: fidelity reads of stock vs candidate, then
        calibration.blocks blocks of B / C / B' measured one server at a time, with a
        quiescence check before and after each. Only allowlisted flags reach vllm. The stock
        or incumbent server always starts first: if it fails the host is at fault (retry);
        if only the candidate then fails to start, its options are (reject)."""
        spec = job["runtime"]
        cal = runtime.Calibration.from_json(spec["calibration"])
        base_flags = runtime.serving_argv(cal.profile)
        kernels = {
            "B": spec.get("incumbent_kernel"),
            "C": spec.get("candidate_kernel"),
        }
        flags = {
            s: [
                *base_flags,
                *runtime.options_argv(spec["incumbent" if s == "B" else "candidate"]),
                *runtime.kernel_argv(kernels[s]),
            ]
            for s in ("B", "C")
        }
        build = getattr(self.launcher, "build", None)
        if any(kernels.values()) and build is None:
            raise JobFailed("this worker cannot run kernels: no sandbox launcher", retry=True)
        base = await asyncio.to_thread(base_snapshot, self.workdir / "base", self.fetch)
        try:
            model, evidence["champion_files"] = await asyncio.to_thread(
                self._champion, job["champion"], base
            )
        except JobFailed as error:
            raise JobFailed(f"champion: {error.reason}", retry=True) from None
        if kernels["C"] is not None:
            assert build is not None
            arch = int(str(cal.profile["compute_cap"]).replace(".", ""))
            evidence["kernel_build"] = await build(kernels["C"], arch)  # fault: the candidate's
        share = float(cal.profile["gpu_memory_utilization"])
        placements: list[Any] = []

        def launch(side: str, served: str) -> Any:
            return self._launch(
                cal, model, served, flags[side] if side in flags else base_flags,
                kernels.get(side), share, placements,
            )  # fmt: skip

        # fidelity: stock ("champion", pristine flags, independent of B) then candidate
        # ("challenger"), each alone on the GPU at the calibrated share, like the timing
        self._require_quiescent("before stock fidelity")
        async with launch("stock", "champion") as urls:
            counts = await self._read(job, urls, ("champion",))
        self._require_quiescent("before candidate fidelity")
        try:
            async with launch("C", "challenger") as urls:
                candidate = await self._read(job, urls, ("challenger",))
        except ServeFailed as error:
            # stock served healthily alone just before, on the same pinned build
            raise _candidate_fault(error, "challenger") from None
        except JobFailed as error:
            raise self._content_fault(error, "challenger") from None
        counts["errors"] += candidate["errors"]
        blocks = []
        for number in range(cal.blocks):
            seconds, quiescent = {}, []
            for side in runtime.SIDES:
                self._require_quiescent(f"before {side}")
                try:
                    async with launch("C" if side == "C" else "B", "champion") as urls:
                        seconds[side], tasks = await self._measure(job, cal, urls["champion"])
                except ServeFailed as error:
                    # C's server runs as "champion"; B served healthily just before it
                    raise _candidate_fault(error, "champion" if side == "C" else None) from None
                except JobFailed as error:
                    if side != "C":
                        raise
                    raise self._content_fault(error, "champion") from None
                await self._post_timings(job, number, side, tasks)
                quiescent.append(self.launcher.quiescent())
                if not quiescent[-1]:
                    break
            blocks.append(
                {"order": list(runtime.SIDES), "seconds": seconds, "quiescent": quiescent}
            )
            if not all(quiescent):
                break  # reported as is: the verdict is NO_DECISION
        evidence["placements"] = placements
        return {**counts, "runtime": {"profile": cal.profile, "blocks": blocks}}

    def _content_fault(self, error: JobFailed, served: str) -> JobFailed:
        """A candidate run failing mid-run is the candidate's only on a content fault of its
        own server (a 5xx or a broken answer); a crash, a hang or a lost channel may be the
        fresh placement's, and retries (bounded by MAX_ATTEMPTS)."""
        fault = getattr(self.launcher, "content_fault", lambda _: None)(served)
        if not error.retry or fault is None:
            return error
        return JobFailed(f"the candidate's server gave a broken answer: {fault}"[:500], False)

    @asynccontextmanager
    async def _launch(
        self,
        cal: runtime.Calibration,
        model: Path,
        served: str,
        argv: Sequence[str],
        kernel: Mapping[str, Any] | None,
        share: float,
        placements: list[Any],
    ) -> AsyncIterator[dict[str, dict[str, str]]]:
        """One run: serve, then check the profile this very run measured (the bootstrap reads
        the GPU before any miner code) before a single case is sent; a mismatch is the host's."""
        kwargs = {"kernel": {served: kernel}} if kernel else {}
        async with self.launcher({served: model}, {served: list(argv)}, share, **kwargs) as urls:
            measured = self.launcher.profile()
            if measured != cal.profile:
                wrong = sorted(k for k in {*measured, *cal.profile}
                               if measured.get(k) != cal.profile.get(k))  # fmt: skip
                raise JobFailed(f"this run does not match the calibrated profile: {wrong}", True)
            placements.extend(getattr(self.launcher, "placements", [])[-1:])
            yield urls

    async def _post_timings(
        self, job: dict[str, Any], block: int, side: str, tasks: list[dict[str, Any]]
    ) -> None:
        """Raw outputs and latencies; the container scores them, the worker never does."""
        items = [{**t, "block": block, "side": side} for t in tasks]
        for batch in _batches(items):
            await self.api.call(
                "POST",
                f"/v1/worker/jobs/{job['job']}/timings",
                json={"lease": job["lease"], "items": batch},
            )

    def _require_quiescent(self, when: str) -> None:
        if not self.launcher.quiescent():
            raise JobFailed(f"the GPU is not quiescent {when}", retry=True)

    async def _measure(
        self, job: dict[str, Any], cal: runtime.Calibration, urls: Mapping[str, str]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Monotonic seconds per cell and every timed task's raw output and latency (ms from
        the worker's clock); cold cells first, a warm cell after one untimed pass."""
        client = self.inference or httpx.AsyncClient()
        seconds: dict[str, float] = {}
        tasks: list[dict[str, Any]] = []
        try:
            for name, cell in sorted(cal.cells.items(), key=lambda item: item[1].warm):
                cases = [
                    runtime.cell_case(job["runtime"]["seed"], name, cell, i)
                    for i in range(cell.cases)
                ]
                if cell.warm:
                    await self._timed(client, urls, cases, cell.concurrency)
                start = time.monotonic()
                results = await self._timed(client, urls, cases, cell.concurrency)
                seconds[name] = time.monotonic() - start
                tasks += [
                    {"cell": name, "case_index": i, "ms": ms, **item}
                    for i, (ms, item) in enumerate(results)
                ]
        finally:
            if self.inference is None:
                await client.aclose()
        return seconds, tasks

    async def _timed(
        self,
        client: httpx.AsyncClient,
        urls: Mapping[str, str],
        cases: Sequence[Case],
        concurrency: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        """(latency ms, output item) per task; the latency counts from the moment the task
        may run, so the server's own queueing is included. A harness task is a whole
        episode."""
        limit = asyncio.Semaphore(concurrency)

        async def call(url: str, body: Mapping[str, Any]) -> dict[str, Any] | None:
            try:
                response = await client.post(url, json=body, timeout=READ_TIMEOUT)
                data = response.json() if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                return None
            return data if isinstance(data, dict) else None

        async def task(case: Case) -> tuple[float, dict[str, Any]]:
            async with limit:
                start = time.monotonic()
                item = await self._task(call, urls, case)
                return (time.monotonic() - start) * 1000, item

        return list(await asyncio.gather(*(task(case) for case in cases)))

    @staticmethod
    async def _task(
        call: Callable[[str, Mapping[str, Any]], Awaitable[dict[str, Any] | None]],
        urls: Mapping[str, str],
        case: Case,
    ) -> dict[str, Any]:
        """The raw output item, as a duel reports it; never a verdict on it."""
        body = case.body
        if case.track not in tracks.ENVS:
            data = await call(urls["reader"] + "/v1/systemone", body)
            if data is None:
                return {"error": "the reader failed"}
            answers = data.get("answers")
            return {
                "answers": {q: slim(a) for q, a in answers.items()}
                if isinstance(answers, dict)
                else {},
                "reads": reads_of(data),
            }
        failed = False

        async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
            nonlocal failed
            request = {
                "model": "champion",
                "messages": messages,
                "max_tokens": int(body["limits"]["max_tokens"]),
            }
            data = await call(urls["chat"] + "/v1/chat/completions", request)
            try:
                content = data["choices"][0]["message"]["content"] if data else None
            except (KeyError, IndexError, TypeError):
                content = None
            if not isinstance(content, str):
                failed = True
                return ""
            return content

        transcript = await harness.run_episode(tracks.ENVS[case.track], body, generate)
        return {"error": "the chat server failed"} if failed else {"transcript": transcript}

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
        # the champion's config is the one the container crowned: the base's, or the pinned
        # NVFP4 export's once the champion migrated (nothing else ever becomes champion)
        allowed = {pins.BASE_FILES["config.json"], pins.NVFP4_CONFIG_SHA256}
        config = manifest["files"].get("config.json")
        if config not in allowed:
            raise JobFailed("the champion's config.json is neither the base's nor NVFP4's", True)
        resolved = assemble(manifest, base, directory, self.fetch, config)
        record.write_text(json.dumps(resolved, sort_keys=True))
        return directory, resolved

    async def _read(
        self,
        job: dict[str, Any],
        urls: Urls,
        sides: Sequence[str] = SIDES,
    ) -> dict[str, Any]:
        """Page through every case, run `sides` and post the answer items in batches. The
        container stores each side's results on their own, so sides may come in separate
        passes (the runtime fidelity serves one side at a time)."""
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

            async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
                request = {
                    "model": side,
                    "messages": messages,
                    "max_tokens": max_tokens,
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
                items = await asyncio.gather(*(run(side, c) for c in cases for side in sides))
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

    async def run_forever(self, idle: float = 30.0, until_empty: bool = False) -> None:
        """Drain the queue or poll forever. Scheduled drains fail after bounded API retries."""
        while True:
            try:
                ran = await self.run_once()
            except ApiUnavailable as error:
                if until_empty:
                    raise
                print(f"worker: {error}", file=sys.stderr, flush=True)
                await asyncio.sleep(idle)
                continue
            if not ran:
                if until_empty:
                    return
                await asyncio.sleep(idle)


def _candidate_fault(error: ServeFailed, candidate: str | None) -> JobFailed:
    """The candidate's server failing to start after a reference server was healthy on the
    same GPU is the candidate's fault; any other start failure is the host's (retry)."""
    if candidate is not None and error.side == candidate:
        return JobFailed(f"the candidate options failed to serve: {error.reason}", retry=False)
    return error


def _batches(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split answers into request bodies under BATCH_BYTES and BATCH_HARNESS transcripts."""
    batches: list[list[dict[str, Any]]] = [[]]
    size = replays = 0
    for item in items:
        length = len(json.dumps(item, separators=(",", ":"))) + 1
        harness_item = "transcript" in item
        if batches[-1] and (
            size + length > BATCH_BYTES
            or len(batches[-1]) >= 2000
            or (harness_item and replays >= BATCH_HARNESS)
        ):
            batches.append([])
            size = replays = 0
        batches[-1].append(item)
        size += length
        replays += harness_item
    return [batch for batch in batches if batch]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
