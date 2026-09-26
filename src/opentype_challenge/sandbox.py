"""Runtime-lane execution in fresh GPU sandboxes: a trusted bootstrap inside, a relay outside.

Every runtime run (stock fidelity, candidate fidelity, each B / C / B' of every block) starts
one fresh sandbox (Modal: gpu=B300, block_network, no secret, no OIDC token, the verified
champion snapshot mounted read-only) whose main process is `python -m opentype_challenge.sandbox
serve`, running as root. Its stdin and stdout are the only channel to the controller:

1. The controller writes one init line (side, allowlisted vllm flags, optional kernel).
2. The bootstrap measures the GPU identity (nvidia-smi) before any miner code exists in the
   sandbox, writes the kernel file root-owned and read-only, starts `vllm serve` as uid 65534
   and the pinned structured_server.py as uid 65533 (separate sessions, no new privileges,
   rlimits, stdin /dev/null, stdout and stderr to root-owned log files, no inherited fd), and
   reports {"identity"} then {"ready"} or {"failed"}.
3. It relays JSON-line requests {id, to: reader|chat, path, body} to 127.0.0.1 and answers
   {id, status, body}. No child ever holds the stdout pipe, so no child writes a frame.

The controller (SandboxLauncher) keeps the worker token, the case gold and every clock: it
serves the Worker through an httpx transport over that channel, so the Worker's own
time.monotonic() measures each request end to end, sandbox relay included (identically for
B, C and B'). The sandbox never sees gold or scores; the container scores every output.

What this does not prove: the isolation is Modal's gVisor sandbox plus Unix uids, not a formal
boundary; a kernel may still be approximate, which the fidelity reads and the timed answer
divergence (runtime.divergence) catch statistically, not absolutely.

`build`: the same bootstrap in a CPU-only sandbox (no GPU, no volume, no network) compiles
the kernel for the calibrated CUDA arch as uid 65534 with CPU, memory and time limits. A
compile failure is the candidate's; nothing compiled leaves the sandbox (the GPU run compiles
the same signed source again).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from . import pins, runtime
from .worker import (
    CANVAS,
    HEALTH_TIMEOUT,
    MAX_MODEL_LEN,
    STRUCTURED_SERVER,
    JobFailed,
    ServeFailed,
    VllmLauncher,
    scrubbed_env,
    sha256_file,
)

VLLM_UID = 10001  # dedicated: 65534 is nobody, the overflow uid of user namespaces
READER_UID = 10002
MODEL_MOUNT = "/model"
KERNEL_DIR = Path("/opt/opentype-kernel")
LOG_DIR = Path("/var/log/opentype")
MAX_FRAME = 8 * 1024 * 1024  # one relay line, either direction
MAX_BODY = 6 * 1024 * 1024  # one relayed response body
RELAY_WORKERS = runtime.MAX_CONCURRENCY
BUILD_SECONDS = 300
BUILD_MEMORY = 6 << 30  # RLIMIT_AS of the compile; the build sandbox gets 12 GiB
BUILD_FILE_BYTES = 256 << 20  # the compile writes a log and a Triton cache, nothing large
CHILD_FILE_BYTES = 8 << 30  # logs, Triton and torch caches of one run
PLUGIN = "opentype_kernel"  # the vllm.general_plugins entry point (kernel_slot.register)
PIECE = 12 * 1024  # bytes of a frame per physical line; base64 keeps a line under 16.1 KiB
LINE_MAX = 16 * 1024 + 64


# ---------------------------------------------------------------------------
# The wire: each logical frame (one JSON line, <= MAX_FRAME bytes) travels as physical lines
# "<seq> <more> <base64 piece>\n" of <= LINE_MAX bytes, numbered per direction. A dropped,
# duplicated, reordered or oversized line is detected (Modal's log path drops stdout over a
# rate limit), never silently misread.


def encode(frame: bytes, seq: int) -> tuple[bytes, int]:
    """Physical lines of one logical frame from sequence number `seq`; (lines, next seq)."""
    import base64

    pieces = [frame[i : i + PIECE] for i in range(0, len(frame), PIECE)] or [b""]
    out = []
    for index, piece in enumerate(pieces):
        more = int(index < len(pieces) - 1)
        out.append(b"%d %d %s\n" % (seq, more, base64.b64encode(piece)))
        seq += 1
    return b"".join(out), seq


class Reassembler:
    """Logical frames from physical lines, in order, each capped at MAX_FRAME."""

    def __init__(self) -> None:
        self.seq, self.parts, self.size = 0, list[bytes](), 0

    def feed(self, line: bytes) -> bytes | None:
        """The logical frame `line` completes, or None; ValueError on any protocol break."""
        import base64
        import binascii

        if len(line) > LINE_MAX:
            raise ValueError("a relay line exceeds its cap")
        try:
            fields = line.strip().split(b" ")
            if len(fields) == 2:
                fields.append(b"")  # an empty piece (strip removed its separator)
            seq, more, data = fields
            piece = base64.b64decode(data, validate=True)
            number, flag = int(seq), int(more)
        except (ValueError, binascii.Error):
            raise ValueError("a malformed relay line") from None
        if number != self.seq or flag not in (0, 1):
            raise ValueError(f"relay line {number} where {self.seq} was due")
        self.seq += 1
        self.size += len(piece)
        if self.size > MAX_FRAME:
            raise ValueError("a relay frame exceeds its cap")
        self.parts.append(piece)
        if flag:
            return None
        frame, self.parts, self.size = b"".join(self.parts), [], 0
        return frame


class _Inbox:
    """Logical frames from the bootstrap's stdin."""

    def __init__(self, stream: Any):
        self.stream, self.wire = stream, Reassembler()

    def read(self) -> bytes | None:
        """The next logical frame, or None at EOF; ValueError on a protocol break."""
        while True:
            line = self.stream.readline(LINE_MAX + 1)
            if not line:
                return None
            frame = self.wire.feed(line)
            if frame is not None:
                return frame


# ---------------------------------------------------------------------------
# Inside the sandbox (root).


def _cap(kind: int, want: int) -> None:
    """Lower a limit to `want`, never above the inherited hard limit (raising it fails once
    the child dropped root: CPython runs preexec_fn after setuid)."""
    hard = resource.getrlimit(kind)[1]
    value = want if hard == resource.RLIM_INFINITY else min(want, hard)
    resource.setrlimit(kind, (value, value))


def _limits(
    cpu_seconds: int | None, memory: int | None, file_bytes: int = CHILD_FILE_BYTES
) -> Callable[[], None]:
    """preexec: rlimits and no_new_privs for a demoted child (runs after fork, before exec)."""

    def apply() -> None:
        _cap(resource.RLIMIT_CORE, 0)
        _cap(resource.RLIMIT_FSIZE, file_bytes)
        _cap(resource.RLIMIT_NOFILE, 65536)
        _cap(resource.RLIMIT_NPROC, 8192)
        if cpu_seconds is not None:
            _cap(resource.RLIMIT_CPU, cpu_seconds)
        if memory is not None:
            _cap(resource.RLIMIT_AS, memory)
        import ctypes

        pr_set_no_new_privs = 38
        ctypes.CDLL(None, use_errno=True).prctl(pr_set_no_new_privs, 1, 0, 0, 0)

    return apply


def _home(uid: int | None, name: str) -> Path:
    home = Path("/tmp") / f"opentype-{name}"  # noqa: S108 - per-child private home
    home.mkdir(mode=0o700, exist_ok=True)
    if uid is not None:
        os.chown(home, uid, uid)
    return home


def _spawn(
    command: Sequence[str],
    uid: int | None,
    name: str,
    extra_env: Mapping[str, str],
    log_dir: Path,
    cpu_seconds: int | None = None,
    memory: int | None = None,
    file_bytes: int = CHILD_FILE_BYTES,
) -> subprocess.Popen[bytes]:
    """A child as `uid` (None: unchanged, tests only) with a private home, its output in a
    root-owned log file and nothing inherited but that file."""
    home = _home(uid, name)
    env = {
        **scrubbed_env(),
        "HOME": str(home),
        "TMPDIR": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "TRITON_CACHE_DIR": str(home / ".triton"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "VLLM_PLUGINS": PLUGIN,  # only ours; it is a no-op without a kernel
        **extra_env,
    }
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = (log_dir / f"{name}.log").open("wb")
    demote: dict[str, Any] = {}
    if uid is not None:
        demote = {"user": uid, "group": uid, "extra_groups": [], "umask": 0o077}
    try:
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=home,
            close_fds=True,
            start_new_session=True,
            preexec_fn=_limits(cpu_seconds, memory, file_bytes),  # noqa: PLW1509 - single-threaded here
            **demote,
        )
    finally:
        log.close()


def _stop(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)


def _nvidia(query: str) -> list[str]:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def identity(reader: Path) -> dict[str, Any]:
    """What this sandbox runs on, read before any miner code exists in it."""
    import importlib.metadata

    rows = [
        [c.strip() for c in r.split(",")] for r in _nvidia("name,driver_version,compute_cap,uuid")
    ]
    try:
        image = json.loads(Path("/opt/opentype/build.json").read_text())["vllm_image"]
    except (OSError, ValueError, KeyError, TypeError):
        image = None
    try:
        version: str | None = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "gpu": ",".join(sorted({r[0] for r in rows})) if rows else None,
        "driver": ",".join(sorted({r[1] for r in rows})) if rows else None,
        "compute_cap": ",".join(sorted({r[2] for r in rows})) if rows else None,
        "gpu_uuids": sorted(r[3] for r in rows if len(r) > 3),
        "vllm_image": image,
        "vllm_version": version,
        "structured_server_sha256": sha256_file(reader) if reader.exists() else None,
        "euid": os.geteuid(),
    }


def write_kernel(kernel: Mapping[str, Any], directory: Path) -> Path:
    """The signed source, root-owned and read-only, at a path no child can write."""
    source = str(kernel["source"]).encode()
    if hashlib.sha256(source).hexdigest() != kernel["sha256"]:
        raise ValueError("the kernel source does not match its sha256")
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    path = directory / "kernel.py"
    with contextlib.suppress(FileNotFoundError):
        path.chmod(0o600)
    path.write_bytes(source)
    path.chmod(0o444)
    return path


def _frame(frame: Mapping[str, Any]) -> bytes:
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


class _Out:
    """The one writer of the relay stdout."""

    def __init__(self, stream: Any):
        self.stream, self.lock, self.seq = stream, threading.Lock(), 0

    def __call__(self, frame: Mapping[str, Any]) -> None:
        line = _frame(frame)
        with self.lock:
            wire, self.seq = encode(line, self.seq)
            self.stream.write(wire)
            self.stream.flush()


def _wait_health(url: str, processes: Sequence[subprocess.Popen[bytes]], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            return f"a serving process exited before {url} was healthy"
        with contextlib.suppress(OSError, urllib.error.URLError):
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - loopback
                if response.status == 200:
                    return ""
        time.sleep(0.5)
    return f"{url} never became healthy"


def _forward(base: str, request: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    """One relayed POST to a loopback server; the body comes back as parsed JSON."""
    data = json.dumps(request["body"]).encode()
    http = urllib.request.Request(  # noqa: S310 - loopback only, path from the controller
        base + str(request["path"]),
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:  # noqa: S310
            status, raw = response.status, response.read(MAX_BODY + 1)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(MAX_BODY + 1)
    except (OSError, urllib.error.URLError) as error:  # refused, reset, timed out: a crash or
        # a hang, which a fresh placement's hardware can cause as well as the served code
        return {"id": request["id"], "status": 599, "body": {"error": repr(error)[:300]}}
    if len(raw) > MAX_BODY:
        return _upstream(request, "response too large")
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        return _upstream(request, "not json")
    reply = {"id": request["id"], "status": status, "body": body}
    try:
        if len(_frame(reply)) > MAX_FRAME:
            return _upstream(request, "response too large")
    except (ValueError, RecursionError):
        return _upstream(request, "not json")
    return reply


def _upstream(request: Mapping[str, Any], error: str) -> dict[str, Any]:
    """The served process answered, with bytes the relay refuses (not JSON, too large). Only
    this is attributed to what it serves; a 5xx is not (the pinned reader turns a dead or
    failing vllm into a 500/502, and a GPU fault can make a live vllm answer 500)."""
    return {"id": request["id"], "status": 599, "origin": "upstream", "body": {"error": error}}


def serve(
    stdin: Any,
    stdout: Any,
    *,
    vllm: Sequence[str] = ("vllm",),
    reader: Path = STRUCTURED_SERVER,
    demote: bool = True,
    kernel_dir: Path = KERNEL_DIR,
    log_dir: Path = LOG_DIR,
    health_timeout: float = HEALTH_TIMEOUT,
    port_base: int = 8100,
) -> int:
    """The serve bootstrap. demote=False, a fake vllm and port_base are for local tests only."""
    out, inbox = _Out(stdout), _Inbox(stdin)
    try:
        init = json.loads(inbox.read() or b"")
        side, argv = str(init["side"]), [str(a) for a in init["argv"]]
        share = float(init["share"])
    except (ValueError, KeyError, TypeError, RecursionError):
        out({"failed": "malformed init"})
        return 2
    out({"identity": identity(reader)})  # before any miner code exists here
    if demote and os.geteuid() != 0:
        out({"failed": "the bootstrap must run as root to demote its children"})
        return 2
    kernel_env: dict[str, str] = {}
    if init.get("kernel"):
        try:
            path = write_kernel(init["kernel"], kernel_dir)
        except (ValueError, KeyError, TypeError, OSError) as error:
            out({"failed": f"kernel: {error}"})
            return 2
        kernel_env = {"OPENTYPE_KERNEL_FILE": str(path), "OPENTYPE_KERNEL_SHA256": path_sha(path)}
    launcher = VllmLauncher(
        canvas=int(init.get("canvas", CANVAS)),
        max_model_len=int(init.get("max_model_len", MAX_MODEL_LEN)),
        vllm=tuple(vllm),
        reader=reader,
        port_base=port_base,
    )
    serve_cmd, reader_cmd = launcher.commands(side, Path(str(init["model"])), argv, share)
    vllm_port, reader_port = launcher.ports(side)
    processes: list[subprocess.Popen[bytes]] = []
    try:
        # The pinned reader binds its port before any miner code runs, so the vllm process
        # (which loads the kernel) can never listen in its place.
        try:
            processes.append(
                _spawn(reader_cmd, READER_UID if demote else None, "reader", {}, log_dir)
            )
            problem = _wait_health(
                f"http://127.0.0.1:{reader_port}/health", processes, health_timeout
            )
            if not problem:
                processes.append(
                    _spawn(serve_cmd, VLLM_UID if demote else None, "vllm", kernel_env, log_dir)
                )
                problem = _wait_health(
                    f"http://127.0.0.1:{vllm_port}/health", processes, health_timeout
                )
        except (OSError, subprocess.SubprocessError) as error:
            problem = f"spawn: {error!r}"[:300]
        if problem:
            out({"failed": problem, "logs": _tails(log_dir)})
            return 1
        out({"ready": True})
        bases = {
            "chat": f"http://127.0.0.1:{vllm_port}",
            "reader": f"http://127.0.0.1:{reader_port}",
        }
        names = ("reader", "vllm")
        with ThreadPoolExecutor(RELAY_WORKERS) as pool:
            while True:
                try:
                    line = inbox.read()
                except ValueError as error:
                    out({"failed": f"relay: {error}"})
                    break
                if line is None:
                    break  # the controller closed the channel
                gone = [n for n, p in zip(names, processes, strict=True) if p.poll() is not None]
                if gone:  # nothing is relayed once a served process is gone
                    out({"exited": gone[0], "logs": _tails(log_dir)})
                    break
                try:
                    request = json.loads(line)
                    base = bases[request["to"]]
                    timeout = float(request.get("timeout", 300))
                except (ValueError, KeyError, TypeError, RecursionError):
                    out({"failed": "malformed request"})
                    break
                pool.submit(_relay, out, base, request, timeout, tuple(processes))
    finally:
        _stop(processes)
    return 0


LOG_TAIL = 4000  # bytes of each child's log in a failure frame (operator diagnosis only)


def _tail(path: Path, size: int) -> str:
    """The last `size` bytes of a child's log, never the whole file (a child may fill it)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - size))
            return handle.read(size).decode(errors="replace")
    except OSError:
        return ""


def _tails(log_dir: Path) -> dict[str, str]:
    return {name: _tail(log_dir / f"{name}.log", LOG_TAIL) for name in ("reader", "vllm")}


EXIT_GRACE = 1.0  # seconds a reply waits for a served process that may be dying


def _relay(
    out: _Out,
    base: str,
    request: Mapping[str, Any],
    timeout: float,
    processes: Sequence[subprocess.Popen[bytes]],
) -> None:
    """One request, always answered. A reply counts only if every served process is still
    alive once it is received: a live reader still holds its port (bound before any miner
    code ran; another uid cannot share it), so a reader reply came from it; and a reply given
    while vllm died (the reader masks a dead upstream as a 500/502) is the placement's
    question, not an answer. Such a reply becomes a transport failure (retried)."""
    try:
        reply = _forward(base, request, timeout)
    except Exception as error:  # noqa: BLE001 - every request gets an answer
        reply = {"id": request["id"], "status": 599, "body": {"error": f"relay: {error!r}"[:300]}}
    if int(reply.get("status", 599)) >= 500 or reply.get("origin"):
        with contextlib.suppress(subprocess.TimeoutExpired):
            processes[-1].wait(timeout=EXIT_GRACE)  # an engine in the middle of dying
    if any(p.poll() is not None for p in processes):
        reply = {"id": request["id"], "status": 599, "body": {"error": "a served process exited"}}
    out(reply)


def path_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(stdin: Any, stdout: Any, *, demote: bool = True, kernel_dir: Path = KERNEL_DIR) -> int:
    """The build bootstrap: compile the kernel for one arch as an unprivileged, capped child."""
    out = _Out(stdout)
    try:
        init = json.loads(_Inbox(stdin).read() or b"")
        path = write_kernel(init["kernel"], kernel_dir)
        arch = int(init["arch"])
    except (ValueError, KeyError, TypeError, OSError, RecursionError) as error:
        out({"failed": f"malformed build: {error}"})
        return 2
    if demote and os.geteuid() != 0:
        out({"failed": "the bootstrap must run as root to demote its children"})
        return 2
    command = [sys.executable, "-m", "opentype_challenge.kernel_slot", str(path), path_sha(path)]
    logs = kernel_dir / "logs"
    try:
        child = _spawn(
            [*command, str(arch)],
            VLLM_UID if demote else None,
            "build",
            {},
            logs,
            cpu_seconds=BUILD_SECONDS,
            memory=BUILD_MEMORY,
            file_bytes=BUILD_FILE_BYTES,
        )
    except (OSError, subprocess.SubprocessError) as error:
        out({"failed": f"spawn: {error!r}"[:300]})
        return 2
    try:
        code = child.wait(timeout=BUILD_SECONDS + 30)
    except subprocess.TimeoutExpired:
        _stop([child])
        out({"build_failed": "the compile timed out"})
        return 1
    tail = _tail(logs / "build.log", 2000)
    if code != 0:
        out({"build_failed": tail})
        return 1
    out({"built": tail.strip().splitlines()[-1] if tail.strip() else ""})
    return 0


def main(argv: Sequence[str]) -> int:
    mode = argv[0] if argv else ""
    if mode == "serve":
        return serve(sys.stdin.buffer, sys.stdout.buffer)
    if mode == "build":
        return build(sys.stdin.buffer, sys.stdout.buffer)
    print("usage: python -m opentype_challenge.sandbox serve|build", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# The controller side (trusted, holds the token; never imports miner code).


class Channel(Protocol):
    async def send(self, line: str) -> None: ...

    def chunks(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> bool:
        """Stop the sandbox and wait for it; True once it is gone."""
        ...


class Backend(Protocol):
    gpu: str

    def model_path(self, local: Path) -> str:
        """Where the controller's verified model directory appears inside a sandbox."""
        ...

    async def start(self, mode: str, model: Path | None) -> Channel: ...


async def _lines(channel: Channel) -> AsyncIterator[dict[str, Any]]:
    """JSON frames from a sandbox's stdout (the wire's lines reassembled, each frame capped at
    MAX_FRAME); a lost or malformed line fails the channel."""
    buffer, wire = b"", Reassembler()
    async for chunk in channel.chunks():
        buffer += chunk.encode() if isinstance(chunk, str) else chunk
        if len(buffer) > LINE_MAX and b"\n" not in buffer:
            raise JobFailed("a sandbox relay line exceeded its cap", retry=True)
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            try:
                raw = wire.feed(line)
                frame = json.loads(raw) if raw is not None else None
            except (ValueError, RecursionError) as error:
                raise JobFailed(f"a sandbox broke the relay: {error}", retry=True) from None
            if raw is None:
                continue
            if not isinstance(frame, dict):
                raise JobFailed("a sandbox wrote a malformed frame", retry=True)
            yield frame


class _Sender:
    """Logical frames onto a channel, numbered for the sandbox's reassembler."""

    def __init__(self, channel: Channel):
        self.channel, self.seq = channel, 0

    async def __call__(self, frame: Mapping[str, Any]) -> None:
        wire, self.seq = encode(_frame(frame), self.seq)
        await self.channel.send(wire.decode())


@dataclass
class _Live:
    side: str
    channel: Channel
    frames: AsyncIterator[dict[str, Any]]
    pending: dict[int, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    pump: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dead: str = ""
    send: _Sender | None = None


class RelayTransport(httpx.AsyncBaseTransport):
    """http://<side>.<reader|chat>/<path> requests over a live sandbox's channel."""

    def __init__(self, launcher: SandboxLauncher):
        self.launcher = launcher
        self.next = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        side, _, to = request.url.host.partition(".")
        live = self.launcher.live.get(side)
        if live is None or to not in ("reader", "chat") or live.dead:
            raise httpx.ConnectError(f"no live sandbox for {side}", request=request)
        timeout = (request.extensions.get("timeout") or {}).get("read") or 300.0
        self.next += 1
        number = self.next
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        live.pending[number] = future
        body = json.loads(await request.aread() or b"null")
        frame = {"id": number, "to": to, "path": request.url.path, "body": body, "timeout": timeout}
        try:
            async with live.lock:
                assert live.send is not None
                await live.send(frame)
            reply = await asyncio.wait_for(future, timeout + 5)
        except TimeoutError:
            raise httpx.ReadTimeout("the sandbox did not answer", request=request) from None
        finally:
            live.pending.pop(number, None)
        status = int(reply["status"])
        if reply.get("origin") == "upstream":
            # the served process answered bytes the relay refuses (not JSON, too large)
            self.launcher.faults.append({"side": side, "body": reply.get("body")})
            return httpx.Response(502, json=reply.get("body"), request=request)
        if status == 599:  # no answer: a crash, a hang, a dead process or a lost channel
            raise httpx.ConnectError(str(reply.get("body")), request=request)
        # a 5xx stays ambiguous (hardware or the served code): the worker retries it
        return httpx.Response(status, json=reply.get("body"), request=request)


@dataclass
class SandboxLauncher:
    """The Worker's Launcher for the runtime lane: each launch is one fresh sandbox per side."""

    backend: Backend
    canvas: int = CANVAS
    max_model_len: int = MAX_MODEL_LEN
    dtype: str = "bfloat16"
    ready_timeout: float = HEALTH_TIMEOUT + 600
    live: dict[str, _Live] = field(default_factory=dict)
    placements: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)  # never sent to the API
    faults: list[dict[str, Any]] = field(default_factory=list)  # content faults, this launch
    _profile: dict[str, Any] | None = None
    _open: int = 0
    sandboxed = True  # one fresh sandbox per side and launch (Worker._duel_sandboxed)

    def evidence(self) -> dict[str, Any]:
        return {
            "executor": "modal-sandbox",
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
        }

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=RelayTransport(self))

    def urls(self, side: str) -> dict[str, str]:
        return {"reader": f"http://{side}.reader", "chat": f"http://{side}.chat"}

    def profile(self) -> dict[str, Any]:
        """The profile of the latest launch: what the bootstrap measured before any miner code,
        the flags this controller sent and the verified weights it mounted."""
        if self._profile is None:
            raise JobFailed("no sandbox has reported its identity yet", retry=True)
        return dict(self._profile)

    def content_fault(self, side: str) -> dict[str, Any] | None:
        """The first answer a server of `side` gave since its launch that the relay refused
        (not JSON, too large) while every served process stayed alive: evidence against what
        it serves. A 5xx, a crash, a hang or a lost channel is not: a fresh placement's
        hardware causes those too."""
        return next((f for f in self.faults if f["side"] == side), None)

    def quiescent(self) -> bool:
        """Every sandbox this launcher started is gone (a fresh one serves each run)."""
        return self._open == 0

    def serving(self) -> bool:
        """A sandbox may still mount a model directory (one did not stop)."""
        return self._open != 0

    async def build(self, kernel: Mapping[str, Any], arch: int) -> str:
        channel = await self.backend.start("build", None)
        try:
            await _Sender(channel)({"kernel": dict(kernel), "arch": arch})
            async with asyncio.timeout(BUILD_SECONDS + 300):
                async for frame in _lines(channel):
                    if "built" in frame:
                        return str(frame["built"])[:200]
                    if "build_failed" in frame:
                        raise JobFailed(
                            f"the kernel failed to compile: {str(frame['build_failed'])[-500:]}",
                            retry=False,
                        )
                    if "failed" in frame:
                        raise JobFailed(f"build sandbox: {frame['failed']}", retry=True)
        except TimeoutError:
            raise JobFailed("the build sandbox did not answer", retry=True) from None
        finally:
            await channel.close()
        raise JobFailed("the build sandbox exited without a result", retry=True)

    @asynccontextmanager
    async def __call__(
        self,
        models: Mapping[str, Path],
        extra: Mapping[str, Sequence[str]] | None = None,
        share: float | None = None,
        kernel: Mapping[str, Mapping[str, Any] | None] | None = None,
    ) -> AsyncIterator[dict[str, dict[str, str]]]:
        started: list[_Live] = []
        try:
            for side, model in models.items():
                argv = list((extra or {}).get(side, ()))
                started.append(
                    await self._start(side, model, argv, share, (kernel or {}).get(side))
                )
            yield {side: self.urls(side) for side in models}
        finally:
            for live in started:
                await self._finish(live)

    async def _start(
        self,
        side: str,
        model: Path,
        argv: list[str],
        share: float | None,
        kernel: Mapping[str, Any] | None,
    ) -> _Live:
        weights = weights_identity(model)  # the controller's own copy: verified, never mounted rw
        self.faults = [f for f in self.faults if f["side"] != side]
        channel = await self.backend.start("serve", model)
        self._open += 1
        live = _Live(side, channel, _lines(channel))
        live.send = _Sender(channel)
        self.live[side] = live
        init = {
            "side": side,
            "model": self.backend.model_path(model),
            "argv": argv,
            "share": share if share is not None else 0.45,
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
            "kernel": dict(kernel) if kernel else None,
        }
        try:
            await live.send(init)
            measured: dict[str, Any] | None = None
            async with asyncio.timeout(self.ready_timeout):
                async for frame in live.frames:
                    if "identity" in frame and measured is None:
                        measured = frame["identity"] if isinstance(frame["identity"], dict) else {}
                    elif "ready" in frame and measured is not None:
                        break
                    elif "failed" in frame:
                        self.failures.append({"side": side, **frame})  # bounded by MAX_FRAME
                        raise ServeFailed(f"sandbox: {str(frame['failed'])[:300]}", side)
                    else:
                        raise JobFailed("a sandbox broke the start protocol", retry=True)
                else:
                    raise ServeFailed("the sandbox exited before it was ready", side)
        except TimeoutError:
            await self._finish(live)
            raise ServeFailed("the sandbox never became ready", side) from None
        except BaseException:
            await self._finish(live)
            raise
        self._profile = self._measured_profile(measured, argv, share, weights)
        self.placements.append(
            {
                "side": side,
                "gpu_uuids": measured.get("gpu_uuids"),
                "kernel": runtime.kernel_ref(kernel),
            }
        )
        live.pump = asyncio.create_task(self._pump(live))
        return live

    def _measured_profile(
        self,
        measured: Mapping[str, Any],
        argv: Sequence[str],
        share: float | None,
        weights: Mapping[str, str | None],
    ) -> dict[str, Any]:
        def flag(name: str) -> str | None:
            return argv[argv.index(name) + 1] if name in argv[:-1] else None

        return {
            "vllm_image": measured.get("vllm_image"),
            "vllm_version": measured.get("vllm_version"),
            "structured_server_sha256": measured.get("structured_server_sha256"),
            "base": f"{pins.BASE_REPO}@{pins.BASE_REVISION}",
            **weights,
            "dtype": self.dtype,
            "kv_cache_dtype": flag("--kv-cache-dtype"),
            "attention_backend": flag("--attention-backend"),
            "moe_backend": flag("--moe-backend"),
            "canvas": self.canvas,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": share,
            "executor": "modal-sandbox",
            "gpu_type": self.backend.gpu,
            "gpu": measured.get("gpu"),
            "driver": measured.get("driver"),
            "compute_cap": measured.get("compute_cap"),
        }

    async def _pump(self, live: _Live) -> None:
        """Route each reply frame to its request; any protocol break ends the sandbox's use."""
        try:
            async for frame in live.frames:
                number = frame.get("id")
                future = live.pending.get(number) if isinstance(number, int) else None
                if "exited" in frame:
                    self.failures.append({"side": live.side, **frame})
                    live.dead = f"the {frame['exited']} process exited"
                    break
                if future is None or "status" not in frame:
                    live.dead = "an unexpected frame"
                    break
                if not future.done():
                    future.set_result(frame)
            else:
                live.dead = live.dead or "the sandbox closed its channel"
        except JobFailed as error:
            live.dead = error.reason
        except Exception as error:  # noqa: BLE001 - the channel broke: fail every waiter
            live.dead = repr(error)[:300]
        for future in live.pending.values():
            if not future.done():
                future.set_result({"status": 599, "body": {"error": live.dead}})

    async def _finish(self, live: _Live) -> None:
        if live.pump is not None:
            live.pump.cancel()
            with contextlib.suppress(BaseException):
                await live.pump
        with contextlib.suppress(BaseException):
            await live.frames.aclose()  # type: ignore[attr-defined]  # an async generator
        gone = await live.channel.close()
        if self.live.get(live.side) is live:
            del self.live[live.side]
        if gone:
            self._open -= 1


def weights_identity(model: Path) -> dict[str, str | None]:
    """The served weights as the profile names them: NVFP4 only when both the config and the
    tensor layout of every shard match the pinned export."""
    config = model / "config.json"
    config_sha = sha256_file(config) if config.exists() else None
    schema = tensor_schema(model)
    nvfp4 = config_sha == pins.NVFP4_CONFIG_SHA256 and schema == pins.NVFP4_SCHEMA_SHA256
    return {
        "weights": "modelopt-nvfp4" if nvfp4 else "unverified",
        "weights_config_sha256": config_sha,
        "weights_schema_sha256": schema,
    }


def stage_nvfp4(directory: Path, fetch: Any) -> Path:
    """The pinned NVFP4 export plus the base's support files in `directory`, every file
    sha256-verified (the export's own chat template differs from the base's, so the base's
    is used, as for any champion). Refuses unless the result verifies as modelopt-nvfp4."""
    from .worker import assemble, base_snapshot

    base = base_snapshot(directory.parent / "base", fetch)
    manifest = {"repo": pins.NVFP4_REPO, "revision": pins.NVFP4_REVISION, "files": pins.NVFP4_FILES}
    assemble(manifest, base, directory, fetch, pins.NVFP4_CONFIG_SHA256)
    if weights_identity(directory)["weights"] != "modelopt-nvfp4":
        raise JobFailed("the staged snapshot does not verify as the pinned NVFP4 export", False)
    return directory


MAX_HEADER = 100 * 1024 * 1024


def tensor_schema(model: Path) -> str | None:
    """sha256 of the canonical {tensor: [dtype, shape]} over the index's shards (headers only)."""
    try:
        index = json.loads((model / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        schema: dict[str, list[Any]] = {}
        for shard in shards:
            if "/" in shard or shard.startswith("."):
                return None
            with (model / shard).open("rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                if not 0 < size <= MAX_HEADER:
                    return None
                header = json.loads(handle.read(size))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if name in schema:
                    return None
                schema[name] = [meta["dtype"], meta["shape"]]
        if set(schema) != set(index["weight_map"]):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    text = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Backends.


@dataclass
class ModalBackend:
    """Fresh Modal sandboxes: B300, network blocked, no secret, no OIDC token, the model
    directory mounted read-only from the controller's volume (only that directory)."""

    app: Any
    image: Any
    volume: Any
    root: Path  # where the controller mounts `volume`
    gpu: str = runtime.GPU_TYPE
    timeout: int = 6 * 3600
    cpu: tuple[float, float] = (8.0, 16.0)
    memory: tuple[int, int] = (65536, 131072)
    # the controller writes the model directories into `volume` (its only writer) and commits
    # them before each sandbox mounts them read-only; False for a volume staged in advance
    commit: bool = False
    # "exec": the bootstrap runs as an exec'd process whose stdio goes through the task
    # command router; "entrypoint": it is the sandbox's own command, whose stdout Modal serves
    # from its log pipeline (rate limited: it drops output; kept for the relay probe only)
    stdio: str = "exec"

    def model_path(self, local: Path) -> str:
        return MODEL_MOUNT

    async def start(self, mode: str, model: Path | None) -> Channel:
        import modal
        from modal.stream_type import StreamType

        volumes = {}
        if model is not None:
            sub = await asyncio.to_thread(lambda: model.resolve().relative_to(self.root.resolve()))
            # staging commits the snapshot once (stage_nvfp4); a read-only controller mount
            # never writes. A worker's work volume is committed here, between its downloads
            # and any mount: nothing writes it while a sandbox serves.
            if self.commit:
                await self.volume.commit.aio()
            volumes = {
                MODEL_MOUNT: self.volume.with_mount_options(read_only=True, sub_path=str(sub))
            }
        command = ("python3", "-m", "opentype_challenge.sandbox", mode)
        entry = command if self.stdio == "entrypoint" else ("sleep", "infinity")
        sandbox = await modal.Sandbox.create.aio(
            *entry,
            app=self.app,
            image=self.image,
            gpu=self.gpu if mode == "serve" else None,
            cpu=self.cpu if mode == "serve" else (2.0, 4.0),
            memory=self.memory if mode == "serve" else (8192, 12288),
            block_network=True,
            secrets=[],
            include_oidc_identity_token=False,
            volumes=volumes,
            timeout=self.timeout if mode == "serve" else BUILD_SECONDS + 600,
        )
        if self.stdio == "entrypoint":
            return _ModalChannel(sandbox, sandbox)
        try:
            # stderr is discarded: an unread pipe would fill; the bootstrap reports on stdout
            process = await sandbox.exec.aio(*command, text=False, stderr=StreamType.DEVNULL)
        except BaseException:
            with contextlib.suppress(Exception):
                await sandbox.terminate.aio()
            raise
        return _ModalChannel(sandbox, process)


STDIN_CHUNK = 1 << 20  # Modal buffers at most 2 MiB of sandbox stdin between drains


async def _send_chunked(stdin: Any, line: str) -> None:
    """One line as bytes in chunks under the SDK's write buffer, each drained before the next
    (the reader reassembles bytes, so a chunk may end inside a UTF-8 character)."""
    data = line.encode()
    for start in range(0, len(data), STDIN_CHUNK):
        stdin.write(data[start : start + STDIN_CHUNK])
        await stdin.drain.aio()


class _ModalChannel:
    """stdio of `process` (the sandbox itself, or a process exec'd in it)."""

    def __init__(self, sandbox: Any, process: Any):
        self.sandbox, self.process = sandbox, process

    async def send(self, line: str) -> None:
        await _send_chunked(self.process.stdin, line)

    async def chunks(self) -> AsyncIterator[bytes]:
        async for chunk in self.process.stdout:
            yield chunk.encode() if isinstance(chunk, str) else chunk

    async def close(self) -> bool:
        with contextlib.suppress(Exception):
            self.process.stdin.write_eof()
        try:
            await self.sandbox.terminate.aio(wait=True)
        except Exception:  # noqa: BLE001 - reported as not quiescent
            return False
        return True


@dataclass
class ProcessBackend:
    """A local bootstrap process with no privilege drop: tests and CPU smoke runs only; it
    isolates nothing and a calibration never names it (executor stays modal-sandbox)."""

    command: Sequence[str]  # e.g. [python, -c, "...sandbox.serve(..., demote=False)"]
    build_command: Sequence[str] = ()
    gpu: str = "none"

    def model_path(self, local: Path) -> str:
        return str(local)

    async def start(self, mode: str, model: Path | None) -> Channel:
        command = self.command if mode == "serve" else self.build_command
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=MAX_FRAME * 2,
            start_new_session=True,
        )
        return _ProcessChannel(process)


class _ProcessChannel:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    async def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line.encode())
        await self.process.stdin.drain()

    async def chunks(self) -> AsyncIterator[bytes]:
        assert self.process.stdout is not None
        while chunk := await self.process.stdout.read(1 << 16):
            yield chunk

    async def close(self) -> bool:
        if self.process.stdin is not None:
            with contextlib.suppress(Exception):
                self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), 60)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()
        return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
