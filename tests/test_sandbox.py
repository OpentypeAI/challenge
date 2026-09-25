"""The runtime lane's sandbox protocol on CPU: the serve bootstrap, the relay and the launcher,
with the fake vllm and reader and no privilege drop (ProcessBackend isolates nothing; this
checks the protocol, not the isolation). Kernel admission, the NVFP4 identity and the timed
answer divergence are pure."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from opentype_challenge import crypto, pins, runtime, sandbox, tracks
from opentype_challenge.worker import JobFailed, ServeFailed

from .fake_inference import answer, blur

FAKE = Path(__file__).with_name("fake_inference.py")
KERNEL = """import triton
import triton.language as tl

EPS_FLOOR = 1e-12


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(var + eps) * w
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)
"""


def _port_base() -> int:
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port + 11 <= 65535:
            return port


HOSTILE = Path(__file__).with_name("hostile_vllm.py")


PROXY = Path(__file__).with_name("proxy_reader.py")


def _backend(tmp_path: Path, hostile: bool = False, proxy: bool = False) -> sandbox.ProcessBackend:
    """The serve bootstrap as a local process: the fake reader (which records its pid), or a
    reader proxying to its upstream as the pinned one does; the fake vllm or the hostile one."""
    base = _port_base()
    pid_file = tmp_path / "reader.pid"
    reader = tmp_path / "structured_server.py"
    future = "from __future__ import annotations\n"
    record = f"import os\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
    source = (PROXY if proxy else FAKE).read_text()
    reader.write_text(source.replace(future, future + record, 1))
    vllm = (
        (sys.executable, str(HOSTILE), str(pid_file), str(base + 10))
        if hostile
        else (sys.executable, str(FAKE))
    )
    code = (
        "import sys; from pathlib import Path; from opentype_challenge import sandbox; "
        "sys.exit(sandbox.serve(sys.stdin.buffer, sys.stdout.buffer, "
        f"vllm={vllm!r}, reader=Path({str(reader)!r}), "
        f"demote=False, kernel_dir=Path({str(tmp_path / 'k')!r}), "
        f"log_dir=Path({str(tmp_path / 'logs')!r}), health_timeout=30, "
        f"port_base={base}))"
    )
    return sandbox.ProcessBackend([sys.executable, "-c", code])


def _model(tmp_path: Path, skill: str = "exact") -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text(skill)
    return model


def test_serve_relays_reads_through_the_launcher(tmp_path):
    launcher = sandbox.SandboxLauncher(_backend(tmp_path), canvas=64, ready_timeout=60)
    model = _model(tmp_path)
    cell = runtime.Cell("decisions", 1, 1, 1000.0, 1.0, False)
    body = runtime.cell_case("seed", "short", cell, 0).body

    async def go() -> tuple[int, dict, dict]:
        argv = runtime.serving_argv({"moe_backend": "cutlass", **runtime.PROFILE_FIXED})
        async with launcher({"champion": model}, {"champion": argv}, 0.9) as urls:
            assert launcher.quiescent() is False
            profile = launcher.profile()
            async with launcher.client() as client:
                reply = await client.post(urls["champion"]["reader"] + "/v1/systemone", json=body)
                missing = await client.post(urls["champion"]["chat"] + "/v1/nothing", json={})
                assert missing.status_code == 404
        return reply.status_code, reply.json(), profile

    try:
        status, reply, profile = asyncio.run(go())
    except Exception:
        logs = tmp_path / "logs"
        for log in sorted(logs.glob("*.log")) if logs.exists() else []:
            print(log.name, log.read_text()[-2000:])
        raise
    assert launcher.quiescent()
    assert status == 200 and set(reply["answers"]) == set(body["questions"])
    # the profile names what the controller sent and what the bootstrap measured
    assert profile["kv_cache_dtype"] == "bfloat16"
    assert profile["attention_backend"] == "TRITON_ATTN" and profile["moe_backend"] == "cutlass"
    assert profile["executor"] == "modal-sandbox" and profile["gpu_type"] == "none"
    assert profile["weights"] == "unverified"  # the fake model is not the NVFP4 export
    assert launcher.placements[-1]["side"] == "champion"


def test_a_failing_server_is_the_sides_fault(tmp_path):
    backend = _backend(tmp_path)
    backend.command = [*backend.command[:2], backend.command[2].replace(str(FAKE), "/nope.py")]
    launcher = sandbox.SandboxLauncher(backend, canvas=64, ready_timeout=60)

    async def go() -> None:
        async with launcher({"challenger": _model(tmp_path)}, {"challenger": []}, 0.9):
            pass

    with pytest.raises(ServeFailed) as error:
        asyncio.run(go())
    assert error.value.side == "challenger"
    assert launcher.quiescent()
    tails = launcher.failures[-1]["logs"]
    assert "nope.py" in tails["vllm"] and len(tails["vllm"]) <= sandbox.LOG_TAIL


def _wire(*frames: bytes, seq: int = 0) -> bytes:
    out = b""
    for frame in frames:
        lines, seq = sandbox.encode(frame, seq)
        out += lines
    return out


def _frames(data: bytes, step: int = 1 << 20) -> list[dict]:
    class Channel:
        async def chunks(self):
            for i in range(0, len(data), step):
                yield data[i : i + step]

    async def go() -> list[dict]:
        return [f async for f in sandbox._lines(Channel())]  # type: ignore[arg-type]

    return asyncio.run(go())


def test_the_relay_refuses_oversized_and_malformed_frames():
    assert _frames(_wire(b'{"ready":true}')) == [{"ready": True}]
    with pytest.raises(JobFailed, match="malformed"):
        _frames(_wire(b"[1]"))
    with pytest.raises(JobFailed, match="malformed"):
        _frames(b'{"ready":true}\n')  # a bare JSON line is not the wire
    with pytest.raises(JobFailed, match="cap"):
        _frames(b"x" * (sandbox.LINE_MAX + 1))
    with pytest.raises(JobFailed, match="cap"):
        _frames(_wire(b"x" * (sandbox.MAX_FRAME + 1)))


def test_a_large_unicode_frame_crosses_the_wire_in_short_lines():
    """Modal drops or splits long stdout lines: a 5 MiB emoji frame travels as <= 16 KiB
    lines, delivered in odd-sized chunks, and comes back byte for byte."""
    body = "\U0001f600" * (5 * (1 << 20) // 4)
    frame = json.dumps({"id": 1, "body": body}, ensure_ascii=False).encode()
    data = _wire(b'{"ready":true}', frame, b"{}")
    assert max(len(line) for line in data.split(b"\n")) <= sandbox.LINE_MAX
    assert _frames(data, step=4093) == [{"ready": True}, {"id": 1, "body": body}, {}]


@pytest.mark.parametrize("damage", ["drop", "repeat", "swap", "truncate"])
def test_a_damaged_wire_is_detected_never_misread(damage):
    frame = json.dumps({"body": "\u00e9" * 40000}).encode()
    lines = _wire(frame, b"{}").split(b"\n")[:-1]
    assert len(lines) > 3
    if damage == "drop":
        del lines[1]
    elif damage == "repeat":
        lines.insert(1, lines[1])
    elif damage == "swap":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines[1] = lines[1][:-7]
    with pytest.raises(JobFailed, match="relay"):
        _frames(b"\n".join(lines) + b"\n")


def test_the_bootstrap_inbox_reassembles_and_rejects():
    import io

    frame = json.dumps({"x": "\U0001f600" * 20000}, ensure_ascii=False).encode()
    inbox = sandbox._Inbox(io.BytesIO(_wire(frame, b"{}")))
    assert inbox.read() == frame and inbox.read() == b"{}" and inbox.read() is None
    with pytest.raises(ValueError):
        sandbox._Inbox(io.BytesIO(_wire(b"{}", seq=5))).read()


def test_the_kernel_file_must_match_its_signed_sha(tmp_path):
    kernel = runtime.normalize_kernel({"slot": "rms_norm", "source": KERNEL})
    assert kernel is not None
    path = sandbox.write_kernel(kernel, tmp_path / "k")
    assert path.read_text() == KERNEL and (path.stat().st_mode & 0o777) == 0o444
    with pytest.raises(ValueError, match="sha256"):
        sandbox.write_kernel({**kernel, "sha256": "0" * 64}, tmp_path / "k2")


# -- kernel admission (parsed, never imported) ---------------------------------------------


def test_kernel_admission_accepts_a_plain_triton_kernel():
    kernel = runtime.normalize_kernel({"slot": "rms_norm", "source": KERNEL})
    assert kernel == {
        "slot": "rms_norm",
        "source": KERNEL,
        "sha256": hashlib.sha256(KERNEL.encode()).hexdigest(),
    }
    argv = runtime.kernel_argv(kernel)
    assert argv[0] == "--ir-op-priority" and json.loads(argv[1]) == {"rms_norm": ["opentype"]}
    assert runtime.kernel_argv(None) == []


@pytest.mark.parametrize(
    "source",
    [
        "import os\n" + KERNEL,
        "from . import x\n" + KERNEL,
        KERNEL + "\nprint('hi')\n",
        KERNEL + "\nX = open('/etc/passwd')\n",
        KERNEL.replace("@triton.jit", "@triton.jit\n@evil"),
        KERNEL.replace("@triton.jit", "@__import__('os').system('x')"),
        KERNEL.replace("eps,", "eps=__import__('os'),"),
        KERNEL + "\n\n@triton.jit\ndef rms_norm_kernel(x):\n    pass\n",
        KERNEL.replace("rms_norm_kernel", "other_kernel"),
        "class A:\n    pass\n" + KERNEL,
        "x" * (runtime.KERNEL_MAX_BYTES + 1),
        "def (",
    ],
)
def test_kernel_admission_refuses_code_that_runs_at_import(source):
    with pytest.raises(runtime.RuntimeError_):
        runtime.normalize_kernel({"slot": "rms_norm", "source": source})


def test_kernel_admission_refuses_unknown_slots_and_shapes():
    for raw in (
        {"slot": "gelu", "source": KERNEL},
        {"slot": "rms_norm"},
        {"slot": "rms_norm", "source": KERNEL, "path": "/x"},
        "rms_norm",
    ):
        with pytest.raises(runtime.RuntimeError_):
            runtime.normalize_kernel(raw)


def test_a_kernel_changes_the_signed_digest_and_options_alone_do_not():
    target, profile = {"champion": 1, "digest": "a" * 64}, "b" * 64
    plain = crypto.runtime_digest("s", target, profile, {"max_num_seqs": 8})
    assert plain == crypto.runtime_digest("s", target, profile, {"max_num_seqs": 8}, None)
    kernel = runtime.normalize_kernel({"slot": "rms_norm", "source": KERNEL})
    assert kernel is not None
    signed = crypto.runtime_digest("s", target, profile, {"max_num_seqs": 8}, kernel)
    other = {**kernel, "sha256": "c" * 64}
    assert signed != plain != crypto.runtime_digest("s", target, profile, {}, other)


# -- the NVFP4 identity ----------------------------------------------------------------------


def _shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header: dict[str, object] = {
        n: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for n, (d, s) in tensors.items()
    }
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_tensor_schema_reads_headers_only_and_binds_the_layout(tmp_path, monkeypatch):
    model = tmp_path / "m"
    model.mkdir()
    tensors = {"a.weight": ("U8", [4, 8]), "a.weight_scale": ("F8_E4M3", [4, 1])}
    _shard(model / "model-00001-of-00001.safetensors", tensors)
    index = {"weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "config.json").write_text("{}")
    schema = sandbox.tensor_schema(model)
    assert schema is not None
    monkeypatch.setattr(pins, "NVFP4_SCHEMA_SHA256", schema)
    monkeypatch.setattr(pins, "NVFP4_CONFIG_SHA256", sandbox.sha256_file(model / "config.json"))
    assert sandbox.weights_identity(model)["weights"] == "modelopt-nvfp4"
    # a BF16 tensor in place of the packed FP4 one is another layout
    _shard(model / "model-00001-of-00001.safetensors", {**tensors, "a.weight": ("BF16", [4, 16])})
    assert sandbox.weights_identity(model)["weights"] == "unverified"
    # a shard outside the directory is never opened
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "../x.safetensors"}})
    )
    assert sandbox.tensor_schema(model) is None


# -- timed answer divergence -----------------------------------------------------------------


def test_divergence_compares_c_and_b2_against_b():
    cell = runtime.Cell("decisions", 1, 1, 1000.0, 1.0, False)
    case = runtime.cell_case("seed", "short", cell, 0)
    gold = tracks.solve_body(case.body)

    def vectors(skill: str) -> dict:
        answers = {
            qid: answer(q, blur(skill, qid, case.body["seed"], gold[qid]))
            for qid, q in case.body["questions"].items()
        }
        return runtime.answer_vectors(case, {"answers": answers}) or {}

    exact = vectors("exact")
    assert exact and runtime.answer_vectors(case, {"answers": {}, "error": "x"}) == {
        q: [] for q in exact
    }

    def task(side: str, v: dict | None) -> dict:
        return {"block": 0, "side": side, "cell": "short", "case_index": 0, "vectors": v}

    same = runtime.divergence(1, [task("B", exact), task("C", exact), task("B2", exact)])
    assert same == [{"C": 0.0, "B2": 0.0}]
    wrong: dict[str, list[float]] = {
        q: [] for q in exact
    }  # every answer invalid: as far as it gets
    moved = runtime.divergence(1, [task("B", exact), task("C", wrong), task("B2", exact)])
    assert moved == [{"C": 1.0, "B2": 0.0}]
    missing = runtime.divergence(1, [task("B", exact), task("B2", exact)])
    assert missing == [{"C": 1.0, "B2": 0.0}]  # C skipping a case B answered counts fully


def test_stage_nvfp4_verifies_every_file_and_the_layout(tmp_path, monkeypatch):
    """The staging path with a fake hub: pinned digests, base support files, the schema."""
    src = tmp_path / "hub"
    src.mkdir()
    tensors = {"a.weight": ("U8", [4, 8]), "a.weight_scale": ("F8_E4M3", [4, 1])}
    shard = "model-00001-of-00001.safetensors"
    _shard(src / shard, tensors)
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard)})
    )
    (src / "config.json").write_text('{"quantization_config": {}}')
    (src / "tokenizer.json").write_text("{}")
    digest = {p.name: sandbox.sha256_file(p) for p in src.iterdir()}
    served: list[str] = []

    def fetch(repo: str, revision: str, name: str, directory: Path) -> Path:
        served.append(f"{repo}/{name}")
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes((src / name).read_bytes())
        return directory / name

    weights = {k: v for k, v in digest.items() if k != "tokenizer.json"}
    monkeypatch.setattr(pins, "BASE_SUPPORT_FILES", {"tokenizer.json": digest["tokenizer.json"]})
    monkeypatch.setattr(pins, "NVFP4_FILES", weights)
    monkeypatch.setattr(pins, "NVFP4_CONFIG_SHA256", digest["config.json"])
    monkeypatch.setattr(pins, "NVFP4_SCHEMA_SHA256", "0" * 64)
    with pytest.raises(JobFailed, match="NVFP4"):
        sandbox.stage_nvfp4(tmp_path / "snap" / "m", fetch)  # a layout other than the pin
    schema = sandbox.tensor_schema(tmp_path / "snap" / "m")
    assert schema is not None
    monkeypatch.setattr(pins, "NVFP4_SCHEMA_SHA256", schema)
    model = sandbox.stage_nvfp4(tmp_path / "snap" / "m", fetch)
    assert sandbox.weights_identity(model)["weights"] == "modelopt-nvfp4"
    assert (model / "tokenizer.json").exists()  # the base's support file
    assert f"{pins.BASE_REPO}/tokenizer.json" in served
    (src / shard).write_bytes(b"\0" * 16)  # a tampered shard never verifies
    with pytest.raises(JobFailed, match="sha256"):
        sandbox.stage_nvfp4(tmp_path / "snap2" / "m", fetch)


# -- regressions of the sandbox boundary review (each one a confirmed defect) -----------------


def _hostile(tmp_path: Path, paths: list[str], reader_after: bool = False) -> tuple[list, Any]:
    """POST each path to the hostile chat server through the relay; (status or error, launcher).
    reader_after: then one reader request."""
    launcher = sandbox.SandboxLauncher(_backend(tmp_path, hostile=True), canvas=64)
    launcher.ready_timeout = 60
    results: list[Any] = []

    async def go() -> None:
        async with launcher({"champion": _model(tmp_path)}, {"champion": []}, 0.9) as urls:
            async with launcher.client() as client:
                for path in paths:
                    try:
                        reply = await client.post(
                            urls["champion"]["chat"] + path, json={}, timeout=20
                        )
                        results.append(reply.status_code)
                    except Exception as error:  # noqa: BLE001 - recorded
                        results.append(type(error).__name__)
                if reader_after:
                    try:
                        reply = await client.post(
                            urls["champion"]["reader"] + "/v1/systemone", json={}, timeout=20
                        )
                        results.append(reply.json())
                    except Exception as error:  # noqa: BLE001 - recorded
                        results.append(type(error).__name__)

    asyncio.run(go())
    return results, launcher


def test_non_ascii_replies_never_overflow_the_frame_cap(tmp_path):
    """A 6 MB body of 4-byte characters escaped to ASCII was an 18 MB frame: the relay died
    and every pending request failed as infrastructure. It is one refused reply now."""
    (emoji, after), launcher = _hostile(tmp_path, ["/emoji", "/error"])
    assert emoji == 200 or emoji == 502  # encoded raw it fits the frame, or it is refused
    assert after == 500  # the relay survived
    raw = sandbox._frame({"id": 1, "status": 200, "body": {"x": "\U0001f600" * 1_400_000}})
    assert len(raw) < sandbox.MAX_FRAME


def test_a_deeply_nested_reply_is_answered_at_once(tmp_path):
    """json.loads raised RecursionError in a relay thread, which died without replying: the
    controller waited out the whole read timeout. A reply that is not JSON, all processes
    alive, is a content fault of the served process."""
    import time

    start = time.monotonic()
    (deep, after), launcher = _hostile(tmp_path, ["/deep", "/error"])
    assert deep == 502 and after == 500  # the 500 is relayed, ambiguous
    assert time.monotonic() - start < 15
    assert launcher.content_fault("champion") is not None


def test_a_hung_server_is_not_a_content_fault(tmp_path):
    """No answer may be the placement's (a crash, a hang): it stays a transport error."""
    launcher = sandbox.SandboxLauncher(_backend(tmp_path, hostile=True), canvas=64)

    async def go() -> str:
        async with launcher({"champion": _model(tmp_path)}, {"champion": []}, 0.9) as urls:
            async with launcher.client() as client:
                try:
                    await client.post(urls["champion"]["chat"] + "/hang", json={}, timeout=2)
                except Exception as error:  # noqa: BLE001 - recorded
                    return type(error).__name__
        return "answered"

    assert asyncio.run(go()) in ("ReadTimeout", "ConnectError")  # a transport error
    assert launcher.content_fault("champion") is None


def test_a_dead_reader_ends_the_relay_before_a_spoofed_port_answers(tmp_path):
    """The served process kills the pinned reader and listens on its port: no reader reply
    is relayed once the reader exited (checked before and after every request)."""
    (spoof, reader), launcher = _hostile(tmp_path, ["/spoof"], reader_after=True)
    assert spoof == "ConnectError"  # the reader died while it ran: voided too
    assert reader == "ConnectError"  # never the forged {"answers": {"forged": true}}
    assert launcher.failures and launcher.failures[-1].get("exited") == "reader"


def test_a_reader_dying_during_a_request_voids_its_reply(tmp_path):
    """The race: the reader is alive when the request is relayed and dies before the reply
    is read back; the reply is dropped, not trusted."""

    class Dead:
        def poll(self) -> int:
            return -9

        def wait(self, timeout: float) -> int:
            return -9

    replies: list[dict] = []
    relay: Any = sandbox._relay  # duck-typed writer and processes
    relay(
        replies.append,
        "http://127.0.0.1:1",
        {"id": 3, "to": "reader", "path": "/", "body": {}},
        1,
        (Dead(), Dead()),
    )
    assert replies == [{"id": 3, "status": 599, "body": {"error": "a served process exited"}}]


def test_a_dead_engine_masked_by_the_reader_is_never_the_candidates(tmp_path):
    """The pinned reader answers 500/502 when its vllm fails or is gone. A GPU fault killing
    vllm mid-read must retry, never reject: the reply becomes a transport failure and no
    content fault is recorded."""
    launcher = sandbox.SandboxLauncher(_backend(tmp_path, hostile=True, proxy=True), canvas=64)

    async def go() -> list[Any]:
        seen: list[Any] = []
        async with launcher({"challenger": _model(tmp_path)}, {"challenger": []}, 0.9) as urls:
            async with launcher.client() as client:
                for path in ("/error", "/die", "/error"):
                    try:
                        reply = await client.post(
                            urls["challenger"]["reader"] + path, json={}, timeout=20
                        )
                        seen.append(reply.status_code)
                    except Exception as error:  # noqa: BLE001 - recorded
                        seen.append(type(error).__name__)
        return seen

    # a live engine's 500 is masked as 502: ambiguous, never a content fault; then the engine
    # dies: the reader's masked 500 is voided; then nothing is relayed any more
    assert asyncio.run(go()) == [502, "ConnectError", "ConnectError"]
    assert launcher.content_fault("challenger") is None


def test_a_live_5xx_is_not_the_candidates_either(monkeypatch):
    """Worker side: a 5xx with every process alive stays ambiguous and retries."""
    from opentype_challenge import worker

    class Launcher:
        def content_fault(self, side: str) -> None:
            return None

    instance = worker.Worker(None, None, Launcher())  # type: ignore[arg-type]
    error = worker.JobFailed("challenger http://x returned 500", retry=True)
    assert instance._content_fault(error, "challenger").retry is True


def test_limits_never_raise_an_inherited_hard_limit():
    """preexec_fn runs after setuid: raising a hard limit there failed the spawn."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    code = (
        "import resource, subprocess, sys; "
        "from opentype_challenge import sandbox; "
        "resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256)); "
        "p = subprocess.run([sys.executable, '-c', 'import resource; "
        "print(resource.getrlimit(resource.RLIMIT_NOFILE)[1])'], "
        "preexec_fn=sandbox._limits(None, None), capture_output=True, text=True); "
        "print(p.stdout.strip())"
    )
    import subprocess

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "256"
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft, hard)


def test_a_huge_build_log_is_read_by_its_tail(tmp_path):
    """build() read the whole log a child may fill (8 GiB) into root's memory."""
    log = tmp_path / "build.log"
    with log.open("wb") as handle:
        handle.seek(50 << 20)
        handle.write(b"the end")
    assert sandbox._tail(log, 2000).endswith("the end")
    assert len(sandbox._tail(log, 2000)) <= 2000
    assert sandbox._tail(tmp_path / "missing.log", 10) == ""


def test_a_build_child_is_capped_and_its_failure_is_the_kernels(tmp_path, monkeypatch):
    """The compile child writes past its file cap: the bootstrap answers build_failed."""
    import io

    monkeypatch.setattr(sandbox, "BUILD_FILE_BYTES", 1 << 20)
    kernel = runtime.normalize_kernel({"slot": "rms_norm", "source": KERNEL})
    assert kernel is not None
    real = sandbox._spawn

    def spawn(command, *args, **kwargs):  # the child spams its log instead of compiling
        spam = "import sys\nwhile True: sys.stdout.write('x' * 65536)"
        return real([sys.executable, "-c", spam], *args, **kwargs)

    monkeypatch.setattr(sandbox, "_spawn", spawn)
    stdin = io.BytesIO(_wire(json.dumps({"kernel": kernel, "arch": 103}).encode()))
    stdout = io.BytesIO()
    assert sandbox.build(stdin, stdout, demote=False, kernel_dir=tmp_path / "k") == 1
    (frame,) = _frames(stdout.getvalue())
    assert "build_failed" in frame and len(frame["build_failed"]) <= 2000
    assert (tmp_path / "k" / "logs" / "build.log").stat().st_size <= 1 << 20


def test_stdin_lines_are_sent_in_drained_chunks_under_the_sdk_buffer():
    """Modal's sandbox stdin refuses a write past 2 MiB between drains (BufferError, found by
    the relay probe at 1 MiB of escaped input): a line goes in drained chunks under that."""

    class Stdin:
        limit = 2 << 20

        def __init__(self) -> None:
            self.buffer, self.sent = b"", b""

        def write(self, data: bytes) -> None:
            if len(self.buffer) + len(data) > self.limit:
                raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
            self.buffer += data

        class _Drain:
            def __init__(self, outer: Any) -> None:
                self.outer = outer

            async def aio(self) -> None:
                self.outer.sent += self.outer.buffer
                self.outer.buffer = b""

        @property
        def drain(self) -> Any:
            return Stdin._Drain(self)

    stdin = Stdin()
    line = json.dumps({"pad": "\U0001f600" * 1_500_000}, ensure_ascii=False) + "\n"
    assert len(line.encode()) > 2 * Stdin.limit
    asyncio.run(sandbox._send_chunked(stdin, line))
    assert stdin.sent.decode() == line and stdin.buffer == b""
