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


def _backend(tmp_path: Path) -> sandbox.ProcessBackend:
    reader = tmp_path / "structured_server.py"
    reader.write_text(FAKE.read_text())
    code = (
        "import sys; from pathlib import Path; from opentype_challenge import sandbox; "
        "sys.exit(sandbox.serve(sys.stdin.buffer, sys.stdout.buffer, "
        f"vllm=({sys.executable!r}, {str(FAKE)!r}), reader=Path({str(reader)!r}), "
        f"demote=False, kernel_dir=Path({str(tmp_path / 'k')!r}), "
        f"log_dir=Path({str(tmp_path / 'logs')!r}), health_timeout=30, "
        f"port_base={_port_base()}))"
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


def test_the_relay_refuses_oversized_and_malformed_frames():
    class Channel:
        def __init__(self, text: str):
            self.text = text

        async def chunks(self):
            yield self.text

    async def frames(text: str) -> list[dict]:
        return [f async for f in sandbox._lines(Channel(text))]  # type: ignore[arg-type]

    assert asyncio.run(frames('{"ready":true}\n')) == [{"ready": True}]
    with pytest.raises(JobFailed, match="malformed"):
        asyncio.run(frames("[1]\n"))
    with pytest.raises(JobFailed, match="cap"):
        asyncio.run(frames("x" * (sandbox.MAX_FRAME + 1)))


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
