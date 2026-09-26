"""Runtime lane on Modal: stage the pinned NVFP4 export, then a bounded stock smoke on B300.

    modal run deploy/modal_runtime.py::stage          # CPU, network: download + verify, once
    modal run deploy/modal_runtime.py::smoke --moe-backend cutlass --cases 8
    modal run deploy/modal_runtime.py::kernel_smoke --cases 8 [--control]

`stage` writes the pinned nvidia NVFP4 export plus the base support files into the dedicated
volume `opentype-nvfp4-snapshot` (never the quality worker's volume), verifies every sha256
and the tensor schema, and commits once. Nothing writes the volume afterwards.

`smoke` is the controller: a CPU Function with that volume read-only and no secret. It starts
ONE fresh sandbox (gpu="B300", block_network, secrets=[], no OIDC token, the snapshot mounted
read-only) serving the STOCK model through opentype_challenge.sandbox, sends a few decisions
reads and one ops episode, scores them here (gold never enters the sandbox), prints the
measured profile and exits. No miner code, no worker token, hard timeouts everywhere.
"""

import json
import os
import sys
from pathlib import Path

import modal

# The CUDA 13.0 worker image the B300 platform probe passed on, plus this checkout's package
# (the bootstrap and the kernel slot entry point) installed over the image's copy.
BASE_IMAGE = (
    "ghcr.io/opentypeai/challenge-worker"
    "@sha256:8a6b2081929ff18691526e9e8fc83dab981499272f0e13114353466b5afcd6ee"
)
SNAPSHOT = "/snap"
ROOT = Path(__file__).resolve().parent.parent
DEPLOY, DEPLOY_DIR = ROOT / "deploy", "/opt/opentype-deploy"


# Every file a deploy uploads or Modal auto-mounts (a Function's own file) beside src/.
UPLOADED = (
    "pyproject.toml", "README.md", "LICENSE",
    "deploy/modal_runtime.py", "deploy/modal_runtime_kernels.py",
    "deploy/modal_controller.py", "deploy/modal_pilot.py",
)  # fmt: skip


def source_sha256(root: Path = ROOT) -> str:
    """sha256 over (path, sha256) of every uploaded file: src/ plus UPLOADED. A symlink is
    refused: rglob and Modal's upload could resolve it differently."""
    import hashlib

    tree = [p for p in root.joinpath("src").rglob("*") if "__pycache__" not in p.parts]
    paths = [root / "src", *tree, *(root / name for name in UPLOADED)]
    if missing := [name for name in UPLOADED if not (root / name).is_file()]:
        raise SystemExit(f"missing uploaded source files: {missing}")
    if links := [str(p.relative_to(root)) for p in paths if p.is_symlink()]:
        raise SystemExit(f"refusing symlinks in the uploaded source: {links}")
    digest = hashlib.sha256()
    for path in sorted(p for p in paths if p.is_file()):
        digest.update(
            f"{path.relative_to(root)}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode()
        )
    return digest.hexdigest()


SOURCE_ENV = ("OPENTYPE_SOURCE_SHA256", "OPENTYPE_SOURCE_REVISION")


def source_identity() -> dict[str, str]:
    """What this deploy overlays on BASE_IMAGE: source_sha256() and the operator-declared
    revision. The image identity is the pair: BASE_IMAGE's digest alone omits the overlay."""
    if not modal.is_local():  # in a container: the values this image was built with
        return {k: os.environ.get(k, "") for k in ("OPENTYPE_WORKER_IMAGE", *SOURCE_ENV)}
    return {
        "OPENTYPE_WORKER_IMAGE": BASE_IMAGE,  # the base only; the overlay is named below
        "OPENTYPE_SOURCE_SHA256": source_sha256(),
        "OPENTYPE_SOURCE_REVISION": os.environ.get("OPENTYPE_SOURCE_REVISION", "undeclared"),
    }


app = modal.App("opentype-runtime")
image = (
    modal.Image.from_registry(
        BASE_IMAGE,
        setup_dockerfile_commands=["RUN ln -sf $(command -v python3) /usr/local/bin/python"],
    )
    .entrypoint([])
    # only what the package build reads: never the checkout (it may hold secrets or state)
    .add_local_dir(ROOT / "src", "/opt/opentype-src/src", copy=True, ignore=["**/__pycache__"])
    .add_local_file(ROOT / "pyproject.toml", "/opt/opentype-src/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/opentype-src/README.md", copy=True)
    .add_local_file(ROOT / "LICENSE", "/opt/opentype-src/LICENSE", copy=True)
    # the deploy modules other deploy files import (modal mounts only a function's own file)
    .add_local_file(DEPLOY / "modal_runtime.py", f"{DEPLOY_DIR}/modal_runtime.py", copy=True)
    .add_local_file(
        DEPLOY / "modal_runtime_kernels.py", f"{DEPLOY_DIR}/modal_runtime_kernels.py", copy=True
    )
    .run_commands("uv pip install --system --no-cache --no-deps /opt/opentype-src")
    .env(
        {
            "PYTHONPATH": DEPLOY_DIR,
            **source_identity(),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
)
snapshot = modal.Volume.from_name("opentype-nvfp4-snapshot", create_if_missing=True)


def _directory():  # type: ignore[no-untyped-def]
    from pathlib import Path

    from opentype_challenge import pins

    return Path(SNAPSHOT) / f"nvfp4-{pins.NVFP4_REVISION}"


@app.function(image=image, cpu=4, memory=16384, volumes={SNAPSHOT: snapshot}, timeout=3 * 3600)
def stage() -> dict:
    from opentype_challenge import sandbox
    from opentype_challenge.worker import hf_fetch

    directory = _directory()
    sandbox.stage_nvfp4(directory, hf_fetch)
    snapshot.commit()
    identity = sandbox.weights_identity(directory)
    print(json.dumps(identity))
    return identity


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={SNAPSHOT: snapshot.with_mount_options(read_only=True)},
    timeout=3600,
)
def smoke(moe_backend: str = "cutlass", cases: int = 8, max_model_len: int = 32768) -> dict:
    import asyncio
    import time
    from pathlib import Path

    from opentype_challenge import runtime, sandbox
    from opentype_challenge.worker import Worker

    if moe_backend not in runtime.MOE_BACKENDS or not 1 <= cases <= 32:
        raise SystemExit(f"--moe-backend one of {runtime.MOE_BACKENDS}, --cases 1..32")
    model = _directory()
    if sandbox.weights_identity(model)["weights"] != "modelopt-nvfp4":
        raise SystemExit("run `stage` first: the snapshot does not verify as NVFP4")
    backend = sandbox.ModalBackend(app, image, snapshot, Path(SNAPSHOT), timeout=2400)
    launcher = sandbox.SandboxLauncher(backend, max_model_len=max_model_len, ready_timeout=1800)
    argv = runtime.serving_argv({**runtime.PROFILE_FIXED, "moe_backend": moe_backend})
    decisions = runtime.Cell("decisions", cases, cases, 60000.0, 1.0, False)
    ops = runtime.Cell("ops", 1, 1, 60000.0, 1.0, False)
    work = [runtime.cell_case("smoke", "short", decisions, i) for i in range(cases)]
    work.append(runtime.cell_case("smoke", "chat", ops, 0))

    async def go() -> dict:
        started = time.monotonic()
        async with launcher({"champion": model}, {"champion": argv}, 0.9) as urls:
            ready = time.monotonic() - started
            async with launcher.client() as client:

                async def call(url: str, body: dict) -> dict | None:
                    try:
                        response = await client.post(url, json=body, timeout=600)
                        data = response.json() if response.status_code == 200 else None
                    except Exception:  # noqa: BLE001 - reported per case
                        return None
                    return data if isinstance(data, dict) else None

                rows = []
                for case in work:
                    t0 = time.monotonic()
                    item = await Worker._task(call, urls["champion"], case)
                    rows.append(
                        {
                            "track": case.track,
                            "ms": round((time.monotonic() - t0) * 1000),
                            "ok": runtime.task_ok(case, item),
                            "error": item.get("error"),
                        }
                    )
            profile = launcher.profile()
        return {
            "ready_seconds": round(ready, 1),
            "profile": profile,
            "placements": launcher.placements,
            "cases": rows,
            "ok": sum(r["ok"] for r in rows),
            "quiescent": launcher.quiescent(),
        }

    try:
        result = asyncio.run(go())
    except Exception as error:
        print(json.dumps({"failures": launcher.failures, "quiescent": launcher.quiescent()},
                         indent=2)[-20000:])  # fmt: skip
        # a builtin: the local modal client cannot unpickle this package's exception types
        raise RuntimeError(f"{type(error).__name__}: {error}"[:2000]) from None
    print(json.dumps(result, indent=2))
    return result


# next to this file locally; in the image under DEPLOY_DIR (PYTHONPATH)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # local; the image sets PYTHONPATH
from modal_runtime_kernels import CONTROL_KERNEL, RMS_KERNEL, SINGLE_KERNEL  # noqa: E402

VARIANTS = {"correct": RMS_KERNEL, "single": SINGLE_KERNEL, "control": CONTROL_KERNEL}


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={SNAPSHOT: snapshot.with_mount_options(read_only=True)},
    timeout=3 * 3600,
)
def kernel_smoke(
    moe_backend: str = "cutlass",
    cases: int = 8,
    max_model_len: int = 32768,
    variants: str = "stock,correct",
) -> dict:
    """The kernel slot end to end, no speed claim: build the named kernels in CPU sandboxes,
    then serve the same cases in distinct fresh B300 sandboxes, one per entry of `variants`,
    in order (comma separated; "stock" may repeat, which measures stock-vs-stock spread
    across placements; kernels: correct, single, control). The first stock session is the
    reference for per-case distances. Returns every raw output: keep it private."""
    import asyncio
    import time
    from pathlib import Path

    from opentype_challenge import runtime, sandbox
    from opentype_challenge.worker import Worker

    order = [v.strip() for v in variants.split(",") if v.strip()]
    if (
        moe_backend not in runtime.MOE_BACKENDS
        or not 1 <= cases <= 16
        or not 1 <= len(order) <= 5
        or order[0] != "stock"
        or any(v != "stock" and v not in VARIANTS for v in order)
    ):
        raise SystemExit(
            f"--moe-backend one of {runtime.MOE_BACKENDS}, --cases 1..16, --variants: "
            f"stock first, then up to 4 of stock,{','.join(VARIANTS)}"
        )
    model = _directory()
    if sandbox.weights_identity(model)["weights"] != "modelopt-nvfp4":
        raise SystemExit("run `stage` first: the snapshot does not verify as NVFP4")
    backend = sandbox.ModalBackend(app, image, snapshot, Path(SNAPSHOT), timeout=2400)
    launcher = sandbox.SandboxLauncher(backend, max_model_len=max_model_len, ready_timeout=1800)
    argv = runtime.serving_argv({**runtime.PROFILE_FIXED, "moe_backend": moe_backend})
    cell = runtime.Cell("decisions", cases, cases, 60000.0, 1.0, False)
    work = [runtime.cell_case("kernel-smoke", "short", cell, i) for i in range(cases)]
    kernels = {
        name: runtime.normalize_kernel({"slot": "rms_norm", "source": VARIANTS[name]})
        for name in dict.fromkeys(order)
        if name != "stock"
    }

    async def serve(kernel: dict | None) -> list[dict]:
        extra = [*argv, *runtime.kernel_argv(kernel)]
        async with launcher(
            {"champion": model}, {"champion": extra}, 0.9, {"champion": kernel}
        ) as urls:
            async with launcher.client() as client:

                async def call(url: str, body: dict) -> dict | None:
                    try:
                        response = await client.post(url, json=body, timeout=600)
                        data = response.json() if response.status_code == 200 else None
                    except Exception:  # noqa: BLE001 - reported per case
                        return None
                    return data if isinstance(data, dict) else None

                rows = []
                for case in work:
                    item = await Worker._task(call, urls["champion"], case)
                    rows.append(
                        {
                            "ok": runtime.task_ok(case, item),
                            "error": item.get("error"),
                            "vectors": runtime.answer_vectors(case, item),
                            "item": item,  # the raw output, for later independent comparison
                        }
                    )
                return rows

    progress: dict = {"built": {}, "sessions": {}}
    raw: dict[str, list[dict]] = {}  # returned, never printed: prompts are private

    async def go() -> None:
        arch = 103  # B300 (sm_103), as the profile's compute_cap reports it
        for name, kernel in kernels.items():
            progress["built"][name] = await launcher.build(kernel, arch)
            print(json.dumps({"progress": "built", "kernel": name}), flush=True)
        stock: list[dict] = []
        for number, name in enumerate(order):
            started = time.monotonic()
            rows = await serve(kernels.get(name))
            stock = stock or rows
            label = f"{number}:{name}"
            raw[label] = [r["item"] for r in rows]
            session = {
                "seconds": round(time.monotonic() - started, 1),
                "ok": sum(r["ok"] for r in rows),
                "errors": sum(bool(r["error"]) for r in rows),
                # per case, so a control that changes nothing cannot hide in a mean
                "distance_from_stock": [
                    round(runtime._distance(a["vectors"] or {}, b["vectors"] or {}), 4)
                    for a, b in zip(stock, rows, strict=True)
                ],
                "profile": launcher.profile(),
            }
            progress["sessions"][label] = session
            print(json.dumps({"progress": "session", "name": label, **session}), flush=True)

    error = None
    try:
        asyncio.run(go())
    except BaseException as caught:  # noqa: BLE001 - reported below as a builtin error
        error = f"{type(caught).__name__}: {caught}"[:2000]
    result = {
        **progress,
        "kernels": {n: runtime.kernel_ref(k) for n, k in kernels.items()},
        "placements": launcher.placements,  # distinct sandboxes; the same GPU is not promised
        "quiescent": launcher.quiescent(),
        "error": error,
        "failures": launcher.failures[-4:],
    }
    print(json.dumps(result, indent=2)[-40000:], flush=True)
    if error:
        # a builtin: the local modal client cannot unpickle this package's exception types
        raise RuntimeError(error)
    # the cases are rebuilt from the fixed seed "kernel-smoke" (runtime.cell_case)
    return {**result, "raw": raw}


@app.local_entrypoint()
def kernel_smoke_report(
    out: str, variants: str = "stock,correct", cases: int = 8, moe_backend: str = "cutlass"
) -> None:
    """Run kernel_smoke and keep its whole result, raw outputs included, in a private local
    file (mode 0600), never in logs."""
    import os

    result = kernel_smoke.remote(moe_backend=moe_backend, cases=cases, variants=variants)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(result, handle)
    print(f"wrote {out}")


# A trusted echo: each stdin line comes back as {n, sha256, size} of what arrived, then the
# probe lines it was asked for, made of 4-byte characters, with their own sha256.
PROBE = r"""
import hashlib, json, sys
from opentype_challenge.sandbox import _Inbox, _Out
inbox, out = _Inbox(sys.stdin.buffer), _Out(sys.stdout.buffer)
while (raw := inbox.read()) is not None:
    ask = json.loads(raw)
    if "pad" in ask:
        out({"n": ask["n"], "bytes": len(ask["pad"].encode()),
             "sha": hashlib.sha256(ask["pad"].encode()).hexdigest()})
    else:
        pad = "\U0001f600" * (ask["size"] // 4)
        out({"n": ask["n"], "sha": hashlib.sha256(pad.encode()).hexdigest(), "pad": pad})
"""


@app.function(image=image, cpu=1, memory=2048, timeout=900)
def relay_probe(max_mib: int = 7, stdio: str = "exec") -> dict:
    """Frames of 1 KiB .. max_mib MiB, each direction as its own phase, through a CPU
    sandbox's stdio with the relay's wire codec (no GPU, no volume, no network, no miner
    code). stdio="exec" is the production path (an exec'd process, the command router);
    "entrypoint" reads the sandbox's own stdout (Modal's rate-limited log pipeline). Only
    lengths, digests and timings are reported, never payloads."""
    import asyncio
    import contextlib
    import hashlib
    import time

    from modal.stream_type import StreamType

    from opentype_challenge import sandbox

    if not 1 <= max_mib <= 8 or stdio not in ("exec", "entrypoint"):
        raise SystemExit("--max-mib 1..8, --stdio exec|entrypoint")
    sizes = [1 << 10, 64 << 10, 1 << 20, 4 << 20, max_mib << 20]
    command = ("python3", "-c", PROBE)

    async def go() -> dict:
        box = await modal.Sandbox.create.aio(
            *(command if stdio == "entrypoint" else ("sleep", "infinity")),
            app=app, image=image, cpu=1, memory=2048,
            block_network=True, secrets=[], include_oidc_identity_token=False, timeout=600,
        )  # fmt: skip
        process = box
        if stdio == "exec":
            process = await box.exec.aio(*command, text=False, stderr=StreamType.DEVNULL)
        channel = sandbox._ModalChannel(box, process)
        send, frames = sandbox._Sender(channel), sandbox._lines(channel)
        rows: list[dict] = []
        try:
            for n, size in enumerate(sizes):
                # a frame is <= MAX_FRAME: leave room for the JSON around the payload
                size = min(size, sandbox.MAX_FRAME - 256)
                row: dict = {"size": size}
                rows.append(row)
                pad = "\u00e9" * (size // 2)
                start = time.monotonic()
                row["phase"] = "up"
                await send({"n": n, "pad": pad})
                echo = await asyncio.wait_for(frames.__anext__(), 120)
                row["up_ok"] = echo == {
                    "n": n, "bytes": len(pad.encode()),
                    "sha": hashlib.sha256(pad.encode()).hexdigest(),
                }  # fmt: skip
                row["up_s"] = round(time.monotonic() - start, 2)
                start = time.monotonic()
                row["phase"] = "down"
                await send({"n": n, "size": size})
                line = await asyncio.wait_for(frames.__anext__(), 120)
                got = line.get("pad", "").encode()
                row["down_bytes"] = len(got)
                row["down_ok"] = line.get("n") == n and (
                    hashlib.sha256(got).hexdigest() == line.get("sha")
                )
                row["down_s"] = round(time.monotonic() - start, 2)
                row["phase"] = "done"
        except BaseException as error:  # noqa: BLE001 - the first size that breaks is the answer
            rows.append({"error": repr(error)[:300]})
        finally:
            with contextlib.suppress(BaseException):
                await frames.aclose()
            await channel.close()
        ok = all(r.get("up_ok") and r.get("down_ok") for r in rows)
        return {"stdio": stdio, "rows": rows, "ok": ok}

    result = asyncio.run(go())
    print(json.dumps(result, indent=2))
    return result
