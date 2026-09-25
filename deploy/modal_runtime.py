"""Runtime lane on Modal: stage the pinned NVFP4 export, then a bounded stock smoke on B300.

    modal run deploy/modal_runtime.py::stage          # CPU, network: download + verify, once
    modal run deploy/modal_runtime.py::smoke --moe-backend cutlass --cases 8

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

import modal

# The CUDA 13.0 worker image the B300 platform probe passed on, plus this checkout's package
# (the bootstrap and the kernel slot entry point) installed over the image's copy.
BASE_IMAGE = (
    "ghcr.io/opentypeai/challenge-worker"
    "@sha256:8a6b2081929ff18691526e9e8fc83dab981499272f0e13114353466b5afcd6ee"
)
SNAPSHOT = "/snap"

app = modal.App("opentype-runtime")
image = (
    modal.Image.from_registry(
        BASE_IMAGE,
        setup_dockerfile_commands=["RUN ln -sf $(command -v python3) /usr/local/bin/python"],
    )
    .entrypoint([])
    .add_local_dir(
        ".", "/opt/opentype-src", copy=True, ignore=["**/.venv", "**/.git", "**/__pycache__"]
    )
    .run_commands("uv pip install --system --no-cache --no-deps /opt/opentype-src")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
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
    except Exception:
        print(json.dumps({"failures": launcher.failures}, indent=2)[-20000:])
        raise
    print(json.dumps(result, indent=2))
    return result


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
