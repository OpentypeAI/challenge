"""Bounded B300 pilots of the controller path. No production API, token or store is touched.

    modal run deploy/modal_pilot.py::quality_report --out /tmp/pilot-quality.json
    modal run deploy/modal_pilot.py::calibration_report --out /tmp/pilot-cal.json [--blocks 2]

quality: one real NVFP4 quality duel through the production code (the container app and
store, the Worker, the SandboxLauncher over fresh B300 sandboxes), all inside this trusted
CPU function:
  - an ephemeral in-process container: a fresh sqlite store in a temp dir, random tokens,
    a fake metagraph;
  - the base champion migrates to the pinned official export (POST /v1/admin/champion/nvfp4);
  - the challenger is the same official files under another repo name, so both sides serve
    the same bytes;
  - weights are copied from the read-only staged snapshot, never downloaded;
  - the verdict is whatever the container scores. A crown is not expected, and none is a
    finding.

Intake refuses a clone of the champion, so this pilot inserts that one challenger row
directly (`_pilot_challenger`); everything after intake is the production path.

calibration: stock-only measurement for the runtime calibration. B, B and B' each serve the
official snapshot alone, in a fresh sandbox, at the pinned profile, and time every required
cell (decisions, longctx, ops, sql) with `Worker._measure`, the production timing. The
controller scores each raw output against gold with `runtime.task_ok` (gold stays here). It
reports per run and cell: tasks, ok, errors, seconds and p95. Any run whose cell completes
no task is reported as NO_DECISION: no calibration can come from it. Nothing here chooses
thresholds, relaxes scoring or picks cases: the cells come from `--cells` or the default
below, seeded by `--seed`.

Both entrypoints return raw outputs. The local entrypoint writes them to a new file (mode
0600) and never prints them: prompts are private benchmark inputs.
"""

import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from modal_runtime import SNAPSHOT, _directory, image, snapshot  # noqa: E402

WORK = "/work"
PILOT_REPO = "pilot/same-bytes"
READY = 720  # s per sandbox to be ready (measured 244-409 s on the staged snapshot)
SESSION = 1500  # s per serve session, ready included: a hard bound
MAX_CASES, MAX_CONCURRENCY = 32, 16  # per calibration pilot cell
# All four runtime tracks, cold (the production default of Cell.warm is the operator's). The
# SLO is loose on purpose so that correctness, not a guessed latency bound, decides ok; the
# report keeps every latency for choosing the real one.
CELLS = {
    "decisions": {"track": "decisions", "cases": 16, "concurrency": 8, "slo_ms": 120000.0},
    "longctx": {"track": "longctx", "cases": 4, "concurrency": 2, "slo_ms": 120000.0},
    "ops": {"track": "ops", "cases": 8, "concurrency": 4, "slo_ms": 120000.0},
    "sql": {"track": "sql", "cases": 8, "concurrency": 4, "slo_ms": 120000.0},
}

app = modal.App("opentype-pilot")
work = modal.Volume.from_name("opentype-pilot-work", create_if_missing=True)
CPU_FUNCTION = {
    "image": image,
    "cpu": 4,
    "memory": 16384,
    "timeout": 4 * 3600,
    "max_containers": 1,
}


def _snapshot_fetch(repo: str, revision: str, filename: str, directory: Path) -> Path:
    """Stand-in for the Hub: the staged snapshot's copy of the file (the caller verifies its
    sha256), copied into `directory` as hf_hub_download would write it."""
    import shutil

    from opentype_challenge import pins

    known = {pins.NVFP4_REPO: pins.NVFP4_FILES, PILOT_REPO: pins.NVFP4_FILES,
             pins.BASE_REPO: pins.BASE_SUPPORT_FILES}  # fmt: skip
    if filename not in known.get(repo, {}):
        raise RuntimeError(f"the pilot snapshot does not serve {repo}/{filename}")
    source = _directory() / filename
    target = directory / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _pilot_challenger(store, manifest: dict) -> str:  # type: ignore[no-untyped-def]
    """Intake without the clone refusal: the one challenger of this pilot, as submit would
    queue it (a fixed pilot hotkey, no nonce: nothing is signed or paid here)."""
    import secrets

    from opentype_challenge.crypto import manifest_digest
    from opentype_challenge.store import _dumps

    digest = manifest_digest(manifest["repo"], manifest["revision"], manifest["files"])
    submission = "s_" + secrets.token_hex(8)
    with store._tx() as db:
        db.execute(
            "INSERT INTO submissions (id, hotkey, repo, revision, files, digest, state, "
            "created_at) VALUES (?, 'pilot', ?, ?, ?, ?, 'queued', ?)",
            (submission, manifest["repo"], manifest["revision"], _dumps(manifest["files"]),
             digest, store._now()),
        )  # fmt: skip
        assert store._new_job(db, submission), "the pilot challenger did not queue"
    return submission


@app.function(
    **CPU_FUNCTION,
    volumes={SNAPSHOT: snapshot.with_mount_options(read_only=True), WORK: work},
)
def quality(cases: str = "decisions=32,longctx=8,ops=8,sql=8") -> dict:
    import asyncio
    import secrets
    import shutil
    import tempfile

    import httpx

    from opentype_challenge import pins, sandbox
    from opentype_challenge.app import Config, create_app
    from opentype_challenge.store import Settings
    from opentype_challenge.tracks import TrackPlan
    from opentype_challenge.worker import Api, Worker

    counts = {k: int(v) for k, v in (p.split("=") for p in cases.split(","))}
    if set(counts) - {"decisions", "longctx", "ops", "sql"} or not all(
        0 < n <= 64 for n in counts.values()
    ):
        raise SystemExit("--cases: decisions,longctx,ops,sql counts in 1..64")
    if sandbox.weights_identity(_directory())["weights"] != "modelopt-nvfp4":
        raise SystemExit("run modal_runtime.py::stage first: the snapshot is not NVFP4")
    run_dir = Path(WORK) / f"pilot-{int(time.time())}-{secrets.token_hex(4)}"
    state = Path(tempfile.mkdtemp())
    tokens = {name: secrets.token_hex(32) for name in ("admin", "worker")}
    for name, token in tokens.items():
        (state / f"{name}.token").write_text(token)
        (state / f"{name}.token").chmod(0o600)
    plan = {t: TrackPlan(1.0 / len(counts), n) for t, n in counts.items()}
    config = Config(
        slug="opentype",
        state_dir=state / "data",
        master_url="http://master.pilot",
        internal_token_file=None,
        admin_token_file=state / "admin.token",
        worker_token_file=state / "worker.token",
        settings=Settings(plan=plan),
    )

    def metagraph(request: httpx.Request) -> httpx.Response:
        body = {"epoch": 1, "block": 1, "netuid": 1, "hotkeys": {}}
        return httpx.Response(200, json=body)

    container = create_app(config, transport=httpx.MockTransport(metagraph), beacon=lambda: None)
    store = container.state.store
    backend = sandbox.ModalBackend(app, image, work, Path(WORK), commit=True, timeout=SESSION)
    launcher = sandbox.SandboxLauncher(backend, ready_timeout=READY)
    started = time.monotonic()

    async def go() -> dict:
        transport = httpx.ASGITransport(app=container)
        async with httpx.AsyncClient(transport=transport, base_url="http://pilot") as api:
            admin = {"authorization": f"Bearer {tokens['admin']}"}
            migrated = await api.post("/v1/admin/champion/nvfp4", headers=admin)
            if migrated.status_code != 200:
                raise RuntimeError(f"migration: {migrated.status_code} {migrated.text[:300]}")
            manifest = {"repo": PILOT_REPO, "revision": pins.NVFP4_REVISION,
                        "files": dict(pins.NVFP4_FILES)}  # fmt: skip
            submission = _pilot_challenger(store, manifest)
            worker = Worker(
                Api("http://pilot", tokens["worker"], api), run_dir, launcher,
                fetch=_snapshot_fetch,
            )  # fmt: skip
            ran = await asyncio.wait_for(worker.run_once(), 2 * SESSION + 3600)
            return {"ran": ran, "submission": store.submission(submission)}

    error = None
    try:
        result = asyncio.run(go())
    except BaseException as caught:  # noqa: BLE001 - a builtin error for the local client
        error, result = f"{type(caught).__name__}: {caught}"[:2000], {}
    finally:
        if not launcher.serving():  # else a sandbox may still mount it: kept, reported below
            shutil.rmtree(run_dir, ignore_errors=True)
        work.commit()
    out = {
        "run_dir_kept": str(run_dir) if run_dir.exists() else None,
        **result,
        "seconds": round(time.monotonic() - started, 1),
        "placements": launcher.placements,
        "quiescent": launcher.quiescent(),
        "failures": launcher.failures[-4:],
        "error": error,
    }
    submission = out.get("submission") or {}
    verdict = (submission.get("job") or {}).get("verdict") or {}
    brief = {"state": submission.get("state"), "reason": submission.get("reason")}
    print(json.dumps({**brief, "crown": verdict.get("crown"), "seconds": out["seconds"]}))
    return out  # also on failure: the local entrypoint keeps it, then exits nonzero


@app.function(**CPU_FUNCTION, volumes={SNAPSHOT: snapshot.with_mount_options(read_only=True)})
def calibration(
    blocks: int = 2, seed: str = "calibration-pilot", cells: str = "", moe_backend: str = "cutlass"
) -> dict:
    import asyncio
    import math
    from types import SimpleNamespace

    from opentype_challenge import runtime, sandbox
    from opentype_challenge.worker import Worker

    spec = json.loads(cells) if cells else CELLS
    parsed = {
        name: runtime.Cell(c["track"], int(c["cases"]), int(c["concurrency"]),
                           float(c["slo_ms"]), 1.0 / len(spec), bool(c.get("warm", False)))
        for name, c in spec.items()
    }  # fmt: skip
    missing = set(runtime.CELL_TRACKS) - {c.track for c in parsed.values()}
    over = [
        n
        for n, c in parsed.items()
        if not 1 <= c.cases <= MAX_CASES
        or not 1 <= c.concurrency <= MAX_CONCURRENCY
        or c.track not in runtime.CELL_TRACKS
    ]
    if missing or over or len(parsed) > 8 or not 1 <= blocks <= 3:
        raise SystemExit(
            f"cells must cover {runtime.CELL_TRACKS} (at most 8, cases 1..{MAX_CASES}, "
            f"concurrency 1..{MAX_CONCURRENCY}; refused: {over}); --blocks 1..3"
        )
    if moe_backend not in runtime.MOE_BACKENDS:
        raise SystemExit(f"--moe-backend one of {runtime.MOE_BACKENDS}")
    model = _directory()
    if sandbox.weights_identity(model)["weights"] != "modelopt-nvfp4":
        raise SystemExit("run modal_runtime.py::stage first: the snapshot is not NVFP4")
    backend = sandbox.ModalBackend(app, image, snapshot, Path(SNAPSHOT), timeout=SESSION)
    launcher = sandbox.SandboxLauncher(backend, ready_timeout=READY)
    profile = {**runtime.PROFILE_FIXED, "moe_backend": moe_backend}
    argv, share = runtime.serving_argv(profile), float(profile["gpu_memory_utilization"])
    worker = Worker(None, None, launcher)  # type: ignore[arg-type]
    job = {"runtime": {"seed": seed}}
    cal = SimpleNamespace(cells=parsed)  # only the cells: no thresholds exist yet
    cases = {name: [runtime.cell_case(seed, name, cell, i) for i in range(cell.cases)]
             for name, cell in parsed.items()}  # fmt: skip
    runs: list[dict] = []
    raw: list[dict] = []

    async def session(block: int, side: str) -> None:
        started = time.monotonic()
        async with asyncio.timeout(SESSION):
            async with launcher({"champion": model}, {"champion": argv}, share) as urls:
                ready = time.monotonic() - started
                measured = launcher.profile()
                wrong = sorted(k for k in profile if measured.get(k) != profile[k])
                if runtime.GPU_TYPE not in str(measured.get("gpu")):
                    wrong.append("gpu")
                if runs:  # every run on the first run's build and card type
                    wrong += [k for k in runtime.MEASURED
                              if measured.get(k) != runs[0]["profile"].get(k)]  # fmt: skip
                if wrong:  # before a single case: nothing measured here could calibrate
                    raise RuntimeError(f"run {block}/{side} is off the profile: {wrong}")
                seconds, tasks = await worker._measure(job, cal, urls["champion"])  # type: ignore[arg-type]
        cells_out = {}
        for name, cell in parsed.items():
            mine = [t for t in tasks if t["cell"] == name]
            ok = [runtime.task_ok(cases[name][t["case_index"]], t) for t in mine]
            ms = sorted(t["ms"] for t in mine)
            under = sum(o and t["ms"] <= cell.slo_ms for o, t in zip(ok, mine, strict=True))
            cells_out[name] = {
                "tasks": len(mine), "ok": sum(ok), "ok_under_slo": under,
                "errors": sum("error" in t for t in mine), "seconds": round(seconds[name], 3),
                "p95_ms": ms[max(math.ceil(0.95 * len(ms)) - 1, 0)] if ms else None,
                "goodput": under / seconds[name] if seconds[name] > 0 else 0.0,
            }  # fmt: skip
        runs.append(
            {
                "block": block,
                "side": side,
                "ready_s": round(ready, 1),
                "cells": cells_out,
                "profile": measured,
                "placement": launcher.placements[-1],
                "quiescent": launcher.quiescent(),
            }
        )
        raw.append({"block": block, "side": side, "tasks": tasks})
        ok = {n: c["ok"] for n, c in cells_out.items()}
        line = {"progress": "run", "block": block, "side": side, "ok": ok}
        print(json.dumps({**line, "ready_s": round(ready, 1)}), flush=True)

    async def go() -> None:
        for block in range(blocks):
            for side in runtime.SIDES:  # B, then the second stock run C, then B'
                await session(block, side)

    error = None
    try:
        asyncio.run(go())
    except BaseException as caught:  # noqa: BLE001 - a builtin error for the local client
        error = f"{type(caught).__name__}: {caught}"[:2000]
    zero = sorted({(r["block"], r["side"], n) for r in runs
                   for n, c in r["cells"].items() if c["goodput"] <= 0})  # fmt: skip
    goodput = {
        (r["block"], r["side"], n): c["goodput"] for r in runs for n, c in r["cells"].items()
    }
    drift = {  # |ln(goodput B / goodput B')| per block, as the verdict's max_drift reads it
        name: [
            abs(math.log(goodput[(k, "B", name)] / goodput[(k, "B2", name)]))
            if goodput.get((k, "B", name), 0) > 0 and goodput.get((k, "B2", name), 0) > 0
            else None
            for k in range(blocks)
        ]
        for name in parsed
    }
    decision = "incomplete" if error else ("no_decision" if zero else "measured")
    summary = {
        "decision": decision,
        "reason": f"zero goodput in {zero}" if zero else None,
        "cells": spec, "seed": seed, "blocks": blocks, "moe_backend": moe_backend,
        "runs": runs, "log_drift_b_b2": drift, "error": error,
        "failures": launcher.failures[-4:],
    }  # fmt: skip
    summary["quiescent"] = launcher.quiescent()
    print(json.dumps({"decision": decision, "reason": summary["reason"], "error": error}))
    return {**summary, "raw": raw}  # also on failure: kept locally, then a nonzero exit


def _write_private(out: str, result: dict) -> None:
    import os

    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(result, handle)
    print(f"wrote {out}")
    if result.get("error"):
        raise SystemExit(f"the pilot failed (report kept in {out}): {result['error'][:300]}")


@app.local_entrypoint()
def quality_report(out: str, cases: str = "decisions=32,longctx=8,ops=8,sql=8") -> None:
    _write_private(out, quality.remote(cases=cases))


@app.local_entrypoint()
def calibration_report(out: str, blocks: int = 2, seed: str = "calibration-pilot") -> None:
    _write_private(out, calibration.remote(blocks=blocks, seed=seed))
