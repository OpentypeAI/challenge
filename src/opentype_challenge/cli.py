"""opentype-challenge: serve | worker | generate | audit | miner submit | miner status."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__, bank, generator, harness, tracks
from .generator import Case


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .app import Config, create_app

    app = create_app(Config.from_env())
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, access_log=False)


def _worker(args: argparse.Namespace) -> None:
    import httpx

    from . import pins
    from .worker import STRUCTURED_SERVER, Api, VllmLauncher, Worker, sha256_file

    if sha256_file(STRUCTURED_SERVER) != pins.STRUCTURED_SERVER_SHA256:
        raise SystemExit(f"{STRUCTURED_SERVER} does not match the pinned structured_server.py")
    token = Path(args.token_file).read_text().strip()

    async def main() -> None:
        async with httpx.AsyncClient() as client:
            worker = Worker(
                Api(args.api, token, client),
                Path(args.workdir),
                VllmLauncher(
                    canvas=args.canvas,
                    max_model_len=args.max_model_len,
                    log_dir=Path(args.workdir),
                ),
                concurrency=args.concurrency,
            )
            if args.once:
                await worker.run_once()
            else:
                await worker.run_forever(until_empty=args.until_empty)

    asyncio.run(main())


def _oracle(case: Case) -> list[str] | None:
    """The reference policy's transcript of a harness case, or None when it has no oracle."""
    policy = getattr(tracks.BUILDERS[case.track], "reference_policy", None)
    if policy is None:
        return None

    async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
        return str(policy(messages))

    try:
        return asyncio.run(harness.run_episode(tracks.ENVS[case.track], case.body, generate))
    except ValueError:
        return None


def _public_cases(track: str, level: int | None, n: int, seed: int) -> Iterator[Case]:
    """Public cases of a non-decisions track, built with the empty bank from a public seed."""
    builder = tracks.BUILDERS[track]
    levels = builder.buildable(bank.EMPTY_BANK)
    if level is not None and level not in levels:
        raise SystemExit(f"{track} builds levels {list(levels)} without a teacher bank")
    for index in range(n):
        rng = random.Random(f"opentype-train|{seed}|{track}|{level or '*'}|{index}")
        yield builder.make_case(rng, level or rng.choice(levels), bank.EMPTY_BANK)


def _generate(args: argparse.Namespace) -> None:
    if args.track == "decisions":
        if args.level is None:
            raise SystemExit("decisions needs --level")
        cases: Iterator[Case] = iter(generator.generate(args.family, args.level, args.n, args.seed))
    else:
        cases = _public_cases(args.track, args.level, args.n, args.seed)
    out = open(args.out, "w") if args.out != "-" else sys.stdout  # noqa: SIM115
    try:
        for case in cases:
            row: dict[str, Any] = {
                "track": case.track,
                "family": case.family,
                "level": case.level,
                "request": case.body,
            }
            if case.track in tracks.ENVS:
                row["oracle"] = _oracle(case)
            else:
                row["gold"] = {qid: g.to_json() for qid, g in case.gold.items()}
            out.write(json.dumps(row, separators=(",", ":")) + "\n")  # key order is meaningful
    finally:
        if out is not sys.stdout:
            out.close()


def _fetch_bank(api: str, window: int) -> bank.Bank:
    """A revealed window's bank, read page by page from the public API."""
    import httpx

    rows: list[Any] = []
    while True:
        page = httpx.get(
            f"{api}/v1/windows/{window}/bank", params={"offset": len(rows)}, timeout=60
        ).json()
        rows += page["items"]
        if not page["items"] or len(rows) >= page["total"]:
            return bank.Bank.from_json(rows)


def _audit_job(secret: bytes, job: Mapping[str, Any], window_bank: bank.Bank) -> dict[str, Any]:
    plan = {
        t: tracks.TrackPlan(float(p["weight"]), int(p["cases"])) for t, p in job["plan"].items()
    }
    seed = bank.job_seed(secret, job["id"], job["digest"], job.get("beacon"))
    count = job.get("cases_fetched") or job["cases"]
    digest = bank.cases_digest(
        tracks.job_case(seed, plan, job["mix"], index, window_bank, bool(job.get("judge"))).body
        for index in range(count)
    )
    expected = job.get("cases_sha256")
    return {
        "job": job["id"],
        "cases": count,
        "cases_sha256": digest,
        "published": expected,
        "ok": expected is None or expected == digest,
    }


def _audit(args: argparse.Namespace) -> None:
    """Regenerate a revealed window's duel cases with its bank, beacon, judge flag and plan,
    and check them against the published digests."""
    import httpx

    secret = bytes.fromhex(args.window_secret)
    if args.commitment and bank.commitment(secret) != args.commitment:
        raise SystemExit("the secret does not match the commitment")
    jobs: Sequence[Mapping[str, Any]]
    if args.api:
        api = args.api.rstrip("/")
        window = httpx.get(f"{api}/v1/windows/{args.window}", timeout=60).json()
        if bank.commitment(secret) != window["commitment"]:
            raise SystemExit("the secret does not match the published commitment")
        window_bank = _fetch_bank(api, args.window)
        if window_bank.digest != window["bank_digest"]:
            raise SystemExit("the bank does not match the published bank digest")
        jobs = window.get("jobs", [])
    else:
        window_bank = bank.EMPTY_BANK
        if args.bank_file:
            window_bank = bank.Bank.from_json(json.loads(Path(args.bank_file).read_text()))
        plan = (
            json.loads(args.plan)
            if args.plan
            else {"decisions": {"weight": 1.0, "cases": args.cases}}
        )
        jobs = [
            {
                "id": args.job,
                "digest": args.digest,
                "mix": json.loads(args.mix),
                "plan": plan,
                "beacon": json.loads(args.beacon) if args.beacon else None,
                "judge": args.judge,
                "cases": args.cases,
                "cases_sha256": None,
                "cases_fetched": args.cases,
            }
        ]
    failed = False
    for job in jobs:
        line = _audit_job(secret, job, window_bank)
        failed |= not line["ok"]
        print(json.dumps(line))
    if failed:
        raise SystemExit(1)


def _miner_submit(args: argparse.Namespace) -> None:
    from . import miner

    if args.seed_file:
        signer = miner.seed_signer(Path(args.seed_file).read_text())
    elif args.wallet_name and args.wallet_hotkey:
        signer = miner.wallet_signer(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    else:
        raise SystemExit("pass --seed-file or --wallet-name and --wallet-hotkey")
    manifest = miner.hf_manifest(args.repo, args.revision)
    result = miner.post(args.api, miner.signed_submission(manifest, signer))
    print(result["id"])


def _miner_status(args: argparse.Namespace) -> None:
    from . import miner

    print(json.dumps(miner.status(args.api, args.id), indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="opentype-challenge")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the challenge container")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container network only
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(run=_serve)

    worker = sub.add_parser("worker", help="run the B300 duel worker")
    worker.add_argument("--api", required=True, help="https://<gateway>/challenge/opentype")
    worker.add_argument("--token-file", required=True)
    worker.add_argument("--workdir", required=True)
    worker.add_argument("--canvas", type=int, default=256)
    worker.add_argument("--max-model-len", type=int, default=131072)
    worker.add_argument("--concurrency", type=int, default=64)
    mode = worker.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run at most one job")
    mode.add_argument(
        "--until-empty", action="store_true", help="run jobs until the queue is empty, then exit"
    )
    worker.set_defaults(run=_worker)

    gen = sub.add_parser("generate", help="public training cases (exact targets or oracles)")
    gen.add_argument("--track", choices=tracks.TRACKS, default="decisions")
    gen.add_argument("--family", choices=sorted(generator.FAMILY_BY_NAME))
    gen.add_argument("--level", type=int)
    gen.add_argument("--n", type=int, required=True)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", default="-")
    gen.set_defaults(run=_generate)

    audit = sub.add_parser("audit", help="regenerate a revealed window's duel cases")
    audit.add_argument("--window-secret", required=True)
    audit.add_argument("--commitment")
    audit.add_argument("--api", help="read the window's jobs from the public API")
    audit.add_argument("--window", type=int)
    audit.add_argument("--job")
    audit.add_argument("--digest")
    audit.add_argument("--mix", default='{"1": 1.0}')
    audit.add_argument("--cases", type=int, default=10)
    audit.add_argument("--plan", help="the job's plan as JSON {track: {weight, cases}}")
    audit.add_argument("--beacon", help="the job's drand beacon as JSON {round, randomness}")
    audit.add_argument("--judge", action="store_true", help="the job had a judge")
    audit.add_argument("--bank-file", help="the window's bank rows as JSON")
    audit.set_defaults(run=_audit)

    miner = sub.add_parser("miner", help="submit and track a model")
    msub = miner.add_subparsers(dest="miner_command", required=True)
    submit = msub.add_parser("submit")
    submit.add_argument("--api", required=True)
    submit.add_argument("--repo", required=True)
    submit.add_argument("--revision", required=True)
    submit.add_argument("--seed-file")
    submit.add_argument("--wallet-name")
    submit.add_argument("--wallet-hotkey")
    submit.add_argument("--wallet-path")
    submit.set_defaults(run=_miner_submit)
    stat = msub.add_parser("status")
    stat.add_argument("--api", required=True)
    stat.add_argument("--id", required=True)
    stat.set_defaults(run=_miner_status)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.run(args)
