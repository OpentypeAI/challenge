"""opentype-challenge: serve | worker | generate | audit | miner submit | miner status."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__, bank, generator


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
                VllmLauncher(canvas=args.canvas, log_dir=Path(args.workdir)),
                concurrency=args.concurrency,
            )
            if args.once:
                await worker.run_once()
            else:
                await worker.run_forever()

    asyncio.run(main())


def _generate(args: argparse.Namespace) -> None:
    out = open(args.out, "w") if args.out != "-" else sys.stdout  # noqa: SIM115
    try:
        for case in generator.generate(args.family, args.level, args.n, args.seed):
            row = {
                "family": case.family,
                "level": case.level,
                "request": case.body,
                "gold": {qid: g.to_json() for qid, g in case.gold.items()},
            }
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


def _audit(args: argparse.Namespace) -> None:
    """Regenerate a revealed window's duel cases and check them against the published digests."""
    import httpx

    secret = bytes.fromhex(args.window_secret)
    if args.commitment and bank.commitment(secret) != args.commitment:
        raise SystemExit("the secret does not match the commitment")
    if args.api:
        window = httpx.get(f"{args.api.rstrip('/')}/v1/windows/{args.window}", timeout=60).json()
        if bank.commitment(secret) != window["commitment"]:
            raise SystemExit("the secret does not match the published commitment")
        jobs = window.get("jobs", [])
    else:
        jobs = [
            {
                "id": args.job,
                "digest": args.digest,
                "mix": json.loads(args.mix),
                "cases": args.cases,
                "cases_sha256": None,
                "cases_fetched": args.cases,
            }
        ]
    failed = False
    for job in jobs:
        seed = bank.job_seed(secret, job["id"], job["digest"])
        count = job.get("cases_fetched") or job["cases"]
        digest = bank.cases_digest(
            bank.job_case(seed, job["mix"], index).body for index in range(count)
        )
        expected = job.get("cases_sha256")
        ok = expected is None or expected == digest
        failed |= not ok
        print(
            json.dumps(
                {
                    "job": job["id"],
                    "cases": count,
                    "cases_sha256": digest,
                    "published": expected,
                    "ok": ok,
                }
            )
        )
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
    worker.add_argument("--concurrency", type=int, default=64)
    worker.add_argument("--once", action="store_true", help="run at most one job")
    worker.set_defaults(run=_worker)

    gen = sub.add_parser("generate", help="public training cases with exact soft targets")
    gen.add_argument("--family", choices=sorted(generator.FAMILY_BY_NAME))
    gen.add_argument("--level", type=int, choices=sorted(generator.LEVELS), required=True)
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
