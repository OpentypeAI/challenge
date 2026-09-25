"""OpenType duel worker on Modal: 1x H200 (141 GB), both models at 0.45 of the card.

    modal secret create opentype-worker OPENTYPE_WORKER_TOKEN=<worker.token>
    modal deploy deploy/modal_worker.py

A cron wakes the worker every 10 minutes. It runs duels until the queue is empty and
exits, so the GPUs are billed only while a duel runs. The work volume keeps the champion's
weights between runs. OPENTYPE_API (at deploy time) is the master's /challenge/opentype URL.
"""

import os
import subprocess
import tempfile

import modal

API = os.environ.get("OPENTYPE_API", "https://chain.joinbase.ai/challenge/opentype")

app = modal.App("opentype-worker")
image = (
    modal.Image.from_registry(
        "ghcr.io/opentypeai/challenge-worker:stable",
        # Modal runs its own agent with `python`; the vLLM base only ships `python3`.
        setup_dockerfile_commands=["RUN ln -sf $(command -v python3) /usr/local/bin/python"],
    )
    .entrypoint([])
    .env({"OPENTYPE_API": API})
)
work = modal.Volume.from_name("opentype-worker-work", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,  # 512 GiB: champion + challenger + base support files
    volumes={"/work": work},
    secrets=[modal.Secret.from_name("opentype-worker")],
    timeout=24 * 3600,
    max_containers=1,
    schedule=modal.Cron("*/10 * * * *"),
)
def duel() -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write(os.environ["OPENTYPE_WORKER_TOKEN"])
    os.chmod(handle.name, 0o400)
    try:
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["opentype-challenge", "worker", "--api", os.environ["OPENTYPE_API"],  # noqa: S607
             "--token-file", handle.name, "--workdir", "/work", "--until-empty"],
            check=True,
        )  # fmt: skip
    finally:
        work.commit()
