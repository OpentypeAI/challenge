"""The B300 worker for both lanes: a CPU controller on Modal, every GPU run a fresh sandbox.

    modal secret create opentype-worker OPENTYPE_WORKER_TOKEN=<worker.token>
    modal deploy deploy/modal_controller.py
    modal run deploy/modal_controller.py::quality   # or ::runtime; drains the queue, exits

The controller is a CPU Function. It holds the worker token (its own Modal secret) and a
work volume that only it writes. For each job it:
  - leases the job;
  - downloads and sha256-verifies the weights into the volume, and commits it;
  - serves each side in a fresh sandbox (opentype_challenge.sandbox.ModalBackend) and relays
    the reads over the sandbox's stdio;
  - posts the raw answers.

Each sandbox runs with gpu="B300", block_network, secrets=[], no OIDC token, and only that
side's model directory mounted read-only. Nothing writes the volume while a sandbox serves,
because the runs are sequential.

  quality   NVFP4 duels only (it leases with nvfp4=true): champion then challenger, each
            alone at the pinned quality serving config (runtime.QUALITY_SERVING).
  runtime   the kernel/option lane: a build sandbox, stock and candidate fidelity, then B/C/B'
            blocks at the calibrated profile.

One container per lane (max_containers=1), each lane with its own volume, so each volume has
exactly one writer. No schedule is set here: the operator enables a cron only after the B300
validation (docs/operator.md). After the NVFP4 migration, the H200 worker
(deploy/modal_worker.py) leases nothing, since quality leases are gated on the champion's
format. Stop it then (`modal app stop opentype-worker`).
"""

import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from modal_runtime import BASE_IMAGE, image  # noqa: E402 - the one pinned CUDA 13.0 image

API = os.environ.get("OPENTYPE_API", "https://chain.joinbase.ai/challenge/opentype")
WORK = "/work"

app = modal.App("opentype-controller")
controller_image = image.env({"OPENTYPE_API": API, "OPENTYPE_WORKER_IMAGE": BASE_IMAGE})
volumes = {
    lane: modal.Volume.from_name(f"opentype-controller-{lane}", create_if_missing=True)
    for lane in ("quality", "runtime")
}


def _drain(lane: str) -> None:
    import asyncio

    import httpx

    from opentype_challenge import sandbox
    from opentype_challenge.worker import Api, Worker

    token = os.environ.pop("OPENTYPE_WORKER_TOKEN")  # this process only; never a sandbox's
    backend = sandbox.ModalBackend(app, controller_image, volumes[lane], Path(WORK), commit=True)

    async def main() -> None:
        async with httpx.AsyncClient() as client:
            worker = Worker(
                Api(os.environ["OPENTYPE_API"], token, client),
                Path(WORK),
                sandbox.SandboxLauncher(backend),
                lane=lane,
            )
            await worker.run_forever(until_empty=True)

    try:
        asyncio.run(main())
    finally:
        volumes[lane].commit()


CONTROLLER = {
    "image": controller_image,
    "cpu": 4,
    "memory": 16384,
    "timeout": 24 * 3600,
    "secrets": [modal.Secret.from_name("opentype-worker")],
    "max_containers": 1,
}


@app.function(**CONTROLLER, volumes={WORK: volumes["quality"]})
def quality() -> None:
    _drain("quality")


@app.function(**CONTROLLER, volumes={WORK: volumes["runtime"]})
def runtime() -> None:
    _drain("runtime")
