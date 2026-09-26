"""CPU-only check of the controller's deployed-app guard (modal_controller._drain), on its own
app name: no secret, no volume, no GPU.

    modal deploy deploy/modal_guard_probe.py
    python -c "import modal; print(modal.Function.from_name(
        'opentype-guard-probe', 'probe').remote())"
    modal run deploy/modal_guard_probe.py::probe      # the ephemeral app: must differ
    modal app stop opentype-guard-probe

Expected: the deployed call prints same=True and the `modal run` prints same=False.
"""

import modal

app = modal.App("opentype-guard-probe")


@app.function(cpu=0.25, memory=256, timeout=120, max_containers=1)
def probe() -> dict:
    deployed = modal.App.lookup(app.name).app_id
    result = {"running": app.app_id, "deployed": deployed, "same": app.app_id == deployed}
    print(result)
    return result
