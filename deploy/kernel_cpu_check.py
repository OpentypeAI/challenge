"""CPU check of the kernel slot in the pinned vLLM image, before any GPU is spent.

    docker run --rm --network none --entrypoint bash -v "$PWD":/src:ro <VLLM_IMAGE> -c \
      'pip install -q --no-deps --no-index /src/dist/*.whl && \
       python3 /src/deploy/kernel_cpu_check.py'

It writes the smoke's kernels (the double-rounding "correct", the single-rounding "single"
and the zero "control"), then for each one, as vllm serve would: loads the
general plugins through vllm's own loader with the sandbox's environment, checks the slot is
registered and its op schema inferred, runs the registered op under the Triton interpreter
against vllm's native rms_norm (correct and single within 4 units of |ref| * 2^-7, the
zero control at least 64 off; the interpreter truncates casts, so this bounds the math, not
the bit pattern B300 produces), parses the worker's --ir-op-priority, and runs
`vllm serve --help` with the plugin enabled. No GPU, no network; VLLM_TARGET_DEVICE=cpu
only because this image cannot infer a device without one. Install the package from a
wheel (`uv build --wheel`): the source install needs the network for its build backend.
Prints one JSON line; exits 1 on any failure.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = r"""
import json, os, sys
import torch
from vllm import ir
from vllm.engine.arg_utils import EngineArgs
from vllm.plugins import load_general_plugins
from vllm.utils.argparse_utils import FlexibleArgumentParser

load_general_plugins()
impls = sorted(ir.ops.rms_norm.impls)
schema = str(torch.ops.opentype.rms_norm.default._schema)
worst, nan = 0.0, 0
for n in (256, 512, 2816):
    torch.manual_seed(n)
    x = (torch.randn(8, n) * 3).to(torch.bfloat16)
    w = (torch.randn(n) + 1).to(torch.bfloat16)
    out = torch.ops.opentype.rms_norm(x, w, 1e-6).float()
    ref = ir.ops.rms_norm.impls["native"].impl_fn(x, w, 1e-6).float()
    # every element finite before any difference: max() over floats silently skips NaN
    nan += int((~out.isfinite()).sum()) + int((~ref.isfinite()).sum())
    if nan:
        worst = float("inf")
        continue
    # in units of one bf16 ulp of the reference (2^-7 relative)
    ulp = (ref.abs() * 2.0**-7).clamp_min(2.0**-126)
    worst = max(worst, float(((out - ref).abs() / ulp).max()))
args = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(
    ["--ir-op-priority", json.dumps({"rms_norm": ["opentype"]})]
)
priority = EngineArgs.from_cli_args(args).ir_op_priority.rms_norm
print(json.dumps({"impls": impls, "schema": schema, "nonfinite": nan, "worst_ulp": worst,
                  "priority": priority}))
"""


def main() -> int:
    sys.path.insert(0, str(HERE))
    import modal_runtime_kernels as kernels  # the smoke's sources, without modal

    results, ok = {}, True
    with tempfile.TemporaryDirectory() as tmp:
        variants = (
            ("correct", kernels.RMS_KERNEL),
            ("single", kernels.SINGLE_KERNEL),
            ("control", kernels.CONTROL_KERNEL),
        )
        for name, source in variants:
            path = Path(tmp) / f"{name}.py"
            path.write_text(source)
            env = {
                **os.environ,
                "VLLM_PLUGINS": "opentype_kernel",
                "OPENTYPE_KERNEL_FILE": str(path),
                "OPENTYPE_KERNEL_SHA256": hashlib.sha256(source.encode()).hexdigest(),
                "TRITON_INTERPRET": "1",
                # this image cannot infer a device without a GPU; argument parsing needs one
                "VLLM_TARGET_DEVICE": "cpu",
            }
            run = subprocess.run(  # noqa: S603 - our own interpreter and script
                [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=900
            )
            lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
            row = json.loads(lines[-1]) if run.returncode == 0 and lines else {}
            row["rc"] = run.returncode
            if run.returncode:
                row["stderr"] = run.stderr[-800:]
            cli = subprocess.run(  # noqa: S603 - the image's vllm CLI
                ["vllm", "serve", "--help"],  # noqa: S607 - the image's PATH
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
            row["serve_help_rc"] = cli.returncode
            good = (
                row["rc"] == 0
                and cli.returncode == 0
                and "opentype" in row.get("impls", [])
                and row.get("schema", "").startswith("opentype::rms_norm(Tensor x, Tensor weight")
                and row.get("priority") == ["opentype"]
            )
            if name in ("correct", "single"):
                # fp32 reduction order may round y to the neighbouring bf16 before the weight:
                # a few units of |ref| * 2^-7, never more
                good = good and row.get("nonfinite") == 0 and row.get("worst_ulp", 99) <= 4.0
            else:
                # zeros are 128 units off: the slot's output must visibly change
                good = good and row.get("nonfinite") == 0 and row.get("worst_ulp", 0) >= 64
            row["ok"] = good
            ok = ok and good
            results[name] = row
    print(json.dumps({"ok": ok, **results}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
