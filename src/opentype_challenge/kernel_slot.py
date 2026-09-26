"""The registered kernel slot: a miner's Triton RMSNorm as vLLM IR provider "opentype".

Only ever imported inside a runtime sandbox (sandbox.py), never by the container or the worker:
it executes the miner's file. vLLM loads it in every process through the
`vllm.general_plugins` entry point; it does nothing unless the sandbox bootstrap set
OPENTYPE_KERNEL_FILE and OPENTYPE_KERNEL_SHA256, and then the served argv selects it with
`--ir-op-priority {"rms_norm": ["opentype"]}` (platform defaults are appended after it).

The slot is vllm.ir.ops.rms_norm at the pinned nightly (vllm/ir/ops/layernorm.py): Gemma4's
RMSNorm layers (DiffusionGemma's text stack) call it for every input, attention, feed-forward
and q/k norm. The provider mirrors vLLM's own Triton provider for gelu_and_mul_sparse:
torch.library.triton_op + wrap_triton, so it runs eager and under the inductor lowering pass.
No `from __future__ import annotations` here: triton_op infers the op schema from the real
Tensor annotations of the nested functions, which string annotations would hide.
"""

import hashlib
import importlib.util
import os
import sys
from typing import Any

SLOT = "rms_norm"
PROVIDER = "opentype"
KERNEL = "rms_norm_kernel"
MAX_COLS = 65_536


def load(path: str, sha256: str) -> Any:
    """The miner module, after its bytes match the signed digest (the file is root-owned and
    read-only; the check guards a wrong mount, not the miner)."""
    data = open(path, "rb").read()  # noqa: PTH123
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError("the kernel file does not match its signed sha256")
    spec = importlib.util.spec_from_file_location("opentype_miner_kernel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def num_warps(module: Any, block: int) -> int:
    fixed = getattr(module, "NUM_WARPS", None)
    return int(fixed) if fixed else min(max(block // 256, 1), 8)


def register() -> None:
    """vllm.general_plugins entry point."""
    path, sha256 = os.environ.get("OPENTYPE_KERNEL_FILE"), os.environ.get("OPENTYPE_KERNEL_SHA256")
    if not path or not sha256:
        return
    import torch
    from torch import Tensor
    from torch.library import triton_op, wrap_triton
    from vllm import ir

    if PROVIDER in ir.ops.rms_norm.impls:  # plugins load once per process; be idempotent
        return
    module = load(path, sha256)
    kernel = getattr(module, KERNEL)
    import triton

    def supports(
        x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
    ) -> bool:
        return (
            variance_size is None
            and weight is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and weight.dtype == x.dtype
            and weight.is_contiguous()
            and x.dim() >= 1
            and x.stride(-1) == 1
            and 0 < x.shape[-1] <= MAX_COLS
            and weight.shape == (x.shape[-1],)
        )

    @triton_op("opentype::rms_norm", mutates_args=())
    def _op(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
        n = x.shape[-1]
        rows = x.reshape(-1, n)  # a copy only for non-viewable strides
        out = torch.empty_like(rows)
        if rows.shape[0]:
            block = triton.next_power_of_2(n)
            wrap_triton(kernel)[(rows.shape[0],)](
                rows,
                weight,
                out,
                rows.stride(0),
                out.stride(0),
                n,
                epsilon,
                BLOCK_SIZE=block,
                num_warps=num_warps(module, block),
            )
        return out.reshape(x.shape)

    @ir.ops.rms_norm.register_impl(PROVIDER, supports_args=supports)
    def _impl(
        x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
    ) -> Tensor:
        assert weight is not None and variance_size is None
        return _op(x, weight, epsilon)


def compile_for(source_path: str, sha256: str, arch: int) -> int:
    """Compile the kernel for one CUDA arch with no GPU (the build sandbox): the pinned Triton
    must accept it for bf16 at the model's widths. Returns the cubin size."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    module = load(source_path, sha256)
    kernel = getattr(module, KERNEL)
    signature = {
        "x_ptr": "*bf16",
        "w_ptr": "*bf16",
        "out_ptr": "*bf16",
        "x_row_stride": "i64",
        "out_row_stride": "i64",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_SIZE": "constexpr",
    }
    size = 0
    # hidden 2816 -> 4096; head dims 256 and 512 (q/k norms)
    for block in (256, 512, 4096):
        source = ASTSource(kernel, signature, constexprs={"BLOCK_SIZE": block})
        compiled = triton.compile(
            source,
            target=GPUTarget("cuda", arch, 32),
            options={"num_warps": num_warps(module, block)},
        )
        size += len(compiled.asm["cubin"])
    return size


if __name__ == "__main__":  # python -m opentype_challenge.kernel_slot <file> <sha256> <arch>
    print(compile_for(sys.argv[1], sys.argv[2], int(sys.argv[3])), flush=True)
