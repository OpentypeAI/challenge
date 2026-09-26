"""The kernel smoke's sources (no modal import: deploy/kernel_cpu_check.py reads them too)."""

# Known-correct RMSNorm for the slot (the same math as vllm.ir.ops.rms_norm: fp32 statistics,
# weight in the input dtype); CONTROL writes zeros, so a run with it must visibly break.
RMS_KERNEL = """import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    # native: x.to(weight.dtype) * weight, i.e. y rounded to the dtype, then one rounded
    # product; the fp32 product of the rounded values, rounded once by the store, is that
    y = (x * tl.rsqrt(var + eps)).to(out_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, y * w, mask=mask)
"""
CONTROL_KERNEL = RMS_KERNEL.replace("y * w, mask=mask", "y * w * 0.0, mask=mask")
assert CONTROL_KERNEL != RMS_KERNEL
# One rounding at the store instead of two: what a compiler that drops the intermediate cast
# (inductor without emulate_precision_casts) makes of native. Which one stock actually runs
# on B300 is what kernel_op_probe measures; the CPU interpreter cannot (it truncates casts).
SINGLE_KERNEL = RMS_KERNEL.replace(".to(out_ptr.dtype.element_ty).to(tl.float32)", "")
assert SINGLE_KERNEL != RMS_KERNEL
