"""The runtime lane: vLLM options, a registered kernel slot, the calibration and the verdict.

The lane pays for serving the quality champion's weights faster, on B300 and NVFP4 weights
only. A miner submits options from OPTIONS and, where the calibration opens it, one Triton
kernel for a registered slot (KERNEL_SLOTS); never argv, env, image, reader or a plugin of
their own. A trusted controller runs blocks of incumbent B / candidate C / incumbent B', each
run in a fresh network-blocked, secret-free GPU sandbox (sandbox.py), and reports each timed
task's latency (its own clock) and raw output plus each run's monotonic seconds. The container
scores every output against its own gold (task_ok), measures how far the candidate's answers
drift from a pristine stock reference's (divergence), and recomputes the verdict here; no
success flag or count from the worker or the miner is ever evidence.

Nothing is decided before the operator publishes a calibration (reference-vs-reference
pilot on the pinned profile): without it the lane stays closed and no timing is accepted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import pins
from .generator import Case

LANES = ("quality", "runtime")
# Budgets in ledger units (1e9 = one epoch-mass), fixed by the challenge: an unused lane burns.
BUDGETS = {"quality": 750_000_000, "runtime": 250_000_000}
SIDES = ("B", "C", "B2")
# Registered kernel slots: the vLLM IR op each replaces, as provider "opentype"
# (kernel_slot.py). Only slots a calibration lists accept kernels.
KERNEL_SLOTS = ("rms_norm",)
KERNEL_MAX_BYTES = 48 * 1024
# What a kernel file may import: Triton only. The file runs only inside a sandbox; this check
# keeps obvious non-kernels out of the queue, it is not the security boundary.
KERNEL_IMPORTS = ("triton", "triton.language", "math")
READ_TRACKS = ("decisions", "longctx")
# Operational caps on a calibration (an oversized one would hold the exclusive GPU for days):
# MAX_BLOCKS matches the timings API's block bound.
MAX_BLOCKS = 999
MAX_CELL_CASES = 10_000
MAX_CONCURRENCY = 1024
MAX_RESAMPLES = 100_000
MAX_CELL_NAME = 64  # the timings API's cell bound
CELL_TRACKS = ("decisions", "longctx", "ops", "sql")  # no judge in the runtime lane

# Options checked against vllm/engine/arg_utils.py and vllm/config/scheduler.py at the pinned
# nightly 7f1a5398 (pins.VLLM_IMAGE): each is a parsed `vllm serve` flag there. Their effect
# on DiffusionGemma is measured by the benchmark, never assumed. Anything else is refused.
# name -> (flag, kind, min, max)
OPTIONS: dict[str, tuple[str, type, int, int]] = {
    "max_num_seqs": ("--max-num-seqs", int, 1, 1024),
    "max_num_batched_tokens": ("--max-num-batched-tokens", int, 256, 1_048_576),
    "enable_chunked_prefill": ("--enable-chunked-prefill", bool, 0, 1),
    "enable_prefix_caching": ("--enable-prefix-caching", bool, 0, 1),
}

# The serving profile every measurement runs under. The sandbox bootstrap measures MEASURED
# inside each run's sandbox before any miner code runs, and reads vllm_image from the build
# manifest baked into the image; a calibration pins all of them, and a run whose identity
# differs is infrastructure doubt (no decision).
MEASURED = ("gpu", "driver", "vllm_version", "compute_cap")
GPU_TYPE = "B300"  # Modal's gpu= string; the measured name must contain it too
PROFILE_FIXED: dict[str, Any] = {
    "vllm_image": pins.VLLM_IMAGE,
    "structured_server_sha256": pins.STRUCTURED_SERVER_SHA256,
    "base": f"{pins.BASE_REPO}@{pins.BASE_REVISION}",  # tokenizer, template, processor
    # The champion must be a ModelOpt NVFP4 checkpoint of exactly this config and tensor
    # layout; activations, attention and the KV cache stay bfloat16, pinned by flag.
    "weights": "modelopt-nvfp4",
    "weights_config_sha256": pins.NVFP4_CONFIG_SHA256,
    "weights_schema_sha256": pins.NVFP4_SCHEMA_SHA256,
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "attention_backend": "TRITON_ATTN",
    "canvas": 256,
    "max_model_len": 131072,
    "gpu_memory_utilization": 0.9,
    "executor": "modal-sandbox",
    "gpu_type": GPU_TYPE,
}
# Pinned by the operator per calibration (chosen on the hardware), from this allowlist.
MOE_BACKENDS = ("flashinfer_trtllm", "flashinfer_cutlass", "cutlass")
# Quality duels on NVFP4 weights: each side alone in its own B300 sandbox with the runtime
# lane's pinned flags and memory share. A change is a new version, never an edit.
QUALITY_SERVING = {"version": "quality-b300-v1", "moe_backend": "cutlass"}
PROFILE_KEYS = (*PROFILE_FIXED, "moe_backend", *MEASURED)


class RuntimeError_(ValueError):
    """A refused option set, kernel or calibration."""


def serving_argv(profile: Mapping[str, Any]) -> list[str]:
    """The profile's own vllm flags, the same for every side: never auto-resolved."""
    return [
        "--kv-cache-dtype",
        str(profile["kv_cache_dtype"]),
        "--attention-backend",
        str(profile["attention_backend"]),
        "--moe-backend",
        str(profile["moe_backend"]),
    ]


def quality_serving() -> tuple[list[str], float]:
    """The vllm flags and memory share of a quality duel side on B300 (QUALITY_SERVING)."""
    profile = {**PROFILE_FIXED, "moe_backend": QUALITY_SERVING["moe_backend"]}
    return serving_argv(profile), float(PROFILE_FIXED["gpu_memory_utilization"])


def kernel_argv(kernel: Mapping[str, Any] | None) -> list[str]:
    """Select the registered provider for the kernel's slot (vllm --ir-op-priority)."""
    if not kernel:
        return []
    return ["--ir-op-priority", json.dumps({kernel["slot"]: ["opentype"]})]


def normalize_kernel(raw: Any) -> dict[str, Any] | None:
    """{slot, source, sha256} of a submitted kernel, or None. The source is parsed, never
    imported or executed here: only a sandbox ever runs it (kernel_slot.py)."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"slot", "source"}:
        raise RuntimeError_("kernel must be an object with slot and source")
    slot, source = raw["slot"], raw["source"]
    if slot not in KERNEL_SLOTS:
        raise RuntimeError_(f"kernel slot must be one of {KERNEL_SLOTS}")
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError_("kernel source must be a non-empty string")
    data = source.encode()
    if len(data) > KERNEL_MAX_BYTES:
        raise RuntimeError_(f"kernel source exceeds {KERNEL_MAX_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise RuntimeError_(f"kernel source does not parse: {error}") from None
    kernels = []
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if isinstance(node, ast.ImportFrom) and node.level:
                raise RuntimeError_("kernel source may not use relative imports")
            if any(n not in KERNEL_IMPORTS for n in names):
                raise RuntimeError_(f"kernel source may import only {KERNEL_IMPORTS}")
        elif isinstance(node, ast.FunctionDef):
            if node.name == "rms_norm_kernel":
                kernels.append(node)
        elif isinstance(node, ast.Assign):
            if not all(isinstance(t, ast.Name) for t in node.targets) or not isinstance(
                node.value, ast.Constant
            ):
                raise RuntimeError_("top-level assignments must bind names to constants")
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            raise RuntimeError_("kernel source may hold only imports, constants and functions")
        if isinstance(node, ast.FunctionDef):
            _plain_def(node)
    if len(kernels) != 1 or [ast.unparse(d) for d in kernels[0].decorator_list] != ["triton.jit"]:
        raise RuntimeError_("kernel source must define one @triton.jit rms_norm_kernel")
    return {"slot": slot, "source": source, "sha256": hashlib.sha256(data).hexdigest()}


def _dotted(node: ast.AST | None) -> bool:
    """A name or attribute chain (tl.constexpr), or nothing: evaluates without calls."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node is None or isinstance(node, ast.Name)


def _plain_def(node: ast.FunctionDef) -> None:
    """What `def` evaluates at import (decorators, defaults, annotations) runs no code."""
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    if (
        not all(_dotted(d) and not isinstance(d, ast.Call) for d in node.decorator_list)
        or not all(isinstance(d, ast.Constant) for d in [*args.defaults, *args.kw_defaults] if d)
        or not all(_dotted(a.annotation) for a in every if a is not None)
        or not _dotted(node.returns)
    ):
        raise RuntimeError_(
            f"{node.name}: decorators, defaults and annotations must be names or constants"
        )


def kernel_ref(kernel: Mapping[str, Any] | None) -> dict[str, str] | None:
    """What is signed, stored in digests and shown: the slot and the source digest."""
    return None if not kernel else {"slot": kernel["slot"], "sha256": kernel["sha256"]}


def normalize_candidate(options: Any, kernel: Any) -> tuple[dict[str, Any], dict | None]:
    """A runtime submission: options (possibly none) and at most one kernel, not both empty."""
    normalized_kernel = normalize_kernel(kernel)
    if normalized_kernel is not None and (options is None or options == {}):
        return {}, normalized_kernel
    return normalize_options(options), normalized_kernel


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_options(raw: Any) -> dict[str, Any]:
    """The options a miner may set, strictly typed and bounded; unknown keys are refused."""
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError_("options must be a non-empty object")
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in OPTIONS:
            raise RuntimeError_(f"option {name!r} is not allowed; allowed: {sorted(OPTIONS)}")
        _, kind, low, high = OPTIONS[name]
        if kind is bool:
            if not isinstance(value, bool):
                raise RuntimeError_(f"{name} must be true or false")
        elif isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise RuntimeError_(f"{name} must be an integer in [{low}, {high}]")
        out[name] = value
    # Combinations SchedulerConfig rejects at the pinned nightly (vllm/config/scheduler.py):
    # refused here so they never cost a GPU start. Other failures are measured, not guessed.
    tokens, seqs = out.get("max_num_batched_tokens"), out.get("max_num_seqs")
    if tokens is not None and seqs is not None and tokens < seqs:
        raise RuntimeError_("max_num_batched_tokens must be >= max_num_seqs")
    if (
        tokens is not None
        and out.get("enable_chunked_prefill") is False
        and tokens < PROFILE_FIXED["max_model_len"]
    ):
        raise RuntimeError_(
            "without chunked prefill max_num_batched_tokens must be >= max_model_len"
        )
    return dict(sorted(out.items()))


def options_argv(options: Mapping[str, Any]) -> list[str]:
    """Fixed flags for normalized options (appended after the profile's own flags)."""
    argv: list[str] = []
    for name, value in normalize_options(options).items() if options else ():
        flag, kind, _, _ = OPTIONS[name]
        if kind is bool:
            argv.append(flag if value else "--no-" + flag.removeprefix("--"))
        else:
            argv += [flag, str(value)]
    return argv


@dataclass(frozen=True)
class Cell:
    track: str
    cases: int
    concurrency: int
    slo_ms: float
    weight: float
    warm: bool


@dataclass(frozen=True)
class Calibration:
    """Operator-published after the reference-vs-reference pilot; every threshold is theirs."""

    version: str
    profile: dict[str, Any]
    cells: dict[str, Cell]
    blocks: int
    max_drift: float  # |ln(goodput B / goodput B')| above this in any cell: NO_DECISION
    min_gain: float  # the noise margin the 99 % LCB must clear
    latency_tolerance: float  # every block: candidate p95 <= (1 + this) * min(p95 B, p95 B')
    fidelity_loss_tolerance: float  # candidate loss per decision - stock's
    fidelity_accuracy_tolerance: float  # stock accuracy - candidate's
    # mean total-variation distance of the candidate's timed read answers from the stock
    # reference's, above the B-vs-B' distance of the same block: a kernel exact in the
    # fidelity pass but approximate while timed is rejected
    divergence_tolerance: float
    bootstrap_resamples: int
    credit_per_log_gain: float  # epoch-masses per unit of certified log gain
    credit_cap: float  # epoch-masses per crown
    kernel_slots: tuple[str, ...]  # slots accepting kernels under this calibration

    @property
    def profile_digest(self) -> str:
        return digest(self.profile)

    @classmethod
    def from_json(cls, raw: Any) -> Calibration:
        if not isinstance(raw, Mapping):
            raise RuntimeError_("the calibration must be an object")
        fields = set(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise RuntimeError_(f"calibration keys must be exactly {sorted(fields)}")
        profile = raw["profile"]
        if not isinstance(profile, Mapping) or set(profile) != set(PROFILE_KEYS):
            raise RuntimeError_(f"the profile keys must be exactly {sorted(PROFILE_KEYS)}")
        wrong = [k for k, v in PROFILE_FIXED.items() if profile[k] != v]
        if wrong:
            raise RuntimeError_(f"profile differs from the pinned serving profile: {wrong}")
        if profile["moe_backend"] not in MOE_BACKENDS:
            raise RuntimeError_(f"moe_backend must be one of {MOE_BACKENDS}")
        if not all(isinstance(profile[k], str) and profile[k] for k in MEASURED):
            raise RuntimeError_(f"profile {MEASURED} must be non-empty strings")
        if GPU_TYPE not in profile["gpu"]:
            raise RuntimeError_(f"the runtime lane is calibrated on {GPU_TYPE} only")
        slots = raw["kernel_slots"]
        if (
            not isinstance(slots, list)
            or len(set(slots)) != len(slots)
            or not all(s in KERNEL_SLOTS for s in slots)
        ):
            raise RuntimeError_(f"kernel_slots must be distinct slots from {KERNEL_SLOTS}")
        cells_raw = raw["cells"]
        if not isinstance(cells_raw, Mapping) or not cells_raw:
            raise RuntimeError_("cells must be a non-empty object")
        cells = {}
        for name, cell in cells_raw.items():
            if not isinstance(name, str) or not 0 < len(name) <= MAX_CELL_NAME:
                raise RuntimeError_(f"cell names must be 1 to {MAX_CELL_NAME} characters")
            if not isinstance(cell, Mapping) or set(cell) != set(Cell.__dataclass_fields__):
                raise RuntimeError_(
                    f"cell {name}: keys must be {sorted(Cell.__dataclass_fields__)}"
                )
            if cell["track"] not in CELL_TRACKS:
                raise RuntimeError_(f"cell {name}: track must be one of {CELL_TRACKS}")
            if not (
                _count(cell["cases"])
                and _count(cell["concurrency"])
                and cell["cases"] <= MAX_CELL_CASES
                and cell["concurrency"] <= MAX_CONCURRENCY
            ):
                raise RuntimeError_(
                    f"cell {name}: cases must be an int in [1, {MAX_CELL_CASES}] and "
                    f"concurrency in [1, {MAX_CONCURRENCY}]"
                )
            if not (_positive(cell["slo_ms"]) and _positive(cell["weight"])):
                raise RuntimeError_(f"cell {name}: slo_ms and weight must be positive")
            if not isinstance(cell["warm"], bool):
                raise RuntimeError_(f"cell {name}: warm must be a boolean")
            cells[str(name)] = Cell(**{k: cell[k] for k in Cell.__dataclass_fields__})
        if not math.isclose(sum(c.weight for c in cells.values()), 1.0, abs_tol=1e-9):
            raise RuntimeError_("cell weights must sum to 1 (a missing cell is never renormalized)")
        if not (_count(raw["blocks"]) and 3 <= raw["blocks"] <= MAX_BLOCKS):
            raise RuntimeError_(f"blocks must be an integer in [3, {MAX_BLOCKS}]")
        resamples = raw["bootstrap_resamples"]
        if not (_count(resamples) and 1000 <= resamples <= MAX_RESAMPLES):
            raise RuntimeError_(
                f"bootstrap_resamples must be an integer in [1000, {MAX_RESAMPLES}]"
            )
        for key in (
            "max_drift",
            "min_gain",
            "latency_tolerance",
            "fidelity_loss_tolerance",
            "fidelity_accuracy_tolerance",
            "divergence_tolerance",
            "credit_per_log_gain",
            "credit_cap",
        ):
            if not _nonnegative(raw[key]):
                raise RuntimeError_(f"{key} must be a finite non-negative number")
        if not isinstance(raw["version"], str) or not raw["version"]:
            raise RuntimeError_("version must be a non-empty string")
        return cls(
            **{**raw, "profile": dict(profile), "cells": cells, "kernel_slots": tuple(slots)}
        )

    def public(self) -> dict[str, Any]:
        """Published: distributions, weights, thresholds and profile; never the cases."""
        return {
            "version": self.version,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "cells": {n: vars(c) for n, c in sorted(self.cells.items())},
            **{
                k: getattr(self, k)
                for k in (
                    "blocks",
                    "max_drift",
                    "min_gain",
                    "latency_tolerance",
                    "fidelity_loss_tolerance",
                    "fidelity_accuracy_tolerance",
                    "divergence_tolerance",
                    "bootstrap_resamples",
                    "credit_per_log_gain",
                    "credit_cap",
                )
            },
            "kernel_slots": list(self.kernel_slots),
        }


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0


def _nonnegative(value: Any) -> bool:
    return _finite(value) and value >= 0


@dataclass(frozen=True)
class Fidelity:
    """Container-scored sums of one side's answers on the fidelity cases."""

    loss: float
    decisions: int
    determined: int
    correct: int
    cases: int


def _no_decision(reason: str, **extra: Any) -> dict[str, Any]:
    return {"decision": "no_decision", "reason": reason, "crown": False, **extra}


def _reject(reason: str, **extra: Any) -> dict[str, Any]:
    return {"decision": "reject", "reason": reason, "crown": False, **extra}


def _run(cal: Calibration, run: Any) -> dict[str, tuple[float, float]] | str:
    """{cell: (goodput, p95_ms)} of one run, or why it is malformed."""
    if not isinstance(run, Mapping) or set(run) != set(cal.cells):
        return "a run does not report exactly the calibrated cells"
    out = {}
    for name, cell in cal.cells.items():
        m = run[name]
        if not isinstance(m, Mapping) or set(m) != {"tasks", "ok", "errors", "seconds", "p95_ms"}:
            return f"cell {name}: expected tasks, ok, errors, seconds, p95_ms"
        tasks, ok, seconds, p95 = m["tasks"], m["ok"], m["seconds"], m["p95_ms"]
        if tasks != cell.cases or not isinstance(ok, int) or isinstance(ok, bool):
            return f"cell {name}: tasks must equal {cell.cases} and ok must be an int"
        if not 0 <= ok <= tasks or not _positive(seconds) or not _nonnegative(p95):
            return f"cell {name}: ok, seconds or p95_ms out of range or not finite"
        out[name] = (ok / seconds, float(p95))
    return out


def _lcb(values: Sequence[float], resamples: int, seed: str) -> float:
    """99 % one-sided percentile bootstrap lower bound of the mean, resampling whole blocks."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.01 * resamples)]


def fidelity_tracks(cal: Calibration) -> tuple[str, ...]:
    """Every track a cell measures is quality-guarded, decisions always."""
    return tuple(sorted({"decisions", *(c.track for c in cal.cells.values())}))


def verdict(
    cal: Calibration,
    evidence: Any,
    candidate: Mapping[str, Fidelity],
    stock: Mapping[str, Fidelity],
    fidelity_cases: Mapping[str, int],
    divergences: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Pure: the runtime verdict from the trusted worker's timings, the container's scores of
    the timed outputs (runs_from_tasks), its per-track fidelity scores and each block's timed
    answer divergence from B (divergence). Infrastructure doubt gives NO_DECISION (no credit,
    no penalty); a candidate failing on healthy infrastructure is rejected."""
    if not isinstance(evidence, Mapping):
        return _no_decision("no evidence")
    if evidence.get("profile") != cal.profile:
        return _no_decision("the worker profile differs from the calibrated profile")
    blocks = evidence.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != cal.blocks:
        return _no_decision(f"expected {cal.blocks} blocks")
    if set(fidelity_cases) != set(fidelity_tracks(cal)) or any(
        side.get(track) is None or side[track].cases != cases or side[track].decisions <= 0
        for track, cases in fidelity_cases.items()
        for side in (candidate, stock)
    ):
        return _no_decision("fidelity answers are incomplete")
    gains: list[float] = []
    latency: dict[str, list[float]] = {n: [] for n in cal.cells}
    cost: dict[str, list[float]] = {"B": [], "C": []}
    for block in blocks:
        if not isinstance(block, Mapping) or block.get("order") != list(SIDES):
            return _no_decision("every block must run B, C, B2 in this order")
        if block.get("quiescent") != [True, True, True]:
            return _no_decision("a process stop or GPU quiescence was not verified")
        runs = block.get("runs")
        if not isinstance(runs, Mapping) or set(runs) != set(SIDES):
            return _no_decision("a block misses a run")
        parsed = {side: _run(cal, runs[side]) for side in SIDES}
        for value in parsed.values():
            if isinstance(value, str):
                return _no_decision(f"malformed measurement: {value}")
        b, c, b2 = (parsed[s] for s in SIDES)
        assert not isinstance(b, str) and not isinstance(c, str) and not isinstance(b2, str)
        gain = 0.0
        for name, cell in cal.cells.items():
            (gb, pb), (gc, pc), (gb2, pb2) = b[name], c[name], b2[name]
            if gb <= 0 or gb2 <= 0 or min(pb, pb2) <= 0:
                return _no_decision(f"cell {name}: the reference completed no task under SLO")
            if abs(math.log(gb / gb2)) > cal.max_drift:
                return _no_decision(f"cell {name}: the reference drifted between B and B'")
            if gc <= 0:
                return _reject(f"cell {name}: the candidate completed no task under SLO")
            gain += cell.weight * math.log(gc / max(gb, gb2))
            latency[name].append(pc / min(pb, pb2))
        gains.append(gain)
        cost["B"].append(_seconds_per_ok(runs["B"]))
        cost["C"].append(_seconds_per_ok(runs["C"]))
    detail: dict[str, Any] = {
        "block_gains": gains,
        "gain_mean": sum(gains) / len(gains),
        "latency_ratio_median": {n: _median(v) for n, v in sorted(latency.items())},
        "latency_ratio_max": {n: max(v) for n, v in sorted(latency.items())},
        # shown, never paid on its own: the same speed-up already drives goodput
        "seconds_per_ok_task": {s: _median(v) for s, v in cost.items()},
        "calibration": cal.version,
    }
    detail["fidelity"] = {}
    for track in sorted(fidelity_cases):
        fc, fs = candidate[track], stock[track]
        loss_c, loss_s = fc.loss / fc.decisions, fs.loss / fs.decisions
        acc_c = fc.correct / fc.determined if fc.determined else 1.0
        acc_s = fs.correct / fs.determined if fs.determined else 1.0
        detail["fidelity"][track] = {
            "loss_candidate": loss_c,
            "loss_stock": loss_s,
            "accuracy_candidate": acc_c,
            "accuracy_stock": acc_s,
        }
        if loss_c - loss_s > cal.fidelity_loss_tolerance:
            return _reject(f"{track}: loss regressed against the stock reference", **detail)
        if acc_s - acc_c > cal.fidelity_accuracy_tolerance:
            return _reject(f"{track}: accuracy regressed against the stock reference", **detail)
    # the timed answers: a candidate may not drift from B further than B' drifts from B
    drift = divergences or []
    if len(drift) != len(blocks) or not all(
        isinstance(d, Mapping) and _finite(d.get("C")) and _finite(d.get("B2")) for d in drift
    ):
        return _no_decision("the timed answers are incomplete", **detail)
    detail["divergence"] = [{"C": d["C"], "B2": d["B2"]} for d in drift]
    if any(d["C"] - d["B2"] > cal.divergence_tolerance for d in drift):
        return _reject("the timed answers drifted from the stock reference", **detail)
    # every block must hold the guard: a median would let a minority of slow blocks through
    slow = [n for n, v in latency.items() if max(v) > 1 + cal.latency_tolerance]
    if slow:
        return _reject(f"p95 latency regressed in {slow}", **detail)
    g_lcb = _lcb(gains, cal.bootstrap_resamples, digest(evidence))
    detail["g_lcb"] = g_lcb
    if g_lcb <= cal.min_gain:
        return _reject("no certified gain", **detail)
    return {"decision": "crown", "reason": "certified gain", "crown": True, **detail}


def cell_case(seed: str, name: str, cell: Cell, index: int) -> Case:
    """Case `index` of a cell's private workload: public generators, the empty bank and the
    job's secret seed. Every side of every block runs the same cases; the container rebuilds
    them from the seed to score the outputs."""
    from . import generator, tracks
    from .bank import EMPTY_BANK

    plan = {cell.track: tracks.TrackPlan(1.0, cell.cases)}
    mix = {str(level): 1.0 for level in generator.LEVELS}
    return tracks.job_case(f"{seed}|runtime|{name}", plan, mix, index, EMPTY_BANK, False)


def task_ok(case: Case, item: Mapping[str, Any]) -> bool:
    """A timed task succeeded only if its raw output, scored here against gold, is right:
    a harness episode that replays to zero loss, or a read whose every answer is valid and
    whose every determined answer has the gold argmax, untied. An error item never succeeds."""
    from . import scoring, tracks

    if "error" in item:
        return False
    if case.track in tracks.ENVS:
        score = tracks.score_item(case, item)
        return score is not None and score.correct == 1
    answers, reads = item.get("answers"), item.get("reads")
    if not isinstance(answers, Mapping) or not isinstance(reads, Mapping):
        return False
    for qid, gold in case.gold.items():
        loss, right = scoring.decision_loss(gold, answers.get(qid), reads.get(qid))
        if loss >= 1.0:
            return False
        if gold.determined:
            vector = scoring._vector(gold, answers.get(qid)) or []
            if not right or sorted(vector)[-1] == sorted(vector)[-2]:  # a tie decides nothing
                return False
    return True


def answer_vectors(case: Case, item: Mapping[str, Any]) -> dict[str, list[float]] | None:
    """A read task's answers as gold-aligned probability vectors (None for a harness task);
    an invalid answer is an empty vector, as far from anything as it gets."""
    from . import scoring

    if case.track not in READ_TRACKS:
        return None
    answers = item.get("answers") if "error" not in item else None
    answers = answers if isinstance(answers, Mapping) else {}
    return {qid: scoring._vector(gold, answers.get(qid)) or [] for qid, gold in case.gold.items()}


def _distance(a: Mapping[str, list[float]], b: Mapping[str, list[float]]) -> float:
    """Mean total-variation distance over the case's questions (1 when either is invalid)."""
    per = [
        0.5 * sum(abs(p - q) for p, q in zip(a[k], b[k], strict=True))
        if a.get(k) and b.get(k) and len(a[k]) == len(b[k])
        else 1.0
        for k in sorted(set(a) | set(b))
    ]
    return sum(per) / len(per) if per else 0.0


def divergence(blocks: int, tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Per block, the mean distance of C's and of B2's timed read answers from B's on the same
    cases ({block, side, cell, case_index, vectors}); a case B answered but another side did
    not counts as distance 1. Blocks without read cells give 0 for both."""
    by: dict[tuple[int, str, str, int], Mapping[str, list[float]]] = {}
    for t in tasks:
        if t.get("vectors") is not None:
            by[(t["block"], t["side"], t["cell"], t["case_index"])] = t["vectors"]
    out = []
    for block in range(blocks):
        keys = [k for k in by if k[0] == block and k[1] == "B"]
        row = {}
        for side in ("C", "B2"):
            d = [_distance(by[k], by.get((block, side, k[2], k[3]), {"": []})) for k in keys]
            row[side] = sum(d) / len(d) if d else 0.0
        out.append(row)
    return out


def runs_from_tasks(
    cal: Calibration, blocks: Any, tasks: Sequence[Mapping[str, Any]]
) -> list[Any] | None:
    """The worker's blocks with each run's cell metrics rebuilt from the container-scored
    tasks ({block, side, cell, ms, ok, error}) and the worker's monotonic seconds per run.
    None when the blocks are malformed (the verdict then says why)."""
    if not isinstance(blocks, list):
        return None
    by_run: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for task in tasks:
        by_run.setdefault((task["block"], task["side"], task["cell"]), []).append(task)
    out: list[Any] = []
    for number, block in enumerate(blocks):
        if not isinstance(block, Mapping) or not isinstance(block.get("seconds"), Mapping):
            out.append(block)
            continue
        runs: dict[str, Any] = {}
        for side, cells in block["seconds"].items():
            if side not in SIDES or not isinstance(cells, Mapping):
                continue
            run: dict[str, Any] = {}
            for name, seconds in cells.items():
                cell = cal.cells.get(name)
                if cell is None:
                    continue
                done = by_run.get((number, side, name), [])
                ms = sorted(t["ms"] for t in done)
                run[name] = {
                    "tasks": len(done),
                    "ok": sum(bool(t["ok"]) and t["ms"] <= cell.slo_ms for t in done),
                    "errors": sum(bool(t["error"]) for t in done),
                    "seconds": seconds,
                    "p95_ms": ms[max(math.ceil(0.95 * len(ms)) - 1, 0)] if ms else 0.0,
                }
            runs[side] = run
        out.append({k: v for k, v in block.items() if k != "seconds"} | {"runs": runs})
    return out


def credit_units(cal: Calibration, g_lcb: float, units: int) -> int:
    return max(0, min(int(g_lcb * cal.credit_per_log_gain * units), int(cal.credit_cap * units)))


def _seconds_per_ok(run: Mapping[str, Any]) -> float:
    ok = sum(m["ok"] for m in run.values())
    return sum(m["seconds"] for m in run.values()) / ok if ok else math.inf


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
