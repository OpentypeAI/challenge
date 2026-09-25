"""The runtime lane: vLLM options, the operator calibration and the pure verdict.

The lane pays for serving the quality champion's weights faster. A miner submits only
options from OPTIONS (no argv, env, image, plugin, reader or kernel); a trusted worker runs
blocks of incumbent B / candidate C / incumbent B' one process at a time on one exclusive
GPU and reports each timed task's latency and raw output plus each run's monotonic seconds.
The container scores every output against its own gold (task_ok) and recomputes the verdict
here; no success flag or count from the worker or the miner is ever evidence.

Nothing is decided before the operator publishes a calibration (reference-vs-reference
pilot on the pinned profile): without it the lane stays closed and no timing is accepted.
"""

from __future__ import annotations

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
KERNELS = "disabled: no GPU backend with verified isolation (docs/operator.md, runtime lane)"
SIDES = ("B", "C", "B2")
# Operational caps on a calibration (an oversized one would hold the exclusive GPU for days):
# MAX_BLOCKS matches the timings API's block bound.
MAX_BLOCKS = 999
MAX_CELL_CASES = 10_000
MAX_CONCURRENCY = 1024
MAX_RESAMPLES = 100_000
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

# The serving profile every measurement runs under. The worker measures MEASURED on its own
# host and reads vllm_image from the build manifest baked into its image; a calibration pins
# all of them, and a worker that cannot read one refuses the job.
MEASURED = ("gpu", "driver", "vllm_version")
PROFILE_FIXED: dict[str, Any] = {
    "vllm_image": pins.VLLM_IMAGE,
    "structured_server_sha256": pins.STRUCTURED_SERVER_SHA256,
    "base": f"{pins.BASE_REPO}@{pins.BASE_REVISION}",
    "dtype": "bfloat16",
    "canvas": 256,
    "max_model_len": 131072,
    "gpu_memory_utilization": 0.9,
}


class RuntimeError_(ValueError):
    """A refused option set or calibration."""


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
    latency_tolerance: float  # candidate p95 <= (1 + this) * min(p95 B, p95 B')
    fidelity_loss_tolerance: float  # candidate loss per decision - stock's
    fidelity_accuracy_tolerance: float  # stock accuracy - candidate's
    bootstrap_resamples: int
    credit_per_log_gain: float  # epoch-masses per unit of certified log gain
    credit_cap: float  # epoch-masses per crown

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
        if not isinstance(profile, Mapping) or set(profile) != {*PROFILE_FIXED, *MEASURED}:
            raise RuntimeError_(f"the profile must hold the fixed profile keys and {MEASURED}")
        wrong = [k for k, v in PROFILE_FIXED.items() if profile[k] != v]
        if wrong:
            raise RuntimeError_(f"profile differs from the pinned serving profile: {wrong}")
        if not all(isinstance(profile[k], str) and profile[k] for k in MEASURED):
            raise RuntimeError_(f"profile {MEASURED} must be non-empty strings")
        cells_raw = raw["cells"]
        if not isinstance(cells_raw, Mapping) or not cells_raw:
            raise RuntimeError_("cells must be a non-empty object")
        cells = {}
        for name, cell in cells_raw.items():
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
            "credit_per_log_gain",
            "credit_cap",
        ):
            if not _nonnegative(raw[key]):
                raise RuntimeError_(f"{key} must be a finite non-negative number")
        if not isinstance(raw["version"], str) or not raw["version"]:
            raise RuntimeError_("version must be a non-empty string")
        return cls(**{**raw, "profile": dict(profile), "cells": cells})

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
                    "bootstrap_resamples",
                    "credit_per_log_gain",
                    "credit_cap",
                )
            },
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
) -> dict[str, Any]:
    """Pure: the runtime verdict from the trusted worker's timings, the container's scores of
    the timed outputs (runs_from_tasks) and its per-track fidelity scores. Infrastructure
    doubt gives NO_DECISION (no credit, no penalty); a candidate failing on healthy
    infrastructure is rejected."""
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
    slow = [n for n, v in latency.items() if _median(v) > 1 + cal.latency_tolerance]
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
