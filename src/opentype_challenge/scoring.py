"""Per-decision half-Brier, the paired duel statistic, the crown rule and the level ladder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeGuard

from .generator import NOUL, Gold

Z99 = 2.326
G_MIN = -math.log(0.95)  # a crown removes at least 5 % of the champion's loss
GUARD_MAX = 0.002  # UCB99 of the error-rate increase allowed on retired levels
EARLY_STOP_DECISIONS = 5_000
EARLY_STOP_SE = 3.0
RETIRE_ACCURACY = 0.999
MIX_FLOOR = 0.05  # every active level keeps at least this share of the active mix
GUARD_SHARE = 0.10  # share of duel cases drawn from retired levels
LABEL_MASS_MIN = 0.5
NORM_TOLERANCE = 1e-3
# Poisson UCB99 of a count observed as zero: a side's summed loss is floored here, so one duel
# certifies at most ln(champion loss / 4.6) nats however perfect the challenger looks.
ZERO_LOSS_FLOOR = -math.log(0.01)


def _finite(value: Any) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _vector(gold: Gold, answer: Any) -> list[float] | None:
    """The answer as a probability vector aligned with gold.options, or None when invalid."""
    if not isinstance(answer, Mapping):
        return None
    if gold.kind == "noul":
        p = answer.get("noul")
        if not _finite(p) or not 0.0 <= float(p) <= 1.0:
            return None
        return [float(p), 1.0 - float(p)]
    probs = answer.get("probabilities")
    if not isinstance(probs, Mapping):
        return None
    if gold.kind == "score":
        legend = {str(i): name for i, name in enumerate(gold.options)}
        if answer.get("legend") != legend:
            return None
        keys: Sequence[str] = list(legend)
    else:
        keys = gold.options
    if set(probs) != set(keys) or not all(_finite(probs[k]) for k in keys):
        return None
    vector = [float(probs[k]) for k in keys]
    if min(vector) < 0.0 or abs(sum(vector) - 1.0) > NORM_TOLERANCE:
        return None
    return vector


def decision_loss(gold: Gold, answer: Any, read: Any) -> tuple[float, bool]:
    """(half-Brier loss, argmax correct). Forfeit (1.0, False) for a missing, invalid,
    non-finite or unnormalised answer, a label mass below 0.5 or an off-label argmax."""
    if gold.kind == "noul" and len(gold.options) != len(NOUL):
        raise ValueError("noul gold must have two outcomes")
    vector = _vector(gold, answer)
    if (
        vector is None
        or not isinstance(read, Mapping)
        or not _finite(read.get("label_mass"))
        or read["label_mass"] < LABEL_MASS_MIN
        or read.get("argmax_is_label") is not True
    ):
        return 1.0, False
    loss = 0.5 * sum((p - g) ** 2 for p, g in zip(vector, gold.probs, strict=True))
    top = max(range(len(vector)), key=vector.__getitem__)
    return loss, top == max(range(len(gold.probs)), key=gold.probs.__getitem__)


@dataclass(frozen=True)
class CaseScore:
    """One side of one case: the cluster sums used by every statistic."""

    loss: float
    decisions: int
    determined: int
    correct: int  # argmax == gold argmax, determined items only
    under_loss: float  # half-Brier on underdetermined items
    under: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def score_case(
    gold: Mapping[str, Gold],
    answers: Mapping[str, Any] | None,
    reads: Mapping[str, Any] | None,
) -> CaseScore:
    loss = under_loss = 0.0
    determined = correct = under = 0
    for qid, item in gold.items():
        answer = answers.get(qid) if isinstance(answers, Mapping) else None
        read = reads.get(qid) if isinstance(reads, Mapping) else None
        item_loss, right = decision_loss(item, answer, read)
        loss += item_loss
        if item.determined:
            determined += 1
            correct += right
        else:
            under += 1
            under_loss += item_loss
    return CaseScore(loss, len(gold), determined, correct, under_loss, under)


# ---------------------------------------------------------------------------
# Paired statistics on cluster sums (cluster = case).


def log_ratio(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(g, se) for g = ln(mean a / mean b), delta method on paired cluster sums.

    ponytail: delta-method SE (power_check.lcb) instead of the 10 000-resample cluster
    bootstrap: same estimand, ~1000x cheaper; swap in the bootstrap if an audit near the
    bar disagrees.
    """
    if len(b) != len(a):
        raise ValueError("paired samples differ in length")
    return log_ratio_moments(
        len(a),
        sum(a),
        sum(b),
        sum(x * x for x in a),
        sum(y * y for y in b),
        sum(x * y for x, y in zip(a, b, strict=True)),
    )


def log_ratio_moments(
    n: int, sa: float, sb: float, saa: float, sbb: float, sab: float
) -> tuple[float, float]:
    """log_ratio from running sums, so SQLite can aggregate a duel in one query."""
    if n < 2:
        return 0.0, math.inf
    ma, mb = max(sa, ZERO_LOSS_FLOOR) / n, max(sb, ZERO_LOSS_FLOOR) / n
    va = max(saa - sa * sa / n, 0.0) / (n - 1)
    vb = max(sbb - sb * sb / n, 0.0) / (n - 1)
    cov = (sab - sa * sb / n) / (n - 1)
    se = math.sqrt(max(va / ma**2 + vb / mb**2 - 2 * cov / (ma * mb), 0.0) / n)
    return math.log(ma / mb), se


def lcb(a: Sequence[float], b: Sequence[float]) -> float:
    g, se = log_ratio(a, b)
    return g - Z99 * se


def ratio_ucb(diff: Sequence[float], weight: Sequence[float]) -> float:
    """UCB99 of sum(diff) / sum(weight), cluster-robust (linearised ratio)."""
    n, total = len(diff), sum(weight)
    if n < 2 or total <= 0:
        return 0.0 if n == 0 else math.inf
    r = sum(diff) / total
    resid = sum((d - r * w) ** 2 for d, w in zip(diff, weight, strict=True))
    return r + Z99 * math.sqrt(n / (n - 1) * resid) / total


def wilson_lower(correct: int, total: int) -> float:
    """Wilson score LCB99 of an accuracy.

    ponytail: counts decisions as independent; errors cluster by case, so this is mildly
    optimistic. Switch to a cluster bound if a retired level ever regresses.
    """
    if total == 0:
        return 0.0
    p, z2 = correct / total, Z99**2
    centre = p + z2 / (2 * total)
    spread = Z99 * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2))
    return (centre - spread) / (1 + z2 / total)


@dataclass(frozen=True)
class Paired:
    index: int
    level: int
    champion: CaseScore
    challenger: CaseScore


def early_stop(pairs: Iterable[Paired], retired: set[int]) -> bool:
    """Stop once 5 000 paired decisions show g < 0 by more than 3 SE."""
    active = [p for p in pairs if p.level not in retired]
    if sum(p.champion.decisions for p in active) < EARLY_STOP_DECISIONS:
        return False
    g, se = log_ratio([p.champion.loss for p in active], [p.challenger.loss for p in active])
    return g + EARLY_STOP_SE * se < 0


def _level_metrics(pairs: Sequence[Paired]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in sorted({p.level for p in pairs}):
        rows = [p for p in pairs if p.level == level]
        entry: dict[str, Any] = {"cases": len(rows)}
        for side in ("champion", "challenger"):
            scores = [getattr(p, side) for p in rows]
            determined = sum(s.determined for s in scores)
            under = sum(s.under for s in scores)
            entry[side] = {
                "determined": determined,
                "correct": sum(s.correct for s in scores),
                "accuracy": sum(s.correct for s in scores) / determined if determined else None,
                "under": under,
                "brier": sum(s.under_loss for s in scores) / under if under else None,
                "loss": sum(s.loss for s in scores),
            }
        out[str(level)] = entry
    return out


def verdict(pairs: Sequence[Paired], retired: set[int], stopped: bool) -> dict[str, Any]:
    """The crown rule. Halves are the even and odd case indices."""
    active = [p for p in pairs if p.level not in retired]
    guard = [p for p in pairs if p.level in retired]
    a = [p.champion.loss for p in active]
    b = [p.challenger.loss for p in active]
    g, se = log_ratio(a, b)
    halves = [
        lcb(
            [p.champion.loss for p in active if p.index % 2 == h],
            [p.challenger.loss for p in active if p.index % 2 == h],
        )
        for h in (0, 1)
    ]
    g_lcb = min(halves)
    guard_ucb = ratio_ucb(
        [
            (p.challenger.determined - p.challenger.correct)
            - (p.champion.determined - p.champion.correct)
            for p in guard
        ],
        [p.champion.determined for p in guard],
    )
    crown = not stopped and g_lcb >= G_MIN and guard_ucb <= GUARD_MAX
    return {
        "crown": crown,
        "early_stop": stopped,
        "g": g,
        "se": se,
        "g_lcb": g_lcb,
        "g_lcb_halves": halves,
        "g_min": G_MIN,
        "guard_ucb": guard_ucb,
        "guard_max": GUARD_MAX,
        "cases": len(pairs),
        "decisions": sum(p.champion.decisions for p in pairs),
        "levels": _level_metrics(pairs),
    }


# ---------------------------------------------------------------------------
# Frontier ladder.


def ladder_state(
    order: Sequence[int], width: int, retired: set[int]
) -> tuple[list[int], list[int], list[int]]:
    """(active, retired, pending): the first `width` unretired levels are active. When every
    level is retired the hardest one stays active until the operator adds levels."""
    alive = [level for level in order if level not in retired]
    if not alive:
        return [order[-1]], list(order[:-1]), []
    return alive[:width], [lv for lv in order if lv in retired], alive[width:]


def duel_mix(
    active: Sequence[int], retired: Sequence[int], errors: Mapping[int, float]
) -> dict[int, float]:
    """Case share per level: active levels by champion error rate (floored), retired levels
    split GUARD_SHARE evenly. errors[level] is the champion's error rate (1.0 when unknown)."""
    if not active:
        raise ValueError("the ladder has no active level")
    raw = {level: max(errors.get(level, 1.0), 1e-6) for level in active}
    total = sum(raw.values())
    share = {level: max(value / total, MIX_FLOOR) for level, value in raw.items()}
    total = sum(share.values())
    active_mass = 1.0 - GUARD_SHARE if retired else 1.0
    mix = {level: active_mass * value / total for level, value in share.items()}
    for level in retired:
        mix[level] = GUARD_SHARE / len(retired)
    return mix
