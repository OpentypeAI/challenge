"""Track plan, case interleave, case construction, scoring and read-solver dispatch
(docs/tracks.md §8, §10).

A duel is a fixed number of cases per track, interleaved so that any prefix holds the tracks
in proportion. Every case is a pure function of (seed, plan, mix, index, bank, judge).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import generator, longctx, ops, paint, sqltask
from .bank import Bank, pick_level
from .generator import Case, GeneratorError
from .harness import MAX_OUTPUT_CHARS, Env, replay
from .scoring import CaseScore, harness_score, score_case

TRACKS = ("decisions", "longctx", "ops", "sql", "paint")
ENVS: dict[str, Env] = {"ops": ops.ENV, "sql": sqltask.ENV, "paint": paint.ENV}
BUILDERS = {"longctx": longctx, "ops": ops, "sql": sqltask, "paint": paint}
SEALED_SHARE = 0.3
PROSE_SHARE = 0.5


@dataclass(frozen=True)
class TrackPlan:
    weight: float
    cases: int


DEFAULT_PLAN: dict[str, TrackPlan] = {
    "decisions": TrackPlan(0.35, 20_000),
    "longctx": TrackPlan(0.25, 800),
    "ops": TrackPlan(0.15, 300),
    "sql": TrackPlan(0.10, 300),
    "paint": TrackPlan(0.15, 200),
}

PlanKey = tuple[tuple[str, TrackPlan], ...]


def _key(plan: Mapping[str, TrackPlan]) -> PlanKey:
    unknown = sorted(set(plan) - set(TRACKS))
    if unknown:
        raise ValueError(f"unknown tracks {unknown}")
    for track in sorted(plan):
        if plan[track].cases < 0 or not plan[track].weight >= 0:
            raise ValueError(f"track {track}: cases and weight must be non-negative")
    return tuple((t, plan[t]) for t in TRACKS if t in plan)


def _levels(track: str, bank: Bank, judge: bool) -> tuple[int, ...]:
    """Buildable levels of a non-decisions track (paint depict needs the judge)."""
    levels: tuple[int, ...] = BUILDERS[track].buildable(bank)
    if track == "paint" and not judge:
        levels = tuple(level for level in levels if paint.LEVELS[level] != "depict")
    return levels


def effective_plan(
    plan: Mapping[str, TrackPlan], bank: Bank, judge: bool = True
) -> dict[str, TrackPlan]:
    """The plan without tracks that build no level or have no case, weights renormalised."""
    kept = {
        t: p
        for t, p in _key(plan)
        if p.cases > 0 and p.weight > 0 and (t == "decisions" or _levels(t, bank, judge))
    }
    total = sum(p.weight for p in kept.values())
    if not kept:
        raise ValueError("the plan builds no case")
    return {t: TrackPlan(p.weight / total, p.cases) for t, p in kept.items()}


@lru_cache(maxsize=16)
def _schedule(key: PlanKey) -> tuple[str, ...]:
    # ponytail: materialises the whole order (~22k slots for DEFAULT_PLAN, sorted once per
    # plan); fine to ~1e6 cases. Past that, count keys below index/total per track instead.
    order = {t: i for i, t in enumerate(TRACKS)}
    slots = [((k + 0.5) / p.cases, order[t], t) for t, p in key for k in range(p.cases)]
    return tuple(t for _, _, t in sorted(slots))


def track_of(index: int, plan: Mapping[str, TrackPlan]) -> str:
    """The track of case `index`: the k-th case of track t sits at key (k + 0.5) / n_t, merged
    by (key, track order), so every prefix holds the tracks in proportion."""
    schedule = _schedule(_key(plan))
    if not 0 <= index < len(schedule):
        raise IndexError(f"case {index} is outside the plan's {len(schedule)} cases")
    return schedule[index]


def job_case(
    seed: str,
    plan: Mapping[str, TrackPlan],
    mix: Mapping[str, float],
    index: int,
    bank: Bank,
    judge: bool = True,
) -> Case:
    """Case `index` of a duel. A decisions-only plan with an empty bank gives v1's case."""
    track = track_of(index, effective_plan(plan, bank, judge))
    rng = random.Random(f"{seed}|{index}")
    if track != "decisions":
        return BUILDERS[track].make_case(rng, rng.choice(_levels(track, bank, judge)), bank)
    level = pick_level(rng, mix)
    sealed = bank.families()
    # no draw without sealed families or prose, so an empty bank reproduces v1's rng stream
    if sealed and rng.random() < SEALED_SHARE:
        family = rng.choice(sealed)
    else:
        family = rng.choice(generator.FAMILIES)
    prose = [item.payload for item in bank.of("prose") if item.payload.get("family") == family.name]
    chosen = rng.choice(prose) if prose and rng.random() < PROSE_SHARE else None
    return generator.make_case(rng, family, level, prose=chosen)


def _transcript(case: Case, value: Any) -> list[str] | None:
    """The raw outputs, or None when malformed: not a list of str, more outputs than turns,
    or an output longer than the worker's truncation."""
    turns = int(case.body["limits"]["turns"])
    if not isinstance(value, list) or len(value) > turns:
        return None
    if not all(isinstance(out, str) and len(out) <= MAX_OUTPUT_CHARS for out in value):
        return None
    return value


def score_item(case: Case, item: Mapping[str, Any]) -> CaseScore | None:
    """One side's score. An error item or a malformed transcript forfeits; None only when
    the harness loss needs the judge (paint depict)."""
    if case.track not in ENVS:
        if "error" in item:
            return score_case(case.gold, None, None)
        return score_case(case.gold, item.get("answers"), item.get("reads"))
    outputs = None if "error" in item else _transcript(case, item.get("transcript"))
    if outputs is None:
        return harness_score(1.0)
    _, loss = replay(ENVS[case.track], case.body, outputs)
    return None if loss is None else harness_score(loss)


def solve_body(body: Mapping[str, Any]) -> dict[str, list[float]]:
    """Exact gold of a template-rendered read body from its text, by instructions marker."""
    first = str(body.get("instructions", "")).split("\n", 1)[0]
    if first.startswith(generator.DECISIONS_MARKER):
        return generator.solve(body)
    if first.startswith(longctx.MARKER):
        return longctx.solve(body)
    raise GeneratorError("not a read request: unknown instructions marker")
