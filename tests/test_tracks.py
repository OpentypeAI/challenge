"""Track plan, interleave, case construction, per-item scoring and read-solver dispatch."""

import asyncio
import json
import random

import pytest

from opentype_challenge import bank as b
from opentype_challenge import generator as g
from opentype_challenge import harness, tracks
from opentype_challenge.bank import EMPTY_BANK, Bank, BankItem
from opentype_challenge.tracks import DEFAULT_PLAN, TrackPlan

from .test_generator import prose_payload, sealed_payload
from .test_paint import DEPICT

MIX = {"1": 0.5, "2": 0.5}
SMALL = {
    "decisions": TrackPlan(0.35, 40),
    "longctx": TrackPlan(0.25, 10),
    "ops": TrackPlan(0.15, 10),
    "sql": TrackPlan(0.10, 10),
    "paint": TrackPlan(0.15, 10),
}


def rich_bank() -> Bank:
    sealed = g.family_from_json(sealed_payload())
    items = [BankItem.make("family", sealed_payload()), BankItem.make("depict", DEPICT)]
    for i, family in enumerate([sealed, *g.FAMILIES] * 3):
        items.append(BankItem.make("prose", prose_payload(random.Random(f"tb|{i}"), family, 2)))
    return Bank(tuple(items))


def only(track: str, cases: int = 1) -> dict[str, TrackPlan]:
    return {track: TrackPlan(1.0, cases)}


def test_default_plan():
    assert tracks.TRACKS == ("decisions", "longctx", "ops", "sql", "paint")
    assert set(tracks.ENVS) == {"ops", "sql", "paint"}
    assert {t: (p.weight, p.cases) for t, p in DEFAULT_PLAN.items()} == {
        "decisions": (0.35, 20_000),
        "longctx": (0.25, 800),
        "ops": (0.15, 300),
        "sql": (0.10, 300),
        "paint": (0.15, 200),
    }
    assert sum(p.weight for p in DEFAULT_PLAN.values()) == pytest.approx(1.0)


def test_track_of_prefixes_are_proportional():
    """At prefix n with last key x, every track holds x * n_t cases to within one case (the
    rounding of each track's own grid), hence n * share to within half the track count."""
    total = sum(p.cases for p in DEFAULT_PLAN.values())
    order = [tracks.track_of(i, DEFAULT_PLAN) for i in range(total)]
    counts = dict.fromkeys(DEFAULT_PLAN, 0)
    for n, track in enumerate(order, 1):
        counts[track] += 1
        x = (counts[track] - 0.5) / DEFAULT_PLAN[track].cases  # key of case n
        for t, plan in DEFAULT_PLAN.items():
            assert abs(counts[t] - x * plan.cases) <= 1, (n, t)
            assert abs(counts[t] - n * plan.cases / total) <= len(DEFAULT_PLAN) / 2, (n, t)
    assert counts == {t: p.cases for t, p in DEFAULT_PLAN.items()}
    with pytest.raises(IndexError):
        tracks.track_of(total, DEFAULT_PLAN)


def test_effective_plan_drops_unbuildable_tracks():
    plan = {"decisions": TrackPlan(0.5, 10), "paint": TrackPlan(0.5, 10)}
    assert set(tracks.effective_plan(plan, EMPTY_BANK, judge=False)) == {"decisions", "paint"}
    eff = tracks.effective_plan({**plan, "ops": TrackPlan(0.5, 0)}, EMPTY_BANK)
    assert {t: p.weight for t, p in eff.items()} == {"decisions": 0.5, "paint": 0.5}
    eff = tracks.effective_plan(DEFAULT_PLAN, EMPTY_BANK)
    assert sum(p.weight for p in eff.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tracks.effective_plan({"bogus": TrackPlan(1.0, 1)}, EMPTY_BANK)


def test_decisions_only_plan_with_an_empty_bank_is_v1():
    for i in range(20):
        v2 = tracks.job_case("seed", only("decisions", 20), MIX, i, EMPTY_BANK)
        assert v2 == b.job_case("seed", MIX, i)


@pytest.mark.parametrize("bank", [EMPTY_BANK, rich_bank()], ids=["empty", "rich"])
def test_job_case_is_deterministic_for_every_track(bank):
    seen = set()
    total = sum(p.cases for p in SMALL.values())
    for i in range(0, total, 3):
        a = tracks.job_case("s", SMALL, MIX, i, bank)
        again = tracks.job_case("s", SMALL, MIX, i, Bank(tuple(reversed(bank.items))))
        assert a == again
        assert a.track == tracks.track_of(i, tracks.effective_plan(SMALL, bank))
        assert json.loads(json.dumps(a.body, sort_keys=True)) == a.body
        seen.add(a.track)
    assert seen == set(tracks.TRACKS)


def test_paint_depict_needs_a_judge_and_a_depict_item():
    levels = {
        (name, judge): {
            tracks.job_case("p", only("paint", 60), MIX, i, bank, judge).level for i in range(60)
        }
        for name, bank in (("empty", EMPTY_BANK), ("rich", rich_bank()))
        for judge in (True, False)
    }
    assert levels[("rich", True)] == {1, 2, 3}
    assert levels[("rich", False)] == levels[("empty", True)] == levels[("empty", False)] == {1, 2}


def test_sealed_families_and_prose_are_used():
    bank = rich_bank()
    sealed = bank.families()[0].name
    cases = [tracks.job_case("f", only("decisions", 600), MIX, i, bank) for i in range(600)]
    share = sum(c.family == sealed for c in cases) / len(cases)
    assert 0.24 <= share <= 0.36
    prose = {item.payload["text"] for item in bank.of("prose")}
    used = sum(c.body["state"] in prose for c in cases) / len(cases)
    assert 0.1 <= used <= 0.5  # half of the cases whose family has bank prose


@pytest.mark.parametrize("track", ["decisions", "longctx"])
def test_solve_body_matches_gold(track):
    for i in range(6):
        case = tracks.job_case("solve", only(track, 6), {"1": 1.0}, i, EMPTY_BANK)
        solved = tracks.solve_body(case.body)
        assert set(solved) == set(case.gold)
        for qid, gold in case.gold.items():
            assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)
    with pytest.raises(g.GeneratorError):
        tracks.solve_body({"instructions": "Hello", "state": ""})


def test_score_item_reads_and_forfeits():
    case = tracks.job_case("r", only("decisions"), {"2": 1.0}, 0, EMPTY_BANK)
    forfeit = tracks.score_item(case, {"error": "HTTP 400"})
    assert forfeit is not None and forfeit.loss == len(case.gold) and forfeit.correct == 0
    assert tracks.score_item(case, {"answers": {}, "reads": {}}) == forfeit


def test_score_item_harness_replays_and_forfeits():
    case = tracks.job_case("h", only("ops"), MIX, 0, EMPTY_BANK)
    assert case.track == "ops"
    lost = tracks.score_item(case, {"error": "HTTP 400"})
    assert lost is not None and (lost.loss, lost.correct, lost.decisions) == (1.0, 0, 1)
    turns = case.body["limits"]["turns"]
    for bad in (None, "x", [1], ["{}"] * (turns + 1), ["x" * (harness.MAX_OUTPUT_CHARS + 1)]):
        assert tracks.score_item(case, {"transcript": bad}) == lost

    async def generate(messages, seed):
        return tracks.ops.reference_policy(messages)

    outputs = asyncio.run(harness.run_episode(tracks.ENVS["ops"], case.body, generate))
    won = tracks.score_item(case, {"transcript": outputs})
    assert won is not None and (won.loss, won.correct) == (0.0, 1)


def test_score_item_depict_needs_the_judge():
    bank = Bank((BankItem.make("depict", DEPICT),))
    case = next(
        c
        for i in range(60)
        if (c := tracks.job_case("d", only("paint", 60), MIX, i, bank)).level == 3
    )
    assert tracks.score_item(case, {"transcript": ['{"tool": "done", "args": {}}']}) is None
    lost = tracks.score_item(case, {"error": "HTTP 400"})
    assert lost is not None and lost.loss == 1.0
