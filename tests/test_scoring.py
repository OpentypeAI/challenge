import math
import random

import pytest

from opentype_challenge import scoring as s
from opentype_challenge.generator import Gold

READ = {"label_mass": 0.9, "argmax_is_label": True}
CHOICE = Gold("choice", ("a", "b", "c"), (0.0, 1.0, 0.0))
NOUL = Gold("noul", ("yes", "no"), (0.25, 0.75))
SCORE = Gold("score", ("low", "mid", "high"), (0.0, 0.0, 1.0))


def choice(probs):
    return {"choice": "b", "probabilities": dict(zip("abc", probs)), "confidence": 1}  # noqa: B905


def test_half_brier_per_type():
    assert s.decision_loss(CHOICE, choice([0, 1, 0]), READ) == (0.0, True)
    loss, right = s.decision_loss(CHOICE, choice([0.5, 0.5, 0]), READ)
    assert loss == pytest.approx(0.25) and right is False  # ties break to the first option
    assert s.decision_loss(NOUL, {"noul": 0.25}, READ)[0] == pytest.approx(0.0)
    assert s.decision_loss(NOUL, {"noul": 1.0}, READ)[0] == pytest.approx(0.5625)
    score = {
        "score": 2.0,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
    }
    assert s.decision_loss(SCORE, score, READ) == (pytest.approx(0.04), True)


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        choice([0, 1]),
        choice([0.5, 0.6, 0.0]),
        choice([float("nan"), 1, 0]),
        choice([-0.1, 1.1, 0]),
        {"probabilities": {"a": 0, "b": True, "c": 0}},
    ],
)
def test_invalid_answers_forfeit(answer):
    assert s.decision_loss(CHOICE, answer, READ) == (1.0, False)


def test_unparseable_reads_forfeit():
    good = choice([0, 1, 0])
    assert s.decision_loss(CHOICE, good, None) == (1.0, False)
    assert s.decision_loss(CHOICE, good, {"label_mass": 0.4, "argmax_is_label": True})[0] == 1
    assert s.decision_loss(CHOICE, good, {"label_mass": 0.9, "argmax_is_label": False})[0] == 1
    assert s.decision_loss(NOUL, {"noul": float("inf")}, READ)[0] == 1
    wrong_legend = {"legend": {"0": "mid", "1": "low", "2": "high"}, "probabilities": {}}
    assert s.decision_loss(SCORE, wrong_legend, READ)[0] == 1


def test_score_case_counts_determined_and_underdetermined():
    gold = {"q1": CHOICE, "q2": NOUL}
    score = s.score_case(
        gold, {"q1": choice([0, 1, 0]), "q2": {"noul": 0.5}}, {"q1": READ, "q2": READ}
    )
    assert (score.decisions, score.determined, score.correct, score.under) == (2, 1, 1, 1)
    assert score.loss == pytest.approx(0.0625) and score.under_loss == pytest.approx(0.0625)
    missing = s.score_case(gold, None, None)
    assert missing.loss == 2 and missing.correct == 0


def test_log_ratio_matches_moments_and_power_check():
    rng = random.Random(1)
    a = [rng.random() for _ in range(500)]
    b = [x * 0.8 + rng.random() * 0.1 for x in a]
    g, se = s.log_ratio(a, b)
    assert g == pytest.approx(math.log(sum(a) / sum(b)))
    # power_check.lcb, verbatim formula
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) / (n - 1)
    vb = sum((y - mb) ** 2 for y in b) / (n - 1)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (n - 1)
    reference = math.log(ma / mb) - s.Z99 * math.sqrt(
        max(va / ma**2 + vb / mb**2 - 2 * cov / (ma * mb), 0) / n
    )
    assert s.lcb(a, b) == pytest.approx(reference, rel=1e-9)
    assert s.log_ratio(a[:1], b[:1]) == (0.0, math.inf)


# -- power (adapted from power_check.py) ---------------------------------------------

PER_CASE = 6
HARD_SHARE = 1 / 3


def simulated_duel(rng, acc, r, n_dec, churn=0.25):
    """0/1 errors clustered by case; a third of cases are hard (power_check.duel)."""
    pc_h = (1 - acc) / HARD_SHARE
    pch_h = (1 - r) * pc_h
    stay = churn * pch_h / (1 - pc_h)
    keep = (1 - churn) * (1 - r)
    pairs = []
    for index in range(n_dec // PER_CASE):
        la = lb = 0
        hard = rng.random() < HARD_SHARE
        for _ in range(PER_CASE):
            wrong_c = hard and rng.random() < pc_h
            wrong_ch = hard and rng.random() < (keep if wrong_c else stay)
            la += wrong_c
            lb += wrong_ch
        pairs.append(
            s.Paired(
                index,
                3,
                s.CaseScore(la, PER_CASE, PER_CASE, PER_CASE - la, 0.0, 0),
                s.CaseScore(lb, PER_CASE, PER_CASE, PER_CASE - lb, 0.0, 0),
            )
        )
    return pairs


def crown_rate(acc, r, n_dec, sims, seed=7):
    rng = random.Random(seed)
    return (
        sum(
            s.verdict(simulated_duel(rng, acc, r, n_dec), set(), False)["crown"]
            for _ in range(sims)
        )
        / sims
    )


def test_equal_models_never_crown():
    assert crown_rate(0.98, 0.0, 60_000, sims=20) == 0.0
    assert crown_rate(0.95, 0.0, 60_000, sims=10, seed=3) == 0.0


def test_twenty_percent_error_reduction_at_98_crowns():
    assert crown_rate(0.98, 0.20, 60_000, sims=10) >= 0.9


def test_saturated_level_is_underpowered():
    assert crown_rate(0.995, 0.10, 60_000, sims=10) <= 0.5


def test_early_stop_fires_on_a_clearly_worse_challenger():
    rng = random.Random(5)
    worse = [
        s.Paired(p.index, p.level, p.challenger, p.champion)
        for p in simulated_duel(rng, 0.9, 0.5, 6_000)
    ]
    assert s.early_stop(worse, set())
    assert not s.early_stop(worse[:500], set())  # fewer than 5 000 decisions
    rng = random.Random(6)
    assert not s.early_stop(simulated_duel(rng, 0.9, 0.0, 6_000), set())


def test_regression_guard_blocks_a_crown():
    rng = random.Random(8)
    pairs = simulated_duel(rng, 0.9, 0.5, 30_000)
    guard = [
        s.Paired(
            100_000 + i,
            1,
            s.CaseScore(0, 6, 6, 6, 0.0, 0),
            s.CaseScore(1 if i % 20 == 0 else 0, 6, 6, 6 - (1 if i % 20 == 0 else 0), 0.0, 0),
        )
        for i in range(2000)
    ]
    assert s.verdict(pairs, {1}, False)["crown"]
    result = s.verdict(pairs + guard, {1}, False)
    assert result["guard_ucb"] > s.GUARD_MAX and not result["crown"]


def test_wilson_lower_bound_and_retirement_threshold():
    assert s.wilson_lower(0, 0) == 0
    assert s.wilson_lower(1000, 1000) < 1
    assert s.wilson_lower(20_000, 20_000) >= s.RETIRE_ACCURACY
    assert s.wilson_lower(19_990, 20_000) < s.RETIRE_ACCURACY


def test_ladder_and_mix():
    assert s.ladder_state([1, 2, 3, 4], 2, {1}) == ([2, 3], [1], [4])
    assert s.ladder_state([1, 2], 2, {1, 2}) == ([2], [1], [])
    mix = s.duel_mix([2, 3], [1], {2: 0.001, 3: 0.3})
    assert sum(mix.values()) == pytest.approx(1)
    assert mix[1] == pytest.approx(s.GUARD_SHARE)
    assert mix[2] >= s.MIX_FLOOR * (1 - s.GUARD_SHARE) * 0.9 and mix[3] > mix[2]
    assert s.duel_mix([4], [], {}) == {4: 1.0}


# -- v2: tracks, composite, per-track guard (docs/tracks.md §8, §11) -----------------


def retrack(pairs, track, offset=0):
    return [s.Paired(p.index + offset, p.level, p.champion, p.challenger, track) for p in pairs]


def harness_duel(rng, n, champion_loss, challenger_loss, track, offset):
    """Per-case 0/1 harness losses with the given rates."""
    return [
        s.Paired(
            offset + i,
            1,
            s.harness_score(float(rng.random() < champion_loss)),
            s.harness_score(float(rng.random() < challenger_loss)),
            track,
        )
        for i in range(n)
    ]


def test_harness_score_is_one_decision():
    assert s.harness_score(0.0) == s.CaseScore(0.0, 1, 1, 1, 0.0, 0)
    assert s.harness_score(0.25) == s.CaseScore(0.25, 1, 1, 0, 0.0, 0)
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            s.harness_score(bad)


def test_composite_of_one_track_is_v1_log_ratio():
    rng = random.Random(2)
    a = [rng.random() for _ in range(300)]
    b = [x * 0.7 + rng.random() * 0.2 for x in a]
    pairs = [
        s.Paired(i, 3, s.CaseScore(x, 1, 1, 1, 0, 0), s.CaseScore(y, 1, 1, 1, 0, 0))
        for i, (x, y) in enumerate(zip(a, b, strict=True))
    ]
    m = s.moments(pairs)
    assert list(m) == ["decisions"]
    g, se = s.composite(m, {"decisions": 0.35, "ops": 0.15})
    assert (g, se) == pytest.approx(s.log_ratio(a, b), rel=1e-12)
    assert s.composite({}, {"decisions": 1.0}) == (0.0, math.inf)


def test_composite_weights_tracks():
    m1 = s.moments(retrack(simulated_duel(random.Random(1), 0.9, 0.5, 6_000), "decisions"))
    m2 = s.moments(harness_duel(random.Random(2), 400, 0.5, 0.4, "ops", 0))
    (g1, se1), (g2, se2) = (s.log_ratio_moments(*m1["decisions"]), s.log_ratio_moments(*m2["ops"]))
    g, se = s.composite({**m1, **m2}, {"decisions": 3.0, "ops": 1.0})
    assert g == pytest.approx(0.75 * g1 + 0.25 * g2)
    assert se == pytest.approx(math.sqrt((3 * se1) ** 2 + se2**2) / 4)


def test_weights_none_keeps_v1_verdict():
    pairs = simulated_duel(random.Random(8), 0.9, 0.5, 30_000)
    v1 = s.verdict(pairs, set(), False)
    v2 = s.verdict(pairs, set(), False, {"decisions": 1.0})
    assert "tracks" not in v1 and v2["tracks"]["decisions"]["pairs"] == len(pairs)
    for key in ("crown", "g", "se", "g_lcb", "guard_ucb", "levels"):
        assert v2[key] == pytest.approx(v1[key]) if key != "levels" else v2[key] == v1[key]


WEIGHTS = {"decisions": 0.35, "longctx": 0.25, "ops": 0.15, "sql": 0.10, "paint": 0.15}


def better_everywhere(ops_challenger=0.3, ops_cases=300):
    rng = random.Random(11)
    pairs = retrack(simulated_duel(rng, 0.9, 0.5, 30_000), "decisions")
    pairs += retrack(simulated_duel(rng, 0.9, 0.5, 6_000), "longctx", 100_000)
    pairs += harness_duel(rng, ops_cases, 0.5, ops_challenger, "ops", 200_000)
    pairs += harness_duel(rng, 300, 0.5, 0.3, "sql", 300_000)
    pairs += harness_duel(rng, 200, 0.5, 0.3, "paint", 400_000)
    return pairs


def test_better_on_every_track_is_crowned():
    result = s.verdict(better_everywhere(), set(), False, WEIGHTS)
    assert result["crown"], result
    assert set(result["tracks"]) == set(WEIGHTS)
    assert all(not t["regressed"] for t in result["tracks"].values())
    ops = result["tracks"]["ops"]
    assert ops["pairs"] == 300 and ops["accuracy"]["challenger"] > ops["accuracy"]["champion"]
    assert s.verdict(better_everywhere(), set(), True, WEIGHTS)["crown"] is False


def test_a_track_regression_blocks_the_crown_only_with_30_pairs():
    regressed = s.verdict(better_everywhere(ops_challenger=0.95), set(), False, WEIGHTS)
    assert regressed["tracks"]["ops"]["regressed"] and not regressed["crown"]
    assert regressed["g_lcb"] >= s.G_MIN  # the composite alone would crown
    few = s.verdict(better_everywhere(ops_challenger=1.0, ops_cases=29), set(), False, WEIGHTS)
    assert few["tracks"]["ops"]["pairs"] == 29 and not few["tracks"]["ops"]["regressed"]
    assert few["crown"], few


def test_retired_guard_applies_to_decisions_only():
    pairs = better_everywhere()
    # level 1 of ops is not a retired decisions level: it stays active
    result = s.verdict(pairs, {1}, False, WEIGHTS)
    assert result["tracks"]["ops"]["pairs"] == 300 and result["guard_ucb"] == 0.0


def test_early_stop_with_weights_uses_the_composite():
    rng = random.Random(5)
    worse = [
        s.Paired(p.index, p.level, p.challenger, p.champion)
        for p in simulated_duel(rng, 0.9, 0.5, 6_000)
    ]
    assert s.early_stop(worse, set(), {"decisions": 1.0}) == s.early_stop(worse, set())
    worse_ops = harness_duel(random.Random(3), 2_000, 0.3, 0.8, "ops", 100_000)
    even = retrack(simulated_duel(random.Random(4), 0.9, 0.0, 6_000), "decisions")
    assert s.early_stop(even + worse_ops, set(), {"decisions": 0.5, "ops": 0.5})
    assert not s.early_stop(even + worse_ops, set(), {"decisions": 1.0})


@pytest.mark.parametrize(
    "plan",
    [
        {"decisions": (0.5, 300), "ops": (0.5, 300)},
        {"decisions": (0.4, 20_000), "longctx": (0.2, 800), "ops": (0.2, 300), "sql": (0.2, 300)},
    ],
)
def test_halves_split_every_track_evenly(plan):
    from opentype_challenge import tracks

    full = {t: tracks.TrackPlan(w, n) for t, (w, n) in plan.items()}
    zero = s.CaseScore(0.0, 1, 1, 1, 0.0, 0)
    pairs = [
        s.Paired(i, 1, zero, zero, tracks.track_of(i, full))
        for i in range(sum(n for _, n in plan.values()))
    ]
    halves = s._halves(pairs, {p.index for p in pairs})
    for track in plan:
        counts = [sum(p.track == track for p in half) for half in halves]
        assert abs(counts[0] - counts[1]) <= 1, (track, counts)


def test_default_plan_balances_the_track_errors():
    """Null duel under DEFAULT_PLAN: no track dominates the composite SE, which is ~0.017
    (0.027 with the former 20k-decisions plan)."""
    from opentype_challenge.tracks import DEFAULT_PLAN

    rng = random.Random(0)
    n = {t: p.cases for t, p in DEFAULT_PLAN.items()}
    pairs = retrack(simulated_duel(rng, 0.9, 0.0, PER_CASE * n["decisions"]), "decisions")
    pairs += retrack(simulated_duel(rng, 0.9, 0.0, PER_CASE * n["longctx"]), "longctx", 10**5)
    for i, track in enumerate(("ops", "sql", "paint")):
        pairs += harness_duel(rng, n[track], 0.4, 0.4, track, 2 * 10**5 + i * 10**4)
    weights = {t: p.weight for t, p in DEFAULT_PLAN.items()}
    result = s.verdict(pairs, set(), False, weights)
    shares = [weights[t] * m["se"] for t, m in result["tracks"].items()]
    assert max(shares) < 4 * min(shares), result["tracks"]
    assert result["se"] < 0.02
