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
