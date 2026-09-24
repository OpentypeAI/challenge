import json
import random

import pytest

from opentype_challenge import generator as g


def cases(n_per_level: int = 60, levels=tuple(g.LEVELS)):
    for level in levels:
        for i in range(n_per_level):
            rng = random.Random(f"test|{level}|{i}")
            yield g.make_case(rng, rng.choice(g.FAMILIES), level)


def test_families_have_closed_domains_and_question_mix():
    assert len(g.FAMILIES) >= 4
    for family in g.FAMILIES:
        kinds = {q.kind for q in family.questions}
        assert kinds == {"choice", "noul", "score"}
        for question in family.questions:
            assert 2 <= len(question.options) <= 26
            assert len(set(question.options)) == len(question.options)
        for fact in family.facts:
            assert len(fact.domain) >= 2 and len(set(fact.domain)) == len(fact.domain)


def test_case_shape_matches_the_structured_server_contract():
    for case in cases(10):
        body = case.body
        assert set(body) == {"model", "instructions", "state", "questions", "samples", "seed"}
        assert body["samples"] == "auto"
        assert 5 <= len(body["questions"]) <= 7
        for qid, question in body["questions"].items():
            assert ":" not in qid and "\n" not in qid
            gold = case.gold[qid]
            if question["type"] == "choice":
                assert list(question["criteria"]) == list(gold.options)
            elif question["type"] == "score":
                assert question["criteria"] == list(gold.options)
            else:
                assert set(question["criteria"]) == {"true", "false"}
                assert gold.options == g.NOUL
        assert len(json.dumps(body)) < 64_000


def test_gold_is_a_distribution_and_the_true_world_is_possible():
    for case in cases():
        for qid, gold in case.gold.items():
            assert abs(sum(gold.probs) - 1) < 1e-12 and min(gold.probs) >= 0
            assert gold.probs[case.realized[qid]] > 0


def test_n_version_interpreter_and_compiled_evaluator_agree():
    rng = random.Random(3)
    compiled = g.compiled_evaluator()
    family = g.FAMILIES[0]
    for _ in range(2000):
        rule = g._sample_rule(rng, family.facts, (), ("a", "b", "c"), (1, 8), 3)
        world = {f.name: rng.choice(f.domain) for f in family.facts}
        assert g.interpret(rule, world) == compiled(rule, world)


def test_text_is_the_rule_and_the_reference_solver_reproduces_gold():
    """The criteria text parses back to the rule, and a solver that sees only the request
    (never the world) recomputes the exact posterior."""
    for case in cases(40):
        solved = g.solve(case.body)
        for qid, gold in case.gold.items():
            assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)


def test_render_round_trips_through_the_extractor():
    rng = random.Random(9)
    for family in g.FAMILIES:
        for level in g.LEVELS.values():
            known = {f.name: rng.choice(f.domain) for f in family.facts if rng.random() < 0.7}
            state = g.render_state(rng, family, known, level)
            assert g.extract(family, state) == known


def test_duplicate_fact_statement_is_not_extracted():
    family = g.FAMILIES[0]
    state = "Plan tier: pro\nThe plan tier is free."
    assert g.extract(family, state) is None


def test_posterior_is_calibrated_against_realized_worlds():
    """E[g(realized)] == E[sum g^2] when hidden facts really are uniform."""
    hit = expected = 0.0
    under = 0
    for case in cases(120, levels=(2, 3, 4, 5, 6)):
        for qid, gold in case.gold.items():
            if not gold.determined:
                under += 1
                hit += gold.probs[case.realized[qid]]
                expected += sum(p * p for p in gold.probs)
    assert under > 300
    assert abs(hit - expected) / expected < 0.06


def test_probe_transforms_are_exact():
    reordered = mirrored = 0
    for case in cases(40, levels=(3, 4, 5, 6)):
        for qid, gold in case.gold.items():
            if qid.endswith("_reordered"):
                reordered += 1
                source = case.gold[qid.removesuffix("_reordered")]
                assert dict(zip(gold.options, gold.probs, strict=True)) == dict(
                    zip(source.options, source.probs, strict=True)
                )
                assert gold.options != source.options
            if qid.endswith("_mirror"):
                mirrored += 1
                source = case.gold[qid.removesuffix("_mirror")]
                assert gold.probs[0] == pytest.approx(1 - source.probs[0], abs=1e-12)
    assert reordered > 5 and mirrored > 5


def test_level_one_is_fully_determined_and_harder_levels_hide_facts():
    easy = [gold for case in cases(30, levels=(1,)) for gold in case.gold.values()]
    assert all(gold.determined for gold in easy)
    hard = [gold for case in cases(60, levels=(6,)) for gold in case.gold.values()]
    assert any(not gold.determined for gold in hard)


def test_generation_is_deterministic():
    a = g.generate("invoice_approval", 4, 5, seed=11)
    b = g.generate("invoice_approval", 4, 5, seed=11)
    assert [c.body for c in a] == [c.body for c in b]
    assert [c.gold for c in a] == [c.gold for c in b]
    assert g.generate(None, 4, 5, seed=12)[0].body != a[0].body


def test_gold_json_round_trip():
    case = next(cases(1))
    for gold in case.gold.values():
        assert g.Gold.from_json(json.loads(json.dumps(gold.to_json()))) == gold
