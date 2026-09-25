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


# ---------------------------------------------------------------------------
# v2: family payloads, sealed families, prose states (docs/tracks.md §2, §4, §10).


def sealed_payload(tag: str = "0badcafe") -> dict:
    """A valid sealed family payload: a public family's shape, renamed."""
    payload = g.family_to_json(g.FAMILY_BY_NAME["invoice_approval"])
    payload["name"] = f"sealed_{tag}"
    payload["title"] = "vendor claim"
    payload["subject"] = "supplier"
    for fact in payload["facts"]:
        fact["name"] = "sx_" + fact["name"]
        fact["label"] = "claimed " + fact["label"]
    for derived in payload["derived"]:
        derived["name"] = "dx_" + derived["name"]
    for question in payload["questions"]:
        question["id"] = "qx_" + question["id"]
    return payload


def test_case_track_defaults_keep_v1_call_sites():
    case = next(cases(1))
    assert case.track == "decisions" and case.private == {}
    assert g.Case("x", 1, {}, {}, {}).track == "decisions"


def test_every_public_family_round_trips_exactly():
    for family in g.FAMILIES:
        payload = json.loads(json.dumps(g.family_to_json(family)))
        assert g.family_from_json(payload) == family
        assert g.family_to_json(g.family_from_json(payload)) == payload


def test_sealed_family_round_trips():
    payload = sealed_payload()
    family = g.family_from_json(payload)
    assert family.name == "sealed_0badcafe" and family.title == "vendor claim"
    assert g.family_to_json(family) == payload


def _no_scores(payload: dict) -> None:
    for question in payload["questions"]:
        if question["kind"] == "score":
            question.update(kind="noul", options=["yes", "no"])


REJECTIONS = {  # reason: (mutation, expected message fragment)
    "bad name": (lambda p: p.update(name="sealed_xyz"), "sealed_<8 hex>"),
    "public name, other shape": (lambda p: p.update(name="support_ticket"), "differs"),
    "public title": (lambda p: p.update(title="invoice"), "title of a public"),
    "extra key": (lambda p: p.update(extra=1), "expected keys"),
    "too few facts": (lambda p: p.update(facts=p["facts"][:5]), "facts: expected a list of 6"),
    "too many facts": (
        lambda p: p["facts"].extend(
            {"name": f"f{i}", "label": f"extra fact {i}", "domain": ["a", "b"], "unit": ""}
            for i in range(3)
        ),
        "facts: expected a list of 6..12",
    ),
    "domain of one": (lambda p: p["facts"][0].update(domain=["new"]), "domain: expected"),
    "domain of nine": (
        lambda p: p["facts"][0].update(domain=[f"v{i}" for i in range(9)]),
        "domain: expected a list of 2..8",
    ),
    "mixed domain": (lambda p: p["facts"][0].update(domain=["a", 1]), r"domain\[1\]"),
    "descending ints": (lambda p: p["facts"][6].update(domain=[500, 50]), "ascend"),
    "repeated value": (lambda p: p["facts"][0].update(domain=["a", "a"]), "not distinct"),
    "not snake value": (lambda p: p["facts"][0].update(domain=["Big", "small"]), "domain"),
    "bool domain": (lambda p: p["facts"][0].update(domain=[True, False]), "domain"),
    "label not lower case": (
        lambda p: p["facts"][0].update(label="Vendor Status"),
        r"facts\[0\].label",
    ),
    "duplicate label": (
        lambda p: p["facts"][1].update(label=p["facts"][0]["label"]),
        "labels must be unique",
    ),
    "duplicate name": (
        lambda p: p["facts"][1].update(name=p["facts"][0]["name"]),
        "names .* unique",
    ),
    "question named like a fact": (
        lambda p: p["questions"][0].update(id=p["facts"][0]["name"]),
        "names .* unique",
    ),
    "no derived": (lambda p: p.update(derived=[]), "derived: expected a list of 1..3"),
    "derived of five": (
        lambda p: p["derived"][0].update(domain=["a", "b", "c", "d", "e"]),
        "derived",
    ),
    "five questions": (lambda p: p.update(questions=p["questions"][:5]), "questions: expected"),
    "missing kind": (_no_scores, "every kind"),
    "unknown kind": (lambda p: p["questions"][0].update(kind="free"), "kind"),
    "choice pool of three": (
        lambda p: p["questions"][0].update(options=["a", "b", "c"]),
        "options: expected a list of 4..26",
    ),
    "score of two": (
        lambda p: next(q for q in p["questions"] if q["kind"] == "score").update(
            options=["low", "high"]
        ),
        "options: expected a list of 3..6",
    ),
    "noul options": (
        lambda p: next(q for q in p["questions"] if q["kind"] == "noul").update(
            options=["no", "yes"]
        ),
        "exactly",
    ),
    "prompt without ?": (lambda p: p["questions"][0].update(prompt="Decide"), "prompt"),
    "probe suffix": (lambda p: p["questions"][0].update(id="qx_decision_mirror"), "probe"),
    "value 'and'": (lambda p: p["facts"][0].update(domain=["and", "b"]), "rule grammar"),
    "derived value 'and'": (lambda p: p["derived"][0].update(domain=["and", "b"]), "grammar"),
    "not an object": (lambda p: p["facts"].__setitem__(0, "fact"), r"facts\[0\]"),
}


@pytest.mark.parametrize("reason", sorted(REJECTIONS))
def test_family_payload_rejections(reason):
    mutate, message = REJECTIONS[reason]
    payload = sealed_payload()
    mutate(payload)
    with pytest.raises(g.GeneratorError, match=message):
        g.family_from_json(payload)


def test_ambiguous_render_is_rejected_by_the_template_check():
    """Two facts whose labels and values nest render the same line: rejected."""
    payload = sealed_payload()
    payload["facts"][0].update(label="code", domain=["one_is_two", "b"])
    payload["facts"][1].update(label="code is one", domain=["two", "c"])
    with pytest.raises(g.GeneratorError, match="ambiguous"):
        g.family_from_json(payload)


def test_sealed_family_cases_solve_exactly_from_the_request():
    family = g.family_from_json(sealed_payload())
    for level in g.LEVELS:
        for i in range(8):
            case = g.make_case(random.Random(f"sealed|{level}|{i}"), family, level)
            assert case.body["instructions"].startswith("Record type: vendor claim.\n")
            solved = g.solve(case.body)
            for qid, gold in case.gold.items():
                assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)


def test_sealed_integer_and_unit_domains_are_rebuilt_from_text():
    family, _ = g.read_context(g.context_text(g.family_from_json(sealed_payload()), []))
    source = g.family_from_json(sealed_payload())
    assert [(f.name, f.label, f.domain, f.unit) for f in family.facts] == [
        (f.name, f.label, f.domain, f.unit) for f in source.facts
    ]


def test_sample_known_has_the_v1_distribution():
    family = g.FAMILIES[2]
    for i in range(50):
        a, b = random.Random(f"k|{i}"), random.Random(f"k|{i}")
        known, hidden = g.sample_known(a, family, 3)
        world = {f.name: b.choice(f.domain) for f in family.facts}
        names = [f.name for f in family.facts]
        drawn = set(b.sample(names, b.randint(0, 3)))
        assert hidden == [n for n in names if n in drawn]
        assert known == {n: world[n] for n in names if n not in drawn}
        assert list(known) == [n for n in names if n in known]


def prose_payload(rng, family, level):
    known, hidden = g.sample_known(rng, family, g.LEVELS[level].hidden)
    text = "Hi team, " + "; ".join(
        f"{family.fact(n).label} came out as {family.fact(n).show(v)}" for n, v in known.items()
    )
    return {"family": family.name, "known": known, "hidden": hidden, "text": text, "style": "email"}


def test_prose_state_is_verbatim_and_gold_is_the_posterior_given_known():
    families = [*g.FAMILIES, g.family_from_json(sealed_payload())]
    for level in (1, 3, 5, 8):
        for i in range(10):
            rng = random.Random(f"prose|{level}|{i}")
            family = families[i % len(families)]
            prose = json.loads(json.dumps(prose_payload(rng, family, level)))
            case = g.make_case(rng, family, level, prose=prose)
            assert case.body["state"] == prose["text"]
            known = {n: prose["known"][n] for n in sorted(prose["known"])}
            solved = g.solve_known(case.body, known)
            for qid, gold in case.gold.items():
                assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)
                assert gold.probs[case.realized[qid]] > 0


def test_prose_payload_must_match_its_family():
    family = g.FAMILIES[0]
    good = prose_payload(random.Random(1), family, 4)
    bad = [
        {**good, "family": "invoice_approval"},
        {**good, "known": {**good["known"], "nope": 1}},
        {**good, "known": {**good["known"], "tier": "platinum"}},
        {**good, "known": {**good["known"], "prior_tickets": "3"}},
        {**good, "hidden": ["tier"]},
        {**good, "text": ""},
        {**{k: v for k, v in good.items() if k != "hidden"}},
    ]
    for prose in bad:
        with pytest.raises(g.GeneratorError):
            g.make_case(random.Random(2), family, 4, prose=prose)


def test_v1_output_is_unchanged_without_prose():
    """Pinned digest of v1 bodies: the prose path must not move the template rng."""
    import hashlib

    digest = hashlib.sha256()
    for level in (1, 3, 6):
        for i in range(10):
            rng = random.Random(f"opentype-pin|{level}|{i}")
            case = g.make_case(rng, rng.choice(g.FAMILIES), level)
            digest.update(json.dumps(case.body, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    assert digest.hexdigest() == (
        "55144c773ef5e5cc1b25359ca9f68803f15a967b1083440111ff119d0b52c596"
    )
