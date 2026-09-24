import json
import random
import re
import time

import pytest

from opentype_challenge import generator as g
from opentype_challenge import longctx as lc
from opentype_challenge.bank import EMPTY_BANK, Bank, BankItem

from .test_generator import prose_payload, sealed_payload


def canonical(body) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def rng(*parts) -> random.Random:
    return random.Random("|".join(map(str, ("longctx", *parts))))


def target_block(case) -> tuple[list[str], int, int]:
    """State lines and the [start, end) line range of the target record's facts."""
    lines = case.body["state"].split("\n")
    start = lines.index(f"Record #{case.private['target']}") + 1
    end = lines.index("", start) if "" in lines[start:] else len(lines)
    return lines, start, end


def rich_bank() -> Bank:
    sealed = g.family_from_json(sealed_payload())
    items = [BankItem.make("family", sealed_payload())]
    for i, family in enumerate([sealed, *g.FAMILIES] * 6):
        prose = prose_payload(random.Random(f"bank|{i}"), family, 4)
        items.append(BankItem.make("prose", prose))
    return Bank(tuple(items))


def test_levels_and_buildable():
    assert tuple(lc.LEVELS) == (1, 2, 3, 4, 5)
    assert [spec.tokens for spec in lc.LEVELS.values()] == [8000, 16000, 32000, 64000, 100000]
    assert lc.buildable(EMPTY_BANK) == (1, 2, 3, 4, 5)
    assert lc.buildable(rich_bank()) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize("level", sorted(lc.LEVELS))
def test_solver_recomputes_gold_from_the_text(level):
    bank = rich_bank()
    for seed in range(6 if level < 5 else 3):
        case = lc.make_case(rng(level, seed), level, bank if seed % 2 else EMPTY_BANK)
        assert case.track == "longctx" and case.level == level
        solved = lc.solve(case.body)
        assert set(solved) == set(case.gold)
        for qid, gold in case.gold.items():
            assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)
            assert gold.probs[case.realized[qid]] > 0


@pytest.mark.parametrize("level", sorted(lc.LEVELS))
def test_body_fills_the_budget_and_stays_small(level):
    budget = lc.LEVELS[level].tokens * lc.CHARS_PER_TOKEN
    for seed in range(3):
        case = lc.make_case(rng("size", level, seed), level, rich_bank())
        size = len(canonical(case.body).encode())
        assert 0.85 * budget <= size <= 1.15 * budget
        assert size < lc.MAX_BODY_BYTES


def test_body_shape_and_instructions():
    case = lc.make_case(rng("shape"), 3, EMPTY_BANK)
    body = case.body
    assert set(body) == {"model", "instructions", "state", "questions", "samples", "seed"}
    lines = body["instructions"].split("\n")
    family = g.FAMILY_BY_NAME[case.family]
    assert lines[0] == f"Dossier: {family.title}."
    assert "Facts and allowed values:" in lines
    assert f"Target record: #{case.private['target']}. Every question is about this record." in (
        lines
    )
    assert "Dossier conventions:" in lines
    for question in body["questions"].values():
        assert g.RULES_HEADER in question["instructions"]


def test_level_five_is_fast():
    start = time.perf_counter()
    lc.make_case(rng("fast"), 5, rich_bank())
    assert time.perf_counter() - start < 1.0


def test_same_seed_same_case():
    bank = rich_bank()
    for level in lc.LEVELS:
        a = lc.make_case(rng("det", level), level, bank)
        b = lc.make_case(rng("det", level), level, bank)
        assert canonical(a.body) == canonical(b.body)
        assert a.gold == b.gold and a.private == b.private
    assert canonical(lc.make_case(rng("det", 9), 2, bank).body) != canonical(
        lc.make_case(rng("det", 8), 2, bank).body
    )


def test_near_duplicates_are_transpositions_present_from_level_two():
    seen = 0
    for level in (2, 3, 4, 5):
        for seed in range(4):
            case = lc.make_case(rng("near", level, seed), level, EMPTY_BANK)
            target, near = case.private["target"], case.private["near"]
            low, high = lc.LEVELS[level].near
            assert low <= len(near) <= high and len(near) >= 1
            state = case.body["state"]
            for record in near:
                assert record != target and sorted(record) == sorted(target)
                assert sum(a != b for a, b in zip(record, target, strict=True)) == 2
                assert state.count(f"Record #{record}\n") == 1
                seen += 1
            assert state.count(f"Record #{target}\n") == 1
            ids = re.findall(r"^Record #(\d{5})$", state, re.MULTILINE)
            assert len(ids) == len(set(ids))
            others = set(ids) - {target, *near}
            assert not others & set(lc._swaps(target))
    assert seen > 10


def test_near_duplicates_are_never_confused():
    """Reading a near-duplicate as the target gives different gold somewhere: the solver
    must pick the exact id."""
    differs = 0
    for seed in range(20):
        case = lc.make_case(rng("confuse", seed), 4, EMPTY_BANK)
        body = dict(case.body)
        record = case.private["near"][0]
        body["instructions"] = body["instructions"].replace(
            f"Target record: #{case.private['target']}.", f"Target record: #{record}."
        )
        wrong = lc.solve(body)
        differs += any(wrong[q] != list(gold.probs) for q, gold in case.gold.items())
        assert lc.solve(case.body) == pytest.approx(
            {q: list(gold.probs) for q, gold in case.gold.items()}, abs=1e-12
        )
    assert differs >= 10


def test_corrections_apply_in_order_and_the_last_wins():
    amended = 0
    for seed in range(40):
        case = lc.make_case(rng("fix", seed), 5, EMPTY_BANK)
        target = case.private["target"]
        family = g.FAMILY_BY_NAME[case.family]
        lines, block, end = target_block(case)
        known = g.extract(family, "\n".join(lines[block:end]))
        assert known is not None
        prefix = f"Correction to record #{target}: the "
        fixes = [line for line in lines[end:] if line.startswith(prefix)]
        for line in fixes:
            for fact in family.facts:
                for value in fact.domain:
                    if line == f"{prefix}{fact.label} is {fact.show(value)}.":
                        known[fact.name] = value
        assert known == case.private["known"]
        amended += bool(fixes)
        if fixes:
            # A correction before the last one loses; one after it wins.
            first = len(lines) - 1 - lines[::-1].index(fixes[-1])
            fact = next(f for f in family.facts if fixes[-1].startswith(f"{prefix}{f.label} is"))
            other = next(v for v in fact.domain if v != case.private["known"][fact.name])
            lines.insert(first, f"{prefix}{fact.label} is {fact.show(other)}.")
            earlier = dict(case.body, state="\n".join(lines))
            assert lc.solve(earlier) == lc.solve(case.body)
            lines.insert(first + 2, f"{prefix}{fact.label} is {fact.show(other)}.")
            later = dict(case.body, state="\n".join(lines))
            want = {**case.private["known"], fact.name: other}
            assert lc.solve(later) == g.solve_known(case.body, want)
    assert amended >= 10


def test_corrections_may_state_a_fact_the_record_left_out():
    found = 0
    for seed in range(60):
        case = lc.make_case(rng("fill", seed), 4, EMPTY_BANK)
        family = g.FAMILY_BY_NAME[case.family]
        lines, block, end = target_block(case)
        stated = g.extract(family, "\n".join(lines[block:end]))
        assert stated is not None
        found += bool(set(case.private["known"]) - set(stated))
    assert found >= 1


def test_needle_depth_spreads_over_the_dossier():
    depths = []
    for seed in range(60):
        case = lc.make_case(rng("depth", seed), 1, EMPTY_BANK)
        ids = re.findall(r"^Record #(\d{5})$", case.body["state"], re.MULTILINE)
        depths.append(ids.index(case.private["target"]) / (len(ids) - 1))
    assert min(depths) < 0.15 and max(depths) > 0.85
    assert 0.35 < sum(depths) / len(depths) < 0.65


def test_bank_sealed_families_and_prose_are_used():
    bank = rich_bank()
    sealed = bank.families()[0]
    texts = {item.payload["text"] for item in bank.of("prose")}
    families = set()
    prose_used = 0
    for seed in range(30):
        case = lc.make_case(rng("bank", seed), 2, bank)
        families.add(case.family)
        state = case.body["state"]
        prose_used += sum(text in state for text in texts)
        lines, block, end = target_block(case)
        assert "\n".join(lines[block:end]) not in texts
        if case.family == sealed.name:
            assert case.body["instructions"].startswith(f"Dossier: {sealed.title}.\n")
            solved = lc.solve(case.body)
            for qid, gold in case.gold.items():
                assert solved[qid] == pytest.approx(list(gold.probs), abs=1e-12)
    assert sealed.name in families
    assert prose_used > 30


def test_each_solver_rejects_the_other_track():
    with pytest.raises(g.GeneratorError):
        lc.solve(g.generate(None, 2, 1, seed=1)[0].body)
    with pytest.raises(g.GeneratorError):
        g.solve(lc.make_case(rng("other"), 1, EMPTY_BANK).body)
