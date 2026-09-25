"""Long-context read track: typed decisions about one record buried in a dossier of many
records of the same family (docs/tracks.md §4).

The questions, rules and exact gold are the decisions track's; the state is a dossier of
"Record #NNNNN" blocks separated by blank lines, with "Correction to record #NNNNN: ..."
lines after the records they amend (last one wins) and near-duplicate ids (a transposition
of two distinct digits of the target id). The target record is always template-rendered,
so its effective known facts are exact; other records may be bank prose.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import generator as g
from .bank import Bank


@dataclass(frozen=True)
class Level:
    tokens: int  # budget of the served body, estimated at CHARS_PER_TOKEN
    amendments: tuple[int, int]
    near: tuple[int, int]  # near-duplicate ids of the target
    rules: int  # decisions level of the rules, derived facts and the target's hidden facts


LEVELS: dict[int, Level] = {
    1: Level(8_000, (0, 2), (0, 1), 2),
    2: Level(16_000, (1, 4), (1, 2), 3),
    3: Level(32_000, (2, 6), (1, 3), 3),
    4: Level(64_000, (3, 8), (2, 4), 4),
    5: Level(100_000, (4, 10), (2, 5), 4),
}
CHARS_PER_TOKEN = 4
MAX_BODY_BYTES = 600_000
SEALED_SHARE = 0.3  # as the decisions track
PROSE_SHARE = 0.5  # of non-target records, when the bank has prose of the family
TARGET_SHARE = 0.4  # of amendments; NEAR_SHARE go to near-duplicates, the rest elsewhere
NEAR_SHARE = 0.3
MARKER = "Dossier: "
CONVENTIONS = (
    "Dossier conventions:",
    '- The state is a dossier of many records. Each record starts with a line "Record '
    '#NNNNN" (a five-digit id) and ends at the next blank line.',
    '- A line "Correction to record #NNNNN: the <label> is <value>." sets that fact of '
    "that record. Corrections apply in order and the last one wins; a correction may state "
    "a fact the record left out.",
    "- A different id is a different record, even when it looks alike.",
    "- A fact that neither the target record nor a correction to it states is equally "
    "likely to take each of its allowed values, independently of the other facts.",
)
_TARGET = re.compile(r"^Target record: #(\d{5})\.", re.MULTILINE)


def buildable(bank: Bank) -> tuple[int, ...]:
    """Every level: public families and template records always suffice."""
    return tuple(LEVELS)


def header(record: str) -> str:
    return f"Record #{record}"


def correction(record: str, fact: g.Fact, value: g.Value) -> str:
    return f"Correction to record #{record}: the {fact.label} is {fact.show(value)}."


def _jlen(text: str) -> int:
    """Characters the text takes inside the served JSON body."""
    return len(json.dumps(text)) - 2


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _swaps(record: str) -> list[str]:
    """Every id that transposes two distinct digits of record, sorted."""
    out = set()
    for i in range(len(record)):
        for j in range(i + 1, len(record)):
            if record[i] != record[j]:
                digits = list(record)
                digits[i], digits[j] = digits[j], digits[i]
                out.add("".join(digits))
    return sorted(out)


def _fresh_id(rng: random.Random, taken: set[str]) -> str:
    while True:
        record = f"{rng.randrange(100_000):05d}"
        if record not in taken:
            taken.add(record)
            return record


def _usable(text: Any) -> bool:
    """Prose that cannot be mistaken for dossier structure (ids, corrections)."""
    return (
        isinstance(text, str)
        and bool(text.strip())
        and "#" not in text
        and "correction to record" not in text.lower()
    )


def _block(text: str) -> str:
    """Prose as one record block: no blank line inside."""
    return re.sub(r"\n\s*\n", "\n", text.strip())


def _template(
    rng: random.Random, family: g.Family, known: Mapping[str, g.Value], spec: g.Level
) -> str:
    return "\n".join(g.render_state(rng, family, known, spec).split("\n")[1:])


def _target_lines(
    rng: random.Random, family: g.Family, known: Mapping[str, g.Value], spec: g.Level
) -> str:
    for _ in range(g.MAX_ATTEMPTS):
        text = _template(rng, family, known, spec)
        if text and "\n\n" not in text and g.extract(family, text) == known:
            return text
    raise g.GeneratorError(f"{family.name}: every target render failed the round trip")


def instructions(family: g.Family, derived: Sequence[tuple[str, g.Rule]], target: str) -> str:
    context = g.context_text(family, derived).split("\n")
    return "\n".join(
        [
            f"{MARKER}{family.title}.",
            *context[1:-1],
            *CONVENTIONS,
            f"Target record: #{target}. Every question is about this record.",
            context[-1],
        ]
    )


def make_case(rng: random.Random, level: int, bank: Bank) -> g.Case:
    spec = LEVELS[level]
    rules = g.LEVELS[spec.rules]
    sealed = bank.families()
    family = rng.choice(sealed if sealed and rng.random() < SEALED_SHARE else g.FAMILIES)
    prose = [
        item.payload
        for item in bank.of("prose")
        if item.payload.get("family") == family.name and _usable(item.payload.get("text"))
    ]

    target = f"{rng.randrange(100_000):05d}"
    while len(_swaps(target)) < spec.near[1]:
        target = f"{rng.randrange(100_000):05d}"
    near = rng.sample(_swaps(target), rng.randint(*spec.near))
    taken = {target, *_swaps(target)}  # other ids never look like the target

    derived, items = g.sample_program(rng, family, rules)
    known, _ = g.sample_known(rng, family, rules.hidden)
    world = {
        f.name: known[f.name] if f.name in known else rng.choice(f.domain) for f in family.facts
    }
    records: list[tuple[str, str]] = []  # (id, lines) of the non-target records
    for record in near:
        twin = dict(known)
        for fact in rng.sample(family.facts, rng.randint(1, 2)):
            twin[fact.name] = rng.choice([v for v in fact.domain if v != known.get(fact.name)])
        records.append((record, _template(rng, family, twin, rules)))
    target_text = _target_lines(rng, family, known, rules)

    text = instructions(family, derived, target)
    shell = {
        "model": g.MODEL_NAME,
        "instructions": text,
        "state": "",
        "questions": g.grade(family, known, world, derived, items)[0],
        "samples": "auto",
        "seed": 0,
    }
    amendments = rng.randint(*spec.amendments)
    longest = max(_jlen(correction(target, f, v)) + 2 for f in family.facts for v in f.domain)
    reserve = longest * amendments + 10  # corrections and the seed's digits
    room = spec.tokens * CHARS_PER_TOKEN - len(_canonical(shell)) - reserve
    size = sum(_jlen(f"{header(r)}\n{t}") + 4 for r, t in [(target, target_text), *records])
    while True:
        if prose and rng.random() < PROSE_SHARE:
            lines = _block(str(rng.choice(prose)["text"]))
        else:
            lines = _template(rng, family, g.sample_known(rng, family, rules.hidden)[0], rules)
        record = _fresh_id(rng, taken)
        cost = _jlen(f"{header(record)}\n{lines}") + 4
        if size + cost > room:
            break
        size += cost
        records.insert(rng.randint(0, len(records)), (record, lines))
    position = rng.randint(0, len(records))  # needle depth: uniform over the dossier
    records.insert(position, (target, target_text))

    # ponytail: amendments land after their record at a uniform gap; a correction to a
    # record never precedes it, so the solver need not check positions.
    others = [r for r, _ in records if r != target and r not in near]
    gaps: dict[int, list[tuple[str, g.Fact, g.Value]]] = {}
    index = {r: i for i, (r, _) in enumerate(records)}
    for _ in range(amendments):
        draw = rng.random()
        if draw < TARGET_SHARE:
            record = target
        elif draw < TARGET_SHARE + NEAR_SHARE and near:
            record = rng.choice(near)
        else:
            record = rng.choice(others or [target])
        fact = rng.choice(family.facts)
        gap = rng.randint(index[record] + 1, len(records))
        gaps.setdefault(gap, []).append((record, fact, rng.choice(fact.domain)))

    chunks: list[str] = []
    effective = dict(known)
    for i, (record, lines) in enumerate(records, 1):
        chunks.append(f"{header(record)}\n{lines}" if lines else header(record))
        for amended, fact, value in gaps.get(i, []):
            chunks.append(correction(amended, fact, value))
            if amended == target:
                effective[fact.name] = value
                world[fact.name] = value  # the true world agrees with the dossier
    effective = {f.name: effective[f.name] for f in family.facts if f.name in effective}
    questions, gold, realized = g.grade(family, effective, world, derived, items)
    body = {
        "model": g.MODEL_NAME,
        "instructions": text,
        "state": "\n\n".join(chunks),
        "questions": questions,
        "samples": "auto",
        "seed": rng.getrandbits(31),
    }
    if len(_canonical(body).encode()) >= MAX_BODY_BYTES:
        raise g.GeneratorError(f"longctx L{level}: body exceeds {MAX_BODY_BYTES} bytes")
    private = {"target": target, "near": near, "known": effective, "position": position}
    return g.Case(family.name, level, body, gold, realized, track="longctx", private=private)


def solve(body: Mapping[str, Any]) -> dict[str, list[float]]:
    """Exact posterior from the dossier text alone: the target's block, then its corrections
    in order."""
    text = str(body["instructions"])
    match = _TARGET.search(text)
    if not text.startswith(MARKER) or match is None:
        raise g.GeneratorError("not a dossier request")
    family, _ = g.read_context(text)
    target = match.group(1)
    lines = str(body["state"]).split("\n")
    if lines.count(header(target)) != 1:
        raise g.GeneratorError(f"record #{target} does not appear exactly once")
    start = lines.index(header(target)) + 1
    end = start
    while end < len(lines) and lines[end]:
        end += 1
    known = g.extract(family, "\n".join(lines[start:end]))
    if known is None:
        raise g.GeneratorError(f"record #{target} states a fact twice")
    fixes = {
        f"the {fact.label} is {fact.show(value)}.": (fact.name, value)
        for fact in family.facts
        for value in fact.domain
    }
    prefix = f"Correction to record #{target}: "
    for line in lines[end:]:
        if line.startswith(prefix):
            hit = fixes.get(line.removeprefix(prefix))
            if hit is None:
                raise g.GeneratorError(f"unreadable correction {line!r}")
            known[hit[0]] = hit[1]
    return g.solve_known(body, known)
