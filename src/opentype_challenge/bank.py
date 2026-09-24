"""Duel cases: a window secret, a drand beacon, a per-job seed and the window's teacher bank
give every case exactly (docs/tracks.md §2, §8)."""

from __future__ import annotations

import hashlib
import hmac
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import httpx

from .generator import FAMILIES, Case, make_case

if TYPE_CHECKING:
    from .generator import Family

BANK_KINDS = ("family", "prose", "ops_story", "depict")
DRAND_URL = "https://api.drand.sh/public/latest"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def item_key(kind: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical([kind, payload]).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BankItem:
    kind: str
    key: str
    payload: dict[str, Any]

    @classmethod
    def make(cls, kind: str, payload: dict[str, Any]) -> BankItem:
        if kind not in BANK_KINDS:
            raise ValueError(f"unknown bank kind {kind!r}")
        return cls(kind, item_key(kind, payload), payload)

    def to_json(self) -> list[Any]:
        return [self.kind, self.key, self.payload]


def bank_digest(items: Iterable[BankItem]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: (i.kind, i.key)):
        digest.update(canonical(item.to_json()).encode() + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class Bank:
    """The frozen teacher items of one window; private until the window closes."""

    items: tuple[BankItem, ...] = ()

    def of(self, kind: str) -> tuple[BankItem, ...]:
        return self._by_kind.get(kind, ())

    @cached_property
    def _by_kind(self) -> dict[str, tuple[BankItem, ...]]:
        out: dict[str, list[BankItem]] = {}
        for item in sorted(self.items, key=lambda i: (i.kind, i.key)):
            out.setdefault(item.kind, []).append(item)
        return {kind: tuple(items) for kind, items in out.items()}

    @cached_property
    def _families(self) -> tuple[Family, ...]:
        from .generator import family_from_json

        return tuple(family_from_json(item.payload) for item in self.of("family"))

    def families(self) -> tuple[Family, ...]:
        """The sealed families of this window, parsed and validated."""
        return self._families

    @cached_property
    def digest(self) -> str:
        return bank_digest(self.items)

    @classmethod
    def from_json(cls, rows: Sequence[Sequence[Any]]) -> Bank:
        return cls(tuple(BankItem(str(k), str(key), dict(p)) for k, key, p in rows))

    def to_json(self) -> list[list[Any]]:
        return [item.to_json() for item in sorted(self.items, key=lambda i: (i.kind, i.key))]


EMPTY_BANK = Bank()


def drand_beacon(client: httpx.Client | None = None, timeout: float = 5.0) -> dict[str, Any] | None:
    """The latest drand round {round, randomness}, or None when drand is unreachable."""
    try:
        if client is None:
            response = httpx.get(DRAND_URL, timeout=timeout)
        else:
            response = client.get(DRAND_URL, timeout=timeout)
        data = response.json()
        round_, randomness = data["round"], data["randomness"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    if type(round_) is not int or round_ < 1:
        return None
    if not isinstance(randomness, str) or len(randomness) != 64:
        return None
    try:
        bytes.fromhex(randomness)
    except ValueError:
        return None
    return {"round": round_, "randomness": randomness.lower()}


def commitment(secret: bytes) -> str:
    return hashlib.sha256(secret).hexdigest()


def job_seed(
    secret: bytes, job_id: str, digest: str, beacon: Mapping[str, Any] | None = None
) -> str:
    """HMAC(secret_w, job | digest [| drand round | randomness]); v1 seed without a beacon."""
    message = f"{job_id}|{digest}"
    if beacon is not None:
        message += f"|{beacon['round']}|{beacon['randomness']}"
    return hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()


def pick_level(rng: random.Random, mix: Mapping[str, float]) -> int:
    levels = sorted(mix, key=int)
    point = rng.random() * sum(mix[level] for level in levels)
    for level in levels:
        point -= mix[level]
        if point < 0:
            return int(level)
    return int(levels[-1])


def job_case(seed: str, mix: Mapping[str, float], index: int) -> Case:
    rng = random.Random(f"{seed}|{index}")
    level = pick_level(rng, mix)
    return make_case(rng, rng.choice(FAMILIES), level)


def case_line(body: Mapping[str, Any]) -> bytes:
    """One line of the cases digest: the canonical JSON of a served request body."""
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def cases_digest(bodies: Iterable[Mapping[str, Any]]) -> str:
    """sha256 over the served request bodies in case order (worker evidence == audit)."""
    digest = hashlib.sha256()
    for body in bodies:
        digest.update(case_line(body))
    return digest.hexdigest()
