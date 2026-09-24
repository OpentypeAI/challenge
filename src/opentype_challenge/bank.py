"""Duel cases: a window secret, a per-job seed and a level mix give every case exactly.

ponytail: the seed is HMAC(secret_w, job_id | challenger digest) with a commit-reveal
window secret; no drand round is mixed in yet. Add the round at lease time if the operator
itself must be unable to pick favourable cases.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

from .generator import FAMILIES, Case, make_case


def commitment(secret: bytes) -> str:
    return hashlib.sha256(secret).hexdigest()


def job_seed(secret: bytes, job_id: str, digest: str) -> str:
    return hmac.new(secret, f"{job_id}|{digest}".encode(), hashlib.sha256).hexdigest()


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
