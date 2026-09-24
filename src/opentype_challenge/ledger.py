"""A ledger of certified gain: crowns create entitlements, epochs pay them FIFO.

Amounts are integer units of 1e-9 epoch-mass so repeated payments never drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .scoring import G_MIN

UNITS = 10**9  # one epoch-mass


@dataclass(frozen=True)
class Entitlement:
    id: int
    hotkey: str
    amount: int
    paid: int

    @property
    def outstanding(self) -> int:
        return self.amount - self.paid


def entitlement_units(g_lcb: float, window_total: int, window_cap: float | None) -> int:
    """E = g_LCB / g_min epoch-masses, clamped by the optional per-window cap."""
    units = int(g_lcb / G_MIN * UNITS)
    if window_cap is not None:
        units = min(units, max(int(window_cap * UNITS) - window_total, 0))
    return max(units, 0)


def pay(entitlements: Sequence[Entitlement], budget: int = UNITS) -> list[tuple[int, int]]:
    """FIFO payments (entitlement id, units) up to one epoch-mass; the rest burns."""
    payments = []
    for item in sorted(entitlements, key=lambda e: e.id):
        if budget <= 0:
            break
        amount = min(item.outstanding, budget)
        if amount > 0:
            payments.append((item.id, amount))
            budget -= amount
    return payments
