"""A ledger of certified gain: crowns create entitlements, epochs pay them FIFO.

Amounts are integer units of 1e-9 epoch-mass so repeated payments never drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext

from .scoring import G_MIN

UNITS = 10**9  # one epoch-mass
HALF_LIFE_SECONDS = 36 * 60 * 60


def decayed_units(crowned_at: int, epoch_at: int, budget: int = UNITS) -> int:
    """Full budget for 36 hours, then a continuous 36-hour half-life; round down."""
    if any(type(value) is not int or value < 0 for value in (crowned_at, epoch_at, budget)):
        raise ValueError("reward timestamps and budget must be nonnegative integers")
    if crowned_at > epoch_at:
        raise ValueError("champion was crowned after the reward epoch")
    elapsed = max(0, epoch_at - crowned_at - HALF_LIFE_SECONDS)
    if elapsed == 0 or budget == 0:
        return budget
    # Once even a whole-half-life bound is below one unit, everything burns.
    if elapsed // HALF_LIFE_SECONDS >= budget.bit_length():
        return 0
    with localcontext() as context:
        context.prec = max(50, len(str(budget)) + 30)
        return int(Decimal(budget) * Decimal(2) ** (-Decimal(elapsed) / HALF_LIFE_SECONDS))


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
