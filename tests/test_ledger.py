import json
import math
import sqlite3

from opentype_challenge import ledger
from opentype_challenge.ledger import UNITS, Entitlement
from opentype_challenge.scoring import G_MIN
from opentype_challenge.store import Settings, Store


def test_entitlement_is_g_lcb_over_g_min():
    assert ledger.entitlement_units(G_MIN, 0, None) == UNITS
    assert ledger.entitlement_units(2.5 * G_MIN, 0, None) == int(2.5 * UNITS)
    assert ledger.entitlement_units(-1.0, 0, None) == 0


def test_window_cap_clamps_total_entitlement():
    assert ledger.entitlement_units(3 * G_MIN, 0, 2.0) == 2 * UNITS
    assert ledger.entitlement_units(3 * G_MIN, int(1.5 * UNITS), 2.0) == UNITS // 2
    assert ledger.entitlement_units(3 * G_MIN, 5 * UNITS, 2.0) == 0


def test_fifo_pays_up_to_one_epoch_mass():
    owed = [Entitlement(2, "b", 3 * UNITS // 2, 0), Entitlement(1, "a", UNITS // 2, UNITS // 4)]
    assert ledger.pay(owed) == [(1, UNITS // 4), (2, 3 * UNITS // 4)]
    assert ledger.pay([Entitlement(1, "a", UNITS // 5, 0)]) == [(1, UNITS // 5)]
    assert ledger.pay([]) == []


def _store(tmp_path, clock=lambda: 1_790_000_000.0):
    return Store(tmp_path, Settings(duel_cases=10), clock)


def _grant(store: Store, hotkey: str, units: int) -> None:
    db = sqlite3.connect(store.path)
    with db:
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at) "
            "VALUES (?, 1, 1, ?, 0)",
            (hotkey, units),
        )
    db.close()


def test_epochs_are_persisted_replayed_and_burn_the_remainder(tmp_path):
    store = _store(tmp_path)
    empty = json.loads(store.weights(10, "opentype"))
    assert empty["weights"] == {} and empty["full_share_mass"] == 1.0
    assert empty["metadata"]["burned"] == 1.0

    _grant(store, "5Alice", int(1.5 * UNITS))
    _grant(store, "5Bob", UNITS // 4)
    first = store.weights(11, "opentype")
    assert json.loads(first)["weights"] == {"5Alice": 1.0}
    assert store.weights(11, "opentype") == first  # replay is byte-identical

    second = json.loads(store.weights(12, "opentype"))
    assert second["weights"] == {"5Alice": 0.5, "5Bob": 0.25}
    assert math.isclose(second["metadata"]["burned"], 0.25)
    assert json.loads(store.weights(13, "opentype"))["weights"] == {}

    # replays never pay twice, and a reopened store replays the same bytes
    assert _store(tmp_path).weights(11, "opentype") == first


def test_entitlements_survive_dethronement(tmp_path):
    store = _store(tmp_path)
    _grant(store, "5Old", 2 * UNITS)
    _grant(store, "5New", UNITS)
    paid = [json.loads(store.weights(e, "opentype"))["weights"] for e in (1, 2, 3)]
    assert paid == [{"5Old": 1.0}, {"5Old": 1.0}, {"5New": 1.0}]
