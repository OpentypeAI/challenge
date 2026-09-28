"""Continuous champion reward, measured at the reward epoch rather than request time."""

import json
import sqlite3

import pytest

from opentype_challenge.ledger import HALF_LIFE_SECONDS, UNITS, decayed_units
from opentype_challenge.store import Settings, Store, StoreError

from .conftest import ADMIN, INTERNAL, SLUG, Clock, bearer

T0 = 1_790_000_000
H = 3600


def test_full_grace_then_continuous_half_lives():
    crowned = 1_800_000_000
    for hours, expected in [(0, UNITS), (36, UNITS), (72, UNITS // 2), (108, UNITS // 4)]:
        assert decayed_units(crowned, crowned + hours * 3600) == expected
    assert decayed_units(crowned, crowned + HALF_LIFE_SECONDS + 1) < UNITS
    assert decayed_units(crowned, crowned + 54 * 3600) == 707106781
    assert decayed_units(crowned, crowned + 72 * 3600 - 1) > UNITS // 2
    assert decayed_units(crowned, crowned + 72 * 3600 + 1) < UNITS // 2


def test_independent_lane_budgets_round_down_without_redistribution():
    at = 72 * 3600
    assert decayed_units(0, at, 750_000_000) == 375_000_000
    assert decayed_units(0, at, 250_000_000) == 125_000_000
    assert decayed_units(0, at, 3) == 1
    assert decayed_units(0, 10**30) == 0
    assert decayed_units(0, at, 0) == 0


@pytest.mark.parametrize("args", [(2, 1), (-1, 1), (0, -1), (0, 1, -1), (True, 1), (0, 1.5)])
def test_invalid_reward_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        decayed_units(*args)


def test_decay_is_monotonic_and_new_crown_restores_full_budget():
    amounts = [decayed_units(0, hour * 3600) for hour in range(181)]
    assert amounts == sorted(amounts, reverse=True)
    assert decayed_units(180 * 3600, 180 * 3600) == UNITS


# -- store integration ---------------------------------------------------------------


def _store(tmp_path, clock, **settings):
    """The base champion is installed well before any crown the tests add."""
    now, clock.now = clock.now, T0 - 1000 * H
    store = Store(tmp_path, Settings(duel_cases=10, **settings), clock)
    clock.now = now
    return store


def _crown(store, hotkey, at, *, empty_bank=False, state="crowned", credit=0):
    """A miner champion certified by a crowned quality duel, as _crown leaves it."""
    db = sqlite3.connect(store.path)
    with db:
        window = db.execute("SELECT max(id) FROM windows").fetchone()[0]
        if not empty_bank:
            db.execute("UPDATE windows SET bank_digest='nonempty' WHERE id=?", (window,))
        job = f"job-{hotkey}-{at}"
        db.execute(
            "INSERT INTO jobs (id, submission_id, champion_id, window_id, seed, mix, retired, "
            "cases, state, verdict, created_at) VALUES (?, 's', 1, ?, '', '', '', 1, ?, ?, ?)",
            (job, window, state, json.dumps({"crown": True, "g_lcb": 0.1}), at),
        )
        cid = db.execute(
            "INSERT INTO champions (submission_id, hotkey, repo, revision, files, digest, "
            "job_id, g_lcb, window_id, crowned_at) VALUES ('s', ?, 'r', 'v', '{}', 'd', ?, "
            "0.1, ?, ?)",
            (hotkey, job, window, at),
        ).lastrowid
        if credit:
            db.execute(
                "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, paid, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (hotkey, cid, window, credit, credit, at),
            )
    db.close()
    return cid


def _activate(store, clock, epoch=10):
    store.set_reward_decay(epoch, int(clock.now) + 1)
    clock.now += 1
    json.loads(store.weights(epoch - 1, SLUG))  # a pre-cutoff epoch: legacy rule, burns
    return store.reward_decay_status()


def _paid(store, epoch, epoch_at):
    return json.loads(store.weights(epoch, SLUG, epoch_at))


def test_decay_pays_the_reigning_champion_by_chain_time_and_burns_the_rest(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    _crown(store, "5Champ", T0 - 72 * H)
    status = _activate(store, clock)
    assert status["active"]["epoch"] == 10 and status["pending"] is None
    body = _paid(store, 10, T0)
    assert body["weights"] == {"5Champ": 0.5}
    assert body["metadata"]["burned"] == 0.5 and body["full_share_mass"] == 1.0
    # a late or replayed request returns the frozen bytes, never repays
    clock.now += 10 * H
    assert store.weights(10, SLUG, T0 + 9 * H) == store.weights(10, SLUG)
    clock.now = T0 + 37 * H
    # a delayed request is priced at its epoch's chain time, not at request time
    assert _paid(store, 11, T0 + 36 * H)["weights"] == {"5Champ": 0.25}
    # a new certified champion restarts at full reward; the old one gets nothing
    _crown(store, "5Next", T0 + 40 * H)
    clock.now = T0 + 41 * H
    assert _paid(store, 12, T0 + 41 * H)["weights"] == {"5Next": 1.0}
    rows = sqlite3.connect(store.path).execute("SELECT count(*) FROM decay_payments")
    assert rows.fetchone()[0] == 3
    # reopening the store replays identical bodies
    assert _store(tmp_path, clock).weights(12, SLUG) == store.weights(12, SLUG)


def test_future_crown_admin_champion_and_uncertified_jobs_pay_nothing(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    _activate(store, clock)
    assert _paid(store, 10, T0)["weights"] == {}  # base champion: no hotkey
    _crown(store, "5Future", T0 + H)
    clock.now = T0 + 3 * H
    assert _paid(store, 11, T0 + H - 1)["weights"] == {}  # crowned after the epoch ended
    _crown(store, "5Revoked", T0 + 2 * H, state="rejected")
    assert _paid(store, 12, T0 + 2 * H)["weights"] == {}  # newest champion is not certified
    assert _paid(store, 13, T0 + 2 * H)["metadata"]["champion"] is not None


def test_epoch_at_is_required_and_never_in_the_future(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    _activate(store, clock)
    for bad in (None, int(clock.now) + 3600):
        with pytest.raises(StoreError) as error:
            store.weights(10, SLUG, bad)
        assert error.value.status == 422
    assert json.loads(store.weights(10, SLUG, T0))["epoch"] == 10


def test_caps_stay_cumulative_across_the_switch(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock, empty_bank_cap=1.5)
    _crown(store, "5Empty", T0, empty_bank=True, credit=UNITS)  # 1.0 already paid as credit
    _activate(store, clock)
    assert _paid(store, 10, T0)["weights"] == {"5Empty": 0.5}
    assert _paid(store, 11, T0)["weights"] == {}
    zero_clock = Clock(T0)
    zero = _store(tmp_path / "z", zero_clock)
    _crown(zero, "5Empty", T0, empty_bank=True)
    _activate(zero, zero_clock)
    assert _paid(zero, 10, T0)["weights"] == {}  # default empty-bank cap 0


def test_cutoff_needs_zero_debt_and_cancels_rather_than_dropping_credit(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    store.set_reward_decay(10, T0 + 60)
    with pytest.raises(StoreError):
        store.set_reward_decay(11, T0 + 120)  # already pending
    db = sqlite3.connect(store.path)
    with db:  # a crown before the cutoff still mints a normal credit
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at) "
            "VALUES ('5Owed', 1, 1, ?, 0)",
            (UNITS * 2,),
        )
    db.close()
    clock.now = T0 + 60
    first = json.loads(store.weights(10, SLUG, T0 + 60))
    assert first["weights"] == {"5Owed": 1.0}  # legacy rule kept: the debt is paid
    status = store.reward_decay_status()
    assert status["active"] is None and status["pending"] is None
    assert status["cancelled"]["reason"] == "outstanding credit debt at cutoff"
    # once drained, the operator can schedule again
    assert json.loads(store.weights(11, SLUG))["weights"] == {"5Owed": 1.0}
    store.set_reward_decay(12, T0 + 120)
    assert store.reward_decay_status()["cancelled"] is None


def test_crowns_after_the_cutoff_mint_no_credit(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    store.set_reward_decay(10, T0 + 60)
    clock.now = T0 + 61
    with store._tx() as db:  # the path _crown takes before minting
        store._maybe_activate_decay(db)
    assert store.reward_decay_status()["active"]["epoch"] == 10


def test_scheduling_rules(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    store.weights(5, SLUG)
    with pytest.raises(StoreError):
        store.set_reward_decay(5, T0 + 60)  # epoch already persisted
    with pytest.raises(StoreError):
        store.set_reward_decay(6, T0)  # not prospective
    store.set_lanes_from(6)
    with pytest.raises(StoreError):
        store.set_reward_decay(7, T0 + 60)  # runtime lanes exclude decay
    other = _store(tmp_path / "o", Clock(T0))
    other.set_reward_decay(1, T0 + 60)
    with pytest.raises(StoreError):
        other.set_lanes_from(2)


def test_leaderboard_counts_decay_payments(tmp_path):
    clock = Clock(T0)
    store = _store(tmp_path, clock)
    _crown(store, "5Champ", T0, credit=UNITS // 2)
    _activate(store, clock)
    _paid(store, 10, T0)
    board = store.leaderboard()
    crown = next(c for c in board["crowns"] if c.get("hotkey") == "5Champ")
    assert crown["legacy_paid"] == 0.5 and crown["decay_paid"] == 1.0 and crown["paid"] == 1.5
    assert board["hotkeys"]["5Champ"]["paid"] == 1.5


def test_api_admin_schedule_and_internal_epoch_at(make_client, clock):
    client = make_client()
    admin = bearer(ADMIN)
    assert client.get("/v1/admin/rewards/decay").status_code == 401
    at = int(clock.now) + 60
    put = client.put("/v1/admin/rewards/decay", json={"epoch": 3, "epoch_at": at}, headers=admin)
    assert put.status_code == 200 and put.json()["pending"] == {"epoch": 3, "epoch_at": at}
    clock.now = at
    headers = {**bearer(INTERNAL), "x-platform-challenge-slug": SLUG}
    missing = client.get("/internal/v1/get_weights?epoch=3", headers=headers)
    assert missing.status_code == 422
    ok = client.get(f"/internal/v1/get_weights?epoch=3&epoch_at={at}", headers=headers)
    assert ok.status_code == 200 and ok.json()["metadata"]["policy"] == "champion_decay"
    status = client.get("/v1/admin/rewards/decay", headers=admin).json()
    assert status["active"]["epoch"] == 3
