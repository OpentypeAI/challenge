"""The runtime lane: option allowlist, calibration, the pure verdict, 75/25 allocations, the
v2 -> v3 migration, the signed runtime contract, the store lifecycle and the worker's
sequential B/C/B' orchestration (docs/operator.md, runtime lane)."""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import sqlite3
from contextlib import asynccontextmanager
from typing import Any

import pytest

from opentype_challenge import crypto, harness, ledger, runtime, tracks, worker
from opentype_challenge.crypto import manifest_digest
from opentype_challenge.miner import signed_runtime_submission, signed_submission
from opentype_challenge.runtime import Calibration, Fidelity
from opentype_challenge.store import Settings, Store, StoreError

from .conftest import ADMIN, SLUG, WORKER, Clock, Miner, bearer, weights_manifest
from .fake_inference import answer, blur, chat_reply

PROFILE = {
    **runtime.PROFILE_FIXED,
    "gpu": "NVIDIA H200",
    "driver": "570.00",
    "vllm_version": "0.11.1rc2.dev77+g7f1a5398",
}


def calibration_json(**over: Any) -> dict[str, Any]:
    """Test thresholds only: production has none until the operator's pilot."""
    raw = {
        "version": "pilot-test",
        "profile": PROFILE,
        "cells": {
            "short": {
                "track": "decisions",
                "cases": 4,
                "concurrency": 2,
                "slo_ms": 1000.0,
                "weight": 0.5,
                "warm": False,
            },
            "chat": {
                "track": "ops",
                "cases": 2,
                "concurrency": 1,
                "slo_ms": 5000.0,
                "weight": 0.5,
                "warm": True,
            },
        },
        "blocks": 5,
        "max_drift": 0.05,
        "min_gain": 0.01,
        "latency_tolerance": 0.1,
        "fidelity_loss_tolerance": 0.01,
        "fidelity_accuracy_tolerance": 0.005,
        "bootstrap_resamples": 2000,
        "credit_per_log_gain": 10.0,
        "credit_cap": 2.0,
    }
    raw.update(over)
    return raw


CAL = Calibration.from_json(calibration_json())
GOOD = Fidelity(loss=10.0, decisions=100, determined=80, correct=78, cases=4)
GOODS = {track: GOOD for track in runtime.fidelity_tracks(CAL)}  # decisions and ops
CASES = {track: 4 for track in GOODS}


def run_of(cal: Calibration, seconds: float, p95: float = 500.0, ok: Any = None) -> dict:
    return {
        name: {
            "tasks": cell.cases,
            "ok": cell.cases if ok is None else ok,
            "errors": 0,
            "seconds": seconds,
            "p95_ms": p95,
        }
        for name, cell in cal.cells.items()
    }


def evidence_of(
    cal: Calibration = CAL, gain: float = 0.2, jitter: float = 0.0, **block: Any
) -> dict[str, Any]:
    blocks = []
    for i in range(cal.blocks):
        g = gain + (jitter if i % 2 else -jitter)
        runs = {
            "B": run_of(cal, 10.0),
            "C": run_of(cal, 10.0 / math.exp(g)),
            "B2": run_of(cal, 10.0),
        }
        blocks.append({"order": ["B", "C", "B2"], "runs": runs, "quiescent": [True] * 3, **block})
    return {"profile": cal.profile, "blocks": blocks}


# -- options and calibration --------------------------------------------------------------


def test_options_are_an_allowlist_with_fixed_flags():
    assert runtime.normalize_options({"max_num_seqs": 128, "enable_prefix_caching": False}) == {
        "enable_prefix_caching": False,
        "max_num_seqs": 128,
    }
    assert runtime.options_argv({"max_num_seqs": 128, "enable_chunked_prefill": False}) == [
        "--no-enable-chunked-prefill",
        "--max-num-seqs",
        "128",
    ]
    assert runtime.options_argv({}) == []
    for bad in (
        {"max_num_seqs": 0},
        {"max_num_seqs": True},
        {"max_num_seqs": "64"},
        {"enable_prefix_caching": 1},
        {"kernel": "x.so"},
        {"env": {"LD_PRELOAD": "x"}},
        {"extra_args": "--trust-remote-code"},
        {},
    ):
        with pytest.raises(runtime.RuntimeError_):
            runtime.normalize_options(bad)


def test_calibration_is_strict():
    assert CAL.public()["profile_digest"] == runtime.digest(PROFILE)
    for broken in (
        calibration_json(profile={**PROFILE, "vllm_image": "other"}),
        calibration_json(profile={k: v for k, v in PROFILE.items() if k != "gpu"}),
        calibration_json(cells={"a": {**calibration_json()["cells"]["short"], "weight": 0.7}}),
        calibration_json(
            cells={"a": {**calibration_json()["cells"]["short"], "track": "paint", "weight": 1.0}}
        ),
        calibration_json(blocks=2),
        calibration_json(min_gain=math.nan),
        calibration_json(max_drift=-1),
        {**calibration_json(), "extra": 1},
    ):
        with pytest.raises(runtime.RuntimeError_):
            Calibration.from_json(broken)


# -- the pure verdict -----------------------------------------------------------------------


def test_a_known_gain_crowns_with_its_lcb():
    result = runtime.verdict(CAL, evidence_of(gain=0.2, jitter=0.01), GOODS, GOODS, CASES)
    assert result["decision"] == "crown" and result["crown"]
    assert math.isclose(result["gain_mean"], 0.2 - 0.01 / 5)  # 3 blocks at -j, 2 at +j
    assert 0.185 < result["g_lcb"] < result["gain_mean"]
    assert result["seconds_per_ok_task"]["C"] < result["seconds_per_ok_task"]["B"]


def test_noise_below_the_margin_is_no_crown():
    result = runtime.verdict(CAL, evidence_of(gain=0.005, jitter=0.02), GOODS, GOODS, CASES)
    assert result["decision"] == "reject" and result["reason"] == "no certified gain"


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda e: e["blocks"][0]["runs"]["B2"]["short"].update(seconds=20.0), "drifted"),
        (lambda e: e["blocks"][0]["runs"]["B"]["short"].update(ok=0), "reference completed"),
        (lambda e: e["blocks"][0]["runs"]["C"]["short"].update(seconds=math.nan), "malformed"),
        (lambda e: e["blocks"][0]["runs"]["C"]["short"].update(p95_ms=math.inf), "malformed"),
        (lambda e: e["blocks"][0]["runs"]["C"].pop("chat"), "malformed"),
        (lambda e: e["blocks"][0]["runs"]["C"]["short"].update(tasks=1), "malformed"),
        (lambda e: e["blocks"][0]["runs"]["C"]["short"].update(ok=99), "malformed"),
        (lambda e: e["blocks"][0].update(order=["C", "B", "B2"]), "B, C, B2"),
        (lambda e: e["blocks"][0].update(quiescent=[True, False, True]), "quiescence"),
        (lambda e: e["blocks"].pop(), "blocks"),
        (lambda e: e.update(profile={**PROFILE, "gpu": "B300"}), "profile"),
    ],
)
def test_infrastructure_doubt_is_no_decision(mutate, reason):
    evidence = evidence_of()
    mutate(evidence)
    result = runtime.verdict(CAL, evidence, GOODS, GOODS, CASES)
    assert result["decision"] == "no_decision" and not result["crown"]
    assert reason in result["reason"]


def test_candidate_failures_on_healthy_infrastructure_reject():
    dead = evidence_of()
    dead["blocks"][0]["runs"]["C"]["short"]["ok"] = 0
    assert runtime.verdict(CAL, dead, GOODS, GOODS, CASES)["decision"] == "reject"
    slow = evidence_of()
    for block in slow["blocks"]:
        block["runs"]["C"]["short"]["p95_ms"] = 900.0
    assert "latency" in runtime.verdict(CAL, slow, GOODS, GOODS, CASES)["reason"]
    worse = Fidelity(loss=12.0, decisions=100, determined=80, correct=78, cases=4)
    wrong = Fidelity(loss=10.0, decisions=100, determined=80, correct=70, cases=4)
    for track in GOODS:  # every measured track is guarded, not only decisions
        for bad, reason in ((worse, "loss"), (wrong, "accuracy")):
            result = runtime.verdict(CAL, evidence_of(), {**GOODS, track: bad}, GOODS, CASES)
            assert result["decision"] == "reject" and result["reason"].startswith(track)
            assert reason in result["reason"]
    missing = Fidelity(loss=10.0, decisions=100, determined=80, correct=78, cases=3)
    for track in GOODS:
        partial = {**GOODS, track: missing}
        assert runtime.verdict(CAL, evidence_of(), partial, GOODS, CASES)["decision"] == (
            "no_decision"
        )
    no_ops = {"decisions": GOOD}
    assert runtime.verdict(CAL, evidence_of(), no_ops, no_ops, {"decisions": 4})["decision"] == (
        "no_decision"
    )


def test_declared_candidate_metrics_never_count():
    evidence = evidence_of(gain=0.0)
    evidence["candidate_claims"] = {"speedup": 10.0}
    evidence["blocks"][0]["runs"]["C"]["short"]["speedup"] = 10.0  # unknown key: refused
    assert runtime.verdict(CAL, evidence, GOODS, GOODS, CASES)["decision"] == "no_decision"
    assert runtime.verdict(CAL, evidence_of(gain=0.0), GOODS, GOODS, CASES)["decision"] == "reject"


def test_credit_is_capped_and_never_negative():
    assert runtime.credit_units(CAL, 0.05, ledger.UNITS) == int(0.5 * ledger.UNITS)
    assert runtime.credit_units(CAL, 5.0, ledger.UNITS) == 2 * ledger.UNITS
    assert runtime.credit_units(CAL, -1.0, ledger.UNITS) == 0


# -- allocations -----------------------------------------------------------------------------


def store_of(tmp_path, clock=None, **settings: Any) -> Store:
    return Store(
        tmp_path / "data",
        Settings(**{"duel_cases": 10, "runtime_fidelity_cases": 4, **settings}),
        clock or Clock(),
        beacon=lambda: None,
    )


def grant(store: Store, hotkey: str, units: int, lane: str = "quality") -> None:
    with store._tx() as db:
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at, lane) "
            "VALUES (?, 1, 1, ?, 0, ?)",
            (hotkey, units, lane),
        )


def body(store: Store, epoch: int) -> dict[str, Any]:
    return json.loads(store.weights(epoch, SLUG))


def test_before_activation_the_historical_rule_holds(tmp_path):
    store = store_of(tmp_path)
    grant(store, "5A", 3 * ledger.UNITS)
    first = store.weights(1, SLUG)
    assert json.loads(first)["weights"] == {"5A": 1.0} and "lanes" not in first
    store.set_lanes_from(5)
    assert json.loads(store.weights(4, SLUG))["weights"] == {"5A": 1.0}
    assert store.weights(1, SLUG) == first  # replay unchanged


def test_lane_budgets_burn_what_is_unused(tmp_path):
    store = store_of(tmp_path)
    store.set_lanes_from(1)
    assert body(store, 1)["metadata"]["burned"] == 1.0
    grant(store, "5Q", 10 * ledger.UNITS)  # quality alone: at most 0.75
    only_quality = body(store, 2)
    assert only_quality["weights"] == {"5Q": 0.75}
    assert only_quality["metadata"]["lanes"]["runtime"]["burned"] == 0.25
    assert only_quality["full_share_mass"] == 1.0

    store2 = store_of(tmp_path / "r")
    store2.set_lanes_from(1)
    grant(store2, "5R", 10 * ledger.UNITS, "runtime")  # runtime alone: at most 0.25
    assert body(store2, 1)["weights"] == {"5R": 0.25}
    assert body(store2, 1)["metadata"]["lanes"]["quality"]["burned"] == 0.75

    grant(store, "5R", 10 * ledger.UNITS, "runtime")  # both saturated: exactly 1
    both = body(store, 3)
    assert both["weights"] == {"5Q": 0.75, "5R": 0.25} and both["metadata"]["burned"] == 0.0

    store3 = store_of(tmp_path / "same")
    store3.set_lanes_from(1)
    grant(store3, "5S", ledger.UNITS, "quality")
    grant(store3, "5S", ledger.UNITS, "runtime")
    assert body(store3, 1)["weights"] == {"5S": 1.0}  # one hotkey, both lanes, summed


def test_old_quality_debt_keeps_its_amount_and_pays_from_the_quality_budget(tmp_path):
    store = store_of(tmp_path)
    grant(store, "5Old", 2 * ledger.UNITS)
    assert body(store, 1)["weights"] == {"5Old": 1.0}
    store.set_lanes_from(2)
    paid = [body(store, e)["weights"].get("5Old", 0.0) for e in (2, 3)]
    assert paid == [0.75, 0.25]  # 1.0 owed after epoch 1, repaid at 0.75 per epoch


def test_activation_is_prospective_and_once(tmp_path):
    store = store_of(tmp_path)
    store.weights(7, SLUG)
    with pytest.raises(StoreError, match="already persisted"):
        store.set_lanes_from(7)
    store.set_lanes_from(8)
    with pytest.raises(StoreError, match="already scheduled"):
        store.set_lanes_from(9)


# -- migration -------------------------------------------------------------------------------


def test_v2_migrates_to_v3_preserving_debt_and_epochs(tmp_path):
    store = store_of(tmp_path)
    grant(store, "5A", int(1.5 * ledger.UNITS))
    first = store.weights(1, SLUG)
    store._db.close()
    db = sqlite3.connect(tmp_path / "data" / "opentype.sqlite3")
    db.executescript(  # back to the v2 layout: drop the v3 columns and table
        """
        DROP INDEX submissions_lane;
        ALTER TABLE entitlements DROP COLUMN lane;
        ALTER TABLE submissions DROP COLUMN lane;
        ALTER TABLE submissions DROP COLUMN options;
        ALTER TABLE submissions DROP COLUMN target;
        ALTER TABLE jobs DROP COLUMN lane;
        ALTER TABLE jobs DROP COLUMN incumbent_id;
        ALTER TABLE jobs DROP COLUMN calibration;
        DROP TABLE runtime_incumbents;
        PRAGMA user_version=2;
        """
    )
    db.close()
    migrated = store_of(tmp_path)
    assert migrated._db.execute("PRAGMA user_version").fetchone()[0] == 3
    assert {r[0] for r in migrated._db.execute("SELECT lane FROM entitlements")} == {"quality"}
    assert migrated.weights(1, SLUG) == first  # byte-identical replay
    assert body(migrated, 2)["weights"] == {"5A": 0.5}  # the debt survived
    store_of(tmp_path)  # reopening is a no-op


def test_a_failed_migration_rolls_back(tmp_path):
    store = store_of(tmp_path)
    store._db.close()
    db = sqlite3.connect(tmp_path / "data" / "opentype.sqlite3")
    # v2 layout, and entitlements unreadable: the last column of the migration fails
    db.executescript(
        """
        DROP INDEX submissions_lane;
        ALTER TABLE submissions DROP COLUMN lane;
        ALTER TABLE submissions DROP COLUMN options;
        ALTER TABLE submissions DROP COLUMN target;
        ALTER TABLE entitlements RENAME TO entitlements_elsewhere;
        PRAGMA user_version=2;
        """
    )
    db.close()
    with pytest.raises(sqlite3.OperationalError):
        store_of(tmp_path)
    db = sqlite3.connect(tmp_path / "data" / "opentype.sqlite3")
    assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    columns = {r[1] for r in db.execute("PRAGMA table_info(submissions)")}
    assert not columns & {"lane", "options", "target"}  # the earlier columns rolled back
    db.close()


def test_a_v3_file_restamped_v2_by_an_older_binary_migrates(tmp_path):
    """An older binary opening a v3 file writes user_version=2 but keeps every v3 column:
    the migration reads the columns, so it adds none twice and loses nothing."""
    store = store_of(tmp_path)
    store.set_calibration(calibration_json())
    store.set_lanes_from(1)
    grant(store, "5R", ledger.UNITS, "runtime")
    first = store.weights(1, SLUG)
    store._db.close()
    db = sqlite3.connect(tmp_path / "data" / "opentype.sqlite3")
    db.execute("PRAGMA user_version=2")
    db.close()
    again = store_of(tmp_path)
    assert again._db.execute("PRAGMA user_version").fetchone()[0] == 3
    assert again.weights(1, SLUG) == first
    assert {r[0] for r in again._db.execute("SELECT lane FROM entitlements")} == {"runtime"}
    assert again.runtime_status()["open"]


def test_a_partial_v2_to_v3_file_gets_only_its_missing_columns(tmp_path):
    store = store_of(tmp_path)
    store._db.close()
    db = sqlite3.connect(tmp_path / "data" / "opentype.sqlite3")
    db.executescript(
        "DROP INDEX submissions_lane; ALTER TABLE submissions DROP COLUMN lane; "
        "PRAGMA user_version=2;"
    )
    db.close()
    migrated = store_of(tmp_path)
    columns = {r[1] for r in migrated._db.execute("PRAGMA table_info(submissions)")}
    assert {"lane", "options", "target"} <= columns


# -- the signed contract -------------------------------------------------------------------


def open_lane(client) -> dict[str, Any]:
    put = client.put(
        "/v1/admin/runtime/calibration", json=calibration_json(), headers=bearer(ADMIN)
    )
    assert put.status_code == 200, put.text
    assert client.put("/v1/admin/lanes", json={"epoch": 1}, headers=bearer(ADMIN)).json() == {
        "lanes_from_epoch": 1
    }
    return client.get("/v1/runtime").json()


def runtime_body(state: dict[str, Any], who: Miner, clock: Clock, **over: Any) -> dict:
    options = over.pop("options", {"max_num_seqs": 128})
    return signed_runtime_submission(
        SLUG,
        state["target"],
        state["calibration"]["profile_digest"],
        options,
        who.signer,
        clock.now,
    )


def test_runtime_lane_is_closed_until_calibrated(client, miner, clock):
    state = client.get("/v1/runtime").json()
    assert not state["open"] and state["calibration"] is None
    assert state["kernels"].startswith("disabled")
    fake = {**state, "calibration": {"profile_digest": "0" * 64}}
    assert (
        client.post("/v1/runtime/submissions", json=runtime_body(fake, miner, clock)).status_code
        == 503
    )
    assert client.post("/v1/worker/lease?lane=runtime", headers=bearer(WORKER)).status_code == 204


def test_runtime_signatures_bind_every_field(client, miner, clock):
    state = open_lane(client)
    good = runtime_body(state, miner, clock)
    for field, value in (
        ("options", {"max_num_seqs": 64}),
        ("profile", "1" * 64),
        ("target", {"champion": 1, "digest": "2" * 64}),
    ):
        assert (
            client.post("/v1/runtime/submissions", json={**good, field: value}).status_code == 401
        )
    # a weights signature over the same digest never verifies as a runtime one
    digest = crypto.runtime_digest(
        SLUG, state["target"], state["calibration"]["profile_digest"], good["options"]
    )
    legacy = miner.signer.sign(
        crypto.submit_message(miner.signer.public, digest, good["nonce"], good["exp"])
    )
    assert (
        client.post("/v1/runtime/submissions", json={**good, "signature": legacy.hex()}).status_code
        == 401
    )
    extras: list[dict[str, Any]] = [{"argv": ["--x"]}, {"image": "evil"}, {"env": {}}]
    for extra in extras:
        assert client.post("/v1/runtime/submissions", json={**good, **extra}).status_code == 422
    bad = runtime_body(state, miner, clock)
    bad["options"] = {"kernel": "x"}
    assert client.post("/v1/runtime/submissions", json=bad).status_code == 422
    accepted = client.post("/v1/runtime/submissions", json=good)
    assert accepted.status_code == 201, accepted.text
    assert client.post("/v1/runtime/submissions", json=good).status_code == 409  # nonce
    again = runtime_body(state, miner, clock, options={"max_num_seqs": 64})
    assert "open runtime submission" in client.post("/v1/runtime/submissions", json=again).text
    # the same hotkey still has its own quality slot
    quality = signed_submission(weights_manifest("q"), miner.signer, clock.now)
    assert client.post("/v1/submissions", json=quality).status_code == 201


def test_runtime_submission_must_target_the_current_champion_and_profile(client, miner, clock):
    state = open_lane(client)
    stale = {**state, "target": {"champion": 2, "digest": state["target"]["digest"]}}
    assert (
        client.post("/v1/runtime/submissions", json=runtime_body(stale, miner, clock)).status_code
        == 409
    )
    other = {**state, "calibration": {"profile_digest": "3" * 64}}
    assert (
        client.post("/v1/runtime/submissions", json=runtime_body(other, miner, clock)).status_code
        == 409
    )


def test_an_old_worker_never_receives_runtime_jobs(client, miner, clock):
    state = open_lane(client)
    client.post("/v1/runtime/submissions", json=runtime_body(state, miner, clock))
    assert client.post("/v1/worker/lease", headers=bearer(WORKER)).status_code == 204
    leased = client.post("/v1/worker/lease?lane=runtime", headers=bearer(WORKER)).json()
    assert leased["lane"] == "runtime" and leased["challenger"] is None
    assert leased["runtime"]["candidate"] == {"max_num_seqs": 128}
    assert leased["runtime"]["incumbent"] == {}


# -- store lifecycle -----------------------------------------------------------------------


def quality_submit(store: Store, tag: str) -> dict[str, Any]:
    manifest = weights_manifest(tag)
    digest = manifest_digest(manifest["repo"], manifest["revision"], manifest["files"])
    return store.submit(
        f"5Q-{tag}",
        manifest["repo"],
        manifest["revision"],
        manifest["files"],
        digest,
        secrets.token_hex(16),
        int(store.clock()) + 60,
    )


def runtime_submit(store: Store, hotkey: str, options: dict[str, Any]) -> dict[str, Any]:
    status = store.runtime_status()
    return store.submit_runtime(
        hotkey,
        status["target"],
        CAL.profile_digest,
        runtime.normalize_options(options),
        "d" * 64,
        secrets.token_hex(16),
        int(store.clock()) + 60,
    )


def lane_store(tmp_path) -> Store:
    store = store_of(tmp_path)
    store.set_calibration(calibration_json())
    store.set_lanes_from(1)
    return store


def side_item(case: dict[str, Any], skill: str) -> dict[str, Any]:
    """One side's raw output on a served case, from the fake model of this skill."""
    body_ = case["body"]
    if case["track"] in tracks.ENVS:

        async def generate(messages: list[dict[str, Any]], _seed: int) -> str:
            return chat_reply(skill, messages)

        env = tracks.ENVS[case["track"]]
        return {"transcript": asyncio.run(harness.run_episode(env, body_, generate))}
    gold = tracks.solve_body(body_)
    answers = {
        qid: answer(q, blur(skill, qid, body_["seed"], gold[qid]))
        for qid, q in body_["questions"].items()
    }
    return {
        "answers": answers,
        "reads": {qid: {"label_mass": 1.0, "argmax_is_label": True} for qid in answers},
    }


def answer_both(store: Store, lease: dict[str, Any], skill: dict[str, str]) -> None:
    items = [
        {"case_index": case["index"], "side": side, **side_item(case, skill[side])}
        for case in store.cases(lease["job"], lease["lease"], 0, 100)
        for side in ("champion", "challenger")
    ]
    store.record_answers(lease["job"], lease["lease"], items)


def time_tasks(store: Store, lease: dict[str, Any], skill: dict[str, str] | None = None) -> None:
    """Every B/C/B2 task of every block as a worker reports it: raw output and latency."""
    skill = skill or {}
    cal = Calibration.from_json(lease["runtime"]["calibration"])
    items = []
    for block in range(cal.blocks):
        for side in runtime.SIDES:
            for name, cell in cal.cells.items():
                for i in range(cell.cases):
                    case = runtime.cell_case(lease["runtime"]["seed"], name, cell, i)
                    served = {"index": i, "track": case.track, "body": case.body}
                    item = side_item(served, skill.get(side, "exact"))
                    items.append(
                        {
                            "block": block,
                            "side": side,
                            "cell": name,
                            "case_index": i,
                            "ms": 100.0,
                            **item,
                        }  # fmt: skip
                    )
    store.record_timings(lease["job"], lease["lease"], items)


def seconds_of(evidence: dict[str, Any]) -> dict[str, Any]:
    """The worker-side evidence of evidence_of: per-run seconds only, no counts."""
    return {
        "profile": evidence["profile"],
        "blocks": [
            {
                "order": b["order"],
                "quiescent": b["quiescent"],
                "seconds": {
                    side: {n: m["seconds"] for n, m in run.items()}
                    for side, run in b["runs"].items()
                },
            }
            for b in evidence["blocks"]
        ],
    }


def run_runtime_job(
    store: Store,
    evidence: dict[str, Any],
    skill: str = "exact",
    timed: dict[str, str] | None = None,
) -> dict:
    lease = store.lease("runtime")
    assert lease is not None
    answer_both(store, lease, {"champion": "exact", "challenger": skill})
    time_tasks(store, lease, timed)
    return store.complete(lease["job"], lease["lease"], {"runtime": seconds_of(evidence)})


def test_runtime_crown_pays_from_its_own_budget(tmp_path):
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    result = run_runtime_job(store, evidence_of(gain=0.05))
    assert result["state"] == "crowned", result
    status = store.runtime_status()
    assert status["incumbent"]["hotkey"] == "5R" and status["crowns"][0]["credited"] > 0
    assert store.submission(sid)["job"]["verdict"]["decision"] == "crown"
    paid = body(store, 1)
    assert set(paid["weights"]) == {"5R"} and paid["weights"]["5R"] <= 0.25
    # the quality lane and its leaderboard are untouched
    assert store.leaderboard()["hotkeys"] == {} and store.status()["champion"]["id"] == 1


def test_fidelity_regression_rejects(tmp_path):
    store = lane_store(tmp_path)
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    assert run_runtime_job(store, evidence_of(gain=0.3), skill="base")["state"] == "rejected"


def test_no_decision_retries_then_fails(tmp_path):
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    drift = evidence_of()
    drift["blocks"][0]["runs"]["B2"]["short"]["seconds"] = 30.0
    for _ in range(3):
        run_runtime_job(store, drift)
    final = store.submission(sid)
    assert final["state"] == "failed" and "no decision" in final["reason"]
    assert store.runtime_status()["crowns"] == []


def test_runtime_leases_are_exclusive(tmp_path):
    store = lane_store(tmp_path)
    quality_submit(store, "a")
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    quality = store.lease("quality")
    assert quality is not None and store.lease("runtime") is None  # waits for the GPU
    store.fail(quality["job"], quality["lease"], "done", True, {})
    runtime_lease = store.lease("runtime")
    assert runtime_lease is not None
    assert store.lease("quality") is None  # nothing runs beside a runtime measurement


def test_champion_change_expires_runtime_work_and_recertification_pays_nothing(tmp_path):
    store = lane_store(tmp_path)
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    run_runtime_job(store, evidence_of(gain=0.05))
    first_credit = store.runtime_status()["crowns"][0]["credited"]
    # a queued runtime submission signed against champion 1
    queued = runtime_submit(store, "5Late", {"max_num_seqs": 256})["id"]
    with store._tx() as db:  # a quality crown (champion 2) without running a whole duel
        db.execute(
            "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
            "VALUES ('m/new', 'r2', '{}', ?, 0)",
            ("e" * 64,),
        )
        store._expire_runtime(db)  # what _crown runs after it inserts a champion
    assert store.submission(queued)["state"] == "expired"
    status = store.runtime_status()
    assert status["incumbent"] is None  # back to stock on the new weights
    assert status["target"]["champion"] == 2
    # the same options recertified on the new weights: crowned, but no new credit
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    assert run_runtime_job(store, evidence_of(gain=0.05))["state"] == "crowned"
    crowns = store.runtime_status()["crowns"]
    assert [c["model_champion_id"] for c in crowns] == [1, 2]
    assert crowns[1]["credited"] == 0 and first_credit > 0
    # a real improvement past the best certified gain is paid only for the excess
    runtime_submit(store, "5Next", {"max_num_seqs": 512})
    run_runtime_job(store, evidence_of(gain=0.08))
    assert 0 < store.runtime_status()["crowns"][2]["credited"] <= CAL.credit_cap


def test_a_real_quality_crown_expires_queued_runtime_work(tmp_path):
    store = store_of(tmp_path, duel_cases=240)
    store.set_calibration(calibration_json())
    store.set_lanes_from(1)
    quality_submit(store, "winner")
    queued = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("quality")
    assert lease is not None
    for offset in range(0, lease["cases"], 100):
        page = {**lease}
        items = []
        for case in store.cases(lease["job"], lease["lease"], offset, 100):
            body_, gold = case["body"], tracks.solve_body(case["body"])
            for side, skill in (("champion", "base"), ("challenger", "exact")):
                answers = {
                    qid: answer(q, blur(skill, qid, body_["seed"], gold[qid]))
                    for qid, q in body_["questions"].items()
                }
                reads = {qid: {"label_mass": 1.0, "argmax_is_label": True} for qid in answers}
                items.append(
                    {"case_index": case["index"], "side": side, "answers": answers, "reads": reads}
                )
        store.record_answers(page["job"], page["lease"], items)
    assert store.complete(lease["job"], lease["lease"], {})["state"] == "crowned"
    assert store.submission(queued)["state"] == "expired"
    assert store.runtime_status()["target"]["champion"] == 2


def test_a_calibration_change_mid_job_duels_again(tmp_path):
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None
    store.set_calibration(calibration_json(version="pilot-2"))
    answer_both(store, lease, {"champion": "exact", "challenger": "exact"})
    store.complete(lease["job"], lease["lease"], {"runtime": evidence_of()})
    job = store.submission(sid)["job"]
    assert store.submission(sid)["state"] == "queued" and job["id"] != lease["job"]


# -- the worker ------------------------------------------------------------------------------


class FakeLauncher:
    """Records every server start; quiescent() can be made to fail once."""

    def __init__(self, profile: dict[str, Any] = PROFILE, dirty_after: int | None = None):
        self._profile, self.dirty_after = profile, dirty_after
        self.starts: list[dict[str, Any]] = []

    def evidence(self) -> dict[str, Any]:
        return {}

    def profile(self) -> dict[str, Any]:
        return self._profile

    def quiescent(self) -> bool:
        return self.dirty_after is None or len(self.starts) < self.dirty_after

    @asynccontextmanager
    async def __call__(self, models, extra=None, share=None):
        self.starts.append({"sides": sorted(models), "extra": dict(extra or {}), "share": share})
        yield {
            side: {"reader": f"http://{side}-reader.test", "chat": f"http://{side}-chat.test"}
            for side in models
        }


def bench(
    monkeypatch, launcher: FakeLauncher, candidate: dict[str, Any], posted: list | None = None
) -> dict[str, Any]:
    async def read(self, job, urls, sides=worker.SIDES):
        assert set(urls) == set(sides)  # each pass reads only the server it started
        reads.append(tuple(sides))
        return {"cases_fetched": 4, "cases_sha256": "x", "errors": 0}

    async def measure(self, job, cal, urls):
        return {n: 10.0 for n in cal.cells}, [{"cell": "short", "case_index": 0, "ms": 1.0}]

    async def post_timings(self, job, block, side, tasks):
        (posted if posted is not None else []).append((block, side, tasks))

    monkeypatch.setattr(worker.Worker, "_post_timings", post_timings)
    reads: list[tuple[str, ...]] = []

    monkeypatch.setattr(worker, "base_snapshot", lambda directory, fetch: directory)
    monkeypatch.setattr(worker.Worker, "_champion", lambda self, m, b: (b, {}))
    monkeypatch.setattr(worker.Worker, "_read", read)
    monkeypatch.setattr(worker.Worker, "_measure", measure)
    instance = worker.Worker(None, None, launcher, lane="runtime")  # type: ignore[arg-type]
    instance.workdir = __import__("pathlib").Path("/nonexistent")
    job = {
        "champion": {},
        "runtime": {
            "calibration": calibration_json(),
            "incumbent": {},
            "candidate": candidate,
            "seed": "s",
        },
    }
    return asyncio.run(instance._bench(job, None, {}))  # type: ignore[arg-type]


def verdict_of(out: dict[str, Any]) -> dict[str, Any]:
    """The container's side: the worker's seconds plus tasks scored exact at 100 ms."""
    tasks = [
        {"block": b, "side": side, "cell": n, "ms": 100.0, "ok": True, "error": False}
        for b in range(len(out["runtime"]["blocks"]))
        for side in runtime.SIDES
        for n, cell in CAL.cells.items()
        for _ in range(cell.cases)
    ]
    blocks = runtime.runs_from_tasks(CAL, out["runtime"]["blocks"], tasks)
    measured = {"profile": out["runtime"]["profile"], "blocks": blocks}
    return runtime.verdict(CAL, measured, GOODS, GOODS, CASES)


def test_worker_runs_b_c_b2_one_server_at_a_time(monkeypatch):
    launcher, posted = FakeLauncher(), list[Any]()
    out = bench(monkeypatch, launcher, {"max_num_seqs": 128}, posted)
    assert [(b, side) for b, side, _ in posted] == [
        (b, side) for b in range(CAL.blocks) for side in runtime.SIDES
    ]
    assert "ok" not in json.dumps(out["runtime"])  # the worker reports no success flag
    stock, candidate, *measured = launcher.starts
    # fidelity: one server at a time at the calibrated share, stock pristine and first
    assert stock == {"sides": ["champion"], "extra": {"champion": []}, "share": 0.9}
    assert candidate == {
        "sides": ["challenger"],
        "extra": {"challenger": ["--max-num-seqs", "128"]},
        "share": 0.9,
    }
    assert len(measured) == 3 * CAL.blocks
    assert all(start["sides"] == ["champion"] and start["share"] == 0.9 for start in measured)
    flags = [start["extra"]["champion"] for start in measured]
    assert flags == [[], ["--max-num-seqs", "128"], []] * CAL.blocks
    assert out["runtime"]["profile"] == PROFILE
    assert verdict_of(out)["decision"] == "reject"  # no gain


def test_worker_stops_on_a_dirty_gpu_and_the_verdict_is_no_decision(monkeypatch):
    out = bench(monkeypatch, FakeLauncher(dirty_after=4), {"max_num_seqs": 128})
    blocks = out["runtime"]["blocks"]
    assert len(blocks) == 1 and blocks[0]["quiescent"] == [True, False]
    assert verdict_of(out)["decision"] == "no_decision"


def test_worker_refuses_an_uncalibrated_profile(monkeypatch):
    with pytest.raises(worker.JobFailed, match="calibrated profile") as error:
        bench(monkeypatch, FakeLauncher(profile={**PROFILE, "gpu": "B300"}), {"max_num_seqs": 1})
    assert error.value.retry


def test_worker_never_forwards_miner_text_to_vllm(monkeypatch):
    with pytest.raises(runtime.RuntimeError_):
        bench(monkeypatch, FakeLauncher(), {"max_num_seqs": "1; rm -rf /"})


def test_serving_env_drops_credentials(monkeypatch):
    monkeypatch.setenv("OPENTYPE_WORKER_TOKEN", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    env = worker.scrubbed_env()
    assert not {"OPENTYPE_WORKER_TOKEN", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert "PATH" in env


def test_vllm_command_appends_only_allowlisted_flags():
    serve, _ = worker.VllmLauncher().commands(
        "champion", worker.Path("/m"), runtime.options_argv({"max_num_seqs": 64}), 0.9
    )
    assert serve[-2:] == ["--max-num-seqs", "64"]
    assert serve[serve.index("--gpu-memory-utilization") + 1] == "0.9"


def test_measure_reports_raw_outputs_scored_only_by_the_container(monkeypatch):
    """_measure against fake servers: latencies from the worker's clock, raw outputs only;
    the store's task_ok decides success against gold, so a wrong model never counts."""
    import httpx

    from . import fake_inference as fake

    def serve(skill: str):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            route = fake.systemone if request.url.path == "/v1/systemone" else fake.chat
            code, reply = route(skill, payload)
            return httpx.Response(code, json=reply)

        return handler

    async def go(skill: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
        transport = httpx.MockTransport(serve(skill))
        async with httpx.AsyncClient(transport=transport) as client:
            instance = worker.Worker(None, None, FakeLauncher(), inference=client)  # type: ignore[arg-type]
            job = {"runtime": {"seed": "seed"}}
            urls = {"reader": "http://r.test", "chat": "http://c.test"}
            return await instance._measure(job, CAL, urls)

    def scored(tasks: list[dict[str, Any]]) -> list[bool]:
        return [
            runtime.task_ok(
                runtime.cell_case("seed", t["cell"], CAL.cells[t["cell"]], t["case_index"]), t
            )
            for t in tasks
        ]

    seconds, tasks = asyncio.run(go("exact"))
    assert set(seconds) == set(CAL.cells) and all(v > 0 for v in seconds.values())
    assert len(tasks) == sum(c.cases for c in CAL.cells.values())
    assert all("ok" not in t and t["ms"] >= 0 for t in tasks)
    assert all(scored(tasks))
    _, broken = asyncio.run(go("broken"))  # well-formed replies, wrong content
    assert not any(scored(broken))


def test_task_ok_never_trusts_a_declared_success():
    cell = CAL.cells["short"]
    case = runtime.cell_case("s", "short", cell, 0)
    served = {"index": 0, "track": case.track, "body": case.body}
    good = side_item(served, "exact")
    assert runtime.task_ok(case, good)
    assert not runtime.task_ok(case, {**good, "error": "x"})
    keys_only = {"answers": {q: {} for q in case.body["questions"]}, "reads": good["reads"]}
    assert not runtime.task_ok(case, {**keys_only, "ok": True})
    ops = runtime.cell_case("s", "chat", CAL.cells["chat"], 0)
    ops_served = {"index": 0, "track": ops.track, "body": ops.body}
    assert runtime.task_ok(ops, side_item(ops_served, "exact"))
    assert not runtime.task_ok(ops, side_item(ops_served, "broken"))
    assert not runtime.task_ok(ops, {"transcript": "not a list", "ok": True})


# -- review regressions ----------------------------------------------------------------------


def crown_quality_champion(store: Store) -> None:
    """A quality crown (champion 2) without running a whole duel."""
    with store._tx() as db:
        db.execute(
            "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
            "VALUES ('m/new', 'r2', '{}', ?, 0)",
            ("e" * 64,),
        )
        store._expire_runtime(db)


def test_a_leased_runtime_job_never_moves_to_an_unsigned_champion(tmp_path):
    """Leased against champion 1, champion 2 is crowned, the worker fails with retry: the
    job expires instead of being requeued and retargeted at weights nobody signed for."""
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None and lease["champion"]["repo"] == store.status()["champion"]["repo"]
    crown_quality_champion(store)
    store.fail(lease["job"], lease["lease"], "vllm crashed", True, {})
    final = store.submission(sid)
    assert final["state"] == "expired" and final["job"]["champion"] == 1
    assert store.lease("runtime") is None
    assert store.runtime_status()["crowns"] == [] and body(store, 1)["weights"] == {}


@pytest.mark.parametrize("path", ["lease_expired", "no_decision", "complete"])
def test_every_release_path_expires_a_retargeted_runtime_job(tmp_path, path):
    clock = Clock()
    store = store_of(tmp_path, clock)
    store.set_calibration(calibration_json())
    store.set_lanes_from(1)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None
    crown_quality_champion(store)
    if path == "lease_expired":
        clock.now += 10_000
        assert store.lease("runtime") is None
    else:
        answer_both(store, lease, {"champion": "exact", "challenger": "exact"})
        time_tasks(store, lease)
        drift = evidence_of()
        if path == "no_decision":
            drift["blocks"][0]["runs"]["B2"]["short"]["seconds"] = 30.0
        store.complete(lease["job"], lease["lease"], {"runtime": seconds_of(drift)})
    assert store.submission(sid)["state"] == "expired"
    assert store.runtime_status()["crowns"] == []


def test_a_target_is_never_rewritten(tmp_path):
    store = lane_store(tmp_path)
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    with store._tx() as db:
        job = db.execute("SELECT id FROM jobs WHERE lane='runtime'").fetchone()
        db.execute(
            "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
            "VALUES ('m/new', 'r2', '{}', ?, 0)",
            ("e" * 64,),
        )
        with pytest.raises(StoreError, match="signed target"):
            store._target(db, job["id"])


def test_timed_outputs_are_scored_by_the_container(tmp_path):
    """A candidate whose timed replies are fast but wrong completes nothing under SLO: the
    worker's latencies count only for outputs the container scored right."""
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    result = run_runtime_job(store, evidence_of(gain=0.3), timed={"C": "broken"})
    assert result["state"] == "rejected"
    assert "candidate completed no task" in store.submission(sid)["job"]["verdict"]["reason"]


def test_timings_refuse_foreign_cells_and_quality_jobs(tmp_path):
    store = lane_store(tmp_path)
    quality_submit(store, "q")
    quality = store.lease("quality")
    assert quality is not None
    item = {"block": 0, "side": "C", "cell": "short", "case_index": 0, "ms": 1.0}
    with pytest.raises(StoreError, match="runtime jobs"):
        store.record_timings(quality["job"], quality["lease"], [item])
    store.fail(quality["job"], quality["lease"], "done", False, {})
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    lease = store.lease("runtime")
    assert lease is not None
    for bad in ({**item, "cell": "other"}, {**item, "case_index": 99}, {**item, "block": 9}):
        with pytest.raises(StoreError):
            store.record_timings(lease["job"], lease["lease"], [bad])


def test_fidelity_covers_every_measured_track(tmp_path):
    store = lane_store(tmp_path)
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    lease = store.lease("runtime")
    assert lease is not None and set(lease["plan"]) == {"decisions", "ops"}
    tracks_served = {c["track"] for c in store.cases(lease["job"], lease["lease"], 0, 100)}
    assert tracks_served == {"decisions", "ops"}


def test_queued_runtime_work_drains_quality_leases_boundedly(tmp_path):
    """A waiting runtime job pauses quality leases after runtime_every of them so the GPU
    drains; quality then gets its runtime_every again, and nothing pauses without a runtime
    worker polling."""
    clock = Clock()
    store = store_of(tmp_path, clock, runtime_every=2, max_pending=10)
    store.set_calibration(calibration_json())
    store.set_lanes_from(1)
    for tag in "abcdef":
        quality_submit(store, tag)
    runtime_submit(store, "5R", {"max_num_seqs": 128})
    runtime_submit(store, "5S", {"max_num_seqs": 256})

    def quality_round() -> int:
        leased = 0
        while (lease := store.lease("quality")) is not None:
            store.fail(lease["job"], lease["lease"], "done", False, {})
            leased += 1
        return leased

    assert quality_round() == 6  # no runtime worker has polled: quality is never paused
    first = store.lease("runtime")  # a runtime worker polls; the GPU is free
    assert first is not None
    store.fail(first["job"], first["lease"], "done", False, {})
    for tag in "ghijkl":
        quality_submit(store, tag)
    # the second runtime job waits: quality gets exactly runtime_every leases, then drains
    held = [store.lease("quality"), store.lease("quality")]
    assert all(held) and store.lease("quality") is None
    assert store.lease("runtime") is None  # still waiting for the two leased jobs
    for lease in held:
        assert lease is not None
        store.fail(lease["job"], lease["lease"], "done", False, {})
    second = store.lease("runtime")
    assert second is not None
    store.fail(second["job"], second["lease"], "done", False, {})
    assert quality_round() == 4  # nothing waits any more: quality runs freely
    runtime_submit(store, "5T", {"max_num_seqs": 64})
    for tag in "mno":
        quality_submit(store, tag)
    clock.now += 1_000  # the runtime worker went away: quality is never held for it
    assert quality_round() == 3


def _lease_of(store: Store, job_id: str) -> str:
    return store._db.execute("SELECT lease FROM jobs WHERE id=?", (job_id,)).fetchone()[0]


def test_invalid_scheduler_combinations_are_refused_at_intake():
    with pytest.raises(runtime.RuntimeError_, match="max_num_seqs"):
        runtime.normalize_options({"max_num_seqs": 512, "max_num_batched_tokens": 256})
    with pytest.raises(runtime.RuntimeError_, match="max_model_len"):
        runtime.normalize_options({"max_num_batched_tokens": 8192, "enable_chunked_prefill": False})
    assert runtime.normalize_options(
        {"max_num_batched_tokens": 131072, "enable_chunked_prefill": False}
    )


def test_profile_reads_the_build_and_fails_closed(monkeypatch, tmp_path):
    reader = tmp_path / "structured_server.py"
    reader.write_text("x")
    manifest = tmp_path / "build.json"
    monkeypatch.setattr(worker, "BUILD_MANIFEST", manifest)
    monkeypatch.setattr(worker, "_gpu_identity", lambda: ("NVIDIA H200", "570.00"))
    monkeypatch.setattr(worker, "_package_version", lambda name: "0.11.1")
    launcher = worker.VllmLauncher(reader=reader)
    with pytest.raises(worker.JobFailed, match="vllm_image") as error:
        launcher.profile()  # no manifest baked: refuse, never assume the pin
    assert error.value.retry
    manifest.write_text(json.dumps({"vllm_image": "vllm/other@sha256:1"}))
    profile = launcher.profile()
    assert profile["vllm_image"] == "vllm/other@sha256:1" and profile["vllm_version"] == "0.11.1"
    assert set(profile) == {*runtime.PROFILE_FIXED, *runtime.MEASURED}
    assert profile != {**profile, "vllm_image": runtime.PROFILE_FIXED["vllm_image"]}
    assert worker.VllmLauncher(reader=reader, dtype="float16").profile()["dtype"] == "float16"
    monkeypatch.setattr(worker, "_package_version", lambda name: None)
    with pytest.raises(worker.JobFailed, match="vllm_version"):
        launcher.profile()


def test_calibration_requires_the_measured_build_identity():
    no_version = {k: v for k, v in PROFILE.items() if k != "vllm_version"}
    with pytest.raises(runtime.RuntimeError_):
        Calibration.from_json(calibration_json(profile=no_version))
    with pytest.raises(runtime.RuntimeError_):
        Calibration.from_json(calibration_json(profile={**PROFILE, "vllm_version": ""}))


class FailingLauncher(FakeLauncher):
    """A server of `side` fails to start on start number `at` (0 = fidelity)."""

    def __init__(self, side: str, at: int):
        super().__init__()
        self.side, self.at = side, at

    @asynccontextmanager
    async def __call__(self, models, extra=None, share=None):
        number = len(self.starts)
        self.starts.append({"sides": sorted(models), "extra": dict(extra or {}), "share": share})
        if number == self.at:
            raise worker.ServeFailed("never healthy", self.side)
        yield {s: {"reader": "http://r.test", "chat": "http://c.test"} for s in models}


@pytest.mark.parametrize(
    "side,at,retry",
    [
        ("champion", 0, True),  # stock fidelity never started: the host
        ("challenger", 1, False),  # stock healthy, the candidate never started: rejected
        ("champion", 2, True),  # B never started
        ("champion", 3, False),  # B healthy just before, C never started
        ("champion", 4, True),  # B2 never started
    ],
)
def test_startup_failures_blame_the_candidate_only_after_a_healthy_reference(
    monkeypatch, side, at, retry
):
    with pytest.raises(worker.JobFailed) as error:
        bench(monkeypatch, FailingLauncher(side, at), {"max_num_seqs": 128})
    assert error.value.retry is retry


def test_timings_route_takes_raw_outputs_not_verdicts(client, miner, clock):
    state = open_lane(client)
    client.post("/v1/runtime/submissions", json=runtime_body(state, miner, clock))
    lease = client.post("/v1/worker/lease?lane=runtime", headers=bearer(WORKER)).json()
    item = {"block": 0, "side": "C", "cell": "short", "case_index": 0, "ms": 5.0, "answers": {}}
    url = f"/v1/worker/jobs/{lease['job']}/timings"
    ok = client.post(url, json={"lease": lease["lease"], "items": [item]}, headers=bearer(WORKER))
    assert ok.status_code == 200 and ok.json() == {"accepted": 1, "ok": 0}
    for bad in ({**item, "ok": True}, {**item, "side": "champion"}, {**item, "ms": -1}):
        response = client.post(
            url, json={"lease": lease["lease"], "items": [bad]}, headers=bearer(WORKER)
        )
        assert response.status_code == 422


def test_one_side_fidelity_passes_complete_a_real_job(tmp_path):
    """The real _read, stock pass then candidate pass through the store's API: every case
    gets both sides and the job settles on the container's scores."""
    import httpx

    from . import fake_inference as fake

    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/cases"):
            q = request.url.params
            cases = store.cases(lease["job"], q["lease"], int(q["offset"]), int(q["limit"]))
            return httpx.Response(200, json={"cases": cases})
        payload = json.loads(request.content)
        return httpx.Response(
            200, json=store.record_answers(lease["job"], payload["lease"], payload["items"])
        )

    def inference(request: httpx.Request) -> httpx.Response:
        route = fake.systemone if request.url.path == "/v1/systemone" else fake.chat
        code, reply = route("exact", json.loads(request.content))
        return httpx.Response(code, json=reply)

    async def go() -> None:
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(api)) as api_client,
            httpx.AsyncClient(transport=httpx.MockTransport(inference)) as infer_client,
        ):
            instance = worker.Worker(
                worker.Api("http://api.test", "w", api_client),
                None,  # type: ignore[arg-type]
                FakeLauncher(),
                inference=infer_client,
            )
            for side in ("champion", "challenger"):
                urls = {side: {"reader": "http://r.test", "chat": "http://c.test"}}
                await instance._read(lease, urls, (side,))

    asyncio.run(go())
    time_tasks(store, lease)
    job = store.submission(sid)["job"]
    assert job["paired"] == lease["cases"]  # both one-side passes landed on every case
    store.complete(lease["job"], lease["lease"], {"runtime": seconds_of(evidence_of())})
    assert store.submission(sid)["job"]["verdict"]["decision"] == "crown"


def test_withdrawing_the_calibration_parks_runtime_work(tmp_path):
    store = lane_store(tmp_path)
    sid = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None
    answer_both(store, lease, {"champion": "exact", "challenger": "exact"})
    time_tasks(store, lease)
    store.set_calibration(None)  # mid-job: the job duels again, parked until reopened
    store.complete(lease["job"], lease["lease"], {"runtime": seconds_of(evidence_of())})
    assert store.submission(sid)["state"] == "queued"
    store.requeue(store.submission(sid)["job"]["id"])  # no calibration: still no crash
    assert store.lease("runtime") is None
    store.set_calibration(calibration_json(version="pilot-2"))
    again = store.lease("runtime")
    assert again is not None and again["runtime"]["calibration"]["version"] == "pilot-2"


def test_a_new_profile_expires_work_signed_for_the_old_one(tmp_path):
    store = lane_store(tmp_path)
    queued = runtime_submit(store, "5Q", {"max_num_seqs": 64})["id"]
    other = {**PROFILE, "driver": "575.00"}
    store.set_calibration(calibration_json(profile=other))
    assert store.submission(queued)["state"] == "expired"  # never measured on an unsigned one
    # in flight: the job turns stale on completion and expires rather than duelling again
    store.set_calibration(calibration_json())
    inflight = runtime_submit(store, "5R", {"max_num_seqs": 128})["id"]
    lease = store.lease("runtime")
    assert lease is not None
    store.set_calibration(calibration_json(profile=other))
    answer_both(store, lease, {"champion": "exact", "challenger": "exact"})
    store.complete(lease["job"], lease["lease"], {"runtime": seconds_of(evidence_of())})
    assert store.submission(inflight)["state"] == "expired"
    assert store.lease("runtime") is None


def _one_cell(**over: Any) -> dict[str, Any]:
    return {"a": {**calibration_json()["cells"]["short"], "weight": 1.0, **over}}


@pytest.mark.parametrize(
    "over",
    [
        {"blocks": runtime.MAX_BLOCKS + 1},
        {"bootstrap_resamples": runtime.MAX_RESAMPLES + 1},
        {"cells": _one_cell(cases=runtime.MAX_CELL_CASES + 1)},
        {"cells": _one_cell(concurrency=runtime.MAX_CONCURRENCY + 1)},
    ],
)
def test_calibration_workload_is_bounded(over):
    with pytest.raises(runtime.RuntimeError_):
        Calibration.from_json(calibration_json(**over))
    Calibration.from_json(calibration_json(blocks=runtime.MAX_BLOCKS))  # the timings API's bound
