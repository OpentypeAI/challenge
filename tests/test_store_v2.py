"""v2 store and container: window banks and rotation, byte-bounded case pages, the judgments
lifecycle, the v1 -> v2 migration, the drand beacon and early stop on the composite
(docs/tracks.md §8, §11)."""

from __future__ import annotations

import asyncio
import json
import random
import secrets
import sqlite3
import time
from typing import Any

import pytest

from opentype_challenge import bank, paint, scoring, tracks
from opentype_challenge.bank import EMPTY_BANK, BankItem
from opentype_challenge.crypto import manifest_digest
from opentype_challenge.store import (
    JUDGE_DEADLINE_SECONDS,
    LEASE_SECONDS,
    PAGE_BYTES,
    Settings,
    Store,
    judge_order,
)
from opentype_challenge.tracks import TrackPlan

from .conftest import ADMIN, WORKER, Clock, bearer, weights_manifest
from .test_paint import DEPICT

BEACON = {"round": 4242, "randomness": "ab" * 32}
DONE = json.dumps({"tool": "done", "args": {}})
DRAW = json.dumps(
    {
        "tool": "draw",
        "args": {"commands": [{"op": "rect", "x": 20, "y": 20, "w": 80, "h": 80, "fill": "red"}]},
    }
)


def store_of(tmp_path, clock=None, **kwargs: Any) -> Store:
    settings = kwargs.pop("settings", Settings(duel_cases=10))
    kwargs.setdefault("beacon", lambda: None)
    return Store(tmp_path / "data", settings, clock or Clock(), **kwargs)


def enqueue(store: Store, tag: str) -> dict[str, Any]:
    manifest = weights_manifest(tag)
    digest = manifest_digest(manifest["repo"], manifest["revision"], manifest["files"])
    exp = int(store.clock()) + 60
    return store.submit(
        f"hotkey-{tag}", manifest["repo"], manifest["revision"], manifest["files"], digest,
        secrets.token_hex(16), exp,
    )  # fmt: skip


def lease_of(store: Store) -> dict[str, Any]:
    lease = store.lease()
    assert lease is not None
    return lease


def with_bank(store: Store, items: list[BankItem]) -> None:
    store.set_next_bank(items)
    store.rotate_window()


# -- banks and rotation --------------------------------------------------------------


def test_first_window_is_empty_and_rotation_promotes_the_next_bank(tmp_path):
    clock = Clock()
    store = store_of(tmp_path, clock)
    first = store.window(1)
    assert first["bank_digest"] == EMPTY_BANK.digest and not store.next_bank_ready()
    with pytest.raises(Exception, match="not published"):
        store.window_bank(1, 0, 10)  # open: sealed
    items = [BankItem.make("depict", DEPICT)]
    digest = store.set_next_bank(items)
    assert store.next_bank_ready() and store.window(1)["bank_digest"] == EMPTY_BANK.digest
    assert store.auto_rotate(24 * 3600) is None  # the window is too young
    clock.now += 24 * 3600
    rotated = store.auto_rotate(24 * 3600)
    assert rotated is not None and rotated["bank_digest"] == digest == bank.bank_digest(items)
    assert store.window(2)["bank_digest"] == digest and not store.next_bank_ready()
    assert store.window_bank(1, 0, 10)["items"] == []
    clock.now += 48 * 3600
    assert store.auto_rotate(24 * 3600) is None  # no next bank: the window stays open
    closed = store.rotate_window()  # the admin route still works, with an empty bank
    assert closed["bank_digest"] == EMPTY_BANK.digest
    page = store.window_bank(2, 0, 10)
    assert bank.Bank.from_json(page["items"]).digest == digest and page["total"] == 1


def test_bank_pages_stay_under_the_limit(tmp_path):
    store = store_of(tmp_path)
    big = [
        BankItem.make("prose", {"family": "x", "text": f"{i} " + "y" * 900_000}) for i in range(9)
    ]
    with_bank(store, big)
    store.rotate_window()
    rows: list[Any] = []
    while len(rows) < 9:
        page = store.window_bank(2, len(rows), 1000)
        assert 1 <= len(page["items"]) and len(json.dumps(page)) < PAGE_BYTES
        rows += page["items"]
    assert bank.Bank.from_json(rows).digest == store.window(2)["bank_digest"]


def test_app_builds_the_next_bank_and_publishes_it_after_rotation(make_client):
    built: list[random.Random] = []

    async def builder(rng: random.Random) -> list[BankItem]:
        built.append(rng)
        return [BankItem.make("depict", DEPICT)]

    client = make_client(bank_builder=builder)
    for _ in range(200):
        if client.get("/v1/status").json()["teacher"]["state"] == "ready":
            break
        time.sleep(0.01)
    status = client.get("/v1/status").json()
    assert status["teacher"]["state"] == "ready" and len(built) == 1
    assert status["window"]["bank_digest"] == EMPTY_BANK.digest
    assert client.get("/v1/windows/1/bank").status_code == 404
    rotated = client.post("/v1/admin/window/rotate", headers=bearer(ADMIN)).json()
    assert rotated["bank_digest"] == bank.bank_digest([BankItem.make("depict", DEPICT)])
    assert client.get("/v1/windows/2/bank").status_code == 404  # open
    assert client.get("/v1/windows/1/bank").json()["items"] == []
    assert "token" not in json.dumps(status)


# -- cases ------------------------------------------------------------------------


def test_case_pages_are_byte_bounded_and_match_job_case(tmp_path):
    store = store_of(tmp_path, settings=Settings(plan={"longctx": TrackPlan(1.0, 80)}))
    enqueue(store, "a")
    lease = lease_of(store)
    assert lease is not None and lease["plan"] == {"longctx": {"cases": 80, "weight": 1.0}}
    pages = []
    offset = 0
    while offset < 80:
        page = store.cases(lease["job"], lease["lease"], offset, 200)
        assert page and len(json.dumps({"cases": page}, separators=(",", ":"))) <= PAGE_BYTES
        pages.append(page)
        offset += len(page)
    assert len(pages) > 1
    served = [case for page in pages for case in page]
    assert [c["index"] for c in served] == list(range(80))
    assert {c["track"] for c in served} == {"longctx"}
    job = store._db.execute("SELECT seed, mix FROM jobs").fetchone()
    expected = tracks.job_case(job["seed"], {"longctx": TrackPlan(1.0, 80)}, {}, 7, EMPTY_BANK)
    assert served[7]["body"] == expected.body


def test_answer_model_limits(client):
    def post(item: dict[str, Any]) -> Any:
        body = {"lease": "x", "items": [{"case_index": 0, "side": "champion", **item}]}
        return client.post("/v1/worker/jobs/j_x/answers", headers=bearer(WORKER), json=body)

    assert post({"transcript": ["x"] * 65}).status_code == 422
    assert post({"transcript": ["x" * 8193]}).status_code == 422
    assert post({"transcript": ["x" * 8192] * 64}).status_code == 404  # valid: unknown job


# -- judgments ---------------------------------------------------------------------


def depict_store(tmp_path) -> Store:
    store = store_of(tmp_path, settings=Settings(plan={"paint": TrackPlan(1.0, 12)}), judge=True)
    with_bank(store, [BankItem.make("depict", DEPICT)])
    return store


def answer_paint(store: Store, lease: dict[str, Any]) -> list[dict[str, Any]]:
    """The challenger draws a square then stops; the champion stops on a blank canvas."""
    cases = store.cases(lease["job"], lease["lease"], 0, 100)
    items = [
        {"case_index": c["index"], "side": side, "transcript": outputs}
        for c in cases
        for side, outputs in (("champion", [DONE]), ("challenger", [DRAW, DONE]))
    ]
    result = store.record_answers(lease["job"], lease["lease"], items)
    assert result["accepted"] == 24 and result["continue"] is False
    return cases


def test_judgments_lifecycle_with_an_unjudgeable_case(tmp_path):
    store = depict_store(tmp_path)
    sid = enqueue(store, "a")["id"]
    lease = lease_of(store)
    cases = answer_paint(store, lease)
    depicts = [c for c in cases if c["body"]["task"]["mode"] == "depict"]
    assert 0 < len(depicts) < 12
    assert "rubric" not in json.dumps(cases)
    job = store.complete(lease["job"], lease["lease"], {"w": 1})["job"]
    assert job["state"] == "judging" and job["judgments_pending"] == 2 * len(depicts)
    assert job["paired"] == 12 - len(depicts)  # judged pairs are not paired yet

    pending = store.pending_judgments()
    assert len(pending) == 2 * len(depicts)
    for c, (first, second) in zip(
        depicts, zip(pending[::2], pending[1::2], strict=True), strict=True
    ):
        assert first["case_index"] == second["case_index"] == c["index"]
        assert (first["side"], second["side"]) == judge_order(c["body"]["seed"])
        assert first["brief"] == DEPICT["brief"] and first["rubric"] == DEPICT["rubric"]
    renders = {(p["case_index"], p["side"]): p["png"] for p in pending}
    assert renders[(depicts[0]["index"], "champion")] == paint.blank_png()
    assert renders[(depicts[0]["index"], "challenger")] != paint.blank_png()

    unjudged = depicts[0]["index"]
    first = pending[0]
    assert first["case_index"] == unjudged
    store.record_judgment(first["job"], unjudged, first["side"], None)
    assert store.pending_judgments() and store.settle_judged() == []  # still pending
    for item in store.pending_judgments():
        blank = item["png"] == paint.blank_png()
        loss = None if item["case_index"] == unjudged else (1.0 if blank else 0.0)
        store.record_judgment(item["job"], item["case_index"], item["side"], loss)
    assert store.settle_judged() == [lease["job"]]

    result = store.submission(sid)
    verdict = result["job"]["verdict"]
    assert verdict["unjudged"] == 1  # both renders unreadable: dropped on both sides
    assert verdict["tracks"]["paint"]["pairs"] == 11 == result["job"]["paired"]
    assert verdict["unjudged_max"] == 0.05 and not verdict["crown"]  # 1 of < 12 judged
    assert result["job"]["state"] in {"rejected", "crowned"} or result["state"] == "queued"
    assert result["job"]["evidence"] == {"w": 1}


def test_app_complete_judges_in_order_and_settles(make_client):
    calls: list[bytes] = []

    async def judge(brief: str, rubric: list[str], png: bytes) -> float | None:
        assert brief == DEPICT["brief"] and rubric == DEPICT["rubric"]
        calls.append(png)
        return 1.0 if png == paint.blank_png() else 0.0

    client = make_client(judge=judge, plan={"paint": TrackPlan(1.0, 12)})
    store = client.app.state.store
    with_bank(store, [BankItem.make("depict", DEPICT)])
    status = client.get("/v1/status").json()
    assert status["teacher"]["judge"] is True and status["plan"]["paint"]["cases"] == 12
    sid = enqueue(store, "a")["id"]
    lease = client.post("/v1/worker/lease", headers=bearer(WORKER)).json()
    cases = answer_paint(store, lease)
    depicts = sum(c["body"]["task"]["mode"] == "depict" for c in cases)
    response = client.post(
        f"/v1/worker/jobs/{lease['job']}/complete",
        headers=bearer(WORKER),
        json={"lease": lease["lease"], "evidence": {}},
    )
    assert response.status_code == 200 and response.json()["job"]["state"] != "judging"
    assert len(calls) == 2 * depicts
    verdict = store.submission(sid)["job"]["verdict"]
    assert verdict["unjudged"] == 0 and verdict["tracks"]["paint"]["pairs"] == 12
    assert client.get("/v1/status").json()["tracks"]["paint"]["results"] == 24


def test_release_drops_pending_judgments(tmp_path):
    store = depict_store(tmp_path)
    enqueue(store, "a")
    lease = lease_of(store)
    answer_paint(store, lease)
    store.fail(lease["job"], lease["lease"], "gpu fell over", True, {})
    assert store.pending_judgments() == []


def test_a_retargeted_job_drops_the_previous_attempts_evidence(tmp_path):
    store = store_of(tmp_path)
    sid = enqueue(store, "a")["id"]
    lease = lease_of(store)
    store.fail(lease["job"], lease["lease"], "retry me", True, {"cases_sha256": "stale"})
    assert store.submission(sid)["job"]["evidence"] == {"cases_sha256": "stale"}
    lease_of(store)  # retargeted: a fresh window, mix or seed may differ from attempt 1's
    assert store.submission(sid)["job"]["evidence"] is None


# -- drand beacon ------------------------------------------------------------------


def test_beacon_is_stored_at_first_lease_and_reused_on_retry(tmp_path):
    calls = []

    def beacon() -> dict[str, Any]:
        calls.append(1)
        return BEACON

    store = store_of(tmp_path, beacon=beacon)
    submission = enqueue(store, "a")
    lease = lease_of(store)
    job = store._db.execute("SELECT * FROM jobs WHERE id=?", (lease["job"],)).fetchone()
    secret = store._db.execute("SELECT secret FROM windows WHERE id=1").fetchone()[0]
    assert json.loads(job["beacon"]) == BEACON
    assert job["seed"] == bank.job_seed(secret, job["id"], submission["digest"], BEACON)
    store.fail(lease["job"], lease["lease"], "retry me", True, {})
    again = lease_of(store)
    assert again["job"] == lease["job"] and len(calls) == 1
    assert store._db.execute("SELECT seed FROM jobs").fetchone()[0] == job["seed"]
    store.fail(again["job"], again["lease"], "done", False, {})
    store.rotate_window()
    published = store.window(1)["jobs"][0]
    assert published["beacon"] == BEACON and published["judge"] is False


def test_unreachable_drand_falls_back_to_the_v1_seed(tmp_path):
    store = store_of(tmp_path, beacon=lambda: None)
    submission = enqueue(store, "a")
    lease = lease_of(store)
    job = store._db.execute("SELECT * FROM jobs WHERE id=?", (lease["job"],)).fetchone()
    secret = store._db.execute("SELECT secret FROM windows WHERE id=1").fetchone()[0]
    assert job["beacon"] is None and job["beacon_fetched"] == 1
    assert job["seed"] == bank.job_seed(secret, job["id"], submission["digest"])


def test_the_beacon_follows_the_head_after_an_expired_lease_or_a_race(tmp_path):
    clock = Clock()
    fetched: list[str] = []
    racing: list[Any] = []

    def beacon() -> dict[str, Any]:
        if racing:  # a concurrent fail requeues the earlier job while drand is fetched
            job, lease = racing.pop()
            store.fail(job, lease, "infra", True, {})
        fetched.append(head())
        return BEACON

    def head() -> str:
        return store._db.execute(
            "SELECT j.id FROM jobs j JOIN submissions s ON s.id=j.submission_id "
            "WHERE j.state='queued' ORDER BY s.intake LIMIT 1"
        ).fetchone()[0]

    store = store_of(tmp_path, clock, beacon=beacon)
    enqueue(store, "a")
    first = lease_of(store)
    enqueue(store, "b")
    clock.now += LEASE_SECONDS + 1  # the expired lease is released before the head is read
    again = lease_of(store)
    assert again["job"] == first["job"] and fetched == [first["job"]]
    # a v1 job never fetched a beacon; requeued mid-fetch, it becomes the head and gets one
    store._db.execute("UPDATE jobs SET beacon_fetched=0 WHERE id=?", (first["job"],))
    racing.append((again["job"], again["lease"]))
    third = lease_of(store)
    assert third["job"] == first["job"] and fetched[-1] == first["job"]
    row = store._db.execute("SELECT beacon FROM jobs WHERE id=?", (first["job"],)).fetchone()
    assert json.loads(row["beacon"]) == BEACON
    other = store._db.execute("SELECT beacon_fetched FROM jobs WHERE id != ?", (first["job"],))
    assert other.fetchone()[0] == 0  # b keeps its own fetch for its own lease


# -- early stop --------------------------------------------------------------------


def test_early_stop_uses_the_per_track_composite(tmp_path):
    plan = {"decisions": TrackPlan(0.7, 900), "ops": TrackPlan(0.3, 900)}
    store = store_of(tmp_path, settings=Settings(plan=plan))
    enqueue(store, "a")
    lease = lease_of(store)
    rng = random.Random(3)
    pairs = []
    rows = []
    for index in range(1600):
        track = "decisions" if index % 2 else "ops"
        level = rng.choice([1, 2, 3])
        a, b = rng.random() * 0.3, rng.random() * 0.3 + (0.3 if track == "ops" else -0.01)
        decisions = 8 if track == "decisions" else 1
        scores = [scoring.CaseScore(max(v, 0.0), decisions, decisions, 0, 0.0, 0) for v in (a, b)]
        pairs.append(scoring.Paired(index, level, scores[0], scores[1], track))
        for side, score in zip(("champion", "challenger"), scores, strict=True):
            rows.append(
                (
                    lease["job"],
                    index,
                    side,
                    level,
                    score.loss,
                    decisions,
                    decisions,
                    0,
                    0.0,
                    0,
                    track,
                )
            )
    store._db.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    weights = {t: p.weight for t, p in plan.items()}
    for retired in ([], [1], [1, 2]):
        store._db.execute("UPDATE jobs SET retired=?", (json.dumps(retired),))
        job = store._db.execute("SELECT * FROM jobs").fetchone()
        expected = scoring.early_stop(pairs, set(retired), weights)
        assert store._should_stop(store._db, job) is expected
    store._db.execute("UPDATE jobs SET retired='[]'")
    job = store._db.execute("SELECT * FROM jobs").fetchone()
    assert store._should_stop(store._db, job) is True  # ops regresses, decisions barely helps
    # the decisions track alone favours the challenger: the composite is what stops the duel
    g_decisions, _ = scoring.composite(scoring.moments(pairs), {"decisions": 1.0})
    assert g_decisions > 0


# -- migration ---------------------------------------------------------------------

V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE windows (
  id INTEGER PRIMARY KEY, secret BLOB NOT NULL, commitment TEXT NOT NULL,
  opened_at INTEGER NOT NULL, closed_at INTEGER);
CREATE TABLE nonces (nonce TEXT PRIMARY KEY, hotkey TEXT NOT NULL, exp INTEGER NOT NULL);
CREATE TABLE submissions (
  intake INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, hotkey TEXT NOT NULL,
  repo TEXT NOT NULL, revision TEXT NOT NULL, files TEXT NOT NULL, digest TEXT NOT NULL,
  state TEXT NOT NULL, reason TEXT, created_at INTEGER NOT NULL, job_id TEXT);
CREATE INDEX submissions_hotkey ON submissions (hotkey, state);
CREATE TABLE champions (
  id INTEGER PRIMARY KEY, submission_id TEXT, hotkey TEXT, repo TEXT NOT NULL,
  revision TEXT NOT NULL, files TEXT NOT NULL, digest TEXT NOT NULL, job_id TEXT,
  g_lcb REAL, window_id INTEGER, crowned_at INTEGER NOT NULL);
CREATE TABLE champion_levels (
  champion_id INTEGER NOT NULL, level INTEGER NOT NULL, correct INTEGER NOT NULL,
  determined INTEGER NOT NULL, PRIMARY KEY (champion_id, level)) WITHOUT ROWID;
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, champion_id INTEGER NOT NULL,
  window_id INTEGER NOT NULL, seed TEXT NOT NULL, mix TEXT NOT NULL, retired TEXT NOT NULL,
  cases INTEGER NOT NULL, state TEXT NOT NULL, lease TEXT, lease_expires INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0, stopped INTEGER NOT NULL DEFAULT 0, verdict TEXT,
  evidence TEXT, reason TEXT, created_at INTEGER NOT NULL, finished_at INTEGER);
CREATE INDEX jobs_state ON jobs (state);
CREATE TABLE results (
  job_id TEXT NOT NULL, case_index INTEGER NOT NULL, side TEXT NOT NULL, level INTEGER NOT NULL,
  loss REAL NOT NULL, decisions INTEGER NOT NULL, determined INTEGER NOT NULL,
  correct INTEGER NOT NULL, under_loss REAL NOT NULL, under INTEGER NOT NULL,
  PRIMARY KEY (job_id, case_index, side)) WITHOUT ROWID;
CREATE TABLE entitlements (
  id INTEGER PRIMARY KEY, hotkey TEXT NOT NULL, champion_id INTEGER NOT NULL,
  window_id INTEGER NOT NULL, amount INTEGER NOT NULL, paid INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL);
CREATE TABLE epochs (epoch INTEGER PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE payments (
  epoch INTEGER NOT NULL, entitlement_id INTEGER NOT NULL, amount INTEGER NOT NULL,
  PRIMARY KEY (epoch, entitlement_id));
"""


def test_a_v1_database_migrates_in_place(tmp_path):
    path = tmp_path / "data" / "opentype.sqlite3"
    path.parent.mkdir()
    secret = secrets.token_bytes(32)
    db = sqlite3.connect(path)
    db.executescript(V1_SCHEMA)
    with db:
        db.execute(
            "INSERT INTO meta VALUES ('ladder', ?), ('retired', '[]'), ('crowns_paused', 'false')",
            (json.dumps({"order": [1, 2, 3], "width": 2}),),
        )
        db.execute("INSERT INTO windows VALUES (1, ?, ?, 1000, 2000)", (secret, "c" * 64))
        db.execute("INSERT INTO windows VALUES (2, ?, ?, 2000, NULL)", (secret, "d" * 64))
        db.execute(
            "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
            "VALUES ('base/m', 'r', '{}', 'd', 0)"
        )
        db.execute(
            "INSERT INTO submissions (id, hotkey, repo, revision, files, digest, state, "
            "created_at, job_id) VALUES ('s_1', 'h', 'a/b', 'r', '{}', 'dig', 'rejected', 0, 'j_1')"
        )
        db.execute(
            "INSERT INTO jobs (id, submission_id, champion_id, window_id, seed, mix, retired, "
            "cases, state, created_at, evidence) VALUES ('j_1', 's_1', 1, 1, 'seed', "
            "'{\"1\": 1.0}', '[]', 2, 'rejected', 0, '{\"cases_sha256\": \"x\"}')"
        )
        db.executemany(
            "INSERT INTO results VALUES ('j_1', ?, ?, 1, 0.5, 4, 4, 2, 0.0, 0)",
            [(i, s) for i in range(2) for s in ("champion", "challenger")],
        )
    db.close()

    store = store_of(tmp_path)
    version = store._db.execute("PRAGMA user_version").fetchone()[0]
    assert version == 3
    assert {r[0] for r in store._db.execute("SELECT track FROM results")} == {"decisions"}
    assert store.submission("s_1")["job"]["paired"] == 2
    assert store.window(2)["bank_digest"] == EMPTY_BANK.digest
    job = store.window(1)["jobs"][0]
    assert job["plan"] == {"decisions": {"cases": 2, "weight": 1.0}} and job["beacon"] is None
    assert store.status()["window"]["id"] == 2
    store_of(tmp_path)  # a second open is a no-op


def test_status_counts_only_the_latest_duel_by_primary_key(tmp_path):
    store = store_of(tmp_path)
    rows = [
        ("old-job", i, side, 1, 0.0, 1, 1, 1, 0.0, 0, "decisions")
        for i in range(50)
        for side in ("champion", "challenger")
    ]
    with store._tx() as db:
        db.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    assert store.status()["tracks"]["decisions"]["results"] == 0  # no started job: no scan
    plan = store._db.execute(
        "EXPLAIN QUERY PLAN SELECT track, count(*) FROM results WHERE job_id=? GROUP BY track",
        ("x",),
    ).fetchall()
    assert any("USING PRIMARY KEY" in str(tuple(r)) for r in plan), plan


def test_one_unreadable_render_forfeits_that_side_only(tmp_path):
    store = depict_store(tmp_path)
    sid = enqueue(store, "a")["id"]
    lease = lease_of(store)
    answer_paint(store, lease)
    store.complete(lease["job"], lease["lease"], {})
    for item in store.pending_judgments():
        # the challenger's render is never readable: it cannot hide a loss by being dropped
        loss = None if item["side"] == "challenger" else 0.0
        store.record_judgment(item["job"], item["case_index"], item["side"], loss)
    assert store.settle_judged() == [lease["job"]]
    verdict = store.submission(sid)["job"]["verdict"]
    paint_track = verdict["tracks"]["paint"]
    assert verdict["unjudged"] == 0 and paint_track["pairs"] == 12
    depicts = store._db.execute(
        "SELECT count(*) FROM judgments WHERE side='challenger'"
    ).fetchone()[0]
    assert paint_track["challenger_loss"] >= depicts  # each unreadable side scored loss 1


def test_a_judge_outage_leaves_the_sides_pending(make_client):
    calls: list[int] = []

    async def judge(brief: str, rubric: list[str], png: bytes) -> float | None:
        calls.append(1)
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    client = make_client(judge=judge, plan={"paint": TrackPlan(1.0, 12)})
    store = client.app.state.store
    with_bank(store, [BankItem.make("depict", DEPICT)])
    sid = enqueue(store, "a")["id"]
    lease = client.post("/v1/worker/lease", headers=bearer(WORKER)).json()
    answer_paint(store, lease)
    job = client.post(
        f"/v1/worker/jobs/{lease['job']}/complete",
        headers=bearer(WORKER),
        json={"lease": lease["lease"], "evidence": {}},
    ).json()["job"]
    # each pass (complete's, maybe a racing background one) stops at its first failure
    assert len(calls) <= 2
    assert job["state"] == "judging" and job["judgments_pending"] > 0
    assert store.submission(sid)["job"]["state"] == "judging"
    assert (
        store._db.execute("SELECT count(*) FROM judgments WHERE state != 'pending'").fetchone()[0]
        == 0
    )


def test_a_restart_without_the_teacher_keeps_pending_judgments(tmp_path, make_client):
    store = depict_store(tmp_path)
    sid = enqueue(store, "a")["id"]
    lease = lease_of(store)
    answer_paint(store, lease)
    assert store.complete(lease["job"], lease["lease"], {})["job"]["state"] == "judging"
    pending = len(store.pending_judgments())
    client = make_client(plan={"paint": TrackPlan(1.0, 12)})  # teacher off: judge is None
    client.portal.call(client.app.state.tick)
    again = client.app.state.store
    assert len(again.pending_judgments()) == pending  # not scored unjudged
    assert again.submission(sid)["job"]["state"] == "judging"


def test_a_judging_job_settles_past_its_deadline(tmp_path):
    clock = Clock()
    store = store_of(
        tmp_path, clock, settings=Settings(plan={"paint": TrackPlan(1.0, 12)}), judge=True
    )
    with_bank(store, [BankItem.make("depict", DEPICT)])
    sid = enqueue(store, "a")["id"]
    lease = lease_of(store)
    cases = answer_paint(store, lease)
    depicts = sum(c["body"]["task"]["mode"] == "depict" for c in cases)
    assert store.complete(lease["job"], lease["lease"], {})["job"]["state"] == "judging"
    assert store.settle_judged() == []
    clock.now += JUDGE_DEADLINE_SECONDS + 1  # judge down, or a restart without the teacher
    assert store.settle_judged() == [lease["job"]]
    verdict = store.submission(sid)["job"]["verdict"]
    assert verdict["unjudged"] == depicts and not verdict["crown"]
    assert verdict["tracks"]["paint"]["pairs"] == 12 - depicts
    assert store.pending_judgments() == []


def test_rotation_does_not_wait_for_the_judging_backlog(tmp_path, make_client, clock):
    async def judge(brief: str, rubric: list[str], png: bytes) -> float | None:
        await asyncio.Event().wait()  # a very slow gateway
        return None

    client = make_client(judge=judge, plan={"paint": TrackPlan(1.0, 12)})
    store = client.app.state.store
    with_bank(store, [BankItem.make("depict", DEPICT)])
    enqueue(store, "a")
    lease = lease_of(store)
    answer_paint(store, lease)
    store.complete(lease["job"], lease["lease"], {})
    store.set_next_bank([])
    clock.now += 24 * 3600

    async def bounded() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.app.state.tick(), 0.5)

    client.portal.call(bounded)
    assert store.window(3)["bank_digest"] == EMPTY_BANK.digest  # rotated all the same


@pytest.mark.parametrize(
    "plan",
    [{"Decisions": TrackPlan(1.0, 10)}, {"decisions": TrackPlan(0.0, 10)}],
)
def test_a_plan_that_builds_nothing_fails_startup(make_client, plan):
    with pytest.raises(ValueError):
        make_client(plan=plan)
