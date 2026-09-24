"""v2 store and container: window banks and rotation, byte-bounded case pages, the judgments
lifecycle, the v1 -> v2 migration, the drand beacon and early stop on the composite
(docs/tracks.md §8, §11)."""

from __future__ import annotations

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
from opentype_challenge.store import PAGE_BYTES, Settings, Store, judge_order
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
    for item in pending:
        blank = item["png"] == paint.blank_png()
        loss = None if item["case_index"] == unjudged else (1.0 if blank else 0.0)
        store.record_judgment(item["job"], item["case_index"], item["side"], loss)
        if item["case_index"] == unjudged:
            break  # the other side of an unjudged case is never judged
    assert store.pending_judgments() and store.settle_judged() == []  # still pending
    for item in store.pending_judgments():
        store.record_judgment(item["job"], item["case_index"], item["side"], 0.0)
    assert store.settle_judged() == [lease["job"]]

    result = store.submission(sid)
    verdict = result["job"]["verdict"]
    assert verdict["unjudged"] == 1
    assert verdict["tracks"]["paint"]["pairs"] == 11 == result["job"]["paired"]
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
    assert version == 2
    assert {r[0] for r in store._db.execute("SELECT track FROM results")} == {"decisions"}
    assert store.submission("s_1")["job"]["paired"] == 2
    assert store.window(2)["bank_digest"] == EMPTY_BANK.digest
    job = store.window(1)["jobs"][0]
    assert job["plan"] == {"decisions": {"cases": 2, "weight": 1.0}} and job["beacon"] is None
    assert store.status()["window"]["id"] == 2
    store_of(tmp_path)  # a second open is a no-op
