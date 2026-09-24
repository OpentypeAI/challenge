"""SQLite state: windows, submissions, duel jobs, results, champions, ledger and epochs."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import bank, ledger, pins, scoring
from .crypto import manifest_digest
from .generator import Case

LEASE_SECONDS = 1800  # renewed by every answers batch
MAX_ATTEMPTS = 3  # infrastructure retries of one job before the submission fails
DEFAULT_LADDER = {"order": [1, 2, 3, 4, 5, 6, 7, 8], "width": 2}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS windows (
  id INTEGER PRIMARY KEY, secret BLOB NOT NULL, commitment TEXT NOT NULL,
  opened_at INTEGER NOT NULL, closed_at INTEGER);
CREATE TABLE IF NOT EXISTS nonces (
  nonce TEXT PRIMARY KEY, hotkey TEXT NOT NULL, exp INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS submissions (
  intake INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, hotkey TEXT NOT NULL,
  repo TEXT NOT NULL, revision TEXT NOT NULL, files TEXT NOT NULL, digest TEXT NOT NULL,
  state TEXT NOT NULL, reason TEXT, created_at INTEGER NOT NULL, job_id TEXT);
CREATE INDEX IF NOT EXISTS submissions_hotkey ON submissions (hotkey, state);
CREATE TABLE IF NOT EXISTS champions (
  id INTEGER PRIMARY KEY, submission_id TEXT, hotkey TEXT, repo TEXT NOT NULL,
  revision TEXT NOT NULL, files TEXT NOT NULL, digest TEXT NOT NULL, job_id TEXT,
  g_lcb REAL, window_id INTEGER, crowned_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS champion_levels (
  champion_id INTEGER NOT NULL, level INTEGER NOT NULL, correct INTEGER NOT NULL,
  determined INTEGER NOT NULL, PRIMARY KEY (champion_id, level)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, champion_id INTEGER NOT NULL,
  window_id INTEGER NOT NULL, seed TEXT NOT NULL, mix TEXT NOT NULL, retired TEXT NOT NULL,
  cases INTEGER NOT NULL, state TEXT NOT NULL, lease TEXT, lease_expires INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0, stopped INTEGER NOT NULL DEFAULT 0, verdict TEXT,
  evidence TEXT, reason TEXT, created_at INTEGER NOT NULL, finished_at INTEGER);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state);
CREATE TABLE IF NOT EXISTS results (
  job_id TEXT NOT NULL, case_index INTEGER NOT NULL, side TEXT NOT NULL, level INTEGER NOT NULL,
  loss REAL NOT NULL, decisions INTEGER NOT NULL, determined INTEGER NOT NULL,
  correct INTEGER NOT NULL, under_loss REAL NOT NULL, under INTEGER NOT NULL,
  PRIMARY KEY (job_id, case_index, side)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS entitlements (
  id INTEGER PRIMARY KEY, hotkey TEXT NOT NULL, champion_id INTEGER NOT NULL,
  window_id INTEGER NOT NULL, amount INTEGER NOT NULL, paid INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS epochs (epoch INTEGER PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payments (
  epoch INTEGER NOT NULL, entitlement_id INTEGER NOT NULL, amount INTEGER NOT NULL,
  PRIMARY KEY (epoch, entitlement_id));
"""


class StoreError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Settings:
    duel_cases: int = 40_000
    max_pending: int = 4
    window_cap: float | None = None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=1024)
def _cached_case(seed: str, mix: str, index: int) -> Case:
    return bank.job_case(seed, json.loads(mix), index)


class Store:
    def __init__(self, state_dir: Path, settings: Settings, clock: Callable[[], float] = time.time):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "opentype.sqlite3"
        self.settings, self.clock = settings, clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)  # idempotent DDL; executescript commits on its own
        with self._tx() as db:
            if db.execute("SELECT 1 FROM champions").fetchone() is None:
                base = (pins.BASE_REPO, pins.BASE_REVISION, pins.BASE_FILES)
                db.execute(
                    "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (*base[:2], _dumps(base[2]), manifest_digest(*base), self._now()),
                )
            if db.execute("SELECT 1 FROM windows").fetchone() is None:
                self._open_window(db)
            db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('ladder', ?), ('retired', '[]'), "
                "('crowns_paused', 'false')",
                (_dumps(DEFAULT_LADDER),),
            )

    # -- plumbing ----------------------------------------------------------

    def _now(self) -> int:
        return int(self.clock())

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def healthy(self) -> bool:
        try:
            with self._tx() as db:
                db.execute("INSERT OR REPLACE INTO meta VALUES ('health', ?)", (str(self._now()),))
            return True
        except sqlite3.Error:
            return False

    def _meta(self, db: sqlite3.Connection, key: str) -> Any:
        return json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])

    def _set_meta(self, db: sqlite3.Connection, key: str, value: Any) -> None:
        db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, _dumps(value)))

    # -- windows -----------------------------------------------------------

    def _open_window(self, db: sqlite3.Connection) -> None:
        secret = secrets.token_bytes(32)
        db.execute(
            "INSERT INTO windows (secret, commitment, opened_at) VALUES (?, ?, ?)",
            (secret, bank.commitment(secret), self._now()),
        )

    def _window(self, db: sqlite3.Connection) -> sqlite3.Row:
        row: sqlite3.Row = db.execute(
            "SELECT * FROM windows WHERE closed_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row

    def rotate_window(self) -> dict[str, Any]:
        with self._tx() as db:
            old = self._window(db)
            db.execute("UPDATE windows SET closed_at=? WHERE id=?", (self._now(), old["id"]))
            self._open_window(db)
            new = self._window(db)
        return {"closed": old["id"], "opened": new["id"], "commitment": new["commitment"]}

    def windows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM windows ORDER BY id").fetchall()
        return [_window_json(row) for row in rows]

    def window(self, window_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM windows WHERE id=?", (window_id,)).fetchone()
            if row is None:
                raise StoreError(404, "unknown window")
            out = _window_json(row)
            if row["closed_at"] is not None:
                out["secret"] = row["secret"].hex()
                jobs = self._db.execute(
                    "SELECT j.id, j.mix, j.cases, j.state, j.evidence, s.digest FROM jobs j "
                    "JOIN submissions s ON s.id = j.submission_id "
                    "WHERE j.window_id=? AND j.state != 'queued' ORDER BY j.created_at, j.id",
                    (window_id,),
                ).fetchall()
                out["jobs"] = [
                    {
                        "id": j["id"],
                        "digest": j["digest"],
                        "mix": json.loads(j["mix"]),
                        "cases": j["cases"],
                        "state": j["state"],
                        "cases_sha256": _evidence(j, "cases_sha256"),
                        "cases_fetched": _evidence(j, "cases_fetched"),
                    }
                    for j in jobs
                ]
        return out

    # -- ladder and champion -----------------------------------------------

    def set_ladder(self, order: Sequence[int], width: int) -> dict[str, Any]:
        from .generator import LEVELS

        if not order or len(set(order)) != len(order) or not set(order) <= set(LEVELS):
            raise StoreError(400, f"order must list distinct levels from {sorted(LEVELS)}")
        if not 1 <= width <= len(order):
            raise StoreError(400, "width must be between 1 and the number of levels")
        with self._tx() as db:
            self._set_meta(db, "ladder", {"order": list(order), "width": width})
            retired = [lv for lv in self._meta(db, "retired") if lv in order]
            self._set_meta(db, "retired", retired)
        return {"order": list(order), "width": width, "retired": retired}

    def set_crowns_paused(self, paused: bool) -> None:
        with self._tx() as db:
            self._set_meta(db, "crowns_paused", paused)
            if not paused:
                self._finalize(db)

    def _champion(self, db: sqlite3.Connection) -> sqlite3.Row:
        row: sqlite3.Row = db.execute("SELECT * FROM champions ORDER BY id DESC LIMIT 1").fetchone()
        return row

    def _champion_levels(self, db: sqlite3.Connection, champion_id: int) -> dict[int, tuple]:
        rows = db.execute(
            "SELECT level, correct, determined FROM champion_levels WHERE champion_id=?",
            (champion_id,),
        ).fetchall()
        return {r["level"]: (r["correct"], r["determined"]) for r in rows}

    def _ladder(self, db: sqlite3.Connection) -> tuple[list[int], list[int], list[int]]:
        ladder = self._meta(db, "ladder")
        return scoring.ladder_state(
            ladder["order"], ladder["width"], set(self._meta(db, "retired"))
        )

    def _mix(self, db: sqlite3.Connection, champion_id: int) -> tuple[dict[str, float], list[int]]:
        active, retired, _ = self._ladder(db)
        stats = self._champion_levels(db, champion_id)
        errors = {
            level: 1.0 - correct / determined
            for level, (correct, determined) in stats.items()
            if determined
        }
        mix = scoring.duel_mix(active, retired, errors)
        return {str(level): share for level, share in sorted(mix.items())}, retired

    # -- intake --------------------------------------------------------------

    def submit(
        self,
        hotkey: str,
        repo: str,
        revision: str,
        files: Mapping[str, str],
        digest: str,
        nonce: str,
        exp: int,
    ) -> dict[str, Any]:
        with self._tx() as db:
            # An expired nonce can never be replayed (exp is checked first), so prune it.
            db.execute("DELETE FROM nonces WHERE exp < ?", (self._now(),))
            if db.execute("SELECT 1 FROM nonces WHERE nonce=?", (nonce,)).fetchone():
                raise StoreError(409, "nonce already used")
            if db.execute(
                "SELECT 1 FROM submissions WHERE hotkey=? AND state='queued'", (hotkey,)
            ).fetchone():
                raise StoreError(409, "this hotkey already has an open submission")
            pending = db.execute(
                "SELECT count(*) FROM submissions WHERE state='queued'"
            ).fetchone()[0]
            if pending >= self.settings.max_pending:
                raise StoreError(429, "the duel queue is full, retry later")
            champion = self._champion(db)
            if json.loads(champion["files"]) == dict(files):
                raise StoreError(409, "the manifest is a clone of the champion")
            db.execute("INSERT INTO nonces VALUES (?, ?, ?)", (nonce, hotkey, exp))
            submission_id = "s_" + secrets.token_hex(8)
            db.execute(
                "INSERT INTO submissions (id, hotkey, repo, revision, files, digest, state, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
                (submission_id, hotkey, repo, revision, _dumps(dict(files)), digest, self._now()),
            )
            self._new_job(db, submission_id)
            return self._submission(db, submission_id)

    def _new_job(self, db: sqlite3.Connection, submission_id: str) -> str:
        job_id = "j_" + secrets.token_hex(8)
        db.execute(
            "INSERT INTO jobs (id, submission_id, champion_id, window_id, seed, mix, retired, "
            "cases, state, created_at) VALUES (?, ?, 0, 0, '', '{}', '[]', ?, 'queued', ?)",
            (job_id, submission_id, self.settings.duel_cases, self._now()),
        )
        self._target(db, job_id)
        db.execute("UPDATE submissions SET job_id=? WHERE id=?", (job_id, submission_id))
        return job_id

    def _target(self, db: sqlite3.Connection, job_id: str) -> None:
        """Point a job at the current champion and window (fresh seed and level mix)."""
        job = db.execute(
            "SELECT j.id, s.digest FROM jobs j JOIN submissions s ON s.id=j.submission_id "
            "WHERE j.id=?",
            (job_id,),
        ).fetchone()
        champion, window = self._champion(db), self._window(db)
        mix, retired = self._mix(db, champion["id"])
        db.execute(
            "UPDATE jobs SET champion_id=?, window_id=?, seed=?, mix=?, retired=? WHERE id=?",
            (
                champion["id"],
                window["id"],
                bank.job_seed(window["secret"], job_id, job["digest"]),
                _dumps(mix),
                _dumps(retired),
                job_id,
            ),
        )

    def _submission(self, db: sqlite3.Connection, submission_id: str) -> dict[str, Any]:
        row = db.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
        if row is None:
            raise StoreError(404, "unknown submission")
        out = {
            "id": row["id"],
            "hotkey": row["hotkey"],
            "repo": row["repo"],
            "revision": row["revision"],
            "digest": row["digest"],
            "state": row["state"],
            "reason": row["reason"],
            "intake": row["intake"],
            "created_at": _iso(row["created_at"]),
        }
        job = db.execute("SELECT * FROM jobs WHERE id=?", (row["job_id"],)).fetchone()
        if job is not None:
            done = self._paired_count(db, job["id"])
            out["job"] = {
                "id": job["id"],
                "state": job["state"],
                "champion": job["champion_id"],
                "window": job["window_id"],
                "cases": job["cases"],
                "paired": done,
                "attempts": job["attempts"],
                "verdict": json.loads(job["verdict"]) if job["verdict"] else None,
                "evidence": json.loads(job["evidence"]) if job["evidence"] else None,
                "reason": job["reason"],
            }
        return out

    def submission(self, submission_id: str) -> dict[str, Any]:
        with self._lock:
            return self._submission(self._db, submission_id)

    # -- worker ---------------------------------------------------------------

    def lease(self) -> dict[str, Any] | None:
        now = self._now()
        with self._tx() as db:
            expired = db.execute(
                "SELECT id FROM jobs WHERE state='leased' AND lease_expires < ?", (now,)
            ).fetchall()
            for job in expired:
                self._release(db, job["id"], "lease expired")
            if expired:
                self._finalize(db)
            job = db.execute(
                "SELECT j.* FROM jobs j JOIN submissions s ON s.id=j.submission_id "
                "WHERE j.state='queued' ORDER BY s.intake LIMIT 1"
            ).fetchone()
            if job is None:
                return None
            champion = self._champion(db)
            self._target(db, job["id"])  # current champion, window and ladder mix
            lease = secrets.token_hex(16)
            db.execute(
                "UPDATE jobs SET state='leased', lease=?, lease_expires=? WHERE id=?",
                (lease, now + LEASE_SECONDS, job["id"]),
            )
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
            submission = db.execute(
                "SELECT * FROM submissions WHERE id=?", (job["submission_id"],)
            ).fetchone()
            return {
                "job": job["id"],
                "lease": lease,
                "lease_expires": _iso(now + LEASE_SECONDS),
                "cases": job["cases"],
                "champion": _manifest(champion),
                "challenger": _manifest(submission),
                "base": {"repo": pins.BASE_REPO, "revision": pins.BASE_REVISION},
            }

    def _release(self, db: sqlite3.Connection, job_id: str, reason: str) -> None:
        """Give a job back to the queue after an infrastructure failure, or fail it."""
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        db.execute("DELETE FROM results WHERE job_id=?", (job_id,))
        attempts = job["attempts"] + 1
        if attempts >= MAX_ATTEMPTS:
            self._terminal(db, job_id, "failed", f"{reason} ({attempts} attempts)")
        else:
            db.execute(
                "UPDATE jobs SET state='queued', lease=NULL, lease_expires=NULL, attempts=?, "
                "stopped=0, reason=? WHERE id=?",
                (attempts, reason, job_id),
            )

    def _terminal(self, db: sqlite3.Connection, job_id: str, state: str, reason: str) -> None:
        db.execute(
            "UPDATE jobs SET state=?, reason=?, lease=NULL, finished_at=? WHERE id=?",
            (state, reason, self._now(), job_id),
        )
        if state != "superseded":
            db.execute(
                "UPDATE submissions SET state=?, reason=? WHERE job_id=?", (state, reason, job_id)
            )

    def _leased(self, db: sqlite3.Connection, job_id: str, lease: str) -> sqlite3.Row:
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise StoreError(404, "unknown job")
        if job["state"] != "leased" or job["lease"] != lease:
            raise StoreError(409, "the job is not leased under this lease")
        return job

    def heartbeat(self, job_id: str, lease: str) -> dict[str, Any]:
        """Extend a lease while the worker downloads weights and starts the servers."""
        with self._tx() as db:
            self._leased(db, job_id, lease)
            expires = self._now() + LEASE_SECONDS
            db.execute("UPDATE jobs SET lease_expires=? WHERE id=?", (expires, job_id))
            stale = (
                db.execute("SELECT champion_id FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
                != self._champion(db)["id"]
            )
        return {"lease_expires": _iso(expires), "stale": stale}

    def cases(self, job_id: str, lease: str, offset: int, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            job = self._leased(self._db, job_id, lease)
        end = min(offset + limit, job["cases"])
        return [
            {"index": i, "body": _cached_case(job["seed"], job["mix"], i).body}
            for i in range(offset, end)
        ]

    def record_answers(
        self, job_id: str, lease: str, items: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        with self._lock:
            job = self._leased(self._db, job_id, lease)
        rows = []
        for item in items:
            index = item["case_index"]
            if not 0 <= index < job["cases"]:
                raise StoreError(400, f"case_index {index} is outside the job")
            case = _cached_case(job["seed"], job["mix"], index)
            if item.get("error") is not None:
                score = scoring.score_case(case.gold, None, None)
            else:
                score = scoring.score_case(case.gold, item.get("answers"), item.get("reads"))
            rows.append(
                (
                    job_id,
                    index,
                    item["side"],
                    case.level,
                    score.loss,
                    score.decisions,
                    score.determined,
                    score.correct,
                    score.under_loss,
                    score.under,
                )
            )
        with self._tx() as db:
            job = self._leased(db, job_id, lease)
            db.executemany(
                "INSERT OR IGNORE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            db.execute(
                "UPDATE jobs SET lease_expires=? WHERE id=?",
                (self._now() + LEASE_SECONDS, job_id),
            )
            stopped = bool(job["stopped"]) or self._should_stop(db, job_id, job["retired"])
            if stopped and not job["stopped"]:
                db.execute("UPDATE jobs SET stopped=1 WHERE id=?", (job_id,))
            stale = job["champion_id"] != self._champion(db)["id"]
            paired = self._paired_count(db, job_id)
        return {
            "accepted": len(rows),
            "paired": paired,
            "continue": not (stopped or stale or paired >= job["cases"]),
            "early_stop": stopped,
            "stale": stale,
        }

    def _paired_count(self, db: sqlite3.Connection, job_id: str) -> int:
        return int(
            db.execute(
                "SELECT count(*) FROM results a JOIN results b ON b.job_id=a.job_id "
                "AND b.case_index=a.case_index AND b.side='challenger' "
                "WHERE a.job_id=? AND a.side='champion'",
                (job_id,),
            ).fetchone()[0]
        )

    def _should_stop(self, db: sqlite3.Connection, job_id: str, retired: str) -> bool:
        """Early stop on the active levels; retired is the job's JSON list of guard levels."""
        row = db.execute(
            "SELECT count(*), coalesce(sum(a.decisions), 0), coalesce(sum(a.loss), 0), "
            "coalesce(sum(b.loss), 0), coalesce(sum(a.loss*a.loss), 0), "
            "coalesce(sum(b.loss*b.loss), 0), coalesce(sum(a.loss*b.loss), 0) "
            "FROM results a JOIN results b ON b.job_id=a.job_id AND b.case_index=a.case_index "
            "AND b.side='challenger' WHERE a.job_id=? AND a.side='champion' "
            "AND a.level NOT IN (SELECT value FROM json_each(?))",
            (job_id, retired),
        ).fetchone()
        n, decisions, sa, sb, saa, sbb, sab = row
        if decisions < scoring.EARLY_STOP_DECISIONS:
            return False
        g, se = scoring.log_ratio_moments(n, sa, sb, saa, sbb, sab)
        return g + scoring.EARLY_STOP_SE * se < 0

    def _pairs(self, db: sqlite3.Connection, job_id: str) -> list[scoring.Paired]:
        rows = db.execute(
            "SELECT a.case_index, a.level, a.loss, a.decisions, a.determined, a.correct, "
            "a.under_loss, a.under, b.loss, b.decisions, b.determined, b.correct, "
            "b.under_loss, b.under FROM results a JOIN results b ON b.job_id=a.job_id "
            "AND b.case_index=a.case_index AND b.side='challenger' "
            "WHERE a.job_id=? AND a.side='champion' ORDER BY a.case_index",
            (job_id,),
        ).fetchall()
        return [
            scoring.Paired(r[0], r[1], scoring.CaseScore(*r[2:8]), scoring.CaseScore(*r[8:14]))
            for r in rows
        ]

    def complete(self, job_id: str, lease: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        with self._tx() as db:
            job = self._leased(db, job_id, lease)
            stale = job["champion_id"] != self._champion(db)["id"]
            paired = self._paired_count(db, job_id)
            if not (job["stopped"] or stale) and paired < job["cases"]:
                raise StoreError(409, f"{paired} of {job['cases']} cases answered by both sides")
            pairs = self._pairs(db, job_id)
            result = scoring.verdict(pairs, set(json.loads(job["retired"])), bool(job["stopped"]))
            db.execute(
                "UPDATE jobs SET state='scored', lease=NULL, verdict=?, evidence=?, "
                "finished_at=? WHERE id=?",
                (_dumps(result), _dumps(dict(evidence)), self._now(), job_id),
            )
            self._add_level_stats(db, job["champion_id"], pairs, "champion")
            self._retire_levels(db)
            self._finalize(db)
            return self._submission(db, job["submission_id"])

    def fail(
        self, job_id: str, lease: str, reason: str, retry: bool, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self._tx() as db:
            job = self._leased(db, job_id, lease)
            db.execute("UPDATE jobs SET evidence=? WHERE id=?", (_dumps(dict(evidence)), job_id))
            if retry:
                self._release(db, job_id, reason)
            else:
                self._terminal(db, job_id, "rejected", reason)
            self._finalize(db)
            return self._submission(db, job["submission_id"])

    def requeue(self, job_id: str) -> dict[str, Any]:
        with self._tx() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise StoreError(404, "unknown job")
            if job["state"] == "crowned":
                raise StoreError(409, "a crowned job cannot be requeued")
            db.execute("DELETE FROM results WHERE job_id=?", (job_id,))
            self._terminal(db, job_id, "superseded", "requeued by the operator")
            db.execute(
                "UPDATE submissions SET state='queued', reason=NULL WHERE id=?",
                (job["submission_id"],),
            )
            self._new_job(db, job["submission_id"])
            return self._submission(db, job["submission_id"])

    # -- crowns ---------------------------------------------------------------

    def _add_level_stats(
        self, db: sqlite3.Connection, champion_id: int, pairs: Sequence[scoring.Paired], side: str
    ) -> None:
        totals: dict[int, list[int]] = {}
        for pair in pairs:
            score = getattr(pair, side)
            entry = totals.setdefault(pair.level, [0, 0])
            entry[0] += score.correct
            entry[1] += score.determined
        for level, (correct, determined) in totals.items():
            db.execute(
                "INSERT INTO champion_levels VALUES (?, ?, ?, ?) ON CONFLICT DO UPDATE SET "
                "correct=correct+excluded.correct, determined=determined+excluded.determined",
                (champion_id, level, correct, determined),
            )

    def _retire_levels(self, db: sqlite3.Connection) -> None:
        """Active levels the champion has mastered (accuracy LCB99 >= 99.9 %) retire."""
        active, _, _ = self._ladder(db)
        stats = self._champion_levels(db, self._champion(db)["id"])
        retired = self._meta(db, "retired")
        for level in active:
            correct, determined = stats.get(level, (0, 0))
            if scoring.wilson_lower(correct, determined) >= scoring.RETIRE_ACCURACY:
                retired.append(level)
        self._set_meta(db, "retired", retired)

    def _finalize(self, db: sqlite3.Connection) -> None:
        """Settle scored jobs in intake order: stale ones duel the current champion again,
        losers are rejected, and a winner is crowned once every earlier intake against the
        same champion is settled."""
        paused = self._meta(db, "crowns_paused")
        progress = True
        while progress:
            progress = False
            champion = self._champion(db)
            scored = db.execute(
                "SELECT j.*, s.intake, s.files FROM jobs j JOIN submissions s "
                "ON s.id=j.submission_id WHERE j.state='scored' ORDER BY s.intake"
            ).fetchall()
            for job in scored:
                if job["champion_id"] != champion["id"]:
                    self._terminal(db, job["id"], "superseded", "the champion changed")
                    if json.loads(job["files"]) == json.loads(champion["files"]):
                        db.execute(
                            "UPDATE submissions SET state='rejected', reason=? WHERE id=?",
                            ("the manifest is a clone of the champion", job["submission_id"]),
                        )
                    else:
                        self._new_job(db, job["submission_id"])
                    progress = True
                    break
                result = json.loads(job["verdict"])
                if not result["crown"]:
                    reason = "early stop" if result["early_stop"] else "no certified gain"
                    self._terminal(db, job["id"], "rejected", reason)
                    progress = True
                    break
                blocked = db.execute(
                    "SELECT 1 FROM jobs j JOIN submissions s ON s.id=j.submission_id "
                    "WHERE j.champion_id=? AND s.intake < ? "
                    "AND j.state IN ('queued', 'leased', 'scored')",
                    (champion["id"], job["intake"]),
                ).fetchone()
                if paused or blocked:
                    continue
                self._crown(db, job)
                progress = True
                break

    def _crown(self, db: sqlite3.Connection, job: sqlite3.Row) -> None:
        """ponytail: no anchor telemetry (typed-decisions test, PhishNChips, Laya probes,
        jev-harness-lab) and so no automatic RT-7 pause; the operator pauses crowns by hand
        with PUT /v1/admin/crowns. Add an anchors job on the worker when anchors are wired."""
        submission = db.execute(
            "SELECT * FROM submissions WHERE id=?", (job["submission_id"],)
        ).fetchone()
        result = json.loads(job["verdict"])
        window = self._window(db)
        cursor = db.execute(
            "INSERT INTO champions (submission_id, hotkey, repo, revision, files, digest, "
            "job_id, g_lcb, window_id, crowned_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission["id"],
                submission["hotkey"],
                submission["repo"],
                submission["revision"],
                submission["files"],
                submission["digest"],
                job["id"],
                result["g_lcb"],
                window["id"],
                self._now(),
            ),
        )
        champion_id = cursor.lastrowid
        assert champion_id is not None
        self._add_level_stats(db, champion_id, self._pairs(db, job["id"]), "challenger")
        window_total = db.execute(
            "SELECT coalesce(sum(amount), 0) FROM entitlements WHERE window_id=?",
            (window["id"],),
        ).fetchone()[0]
        amount = ledger.entitlement_units(result["g_lcb"], window_total, self.settings.window_cap)
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (submission["hotkey"], champion_id, window["id"], amount, self._now()),
        )
        self._terminal(db, job["id"], "crowned", f"crowned as champion {champion_id}")
        self._retire_levels(db)

    # -- weights ---------------------------------------------------------------

    def weights(self, epoch: int, slug: str) -> str:
        """The persisted body for an epoch; the first call pays the ledger FIFO.

        ponytail: publication is checked once, when the worker downloads the challenger
        anonymously at duel time; it is not re-checked here before paying. Add a HEAD per
        crowned file when a withdrawn repo must stop its payments."""
        with self._tx() as db:
            row = db.execute("SELECT body FROM epochs WHERE epoch=?", (epoch,)).fetchone()
            if row is not None:
                return str(row["body"])
            owed = [
                ledger.Entitlement(r["id"], r["hotkey"], r["amount"], r["paid"])
                for r in db.execute(
                    "SELECT * FROM entitlements WHERE paid < amount ORDER BY id"
                ).fetchall()
            ]
            payments = ledger.pay(owed)
            by_id = {
                r["id"]: r
                for r in db.execute("SELECT id, hotkey, champion_id FROM entitlements").fetchall()
            }
            weights: dict[str, float] = {}
            paid_units = 0
            for entitlement_id, amount in payments:
                db.execute(
                    "UPDATE entitlements SET paid = paid + ? WHERE id=?", (amount, entitlement_id)
                )
                db.execute("INSERT INTO payments VALUES (?, ?, ?)", (epoch, entitlement_id, amount))
                hotkey = by_id[entitlement_id]["hotkey"]
                weights[hotkey] = weights.get(hotkey, 0) + amount
                paid_units += amount
            body = {
                "challenge_slug": slug,
                "epoch": epoch,
                "weights": {k: v / ledger.UNITS for k, v in sorted(weights.items())},
                "full_share_mass": 1.0,
                "metadata": {
                    "payments": [
                        {
                            "entitlement": eid,
                            "champion": by_id[eid]["champion_id"],
                            "hotkey": by_id[eid]["hotkey"],
                            "mass": amount / ledger.UNITS,
                        }
                        for eid, amount in payments
                    ],
                    "burned": (ledger.UNITS - paid_units) / ledger.UNITS,
                },
                "computed_at": _iso(self._now()),
            }
            text = json.dumps(body, sort_keys=True, separators=(",", ":"))
            db.execute("INSERT INTO epochs VALUES (?, ?)", (epoch, text))
            return text

    # -- public views ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            db = self._db
            champion = self._champion(db)
            active, retired, pending = self._ladder(db)
            stats = self._champion_levels(db, champion["id"])
            window = self._window(db)
            queue = db.execute(
                "SELECT s.id, s.hotkey, s.intake, j.state, j.id AS job FROM submissions s "
                "JOIN jobs j ON j.id=s.job_id WHERE s.state='queued' ORDER BY s.intake"
            ).fetchall()
            levels = []
            for level in [*active, *retired, *pending]:
                correct, determined = stats.get(level, (0, 0))
                levels.append(
                    {
                        "level": level,
                        "state": "active"
                        if level in active
                        else "retired"
                        if level in retired
                        else "pending",
                        "champion_accuracy": correct / determined if determined else None,
                        "champion_accuracy_lcb99": scoring.wilson_lower(correct, determined),
                        "determined": determined,
                    }
                )
            mix, _ = self._mix(db, champion["id"])
            return {
                "champion": _champion_json(champion),
                "queue": [dict(row) for row in queue],
                "levels": levels,
                "next_duel_mix": mix,
                "window": {
                    "id": window["id"],
                    "commitment": window["commitment"],
                    "opened_at": _iso(window["opened_at"]),
                },
                "crowns_paused": self._meta(db, "crowns_paused"),
                "constants": {
                    "g_min": scoring.G_MIN,
                    "z": scoring.Z99,
                    "guard_max": scoring.GUARD_MAX,
                    "duel_cases": self.settings.duel_cases,
                    "early_stop_decisions": scoring.EARLY_STOP_DECISIONS,
                    "early_stop_se": scoring.EARLY_STOP_SE,
                    "retire_accuracy": scoring.RETIRE_ACCURACY,
                    "max_pending": self.settings.max_pending,
                    "window_entitlement_cap": self.settings.window_cap,
                    "base": {"repo": pins.BASE_REPO, "revision": pins.BASE_REVISION},
                },
            }

    def leaderboard(self) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, e.amount, e.paid FROM champions c "
                "LEFT JOIN entitlements e ON e.champion_id=c.id ORDER BY c.id"
            ).fetchall()
        crowns = []
        totals: dict[str, dict[str, float]] = {}
        for row in rows:
            entry = _champion_json(row)
            if row["amount"] is not None:
                entry.update(
                    entitlement=row["amount"] / ledger.UNITS,
                    paid=row["paid"] / ledger.UNITS,
                    outstanding=(row["amount"] - row["paid"]) / ledger.UNITS,
                )
                total = totals.setdefault(row["hotkey"], {"entitlement": 0.0, "paid": 0.0})
                total["entitlement"] += row["amount"] / ledger.UNITS
                total["paid"] += row["paid"] / ledger.UNITS
            crowns.append(entry)
        return {"crowns": crowns, "hotkeys": totals}


def _evidence(row: sqlite3.Row, key: str) -> Any:
    return json.loads(row["evidence"]).get(key) if row["evidence"] else None


def _manifest(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "repo": row["repo"],
        "revision": row["revision"],
        "files": json.loads(row["files"]),
        "digest": row["digest"],
    }


def _champion_json(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "hotkey": row["hotkey"],
        "repo": row["repo"],
        "revision": row["revision"],
        "digest": row["digest"],
        "g_lcb": row["g_lcb"],
        "crowned_at": _iso(row["crowned_at"]),
    }


def _window_json(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "commitment": row["commitment"],
        "opened_at": _iso(row["opened_at"]),
        "closed_at": _iso(row["closed_at"]) if row["closed_at"] is not None else None,
        "revealed": row["closed_at"] is not None,
    }
