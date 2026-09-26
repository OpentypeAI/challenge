"""SQLite state: windows and their banks, submissions, duel jobs, results, judgments,
champions, ledger and epochs (docs/tracks.md §8, §11)."""

from __future__ import annotations

import json
import random
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import bank, harness, ledger, paint, pins, runtime, scoring, tracks
from .crypto import manifest_digest
from .generator import Case
from .tracks import TrackPlan
from .worker import NOT_LEASED

LEASE_SECONDS = 1800  # renewed by every answers batch
MAX_ATTEMPTS = 3  # infrastructure retries of one job before the submission fails
DEFAULT_LADDER = {"order": [1, 2, 3, 4, 5, 6, 7, 8], "width": 2}
DEFAULT_DUEL_CASES = 40_000
PAGE_BYTES = 6 * 1024 * 1024  # case and bank pages stay under this much JSON
CASE_CACHE_BYTES = 128 * 1024 * 1024
BANK_CACHE = 4  # parsed banks kept in memory (one per window)
SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS windows (
  id INTEGER PRIMARY KEY, secret BLOB NOT NULL, commitment TEXT NOT NULL,
  opened_at INTEGER NOT NULL, closed_at INTEGER, bank TEXT NOT NULL DEFAULT '[]',
  bank_digest TEXT NOT NULL DEFAULT '{EMPTY_DIGEST}');
CREATE TABLE IF NOT EXISTS nonces (
  nonce TEXT PRIMARY KEY, hotkey TEXT NOT NULL, exp INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS submissions (
  intake INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, hotkey TEXT NOT NULL,
  repo TEXT NOT NULL, revision TEXT NOT NULL, files TEXT NOT NULL, digest TEXT NOT NULL,
  state TEXT NOT NULL, reason TEXT, created_at INTEGER NOT NULL, job_id TEXT,
  lane TEXT NOT NULL DEFAULT 'quality', options TEXT, target INTEGER, profile TEXT,
  kernel TEXT);
CREATE INDEX IF NOT EXISTS submissions_hotkey ON submissions (hotkey, state);
CREATE INDEX IF NOT EXISTS submissions_lane ON submissions (lane, state);
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
  evidence TEXT, reason TEXT, created_at INTEGER NOT NULL, finished_at INTEGER,
  plan TEXT NOT NULL DEFAULT '', beacon TEXT, beacon_fetched INTEGER NOT NULL DEFAULT 0,
  judge INTEGER NOT NULL DEFAULT 0, lane TEXT NOT NULL DEFAULT 'quality',
  incumbent_id INTEGER NOT NULL DEFAULT 0, calibration TEXT);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state);
CREATE TABLE IF NOT EXISTS results (
  job_id TEXT NOT NULL, case_index INTEGER NOT NULL, side TEXT NOT NULL, level INTEGER NOT NULL,
  loss REAL NOT NULL, decisions INTEGER NOT NULL, determined INTEGER NOT NULL,
  correct INTEGER NOT NULL, under_loss REAL NOT NULL, under INTEGER NOT NULL,
  track TEXT NOT NULL DEFAULT 'decisions',
  PRIMARY KEY (job_id, case_index, side)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS judgments (
  job_id TEXT NOT NULL, case_index INTEGER NOT NULL, side TEXT NOT NULL,
  track TEXT NOT NULL, level INTEGER NOT NULL, seed INTEGER NOT NULL, png BLOB NOT NULL,
  brief TEXT NOT NULL, rubric TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', loss REAL,
  PRIMARY KEY (job_id, case_index, side));
CREATE TABLE IF NOT EXISTS entitlements (
  id INTEGER PRIMARY KEY, hotkey TEXT NOT NULL, champion_id INTEGER NOT NULL,
  window_id INTEGER NOT NULL, amount INTEGER NOT NULL, paid INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL, lane TEXT NOT NULL DEFAULT 'quality');
CREATE TABLE IF NOT EXISTS runtime_incumbents (
  id INTEGER PRIMARY KEY, submission_id TEXT NOT NULL, hotkey TEXT NOT NULL,
  model_champion_id INTEGER NOT NULL, options TEXT NOT NULL, options_digest TEXT NOT NULL,
  profile_digest TEXT NOT NULL, calibration TEXT NOT NULL, job_id TEXT NOT NULL,
  g_lcb REAL NOT NULL, total_gain REAL NOT NULL, credited INTEGER NOT NULL,
  crowned_at INTEGER NOT NULL, kernel TEXT);
CREATE TABLE IF NOT EXISTS runtime_tasks (
  job_id TEXT NOT NULL, block INTEGER NOT NULL, side TEXT NOT NULL, cell TEXT NOT NULL,
  case_index INTEGER NOT NULL, ms REAL NOT NULL, ok INTEGER NOT NULL, error INTEGER NOT NULL,
  vectors TEXT,
  PRIMARY KEY (job_id, block, side, cell, case_index)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS epochs (epoch INTEGER PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payments (
  epoch INTEGER NOT NULL, entitlement_id INTEGER NOT NULL, amount INTEGER NOT NULL,
  PRIMARY KEY (epoch, entitlement_id));
""".replace("{EMPTY_DIGEST}", bank.EMPTY_BANK.digest)

# In-place migrations, one transaction: every column v1 lacks for v2, then v2 lacks for v3
# (the lanes; existing rows become quality). Each column is added only when it is missing, so
# the migration reads the schema, not user_version: a v3 file that an older binary re-stamped
# as v2 migrates to v3 without touching its data.
MIGRATIONS = (
    ("windows", "bank", "TEXT NOT NULL DEFAULT '[]'"),
    ("windows", "bank_digest", f"TEXT NOT NULL DEFAULT '{bank.EMPTY_BANK.digest}'"),
    ("jobs", "plan", "TEXT NOT NULL DEFAULT ''"),
    ("jobs", "beacon", "TEXT"),
    ("jobs", "beacon_fetched", "INTEGER NOT NULL DEFAULT 0"),
    ("jobs", "judge", "INTEGER NOT NULL DEFAULT 0"),
    ("results", "track", "TEXT NOT NULL DEFAULT 'decisions'"),
    ("submissions", "lane", "TEXT NOT NULL DEFAULT 'quality'"),
    ("submissions", "options", "TEXT"),
    ("submissions", "target", "INTEGER"),
    ("submissions", "profile", "TEXT"),
    ("jobs", "lane", "TEXT NOT NULL DEFAULT 'quality'"),
    ("jobs", "incumbent_id", "INTEGER NOT NULL DEFAULT 0"),
    ("jobs", "calibration", "TEXT"),
    ("entitlements", "lane", "TEXT NOT NULL DEFAULT 'quality'"),
    # the kernel slot and the timed answer vectors (still v3: nullable additions only)
    ("submissions", "kernel", "TEXT"),
    ("runtime_incumbents", "kernel", "TEXT"),
    ("runtime_tasks", "vectors", "TEXT"),
)
CREATED_WHOLE = ("runtime_incumbents", "runtime_tasks")  # v3 tables a v2 file lacks
RESULT_COLUMNS = (
    "job_id, case_index, side, level, loss, decisions, determined, correct, under_loss, under, "
    "track"
)
WINDOW_COLUMNS = "id, secret, commitment, opened_at, closed_at, bank_digest"


RUNTIME_WORKER_SECONDS = 120  # a runtime worker polls every 30 s when idle
EXPIRED = "the quality champion or the calibrated profile changed: sign a new runtime submission"
EXPIRED_FORMAT = "the champion changed weight format (NVFP4): resubmit in the champion's format"
UNJUDGED_MAX = 0.05  # share of judged cases a crowned duel may drop as unreadable
JUDGE_DEADLINE_SECONDS = 6 * 3600  # a judging job settles without its missing sides after this


class StoreError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Settings:
    duel_cases: int = DEFAULT_DUEL_CASES
    max_pending: int = 4
    window_cap: float | None = None
    plan: Mapping[str, TrackPlan] | None = None
    # entitlement cap (epoch-masses) of a crown whose duel ran on the empty bank: its cases
    # are all public templates a miner can train on, so by default it pays nothing
    empty_bank_cap: float = 0.0
    # fidelity read cases a runtime candidate and the stock reference both answer
    runtime_fidelity_cases: int = 2_000
    # quality leases between two runtime leases while a runtime job waits
    runtime_every: int = 4

    def track_plan(self) -> Mapping[str, TrackPlan]:
        """The configured plan; v1's duel_cases gives a decisions-only plan (tests)."""
        if self.plan is not None:
            return self.plan
        if self.duel_cases != DEFAULT_DUEL_CASES:
            return {"decisions": TrackPlan(1.0, self.duel_cases)}
        return tracks.DEFAULT_PLAN


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def plan_to_json(plan: Mapping[str, TrackPlan]) -> dict[str, dict[str, Any]]:
    return {t: {"weight": p.weight, "cases": p.cases} for t, p in sorted(plan.items())}


def plan_from_json(value: Mapping[str, Any]) -> dict[str, TrackPlan]:
    return {t: TrackPlan(float(p["weight"]), int(p["cases"])) for t, p in sorted(value.items())}


def _job_plan(job: sqlite3.Row) -> dict[str, TrackPlan]:
    """The job's effective plan; a v1 job (no plan) is decisions-only."""
    if not job["plan"]:
        return {"decisions": TrackPlan(1.0, job["cases"])}
    return plan_from_json(json.loads(job["plan"]))


class _CaseCache:
    """LRU of built cases bounded by the bytes of their served JSON (plus private data)."""

    def __init__(self, limit: int = CASE_CACHE_BYTES):
        self.limit, self.size = limit, 0
        self._items: OrderedDict[tuple[Any, ...], tuple[Case, int]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...], build: Callable[[], Case]) -> tuple[Case, int]:
        """(case, length of its body's JSON)."""
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit
        case = build()  # outside the lock: building a case can take a while
        length = len(_dumps(case.body))
        cost = length + len(_dumps(case.private)) + 1024
        with self._lock:
            if key not in self._items:
                self._items[key] = (case, length)
                self.size += cost
                while self.size > self.limit and len(self._items) > 1:
                    _, (old, old_length) = self._items.popitem(last=False)
                    self.size -= old_length + len(_dumps(old.private)) + 1024
        return case, length


def _final_png(case: Case, transcript: Sequence[str]) -> bytes:
    """The container's own render of a paint episode replayed from the raw outputs."""
    state, _ = harness.replay(tracks.ENVS[case.track], case.body, transcript)
    return paint.render_png([command for draw in state["draws"] for command in draw])


def judge_order(seed: int) -> tuple[str, str]:
    """The side judged first, drawn from the case seed (§6)."""
    if random.Random(f"judge|{seed}").random() < 0.5:
        return ("champion", "challenger")
    return ("challenger", "champion")


class Store:
    def __init__(
        self,
        state_dir: Path,
        settings: Settings,
        clock: Callable[[], float] = time.time,
        *,
        judge: bool = False,
        beacon: Callable[[], dict[str, Any] | None] = bank.drand_beacon,
    ):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "opentype.sqlite3"
        self.settings, self.clock = settings, clock
        self.judge, self.beacon = judge, beacon
        self.teacher_configured = False
        self.teacher_building = False
        self._lock = threading.Lock()
        self._cases = _CaseCache()
        self._banks: OrderedDict[int, bank.Bank] = OrderedDict()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        existing = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='windows'"
        ).fetchone()
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"state schema v{version} is newer than this build's")
        if existing:
            self._migrate()
        self._db.executescript(SCHEMA)  # idempotent DDL; executescript commits on its own
        self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        with self._tx() as db:
            if db.execute("SELECT 1 FROM champions").fetchone() is None:
                base = (pins.BASE_REPO, pins.BASE_REVISION, pins.BASE_FILES)
                db.execute(
                    "INSERT INTO champions (repo, revision, files, digest, crowned_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (*base[:2], _dumps(base[2]), manifest_digest(*base), self._now()),
                )
            if db.execute("SELECT 1 FROM windows").fetchone() is None:
                self._open_window(db, [])  # the first window opens with the empty bank
            db.execute(
                "INSERT OR IGNORE INTO meta VALUES ('ladder', ?), ('retired', '[]'), "
                "('crowns_paused', 'false')",
                (_dumps(DEFAULT_LADDER),),
            )

    # -- plumbing ----------------------------------------------------------

    def _migrate(self) -> None:
        """One transaction: add the missing columns; a failure leaves the file untouched."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for table, column, ddl in MIGRATIONS:
                have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}
                if not have and table in CREATED_WHOLE:
                    continue  # a v2 file lacks it: SCHEMA creates it whole below
                if column not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

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

    def _meta_opt(self, db: sqlite3.Connection, key: str) -> Any:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    # -- lanes ---------------------------------------------------------------

    def _calibration_raw(self, db: sqlite3.Connection) -> str | None:
        value = self._meta_opt(db, "runtime_calibration")
        return None if value is None else _dumps(value)

    def _calibration(self, db: sqlite3.Connection) -> runtime.Calibration | None:
        value = self._meta_opt(db, "runtime_calibration")
        return None if value is None else runtime.Calibration.from_json(value)

    def _runtime_open(self, db: sqlite3.Connection) -> bool:
        """Runtime intake needs the operator's calibration, the lane split scheduled and an
        NVFP4 quality champion: the lane measures NVFP4 weights only, so while the champion is
        a BF16 checkpoint (every champion before the NVFP4 migration) it stays closed."""
        return (
            self._calibration(db) is not None
            and self._meta_opt(db, "lanes_from_epoch") is not None
            and self._champion_nvfp4(db)
        )

    def _champion_nvfp4(self, db: sqlite3.Connection) -> bool:
        """The champion's manifest carries the pinned NVFP4 config. Only two paths make such
        a champion: the migration, which installs exactly the pinned official export, and a
        crown, whose challenger the worker verified (digests and the pinned tensor layout)
        before it served a case. The profile binds the layout again on every runtime run."""
        champion = self._champion(db)
        files = json.loads(champion["files"])
        if files.get("config.json") != pins.NVFP4_CONFIG_SHA256:
            return False
        official = (champion["repo"], champion["revision"], files) == (
            pins.NVFP4_REPO, pins.NVFP4_REVISION, pins.NVFP4_FILES
        )  # fmt: skip
        return official or champion["job_id"] is not None  # the migration, or a crown

    def _incumbent(self, db: sqlite3.Connection) -> sqlite3.Row | None:
        """The runtime incumbent certified on the current champion's weights and the current
        profile; None means the stock configuration."""
        calibration = self._calibration(db)
        if calibration is None:
            return None
        row: sqlite3.Row | None = db.execute(
            "SELECT * FROM runtime_incumbents WHERE model_champion_id=? AND profile_digest=? "
            "ORDER BY id DESC LIMIT 1",
            (self._champion(db)["id"], calibration.profile_digest),
        ).fetchone()
        return row

    def set_calibration(self, raw: Any) -> dict[str, Any] | None:
        """Publish (or, with None, withdraw) the runtime calibration. Runtime jobs duelling
        under the previous one become stale and duel again under the new one."""
        calibration = None if raw is None else runtime.Calibration.from_json(raw)
        with self._tx() as db:
            if calibration is None:
                db.execute("DELETE FROM meta WHERE key='runtime_calibration'")
            else:
                self._set_meta(db, "runtime_calibration", raw)
            self._expire_off_target(db)  # queued work signed for another profile
            self._finalize(db)
        return None if calibration is None else calibration.public()

    def migrate_nvfp4(self) -> dict[str, Any]:
        """The operator's one-way, prospective reset of the quality lane to NVFP4: the pinned
        official export (pins.NVFP4_REPO@NVFP4_REVISION) becomes the champion. It claims no
        equivalence with the BF16 champion it follows: a new row with no hotkey and no
        entitlement; every earlier champion, entitlement, payment and epoch stays as it is
        and old debts keep paying. From here on quality intake takes only the NVFP4 config
        (and the worker the pinned tensor layout), so no BF16 duel ever runs again; open BF16
        work expires, a leased job turns stale, and BF16 workers lease nothing."""
        repo, revision, files = pins.NVFP4_REPO, pins.NVFP4_REVISION, pins.NVFP4_FILES
        with self._tx() as db:
            if self._champion_nvfp4(db):
                raise StoreError(409, "the champion is already NVFP4")
            previous = self._champion(db)
            base = manifest_digest(pins.BASE_REPO, pins.BASE_REVISION, pins.BASE_FILES)
            if previous["digest"] != base:
                # converting a mined BF16 champion needs a checkpoint proven to derive from
                # it; nothing here can prove that, so it is not offered
                raise StoreError(409, "only the base champion migrates (to the official export)")
            db.execute(
                "INSERT INTO champions (repo, revision, files, digest, window_id, crowned_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    repo,
                    revision,
                    _dumps(dict(files)),
                    manifest_digest(repo, revision, files),
                    self._window(db)["id"],
                    self._now(),
                ),
            )
            self._set_meta(db, "nvfp4_migration", {"from": previous["id"], "at": self._now()})
            # queued BF16 work expires now; leased and judging jobs are stale (another
            # champion) and expire when their worker completes or fails them; scored ones
            # are superseded, and their resubmission expires (another format)
            self._expire_off_target(db)
            self._finalize(db)
            return _champion_json(self._champion(db))

    def set_lanes_from(self, epoch: int) -> int:
        """Schedule the 75/25 split from `epoch` on: once, and only past every persisted
        epoch, so no published epoch changes."""
        with self._tx() as db:
            if self._meta_opt(db, "lanes_from_epoch") is not None:
                raise StoreError(409, "the lane split is already scheduled")
            last = db.execute("SELECT max(epoch) FROM epochs").fetchone()[0]
            if last is not None and epoch <= last:
                raise StoreError(409, f"epoch {last} is already persisted; pick a later one")
            self._set_meta(db, "lanes_from_epoch", epoch)
        return epoch

    def submit_runtime(
        self,
        hotkey: str,
        target: Mapping[str, Any],
        profile_digest: str,
        options: Mapping[str, Any],
        digest: str,
        nonce: str,
        exp: int,
        kernel: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A signed vLLM option set and/or kernel for the current champion's weights on the
        pinned profile."""
        with self._tx() as db:
            if not self._runtime_open(db):
                raise StoreError(
                    503,
                    "the runtime lane is closed until the operator calibrates it on an NVFP4 "
                    "champion",
                )
            calibration = self._calibration(db)
            assert calibration is not None
            if kernel is not None and kernel["slot"] not in calibration.kernel_slots:
                raise StoreError(
                    503, f"the {kernel['slot']} kernel slot is not open under this calibration"
                )
            db.execute("DELETE FROM nonces WHERE exp < ?", (self._now(),))
            if db.execute("SELECT 1 FROM nonces WHERE nonce=?", (nonce,)).fetchone():
                raise StoreError(409, "nonce already used")
            champion = self._champion(db)
            if target != {"champion": champion["id"], "digest": champion["digest"]}:
                raise StoreError(409, "the target is not the current quality champion")
            if profile_digest != calibration.profile_digest:
                raise StoreError(409, "the profile is not the calibrated profile")
            if db.execute(
                "SELECT 1 FROM submissions WHERE hotkey=? AND state='queued' AND lane='runtime'",
                (hotkey,),
            ).fetchone():
                raise StoreError(409, "this hotkey already has an open runtime submission")
            pending = db.execute(
                "SELECT count(*) FROM submissions WHERE state='queued' AND lane='runtime'"
            ).fetchone()[0]
            if pending >= self.settings.max_pending:
                raise StoreError(429, "the runtime queue is full, retry later")
            incumbent = self._incumbent(db)
            if incumbent is not None and (
                json.loads(incumbent["options"]) == dict(options)
                and _kernel_sha(incumbent["kernel"]) == (kernel or {}).get("sha256")
            ):
                raise StoreError(409, "the candidate is the runtime incumbent")
            db.execute("INSERT INTO nonces VALUES (?, ?, ?)", (nonce, hotkey, exp))
            submission_id = "r_" + secrets.token_hex(8)
            db.execute(
                "INSERT INTO submissions (id, hotkey, repo, revision, files, digest, state, "
                "created_at, lane, options, target, profile, kernel) "
                "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, 'runtime', ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    champion["repo"],
                    champion["revision"],
                    champion["files"],
                    digest,
                    self._now(),
                    _dumps(dict(options)),
                    champion["id"],
                    profile_digest,
                    _dumps(dict(kernel)) if kernel else None,
                ),
            )
            self._new_job(db, submission_id)
            return self._submission(db, submission_id)

    def _runtime_fidelity(
        self, db: sqlite3.Connection, job_id: str, side: str
    ) -> dict[str, runtime.Fidelity]:
        rows = db.execute(
            "SELECT track, sum(loss), sum(decisions), sum(determined), sum(correct), count(*) "
            "FROM results WHERE job_id=? AND side=? GROUP BY track",
            (job_id, side),
        ).fetchall()
        return {r[0]: runtime.Fidelity(float(r[1]), r[2], r[3], r[4], r[5]) for r in rows}

    def _settle_runtime(self, db: sqlite3.Connection, job: sqlite3.Row) -> None:
        """The verdict is recomputed here from the trusted worker's measurements and the
        fidelity answers the container scored; nothing the candidate declares counts."""
        calibration = runtime.Calibration.from_json(json.loads(job["calibration"]))
        evidence = json.loads(job["evidence"]) if job["evidence"] else {}
        measured = evidence.get("runtime")
        tasks = [
            dict(t)
            for t in db.execute(
                "SELECT block, side, cell, case_index, ms, ok, error, vectors "
                "FROM runtime_tasks WHERE job_id=?",
                (job["id"],),
            ).fetchall()
        ]
        for task in tasks:
            task["vectors"] = json.loads(task["vectors"]) if task["vectors"] else None
        if isinstance(measured, Mapping):
            blocks = runtime.runs_from_tasks(calibration, measured.get("blocks"), tasks)
            measured = {"profile": measured.get("profile"), "blocks": blocks}
        result = runtime.verdict(
            calibration,
            measured,
            self._runtime_fidelity(db, job["id"], "challenger"),
            self._runtime_fidelity(db, job["id"], "champion"),
            {track: plan.cases for track, plan in _job_plan(job).items()},
            runtime.divergence(calibration.blocks, tasks),
        )
        db.execute(
            "UPDATE jobs SET state='scored', lease=NULL, verdict=?, finished_at=? WHERE id=?",
            (_dumps(result), self._now(), job["id"]),
        )
        self._finalize(db)

    def _finalize_runtime(self, db: sqlite3.Connection, paused: bool) -> None:
        """Scored runtime jobs in intake order: stale ones duel again (or expire when the
        champion changed), NO_DECISION is an infrastructure retry, the rest reject or crown."""
        progress = True
        while progress:
            progress = False
            scored = db.execute(
                "SELECT j.*, s.intake FROM jobs j JOIN submissions s ON s.id=j.submission_id "
                "WHERE j.state='scored' AND j.lane='runtime' ORDER BY s.intake"
            ).fetchall()
            for job in scored:
                if self._stale(db, job):
                    self._terminal(db, job["id"], "superseded", "the runtime target changed")
                    self._new_job(db, job["submission_id"])
                    progress = True
                    break
                result = json.loads(job["verdict"])
                if result["decision"] == "no_decision":
                    self._release(db, job["id"], f"no decision: {result['reason']}")
                    progress = True
                    break
                if not result["crown"]:
                    self._terminal(db, job["id"], "rejected", result["reason"])
                    progress = True
                    break
                blocked = db.execute(
                    "SELECT 1 FROM jobs j JOIN submissions s ON s.id=j.submission_id "
                    "WHERE j.lane='runtime' AND s.intake < ? "
                    "AND j.state IN ('queued', 'leased', 'judging', 'scored')",
                    (job["intake"],),
                ).fetchone()
                if paused or blocked:
                    continue
                self._crown_runtime(db, job)
                progress = True
                break

    def _crown_runtime(self, db: sqlite3.Connection, job: sqlite3.Row) -> None:
        """A new incumbent. Credit pays only the certified gain over stock past the best one
        already certified on this profile, so recertifying on new weights, or resubmitting an
        option set that was already paid, earns nothing."""
        submission = db.execute(
            "SELECT * FROM submissions WHERE id=?", (job["submission_id"],)
        ).fetchone()
        calibration = runtime.Calibration.from_json(json.loads(job["calibration"]))
        result = json.loads(job["verdict"])
        incumbent = self._incumbent(db)
        total = (incumbent["total_gain"] if incumbent else 0.0) + result["g_lcb"]
        best = db.execute(
            "SELECT coalesce(max(total_gain), 0) FROM runtime_incumbents WHERE profile_digest=?",
            (calibration.profile_digest,),
        ).fetchone()[0]
        amount = runtime.credit_units(calibration, max(total - best, 0.0), ledger.UNITS)
        cursor = db.execute(
            "INSERT INTO runtime_incumbents (submission_id, hotkey, model_champion_id, options, "
            "options_digest, profile_digest, calibration, job_id, g_lcb, total_gain, credited, "
            "crowned_at, kernel) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission["id"],
                submission["hotkey"],
                job["champion_id"],
                submission["options"],
                runtime.digest(
                    {
                        "options": json.loads(submission["options"]),
                        "kernel": runtime.kernel_ref(_kernel(submission["kernel"])),
                    }
                    if submission["kernel"]
                    else json.loads(submission["options"])
                ),
                calibration.profile_digest,
                calibration.version,
                job["id"],
                result["g_lcb"],
                total,
                amount,
                self._now(),
                submission["kernel"],
            ),
        )
        incumbent_id = cursor.lastrowid
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at, lane) "
            "VALUES (?, ?, ?, ?, ?, 'runtime')",
            (submission["hotkey"], incumbent_id, self._window(db)["id"], amount, self._now()),
        )
        self._terminal(db, job["id"], "crowned", f"runtime incumbent {incumbent_id}")

    # -- windows -----------------------------------------------------------

    def _open_window(self, db: sqlite3.Connection, rows: Sequence[Sequence[Any]]) -> None:
        """A new window with its sealed bank; the commitment and bank digest go public now."""
        secret = secrets.token_bytes(32)
        sealed = bank.Bank.from_json(rows)
        db.execute(
            "INSERT INTO windows (secret, commitment, opened_at, bank, bank_digest) "
            "VALUES (?, ?, ?, ?, ?)",
            (secret, bank.commitment(secret), self._now(), _dumps(sealed.to_json()), sealed.digest),
        )

    def _window(self, db: sqlite3.Connection) -> sqlite3.Row:
        row: sqlite3.Row = db.execute(
            f"SELECT {WINDOW_COLUMNS} FROM windows WHERE closed_at IS NULL "  # noqa: S608
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row

    def _bank(self, db: sqlite3.Connection, window_id: int) -> bank.Bank:
        """A window's bank, parsed once and cached (a window's bank never changes)."""
        cached = self._banks.get(window_id)
        if cached is not None:
            self._banks.move_to_end(window_id)
            return cached
        row = db.execute("SELECT bank FROM windows WHERE id=?", (window_id,)).fetchone()
        loaded = bank.Bank.from_json(json.loads(row["bank"])) if row else bank.EMPTY_BANK
        self._banks[window_id] = loaded
        while len(self._banks) > BANK_CACHE:
            self._banks.popitem(last=False)
        return loaded

    def _rotate(self, db: sqlite3.Connection) -> dict[str, Any]:
        """Close the open window (revealing its secret and bank) and promote the next bank,
        or open with an empty bank when none is ready."""
        old = self._window(db)
        row = db.execute("SELECT value FROM meta WHERE key='next_bank'").fetchone()
        rows = json.loads(row["value"]) if row else []
        db.execute("DELETE FROM meta WHERE key='next_bank'")
        db.execute("UPDATE windows SET closed_at=? WHERE id=?", (self._now(), old["id"]))
        self._open_window(db, rows)
        new = self._window(db)
        return {
            "closed": old["id"],
            "opened": new["id"],
            "commitment": new["commitment"],
            "bank_digest": new["bank_digest"],
        }

    def rotate_window(self) -> dict[str, Any]:
        with self._tx() as db:
            return self._rotate(db)

    def auto_rotate(self, min_age: float) -> dict[str, Any] | None:
        """Rotate once the next bank is ready and the open window is at least min_age old."""
        with self._tx() as db:
            ready = db.execute("SELECT 1 FROM meta WHERE key='next_bank'").fetchone()
            if not ready or self._now() - self._window(db)["opened_at"] < min_age:
                return None
            return self._rotate(db)

    def set_next_bank(self, items: Sequence[bank.BankItem]) -> str:
        """Store the next window's bank; it is sealed when that window opens."""
        sealed = bank.Bank(tuple(items))
        with self._tx() as db:
            self._set_meta(db, "next_bank", sealed.to_json())
        return sealed.digest

    def next_bank_ready(self) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM meta WHERE key='next_bank'").fetchone()
        return row is not None

    def windows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {WINDOW_COLUMNS} FROM windows ORDER BY id"  # noqa: S608
            ).fetchall()
        return [_window_json(row) for row in rows]

    def window(self, window_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                f"SELECT {WINDOW_COLUMNS} FROM windows WHERE id=?",  # noqa: S608
                (window_id,),
            ).fetchone()
            if row is None:
                raise StoreError(404, "unknown window")
            out = _window_json(row)
            if row["closed_at"] is not None:
                out["secret"] = row["secret"].hex()
                jobs = self._db.execute(
                    "SELECT j.id, j.mix, j.cases, j.state, j.evidence, j.plan, j.beacon, "
                    "j.judge, s.digest FROM jobs j JOIN submissions s ON s.id = j.submission_id "
                    "WHERE j.window_id=? AND j.state != 'queued' ORDER BY j.created_at, j.id",
                    (window_id,),
                ).fetchall()
                out["jobs"] = [
                    {
                        "id": j["id"],
                        "digest": j["digest"],
                        "mix": json.loads(j["mix"]),
                        "plan": plan_to_json(_job_plan(j)),
                        "beacon": json.loads(j["beacon"]) if j["beacon"] else None,
                        "judge": bool(j["judge"]),
                        "cases": j["cases"],
                        "state": j["state"],
                        "cases_sha256": _evidence(j, "cases_sha256"),
                        "cases_fetched": _evidence(j, "cases_fetched"),
                    }
                    for j in jobs
                ]
        return out

    def window_bank(self, window_id: int, offset: int, limit: int) -> dict[str, Any]:
        """A closed window's bank rows [kind, key, payload], paged under PAGE_BYTES."""
        with self._lock:
            row = self._db.execute(
                "SELECT closed_at, bank, bank_digest FROM windows WHERE id=?", (window_id,)
            ).fetchone()
        if row is None or row["closed_at"] is None:
            raise StoreError(404, "the bank of an open or unknown window is not published")
        rows = json.loads(row["bank"])
        items: list[Any] = []
        size = 0
        for item in rows[offset : offset + limit]:
            length = len(_dumps(item)) + 1
            if items and size + length > PAGE_BYTES - 4096:
                break
            items.append(item)
            size += length
        return {
            "window": window_id,
            "bank_digest": row["bank_digest"],
            "total": len(rows),
            "offset": offset,
            "items": items,
        }

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
                "SELECT 1 FROM submissions WHERE hotkey=? AND state='queued' AND lane='quality'",
                (hotkey,),
            ).fetchone():
                raise StoreError(409, "this hotkey already has an open submission")
            pending = db.execute(
                "SELECT count(*) FROM submissions WHERE state='queued' AND lane='quality'"
            ).fetchone()[0]
            if pending >= self.settings.max_pending:
                raise StoreError(429, "the duel queue is full, retry later")
            champion = self._champion(db)
            config = json.loads(champion["files"])["config.json"]
            if files.get("config.json") != config:
                # the base's while the champion is BF16; the NVFP4 export's once it migrated
                raise StoreError(422, "config.json must be byte-equal to the champion's")
            if config == pins.NVFP4_CONFIG_SHA256 and "model.safetensors.index.json" not in files:
                # the layout check reads the index before anything is served
                raise StoreError(422, "an NVFP4 checkpoint needs model.safetensors.index.json")
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
        submission = db.execute(
            "SELECT lane FROM submissions WHERE id=?", (submission_id,)
        ).fetchone()
        if not self._submission_current(db, submission["lane"], submission_id):
            # a runtime target is never moved (the miner signs again); a quality submission
            # in the champion's former weight format is resubmitted in the new one
            reason = EXPIRED if submission["lane"] == "runtime" else EXPIRED_FORMAT
            db.execute(
                "UPDATE submissions SET state='expired', reason=? WHERE id=?",
                (reason, submission_id),
            )
            return ""
        job_id = "j_" + secrets.token_hex(8)
        db.execute(
            "INSERT INTO jobs (id, submission_id, champion_id, window_id, seed, mix, retired, "
            "cases, state, created_at, lane) VALUES (?, ?, 0, 0, '', '{}', '[]', ?, 'queued', ?, "
            "?)",
            (job_id, submission_id, self.settings.duel_cases, self._now(), submission["lane"]),
        )
        self._target(db, job_id)
        db.execute("UPDATE submissions SET job_id=? WHERE id=?", (job_id, submission_id))
        return job_id

    def _target_current(self, db: sqlite3.Connection, job: sqlite3.Row) -> bool:
        """A quality job may duel any current champion of its weight format (its config.json);
        a runtime job only the champion and profile its submission signed."""
        return self._submission_current(db, job["lane"], job["submission_id"])

    def _submission_current(self, db: sqlite3.Connection, lane: str, submission_id: str) -> bool:
        if lane == "runtime":
            return self._signed_current(db, submission_id)
        files = db.execute("SELECT files FROM submissions WHERE id=?", (submission_id,)).fetchone()
        config = json.loads(self._champion(db)["files"]).get("config.json")
        return bool(json.loads(files["files"]).get("config.json") == config)

    def _signed_current(self, db: sqlite3.Connection, submission_id: str) -> bool:
        """The champion and the profile a runtime submission signed are still current. With
        no calibration published the profile cannot have moved: the job waits, parked."""
        signed = db.execute(
            "SELECT target, profile FROM submissions WHERE id=?", (submission_id,)
        ).fetchone()
        calibration = self._calibration(db)
        return bool(
            signed["target"] == self._champion(db)["id"]
            and (calibration is None or signed["profile"] == calibration.profile_digest)
        )

    def _target(self, db: sqlite3.Connection, job_id: str) -> None:
        """Point a job at the current champion and window: fresh seed (with the job's drand
        beacon, if any), level mix, effective plan and judge flag."""
        job = db.execute(
            "SELECT j.id, j.beacon, j.lane, j.submission_id, s.digest FROM jobs j "
            "JOIN submissions s ON s.id=j.submission_id WHERE j.id=?",
            (job_id,),
        ).fetchone()
        if not self._target_current(db, job):  # callers expire such jobs first
            raise StoreError(409, "a runtime job never moves off its signed target")
        if job["lane"] == "runtime" and self._calibration(db) is None:
            return  # calibration withdrawn: stays queued; the lease targets it once reopened
        champion, window = self._champion(db), self._window(db)
        mix, retired = self._mix(db, champion["id"])
        judge = self.judge
        if job["lane"] == "runtime":
            # fidelity on every measured track, no judge; the candidate duels stock
            calibration = self._calibration(db)
            assert calibration is not None  # checked above
            measured = runtime.fidelity_tracks(calibration)
            plan = {
                t: TrackPlan(1 / len(measured), self.settings.runtime_fidelity_cases)
                for t in measured
            }
            judge, retired = False, []
            incumbent = self._incumbent(db)
            db.execute(
                "UPDATE jobs SET incumbent_id=?, calibration=? WHERE id=?",
                (
                    incumbent["id"] if incumbent else 0,
                    self._calibration_raw(db),
                    job_id,
                ),
            )
        else:
            plan = tracks.effective_plan(
                self.settings.track_plan(), self._bank(db, window["id"]), self.judge
            )
        beacon = json.loads(job["beacon"]) if job["beacon"] else None
        db.execute(
            "UPDATE jobs SET champion_id=?, window_id=?, seed=?, mix=?, retired=?, plan=?, "
            "cases=?, judge=?, evidence=NULL WHERE id=?",
            (
                champion["id"],
                window["id"],
                bank.job_seed(window["secret"], job_id, job["digest"], beacon),
                _dumps(mix),
                _dumps(retired),
                _dumps(plan_to_json(plan)),
                sum(p.cases for p in plan.values()),
                int(judge),
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
            pending = db.execute(
                "SELECT count(*) FROM judgments WHERE job_id=? AND state='pending'", (job["id"],)
            ).fetchone()[0]
            out["job"] = {
                "id": job["id"],
                "state": job["state"],
                "judgments_pending": pending,
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

    def lease(self, lane: str = "quality", nvfp4: bool = False) -> dict[str, Any] | None:
        """The next job of one lane. A runtime job never runs beside any other job: while one
        is leased nothing else is handed out, and it waits for every leased job to finish.

        ponytail: exclusion is global across this challenge's workers, not per GPU; add a
        per-GPU reservation when runtime jobs are frequent enough for it to cost throughput."""
        if lane not in runtime.LANES:
            raise StoreError(400, f"lane must be one of {runtime.LANES}")
        if lane == "runtime":
            with self._tx() as db:
                self._set_meta(db, "runtime_polled", self._now())
        else:
            with self._lock:
                if nvfp4 != self._champion_nvfp4(self._db):
                    # a worker serves one weight format: BF16 duels on the H200 path, NVFP4
                    # duels on the B300 sandbox path, never mixed within the lane (checked
                    # again in the leasing transaction: a migration may land in between)
                    return None
        return self._lease(lane, nvfp4)

    def _runtime_due(self, db: sqlite3.Connection) -> bool:
        """Quality leases pause so leased quality jobs drain and the runtime job gets the GPU:
        only while a runtime job is queued, a runtime worker polled within
        RUNTIME_WORKER_SECONDS (no worker, no drain) and at least runtime_every quality
        leases went out since the last runtime lease (quality keeps that share)."""
        if not self._runtime_open(db):
            return False
        queued = db.execute("SELECT 1 FROM jobs WHERE lane='runtime' AND state='queued'").fetchone()
        polled = self._meta_opt(db, "runtime_polled") or 0
        since = self._meta_opt(db, "quality_since_runtime") or 0
        return (
            queued is not None
            and self._now() - polled <= RUNTIME_WORKER_SECONDS
            and since >= self.settings.runtime_every
        )

    def _count_lease(self, db: sqlite3.Connection, lane: str) -> None:
        since = self._meta_opt(db, "quality_since_runtime") or 0
        self._set_meta(db, "quality_since_runtime", since + 1 if lane == "quality" else 0)

    def _exclusive_blocked(self, db: sqlite3.Connection, lane: str) -> bool:
        """A leased runtime job blocks every lease; a runtime lease waits for all others."""
        lanes = {r["lane"] for r in db.execute("SELECT lane FROM jobs WHERE state='leased'")}
        return "runtime" in lanes or (lane == "runtime" and bool(lanes))

    def _lease(self, lane: str, nvfp4: bool = False) -> dict[str, Any] | None:
        # The drand beacon is fetched once per job, outside the lock (5 s timeout), and
        # stored on its first lease; retries reuse it. Expired leases are released first so
        # the head the beacon is fetched for is the job leased below; if a concurrent lease
        # or intake still moves the head meanwhile, fetch again for the new head.
        now = self._now()
        with self._tx() as db:
            expired = db.execute(
                "SELECT id FROM jobs WHERE state='leased' AND lease_expires < ?", (now,)
            ).fetchall()
            for job in expired:
                self._release(db, job["id"], "lease expired")
            if expired:
                self._finalize(db)
        beacon: tuple[str, dict[str, Any] | None] | None = None
        for _ in range(3):
            with self._lock:
                head = self._db.execute(
                    "SELECT j.id, j.beacon_fetched FROM jobs j JOIN submissions s "
                    "ON s.id=j.submission_id WHERE j.state='queued' AND j.lane=? "
                    "ORDER BY s.intake LIMIT 1",
                    (lane,),
                ).fetchone()
            if head is None or head["beacon_fetched"] or (beacon and beacon[0] == head["id"]):
                break
            beacon = (head["id"], self.beacon())
        with self._tx() as db:
            if self._exclusive_blocked(db, lane):
                return None
            if lane == "runtime" and not self._runtime_open(db):
                return None
            if lane == "quality" and nvfp4 != self._champion_nvfp4(db):
                return None
            self._expire_off_target(db)
            if lane == "quality" and self._runtime_due(db):
                return None  # drain for the waiting runtime job
            job = db.execute(
                "SELECT j.* FROM jobs j JOIN submissions s ON s.id=j.submission_id "
                "WHERE j.state='queued' AND j.lane=? ORDER BY s.intake LIMIT 1",
                (lane,),
            ).fetchone()
            if job is None:
                return None
            self._count_lease(db, lane)
            champion = self._champion(db)
            if beacon is not None and beacon[0] == job["id"] and not job["beacon_fetched"]:
                db.execute(
                    "UPDATE jobs SET beacon=?, beacon_fetched=1 WHERE id=?",
                    (_dumps(beacon[1]) if beacon[1] else None, job["id"]),
                )
            # else: the head moved three times while drand was fetched; this attempt runs
            # with v1's seed and beacon_fetched stays 0, so a retry fetches one.
            self._target(db, job["id"])  # current champion, window, mix, plan and seed
            lease = secrets.token_hex(16)
            db.execute(
                "UPDATE jobs SET state='leased', lease=?, lease_expires=? WHERE id=?",
                (lease, now + LEASE_SECONDS, job["id"]),
            )
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
            submission = db.execute(
                "SELECT * FROM submissions WHERE id=?", (job["submission_id"],)
            ).fetchone()
            out = {
                "job": job["id"],
                "lease": lease,
                "lease_expires": _iso(now + LEASE_SECONDS),
                "cases": job["cases"],
                "plan": json.loads(job["plan"]),
                "champion": _manifest(champion),
                "challenger": _manifest(submission),
                "base": {"repo": pins.BASE_REPO, "revision": pins.BASE_REVISION},
            }
            if lane == "runtime":
                incumbent = self._incumbent(db)
                out["challenger"] = None  # same weights as the champion
                out["lane"] = "runtime"
                out["runtime"] = {
                    "calibration": json.loads(job["calibration"]),
                    "incumbent": json.loads(incumbent["options"]) if incumbent else {},
                    "candidate": json.loads(submission["options"]),
                    # full sources: only the controller forwards them, into sandboxes
                    "incumbent_kernel": _kernel(incumbent["kernel"]) if incumbent else None,
                    "candidate_kernel": _kernel(submission["kernel"]),
                    "seed": job["seed"],  # the private workload, sealed like a duel's cases
                    "sides": {"champion": "stock", "challenger": "candidate"},
                }
            return out

    def _release(self, db: sqlite3.Connection, job_id: str, reason: str) -> None:
        """Give a job back to the queue after an infrastructure failure, or fail it. A runtime
        job whose signed target is no longer the champion expires instead."""
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        db.execute("DELETE FROM results WHERE job_id=?", (job_id,))
        db.execute("DELETE FROM judgments WHERE job_id=?", (job_id,))
        db.execute("DELETE FROM runtime_tasks WHERE job_id=?", (job_id,))
        if not self._target_current(db, job):
            self._expire(db, job)
            return
        attempts = job["attempts"] + 1
        if attempts >= MAX_ATTEMPTS:
            self._terminal(db, job_id, "failed", f"{reason} ({attempts} attempts)")
        else:
            db.execute(
                "UPDATE jobs SET state='queued', lease=NULL, lease_expires=NULL, attempts=?, "
                "stopped=0, reason=? WHERE id=?",
                (attempts, reason, job_id),
            )

    def _expire(self, db: sqlite3.Connection, job: sqlite3.Row) -> None:
        """A job that can no longer duel its target: no judge spends anything on it."""
        db.execute("DELETE FROM judgments WHERE job_id=? AND state='pending'", (job["id"],))
        reason = EXPIRED if job["lane"] == "runtime" else EXPIRED_FORMAT
        self._terminal(db, job["id"], "expired", reason)

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
            raise StoreError(409, NOT_LEASED)
        return job

    def heartbeat(self, job_id: str, lease: str) -> dict[str, Any]:
        """Extend a lease while the worker downloads weights and starts the servers."""
        with self._tx() as db:
            self._leased(db, job_id, lease)
            expires = self._now() + LEASE_SECONDS
            db.execute("UPDATE jobs SET lease_expires=? WHERE id=?", (expires, job_id))
            stale = self._stale(
                db, db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )
        return {"lease_expires": _iso(expires), "stale": stale}

    def _stale(self, db: sqlite3.Connection, job: sqlite3.Row) -> bool:
        """The job no longer duels the current state of its lane."""
        if job["champion_id"] != self._champion(db)["id"]:
            return True
        if job["lane"] != "runtime":
            return False
        incumbent = self._incumbent(db)
        calibration = self._calibration(db)
        return job["incumbent_id"] != (incumbent["id"] if incumbent else 0) or (
            calibration is None or job["calibration"] != self._calibration_raw(db)
        )

    def _case(self, job: sqlite3.Row, index: int) -> tuple[Case, int]:
        """Case `index` of a job and the length of its body's JSON (byte-bounded cache)."""
        with self._lock:
            window_bank = self._bank(self._db, job["window_id"])
        plan = _job_plan(job)
        key = (job["seed"], _dumps(plan_to_json(plan)), job["mix"], job["judge"], index)
        return self._cases.get(
            key,
            lambda: tracks.job_case(
                job["seed"], plan, json.loads(job["mix"]), index, window_bank, bool(job["judge"])
            ),
        )

    def cases(self, job_id: str, lease: str, offset: int, limit: int) -> list[dict[str, Any]]:
        """At most `limit` cases and PAGE_BYTES of JSON, and at least one while any remain."""
        with self._lock:
            job = self._leased(self._db, job_id, lease)
        out: list[dict[str, Any]] = []
        size = 16
        for index in range(offset, min(offset + limit, job["cases"])):
            case, length = self._case(job, index)
            length += 64 + len(case.track)
            if out and size + length > PAGE_BYTES:
                break
            out.append({"index": index, "track": case.track, "body": case.body})
            size += length
        return out

    def record_answers(
        self, job_id: str, lease: str, items: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        with self._lock:
            job = self._leased(self._db, job_id, lease)
        rows, judgments = [], []
        for raw in items:
            item = {k: v for k, v in raw.items() if v is not None}
            index = item["case_index"]
            if not 0 <= index < job["cases"]:
                raise StoreError(400, f"case_index {index} is outside the job")
            case, _ = self._case(job, index)
            score = tracks.score_item(case, item)
            if score is None:  # the loss needs the judge: keep the container's own render
                judgments.append(
                    (
                        job_id,
                        index,
                        item["side"],
                        case.track,
                        case.level,
                        int(case.body["seed"]),
                        _final_png(case, item["transcript"]),
                        case.private["brief"],
                        _dumps(list(case.private["rubric"])),
                    )
                )
                continue
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
                    case.track,
                )
            )
        with self._tx() as db:
            job = self._leased(db, job_id, lease)
            db.executemany(
                f"INSERT OR IGNORE INTO results ({RESULT_COLUMNS}) "  # noqa: S608
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            db.executemany(
                "INSERT OR IGNORE INTO judgments (job_id, case_index, side, track, level, seed, "
                "png, brief, rubric) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                judgments,
            )
            db.execute(
                "UPDATE jobs SET lease_expires=? WHERE id=?",
                (self._now() + LEASE_SECONDS, job_id),
            )
            stopped = bool(job["stopped"]) or (
                job["lane"] == "quality" and self._should_stop(db, job)
            )
            if stopped and not job["stopped"]:
                db.execute("UPDATE jobs SET stopped=1 WHERE id=?", (job_id,))
            stale = self._stale(db, job)
            paired = self._paired_count(db, job_id)
            answered = self._answered_count(db, job_id)
        return {
            "accepted": len(rows) + len(judgments),
            "paired": paired,
            "judging": len(judgments),
            "continue": not (stopped or stale or answered >= job["cases"]),
            "early_stop": stopped,
            "stale": stale,
        }

    def _answered_count(self, db: sqlite3.Connection, job_id: str) -> int:
        """Cases both sides answered: scored or waiting for the judge."""
        return int(
            db.execute(
                "SELECT count(*) FROM (SELECT case_index FROM (SELECT case_index, side "
                "FROM results WHERE job_id=? UNION SELECT case_index, side FROM judgments "
                "WHERE job_id=?) GROUP BY case_index HAVING count(*) = 2)",
                (job_id, job_id),
            ).fetchone()[0]
        )

    def _paired_count(self, db: sqlite3.Connection, job_id: str) -> int:
        return int(
            db.execute(
                "SELECT count(*) FROM results a JOIN results b ON b.job_id=a.job_id "
                "AND b.case_index=a.case_index AND b.side='challenger' "
                "WHERE a.job_id=? AND a.side='champion'",
                (job_id,),
            ).fetchone()[0]
        )

    def _should_stop(self, db: sqlite3.Connection, job: sqlite3.Row) -> bool:
        """Early stop on the composite of per-track SQL moments, guard levels excluded."""
        rows = db.execute(
            "SELECT a.track, count(*), coalesce(sum(a.decisions), 0), coalesce(sum(a.loss), 0), "
            "coalesce(sum(b.loss), 0), coalesce(sum(a.loss*a.loss), 0), "
            "coalesce(sum(b.loss*b.loss), 0), coalesce(sum(a.loss*b.loss), 0) "
            "FROM results a JOIN results b ON b.job_id=a.job_id AND b.case_index=a.case_index "
            "AND b.side='challenger' WHERE a.job_id=? AND a.side='champion' "
            "AND NOT (a.track='decisions' AND a.level IN (SELECT value FROM json_each(?))) "
            "GROUP BY a.track ORDER BY a.track",
            (job["id"], job["retired"]),
        ).fetchall()
        if sum(r[2] for r in rows) < scoring.EARLY_STOP_DECISIONS:
            return False
        moments = {r[0]: (r[1], r[3], r[4], r[5], r[6], r[7]) for r in rows}
        g, se = scoring.composite(moments, _weights(job))
        return g + scoring.EARLY_STOP_SE * se < 0

    def _pairs(self, db: sqlite3.Connection, job_id: str) -> list[scoring.Paired]:
        rows = db.execute(
            "SELECT a.case_index, a.level, a.loss, a.decisions, a.determined, a.correct, "
            "a.under_loss, a.under, b.loss, b.decisions, b.determined, b.correct, "
            "b.under_loss, b.under, a.track FROM results a JOIN results b "
            "ON b.job_id=a.job_id AND b.case_index=a.case_index AND b.side='challenger' "
            "WHERE a.job_id=? AND a.side='champion' ORDER BY a.case_index",
            (job_id,),
        ).fetchall()
        return [
            scoring.Paired(
                r[0], r[1], scoring.CaseScore(*r[2:8]), scoring.CaseScore(*r[8:14]), r[14]
            )
            for r in rows
        ]

    def record_timings(
        self, job_id: str, lease: str, items: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Timed runtime tasks: latency from the worker's clock, success only from the
        container scoring the raw output against the case it rebuilds from the job's seed."""
        with self._lock:
            job = self._leased(self._db, job_id, lease)
        if job["lane"] != "runtime":
            raise StoreError(409, "timings belong to runtime jobs")
        calibration = runtime.Calibration.from_json(json.loads(job["calibration"]))
        rows = []
        for raw in items:
            item = {k: v for k, v in raw.items() if v is not None}
            cell = calibration.cells.get(item["cell"])
            if cell is None or not 0 <= item["case_index"] < cell.cases:
                raise StoreError(400, f"no case {item['case_index']} in cell {item['cell']}")
            if not item["block"] < calibration.blocks:
                raise StoreError(400, f"block {item['block']} is outside the calibration")
            case = runtime.cell_case(job["seed"], item["cell"], cell, item["case_index"])
            ok = runtime.task_ok(case, item)
            vectors = runtime.answer_vectors(case, item)
            rows.append(
                (
                    job_id,
                    item["block"],
                    item["side"],
                    item["cell"],
                    item["case_index"],
                    float(item["ms"]),
                    int(ok),
                    int("error" in item),
                    None if vectors is None else _dumps(vectors),
                )
            )
        with self._tx() as db:
            self._leased(db, job_id, lease)
            db.executemany(
                "INSERT OR IGNORE INTO runtime_tasks (job_id, block, side, cell, case_index, ms, "
                "ok, error, vectors) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            db.execute(
                "UPDATE jobs SET lease_expires=? WHERE id=?",
                (self._now() + LEASE_SECONDS, job_id),
            )
        return {"accepted": len(rows), "ok": sum(r[6] for r in rows)}

    def complete(self, job_id: str, lease: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        with self._tx() as db:
            job = self._leased(db, job_id, lease)
            stale = self._stale(db, job)
            answered = self._answered_count(db, job_id)
            if not (job["stopped"] or stale) and answered < job["cases"]:
                raise StoreError(409, f"{answered} of {job['cases']} cases answered by both sides")
            db.execute("UPDATE jobs SET evidence=? WHERE id=?", (_dumps(dict(evidence)), job_id))
            pending = db.execute(
                "SELECT 1 FROM judgments WHERE job_id=? AND state='pending'", (job_id,)
            ).fetchone()
            if pending:
                db.execute(
                    # lease_expires is the judging deadline while the job is 'judging'
                    "UPDATE jobs SET state='judging', lease=NULL, lease_expires=? WHERE id=?",
                    (self._now() + JUDGE_DEADLINE_SECONDS, job_id),
                )
            else:
                self._settle(db, job_id)
            return self._submission(db, job["submission_id"])

    def _settle(self, db: sqlite3.Connection, job_id: str) -> None:
        """Score a job whose answers are all in: a pair whose two renders the judge could not
        read is dropped on both sides, then the verdict, the champion's level stats and the
        crown queue. A crown is refused when more than UNJUDGED_MAX of the judged cases were
        dropped, since the drop depends on the renders."""
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job["lane"] == "runtime":
            self._settle_runtime(db, job)
            return
        unjudged = db.execute(
            "SELECT case_index FROM judgments WHERE job_id=? GROUP BY case_index "
            "HAVING sum(state='unjudged') = 2 OR sum(state='expired') > 0",
            (job_id,),
        ).fetchall()
        judged = db.execute(
            "SELECT count(DISTINCT case_index) FROM judgments WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        db.executemany(
            "DELETE FROM results WHERE job_id=? AND case_index=?",
            [(job_id, r[0]) for r in unjudged],
        )
        pairs = self._pairs(db, job_id)
        result = scoring.verdict(
            pairs, set(json.loads(job["retired"])), bool(job["stopped"]), _weights(job)
        )
        result["unjudged"] = len(unjudged)
        result["unjudged_max"] = UNJUDGED_MAX
        if judged and len(unjudged) > UNJUDGED_MAX * judged:
            result["crown"] = False
        db.execute(
            "UPDATE jobs SET state='scored', lease=NULL, verdict=?, finished_at=? WHERE id=?",
            (_dumps(result), self._now(), job_id),
        )
        self._add_level_stats(db, job["champion_id"], pairs, "champion")
        self._retire_levels(db)
        self._finalize(db)

    # -- judging ----------------------------------------------------------------

    def pending_judgments(self, limit: int = 64) -> list[dict[str, Any]]:
        """Pending judgments of up to `limit` cases, each case's sides in the order drawn
        from its seed."""
        with self._lock:
            cases = self._db.execute(
                "SELECT DISTINCT job_id, case_index FROM judgments WHERE state='pending' "
                "ORDER BY job_id, case_index LIMIT ?",
                (limit,),
            ).fetchall()
            rows = [
                self._db.execute(
                    "SELECT job_id, case_index, side, seed, png, brief, rubric FROM judgments "
                    "WHERE job_id=? AND case_index=? AND state='pending'",
                    (c["job_id"], c["case_index"]),
                ).fetchall()
                for c in cases
            ]
        out = []
        for group in rows:
            by_side = {r["side"]: r for r in group}
            for side in judge_order(group[0]["seed"]):
                if side in by_side:
                    r = by_side[side]
                    out.append(
                        {
                            "job": r["job_id"],
                            "case_index": r["case_index"],
                            "side": side,
                            "png": bytes(r["png"]),
                            "brief": r["brief"],
                            "rubric": json.loads(r["rubric"]),
                        }
                    )
        return out

    def record_judgment(self, job_id: str, case_index: int, side: str, loss: float | None) -> None:
        """A judged side becomes a result. None (the judge could not read this side's
        render) forfeits the side with loss 1, as a render is the model's own output; when
        both sides are unreadable the case is unjudged and dropped on both sides at settle."""
        with self._tx() as db:
            row = db.execute(
                "SELECT * FROM judgments WHERE job_id=? AND case_index=? AND side=? "
                "AND state='pending'",
                (job_id, case_index, side),
            ).fetchone()
            if row is None:
                return  # the job was released or requeued meanwhile
            score = scoring.harness_score(1.0 if loss is None else loss)
            db.execute(
                "UPDATE judgments SET state=?, loss=? WHERE job_id=? AND case_index=? AND side=?",
                ("unjudged" if loss is None else "judged", score.loss, job_id, case_index, side),
            )
            db.execute(
                f"INSERT OR IGNORE INTO results ({RESULT_COLUMNS}) "  # noqa: S608
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    case_index,
                    side,
                    row["level"],
                    score.loss,
                    score.decisions,
                    score.determined,
                    score.correct,
                    score.under_loss,
                    score.under,
                    row["track"],
                ),
            )

    def settle_judged(self) -> list[str]:
        """Settle every judging job with no pending judgment, exactly like complete. A job
        past its judging deadline (judge down, or a restart without the teacher) first has
        its pending sides expired: those cases drop on both sides and count as unjudged, so
        the crown is refused past UNJUDGED_MAX instead of blocking later crowns forever."""
        with self._tx() as db:
            db.execute(
                "UPDATE judgments SET state='expired' WHERE state='pending' AND job_id IN "
                "(SELECT id FROM jobs WHERE state='judging' AND lease_expires < ?)",
                (self._now(),),
            )
            jobs = db.execute(
                "SELECT id FROM jobs WHERE state='judging' AND NOT EXISTS (SELECT 1 FROM "
                "judgments WHERE judgments.job_id=jobs.id AND state='pending') ORDER BY id"
            ).fetchall()
            for job in jobs:
                self._settle(db, job["id"])
        return [job["id"] for job in jobs]

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
            db.execute("DELETE FROM judgments WHERE job_id=?", (job_id,))
            db.execute("DELETE FROM runtime_tasks WHERE job_id=?", (job_id,))
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
            if pair.track != "decisions":  # the ladder is the decisions track's
                continue
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
                "ON s.id=j.submission_id WHERE j.state='scored' AND j.lane='quality' "
                "ORDER BY s.intake"
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
                    "WHERE j.champion_id=? AND s.intake < ? AND j.lane='quality' "
                    "AND j.state IN ('queued', 'leased', 'judging', 'scored')",
                    (champion["id"], job["intake"]),
                ).fetchone()
                if paused or blocked:
                    continue
                self._crown(db, job)
                progress = True
                break
        self._finalize_runtime(db, paused)

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
            "SELECT coalesce(sum(amount), 0) FROM entitlements WHERE window_id=? "
            "AND lane='quality'",
            (window["id"],),
        ).fetchone()[0]
        amount = ledger.entitlement_units(result["g_lcb"], window_total, self.settings.window_cap)
        duel_bank = db.execute(
            "SELECT bank_digest FROM windows WHERE id=?", (job["window_id"],)
        ).fetchone()[0]
        if duel_bank == bank.EMPTY_BANK.digest:
            amount = min(amount, int(self.settings.empty_bank_cap * ledger.UNITS))
        db.execute(
            "INSERT INTO entitlements (hotkey, champion_id, window_id, amount, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (submission["hotkey"], champion_id, window["id"], amount, self._now()),
        )
        self._terminal(db, job["id"], "crowned", f"crowned as champion {champion_id}")
        self._retire_levels(db)
        self._expire_off_target(db)

    def _expire_off_target(self, db: sqlite3.Connection) -> None:
        """Queued jobs that can no longer duel the current target expire: a runtime job whose
        signed champion or profile moved (the target is never moved), a quality submission in
        another weight format than the champion's. A leased or judging one turns stale and
        expires when it completes or is released, never under a worker's feet."""
        for queued in db.execute("SELECT * FROM jobs WHERE state='queued'").fetchall():
            if not self._target_current(db, queued):
                self._expire(db, queued)

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
            start = self._meta_opt(db, "lanes_from_epoch")
            split = start is not None and epoch >= start
            # before the split the historical rule holds: quality credits share one epoch-mass
            budgets = runtime.BUDGETS if split else {"quality": ledger.UNITS}
            by_id = {
                r["id"]: r
                for r in db.execute(
                    "SELECT id, hotkey, champion_id, lane FROM entitlements"
                ).fetchall()
            }
            payments: list[tuple[int, int]] = []
            lanes: dict[str, dict[str, float]] = {}
            for lane, budget in budgets.items():
                owed = [
                    ledger.Entitlement(r["id"], r["hotkey"], r["amount"], r["paid"])
                    for r in db.execute(
                        "SELECT * FROM entitlements WHERE paid < amount AND lane=? ORDER BY id",
                        (lane,),
                    ).fetchall()
                ]
                paid = ledger.pay(owed, budget)
                payments += paid
                spent = sum(amount for _, amount in paid)
                lanes[lane] = {
                    "budget": budget / ledger.UNITS,
                    "paid": spent / ledger.UNITS,
                    "burned": (budget - spent) / ledger.UNITS,  # never given to the other lane
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
            rows = []
            for eid, amount in payments:
                entry = {
                    "entitlement": eid,
                    "champion": by_id[eid]["champion_id"],
                    "hotkey": by_id[eid]["hotkey"],
                    "mass": amount / ledger.UNITS,
                }
                if split:
                    entry["lane"] = by_id[eid]["lane"]
                rows.append(entry)
            metadata: dict[str, Any] = {
                "payments": rows,
                "burned": (ledger.UNITS - paid_units) / ledger.UNITS,
            }
            if split:
                metadata["lanes"] = lanes
            body = {
                "challenge_slug": slug,
                "epoch": epoch,
                "weights": {k: v / ledger.UNITS for k, v in sorted(weights.items())},
                "full_share_mass": 1.0,
                "metadata": metadata,
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
                "JOIN jobs j ON j.id=s.job_id WHERE s.state='queued' AND s.lane='quality' "
                "ORDER BY s.intake"
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
            plan = tracks.effective_plan(
                self.settings.track_plan(), self._bank(db, window["id"]), self.judge
            )
            teacher = (
                "off"
                if not self.teacher_configured
                else "building"
                if self.teacher_building
                else "ready"
                if db.execute("SELECT 1 FROM meta WHERE key='next_bank'").fetchone()
                else "configured"
            )
            # the latest started duel only: a PK range scan of at most one plan's rows, never
            # the all-time table, since every store call waits on this lock
            track_counts = {
                r[0]: r[1]
                for r in db.execute(
                    "SELECT track, count(*) FROM results WHERE job_id=(SELECT id FROM jobs "
                    "WHERE state != 'queued' ORDER BY created_at DESC, id DESC LIMIT 1) "
                    "GROUP BY track ORDER BY track"
                ).fetchall()
            }
            judging = db.execute("SELECT count(*) FROM judgments WHERE state='pending'").fetchone()[
                0
            ]
            return {
                "champion": _champion_json(champion),
                "queue": [dict(row) for row in queue],
                "levels": levels,
                "next_duel_mix": mix,
                "window": {
                    "id": window["id"],
                    "commitment": window["commitment"],
                    "bank_digest": window["bank_digest"],
                    "opened_at": _iso(window["opened_at"]),
                },
                "plan": plan_to_json(plan),
                "tracks": {
                    t: {"weight": p.weight, "cases": p.cases, "results": track_counts.get(t, 0)}
                    for t, p in sorted(plan.items())
                },
                "teacher": {"state": teacher, "judge": self.judge, "judgments_pending": judging},
                "crowns_paused": self._meta(db, "crowns_paused"),
                "lanes": {
                    "from_epoch": self._meta_opt(db, "lanes_from_epoch"),
                    "runtime_open": self._runtime_open(db),
                },
                "constants": {
                    "g_min": scoring.G_MIN,
                    "z": scoring.Z99,
                    "guard_max": scoring.GUARD_MAX,
                    "duel_cases": sum(p.cases for p in plan.values()),
                    "early_stop_decisions": scoring.EARLY_STOP_DECISIONS,
                    "early_stop_se": scoring.EARLY_STOP_SE,
                    "retire_accuracy": scoring.RETIRE_ACCURACY,
                    "max_pending": self.settings.max_pending,
                    "window_entitlement_cap": self.settings.window_cap,
                    "empty_bank_entitlement_cap": self.settings.empty_bank_cap,
                    "base": {"repo": pins.BASE_REPO, "revision": pins.BASE_REVISION},
                },
            }

    def runtime_status(self) -> dict[str, Any]:
        """The runtime lane: open or not, calibration, incumbent, queue and allocations."""
        with self._lock:
            db = self._db
            calibration = self._calibration(db)
            champion = self._champion(db)
            incumbent = self._incumbent(db)
            queue = db.execute(
                "SELECT s.id, s.hotkey, s.intake, j.state, j.id AS job FROM submissions s "
                "JOIN jobs j ON j.id=s.job_id WHERE s.state='queued' AND s.lane='runtime' "
                "ORDER BY s.intake"
            ).fetchall()
            history = db.execute(
                "SELECT id, hotkey, model_champion_id, options, profile_digest, calibration, "
                "job_id, g_lcb, credited, crowned_at FROM runtime_incumbents ORDER BY id"
            ).fetchall()
            return {
                "open": self._runtime_open(db),
                "lanes_from_epoch": self._meta_opt(db, "lanes_from_epoch"),
                "budgets": {k: v / ledger.UNITS for k, v in runtime.BUDGETS.items()},
                "options": {
                    k: {"flag": v[0], "min": v[2], "max": v[3]} for k, v in runtime.OPTIONS.items()
                },
                "kernels": {
                    "slots": list(runtime.KERNEL_SLOTS),
                    "open": list(calibration.kernel_slots) if calibration else [],
                    "max_bytes": runtime.KERNEL_MAX_BYTES,
                },
                "weights": {
                    "format": "modelopt-nvfp4",
                    "champion_nvfp4": self._champion_nvfp4(db),
                    "reference": {
                        "repo": pins.NVFP4_REPO,
                        "revision": pins.NVFP4_REVISION,
                        "files": pins.NVFP4_FILES,
                    },
                },
                "gpu": runtime.GPU_TYPE,
                "calibration": calibration.public() if calibration else None,
                "target": {"champion": champion["id"], "digest": champion["digest"]},
                "incumbent": None
                if incumbent is None
                else {
                    "id": incumbent["id"],
                    "hotkey": incumbent["hotkey"],
                    "options": json.loads(incumbent["options"]),
                    "kernel": runtime.kernel_ref(_kernel(incumbent["kernel"])),
                },
                "queue": [dict(row) for row in queue],
                "crowns": [
                    {
                        **{k: row[k] for k in row.keys() if k != "options"},
                        "options": json.loads(row["options"]),
                        "credited": row["credited"] / ledger.UNITS,
                        "crowned_at": _iso(row["crowned_at"]),
                    }
                    for row in history
                ],
            }

    def leaderboard(self) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, e.amount, e.paid FROM champions c LEFT JOIN entitlements e "
                "ON e.champion_id=c.id AND e.lane='quality' ORDER BY c.id"
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


def _weights(job: sqlite3.Row) -> dict[str, float]:
    return {t: p.weight for t, p in _job_plan(job).items()}


def _evidence(row: sqlite3.Row, key: str) -> Any:
    return json.loads(row["evidence"]).get(key) if row["evidence"] else None


def _kernel(value: str | None) -> dict[str, Any] | None:
    return json.loads(value) if value else None


def _kernel_sha(value: str | None) -> str | None:
    kernel = _kernel(value)
    return kernel["sha256"] if kernel else None


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
        "bank_digest": row["bank_digest"],
    }
