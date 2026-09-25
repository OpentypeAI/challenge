"""SQL track: an analyst agent querying a sandboxed SQLite database (docs/tracks.md §5.2).

A case is a small shop database (customers, products, orders, order lines and, from level 3,
reviews) and one question rendered from a structured query spec. The gold is the result of
the spec's compiled reference SQL on the same database, recomputed by `loss` from the task:
the served body holds the spec and the rows, never the answer.

Model SQL runs only in `_Sandbox`: a fresh in-memory database per episode, an authorizer that
allows reads of the task tables and nothing else, a function denylist (nondeterminism and
extensions), a VM step limit and tight length limits. Every other output is data.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from .generator import Case, GeneratorError
from .harness import INVALID, Content, Env, parse_action, text

if TYPE_CHECKING:
    from .bank import Bank

HEADER = "OpenType harness: sql"
TURNS = 12
MAX_TOKENS = 1024
MAX_QUERY = 4000  # characters of query text
MAX_ROWS = 50
MAX_OBS = 4000  # characters of one query observation
# VM steps per query: the worst adversarial query then stops in ~0.06 s; reference queries
# take < 10k steps and a four-way analyst join ~200k.
MAX_STEPS = 500_000
MAX_LIST = 20  # answer list length a spec may have
ATTEMPTS = 200

CITIES = (
    "Austin Berlin Bordeaux Cairo Denver Dublin Lagos Lima Lyon Madrid Osaka Oslo Porto "
    "Quito Seoul Tunis Turin Vienna"
).split()
CATEGORIES = ("audio", "books", "garden", "kitchen", "office", "outdoor", "toys", "travel")
STATUSES = ("cancelled", "delivered", "pending", "returned", "shipped")
FIRST = (
    "Ada Ben Chloe David Emma Farid Grace Hugo Ines Jonas Kofi Lea Mateo Nora Omar Priya Rosa "
    "Sami Tara Vera"
).split()
LAST = (
    "Berg Dubois Garcia Haddad Ibrahim Kowalski Larsen Lopez Martin Moreau Nguyen Novak Okafor "
    "Patel Rossi Schmidt Silva Tanaka"
).split()
ADJECTIVES = (
    "Amber Brisk Cedar Coral Dune Ember Frost Harbor Juniper Maple Nova Onyx Pine Quartz Sage Slate"
).split()
NOUNS = "Anchor Beacon Bolt Crest Drift Field Grove Loop Peak Ridge Spark Trail".split()
YEAR = 2025

# name -> column declarations; reviews exists from level 3.
SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "customers": (
        ("customer_id", "INTEGER"),
        ("name", "TEXT"),
        ("city", "TEXT"),
        ("signup_date", "TEXT"),
    ),
    "products": (
        ("product_id", "INTEGER"),
        ("name", "TEXT"),
        ("category", "TEXT"),
        ("price", "REAL"),
    ),
    "orders": (
        ("order_id", "INTEGER"),
        ("customer_id", "INTEGER"),
        ("order_date", "TEXT"),
        ("status", "TEXT"),
    ),
    "order_items": (("order_id", "INTEGER"), ("product_id", "INTEGER"), ("qty", "INTEGER")),
    "reviews": (("product_id", "INTEGER"), ("customer_id", "INTEGER"), ("stars", "INTEGER")),
}


@dataclass(frozen=True)
class Level:
    customers: tuple[int, int]
    products: tuple[int, int]
    orders: tuple[int, int]
    lines: int  # max order_items lines per order
    reviews: tuple[int, int]  # (0, 0): no reviews table


LEVELS: dict[int, Level] = {
    1: Level((20, 40), (20, 30), (40, 80), 2, (0, 0)),
    2: Level((30, 60), (20, 40), (80, 160), 2, (0, 0)),
    3: Level((40, 80), (30, 50), (120, 180), 2, (80, 200)),
    4: Level((60, 100), (40, 60), (140, 180), 2, (100, 200)),
}


@dataclass(frozen=True)
class Kind:
    level: int
    question: str  # str.format template over the params
    sql: str  # str.format template over the SQL-quoted params (and k_plus = k + 1)
    answer: str  # scalar | first (unique top row) | top (k ordered rows) | all (a set)


REVENUE = "oi.qty * p.price"
LINES = (
    "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
    "JOIN orders o ON o.order_id = oi.order_id"
)
KINDS: dict[str, Kind] = {
    "count_city": Kind(
        1,
        "How many customers live in {city}? Answer with a number.",
        "SELECT COUNT(*) FROM customers WHERE city = {city}",
        "scalar",
    ),
    "max_price": Kind(
        1,
        "What is the highest product price in the {category} category? Answer with a number.",
        "SELECT MAX(price) FROM products WHERE category = {category}",
        "scalar",
    ),
    "count_status_range": Kind(
        1,
        "How many orders with status {status} have an order_date from {start} to {end} "
        "inclusive? Answer with a number.",
        "SELECT COUNT(*) FROM orders WHERE status = {status} "
        "AND order_date BETWEEN {start} AND {end}",
        "scalar",
    ),
    "revenue_category": Kind(
        2,
        "What is the total revenue (the sum of qty times price) of {category} products in "
        "orders with status {status}? Answer with a number.",
        f"SELECT SUM({REVENUE}) {LINES} WHERE p.category = {{category}} AND o.status = {{status}}",
        "scalar",
    ),
    "orders_city": Kind(
        2,
        "How many orders were placed by customers who live in {city}? Answer with a number.",
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.city = {city}",
        "scalar",
    ),
    "avg_qty": Kind(
        2,
        "What is the average qty of the order_items lines of {category} products? Answer "
        "with a number.",
        "SELECT AVG(oi.qty) FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
        "WHERE p.category = {category}",
        "scalar",
    ),
    "top_city": Kind(
        3,
        "The customers of which city placed the most orders with an order_date from {start} "
        "to {end} inclusive? Answer with the city name.",
        "SELECT c.city, COUNT(*) AS n FROM orders o JOIN customers c "
        "ON c.customer_id = o.customer_id WHERE o.order_date BETWEEN {start} AND {end} "
        "GROUP BY c.city ORDER BY n DESC, c.city LIMIT 2",
        "first",
    ),
    "top_products": Kind(
        3,
        "Which are the {k} products with the most units sold (the sum of qty) in orders with "
        "status {status}? Answer with a list of product names, most units first.",
        f"SELECT p.name, SUM(oi.qty) AS units {LINES} WHERE o.status = {{status}} "
        "GROUP BY p.product_id ORDER BY units DESC, p.name LIMIT {k_plus}",
        "top",
    ),
    "best_rated": Kind(
        3,
        "Which {category} product has the highest average review stars? Answer with the "
        "product name.",
        "SELECT p.name, AVG(r.stars) AS s FROM reviews r JOIN products p "
        "ON p.product_id = r.product_id WHERE p.category = {category} "
        "GROUP BY p.product_id ORDER BY s DESC, p.name LIMIT 2",
        "first",
    ),
    "big_spenders": Kind(
        4,
        "Which customers spent more than {amount} in total (the sum of qty times price) on "
        "orders with an order_date from {start} to {end} inclusive? Answer with a list of "
        "customer names, in any order.",
        f"SELECT c.name, SUM({REVENUE}) AS spend {LINES} JOIN customers c "
        "ON c.customer_id = o.customer_id WHERE o.order_date BETWEEN {start} AND {end} "
        "GROUP BY c.customer_id HAVING spend > {amount} ORDER BY c.name",
        "all",
    ),
    "top_category_month": Kind(
        4,
        "Which product category had the highest revenue (the sum of qty times price) in "
        "orders with an order_date in the month {month} (YYYY-MM)? Answer with the category "
        "name.",
        f"SELECT p.category, SUM({REVENUE}) AS revenue {LINES} "
        "WHERE substr(o.order_date, 1, 7) = {month} "
        "GROUP BY p.category ORDER BY revenue DESC, p.category LIMIT 2",
        "first",
    ),
    "repeat_customers": Kind(
        4,
        "How many customers who live in {city} have at least {n} orders with status "
        "{status}? Answer with a number.",
        "SELECT COUNT(*) FROM (SELECT o.customer_id FROM orders o JOIN customers c "
        "ON c.customer_id = o.customer_id WHERE c.city = {city} AND o.status = {status} "
        "GROUP BY o.customer_id HAVING COUNT(*) >= {n})",
        "scalar",
    ),
}
INT_PARAMS = frozenset({"k", "n", "amount"})

SYSTEM_RULES = (
    "Tools (one call per turn):\n"
    '- sql {"query": string} -> runs one read-only SELECT (or WITH / VALUES) statement and '
    f"returns the columns and at most {MAX_ROWS} rows, cut to {MAX_OBS} characters\n"
    '- answer {"value": number | string | [string]} -> gives the final answer and ends the '
    "conversation\n"
    "\n"
    "SQLite dialect. Dates are ISO text (YYYY-MM-DD): compare them as strings or use substr; "
    "the date and time functions, random() and writes are not available. Numbers are "
    "compared after rounding to 2 decimals, names case-insensitively.\n"
    'Reply with exactly one JSON object per turn and nothing else: {"tool": "<name>", '
    '"args": {...}}. The conversation ends at answer or when the turns run out.'
)

# ---------------------------------------------------------------------------
# Sandbox.

ALLOWED = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
# Nondeterministic, clock-reading or extension functions: replay must be exact.
DENIED_FUNCTIONS = frozenset(
    {
        "changes",
        "current_date",
        "current_time",
        "current_timestamp",
        "date",
        "datetime",
        "julianday",
        "last_insert_rowid",
        "load_extension",
        "random",
        "randomblob",
        "sqlite_compileoption_get",
        "sqlite_compileoption_used",
        "sqlite_source_id",
        "sqlite_version",
        "strftime",
        "time",
        "timediff",
        "total_changes",
        "unixepoch",
    }
)
LIMITS = (
    # ponytail: LENGTH 1000 keeps quadratic string functions (instr, replace, LIKE) under
    # about 0.5 s; raise it only with a wall-clock guard, which would break exact replay.
    (sqlite3.SQLITE_LIMIT_LENGTH, 1000),
    (sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_QUERY),
    (sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 100),
    (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 20),
    (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100),
    (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
)
_LEAD = re.compile(r"(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?(?:\*/|\Z))*", re.S)
_STATEMENT = re.compile(r"(?:select|with|values)\b", re.I)


class _Sandbox:
    """One in-memory database built from the task rows, then locked down for model SQL.

    Sorts, DISTINCT, windows and materialized CTEs spill to temp files (temp_store=FILE,
    a 1 MiB page cache), so an episode holds a few MB at most (measured ~4 MB peak against
    ~32 MB in memory); the step limit bounds the rest.
    """

    def __init__(self, tables: Mapping[str, Mapping[str, Any]]) -> None:
        self.tables = frozenset(tables)
        con = sqlite3.connect(
            ":memory:", isolation_level=None, cached_statements=0, check_same_thread=False
        )
        con.execute("PRAGMA temp_store = FILE")
        con.execute("PRAGMA cache_size = -1024")
        for name in sorted(tables):
            columns = tables[name]["columns"]
            decl = ", ".join(f"{col} {kind}" for col, kind in columns)
            con.execute(f"CREATE TABLE {name} ({decl})")
            marks = ", ".join("?" * len(columns))
            con.executemany(f"INSERT INTO {name} VALUES ({marks})", tables[name]["rows"])  # noqa: S608
        for limit, value in LIMITS:
            con.setlimit(limit, value)
        con.set_authorizer(self._authorize)
        con.set_progress_handler(lambda: 1, MAX_STEPS)
        self.con = con

    def _authorize(
        self, action: int, arg1: str | None, arg2: str | None, db: str | None, view: str | None
    ) -> int:
        if action not in ALLOWED:
            return sqlite3.SQLITE_DENY
        # A CTE read has no database; any other read must hit a task table or the schema
        # (vtabs such as sqlite_stmt or json_each vary between runs or leak addresses).
        if action == sqlite3.SQLITE_READ and db is not None:
            if db != "main" or arg1 not in self.tables | {"sqlite_master"}:
                return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def query(self, sql: str) -> dict[str, Any]:
        """Columns and up to MAX_ROWS rows, or {"error": ...}; never raises on model SQL."""
        if len(sql) > MAX_QUERY:
            return {"error": f"the query is longer than {MAX_QUERY} characters"}
        lead = _LEAD.match(sql)
        if not _STATEMENT.match(sql, lead.end() if lead else 0):
            return {"error": "only one SELECT, WITH or VALUES statement is allowed"}
        try:
            cursor = self.con.execute(sql)
            rows = cursor.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in cursor.description or ()]
        except (sqlite3.Error, ValueError, OverflowError, MemoryError) as error:
            return {"error": str(error) or type(error).__name__}
        return {
            "columns": columns,
            "rows": [[_cell(v) for v in row] for row in rows[:MAX_ROWS]],
            "truncated": len(rows) > MAX_ROWS,
        }


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return "x'" + value.hex() + "'"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _observation(result: Mapping[str, Any]) -> str:
    """The JSON observation, rows dropped from the end until it fits MAX_OBS."""
    shown = dict(result)
    while True:
        raw = json.dumps(shown, sort_keys=True, separators=(",", ":"))
        if len(raw) <= MAX_OBS or not shown.get("rows"):
            return raw[:MAX_OBS]
        shown = {**shown, "rows": shown["rows"][:-1], "truncated": True}


# ---------------------------------------------------------------------------
# Specs, gold and answers.


def _quote(value: str | int) -> str:
    return str(value) if isinstance(value, int) else "'" + value.replace("'", "''") + "'"


def compile_sql(spec: Mapping[str, Any]) -> str:
    params = spec["params"]
    quoted = {name: _quote(value) for name, value in params.items()}
    if "k" in params:
        quoted["k_plus"] = str(params["k"] + 1)
    return KINDS[spec["kind"]].sql.format(**quoted)


def question(spec: Mapping[str, Any]) -> str:
    return KINDS[spec["kind"]].question.format(**spec["params"])


def _extract(spec: Mapping[str, Any], rows: Sequence[Sequence[Any]]) -> Any:
    """The answer carried by the reference query's rows."""
    shape = KINDS[spec["kind"]].answer
    if shape == "scalar":
        return rows[0][0] if rows else None
    if shape == "first":
        return rows[0][0] if rows else None
    if shape == "top":
        return [row[0] for row in rows[: spec["params"]["k"]]]
    return [row[0] for row in rows]


def _usable(spec: Mapping[str, Any], rows: Sequence[Sequence[Any]]) -> bool:
    """A spec is kept only when its answer is non-trivial and unique (no ties that decide)."""
    shape = KINDS[spec["kind"]].answer
    if shape == "scalar":
        return bool(rows) and isinstance(rows[0][0], int | float) and rows[0][0] > 0
    if shape == "first":
        return bool(rows) and rows[0][1] is not None and (len(rows) < 2 or rows[0][1] > rows[1][1])
    if shape == "top":
        k = spec["params"]["k"]
        values = [row[1] for row in rows]
        return len(rows) >= k and all(a > b for a, b in zip(values, values[1:], strict=False))
    return 1 <= len(rows) <= MAX_LIST


def gold(task: Mapping[str, Any]) -> Any:
    """The compiled reference SQL on the task's database: recomputed, never served."""
    result = _Sandbox(task["tables"]).query(compile_sql(task["spec"]))
    if "error" in result:
        raise GeneratorError(f"reference query failed: {result['error']}")
    return _extract(task["spec"], result["rows"])


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value) if abs(value) < 2**1023 else None  # wider ints overflow float()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _same_number(answer: Any, expected: float) -> bool:
    """Within half a cent of the gold: any rounding to 2 decimals (half-up as SQLite's ROUND,
    or half-even) of the exact gold passes, so the grade names no rounding convention."""
    got = _number(answer)
    return got is not None and abs(got - expected) <= 0.005 + 1e-6 * abs(expected) + 1e-9


def _fold(value: Any) -> str | None:
    return value.strip().casefold() if isinstance(value, str) else None


def correct(spec: Mapping[str, Any], answer: Any, expected: Any) -> bool:
    shape = KINDS[spec["kind"]].answer
    if shape == "scalar":
        return _same_number(answer, float(expected))
    if shape == "first":
        return _fold(answer) is not None and _fold(answer) == _fold(expected)
    if not isinstance(answer, list) or not all(isinstance(v, str) for v in answer):
        return False
    got = [v.strip().casefold() for v in answer]
    want = [str(v).strip().casefold() for v in expected]
    return got == want if shape == "top" else sorted(got) == sorted(want)


# ---------------------------------------------------------------------------
# The env.


def system(task: Mapping[str, Any]) -> str:
    tables = task["tables"]
    lines = [
        f"- {name}({', '.join(f'{c} {k}' for c, k in tables[name]['columns'])}): "
        f"{len(tables[name]['rows'])} rows"
        for name in sorted(tables)
    ]
    return "\n".join(
        [
            HEADER,
            "You are a data analyst. Answer the user's question about the SQLite database "
            "below by querying it with the sql tool, then give the exact answer.",
            "",
            "Tables:",
            *lines,
            "",
            SYSTEM_RULES,
        ]
    )


def reset(task: Mapping[str, Any]) -> dict[str, Any]:
    return {"db": _Sandbox(task["tables"]), "answered": False, "answer": None}


def observe(task: Mapping[str, Any], state: Any) -> Content:
    return text(question(task["spec"]))


def step(
    task: Mapping[str, Any], state: dict[str, Any], action: dict[str, Any] | None
) -> tuple[Content, bool]:
    if state["answered"]:
        return text("The conversation is over."), True
    if action is None:
        return text(INVALID), False
    tool, args = action["tool"], action["args"]
    if tool == "sql":
        if not isinstance(args.get("query"), str):
            return text(_observation({"error": "sql needs args: query (str)"})), False
        return text(_observation(state["db"].query(args["query"]))), False
    if tool == "answer":
        if "value" not in args:
            return text(_observation({"error": "answer needs args: value"})), False
        state["answered"], state["answer"] = True, args["value"]
        return text("Answer recorded."), True
    return text(_observation({"error": "unknown tool; use sql or answer"})), False


def loss(task: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    """0 when the answer equals the gold under §5.2 normalisation, else 1."""
    if not state["answered"]:
        return 1.0
    return 0.0 if correct(task["spec"], state["answer"], gold(task)) else 1.0


ENV = Env("sql", system, reset, observe, step, loss)


def buildable(bank: Bank) -> tuple[int, ...]:
    return tuple(LEVELS)


# ---------------------------------------------------------------------------
# Case construction.


def _day(rng: random.Random) -> date:
    return date(YEAR, 1, 1) + timedelta(days=rng.randrange(365))


def _tables(rng: random.Random, spec: Level) -> dict[str, dict[str, Any]]:
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], rng.randint(*spec.customers))
    customers = [
        [i, name, rng.choice(CITIES), (_day(rng) - timedelta(days=365)).isoformat()]
        for i, name in enumerate(names, 1)
    ]
    titles = rng.sample(
        [f"{a} {b}" for a in ADJECTIVES for b in NOUNS], rng.randint(*spec.products)
    )
    products = [
        [100 + i, title, rng.choice(CATEGORIES), rng.randint(5, 200) + rng.choice((0.0, 0.5, 0.99))]
        for i, title in enumerate(titles, 1)
    ]
    orders, lines = [], []
    for oid in range(1000, 1000 + rng.randint(*spec.orders)):
        orders.append([oid, rng.choice(customers)[0], _day(rng).isoformat(), rng.choice(STATUSES)])
        for product in rng.sample(products, rng.randint(1, spec.lines)):
            lines.append([oid, product[0], rng.randint(1, 4)])
    rows = {"customers": customers, "products": products, "orders": orders, "order_items": lines}
    if spec.reviews[1]:
        rows["reviews"] = [
            [rng.choice(products)[0], rng.choice(customers)[0], rng.randint(1, 5)]
            for _ in range(rng.randint(*spec.reviews))
        ]
    return {name: {"columns": [list(c) for c in SCHEMA[name]], "rows": rows[name]} for name in rows}


def _range(rng: random.Random) -> tuple[str, str]:
    start = _day(rng)
    end = min(start + timedelta(days=rng.randint(30, 120)), date(YEAR, 12, 31))
    return start.isoformat(), end.isoformat()


def _params(rng: random.Random, kind: str) -> dict[str, str | int]:
    start, end = _range(rng)
    draw: dict[str, str | int] = {
        "city": rng.choice(CITIES),
        "category": rng.choice(CATEGORIES),
        "status": rng.choice(STATUSES),
        "start": start,
        "end": end,
        "month": f"{YEAR}-{rng.randint(1, 12):02d}",
        "k": rng.randint(3, 5),
        "n": rng.randint(2, 3),
        "amount": rng.choice((100, 200, 300, 400, 500, 750, 1000)),
    }
    names = set(re.findall(r"\{(\w+)\}", KINDS[kind].question))
    return {name: draw[name] for name in sorted(names)}


def make_case(rng: random.Random, level: int, bank: Bank) -> Case:
    """One sql case: a database and a question whose reference answer is unique."""
    spec = LEVELS[level]
    kinds = sorted(k for k, v in KINDS.items() if v.level == level)
    tables = _tables(rng, spec)
    db = _Sandbox(tables)
    for _ in range(ATTEMPTS):
        kind = rng.choice(kinds)
        query = {"kind": kind, "params": _params(rng, kind)}
        result = db.query(compile_sql(query))
        if "error" not in result and _usable(query, result["rows"]):
            task = {"spec": query, "tables": tables}
            body = {
                "harness": "sql",
                "version": 2,
                "task": task,
                "limits": {"turns": TURNS, "max_tokens": MAX_TOKENS},
                "seed": rng.getrandbits(31),
            }
            return Case("sql", level, body, {}, {}, track="sql")
    raise GeneratorError(f"no usable sql spec at level {level} after {ATTEMPTS} attempts")


# ---------------------------------------------------------------------------
# Reference oracle: sees only the OpenAI messages.


def _content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    raise ValueError("unexpected message content")


def _parse_question(message: str) -> dict[str, Any]:
    """The spec behind a template question; ValueError for anything else."""
    for kind in sorted(KINDS):
        template = KINDS[kind].question
        pattern = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>.+?)", re.escape(template))
        match = re.fullmatch(pattern, message)
        if match is None:
            continue
        params: dict[str, str | int] = {}
        for name, value in match.groupdict().items():
            params[name] = int(value) if name in INT_PARAMS and value.isdigit() else value
        spec = {"kind": kind, "params": params}
        if question(spec) == message:
            return spec
    raise ValueError("not a template sql question")


def reference_policy(messages: Sequence[Mapping[str, Any]]) -> str:
    """The next output of a perfect analyst that reads only the conversation: it runs the
    compiled reference query, then answers from the rows it got back."""
    if len(messages) < 2 or not _content(messages[0].get("content")).startswith(HEADER + "\n"):
        raise ValueError("not an sql conversation")
    spec = _parse_question(_content(messages[1].get("content")))
    call = {"tool": "sql", "args": {"query": compile_sql(spec)}}
    for index in range(2, len(messages) - 1, 2):
        if parse_action(_content(messages[index].get("content"))) == call:
            result = json.loads(_content(messages[index + 1].get("content")))
            if "rows" not in result or result.get("truncated"):
                raise ValueError("the reference query did not return its rows")
            value = _extract(spec, result["rows"])
            return json.dumps({"tool": "answer", "args": {"value": value}}, sort_keys=True)
    return json.dumps(call, sort_keys=True)
