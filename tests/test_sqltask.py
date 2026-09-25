"""SQL track: oracle solvability, wrong answers, sandbox refusals and bombs, determinism."""

import asyncio
import json
import random
import time

import pytest

from opentype_challenge import harness, sqltask
from opentype_challenge.bank import EMPTY_BANK


def case(level, i):
    return sqltask.make_case(random.Random(f"sql-test|{level}|{i}"), level, EMPTY_BANK)


def play(body, agent):
    async def generate(messages, seed):
        return agent(messages)

    return asyncio.run(harness.run_episode(sqltask.ENV, body, generate))


def score(body, outputs):
    return harness.replay(sqltask.ENV, body, outputs)[1]


def answer(value):
    return json.dumps({"tool": "answer", "args": {"value": value}})


def sandbox(level=3):
    return sqltask._Sandbox(case(level, 0).body["task"]["tables"])


def test_oracle_solves_every_level_from_the_conversation_alone():
    kinds = set()
    for level in sqltask.buildable(EMPTY_BANK):
        for i in range(300):
            body = case(level, i).body
            outputs = play(body, sqltask.reference_policy)
            assert score(body, outputs) == 0.0
            assert len(outputs) == 2
            kinds.add(body["task"]["spec"]["kind"])
    assert kinds == set(sqltask.KINDS)


def test_wrong_answers_lose():
    for level in sqltask.LEVELS:
        for i in range(40):
            body = case(level, i).body
            gold = sqltask.gold(body["task"])
            wrong: object = gold + 1 if isinstance(gold, int | float) else "no such value 0x7f3a"
            if isinstance(gold, list):
                wrong = gold[:-1] if len(gold) > 1 else ["no such value"]
            assert score(body, [answer(wrong)]) == 1.0
            assert score(body, [answer(gold)]) == 0.0
            assert score(body, ["I think it is 42."] * 10) == 1.0
            assert score(body, []) == 1.0


def test_answer_normalisation():
    spec = {"kind": "count_city", "params": {"city": "Oslo"}}
    assert sqltask.correct(spec, "12", 12) and sqltask.correct(spec, 12.004, 12)
    assert not sqltask.correct(spec, 12.01, 12) and not sqltask.correct(spec, True, 1)
    first = {"kind": "top_city", "params": {}}
    assert sqltask.correct(first, "  oslo ", "Oslo") and not sqltask.correct(first, 3, "Oslo")
    top = {"kind": "top_products", "params": {"k": 2}}
    assert sqltask.correct(top, ["a", "B"], ["A", "b"]) and not sqltask.correct(
        top, ["b", "a"], ["a", "b"]
    )
    unordered = {"kind": "big_spenders", "params": {}}
    assert sqltask.correct(unordered, ["b", "a"], ["a", "b"])
    assert not sqltask.correct(unordered, ["a"], ["a", "a"])


def test_scalar_accepts_any_two_decimal_rounding():
    spec = {"kind": "avg_qty", "params": {"category": "toys"}}
    for gold in (2.125, 2.625, 3.125):
        low, high = gold - 0.005, gold + 0.005
        for got in (low, high, gold, f"{high:.2f}"):
            assert sqltask.correct(spec, got, gold), (got, gold)
        assert not sqltask.correct(spec, gold + 0.01, gold)
        assert not sqltask.correct(spec, gold - 0.0151, gold)


@pytest.mark.parametrize(
    "query",
    [
        "ATTACH ':memory:' AS x",
        "PRAGMA table_info(orders)",
        "SELECT * FROM pragma_table_info('orders')",
        "DELETE FROM orders",
        "INSERT INTO orders VALUES (1, 1, '2025-01-01', 'x')",
        "CREATE TABLE x (a)",
        "DROP TABLE orders",
        "BEGIN",
        "SAVEPOINT a",
        "VACUUM",
        "ANALYZE",
        "EXPLAIN SELECT 1",
        "SELECT load_extension('x')",
        "SELECT random()",
        "SELECT hex(randomblob(4))",
        "SELECT date('now')",
        "SELECT current_timestamp",
        "SELECT sqlite_version()",
        "SELECT * FROM sqlite_stmt",
        "SELECT * FROM temp.sqlite_master",
        "SELECT * FROM json_each('[1]')",
        "SELECT 1; DROP TABLE orders",
        "SELECT 'a\x00b'",
        "SELECT '\ud800'",
        "SELECT '" + "x" * 4000 + "'",
    ],
)
def test_sandbox_refuses(query):
    result = sandbox().query(query)
    assert set(result) == {"error"}


def test_sandbox_allows_analysis():
    db = sandbox()
    for query in [
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id",
        "WITH x AS (SELECT name FROM customers) SELECT * FROM x",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r LIMIT 5) SELECT * FROM r",
        "SELECT name, RANK() OVER (ORDER BY price DESC) FROM products",
        "SELECT 1 UNION SELECT 2",
        "VALUES (1), (2)",
        "/* a comment */ select name from sqlite_master",
        "SELECT substr(order_date, 1, 7), COUNT(*) FROM orders GROUP BY 1",
    ]:
        assert "rows" in db.query(query), query
    assert db.query("SELECT * FROM orders")["truncated"] is True
    assert len(db.query("SELECT * FROM orders")["rows"]) == sqltask.MAX_ROWS


@pytest.mark.parametrize(
    "query",
    [
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r",
        "SELECT COUNT(*) FROM orders a, orders b, orders c, orders d",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r LIMIT 5000000) "
        "SELECT n FROM r ORDER BY -n LIMIT 1",
        "SELECT group_concat(a.name) FROM customers a, customers b, customers c",
        "SELECT instr(replace(hex(zeroblob(500)), '0', 'ab'), 'x')",
        "SELECT COUNT(*) FROM customers a, customers b WHERE a.name || b.name "
        "LIKE '%a%a%a%a%a%a%a%a%a%a%a%a%a%a%a%z'",
        "SELECT printf('%.*c', 100000000, 'x')",
    ],
)
def test_bombs_stop_fast(query):
    db = sandbox(4)
    start = time.monotonic()
    result = db.query(query)
    assert time.monotonic() - start < 2.0
    assert "error" in result or len(json.dumps(result)) < 10_000


def test_a_full_transcript_of_bombs_replays_fast():
    query = (
        "SELECT DISTINCT printf('%.990c','x')||a.rowid||b.rowid||c.rowid FROM order_items a, "
        "order_items b, order_items c LIMIT 1 OFFSET 5000000"
    )
    outputs = [json.dumps({"tool": "sql", "args": {"query": query}})] * sqltask.TURNS
    start = time.monotonic()
    assert score(case(4, 0).body, outputs) == 1.0
    assert time.monotonic() - start < 1.5  # 2.6 s at the old 2M-step budget


def test_observation_is_bounded():
    body = case(4, 0).body
    long_rows = json.dumps({"tool": "sql", "args": {"query": "SELECT * FROM customers"}})
    state = sqltask.reset(body["task"])
    obs, done = sqltask.step(body["task"], state, harness.parse_action(long_rows))
    assert not done and len(obs[0]["text"]) <= sqltask.MAX_OBS
    json.loads(obs[0]["text"])


def test_determinism_and_size():
    for level in sqltask.LEVELS:
        for i in range(30):
            a, b = case(level, i), case(level, i)
            raw = json.dumps(a.body, sort_keys=True)
            assert raw == json.dumps(b.body, sort_keys=True)
            assert len(raw.encode()) < 200_000
            assert a.track == "sql" and a.gold == {} and a.private == {}
            system = sqltask.system(a.body["task"])
            assert system.startswith("OpenType harness: sql\n")
            assert "gold" not in raw and 'answer":' not in raw


def test_reference_policy_refuses_prose():
    body = case(1, 0).body
    messages = harness.chat_messages(
        sqltask.ENV, body["task"], [], harness.text("How much is the fish?")
    )
    with pytest.raises(ValueError):
        sqltask.reference_policy(messages)
