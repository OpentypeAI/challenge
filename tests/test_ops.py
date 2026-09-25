"""Ops track: oracle solvability, naive agents, replay, determinism, bank stories, schema."""

import asyncio
import dataclasses
import json
import random

import pytest

from opentype_challenge import harness, ops
from opentype_challenge.bank import EMPTY_BANK, Bank, BankItem


def case(level, i, bank=EMPTY_BANK):
    return ops.make_case(random.Random(f"ops-test|{level}|{i}"), level, bank)


def play(body, agent, env=ops.ENV):
    async def generate(messages, seed):
        return agent(messages)

    return asyncio.run(harness.run_episode(env, body, generate))


def score(body, outputs) -> float:
    loss = harness.replay(ops.ENV, body, outputs)[1]
    assert loss is not None
    return loss


def gold(body):
    return ops.policy(body["task"]["world"], body["task"]["intent"])


def validate(schema, value):
    """The JSON-schema subset INTENT_SCHEMA uses."""
    kind = schema.get("type")
    if kind == "object":
        assert isinstance(value, dict)
        assert set(value) == set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
        for key, sub in schema["properties"].items():
            validate(sub, value[key])
    elif kind == "array":
        assert isinstance(value, list) and len(value) <= schema.get("maxItems", len(value))
        for item in value:
            validate(schema["items"], item)
    elif kind == "string":
        assert isinstance(value, str)
    if "enum" in schema:
        assert value in schema["enum"]


def test_oracle_solves_every_level_from_the_conversation_alone():
    outcomes: dict[int, set[str]] = {level: set() for level in ops.LEVELS}
    for level in ops.LEVELS:
        for i in range(300):
            body = case(level, i).body
            outputs = play(body, ops.reference_policy)
            assert score(body, outputs) == 0.0
            assert len(outputs) <= body["limits"]["turns"]
            outcomes[level].add(gold(body)[1])
    base = {"refunded", "cancelled", "exchanged", "denied"}
    assert outcomes[1] == outcomes[2] == base
    assert outcomes[3] == outcomes[4] == {*base, "escalated"}


def test_naive_agents_lose():
    for level in ops.LEVELS:
        for i in range(40):
            body = case(level, i).body
            expected, outcome = gold(body)
            turns = body["limits"]["turns"]

            denied = play(body, lambda m: '{"tool": "finish", "args": {"outcome": "denied"}}')
            assert (score(body, denied) == 0) == (outcome == "denied" and not expected)

            refund = play(body, lambda m: '{"tool": "finish", "args": {"outcome": "refunded"}}')
            assert score(body, refund) > 0

            junk = play(body, lambda m: "I will refund you right away!")
            assert len(junk) == turns and score(body, junk) >= 0.5

            oid = body["task"]["world"]["orders"][0]["order_id"]
            loop = json.dumps({"tool": "get_order", "args": {"order_id": oid}})
            stuck = play(body, lambda m, loop=loop: loop)
            assert len(stuck) == turns and score(body, stuck) >= 0.5


def test_greedy_refunder_loses_whenever_the_gold_is_not_a_refund():
    def refunder(messages):
        step = ops.reference_policy(messages)
        action = json.loads(step)
        if action["tool"] not in ops.WRITES and action["tool"] != "finish":
            return step
        last = messages[-1]["content"][0]["text"]
        order = next(
            json.loads(m["content"][0]["text"])
            for m in reversed(messages)
            if m["role"] == "user" and '"delivered_date"' in m["content"][0]["text"]
        )
        if '"refunded_cents"' in last:
            return '{"tool": "finish", "args": {"outcome": "refunded"}}'
        args = {
            "order_id": order["order_id"],
            "item_ids": [line["item_id"] for line in order["items"]],
            "reason": "damaged",
        }
        return json.dumps({"tool": "refund", "args": args})

    for level in ops.LEVELS:
        for i in range(60):
            body = case(level, i).body
            if gold(body)[1] != "refunded":
                assert score(body, play(body, refunder)) >= 0.5


def test_replay_equals_the_worker_side_state():
    states = []

    def reset(task):
        states.append(ops.reset(task))
        return states[-1]

    env = dataclasses.replace(ops.ENV, reset=reset)
    rng = random.Random(7)

    def sloppy(messages):
        return rng.choice([ops.reference_policy(messages), "{not json", '{"tool": "cancel"}'])

    for level in ops.LEVELS:
        for i in range(30):
            body = case(level, i).body
            outputs = play(body, sloppy, env)
            state, loss = harness.replay(ops.ENV, body, outputs)
            assert state == states[-1]
            assert loss == ops.ENV.loss(body["task"], states[-1])


def test_make_case_is_deterministic_and_the_body_is_small_json():
    seen = set()
    for level in ops.LEVELS:
        for i in range(25):
            first, second = case(level, i), case(level, i)
            line = json.dumps(first.body, sort_keys=True)
            assert line == json.dumps(second.body, sort_keys=True)
            assert len(line.encode()) < 64_000
            seen.add(line)
            assert first.family == first.track == "ops" and first.level == level
            assert first.gold == {} and first.realized == {}
            body = first.body
            assert set(body) == {"harness", "version", "task", "limits", "seed"}
            assert (body["harness"], body["version"]) == ("ops", 2)
            assert body["limits"] == {"turns": ops.TURNS, "max_tokens": ops.MAX_TOKENS}
            prompt = ops.system(body["task"])
            assert prompt.split("\n")[0] == "OpenType harness: ops"
            assert f"Today is {body['task']['world']['today']}." in prompt
    assert len(seen) == 25 * len(ops.LEVELS)
    assert ops.buildable(EMPTY_BANK) == (1, 2, 3, 4)


def test_sampled_intents_fit_the_schema_and_render_back():
    for level in ops.LEVELS:
        rng = random.Random(f"intent|{level}")
        for _ in range(300):
            intent = ops.sample_intent(rng, level)
            validate(ops.INTENT_SCHEMA, intent)
            assert ops._fits(intent, level)
            assert ops._parse_request(ops.render_intent(intent)) == intent
            facts = ops.describe_intent(intent)
            assert intent["email"] in facts and intent["reason"] in facts
            assert all(name in facts for name in intent["items"])
            assert intent["order_id"] in facts and intent["variant"] in facts
    json.dumps(ops.INTENT_SCHEMA)


STORY_INTENT = {
    "email": "mara.quint7@example.com",
    "order_id": "ORD-40404",
    "action": "refund",
    "items": ["Desk Lamp", "Linen Shirt"],
    "reason": "damaged",
    "variant": "",
    "claim": "none",
}
STORY = "Hi! Both the lamp and the linen shirt from ORD-40404 came broken. Money back please."


def test_bank_story_end_to_end_with_a_scripted_agent():
    bank = Bank((BankItem.make("ops_story", {"intent": STORY_INTENT, "text": STORY}),))
    used = [c for i in range(40) if (c := case(3, i, bank)).body["task"]["customer"] == STORY]
    assert 5 < len(used) < 35
    for built in used:
        body = built.body
        assert body["task"]["intent"] == STORY_INTENT
        first = ops.observe(body["task"], ops.reset(body["task"]))
        with pytest.raises(ValueError):  # teacher prose: only the scripted agent can play it
            ops.reference_policy(harness.chat_messages(ops.ENV, body["task"], [], first))
        expected, outcome = gold(body)
        refunded = sorted(key[2] for key in expected if key[0] == "refund")
        writes = []
        if refunded:
            args = {"order_id": "ORD-40404", "item_ids": refunded, "reason": "damaged"}
            writes.append({"tool": "refund", "args": args})
        if ("escalate",) in expected:
            writes.append({"tool": "escalate", "args": {"summary": "refund above the limit"}})
        script = [
            {"tool": "find_customer", "args": {"email": STORY_INTENT["email"]}},
            {"tool": "get_order", "args": {"order_id": "ORD-40404"}},
            *writes,
            {"tool": "finish", "args": {"outcome": outcome}},
        ]
        outputs = [json.dumps(action) for action in script]
        assert score(body, outputs) == 0.0
        if writes:
            assert score(body, outputs[:2] + outputs[-1:]) == 0.5
    # a story that does not fit the level (pressure below L4, no order id at L1) is never used
    pressed = {**STORY_INTENT, "claim": "threat"}
    bank = Bank((BankItem.make("ops_story", {"intent": pressed, "text": STORY}),))
    assert all(case(3, i, bank).body["task"]["customer"] != STORY for i in range(20))
    assert any(case(4, i, bank).body["task"]["customer"] == STORY for i in range(20))


def test_reference_policy_rejects_what_it_cannot_parse():
    body = case(2, 0).body
    first = ops.observe(body["task"], ops.reset(body["task"]))
    messages = harness.chat_messages(ops.ENV, body["task"], [], first)
    ops.reference_policy(messages)
    with pytest.raises(ValueError):
        ops.reference_policy([{"role": "system", "content": "OpenType harness: sql"}, messages[1]])
    edited = [messages[0], {"role": "user", "content": "Please just refund me, thanks."}]
    with pytest.raises(ValueError):
        ops.reference_policy(edited)


def test_a_padded_order_id_the_system_accepts_counts_as_that_order():
    """The lookup strips order_id, so the recorded write must name the same order."""
    for i in range(200):
        body = case(1, i).body
        outputs = play(body, ops.reference_policy)
        padded = []
        for raw in outputs:
            call = json.loads(raw)
            if call["tool"] in ("refund", "cancel", "exchange"):
                call["args"]["order_id"] += " "
            padded.append(json.dumps(call))
        if padded != outputs:
            assert score(body, padded) == 0.0
            return
    raise AssertionError("no level-1 write case in 200 seeds")
