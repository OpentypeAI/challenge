import base64
import hashlib
import io
import json
import random
from typing import Any

import pytest
from PIL import Image

from opentype_challenge import harness
from opentype_challenge import paint as p
from opentype_challenge.bank import EMPTY_BANK, Bank, BankItem

DONE = json.dumps({"tool": "done", "args": {}})
DEPICT: dict[str, Any] = {
    "subject": "a lighthouse",
    "brief": "A red and white lighthouse on a rock, with a yellow beam at night.",
    "rubric": [
        "A tall tower stands on a rock.",
        "The tower has red and white bands.",
        "A yellow beam leaves the top of the tower.",
        "The sky is dark.",
    ],
}
DEPICT_BANK = Bank((BankItem.make("depict", DEPICT),))


def draw(*commands):
    return json.dumps({"tool": "draw", "args": {"commands": list(commands)}})


def rect(x, y, w, h, fill="red"):
    return {"op": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill}


def results(checks, *commands):
    return p._results(checks, p._image([p._command(c) for c in commands]))


def state_after(*outputs, task=None):
    task = task or {"mode": "spec", "brief": "", "checks": []}
    state = p.reset(task)
    observations = [p.step(task, state, harness.parse_action(o))[0] for o in outputs]
    return state, observations


def pixels(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def image_of(content) -> Image.Image:
    [part] = [part for part in content if part["type"] == "image_url"]
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return pixels(base64.b64decode(url.split(",", 1)[1]))


async def test_reference_painter_scores_zero_on_300_spec_seeds():
    for level in (1, 2):
        for i in range(150):
            case = p.make_case(random.Random(f"paint|{level}|{i}"), level, EMPTY_BANK)

            async def generate(messages, seed):
                return p.reference_policy(messages)

            outputs = await harness.run_episode(p.ENV, case.body, generate)
            assert outputs[-1] == DONE
            _, loss = harness.replay(p.ENV, case.body, outputs)
            assert loss == 0, (level, i, case.body["task"]["brief"])


def test_blank_canvas_fails_every_check():
    for level in (1, 2):
        for i in range(20):
            case = p.make_case(random.Random(f"blank|{level}|{i}"), level, EMPTY_BANK)
            assert harness.replay(p.ENV, case.body, [DONE])[1] == 1.0


def test_case_shape_is_deterministic_and_serialisable():
    for level in (1, 2):
        a = p.make_case(random.Random(f"det|{level}"), level, EMPTY_BANK)
        b = p.make_case(random.Random(f"det|{level}"), level, EMPTY_BANK)
        assert a == b
        assert (a.family, a.track, a.level, a.gold, a.realized, a.private) == (
            "paint",
            "paint",
            level,
            {},
            {},
            {},
        )
        body = a.body
        assert set(body) == {"harness", "version", "task", "limits", "seed"}
        assert (body["harness"], body["version"]) == ("paint", 2)
        assert body["limits"] == {"turns": 12, "max_tokens": 1024}
        assert body["task"]["mode"] == "spec"
        assert json.loads(json.dumps(body, sort_keys=True)) == body
        assert p._parse(body["task"]["brief"]) == body["task"]["checks"]


def test_levels_shapes_and_relations():
    for i in range(100):
        one = p.make_case(random.Random(f"lvl|1|{i}"), 1, EMPTY_BANK).body["task"]["checks"]
        two = p.make_case(random.Random(f"lvl|2|{i}"), 2, EMPTY_BANK).body["task"]["checks"]
        for checks, low, high in ((one, 1, 2), (two, 3, 6)):
            shapes = sum(c["n"] for c in checks if c["check"] == "count")
            shapes += sum(c["check"] == "cover" for c in checks)
            assert low <= shapes <= high
        relations = {"left_of", "above", "larger", "inside", "apart"}
        assert not any(c["check"] in relations for c in one)
        assert any(c["check"] in relations for c in two)


def test_system_prompt_and_first_observation():
    case = p.make_case(random.Random("sys"), 1, EMPTY_BANK)
    task = case.body["task"]
    prompt = p.system(task)
    assert prompt.split("\n")[0] == "OpenType harness: paint"
    for name, value in p.PALETTE.items():
        assert f"- {name} {value}" in prompt
    for tool in ('"draw"', '"undo"', '"clear"', '"done"', '"op": "polygon"'):
        assert tool in prompt
    first = p.observe(task, p.reset(task))
    assert task["brief"] in first[0]["text"]
    assert image_of(first).getcolors() == [(256 * 256, (255, 255, 255))]
    assert 10 <= len(p.PALETTE) <= 12


def test_wrong_colour_and_wrong_region_fail_the_right_checks():
    checks = [
        {"check": "count", "color": "red", "n": 1},
        {"check": "within", "color": "red", "region": "top-left quadrant"},
        {"check": "shape", "color": "red", "shape": "rectangle"},
        {"check": "size", "color": "red", "lo": 20, "hi": 30},
        {"check": "stray", "colors": ["red"], "max": p.STRAY},
    ]
    assert results(checks, rect(10, 10, 25, 25)) == [True] * 5
    assert results(checks, rect(10, 10, 25, 25, "blue")) == [False] * 5
    assert results(checks, rect(10, 10, 25, 25, "#e03031")) == [False] * 5
    assert results(checks, rect(150, 10, 25, 25)) == [True, False, True, True, True]
    assert results(checks, rect(10, 10, 25, 40)) == [True, True, True, False, True]
    circle = {"op": "circle", "cx": 30, "cy": 30, "r": 12, "fill": "red"}
    assert results(checks, circle) == [True, True, False, True, True]
    touching = [rect(10, 10, 25, 25), rect(35, 35, 25, 25)]  # corner contact: one shape
    assert results(checks[:1], *touching) == [True]
    assert results(checks[:1], rect(10, 10, 25, 25), rect(36, 35, 25, 25)) == [False]
    stray = [rect(10, 10, 25, 25), rect(200, 200, 26, 26, "black")]  # 676 > 655 stray pixels
    assert results(checks[4:], *stray) == [False]
    assert results(checks[4:], rect(10, 10, 25, 25), rect(200, 200, 25, 26, "black")) == [True]


def test_relation_and_shape_checks():
    red, blue = rect(10, 10, 20, 20), rect(100, 100, 30, 30, "blue")
    rel = [
        {"check": "left_of", "a": "red", "b": "blue"},
        {"check": "above", "a": "red", "b": "blue"},
        {"check": "larger", "a": "blue", "b": "red"},
        {"check": "apart", "a": "red"},
        {"check": "inside", "a": "red", "b": "blue"},
    ]
    assert results(rel, red, blue) == [True, True, True, True, False]
    nested = [rect(100, 100, 40, 40, "blue"), rect(110, 110, 10, 10)]
    assert results(rel[3:], *nested) == [False, True]
    assert results(rel[4:], rect(100, 100, 40, 40, "blue"), rect(100, 110, 10, 10)) == [False]
    assert results(rel[:1], red) == [False]  # a named colour without pixels fails
    cover = [{"check": "cover", "color": "green", "region": "bottom half", "lo": 20, "hi": 30}]
    assert results(cover, rect(0, 128, 256, 32, "green")) == [True]  # 25 %
    assert results(cover, rect(0, 128, 256, 24, "green")) == [False]  # 18.75 %
    # enclosed pixels fill the box: a ring is a rectangle, a disc within a disc stays a circle
    shape = [{"check": "shape", "color": "red", "shape": "rectangle"}]
    ring = [rect(10, 10, 40, 40), rect(15, 15, 30, 30, "white"), rect(20, 20, 5, 5)]
    assert results([*shape, {"check": "count", "color": "red", "n": 2}], *ring) == [True, True]
    assert results(shape, *ring[:2]) == [True]
    circle = [{"check": "shape", "color": "red", "shape": "circle"}]
    disc = {"op": "circle", "cx": 60, "cy": 60, "r": 30, "fill": "red"}
    hole = {"op": "circle", "cx": 60, "cy": 60, "r": 10, "fill": "blue"}
    assert results(circle, disc, hole) == [True]
    triangle = {"op": "polygon", "points": [[10, 60], [70, 60], [40, 10]], "fill": "red"}
    assert results([{"check": "shape", "color": "red", "shape": "triangle"}], triangle) == [True]
    assert results(circle, triangle) == [False]


def test_undo_and_clear():
    first, second = draw(rect(0, 0, 10, 10)), draw(rect(50, 50, 10, 10, "blue"))
    state, obs = state_after(first, second, '{"tool": "undo", "args": {}}')
    assert state["draws"] == [[p._command(rect(0, 0, 10, 10))]]
    assert state["used"] == 2
    assert image_of(obs[-1]).tobytes() == pixels(p.render_png([rect(0, 0, 10, 10)])).tobytes()
    assert "Undid" in obs[-1][0]["text"]
    state, obs = state_after(first, second, '{"tool": "clear", "args": {}}', '{"tool": "undo"}')
    assert state["draws"] == [] and state["used"] == 2
    assert image_of(obs[2]).tobytes() == pixels(p.blank_png()).tobytes()
    assert obs[3] == harness.text("Error: there is no draw to undo.")
    task = {"mode": "spec", "brief": "", "checks": []}
    state = p.reset(task)
    assert p.step(task, state, {"tool": "done", "args": {}}) == (
        harness.text("Done: the final canvas is scored."),
        True,
    )
    assert p.step(task, state, None)[1] is True
    assert p.step(task, p.reset(task), None) == (harness.text(harness.INVALID), False)
    assert "unknown tool" in p.step(task, p.reset(task), {"tool": "fill", "args": {}})[0][0]["text"]


def test_invalid_command_rejects_the_whole_draw():
    bad = [
        {"op": "rect", "x": 0, "y": 0, "w": 10, "h": 10},
        {"op": "rect", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "teal"},
        {"op": "rect", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "#12345"},
        {"op": "rect", "x": 0.5, "y": 0, "w": 10, "h": 10, "fill": "red"},
        {"op": "rect", "x": True, "y": 0, "w": 10, "h": 10, "fill": "red"},
        {"op": "rect", "x": 0, "y": 0, "w": 0, "h": 10, "fill": "red"},
        {"op": "rect", "x": 0, "y": 0, "w": 10, "h": 10, "fill": "red", "z": 1},
        {"op": "line", "x0": 0, "y0": 0, "x1": 9, "y1": 9, "width": 1, "fill": "red"},
        {"op": "polygon", "points": [[0, 0], [5, 5]], "fill": "red"},
        {"op": "polygon", "points": [[0, 0], [5, 5], [5]], "fill": "red"},
        {"op": "star", "fill": "red"},
        "rect",
    ]
    for command in bad:
        state, [obs] = state_after(draw(rect(0, 0, 5, 5), command))
        assert state["draws"] == [] and state["used"] == 0
        assert obs[0]["text"].startswith("Error in command 2:") and len(obs) == 1
        with pytest.raises(ValueError):
            p.render_png([command])
    for args in ('{"commands": []}', '{"commands": {}}', "{}", '{"commands": [], "x": 1}'):
        state, [obs] = state_after(f'{{"tool": "draw", "args": {args}}}')
        assert state["used"] == 0 and obs[0]["text"].startswith("Error")
    state, [obs] = state_after(draw(*[rect(0, 0, 1, 1)] * 65))
    assert state["used"] == 0 and obs[0]["text"].startswith("Error")


def test_clamping_and_hex_colours():
    assert p._command(rect(-1000, 5000, 999, 1)) == rect(-64, 320, 320, 1, "#e03030")
    assert p._command(rect(0, 0, 5, 5, "#AbCdEf"))["fill"] == "#abcdef"
    big = [{"op": "circle", "cx": 128, "cy": 128, "r": 10**9, "fill": "blue"}]
    assert pixels(p.render_png(big)).getcolors() == [(256 * 256, (0x30, 0x50, 0xE0))]
    far = [rect(10**6, 10**6, 5, 5)]
    assert p.render_png(far) == p.blank_png()
    line = {"op": "line", "x0": -500, "y0": 128, "x1": 500, "y1": 128, "width": 1, "color": "red"}
    assert pixels(p.render_png([line])).getcolors() == [
        (65280, (255, 255, 255)),
        (256, (224, 48, 48)),
    ]


def test_command_cap_counts_every_accepted_command():
    full = draw(*[rect(0, 0, 1, 1)] * 64)
    state, obs = state_after(*[full] * 6, '{"tool": "undo", "args": {}}')
    assert state["used"] == 384 and len(state["draws"]) == 5
    over = draw(*[rect(0, 0, 1, 1)] * 17)
    state, obs = state_after(
        *[full] * 6, over, draw(*[rect(0, 0, 1, 1)] * 16), draw(rect(0, 0, 1, 1))
    )
    assert state["used"] == 400
    assert obs[6][0]["text"].startswith("Error: this draw passes the limit of 400")
    assert obs[7][0]["text"].startswith("Drew 16 commands.")
    assert obs[8][0]["text"].startswith("Error: this draw passes the limit")


def test_render_is_deterministic_and_aliased():
    commands = [
        rect(-10, -10, 100, 60, "yellow"),
        {"op": "circle", "cx": 128, "cy": 128, "r": 50, "fill": "blue"},
        {"op": "ellipse", "x0": 200, "y0": 10, "x1": 150, "y1": 90, "fill": "#00ff88"},
        {"op": "line", "x0": 0, "y0": 255, "x1": 255, "y1": 0, "width": 5, "color": "black"},
        {"op": "polygon", "points": [[10, 250], [60, 180], [110, 250], [60, 230]], "fill": "pink"},
    ]
    png = p.render_png(commands)
    assert png == p.render_png(commands)
    image = pixels(png)
    assert image.size == (256, 256) and image.mode == "RGB"
    colours = {colour for _, colour in image.getcolors() or ()}
    assert colours == {
        (255, 255, 255),
        (0xF0, 0xD0, 0x20),
        (0x30, 0x50, 0xE0),
        (0x00, 0xFF, 0x88),
        (0, 0, 0),
        (0xF0, 0x70, 0xB0),
    }
    # the pixels are pinned; the PNG bytes may change with zlib
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    assert digest == PINNED_PIXELS
    assert pixels(p.blank_png()).getcolors() == [(256 * 256, (255, 255, 255))]


PINNED_PIXELS = "6e9351e9c91673b2f56b6ee3681f03ba4018ac341529cdfe0866b03d02a79d44"


def test_depict_case_keeps_the_rubric_private():
    assert p.buildable(EMPTY_BANK) == (1, 2)
    assert p.buildable(DEPICT_BANK) == (1, 2, 3)
    bad = Bank((BankItem.make("depict", {**DEPICT, "rubric": ["one"]}),))
    assert p.buildable(bad) == (1, 2)
    with pytest.raises(ValueError):
        p.make_case(random.Random(1), 3, EMPTY_BANK)
    case = p.make_case(random.Random(1), 3, DEPICT_BANK)
    assert case.body["task"] == {"mode": "depict", "brief": DEPICT["brief"]}
    assert case.private == {"brief": DEPICT["brief"], "rubric": DEPICT["rubric"]}
    served = json.dumps(case.body, sort_keys=True)
    assert not any(item in served for item in [*DEPICT["rubric"], p.STANDARD_RUBRIC])
    assert harness.replay(p.ENV, case.body, [draw(rect(0, 0, 9, 9)), DONE])[1] is None
    task = case.body["task"]
    messages = harness.chat_messages(p.ENV, task, [], p.observe(task, p.reset(task)))
    assert DEPICT["brief"] in messages[1]["content"][0]["text"]
    with pytest.raises(ValueError):
        p.reference_policy(messages)
    with pytest.raises(ValueError):
        p.reference_policy([{"role": "system", "content": "OpenType harness: ops\n"}])


def test_judge_request_numbers_the_rubric_and_upscales():
    png = p.render_png([rect(0, 0, 128, 256, "black"), rect(128, 0, 1, 1, "red")])
    system, parts, schema = p.judge_request(DEPICT["brief"], DEPICT["rubric"], png)
    assert "strict" in system and "written words" in system and "instruction" in system
    prompt = parts[0]["text"]
    assert DEPICT["brief"] in prompt
    for i, item in enumerate([*DEPICT["rubric"], p.STANDARD_RUBRIC], 1):
        assert f"\n{i}. {item}\n" in prompt + "\n"
    assert "6. " not in prompt
    image = image_of(parts)
    assert image.size == (512, 512)
    assert image.getpixel((255, 0)) == (0, 0, 0) and image.getpixel((256, 1)) == (224, 48, 48)
    assert image.getpixel((258, 0)) == (255, 255, 255)
    assert len(image.getcolors() or ()) == 3  # nearest neighbour: no blended pixels
    entries = schema["properties"]["items"]
    assert schema["required"] == ["items"] and schema["additionalProperties"] is False
    assert entries["minItems"] == entries["maxItems"] == 5
    assert entries["items"]["properties"]["id"] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert entries["items"]["properties"]["pass"] == {"type": "boolean"}
    assert entries["items"]["required"] == ["id", "pass"]


def verdict(*passes):
    return {"items": [{"id": i, "pass": ok} for i, ok in enumerate(passes, 1)]}


def test_judge_loss_math_and_validation():
    assert p.judge_loss([verdict(True, True, True, True)]) == 0.0
    assert (
        p.judge_loss([verdict(True, False, True, True), verdict(False, False, True, False)]) == 0.5
    )
    assert p.judge_loss([verdict(False, False)], items=2) == 1.0
    shuffled = {"items": [{"id": 2, "pass": True}, {"id": 1, "pass": False}]}
    assert p.judge_loss([shuffled]) == 0.5
    malformed: list[Any] = [
        [],
        [{}],
        [{"items": []}],
        [{"items": [{"id": 1, "pass": True}, {"id": 1, "pass": True}]}],
        [{"items": [{"id": 1, "pass": True}, {"id": 3, "pass": True}]}],
        [{"items": [{"id": 0, "pass": True}]}],
        [{"items": [{"id": True, "pass": True}]}],
        [{"items": [{"id": 1, "pass": "yes"}]}],
        [{"items": [{"id": 1, "pass": True, "why": "x"}]}],
        [{"items": [{"id": "1", "pass": True}]}],
        [verdict(True, True), verdict(True)],
        ["not a verdict"],
    ]
    for bad in malformed:
        with pytest.raises(ValueError):
            p.judge_loss(bad)
    with pytest.raises(ValueError):
        p.judge_loss([verdict(True, True)], items=3)


def test_chat_history_keeps_only_the_latest_image():
    case = p.make_case(random.Random("chat"), 1, EMPTY_BANK)
    task = case.body["task"]
    state = p.reset(task)
    first = p.observe(task, state)
    outputs = [draw(rect(0, 0, 9, 9)), "no json here", draw(rect(20, 20, 9, 9, "blue"))]
    history = [(o, p.step(task, state, harness.parse_action(o))[0]) for o in outputs]
    messages = harness.chat_messages(p.ENV, task, history, first)
    assert [m["role"] for m in messages] == ["system", "user", *["assistant", "user"] * 3]
    images = [
        (i, part)
        for i, m in enumerate(messages)
        if isinstance(m["content"], list)
        for part in m["content"]
        if part["type"] == "image_url"
    ]
    assert [i for i, _ in images] == [7]
    assert {"type": "text", "text": harness.OMITTED} in messages[1]["content"]
    assert {"type": "text", "text": harness.OMITTED} in messages[3]["content"]
    assert messages[5]["content"] == harness.text(harness.INVALID)
    expected = p.render_png([rect(0, 0, 9, 9), rect(20, 20, 9, 9, "blue")])
    assert image_of(messages[7]["content"]).tobytes() == pixels(expected).tobytes()


def brute_shapes(mask: Image.Image) -> list[tuple[int, int, int]]:
    """The prompt's definitions, literally: 8-connected shapes; a shape fills its own pixels
    plus every pixel that cannot reach the canvas edge by edge steps without crossing it."""
    size = mask.size[0]
    on = {(x, y) for y in range(size) for x in range(size) if mask.getpixel((x, y))}
    seen, out = set(), []
    for start in sorted(on):
        if start in seen:
            continue
        shape, todo = {start}, [start]
        while todo:
            x, y = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in on and q not in shape:
                        shape.add(q)
                        todo.append(q)
        seen |= shape
        edge = [(x, y) for x in range(size) for y in (0, size - 1)]
        edge += [(x, y) for y in range(size) for x in (0, size - 1)]
        reached = {q for q in edge if q not in shape}
        todo = list(reached)
        while todo:
            x, y = todo.pop()
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= q[0] < size and 0 <= q[1] < size and q not in shape and q not in reached:
                    reached.add(q)
                    todo.append(q)
        xs, ys = [x for x, _ in shape], [y for _, y in shape]
        out.append((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, size * size - len(reached)))
    return sorted(out)


def test_shape_labelling_matches_the_literal_definition(monkeypatch):
    monkeypatch.setattr(p, "SIZE", 40)
    rng = random.Random("labels")
    for _ in range(60):
        commands = []
        for _ in range(rng.randint(1, 8)):
            fill = rng.choice(("red", "blue", "white"))
            kind = rng.choice(("rect", "circle", "polygon", "line"))
            if kind == "rect":
                commands.append(
                    rect(
                        rng.randint(-3, 38),
                        rng.randint(-3, 38),
                        rng.randint(1, 30),
                        rng.randint(1, 30),
                        fill,
                    )
                )
            elif kind == "circle":
                commands.append(
                    {
                        "op": "circle",
                        "cx": rng.randint(0, 39),
                        "cy": rng.randint(0, 39),
                        "r": rng.randint(1, 15),
                        "fill": fill,
                    }
                )
            elif kind == "polygon":
                points = [
                    [rng.randint(-3, 42), rng.randint(-3, 42)] for _ in range(rng.randint(3, 6))
                ]
                commands.append({"op": "polygon", "points": points, "fill": fill})
            else:
                ends = {k: rng.randint(-3, 42) for k in ("x0", "y0", "x1", "y1")}
                commands.append({"op": "line", **ends, "width": rng.randint(1, 4), "color": fill})
        image = p._image([p._command(c) for c in commands])
        mask = p._mask(image.split(), p.PALETTE["red"])
        stats = p._stats(mask)
        assert list(stats.shapes) == brute_shapes(mask), commands
        on = [(x, y) for y in range(40) for x in range(40) if mask.getpixel((x, y))]
        assert (stats.area, stats.sx, stats.sy) == (
            len(on),
            sum(x for x, _ in on),
            sum(y for _, y in on),
        )


def test_step_never_raises_on_junk():
    rng = random.Random("junk")
    task = {"mode": "spec", "brief": "", "checks": []}
    values: list[Any] = [None, True, 1.5, "x", [], {}, [1, 2], 10**30, -(10**30), "#123456", "red"]
    keys = "op x y w h cx cy r x0 y0 x1 y1 width fill color points".split()
    for _ in range(300):
        command: dict[str, Any] = {
            k: rng.choice(values + [rng.randint(-500, 500)])
            for k in rng.sample(keys, rng.randint(0, 8))
        }
        command["op"] = rng.choice(["rect", "circle", "ellipse", "line", "polygon", "blob", 3])
        state = p.reset(task)
        observation, done = p.step(task, state, {"tool": "draw", "args": {"commands": [command]}})
        assert not done and observation[0]["type"] == "text"
