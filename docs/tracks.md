# OpenType v2: tracks, teacher, harness (build contract)

Status: the binding contract for v2, kept in step with the code of 2.0.0. Every module below
implements exactly these names and shapes. Where this file and older docs disagree, this file
wins; where it and the code disagree, the code is the reference and this file is fixed.

v1 measured one skill: typed decisions on short, template-rendered records. v2 measures
whether a DiffusionGemma checkpoint is usable in production:

| track | what it measures | gold | scorer |
| --- | --- | --- | --- |
| `decisions` | typed decisions over rules, public and sealed record families, template or teacher prose | exact posterior | half-Brier (v1) |
| `longctx` | the same reads over 8k–128k-token dossiers with amendments and near-duplicate records | exact posterior | half-Brier |
| `ops` | a multi-turn customer-operations agent with tools and a written policy | exact final state + outcome | state diff |
| `sql` | a multi-turn analyst agent querying a sandboxed SQLite database | exact answer | exact match |
| `paint` | a multi-turn painter that sees its canvas after every turn | `spec`: exact pixel checks; `depict`: VLM rubric | checks / judge |

A teacher LLM (`cx/gpt-6-sol` through the operator gateway) writes the private,
per-window content: sealed record families, realistic prose records, customer stories and
drawing briefs with rubrics. The teacher never decides a gold label. Every teacher text is
admitted only when two extractor models of different families recover its exact structured
spec (round trip). Picture tasks are graded by a VLM judge on the container's own render.

## 1. Module ownership and public names

| module | owns | public names (exact) |
| --- | --- | --- |
| `generator.py` | decisions track, families | existing names + `Case.track`, `family_from_json`, `family_to_json`, `sample_known`, `make_case(rng, family, level, prose=None)`, `solve_known(body, known)` |
| `longctx.py` (new) | long-context track | `LEVELS`, `buildable(bank)`, `make_case(rng, level, bank)`, `solve(body)` |
| `harness.py` (new, written) | agent loop, replay, action parsing | `Env`, `parse_action`, `run_episode(env, body, generate)`, `replay(env, body, outputs)`, `chat_messages(env, task, history)`, `MAX_OUTPUT_CHARS`, `INVALID` |
| `ops.py` (new) | ops env | `ENV`, `LEVELS`, `buildable(bank)`, `make_case(rng, level, bank)`, `sample_intent(rng, level)`, `INTENT_SCHEMA`, `describe_intent(intent)`, `render_intent(intent)`, `reference_policy(messages)` |
| `sqltask.py` (new) | sql env | `ENV`, `LEVELS`, `buildable(bank)`, `make_case(rng, level, bank)`, `reference_policy(messages)` |
| `paint.py` (new) | paint env, render, checks, judge prompt | `ENV`, `LEVELS`, `buildable(bank)`, `make_case(rng, level, bank)`, `render_png(commands)`, `blank_png()`, `judge_request`, `judge_loss`, `reference_policy(messages)`, `STANDARD_RUBRIC` |
| `teacher.py` (new) | gateway client, bank builder, judge | `Gateway`, `GatewayError`, `TeacherConfig`, `DEFAULT_TARGETS`, `build_bank`, `round_trip`, `judge_png`, `BANK_KINDS` (re-exported from `bank.py`) |
| `scoring.py` | per-case loss, duel statistics | existing names + `Paired.track`, `harness_score`, `verdict(pairs, retired, stopped, weights)` |
| `tracks.py` (new) | track plan and dispatch | `TRACKS`, `ENVS` (`{"ops": ops.ENV, "sql": sqltask.ENV, "paint": paint.ENV}`), `BUILDERS`, `TrackPlan`, `DEFAULT_PLAN`, `effective_plan(plan, bank, judge=True)`, `track_of`, `job_case(seed, plan, mix, index, bank, judge=True)`, `score_item`, `solve_body(body)` |
| `bank.py` | seeds, windows, bank snapshots | existing names + `BANK_KINDS`, `BankItem`, `Bank`, `EMPTY_BANK`, `bank_digest`, `drand_beacon`, `job_seed(secret, job_id, digest, beacon=None)` |
| `store.py`, `app.py`, `cli.py` | state, routes, CLI | see §8 |
| `worker.py` | B300 duel worker | see §7 |

`generator.Case` gains one field with a default, so v1 call sites keep working:

```python
@dataclass(frozen=True)
class Case:
    family: str  # family name, or the env name for harness tracks ("ops", "sql", "paint")
    level: int
    body: dict[str, Any]  # exactly what the worker receives
    gold: dict[str, Gold]  # read tracks; {} for harness tracks
    realized: dict[str, int]  # tests only, never served
    track: str = "decisions"
    # never served: depict {"brief", "rubric"}, oracle aids for tests
    private: dict[str, Any] = field(default_factory=dict)
```

`body["task"]` of a harness case may hold the env's hidden state (the ops world, the SQL
rows, the paint spec). The worker host sees it and the model never does: the model sees
only `system(task)` and the observations. Anything that must not leave the container, such
as a depict rubric, lives in `Case.private`, which the container rebuilds from the seed and
the bank.

Every `make_case` is a pure function of `(rng, level, bank)`. The same inputs give the same
`Case` in the container and in `opentype-challenge audit`.

## 2. Bank: teacher content per window

A bank is the frozen list of teacher items of one window. It is private while the window
is open and published when the window closes (`GET /v1/windows/{id}/bank`).

```python
@dataclass(frozen=True)
class BankItem:
    kind: str            # one of BANK_KINDS
    key: str             # stable id, unique per (window, kind): sha256 of canonical payload, 16 hex
    payload: dict[str, Any]

BANK_KINDS = ("family", "prose", "ops_story", "depict")

class Bank:                             # bank.py
    items: tuple[BankItem, ...]
    def of(self, kind: str) -> tuple[BankItem, ...]   # stable order: by key
    def families(self) -> tuple[Family, ...]          # sealed families, parsed
    @property
    def digest(self) -> str                           # bank_digest(items)
EMPTY_BANK = Bank(())
```

`bank_digest(items)` is sha256 over `json.dumps([kind, key, payload], sort_keys=True,
separators=(",", ":")) + "\n"` for items sorted by `(kind, key)`.

Payloads (the teacher builder writes them, the tracks read them; nothing else):

```jsonc
// family: a sealed record family, same shape family_to_json emits
{"name": "sealed_<8 hex>", "title": "...", "subject": "...",
 "facts": [{"name": "snake_case", "label": "lower case words", "domain": ["a", "b"] | [0, 5, 10], "unit": ""}],
 "derived": [{"name": "snake_case", "domain": ["x", "y", "z"]}],
 "questions": [{"id": "snake_case", "kind": "choice|noul|score", "prompt": "...?", "options": ["..."]}]}
// constraints (family_from_json raises GeneratorError otherwise): 6..12 facts, each domain 2..8
// distinct values, all str (snake_case) or all int ascending; sealed labels are lower case
// (public families keep their v1 labels, e.g. "time left on the SLA"); 1..3 derived with 2..4 values;
// 6..12 questions with every kind present; choice pools 4..26 snake_case options; score 3..6
// ordered levels; noul options are exactly ["yes", "no"]; names unique across facts, derived
// and questions; labels unique; the template renderer must stay unambiguous.

// prose: a realistic record written by the teacher for one fact vector
{"family": "<public or sealed family name>", "known": {"fact": value}, "hidden": ["fact"],
 "text": "the record as prose", "style": "email|chat|form|log|note"}

// ops_story: a customer message for one ops intent
{"intent": {...ops.INTENT_SCHEMA...}, "text": "the customer's message"}

// depict: a drawing brief
{"subject": "short noun phrase", "brief": "one to three sentences",
 "rubric": ["checkable visual statement", "... 4..8 items"]}
```

Round trip (`teacher.round_trip`): each extractor model reads only the text plus the public
schema and must return the exact spec. For prose, that is every fact of the family, as its
domain value or `"not_stated"`, and it must equal `known` plus `hidden → not_stated`. For
ops stories, that is the exact intent object. An item is kept only when **every** extractor
agrees exactly. Any disagreement discards the item. The teacher is never asked to label an
answer.

Depict negative control: a rubric is kept only when the judge fails **every payload item** on a
blank white canvas (`STANDARD_RUBRIC` is excluded from the control, a blank canvas passes it). `paint.STANDARD_RUBRIC` ("The picture contains no letters, words or
numbers.") is appended to every rubric at scoring time and is not part of the payload.

Build targets per window (`TeacherConfig.targets`, env `OPENTYPE_BANK_TARGETS` as JSON):
`{"family": 4, "prose": 600, "ops_story": 200, "depict": 80}`. Prose covers public and
sealed families. The builder sizes its work from these targets and runs with bounded
concurrency (`OPENTYPE_TEACHER_CONCURRENCY`, default 8).

## 3. Gateway (`teacher.py`)

```python
@dataclass(frozen=True)
class TeacherConfig:
    url: str                           # OPENTYPE_TEACHER_URL, e.g. https://gateway.example
    token_file: Path                   # OPENTYPE_TEACHER_TOKEN_FILE, default /run/secrets/teacher.token
    teacher: str = "cx/gpt-6-sol"      # OPENTYPE_TEACHER_MODEL
    extractors: tuple[str, ...] = ("cx/gpt-5.6-luna", "cc/claude-sonnet-5")  # OPENTYPE_EXTRACTOR_MODELS (comma list)
    judges: tuple[str, ...] = ("cx/gpt-6-sol",)                            # OPENTYPE_JUDGE_MODELS
    concurrency: int = 8
    targets: Mapping[str, int] = ...   # §2
    @classmethod
    def from_env(cls) -> TeacherConfig | None   # None when OPENTYPE_TEACHER_URL is unset

class GatewayError(Exception): retry: bool

class Gateway:
    def __init__(self, config: TeacherConfig, transport: httpx.AsyncBaseTransport | None = None,
                 *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep)  # tests skip backoff
    async def json(self, model: str, system: str, user: str | list[dict], schema: dict,
                   *, max_tokens: int = 4000, temperature: float = 0.0) -> dict
    async def aclose(self) -> None
```

- `POST {url}/v1/chat/completions`, `Authorization: Bearer <token>`,
  `User-Agent: opentype-challenge/<version>`, 180 s timeout. Retry up to 6 times with
  exponential backoff on transport errors, 408, 409, 429 and 5xx. Other 4xx raise
  `GatewayError(retry=False)`.
- The token is read from `token_file` on every call and never logged, echoed or put into
  an exception message.
- Structured output: for `cx/*` models, send `response_format: {type: json_schema,
  json_schema: {name: "submit", strict: true, schema}}`. For other models, send one tool
  `submit` with the schema as `parameters`, `tool_choice: "auto"`, and the system line "Call
  the submit tool exactly once." Read `tool_calls[0].function.arguments`. Fall back to the
  first JSON object in `content`.
- The response is either one JSON object, possibly followed by `data: [DONE]`, or an SSE
  stream of `chat.completion.chunk` events even without `stream: true`. Handle both:
  concatenate `delta.content` and each `tool_calls[i].function.arguments` in order.
- Validate the result against the schema with a small built-in validator (the subset used
  here: `type`, `properties`, `required`, `additionalProperties: false`, `enum`, `items`,
  `minItems`, `maxItems`, `minimum`, `maximum`). An invalid result counts as one failed
  attempt, and the call retries up to 3 times.
- `user` can be a list of OpenAI content parts. Images are
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.

`build_bank(gateway, config, rng, targets, *, families) -> list[BankItem]` builds the four kinds.
`families` holds the public families. The sealed families are built first, so prose can
cover them. It logs counts and discard reasons without logging text or tokens. A failed
item is skipped. It never raises for one item. It raises `GatewayError` only when nothing
at all could be built for a kind with a non-zero target.

## 4. Read tracks

### decisions (generator.py)

`make_case(rng, family, level, prose=None)`. When `prose` (a `prose` payload whose family
is `family`) is given, `known` comes from `prose["known"]` and the state text is
`prose["text"]`. The world's hidden facts are drawn uniformly as in v1, so the gold
posterior uses exactly `known`. Rules, questions, probes, derived facts and gold are
built as in v1. The round-trip check of v1 applies only to template states, since prose
was round-tripped when the bank was built.

- `sample_known(rng, family, hidden_max) -> tuple[dict[str, Value], list[str]]` draws a
  world and a hidden set with the v1 distribution and returns `(known, hidden)`.
- `solve_known(body, known)` is the reference solver with the known facts supplied, for
  prose states.
- `solve(body)` stays template-only.

### longctx (longctx.py)

A dossier is many records of one family in one state text, with record ids such as
`#48213`. The questions concern one target record. Levels set the token budget, estimated
at 4 characters per token:

| level | tokens | records | amendments | near-duplicate ids |
| --- | --- | --- | --- | --- |
| 1 | 8k | fills the budget | 0–2 | 0–1 |
| 2 | 16k | fills | 1–4 | 1–2 |
| 3 | 32k | fills | 2–6 | 1–3 |
| 4 | 64k | fills | 3–8 | 2–4 |
| 5 | 100k (fits the worker's 131072-token context with headroom) | fills | 4–10 | 2–5 |

- Amendments are lines placed after the record they amend (the solver relies on it), of the form
  `Correction to record #48213: the <label> is <value>.` The last amendment wins.
- A near-duplicate id is a transposition of two digits of the target id.
- The target record's position is uniform over the dossier (needle depth).
- The body is the systemone shape of v1. `instructions` holds the family context, the
  dossier conventions and the target id. The gold is the exact posterior over the target's
  effective known facts.
- Records can be template or bank prose. Bank prose may be used only for non-target
  records, because the target's known facts must be exact.
- `solve(body)` recomputes the gold from the text for template dossiers.
- The body must stay under 600,000 bytes.

## 5. Harness tracks (harness.py, ops.py, sqltask.py, paint.py)

### Protocol

```python
# OpenAI content parts: {"type": "text", "text"} | {"type": "image_url", ...}
Content = list[dict[str, Any]]


@dataclass(frozen=True)
class Env:
    name: str
    system: Callable[[dict], str]  # task -> system prompt (tools, policy, format)
    reset: Callable[[dict], Any]  # task -> mutable state
    observe: Callable[[dict, Any], Content]  # first user message
    # action (None = unparseable) -> observation, done
    step: Callable[[dict, Any, dict | None], tuple[Content, bool]]
    # final state -> loss in [0, 1]; None = needs the judge
    loss: Callable[[dict, Any], float | None]
```

The served body of a harness case is:

```jsonc
{"harness": "ops|sql|paint", "version": 2, "task": {...env task, including hidden env state...},
 "limits": {"turns": 12, "max_tokens": 1024}, "seed": 123456789}
```

`limits.turns` is 12 for every env. `limits.max_tokens` is 512 for `ops` and 1024 for `sql`
and `paint`.

It never contains the gold, the expected actions or the rubric.

- `parse_action(text) -> dict | None` strips code fences and returns the first balanced
  top-level JSON object that has a string `tool` and an object `args`. Otherwise it returns
  `None`.
- `chat_messages(env, task, history, first) -> list[dict]` (history = `[(output, observation)]`, first = the first observation) builds `system`, then `user` (the first
  observation), then for each turn `assistant` (the raw output) followed by `user` (the
  observation). Only the **latest** image part is kept. Earlier image parts become the
  text `[earlier canvas omitted]`.
- `async run_episode(env, body, generate) -> list[str]` loops up to `limits.turns`. `generate(messages,
  seed) -> str` is called with `seed = body.seed + turn`. The loop truncates each output
  to `MAX_OUTPUT_CHARS = 8192`, steps the env and stops when `done`. It returns the raw
  outputs, which form the transcript.
- `replay(env, body, outputs) -> tuple[Any, float | None]` rebuilds the state from `reset` and
  the outputs alone, then returns `(final_state, loss)`. The container scores **only**
  through `replay`. The worker's environment is never trusted, and no observation from the
  worker is used.
- An unparseable action gets the observation `Invalid action. Reply with one JSON object:
  {"tool": "<name>", "args": {...}}` and uses up a turn. Running out of turns ends the
  episode, and the loss is computed on the final state.
- Model outputs are data. No model output is ever executed, except SQL inside the §5.2
  sandbox.

### 5.1 ops

- **World.** Customers, orders (items, prices, dates, statuses, `final_sale` flags,
  payment methods) and a written policy (refund windows, cancellation rules, exchanges,
  escalation thresholds) are generated per case. Levels scale the number of orders and
  items, the policy clauses and the ambiguity.
- **Intent.** `sample_intent(rng, level)` is the customer's structured request, for
  example `{"action": "refund", "order_id": "...", "item_ids": [...], "reason": "damaged"}`.
  `INTENT_SCHEMA` is its JSON schema, used for the extractors' round trip. Its `claim` field
  (`none`, `recent_delivery`, `not_final_sale`, `not_shipped`, `already_approved`, `threat`)
  carries the level-4 pressure, so it is round-tripped too. A bank story is used at a level
  only when its intent fits it (an order id at level 1, a claim only at level 4, no more
  items than the level allows).
- **Levels.** 1–4. They scale the named items (2, 2, 3, 4), the extra lines, the customer's
  other orders, other customers, offered variants and quantities, the chance that the
  customer gives no order id (0, 0.6, 0.5, 0.5), the exception clauses (final sale, partial
  refunds, an escalation threshold; levels 3–4) and the pressure clause (level 4).
- **Customer text.** `render_intent(intent)` is the deterministic template text. When the
  bank holds `ops_story` items that fit the level, one of them replaces the template with
  probability 0.5. The case then adopts that story's intent and builds the world around it.
- **Tools** (JSON, one per turn): `find_customer(email)`, `list_orders(customer_id)`,
  `get_order(order_id)`, `refund(order_id, item_ids, reason)`, `cancel(order_id)`,
  `exchange(order_id, item_id, new_sku)`, `escalate(summary)` and `finish(outcome)`, where
  `outcome` is one of `refunded`, `cancelled`, `exchanged`, `denied` or `escalated`.
- **Gold.** `policy(world, intent)` gives the exact expected write set and outcome.
- **Loss.** `0.5·[outcome ≠ gold] + 0.5·(1 − |W ∩ W*| / max(|W|, |W*|, 1))`, where `W` is
  the set of writes performed and `W*` the expected writes. Two empty write sets count as a
  full match (the overlap term is 0), so a correct denial scores 0. A missing `finish` counts
  as a wrong outcome.

### 5.2 sql

- **Task.** Each case generates 4 tables (`customers`, `products`, `orders`,
  `order_items`), plus `reviews` from level 3, with 20–400 rows each, and a question from
  one of 12 structured query kinds (3 per level, levels 1–4): filter, join, group by,
  aggregate, top-k and date ranges. The answer shape is a scalar, a unique first row, an
  ordered top-k list or a set. The gold is the result of the compiled reference SQL on the
  same database, and a case is kept only when that answer is unique.
- **Tools.** `sql(query)` returns at most 50 rows and 4,000 characters. `answer(value)`
  gives the final answer: a number, a string or a list.
- **Sandbox.** The database is an in-memory `sqlite3` created per episode from the task.
  - An authorizer allows only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION` and
    `SQLITE_RECURSIVE`, and denies everything else: `ATTACH`, `PRAGMA` (and `pragma_*`
    table functions), writes, transactions, `VACUUM`, `ANALYZE`.
  - `READ` is allowed only on the task tables, `sqlite_master` and CTEs: virtual tables
    such as `sqlite_stmt` or `json_each` vary between runs or leak addresses.
  - Functions that read the clock, draw randomness or load code are denied by name
    (`random`, `randomblob`, `date`, `datetime`, `current_*`, `load_extension`, ...), so
    replay is exact.
  - Only one statement starting with `SELECT`, `WITH` or `VALUES` runs (`EXPLAIN` is
    refused: it leaks pointers).
  - A progress handler aborts after 500,000 VM steps. Limits: `SQLITE_LIMIT_LENGTH` 1000,
    `SQL_LENGTH` 4000, `LIKE_PATTERN_LENGTH` 100, `COMPOUND_SELECT` 20, `EXPR_DEPTH` 100,
    `ATTACHED` 0, `temp_store=FILE` with a 1 MiB page cache, no statement cache.
  - The query text is capped at 4,000 characters; NUL or unencodable characters come back
    as an error observation.
- **Loss.** 0 when the normalized answer equals the gold, else 1. Numbers compare at
  1e-6 relative after rounding to 2 decimals. Strings are case-folded and stripped. Lists
  are ordered when the spec orders them, otherwise compared as multisets.

### 5.3 paint

- **Canvas.** 256×256 RGB, white. Tools, one per turn:
  - `draw(commands)` takes a list of at most 64 commands:
    - `{"op": "rect", "x", "y", "w", "h", "fill"}`
    - `{"op": "circle", "cx", "cy", "r", "fill"}`
    - `{"op": "ellipse", "x0", "y0", "x1", "y1", "fill"}`
    - `{"op": "line", "x0", "y0", "x1", "y1", "width", "color"}`
    - `{"op": "polygon", "points": [[x, y], ...], "fill"}`
  - `undo()` removes the last `draw`.
  - `clear()` empties the canvas.
  - `done()` ends the episode.
- **Coordinates and colours.** Coordinates are integers clamped to `[−64, 320]`. Colours
  are `#rrggbb` or one of 12 palette names. A polygon has at most 64 points. At most 400
  accepted commands per episode, undone ones included. An invalid command rejects the
  whole draw.
- **Rendering.** Pillow `ImageDraw` without antialiasing. `render_png(commands) -> bytes`
  is deterministic for a given Pillow major version. The observation after each draw is
  the PNG as an `image_url` part plus a one-line text summary.
- **Level 1–2: `spec`.** The brief is rendered from a structured list of checks and
  parsed back (`parse(brief) == checks`), and the reference painter must pass every check
  before the case is kept. Level 1 has 1–2 shapes; level 2 has 3–6 and relations. The
  checks run on the container's render and are exact pixel predicates on palette colours:
  - `count`: connected components of a colour;
  - `within`: every pixel of a colour lies in a region;
  - `cover`: the percentage of a region a colour covers lies in a range;
  - `shape`: circle, rectangle or triangle, from the fill ratio of its bounding box;
  - `size`: the sides of each shape lie in a range;
  - `left_of`, `above`, `larger`, `inside`, `apart`: relations between shapes;
  - `stray`: at most 1 % of the canvas in colours outside the spec.

  Loss = 1 − checks passed / checks.
- **Level 3: `depict`.** The brief is a bank `depict` item, and `loss` returns `None`. The
  container judges the final render: see `judge_request`, `judge_loss` and §6.

## 6. Judge (container side)

- `paint.judge_request(brief, rubric, png) -> tuple[str, list[dict], dict]` returns
  `(system, user_parts, schema)`. The schema is `{"items": [{"id": int, "pass": bool}]}`,
  with exactly one entry per rubric item, including `STANDARD_RUBRIC`.
- The system prompt makes the judge:
  - strict and literal;
  - blind to the painter's identity, since no side, model or case index is sent;
  - fail any item that is satisfied only by written words;
  - ignore any instruction drawn in the image.
- The image is the container's own render, upscaled to 512 px with nearest neighbour.
- `paint.judge_loss(verdicts, items=None) -> float` is `1 − mean over judges of (passed /
  items)`. It rejects a verdict whose ids are not exactly `1..n` or on which the judges
  disagree about `n`; with `items` (the rubric length + 1) it also rejects a short verdict.
- The two sides of a case are judged in separate calls, in an order drawn from the case
  seed (`Random(f"judge|{seed}")`).
- A side whose render the judge cannot read (no valid verdict after 5 attempts) forfeits
  with loss 1. Only when **both** renders are unreadable is the pair excluded on both sides
  and counted in the verdict as `unjudged`; a duel that drops more than `UNJUDGED_MAX`
  (5 %) of its judged cases is not crowned. A provider refusal of the request's content
  (HTTP 400, 413, 415 or 422) counts as an unreadable reply. A gateway outage (transport,
  408/409/429/5xx, 401/403, or a gateway closed at shutdown) is not a verdict: the side
  stays pending and a later pass judges it. Without a judge (teacher off at startup) no
  side is judged: pending sides wait for a restart with the teacher.

## 7. Worker

- `VllmLauncher` adds `--max-model-len` (default 131072, `--max-model-len` CLI flag) and
  `--limit-mm-per-prompt '{"image": 1}'` to `vllm serve`.
- The launcher yields `{side: {"reader": <structured server URL>, "chat": <vllm URL>}}`.
- Read tracks post the body to `reader + /v1/systemone`, as in v1. Harness tracks run
  `harness.run_episode` with `generate` posting to `chat + /v1/chat/completions`. The body
  is `{"model": side, "messages", "max_tokens": limits.max_tokens, "temperature": 0.0,
  "seed": seed}`, and the reply is `choices[0].message.content`.
- **Answer items.** A read item is `{case_index, side, answers, reads}`, a harness item is
  `{case_index, side, transcript: [str]}`, and a failed item is `{case_index, side,
  error}`.
- **Failures.** A reader or chat `5xx`, or a transport error, is an infrastructure failure
  (retry). A `4xx` is an item error, and the case forfeits.
- **Case pages.** A page is `{"cases": [{"index", "track", "body"}]}`. It holds at most
  `limit` cases and at most 6 MiB of JSON, and always at least one case. The worker pages
  until it has all the cases.

## 8. Container

### Plan and cases

- **Plan.** `Settings.plan` maps each track to `TrackPlan(weight, cases)`. It is loaded
  from the env `OPENTYPE_PLAN` (JSON `{track: {"weight", "cases"}}`). The default is
  (7,400 cases in total, sized so `π_t·se_t` is about equal across tracks; on simulated
  losses a null duel has `g − g_LCB ≈ 0.07`, so the minimum crownable composite `g` is
  about `G_MIN + 0.07 ≈ 0.12`):

  ```text
  decisions 0.35/4000   longctx 0.25/800   ops 0.15/1000   sql 0.10/1000   paint 0.15/600
  ```

  `effective_plan(plan, bank, judge)` is what a job runs. A track whose cases cannot all be
  built (for example `depict` without a judge) keeps its weight but only builds the levels
  it can. A track with no buildable level, no case or no weight is dropped, and the
  remaining weights are renormalized. The job stores its effective plan, and the job's case
  count is the plan's total. The app refuses to start when the configured plan builds no
  case on the empty bank without a judge (unknown track, all weights zero).
- **Case order.** `track_of(index, plan)` is a deterministic interleave, so any prefix
  holds the tracks in proportion. The k-th case of track `t` has key `(k + 0.5) / n_t`.
  Cases are merged by `(key, track order)`.
- **Case construction.** `job_case(seed, plan, mix, index, bank, judge)` takes
  `rng = Random(f"{seed}|{index}")`, then:
  - for `decisions`: picks the level from `mix`, uses a sealed family with probability 0.3
    when the bank has one, and uses prose with probability 0.5 when the bank has prose for
    that family;
  - for any other track: picks its level uniformly among the buildable levels; the per-job
    mix and the retired levels apply only to decisions.
- `score_item(case, item) -> CaseScore | None`. It returns `None` only for a harness case
  whose loss needs the judge.

### Scoring

- A harness case scores as `CaseScore(loss, decisions=1, determined=1, correct=int(loss ==
  0), under_loss=0, under=0)`.
- **Per-track statistics.** For each track with at least 2 pairs, `g_t, se_t` is v1's
  `log_ratio` on that track's case sums.
- **Composite.** `g = Σ π_t min(g_t, ln 2) / Σ π_t` (each track's gain is capped at
  `TRACK_GAIN_CAP = ln 2` when two or more tracks are present) and `se = sqrt(Σ π_t² se_t²) / Σ π_t`, over the
  tracks present. `g_LCB` is the min over the two halves of the composite LCB99; a
  pair's half is the parity of its rank within its track (by case index), so every track
  splits evenly between the halves.
- **Crown.** A challenger is crowned when all of these hold (the per-track guard uses all
  active pairs, not the halves):
  - `g_LCB ≥ g_min`;
  - the v1 retired-level guard passes (decisions track only);
  - no track with at least 30 pairs regresses, meaning `g_t + Z99·se_t ≥ −ln 1.02` for
    every such track;
  - at least `min(2, G)` tracks gain, where `G` counts the tracks with at least 30 pairs
    and a track gains when it has at least 30 pairs and `g_t − Z99·se_t > 0` (one overfit
    track cannot carry the crown);
  - the duel was not early-stopped.
- **Early stop.** v1's rule applied to the composite.
- The verdict reports `tracks: {t: {g, se, pairs, champion_loss, challenger_loss,
  accuracy, regressed}}`, `track_guard: {pairs, min, gain_cap, gain_tracks}`, a per-track `gained` flag, `unjudged` (pairs excluded because
  neither side could be judged) and `unjudged_max`. Only decisions pairs feed the ladder
  statistics.

### Windows and the bank builder

- The first window opens with an empty, sealed bank. When a teacher is configured, a
  background task builds the bank of the **next** window.
- `rotate_window` promotes the next window, whose bank is complete. It never swaps a
  window's bank while that window is open.
- The window auto-rotates when the next bank is ready and the current window is at least
  `OPENTYPE_WINDOW_HOURS` old (default 24); the check runs every 30 s in its own loop,
  independent of judging. Without a teacher no
  next bank is ever ready, so windows rotate only on `POST /v1/admin/window/rotate`, which
  always works and opens the new window with the next bank or, when none is ready, the
  empty bank.
- A crown whose duel ran on the empty bank (all cases are public templates a miner can
  train on) still takes the throne, but its entitlement is capped at
  `OPENTYPE_EMPTY_BANK_CAP` epoch-masses (`Settings.empty_bank_cap`, default 0).
- `sha256(secret)` and `bank_digest` are published when a window opens.
- The secret and the bank items are published when the window closes:
  `GET /v1/windows/{id}/bank?offset&limit`, paged under 6 MiB.

### Seeds and storage

- **Seed.** `job_seed(secret, job_id, digest, beacon)` HMACs the drand randomness as well.
  At first lease, the container fetches `https://api.drand.sh/public/latest` with a 5 s
  timeout. It stores `{round, randomness}` on the job, reuses it on every retry and
  publishes it with the window. When drand is unreachable, it stores `null` and the job
  uses v1's seed.
- **Evidence.** Retargeting a job (each lease) clears the previous attempt's evidence, so
  published evidence always comes from the attempt that ran the published seed and plan.
- **Judging.** Pending judgments (the container's PNG, brief and rubric) are stored in a
  `judgments` table. `complete` moves the job to `judging` while any judgment is pending
  and judges inline for up to 20 s. A background task (every 30 s) judges the rest, scores
  them and then settles the job exactly like `complete`.
- **Schema.** `results` gains `track`. The schema version is stored in
  `PRAGMA user_version`. A v1 database is migrated in place.
- **Case cache.** The case cache is bounded by bytes (≤ 128 MiB), not by count.
- **Secrets.** `teacher.token` sits next to the other token files. It is checked once at
  startup: when `OPENTYPE_TEACHER_URL` is unset or the file is missing or empty, the
  teacher (bank builder and judge) is off and everything else works. The gateway re-reads
  the file on every call, so the token rotates without a restart.

## 9. Anti-cheat summary

| attack | defence |
| --- | --- |
| train on the benchmark | teacher content is private per window, and 30 % of decisions use sealed families invented per window. The bank is revealed only after the window closes and is never reused |
| teacher mistakes in gold | the teacher never labels. Gold comes from code (rules, policy, SQL, pixel checks). Text is admitted only after an exact round trip through 2 extractor families |
| exploit the judge | side-blind judging of the container's own render, a no-text rubric item, a blank-canvas negative control per rubric, instruction-in-image resistance, optional multiple judges, and pairs excluded symmetrically |
| a worker that misreports outcomes | the container recomputes every harness outcome by `replay` from raw outputs |
| code execution from model output | none, except SQL in the sqlite authorizer sandbox with a step limit. Paint commands are data |
| weak on one skill, strong on another | the per-track regression guard blocks a crown that regresses significantly on any track |
| an operator picking seeds | commit-reveal of the window secret and bank digest, plus a drand beacon mixed into the job seed |
| surface cues in templates | `tests/test_shortcuts.py`: a bag-of-words naive Bayes on public decisions cases must stay within 3 points of the majority baseline on determined yes/no items |

## 10. Oracles, markers and buildable levels (test and dispatch contract)

- `buildable(bank) -> tuple[int, ...]`, in `longctx`, `ops`, `sqltask` and `paint`, lists
  the levels that can be built with this bank. For example, `paint` returns `(1, 2)` without
  `depict` items and `(1, 2, 3)` with them.
- Every env's `system(task)` starts with the exact line `OpenType harness: <env name>`.
- **Reference oracles.** `reference_policy(messages) -> str`, in `ops`, `sqltask` and
  `paint`, returns the next raw output of an agent that sees **only the conversation**
  (the OpenAI messages that `harness.chat_messages` builds) and solves every
  template-rendered task of every buildable level perfectly (replay loss 0). It proves
  that each task is solvable from what the model sees, just as `generator.solve` does for
  reads. It raises `ValueError` when it cannot, for example on teacher prose or `depict`.
  Test fakes use the oracles to act as a perfect model.
- `tracks.solve_body(body) -> dict[str, list[float]]` returns the exact gold of a read
  body, computed from the text alone. It dispatches on a marker that each read track puts
  on the first line of `instructions`: the v1 `Record type: ...` for decisions, and
  `Dossier: ...` for longctx.
- `generator.solve(body)` also works for **sealed** families. When the title is not
  public, it rebuilds the family from the facts block of `instructions`.
- **Ops intents.**
  - `ops.sample_intent(rng, level) -> dict` returns a self-contained customer request.
    It holds only what a customer can state: email, order id, action, item names, reason
    and, for exchanges, the wanted variant. It validates against `INTENT_SCHEMA`.
  - `describe_intent(intent) -> str` is a factual English description that the teacher
    paraphrases.
  - `render_intent(intent) -> str` is the deterministic template message.
  - `make_case` builds a world consistent with the intent (a bank story's or a sampled
    one). Records that the customer does not state (dates, statuses, `final_sale`, payment,
    policy) come from the world, and the agent must look them up with the tools.
  - At higher levels the customer may pressure the agent or make claims that the records
    contradict. The policy decides, not the customer.

## 11. Cross-module helpers (fixed names)

- `teacher.judge_png(gateway, config, brief, rubric, png) -> float | None` returns the
  judge loss of §6, averaged over `config.judges`. It retries each judge up to 5 times and
  returns `None` when any judge's replies cannot be read; a gateway outage raises
  `GatewayError`. The container and the depict negative
  control both use it.
- `scoring.TrackMoments = tuple[int, float, float, float, float, float]` holds
  `(n, sa, sb, saa, sbb, sab)` for one track.
  - `scoring.composite(moments: Mapping[str, TrackMoments], weights: Mapping[str, float])
    -> tuple[float, float]` returns `(g, se)`, which is v1's `log_ratio_moments` when only
    one track is present.
  - `scoring.verdict(pairs, retired, stopped, weights=None)` and
    `scoring.early_stop(pairs, retired, weights=None)` keep v1 behaviour when `weights` is
    `None`.
  - The store computes the moments of each track in SQL for early stop.
- `store.Settings` gains `plan: Mapping[str, tracks.TrackPlan] | None = None`. `None` means
  `tracks.DEFAULT_PLAN`. v1's `duel_cases` stays as a test convenience: when `plan` is
  `None` and `duel_cases` is not the default, the plan is decisions-only with that many
  cases.
