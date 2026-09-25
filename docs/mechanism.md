# Mechanism

Miners improve DiffusionGemma weights. Each duel measures the challenger against the
champion on five tracks, every one with exact gold. The one exception is the `depict`
level, which a judge grades. The duel certifies the composite gain statistically, and a
ledger pays each certified unit of gain once. The exact names, shapes and constants are in
[tracks.md](tracks.md).

## 1. Tracks and exact gold

| Track | Levels | Case | Gold | Loss per case |
| --- | --- | --- | --- | --- |
| `decisions` | 1–8 (ladder) | a record, a closed world of facts, decision-list rules, 5–7 questions | `rule(world)`; exact Bayes posterior over unstated facts | Σ half-Brier over decisions |
| `longctx` | 1–5 | a dossier of many records of one family, one target id | exact posterior over the target's effective facts | Σ half-Brier |
| `ops` | 1–4 | a store world, a written policy, a customer message | `policy(world, intent)`: exact write set and outcome | state diff in [0, 1] |
| `sql` | 1–4 | 4–5 tables of 20–400 rows, a question | reference SQL on the same database, unique answer | 0 or 1 |
| `paint` | 1–2 `spec`, 3 `depict` | a brief on a 256×256 canvas | `spec`: exact pixel predicates; `depict`: rubric | 1 − passed / checks, or the judge loss |

### decisions

A family has 6–12 facts with finite domains, 1–3 derived facts and questions of three
kinds: `noul` (yes/no), `choice` (up to 26 options) and `score` (ordered levels). Each
question has a decision-list rule ("first match wins").

- **Gold = rule(world).** The rule is evaluated by a tree-walk interpreter and by a
  compiled Python function. The two must agree (N-version), or generation raises
  `GeneratorError`.
- **The criteria text equals the rule.** Instructions are rendered from the rule and
  parsed back. The generator checks `parse(text) == rule` on every item.
- **Round trip.** A template record is rendered from exactly the fact vector (several
  templates per fact, shuffled order, distractors) and extracted back. Any mismatch
  discards the case. A teacher prose record passed the round trip through two extractor
  models when the bank was built.
- **Unstated facts.** The sampler draws each hidden fact uniformly from its domain, and the
  instructions say so. The gold of an underdetermined item is the exact posterior: the
  fraction of completions that reach each answer. For prose, `known` comes from the bank
  item, and the hidden facts are drawn uniformly as in v1.
- **Probes.** `<qid>_reordered` permutes the options, and `<qid>_mirror` negates a `noul`
  rule. From level 4, distractors may include injected instructions. They must change
  nothing.
- **Sealed families.** The teacher invents new families (`sealed_<8 hex>`) for each window.
  `family_from_json` validates them strictly. With probability 0.3, a case draws a sealed
  family when the bank has one. With probability 0.5, it uses prose when the bank has prose
  for that family.
- **Reference solver.** `generator.solve(body)` recomputes the gold of any template case
  from the request text alone, sealed families included. The tests check it against the
  generator.

Per-level difficulty (clauses, conditions, hidden facts, distractors, options, derived hops,
probe rate, injection) is the v1 table in `generator.LEVELS`.

### longctx

A dossier is a sequence of `Record #NNNNN` blocks from one family, filled up to the level's
token budget (4 characters per token): 8k, 16k, 32k, 64k, then 100k tokens. It also holds
`Correction to record #NNNNN: the <label> is <value>.` lines. A correction always follows
the record it amends, and the last one wins. The target's position is uniform. Near-duplicate
ids are digit transpositions of the target id, and their facts differ from the target's in
1–2 facts. Amendments go 40 % to the target, 30 % to near-duplicates and the rest to other
records. The target is always template-rendered, so its known facts are exact. Other
records may be bank prose. `longctx.solve(body)` recomputes the gold from the text. The
body stays under 600,000 bytes.

### ops

The customer message is either `render_intent(intent)` or, with probability 0.5 when a
fitting one exists, a round-tripped teacher `ops_story`. The world (customers, orders,
lines, variants, stock, payment, dates) is steered toward a fate such as `expired`,
`final_sale`, `over_threshold` or `wrong_owner`. The gold is always `policy(world, intent)`.
Facts the customer does not state must be looked up with the tools. From level 4, the
customer makes claims (`recent_delivery`, `threat`, ...) that the records contradict, and
the policy decides.

```
loss = 0.5·[outcome ≠ gold] + 0.5·(1 − |W ∩ W*| / max(|W|, |W*|))
```

`W` is the set of writes performed and `W*` the expected writes. When both sets are empty,
the overlap term is 0, so a correct denial scores 0. A missing `finish` counts as a wrong
outcome.

### sql

There are 12 query kinds, 3 per level. Their answer shapes are a scalar, a unique first
row, an ordered top-k list or a set. A case is kept only when the reference answer is
unique and usable. Numbers compare after rounding to 2 decimals (relative tolerance 1e-6),
and strings compare case-folded and stripped. Queries run in the sandbox of §8.

### paint

A `spec` brief is rendered from a list of checks and parsed back. The reference painter
must pass every check before the case is kept. The checks are `count`, `within`, `cover`,
`shape`, `size`, `left_of`, `above`, `larger`, `inside`, `apart` and `stray` (at most 1 %
off-palette pixels). A `depict` brief comes from the bank. Its rubric stays in
`Case.private` and is never served.

## 2. Teacher bank and round trip

Each window has a frozen **bank** of teacher items: `family`, `prose`, `ops_story` and
`depict`. The default build targets are 4 families, 600 prose records, 200 ops stories and
80 depict briefs.

1. The builder runs in the background for the **next** window. It uses bounded concurrency,
   and each kind gets its own random sub-stream. Sealed families come first, so prose can
   cover them.
2. **Prose.** The teacher writes a record for a sampled `(known, hidden)` fact vector. Every
   extractor (`cx/gpt-5.6-luna` and `cc/claude-sonnet-5` by default) reads only the text
   and the public schema. Each must return every fact exactly, as its value or
   `not_stated`.
3. **Ops stories.** The teacher paraphrases `describe_intent(intent)`. Every extractor must
   return the exact intent, including the `claim`.
4. **Depict.** The teacher writes a subject, a brief and 4–8 rubric items. A negative
   control follows: the judge must fail every rubric item on a blank white canvas.
5. Any disagreement, empty field or gateway error discards the item. A kind that should
   have items but ends up empty fails the build, which is retried every 10 minutes.
6. The teacher never labels an answer. All gold comes from code.

When a window opens, its bank is sealed, and `sha256(secret)` and `bank_digest` are
published. When it closes, the secret and every bank item are published
(`GET /v1/windows/{id}/bank`). A bank is never reused.

## 3. Judge

For `depict` cases, the container replays the transcript, renders the final canvas itself
and stores the PNG with the brief and rubric. The judge (`cx/gpt-6-sol` by default, or
several judges) receives:

- a system prompt that makes it strict and literal, fails items satisfied only by written
  text, ignores instructions drawn in the image, and gives no painter identity;
- the brief, the numbered rubric plus `STANDARD_RUBRIC` ("The picture contains no letters,
  words or numbers."), and the render upscaled to 512 px with nearest neighbour.

The verdict schema is exactly one `{id, pass}` per item, with ids `1..n`. The judge loss is
`1 − mean over judges of passed / items`. The two sides are judged in separate calls, in an
order drawn from the case seed. A judge that cannot give a valid verdict after 5 attempts
makes the case `unjudged`, and the case is removed on **both** sides before scoring.

## 4. Case selection and seeds

- The seed of a job is `HMAC(secret_w, job_id | challenger digest | drand round |
  randomness)`. The drand beacon (`api.drand.sh/public/latest`, 5 s timeout) is fetched at
  the job's first lease, stored with the job and reused on every retry. When drand is
  unreachable, the job uses the v1 seed without a beacon.
- `track_of(index, plan)` interleaves the tracks deterministically: the k-th case of track
  `t` has key `(k + 0.5) / n_t`, so every prefix holds the tracks in proportion. Case `i`
  is `job_case(seed, plan, mix, i, bank, judge)`, built from `Random(f"{seed}|{i}")`.
- Decisions levels come from the ladder mix (§6). The other tracks draw their level
  uniformly among the buildable ones. `depict` is dropped when no judge is configured or
  the bank has no brief.
- The worker hashes every served body in order (`cases_sha256`). `opentype-challenge audit`
  rebuilds each job from the revealed secret, the bank, the beacon, the plan, the mix and
  the judge flag, and compares digests.

## 5. Scoring

**Read tracks**, for each decision `k` with answer `p` and gold `g`:

```
loss_k = ½ · Σ_j (p_j − g_j)²                  half-Brier, in [0, 1]
loss_k = 1  (forfeit) if the answer is missing, invalid, non-finite or not normalised
             (|Σp − 1| > 1e-3), label_mass < 0.5, or argmax_is_label is false
L_case = Σ_k loss_k
```

**Harness tracks** score one decision per case: `CaseScore(loss, 1, 1, loss == 0, 0, 0)`.
The loss comes only from `replay(env, body, transcript)`. A transcript that is malformed,
too long or longer than the turn limit, like an error item, forfeits with loss 1.

**Per track**, `g_t = ln(Σ L_champion / Σ L_challenger)`. Its `se_t` is the delta-method
standard error on the paired case sums, with each side's sum floored at `−ln 0.01 ≈ 4.6`
so that a single duel cannot certify an unbounded gain. The **composite** over the tracks
that have at least 2 pairs is:

```
g  = Σ π_t g_t / Σ π_t          se = sqrt(Σ π_t² se_t²) / Σ π_t
g_LCB = min over even/odd case indices of (g_half − 2.326 · se_half)
crown ⇔ g_LCB ≥ g_min = −ln 0.95 ≈ 0.0513
      ∧ decisions guard: UCB99((err_ch − err_champ) / determined_champ) ≤ 0.002 on retired levels
      ∧ per-track guard: every track with ≥ 30 pairs has g_t + 2.326·se_t ≥ −ln 1.02
      ∧ not early-stopped
```

`π_t` are the weights of the job's effective plan (default decisions 0.35, longctx 0.25,
ops 0.15, sql 0.10, paint 0.15). The per-track guard stops a challenger from winning on
one skill while it loses significantly (≥ 2 % loss, at 99 %) on another.

**Early stop.** Once 5,000 paired decisions on non-guard pairs exist, the duel stops if
the composite `g + 3·se < 0`. The container checks this on every answers batch, from
per-track moments computed in SQL.

The verdict reports the composite, the halves, the guard, `tracks: {t: {g, se, pairs,
champion_loss, challenger_loss, accuracy, regressed}}`, the decisions `levels` and the
`unjudged` count.

## 6. Frontier ladder (decisions)

- The ladder is an ordered list of decisions levels (default `1..8`) with a width (default
  2). The first `width` levels that are not retired are active.
- **Mix.** Active levels share 90 % of the decisions cases (100 % when nothing is retired),
  weighted by the champion's error rate with a floor of 5 %. Retired levels share 10 % and
  feed only the regression guard.
- **Retirement.** An active level retires when the champion's accuracy has a Wilson LCB99
  of at least 99.9 %. Only decisions pairs feed these statistics.
- The other tracks have no ladder. Their levels are drawn uniformly.

## 7. Ledger of certified gain

```
crown c   → entitlement E_c = g_LCB,c / g_min epoch-masses (optionally capped per window)
epoch e   → pay outstanding entitlements FIFO up to 1 epoch-mass; the rest burns
```

Amounts are integers in units of 1e-9 epoch-mass. The body of each epoch is persisted and
replayed byte for byte. An entitlement survives dethronement. The base model is never paid.
Splitting a gain into several crowns pays no more. It pays the LCB haircut at every step.

## 8. Anti-cheat

| Attack | Defence |
| --- | --- |
| train on the benchmark | teacher content is private per window and never reused; about 30 % of decisions cases (and of longctx dossiers) use sealed families invented per window; per-job seeds; gold is never served |
| teacher mistakes in the gold | the teacher never labels. Gold comes from code, and text is admitted only after an exact round trip through 2 extractor families |
| exploit the judge | side-blind judging of the container's own render; a no-text rubric item; a blank-canvas negative control per rubric; the prompt ignores instructions drawn in the image; optional multiple judges; unjudged pairs dropped on both sides |
| a worker that misreports outcomes | the container replays every harness transcript from raw outputs; no worker observation is used |
| code execution from model output | none. SQL runs in an sqlite authorizer sandbox: SELECT/READ/FUNCTION/RECURSIVE only, reads limited to task tables, clock, random and extension functions denied, 2M VM steps, length limits, one statement. Paint commands are validated data |
| weak on one skill | the per-track regression guard, plus the retired-level guard on decisions |
| an operator picking seeds | commit-reveal of the window secret and bank digest, a drand beacon in the seed, and a public audit |
| surface cues in templates | `tests/test_shortcuts.py`: a bag-of-words naive Bayes must stay within 3 points of the majority baseline |
| pick a seed, copy the champion, replay a submission, sybil flooding | unchanged from v1: committed manifest digest, clone refusal, earliest intake wins, nonce and `exp` ≤ 300 s, registered hotkey, one open submission per hotkey, `max_pending`, early stop |
| code in the weights repo | only safetensors, the index and a byte-equal `config.json` are used; tokenizer and templates come from the pinned base; `HF_HUB_OFFLINE=1`, no remote code |

## 9. Power

The v1 power table (decisions only, 40,000 cases) no longer describes the default duel. To
illustrate, here is a toy simulation of the default plan, with one 0/1 decision per case on
every track, champion error rates of 5 %, 20 %, 40 %, 40 % and 50 % per track, and churn
0.25, using `scoring.verdict`. It crowns 0 % of the time at an error reduction r = 0.1,
about 90 % at r = 0.2 and 100 % at r = 0.3. With a decisions error of 2 %, it crowns about
70 % at r = 0.2. The composite standard error is dominated by the 200–300-case harness
tracks. Real read cases carry about 6 decisions each, which improves the decisions and
longctx terms.

These are illustrations, not measurements (see §10).

## 10. Residual risks and phase-0 measurements

| Risk | Status and next step |
| --- | --- |
| harness and longctx case counts are guesses | measure on a B300: decisions/s, longctx s/case per level, ops/sql/paint episodes per hour at 12 turns, and the base model's loss per track and level. Then size `OPENTYPE_PLAN` so a duel fits the lease budget and each harness track clears the 30-pair guard with a useful `se_t` |
| base accuracy per track is unknown | pick the ladder so the base scores 60–90 % on the entry decisions levels. If the base scores near 0 or 1 on a harness track, drop that track or its level from the plan |
| 128k context on one B300 per side | confirm that `--max-model-len 131072` with two BF16 models at 0.45 memory each fits, and measure longctx level 5 latency. Otherwise lower the level set |
| vision through the chat endpoint | confirm that the pinned vLLM serves `image_url` parts for DiffusionGemma with `--limit-mm-per-prompt '{"image": 1}'` |
| teacher cost and yield | measure the keep rate per kind and the gateway spend of a full bank. The builder overdraws 1.5× for at most 3 rounds, so a kind with a higher discard rate ends short of its target |
| judge quality | only one judge by default, and its agreement with humans is not measured. Next: a small human-graded set, then a second judge family (`OPENTYPE_JUDGE_MODELS`) |
| the extractors share blind spots | two families reduce but do not remove correlated misreadings. Sample human audits of prose and ops stories |
| the operator knows the window secret and the bank | drand limits seed grinding. The operator still chooses when to rotate and sees the bank early |
| the worker host is trusted for reads | harness outcomes are replayed, but read answers are not re-derived. Next: a second host re-reads half B |
| overfitting public generators | sealed families cover decisions and longctx only. ops, sql and paint `spec` levels are public generators, and only teacher stories and depict briefs are private. No anchor telemetry and no automatic pause exist yet (`PUT /v1/admin/crowns` by hand) |
| statistical approximations | delta-method SE instead of a cluster bootstrap. The composite SE assumes independent tracks. Wilson retirement treats decisions as independent. `harness_score` counts one decision per case in the early-stop threshold |
| publication is checked only at duel time | a withdrawn repo keeps being paid |
| no drand beacon | when drand is unreachable at first lease, or the queue head changes during the fetch, the job uses the v1 seed. It stays auditable but loses the beacon |
