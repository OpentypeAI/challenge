# Mechanism: TD-Exact

Miners improve DiffusionGemma weights. The challenge measures the improvement on synthetic
typed decisions whose gold is exact, certifies it statistically in a paired duel against
the champion, and pays each certified unit of gain once through a ledger.

## 1. Cases with exact gold

A case is a record type (family), a closed world of 10 facts with finite domains, 3 derived
facts, and 5–7 questions. Each question has a decision-list rule ("first match wins") over
the facts. Question types follow the structured-read protocol:

| type | answer | gold |
| --- | --- | --- |
| `noul` | `{"noul": p_yes}` | `[p_yes, p_no]` |
| `choice` | `{"choice", "probabilities": {option: p}}`, up to 26 options | a distribution over the options |
| `score` | `{"score", "legend": {"0": ..}, "probabilities": {"0": p, ..}}` | a distribution over the ordered levels |

- **Gold = rule(world).** The rule is evaluated by a tree-walk interpreter and by a compiled
  Python function. The two must agree (N-version), or generation stops with a
  `GeneratorError`. It is a bug, never a silent discard.
- **Criteria text equals the rule.** The question instructions are rendered from the rule
  by template and parsed back. `parse(text) == rule` is checked on every item.
- **Round trip.** The record text is rendered from exactly the fact vector (several
  templates per fact, shuffled order, distractors), then extracted back.
  `extract(render(facts)) == facts`, including facts left unstated, or the case is
  discarded.
- **Unstated facts give an exact posterior.** The world sampler really draws a hidden fact
  uniformly from its domain, and the instructions say so. The gold of an underdetermined
  item is therefore the exact Bayes posterior, the fraction of completions that reach each
  answer. It is calibrated against the true world: over 1,679 underdetermined items, the
  realized mass is 1009.1 against 1012.1 expected.
- **Probes are ordinary items.** `<qid>_reordered` permutes the options, so the gold is the
  same permutation. `<qid>_mirror` negates a `noul` rule, so the gold is `1 − g`. Injected
  lines ("ignore the rules…") and distractors about other records must change nothing.
- **Reference solver.** `generator.solve(body)` recomputes every gold from the request text
  alone, without the world. The tests check it against the generator to 1e-12.

Levels scale difficulty:

| level | clauses per rule | conditions per clause | hidden facts | distractors | options | derived (multi-hop) | probe rate | injection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1–2 | 1 | 0 | 0 | 4 | 0 | 0 | no |
| 2 | 2–3 | 1 | ≤1 | ≤1 | 6 | 0 | 0.25 | no |
| 3 | 3–4 | 2 | ≤1 | ≤2 | 8 | 1 | 0.5 | no |
| 4 | 4–5 | 2 | ≤2 | ≤3 | 10 | 1 | 0.5 | yes |
| 5 | 5–6 | 2 | ≤2 | ≤5 | 14 | 2 | 0.5 | yes |
| 6 | 6–7 | 3 | ≤3 | ≤8 | 18 | 2 | 0.5 | yes |
| 7 | 7–8 | 3 | ≤3 | ≤12 | 22 | 3 | 0.5 | yes |
| 8 | 8–10 | 3 | ≤3 | ≤16 | 26 | 3 | 0.5 | yes |

The generator is public. `opentype-challenge generate` gives miners unlimited training data
with exact soft targets. Duel cases come from the same generator under a secret seed.

## 2. Case selection: commit-reveal windows

- A window has a 32-byte secret, and `sha256(secret)` is published while the window is open.
- The seed of a job is `HMAC(secret_w, job_id | challenger manifest digest)`. Case `i` is
  `job_case(seed, mix, i)`: a level drawn from the job's mix, then a family and a case.
- The miner commits its manifest digest before the job exists and cannot know the secret.
- The cases the worker receives never contain gold, realized worlds or levels. The worker
  hashes every served body in order (`cases_sha256`).
- When the operator rotates the window, the secret is published with every job's digest,
  mix and case count. `opentype-challenge audit` regenerates the cases and must reproduce
  each `cases_sha256`.

## 3. Scoring

For each decision `k` of a case, with answer vector `p` and gold `g`:

```
loss_k = ½ · Σ_j (p_j − g_j)²                    half-Brier, in [0, 1]; 0 is reachable
loss_k = 1  (forfeit) if the answer is missing, invalid, non-finite or not normalised
             (|Σp − 1| > 1e-3), if the read's label_mass < 0.5, or if argmax_is_label is false
L_case = Σ_k loss_k                              cluster = case
```

The duel statistic uses the paired cluster sums of the active (non-retired) levels:

```
g      = ln( Σ L_champion / Σ L_challenger )    log error reduction, additive across crowns
se     = delta method on the paired case sums (cluster-robust, covariance included)
LCB99  = g − 2.326 · se
g_LCB  = min(LCB99 on even case indices, LCB99 on odd case indices)
crown  ⇔ g_LCB ≥ g_min = −ln 0.95 ≈ 0.0513   (at least 5 % less loss, on both halves)
        ∧ guard: UCB99( (err_challenger − err_champion) / determined_champion ) ≤ 0.002
             on retired levels
        ∧ not early-stopped
```

- Each side's summed loss is floored at `−ln 0.01 ≈ 4.6`, the Poisson UCB99 of a zero
  count. A perfect challenger can therefore certify at most `ln(Σ L_champion / 4.6)` nats
  in one duel and cannot claim an infinite gain.
- **Early stop.** Once 5,000 paired decisions on active levels exist, the duel stops if
  `g + 3·se < 0`. The container checks this on every answers batch and tells the worker to
  stop. This saves about 80 % of the GPU time on bad submissions.
- **Headline metrics** per level and side: accuracy on determined items (argmax equals the
  gold argmax, target 100 %), mean half-Brier on underdetermined items, and summed loss.

## 4. Power

The table gives P(crown) for a challenger that removes a fraction `r` of the champion's
errors. Errors cluster by case (6 decisions per case, a third of cases hard, churn 0.25),
and both halves must reach LCB99 ≥ g_min. It comes from 200 simulated duels per cell with
this repository's `scoring.verdict` (the `simulated_duel` model of `tests/test_scoring.py`,
adapted from the design's `power_check.py`). The numbers match the design table within
simulation error.

| champion accuracy | r = 0, 240 k | r = 10 %, 60 k | r = 10 %, 240 k | r = 20 %, 60 k | r = 20 %, 240 k |
| --- | --- | --- | --- | --- | --- |
| 95 % | 0.00 | 0.41 | 1.00 | 1.00 | 1.00 |
| 98 % | 0.00 | 0.06 | 0.72 | 0.99 | 1.00 |
| 99 % | 0.00 | 0.01 | 0.28 | 0.77 | 1.00 |
| 99.5 % | 0.00 | 0.01 | 0.07 | 0.32 | 0.99 |

A model equal to the champion is never crowned. With the default 40,000 cases per duel
(about 5.5–6 decisions per case, so ≈ 230 k decisions), a 20 % error reduction is detected
up to 99.5 % champion accuracy. Past that point, no fixed-size test has power, so the
ladder moves the duel to harder levels. `tests/test_scoring.py` asserts four of these
properties on every CI run.

## 5. Frontier ladder

- The ladder is an ordered list of levels (default `1..8`) and a width (default 2). The
  first `width` levels that are not retired are **active**. The rest are pending.
- **Duel mix.** Active levels share 90 % of the cases (100 % when nothing is retired),
  weighted by the champion's error rate on each level (1.0 when unknown), with a floor of
  5 % per level. Retired levels share 10 % evenly and feed only the regression guard.
- **Retirement.** After each scored duel and each crown, an active level on which the
  champion's accuracy has a Wilson LCB99 of at least 99.9 % is retired, and the next
  pending level becomes active.
- The champion's per-level statistics accumulate over every duel it plays. A new champion
  starts from its own crowning duel.
- The operator can replace the ladder (`PUT /v1/admin/ladder`). When every level is
  retired, the hardest one stays active until new levels ship.

## 6. Ledger of certified gain

```
crown c         → entitlement E_c = g_LCB,c / g_min epoch-masses   (a minimal crown = 1 epoch)
epoch e         → pay outstanding entitlements FIFO (oldest crown first) up to 1 epoch-mass
                  weights[hotkey] = paid mass; the remainder burns (full_share_mass = 1.0)
```

- Amounts are integers in units of 1e-9 epoch-mass, so payments never drift. Each epoch
  body is persisted on first computation and replayed byte for byte.
- An entitlement survives dethronement: the network pays each nat of loss removed exactly
  once, to whoever removed it first. Total emission is proportional to Σ g, that is
  `ln(loss₀ / loss_now)`.
- The base model (champion 1) is never paid.
- Splitting one improvement into several crowns ("drip release") pays no more in total and
  pays the LCB haircut at every step.
- The optional `OPENTYPE_WINDOW_ENTITLEMENT_CAP` bounds the total entitlement created per
  window, as protection against ledger drift when the duel mix changes between crowns.
- If nobody improves, everything burns. Emission is never paid for nothing.

## 7. Anti-cheat

| Attack | Defence |
| --- | --- |
| memorise the benchmark | cases are drawn per job from a secret seed; the generator's space is effectively unbounded; the gold is never served |
| fit label noise or teacher quirks | there are none: the gold is exact |
| pick a favourable seed | the seed depends on the window secret and the committed manifest digest |
| copy the champion | a byte-identical manifest is refused at intake and again when a job is re-targeted; a near copy has g ≈ 0 and fails g_min |
| submit several copies of the same gain | earliest intake wins; the later copy duels the new champion and fails |
| sybil flooding or cost DoS | a registered hotkey is required, one open submission per hotkey, `max_pending` (4) queued in total, and early stop |
| replay a signed submission | the nonce is unique and `exp` at most 300 s ahead; the signature covers the manifest digest |
| code in the weights repo | only `*.safetensors`, `model.safetensors.index.json` and `config.json` are accepted. `config.json` must be byte-equal to the base. Tokenizer, chat template and processor files always come from the pinned base. Serving runs with `HF_HUB_OFFLINE=1` and no remote code |
| swap weights after submission | every file's sha256 is committed in the signed manifest and verified at duel time |
| private or gated weights | the worker downloads anonymously; a failure to download is the challenger's fault and a rejection without retry |
| instructions injected into the record | probe and injection items are scored like any other item |
| a worse model that happens to be lucky | LCB99 on both halves, plus a regression guard on mastered levels |
| operator tampering with cases | commit-reveal plus a public audit of `cases_sha256` (the operator still knows the secret; see §8) |

## 8. Residual risks and gaps

| Risk | Status and next step |
| --- | --- |
| overfitting the public generator instead of general skill | there are no sealed families yet and no anchor telemetry (typed-decisions test, PhishNChips, Laya probes, jev-harness-lab). The operator can pause crowns by hand (`PUT /v1/admin/crowns`). Next: sealed families (hash-committed, revealed after 4 windows) and an anchors job with the RT-7 automatic pause |
| template rendering is learnable | a deterministic multi-template renderer and dictionary extractor stand in for the LLM renderer plus two extractor models. Next: an LLM renderer behind the same round-trip check |
| no shortcut audit or human audit yet | the per-window bag-of-words shortcut audit and the 300-case human audit are not automated |
| the operator knows the window secret | no drand round is mixed into the seed. Next: add the round at lease time |
| the worker host is trusted | one host reads both sides. Evidence (image, vLLM version, `structured_server.py` sha256, file digests, cases digest) is bound to the job. Next: a second host re-reads half B |
| B300 throughput and base accuracy are not measured | phase 0: measure decisions/s at canvas 256 and base accuracy per level, choose the ladder so the base scores 60–90 % on the entry levels, then set `OPENTYPE_DUEL_CASES` |
| statistical approximations | a delta-method SE stands in for the 10,000-resample cluster bootstrap, and the Wilson bound treats decisions as independent (mildly optimistic for retirement) |
| publication is checked only at duel time | there is no re-check before each payment. A withdrawn repo keeps being paid |
