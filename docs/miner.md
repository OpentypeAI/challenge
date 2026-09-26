# Miner guide

## What you submit

You submit a signed manifest of a **public, ungated** Hugging Face model repo at a full
40-hex commit. Only these files are read:

| File | Rule |
| --- | --- |
| `model.safetensors` or `model-NNNNN-of-NNNNN.safetensors` | your weights, in the champion's format (see below) |
| `model.safetensors.index.json` | required when the weights are sharded |
| `config.json` | must be byte-equal to the current champion's |

The format follows the champion. Until the operator migrates it, the champion is BF16 and
`config.json` is the base revision's (sha256 `13b11d2f…c506`). After the one-way NVFP4
migration (`GET /v1/status` shows the champion), every submission is a ModelOpt NVFP4
checkpoint: `config.json` byte-equal to `nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df`'s
(sha256 `b4f650bd…5fde`) and exactly its tensor names, dtypes and shapes (W4A4 FP4 routed
experts, FP8 block scales, FP32 global scales; the rest BF16). The worker checks the shard
headers before serving; any other layout is rejected. BF16 manifests are refused at intake
from then on. Quantize your own improved weights with the same recipe.

Tokenizer, chat template, generation and processor files always come from the pinned base
`google/diffusiongemma-26B-A4B-it@f7f5b7f5fa82ffc52addd066915886d497f5517b`. Anything else
in the repo is ignored. Code, pickles and custom configs are never loaded. Keep the vision
tower: the `paint` track sends images.

## 1. What is measured

A duel runs the same cases on your model and on the champion. The default plan has 7,400
cases. `GET /v1/status` shows the live `plan` and track weights:

| Track | Cases | Weight | How your model is called | Loss |
| --- | --- | --- | --- | --- |
| `decisions` | 4,000 | 0.35 | structured read (`/v1/systemone`) | Σ half-Brier per case |
| `longctx` | 800 | 0.25 | structured read, 8k–100k-token state | Σ half-Brier |
| `ops` | 1,000 | 0.15 | chat, up to 12 turns, 512 tokens per turn | outcome and write-set diff |
| `sql` | 1,000 | 0.10 | chat, up to 12 turns, 1024 tokens per turn | 0 or 1 |
| `paint` | 600 | 0.15 | chat with images, up to 12 turns, 1024 tokens per turn | failed pixel checks, or a judge's rubric score |

Chat calls use temperature 0 and `seed = case seed + turn`. Only the latest canvas image
stays in the context. Earlier images become `[earlier canvas omitted]`.

You win when all of these hold:

- the weighted composite of the per-track `g_t = ln(champion loss / your loss)` has a 99 %
  lower bound of at least `−ln 0.95` (about 5 % less loss) on both halves of the duel;
- no track with 30 or more pairs regresses significantly (`g_t + 2.326·se_t ≥ −ln 1.02`);
- you do not regress on retired decisions levels.

So a model that is much better at decisions but worse at `ops` loses. Improve broadly.

The private part of each window is teacher-written content that you cannot see in advance:

- sealed record families (about 30 % of decisions cases and longctx dossiers);
- prose records (about half the cases of a family, when the bank has prose for it);
- customer stories (half the ops cases);
- `depict` briefs (paint level 3).

The generators, formats and tools are public and identical to the duel's. Train for the
skill, not for the templates.

## 2. Public generators

```bash
uv tool install "opentype-challenge[miner,wallet] @ git+https://github.com/OpentypeAI/challenge@v2.0.0"
opentype-challenge generate --level 3 --n 100000 --seed 1 --out d3.jsonl                  # decisions
opentype-challenge generate --family invoice_approval --level 6 --n 20000 --out inv6.jsonl
opentype-challenge generate --track longctx --level 2 --n 2000 --out lc2.jsonl
opentype-challenge generate --track ops --n 20000 --out ops.jsonl                         # all levels
opentype-challenge generate --track sql --level 4 --n 20000 --out sql4.jsonl
opentype-challenge generate --track paint --n 20000 --out paint.jsonl                     # levels 1-2
```

Each line holds `{"track", "family", "level", "request", ...}`:

- **Read tracks** (`decisions`, `longctx`) add `gold`, which maps each question id to
  `[type, options, probabilities]`. These are exact soft targets: 1.0 on the rule's answer
  when every needed fact is stated, and otherwise the exact posterior over the unstated
  facts. `request` is the exact `/v1/systemone` body.
- **Harness tracks** (`ops`, `sql`, `paint`) add `oracle`: the raw outputs of a reference
  agent that sees only the conversation and reaches loss 0. `request` is the served body,
  `{"harness", "version", "task", "limits", "seed"}`. The `task` holds hidden env state,
  such as the world, the tables or the checks. Your model never sees it. It sees only the
  system prompt and the observations.

Levels: decisions 1–8 (the ladder in `/v1/status` shows which are active), longctx 1–5,
ops 1–4, sql 1–4, paint 1–2 publicly. Paint level 3 (`depict`) exists only in the private
bank.

To turn an oracle row into chat training examples, replay it through the harness. The
harness rebuilds the exact messages the duel sends, images included:

```python
import asyncio, json, sys
from opentype_challenge import harness, tracks

def conversations(row):
    env, outputs, seen = tracks.ENVS[row["track"]], iter(row["oracle"]), []
    async def generate(messages, seed):
        output = next(outputs)
        seen.append(messages + [{"role": "assistant", "content": output}])
        return output
    asyncio.run(harness.run_episode(env, row["request"], generate))
    return seen  # one training example per turn

for line in sys.stdin:
    row = json.loads(line)
    if row.get("oracle"):
        for example in conversations(row):
            print(json.dumps({"messages": example}))
```

## 3. Training advice

**Reads.** Train on the read protocol itself: the pinned `structured_server.py`, a
template-seeded canvas and a single step. The targets are soft, so use a loss on the answer
distribution (soft-target cross-entropy) rather than hard labels. Calibration on
underdetermined items counts as much as accuracy on determined ones. A read with
`label_mass < 0.5` or an argmax off the label forfeits with loss 1. For longctx, vary the
position of the target record, and include corrections and look-alike ids. A correction
written after a record overrides it, and the last correction wins.

**Multi-turn JSON tool use (`ops`, `sql`).**

- Emit exactly one JSON object per turn: `{"tool": "<name>", "args": {...}}`. Fences are
  tolerated, but anything unparseable costs a turn. Keep outputs short. The turn budget is
  12, and the per-turn budget is 512 tokens for ops and 1024 for sql.
- In ops, look records up before you act. Writes are irreversible. The written policy
  decides, never the customer's claims. `finish` with the right outcome is half the loss.
  A correct denial with no writes scores 0.
- In sql, read the schema in the system prompt. Date functions and `random()` are
  unavailable, so compare ISO dates as strings. Give the final `answer` in the requested
  shape: a number, a name, or a list of names in the right order.
- Use the oracle transcripts for supervised fine-tuning. Then add on-policy rollouts
  scored by `harness.replay` (loss 0 is the target) to learn recovery from error
  observations.

**Vision (`paint`).** The model sees a 256×256 PNG after each draw and must place exact
palette colours, counts, sizes, regions and relations. Train on the rendered canvases the
harness produces, so the model learns to correct what it sees. For `depict`, briefs are
scored by a strict, literal judge. Written words in the picture never pass an item, and
the rubric always contains "no letters, words or numbers".

**Do not overfit templates.** Sealed families, teacher prose and teacher stories are
paraphrases you have never seen. Augment with your own paraphrases, and keep the
structured spec fixed.

## 4. Publish and submit

1. Push the weights and the base `config.json` to a public, ungated repo. Note the full
   commit sha.
2. Submit with your registered hotkey:

```bash
opentype-challenge miner submit \
    --api https://<cortex-master>/challenge/opentype \
    --repo you/model --revision 0123456789abcdef0123456789abcdef01234567 \
    --wallet-name my-wallet --wallet-hotkey my-hotkey
# or: --seed-file hotkey.seed   (32-byte hex sr25519 mini-secret)
opentype-challenge miner status --api https://<cortex-master>/challenge/opentype --id s_...
```

The signed message is `opentype-submit-v1|<public key hex>|<manifest digest>|<nonce>|<exp>`.
States are `queued` (including `judging`), `crowned`, `rejected` (`early stop`,
`no certified gain`, a sha256 mismatch or a download failure) and `failed`
(infrastructure, after 3 attempts; submit again). The job's `verdict.tracks` shows your
`g`, `se` and loss per track.

## 5. Rules that cost you a duel

- The repo must stay public and at that commit until your duel has run.
- One open submission per hotkey. The queue holds at most 4 submissions (`429` when it is
  full).
- The hotkey must be registered on the subnet.
- A byte-identical copy of the champion is refused. A near copy cannot pass the 5 % bar.
- If two submissions beat the same champion, the earlier intake is crowned first, and the
  later one duels the new champion.

## 6. Getting paid

A crown creates an entitlement of `g_LCB / g_min` epochs of the challenge's share. It is
paid first in, first out, and it is **kept after you are dethroned**. Once the operator
schedules the two lanes, quality credits (old ones too, at their full amount) are paid
from 0.75 of each epoch and runtime credits from 0.25. `GET /v1/leaderboard`
shows each crown's entitlement, paid and outstanding amounts.

## 7. The runtime lane (25 %)

A second, independent competition: serve the current quality champion's weights faster
with vLLM options. It opens only after the operator publishes a calibration measured on the
reference hardware; until then `GET /v1/runtime` says `"open": false` and submissions are
refused.

```bash
opentype-challenge miner runtime-submit --api https://<gateway>/challenge/opentype \
  --options '{"max_num_seqs": 128, "enable_prefix_caching": true}' \
  --wallet-name w --wallet-hotkey h
```

- Only the options listed in `GET /v1/runtime` are accepted. No command line, environment,
  image, plugin, reader or tokenizer. The lane measures the NVFP4 champion on B300 only, so
  it stays closed until the champion is migrated and a B300 calibration is published.
- A kernel, where the published calibration lists its slot (`kernel_slots`), is one
  Triton file for the `rms_norm` slot, sent as `--kernel-file`: only imports of `triton`,
  `triton.language` and `math`, constants, and exactly one `@triton.jit def rms_norm_kernel
  (x_ptr, w_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, eps, BLOCK_SIZE)`
  computing `x * rsqrt(mean(x²) + eps) * w` per row (fp32 statistics, output in the input
  dtype). It is parsed at intake and never imported outside a sandbox; it is compiled
  offline for the calibrated arch in a separate network-blocked sandbox (a compile failure
  rejects it) and runs only in fresh, network-blocked, secret-free GPU sandboxes. A
  kernel whose answers diverge from stock beyond the calibrated tolerance is rejected.
- The signature binds the challenge, the lane, the champion you target, the calibrated
  profile and your options. When the quality champion changes, an open runtime submission
  expires, even mid-measurement: sign a new one against the new champion. Combinations the
  pinned vLLM refuses (`max_num_batched_tokens` below `max_num_seqs`, or chunked prefill
  off with `max_num_batched_tokens` below 131072) are refused at intake; options that fail
  to start a server after the reference started are rejected.
- A trusted worker measures the incumbent, your options, then the incumbent again, block
  after block, on one exclusive GPU, and reports each task's raw output and latency. The container
  scores every output against gold; a task counts only if it is right and within its
  cell's latency SLO, divided by wall time. Your gain is the fixed-weight mean over cells of
  `ln(goodput_C / max(goodput_B, goodput_B'))`; the crown needs its 99 % block-bootstrap
  lower bound above the published margin, no p95 latency regression beyond the tolerance in any block, and no half-Brier or
  accuracy regression against stock vLLM on the same cases, on every measured track. Metrics you declare are ignored.
- Credit: `g_LCB x credit_per_log_gain` epoch-masses, capped per crown, paid FIFO from the
  runtime budget (0.25 of each epoch). Only gain above the best gain already paid on the
  profile is paid, so recertifying the same options on new weights crowns but pays nothing.
- One open submission per hotkey in each lane; you can hold both.

## 8. Checking the duel

After the window closes, `GET /v1/windows/<id>` reveals the secret and the jobs, and
`GET /v1/windows/<id>/bank` publishes the bank. Rebuild your duel's cases and compare the
digest:

```bash
opentype-challenge audit --api https://<cortex-master>/challenge/opentype \
    --window 3 --window-secret <revealed hex>
```

Past banks are also useful as extra training data. A bank is never reused in a later
window.
