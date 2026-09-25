# Miner guide

## What you submit

You submit a signed manifest of a **public, ungated** Hugging Face model repo at a full
40-hex commit. Only these files are read:

| File | Rule |
| --- | --- |
| `model.safetensors` or `model-NNNNN-of-NNNNN.safetensors` | your weights (BF16, same architecture as the base) |
| `model.safetensors.index.json` | required when the weights are sharded |
| `config.json` | must be byte-equal to the base revision's (sha256 `13b11d2f…c506`) |

Tokenizer, chat template, generation and processor files always come from the pinned base
`google/diffusiongemma-26B-A4B-it@f7f5b7f5fa82ffc52addd066915886d497f5517b`. Anything else
in the repo is ignored. Code, pickles and custom configs are never loaded. Keep the vision
tower: the `paint` track sends images.

## 1. What is measured

A duel runs the same cases on your model and on the champion. The default plan has 21,600
cases. `GET /v1/status` shows the live `plan` and track weights:

| Track | Cases | Weight | How your model is called | Loss |
| --- | --- | --- | --- | --- |
| `decisions` | 20,000 | 0.35 | structured read (`/v1/systemone`) | Σ half-Brier per case |
| `longctx` | 800 | 0.25 | structured read, 8k–100k-token state | Σ half-Brier |
| `ops` | 300 | 0.15 | chat, up to 12 turns, 512 tokens per turn | outcome and write-set diff |
| `sql` | 300 | 0.10 | chat, up to 12 turns, 1024 tokens per turn | 0 or 1 |
| `paint` | 200 | 0.15 | chat with images, up to 12 turns, 1024 tokens per turn | failed pixel checks, or a judge's rubric score |

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
paid first in, first out, and it is **kept after you are dethroned**. `GET /v1/leaderboard`
shows each crown's entitlement, paid and outstanding amounts.

## 7. Checking the duel

After the window closes, `GET /v1/windows/<id>` reveals the secret and the jobs, and
`GET /v1/windows/<id>/bank` publishes the bank. Rebuild your duel's cases and compare the
digest:

```bash
opentype-challenge audit --api https://<cortex-master>/challenge/opentype \
    --window 3 --window-secret <revealed hex>
```

Past banks are also useful as extra training data. A bank is never reused in a later
window.
