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
in the repo is ignored. Code, pickles and custom configs are never loaded.

## 1. Training data

```bash
uv tool install "opentype-challenge[miner,wallet] @ git+https://github.com/OpentypeAI/challenge@v1.0.0"
opentype-challenge generate --level 3 --n 100000 --seed 1 --out l3.jsonl
opentype-challenge generate --family invoice_approval --level 6 --n 20000 --out inv6.jsonl
```

Each line is `{"family", "level", "request", "gold"}`:

- `request` is the exact body the duel sends to `/v1/systemone`. It contains `model`,
  `instructions`, `state`, `questions`, `samples: "auto"` and `seed`.
- `gold` maps each question id to `[type, options, probabilities]`. The probabilities are
  exact: 1.0 on the rule's answer when every fact it needs is stated, and the exact
  posterior over the unstated facts otherwise.

Families: `support_ticket`, `invoice_approval`, `security_alert`, `agent_trace`. Levels: 1–8
(see [mechanism.md](mechanism.md#1-cases-with-exact-gold)). `GET /v1/status` shows which
levels are active and the champion's accuracy on each, which tells you where the duel mix
is heading.

Train on the read protocol itself: the pinned `structured_server.py` template-seeded
canvas with a single step. The duel reads your model through that server. The target is
soft (exact probabilities), so a loss on the answer distribution (for example soft-target
CE) fits better than hard labels.

## 2. What the duel scores

Every decision is scored with half-Brier against the exact gold. A missing or invalid
answer, a `label_mass < 0.5` or an argmax off the label forfeits the decision with a loss
of 1. You win the crown when your summed loss is at least 5 % lower than the champion's at
99 % confidence, on both halves of the duel, and you do not regress on retired levels.
Being calibrated on underdetermined items counts as much as being right on determined
ones.

## 3. Publish and submit

1. Push the weights and the base `config.json` to a public, ungated repo. Note the full
   commit sha.
2. Submit with your registered hotkey:

```bash
opentype-challenge miner submit \
    --api https://<cortex-master>/challenge/opentype \
    --repo you/model --revision 0123456789abcdef0123456789abcdef01234567 \
    --wallet-name my-wallet --wallet-hotkey my-hotkey
# or: --seed-file hotkey.seed   (32-byte hex sr25519 mini-secret)
```

The CLI reads the file digests from the hub (the LFS sha256 for weights, a local hash for
small files), refuses a private or gated repo, and signs:

```
opentype-submit-v1|<32-byte public key hex>|<manifest digest>|<nonce>|<exp>
manifest digest = sha256 of canonical JSON {"files": {...sorted}, "repo": ..., "revision": ...}
```

It prints the submission id. Then follow it:

```bash
opentype-challenge miner status --api https://<cortex-master>/challenge/opentype --id s_...
```

States: `queued` (waiting, leased or re-queued), `crowned`, `rejected` (with a reason such
as `early stop`, `no certified gain`, a sha256 mismatch or a download failure) or `failed`
(infrastructure, after 3 attempts; submit again).

## 4. Rules that cost you a duel

- The repo must stay public and at that commit until your duel has run. A download failure
  is your fault.
- One open submission per hotkey. The queue holds at most 4 submissions (`429` when it is
  full, so retry later).
- The hotkey must be registered on the subnet (checked against the master's latest sealed
  metagraph).
- A byte-identical copy of the champion is refused. A near copy cannot pass the 5 % bar.
- If two submissions both beat the same champion, the earlier intake is crowned first, and
  the later one duels the new champion.

## 5. Getting paid

A crown creates an entitlement of `g_LCB / g_min` epochs of the challenge's share (a
minimal crown is worth one full epoch). The entitlement is paid first in, first out across
the following epochs, and it is **kept after you are dethroned**. `GET /v1/leaderboard`
shows each crown's entitlement, the amount paid and the amount outstanding.

## 6. Checking the duel

After the window rotates, `GET /v1/windows/<id>` reveals the secret. Regenerate your duel's
cases and compare the digest with the worker's:

```bash
opentype-challenge audit --api https://<cortex-master>/challenge/opentype \
    --window 3 --window-secret <revealed hex>
```
