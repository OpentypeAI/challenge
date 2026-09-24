# OpenType challenge

The `opentype` challenge of the Cortex subnet. It implements **TD-Exact**: miners fine-tune
DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) and the challenge rewards verifiable
gains on synthetic typed decisions with exact gold.

- **Exact gold.** Every case is a closed world, a decision-list rule and a rendered record,
  so the gold is `rule(world)`. When the record leaves facts out, the gold is the exact
  Bayes posterior. The label has no noise, and 100 % accuracy is reachable.
- **Paired duels on real weights.** One B300 serves the champion and the challenger in
  BF16 through the pinned vLLM structured reads. Both answer the same cases, and each case
  is scored with half-Brier against the gold.
- **Crowns are certified.** The challenger is crowned when the lower 99 % bound of
  `g = ln(champion loss / challenger loss)` reaches `-ln 0.95` on both halves of the duel
  and the retired levels do not regress.
- **A ledger pays certified gain.** Each crown is owed `g_LCB / g_min` epochs of the
  challenge's share, paid first in, first out. What nobody earns burns.

```text
 miner ── signed manifest ──▶ master /challenge/opentype/v1/submissions ─┐
                                                                         ▼
                        ┌──────────── challenge container (this repo, "server") ─────────┐
                        │ intake · windows (commit-reveal) · duel jobs · scoring · ledger │
                        └──▲───────────────────────────┬───────────────────▲────────────┘
   worker bearer, public API                           │ get_weights        │ /v1/status,
   lease · cases · answers · complete                  ▼ (internal token)   │ /v1/windows
 ┌─────────── B300 worker ("worker" image) ───────┐  master: signs leaves,  anyone: audit
 │ HF download + sha256 · vllm serve ×2 ·         │  seals the epoch
 │ structured_server.py ×2 · paired reads         │
 └────────────────────────────────────────────────┘
```

## Miner quickstart

```bash
uv tool install "opentype-challenge[miner,wallet] @ git+https://github.com/OpentypeAI/challenge@v1.0.0"
opentype-challenge generate --level 3 --n 100000 --out train.jsonl   # exact soft targets
# fine-tune, then push safetensors + the base config.json to a public HF repo
opentype-challenge miner submit --api https://<cortex-master>/challenge/opentype \
    --repo you/model --revision <40-hex commit> --wallet-name w --wallet-hotkey h
opentype-challenge miner status --api https://<cortex-master>/challenge/opentype --id s_...
```

See [docs/miner.md](docs/miner.md).

## Operator quickstart

1. Put `internal.token`, `admin.token` and `worker.token` in
   `<BASE_CHALLENGE_SECRETS_DIR>/opentype/`.
2. Add the `opentype` entry to the Cortex challenge registry, image
   `ghcr.io/opentypeai/challenge`.
3. Run `ghcr.io/opentypeai/challenge-worker` on a B300 with the worker token.

See [docs/operator.md](docs/operator.md).

## Development

```bash
uv sync --all-extras --group dev
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest
docker build --target server -t opentype-challenge:local .
```

## Documentation

- [Architecture](docs/architecture.md)
- [Mechanism](docs/mechanism.md): scoring, power, ladder, ledger, anti-cheat and residual risks
- [Miner guide](docs/miner.md)
- [Operator guide](docs/operator.md)
- [API reference](docs/api.md)
- [Changelog](CHANGELOG.md)

Licensed under Apache-2.0.
