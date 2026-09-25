# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-25

The challenge now measures production usability across five tracks, with private per-window
teacher content and a VLM judge for free-form pictures.

### Added

- Tracks and plan (`tracks.py`): `decisions`, `longctx`, `ops`, `sql` and `paint`,
  interleaved so that every prefix of a duel holds the tracks in proportion.
  `OPENTYPE_PLAN` configures them. The default is 0.35/4,000, 0.25/800, 0.15/1,000,
  0.10/1,000 and 0.15/600 (7,400 cases).
- `longctx`: 8k–100k-token dossiers with corrections, near-duplicate ids and a uniform
  target depth. The gold is the exact posterior, recomputed from the text by `solve`.
- Multi-turn harness (`harness.py`): one JSON tool call per turn, at most 12 turns, and only
  the latest image kept in the context. The worker plays the episode, and the container
  scores only by `replay` from the raw outputs.
- `ops`: a store world with a written policy, 8 tools and customer pressure at level 4. The
  gold is the exact write set and outcome.
- `sql`: an in-memory SQLite analyst task behind an authorizer sandbox (read-only, no
  clock, random or extensions, a VM step limit). The gold is the unique answer of the
  reference SQL.
- `paint`: a 256×256 canvas that the model sees after every draw. Levels 1–2 are scored by
  exact pixel checks. Level 3 (`depict`) is graded by a VLM judge on the container's own
  render, with a no-text rubric item and blind, side-ordered calls. Unjudged pairs are
  dropped on both sides.
- Teacher gateway client and bank builder (`teacher.py`). A teacher writes sealed families,
  prose records, customer stories and drawing briefs. Text is admitted only after an exact
  round trip through two extractor model families, and a rubric only after a blank-canvas
  negative control. The token is re-read on every call and never logged.
- Window banks: a background task builds the next window's bank. Windows auto-rotate once
  it is ready (`OPENTYPE_WINDOW_HOURS`). The bank digest is committed when a window opens,
  and the bank is published at `GET /v1/windows/{id}/bank` when it closes.
- A drand beacon is mixed into the job seed, fetched once per job and published with the
  window.
- Composite scoring: a weighted mean of the per-track log ratios, the per-track regression
  guard (`g_t + Z99·se_t ≥ −ln 1.02` for tracks with ≥ 30 pairs), and early stop on the
  composite.
- The `judging` job state and the `judgments` table. `complete` judges inline for up to
  20 s.
- `generate --track {decisions,longctx,ops,sql,paint}`. Harness rows carry `oracle`
  transcripts from reference policies that see only the conversation.
- `audit` rebuilds jobs with their bank, beacon, plan and judge flag, and checks the
  bank digest.
- Worker: `--max-model-len` (default 131072) and `--limit-mm-per-prompt '{"image": 1}'`
  for vLLM. Harness cases run through the chat endpoint.
- `tests/test_shortcuts.py`: a bag-of-words probe must not beat the majority baseline.

### Changed

- Case pages carry `track` and are bounded to 6 MiB. Answer items may carry `transcript`.
  Answers responses add `judging`. Lease responses add `plan`.
- `/v1/status` adds `plan`, `tracks`, `teacher` and `window.bank_digest`.
  `constants.duel_cases` is the plan's total.
- Window listings add `bank_digest`. Revealed jobs add `plan`, `beacon` and `judge`.
- Verdicts add `tracks`, `track_guard` and `unjudged`. `levels` and the ladder cover the
  decisions track only.
- `OPENTYPE_DUEL_CASES` applies only when `OPENTYPE_PLAN` is unset and gives a
  decisions-only plan.
- The case cache is bounded by bytes (128 MiB).
- The state schema moves to version 2 (`PRAGMA user_version`). A v1 database is migrated
  in place.

### Security

- Model output is never executed, except read-only SQL in the sandbox. Paint commands are
  validated data.
- The worker's harness observations are never trusted. Every outcome is replayed in the
  container.

## [1.0.0] - 2026-09-24

### Added

- Cortex challenge container (contract v1) for the `opentype` challenge: `GET /health`
  (readiness), `GET /version` (liveness) and `GET /internal/v1/get_weights`, whose first
  answer per epoch is persisted, replayed byte for byte, and uses `full_share_mass = 1.0`
  so unpaid mass burns.
- Token files are read lazily on each request. The server starts without secrets for the
  supervisor canary, and a route without its token answers `503`.
- TD-Exact generator: 4 typed-decision families and 8 difficulty levels. Gold is exact:
  the rule is evaluated by an interpreter and a compiled evaluator (N-version), with an
  exact Bayes posterior for unstated facts. Cases pass a render/extract round trip and a
  criteria parse check, and carry reordered and mirror probes. A text-level reference
  solver checks every gold.
- `opentype-challenge generate`, which writes training data with exact soft targets.
- Commit-reveal windows, per-job seeds, `cases_sha256` evidence and
  `opentype-challenge audit`.
- Paired scoring: half-Brier with forfeit, `g = ln(Σ L_champion / Σ L_challenger)`,
  delta-method LCB99 on both halves, crown bar `g_min = −ln 0.95`, a regression guard on
  retired levels and early stop.
- Frontier ladder with duel mix weighted by the champion's error, and level retirement at a
  Wilson LCB99 of at least 99.9 %.
- Ledger of certified gain: entitlement `g_LCB / g_min` epoch-masses per crown, paid first
  in, first out, kept after dethronement, with an optional per-window cap. The base model is
  never paid.
- Intake with sr25519 signatures, nonces, a registration check against the master
  metagraph, one open submission per hotkey, a `max_pending` queue bound and clone
  refusal. The earliest intake wins between simultaneous winners, and stale jobs are
  re-targeted at the new champion.
- Worker API (lease, heartbeat, cases, answers, complete, fail) and admin API (window
  rotation, ladder, crown pause, job requeue).
- B300 duel worker: anonymous Hugging Face download with sha256 verification at the
  committed revision. Support files come from the pinned base, and verified champion
  weights are cached. It launches two `vllm serve` processes plus the pinned
  `structured_server.py` for each side, reads both sides in pairs, and separates
  challenger failures from infrastructure failures (3 attempts).
- Miner CLI: `miner submit` (seed file or bittensor wallet) and `miner status`.
- Dockerfile with `server` and `worker` targets, pinned by digest, with the Cortex labels.
  The image creates `/data` owned by `65532:65532`.
- GitHub Actions: `ci.yml` (lint, types, tests and a canary of the server image),
  `images.yml` (build, push `sha-<commit>` and `edge`, attest the exact digest) and
  `release.yml` (annotated tag aliases the tested digests to `vX.Y.Z` and `stable`).

[2.0.0]: https://github.com/OpentypeAI/challenge/releases/tag/v2.0.0
[1.0.0]: https://github.com/OpentypeAI/challenge/releases/tag/v1.0.0
