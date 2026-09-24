# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/OpentypeAI/challenge/releases/tag/v1.0.0
