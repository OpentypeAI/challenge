# Architecture

## Components

| Component | Where | Role |
| --- | --- | --- |
| challenge container | `ghcr.io/opentypeai/challenge` (target `server`) | Cortex contract v1. Handles intake, windows, the duel queue, scoring, the crown rule, the ledger and `get_weights`. It uses SQLite on `/data` and needs no GPU. |
| duel worker | `ghcr.io/opentypeai/challenge-worker` (target `worker`) | A B300 host. It leases a job, downloads and verifies both models, serves them with vLLM and the pinned `structured_server.py`, reads every case on both sides and posts the answers. |
| Cortex master | upstream | Proxies `/challenge/opentype/*`, polls `get_weights` once per completed epoch, signs leaves and seals the epoch. |
| miner CLI | `opentype-challenge miner` | Builds and signs a manifest of a public HF commit, then posts it. |
| auditor | `opentype-challenge audit` | Regenerates the cases of a revealed window and compares them with the published digests. |

## Modules (`src/opentype_challenge/`)

| Module | Contents |
| --- | --- |
| `generator.py` | 4 families (support ticket, invoice approval, security alert, agent trace) and levels 1–8. It holds the rule interpreter, the compiled evaluator (N-version), the parser, the renderer and extractor (round trip), the exact posterior, the probes and the text-level reference solver. It uses only the standard library. |
| `bank.py` | Window commitment, the per-job seed `HMAC(secret_w, job_id \| digest)`, level sampling, `job_case(seed, mix, i)` and the cases digest. |
| `scoring.py` | Half-Brier with forfeit, the paired log-ratio with delta-method LCB99, the crown rule, the regression guard, early stop, the Wilson retirement bound, the ladder and the duel mix. |
| `ledger.py` | Integer entitlements (1e-9 epoch-mass) and FIFO payment. |
| `store.py` | The SQLite state machine: windows, nonces, submissions, jobs, results, champions, per-level champion statistics, entitlements, epochs and payments. |
| `app.py` | FastAPI routes, auth, body limits and the metagraph cache. |
| `worker.py` | Weight assembly and verification, the injectable `Launcher` (`VllmLauncher` in production), the paired read loop, heartbeats and failure classification. |
| `miner.py`, `crypto.py` | Manifests, sr25519 signing and verification, and SS58. |
| `pins.py` | Base model revision and file digests, vLLM image, `structured_server.py` URL and sha256. |
| `cli.py` | `serve`, `worker`, `generate`, `audit`, `miner submit`, `miner status`. |

## Life of a submission

1. **Intake.** A miner signs `opentype-submit-v1|<pubkey hex>|<manifest digest>|<nonce>|<exp>`
   with its sr25519 hotkey. The container validates the manifest, then the signature,
   expiry (at most 300 s) and nonce. It then checks registration (the master's
   `/v1/metagraph/latest`, cached for 30 s), allows one open submission per hotkey and at
   most `max_pending` queued in total, and refuses a byte-identical clone of the champion.
   The submission receives an `intake` number, which is a total order.
2. **Job.** A job binds the submission to the current champion, the open window and a level
   mix. Its seed is `HMAC(secret_w, job_id | challenger digest)`. The job is re-targeted at
   lease time, so a job that waited in the queue always duels the current champion under
   the current ladder.
3. **Lease.** The worker receives both manifests and a 30-minute lease. Heartbeats extend
   the lease every 5 minutes, and every batch of answers renews it.
4. **Duel.** The worker verifies the pinned base support files once and keeps them, and
   keeps the current champion's verified weights between jobs. It downloads the challenger
   anonymously at the committed revision and checks every sha256. It starts
   `vllm serve` and `structured_server.py` for each side, pages cases (100 per page),
   reads each case on both sides with the same body and seed, and posts answers in batches
   under 900 KiB. The container scores every answer as it arrives and returns `continue`.
   It returns `false` on early stop, when the champion changed, or when every case is
   paired.
5. **Complete.** The container computes the verdict and records the champion's per-level
   statistics, retires mastered levels and settles the queue in intake order. A job scored
   against a champion that has since changed is superseded and re-queued against the new
   one. A loser is rejected. A winner is crowned only once every earlier intake against the
   same champion is settled, so earliest intake wins.
6. **Crown.** The challenger becomes the champion, and its per-level accuracy seeds the next
   duel mix. The ledger receives an entitlement of `g_LCB / g_min` epoch-masses.
7. **Weights.** The master's first `get_weights` call for an epoch pays outstanding
   entitlements first in, first out, up to one epoch-mass. The rest burns
   (`full_share_mass = 1.0`). The body is persisted, and every later call returns the same
   bytes.
8. **Reveal.** When the operator rotates the window, the old secret is published. Anyone can
   then regenerate every served case and compare the result with the worker's
   `cases_sha256`.

## Failure classes

| Failure | Classified as | Effect |
| --- | --- | --- |
| manifest invalid, sha256 mismatch, repo private, gated or missing, `config.json` differs from the base | challenger | rejected, no retry |
| reader `4xx` on one case (invalid output) | challenger | that case forfeits (loss 1 per decision) |
| hub or network error, reader `5xx`, a server that never became healthy, any unexpected exception, a champion download problem | infrastructure | re-queued, and `failed` after 3 attempts |
| lease expired (worker died) | infrastructure | re-queued at the next lease call, with the same 3-attempt limit |

## Contract v1 conformance

- The container serves port 8000 as UID 65532 with a read-only root, a `/tmp` tmpfs and
  `/data` as the only writable path. The image creates `/data` owned by `65532:65532`.
- The container reads token files lazily on each request, so `serve` starts without them
  (supervisor canary). A route whose token file is missing answers `503`.
- `/version` is liveness and answers without state or secrets. `/health` is readiness: it
  requires the state volume to be writable and the internal token to be readable.
- `get_weights` requires the internal bearer (`401`) and the slug header (`403`). The first
  answer for an epoch is final and replayed byte for byte, and `full_share_mass` is `1.0`.
- The container needs no outbound access except `GET {CHALLENGE_MASTER_URL}/v1/metagraph/latest`.
  The worker reaches the container only through the master proxy, so worker request bodies
  stay under 1 MiB and responses under 8 MiB.
