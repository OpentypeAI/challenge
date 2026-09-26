# Architecture

## Components

| Component | Where | Role |
| --- | --- | --- |
| challenge container | `ghcr.io/opentypeai/challenge` (target `server`) | Cortex contract v1: intake, windows and banks, the duel queue, case building, replay, judging, scoring, the crown rule, the ledger and `get_weights`. SQLite on `/data`, no GPU. |
| teacher gateway | operator-provided, OpenAI-compatible | `POST /v1/chat/completions` for the teacher, the extractors and the judge. Optional. |
| duel worker | `ghcr.io/opentypeai/challenge-worker` (target `worker`) | A B300 host. It leases a job, downloads and verifies both models, runs `vllm serve` and the pinned `structured_server.py` for each side, runs every case on both sides and posts the results. |
| Cortex master | upstream | Proxies `/challenge/opentype/*`, polls `get_weights` once per epoch, signs leaves and seals the epoch. |
| miner CLI | `opentype-challenge miner` | Builds and signs a manifest of a public HF commit, then posts it. |
| auditor | `opentype-challenge audit` | Rebuilds the cases of a revealed window and compares their digests. |

## Modules (`src/opentype_challenge/`)

| Module | Contents |
| --- | --- |
| `generator.py` | The decisions track: 4 public families, sealed-family JSON (`family_from_json`, `family_to_json`), levels 1–8, the rule interpreter and compiled evaluator (N-version), the parser, the renderer and extractor (round trip), the exact posterior, probes, prose states (`sample_known`, `make_case(..., prose)`) and the text-level solvers `solve` and `solve_known`. Standard library only. |
| `longctx.py` | The long-context track: dossiers, corrections, near-duplicate ids and `solve(body)`. |
| `harness.py` | `Env`, `parse_action`, `chat_messages`, `run_episode` (worker side) and `replay` (container side). |
| `ops.py`, `sqltask.py`, `paint.py` | The three envs: worlds and tasks, tools, `policy` / reference SQL / pixel checks, `reference_policy` oracles. `paint` also holds `render_png` and the judge prompt (`judge_request`, `judge_loss`). |
| `teacher.py` | The gateway client (retries, structured output, SSE, schema validation, token read per call), `round_trip`, `build_bank` and `judge_png`. |
| `bank.py` | `BankItem`, `Bank`, `bank_digest`, the window commitment, `drand_beacon`, `job_seed`, level sampling and the cases digest. |
| `tracks.py` | `TRACKS`, `ENVS`, `TrackPlan`, `DEFAULT_PLAN`, `effective_plan`, `track_of`, `job_case`, `score_item` and `solve_body`. |
| `scoring.py` | Half-Brier with forfeit, `harness_score`, the paired log ratio, `composite`, the per-track guard, the crown rule, early stop, the Wilson bound, the ladder and the duel mix. |
| `ledger.py` | Integer entitlements and FIFO payment. |
| `runtime.py` | The runtime lane: lane budgets (750 000 000 / 250 000 000 units), the vLLM option allowlist and its fixed flags, the operator `Calibration`, the private workload cells, and the pure `verdict` over B / C / B' measurements and fidelity sums. |
| `store.py` | The SQLite state machine: windows and banks, nonces, submissions, jobs (plan, beacon, judge flag), results with `track`, judgments, champions, level statistics, entitlements, epochs. It holds a byte-bounded case cache (128 MiB) and migrates older databases in place in one transaction (`PRAGMA user_version` 3), adding each missing column found by `PRAGMA table_info`, so a v3 file that an older binary re-stamped as v2 migrates without a duplicate column. v3 adds a `lane` to submissions, jobs and entitlements (existing rows are `quality`), `runtime_incumbents` and `runtime_tasks`. Opening a v3 file with an older binary is not supported: it does not know the lanes. |
| `app.py` | FastAPI routes, auth, body limits, the metagraph cache, teacher wiring, and background tasks for judging, settling, auto-rotation and the bank builder. |
| `worker.py` | Weight assembly and verification, `VllmLauncher` (`--max-model-len`, `--limit-mm-per-prompt`), the read and episode loop, heartbeats and failure classification. |
| `miner.py`, `crypto.py` | Manifests, sr25519 signing and verification, SS58. |
| `pins.py` | The base revision and file digests, the vLLM image, the `structured_server.py` URL and sha256. |
| `cli.py` | `serve`, `worker [--lane runtime]`, `generate` (every track), `audit`, `miner submit`, `miner runtime-submit`, `miner status`. |

## Life of a submission

1. **Intake.** The miner signs `opentype-submit-v1|<pubkey hex>|<manifest digest>|<nonce>|<exp>`.
   The container validates the manifest, the signature, the expiry (at most 300 s) and the
   nonce. It checks the registration against the master metagraph (cached 30 s), allows one
   open submission per hotkey and `max_pending` in total, and refuses a clone of the
   champion. Each submission gets a total-order `intake` number.
2. **Job.** A job binds the submission to the champion and the open window. At lease time,
   it is re-targeted to the current champion, window, ladder mix, effective plan and judge
   flag. The first lease fetches the drand beacon and stores it. The seed is
   `job_seed(secret_w, job, digest, beacon)`.
3. **Lease.** The worker receives both manifests, the plan, the case count and a 30-minute
   lease. Heartbeats extend the lease every 5 minutes, and so does every answers batch.
4. **Duel.** The worker verifies the base support files and the champion (cached), then
   downloads the challenger anonymously and checks every sha256. It starts vLLM and the
   structured server for each side. It pages cases (`{index, track, body}`, 100 per page,
   under 6 MiB) and runs each case on both sides:
   - read tracks post the body to `reader/v1/systemone`;
   - harness tracks run `harness.run_episode` against `chat/v1/chat/completions`
     without temperature overrides or per-request seeds (unsupported by the pinned
     diffusion engine) and keep the raw outputs. Case generation and replay are
     reproducible; model-generated transcripts are not guaranteed deterministic.

   Items are posted in batches under 900 KiB.
5. **Scoring as answers arrive.** For every item, the container rebuilds the case from the
   seed and the bank. Read items are scored with half-Brier. Harness items go through
   `replay`. A `depict` item is replayed, rendered by the container and stored as a
   pending judgment. The container answers `continue: false` on early stop, when the
   champion changes, or when every case has both sides.
6. **Complete.** When the job has pending judgments, it moves to `judging`. The container
   judges inline for up to 20 s, and a background task (every 30 s) finishes the rest.
   Unjudged cases are removed on both sides. The job is then settled: the verdict, the
   champion's decisions-level statistics, retirement, and the queue in intake order. A job
   scored against a replaced champion is superseded and re-queued. A loser is rejected. A
   winner is crowned once every earlier intake against the same champion is settled.
7. **Crown.** The challenger becomes the champion, and the ledger receives `g_LCB / g_min`
   epoch-masses.
8. **Weights.** The first `get_weights` of an epoch pays FIFO up to one epoch-mass, burns
   the rest and persists the body.
9. **Reveal.** Rotation, manual or automatic once the next bank is ready and the window is
   at least `OPENTYPE_WINDOW_HOURS` old, publishes the old secret, its jobs (digest, mix,
   plan, beacon, judge flag, `cases_sha256`) and its bank. Anyone can then run the audit.

In parallel, when a teacher is configured, the builder task keeps one **next bank** ready.
It builds the bank, stores it unsealed in `meta.next_bank`, and the next rotation seals it
into the new window.

## Failure classes

| Failure | Classified as | Effect |
| --- | --- | --- |
| manifest invalid, sha256 mismatch, repo private, gated or missing, `config.json` differs from the champion's, NVFP4 tensor layout mismatch | challenger | rejected, no retry |
| reader or chat `4xx` on one case, or a chat reply without `choices[0].message.content` | challenger | that case forfeits for that side (loss 1 per decision) |
| unparseable action, invalid tool call, turns exhausted | challenger (in-episode) | an error observation, or the episode ends and the final state is scored |
| a transcript longer than the turn limit or an output over 8,192 characters | worker or challenger | that side forfeits the case |
| hub or network error, reader or chat `5xx` or transport error, a server that never became healthy, a champion download problem, an unexpected exception | infrastructure | re-queued, and `failed` after 3 attempts |
| lease expired (worker died) | infrastructure | re-queued at the next lease call, same 3-attempt limit |
| a judge that cannot give a valid verdict after 5 attempts | judge | the case is `unjudged` and dropped on both sides |
| a bank build error | teacher | logged and retried after 10 minutes; the current window keeps running and does not rotate automatically |
| drand unreachable | beacon | the job uses the v1 seed and records a `null` beacon |

## Contract v1 conformance

- The container serves port 8000 as UID 65532 with a read-only root, a `/tmp` tmpfs and
  `/data` as the only writable path.
- Token files are read lazily, so `serve` starts without secrets (supervisor canary). A
  route whose token file is missing answers `503`. The teacher token is checked at startup,
  and without it the teacher is off.
- `/version` is liveness. `/health` is readiness: the state volume must be writable and the
  internal token readable.
- `get_weights` requires the internal bearer (`401`) and the slug header (`403`). The first
  answer for an epoch is final and replayed byte for byte, with `full_share_mass: 1.0`.
- Outbound access: `GET {CHALLENGE_MASTER_URL}/v1/metagraph/latest`, plus, optionally,
  `https://api.drand.sh/public/latest` and `{OPENTYPE_TEACHER_URL}/v1/chat/completions`.
- The worker reaches the container only through the master proxy. Request bodies stay
  under 1 MiB and responses under 8 MiB (case pages are capped at 6 MiB).

## Two lanes

One challenge, one Cortex export, two independent competitions (see
[operator.md](operator.md#8-runtime-lane)):

- **quality** (75 %): everything above. Weights, tracks, half-Brier and crowns are unchanged;
  `status.champion` is the quality champion.
- **runtime** (25 %): the quality champion's weights served faster by allowlisted vLLM
  options only. Its own queue, incumbent, credits and FIFO.

Lanes are not tracks: tracks are renormalised inside a duel, lanes never are. From the
scheduled epoch on, `weights` pays each lane with `ledger.pay` from its own budget; a lane
with nothing owed burns its share, it never goes to the other lane. Before that epoch, and
for every persisted epoch, the historical single-budget rule and bytes are unchanged. A
hotkey paid in both lanes gets the sum.

A runtime job pins two identities, the quality champion (`champion_id`, always the
submission's signed target) and the runtime incumbent (`incumbent_id`, 0 = stock), plus the
calibration it runs under. The signed target is never moved: every path that would requeue
or retarget a runtime job (a quality crown, a retry, an expired lease, `NO_DECISION`, a stale
completion) expires it once the target is no longer the champion, and the lane goes back to
stock on the new weights. While a runtime job waits and a runtime worker polled in the last
two minutes, quality leases pause after `runtime_every` (4) of them so the leased quality
jobs drain and the runtime job gets the GPU; quality then gets its `runtime_every` again. Recertifying on new weights crowns but pays only certified gain above the best
already paid on the profile.

The runtime verdict (`runtime.verdict`) is recomputed by the container. The worker reports
each timed task's raw output (read answers or harness transcript) and latency, and each
run's monotonic seconds; it reports no success flag or count. The container rebuilds each
case from the job's seed and counts a task only if its output scores right against gold
(`runtime.task_ok`: every read answer valid with the untied gold argmax, or an episode that
replays to zero loss) within the cell's SLO. Fidelity (stock vs candidate on the same
cases) covers every track a cell measures, decisions always, each guarded separately. Anything malformed, non-finite, missing, drifting, reordered or
non-quiescent is `NO_DECISION`: an infrastructure retry with no credit. A candidate that
completes nothing, regresses half-Brier, accuracy or p95 latency, or shows no gain above the
calibrated margin on healthy infrastructure is rejected.
