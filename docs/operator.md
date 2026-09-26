# Operator guide

Two things run: the **challenge container**, started by the Cortex challenge-supervisor on
the master host, and one or more **duel workers** on B300 hosts that you run yourself.

## 1. Secrets

Every token is a random string, one per file. None of them is ever baked into an image or
the repo.

```bash
dir="$BASE_CHALLENGE_SECRETS_HOST_DIR/opentype"
install -d -m 0700 -o 65532 -g 65532 "$dir"
(umask 077; for n in internal admin worker; do openssl rand -hex 32 >"$dir/$n.token"; done)
chown 65532:65532 "$dir"/*.token
```

| File | Used by | Without it |
| --- | --- | --- |
| `internal.token` | the master, for `GET /internal/v1/get_weights` | `get_weights` returns `503`, `/health` reports not ready, and the epoch share burns |
| `admin.token` | you, for `/v1/admin/*` | the admin routes return `503` |
| `worker.token` | the duel workers, for `/v1/worker/*` (copy it to each worker host) | the worker routes return `503` |
| `teacher.token` (optional) | the container, as the bearer for the teacher gateway | the teacher is off: no bank is built, so there are no sealed families, prose, stories or `depict`, and windows rotate only by hand |

```bash
(umask 077; printf '%s\n' "$GATEWAY_TOKEN" >"$dir/teacher.token"); chown 65532:65532 "$dir/teacher.token"
```

The three challenge tokens are read on every request, so you can rotate one by rewriting
the file without a restart. The teacher token is checked once at startup to decide whether
the teacher runs, then re-read on every gateway call. After startup you can rotate it in
place, but adding it later needs a restart. The container never logs the token or puts it
in an error message.

## 2. Registry entry

The Cortex registry (`deploy/challenges/registry.toml`) runs the server image:

```toml
[[challenge]]
id = "opentype"
image = "ghcr.io/opentypeai/challenge"
channel = "stable"                  # "edge" follows main; or pin = "sha256:..."
source = "https://github.com/OpentypeAI/challenge"
poll_seconds = 300
cpus = 2.0
memory_mib = 2048
pids = 256
proxy_body_limit = 5242880

[challenge.env]
OPENTYPE_MAX_PENDING = "4"          # queued submissions across all hotkeys
OPENTYPE_TEACHER_URL = "https://<gateway>"   # unset = teacher off
OPENTYPE_TEACHER_TOKEN_FILE = "/run/secrets/teacher.token"
# OPENTYPE_PLAN = '{"decisions": {"weight": 0.35, "cases": 4000}, ...}'
# OPENTYPE_WINDOW_ENTITLEMENT_CAP = "20"   # optional cap, in epoch-masses per window
```

The image already sets `CHALLENGE_SLUG=opentype`, `CHALLENGE_STATE_DIR=/data` and the three
`CHALLENGE_*_TOKEN_FILE` paths under `/run/secrets/`.

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHALLENGE_MASTER_URL` | `http://cortex-master:8080` | where the metagraph is read |
| `OPENTYPE_PLAN` | the 5-track default (7,400 cases) | JSON `{track: {"weight", "cases"}}` over `decisions`, `longctx`, `ops`, `sql`, `paint` |
| `OPENTYPE_DUEL_CASES` | `40000` | v1 knob: used only when `OPENTYPE_PLAN` is unset and the value is not the default, and then gives a decisions-only plan |
| `OPENTYPE_MAX_PENDING` | `4` | queued submissions across all hotkeys |
| `OPENTYPE_WINDOW_HOURS` | `24` | minimum window age before auto-rotation to a ready bank |
| `OPENTYPE_WINDOW_ENTITLEMENT_CAP` | none | cap on the entitlement created per window |
| `OPENTYPE_TEACHER_URL` | none | gateway base URL; unset turns the teacher off |
| `OPENTYPE_TEACHER_TOKEN_FILE` | `/run/secrets/teacher.token` | gateway bearer |
| `OPENTYPE_TEACHER_MODEL` | `cx/gpt-6-sol` | writes families, prose, stories and briefs |
| `OPENTYPE_EXTRACTOR_MODELS` | `cx/gpt-5.6-luna,cc/claude-sonnet-5` | round-trip extractors; keep 2 or more families |
| `OPENTYPE_JUDGE_MODELS` | `cx/gpt-6-sol` | depict judges; the loss is their mean |
| `OPENTYPE_TEACHER_CONCURRENCY` | `8` | concurrent gateway calls of the bank builder |
| `OPENTYPE_BANK_TARGETS` | `{"family": 4, "prose": 600, "ops_story": 200, "depict": 80}` | items per window; a partial object overrides only the listed kinds |

The gateway must speak the OpenAI chat-completions API. `cx/*` models get
`response_format: json_schema`. Other models get a `submit` tool. Both plain JSON and SSE
responses are accepted. A malformed teacher setting turns the teacher off with a warning.
It does not stop the container.

### Egress

The server container needs outbound HTTPS to:

- `CHALLENGE_MASTER_URL`, for the metagraph;
- `api.drand.sh`, for the beacon. Without it, jobs fall back to the v1 seed;
- the teacher gateway host, when the teacher is on.

Nothing else is needed. The supervisor verifies the labels
(`io.cortex.challenge.slug=opentype`, `io.cortex.challenge.contract=1`, source) and the
GitHub build-provenance attestation of the pulled digest. It then runs a canary without
secrets, which must answer `/version`, and replaces the container on the same `/data`
volume.

State is a single SQLite database, `/data/opentype.sqlite3` (WAL). It holds windows with
their secrets and banks, the next bank, submissions, jobs, results, pending judgments (with
PNGs), champions, the ledger and every persisted epoch body. A v1 or v2 database is migrated
in place at startup. Take a backup before you upgrade. Rolling back to a binary older than
the runtime lane is not supported on a migrated file: restore the backup instead. Back it up with SQLite's online backup API (for example
`sqlite3.connect(src).backup(dst)`), never with a plain file copy while the container runs.

Burn-in: run with the id registered but absent from the trust root, check
`GET /challenge/opentype/version` and `/v1/status`, then activate it with a signed
algorithm-3 profile. `full_share_mass` is `1.0`, so unpaid mass burns.

## 3. Duel workers (B300)

One worker host needs 1× B300 (both BF16 models, about 52 GB each, at 0.45 of GPU memory
per side, with a 131,072-token context and one image per prompt). It also needs about 250 GB of disk for the kept champion, one challenger and the
base support files, and outbound HTTPS to `huggingface.co` and the master.

`deploy/modal_worker.py` runs a worker on Modal (1x H200, scale to zero).

```bash
install -d -m 0700 /srv/opentype/work
install -m 0400 worker.token /srv/opentype/worker.token
docker run -d --name opentype-worker --restart unless-stopped \
  --gpus all --ipc=host \
  -v /srv/opentype/work:/work \
  -v /srv/opentype/worker.token:/run/secrets/worker.token:ro \
  ghcr.io/opentypeai/challenge-worker:stable \
  --api https://<cortex-master>/challenge/opentype \
  --token-file /run/secrets/worker.token --workdir /work
```

- Before it starts, the worker refuses to run unless `/opt/opentype/structured_server.py`
  matches the pinned sha256.
- Per job, the worker leases through the master proxy, verifies every file sha256, starts
  `vllm serve` and `structured_server.py` for each side on `127.0.0.1`, reads every case
  on both sides and deletes the challenger's weights afterwards. Logs of the four
  processes go to `/work/*.log`.
- Evidence recorded with each job: worker version, image
  (`challenge-worker:sha-<commit>`), vLLM image and version, `structured_server.py` sha256,
  canvas, resolved file digests of both sides, `cases_sha256`, the error count and timings.
- Several workers can share the queue. Each lease is exclusive, lasts 30 minutes and is
  renewed by heartbeats.
- Use `--once` for a single job (useful for phase-0 measurement), `--concurrency` for
  in-flight requests per side (default 64) and `--max-model-len` for the served context
  (default 131072, needed by longctx level 5).
- Harness episodes run up to 12 sequential turns each. Their throughput depends on
  `--concurrency`, not on page size.

Verify a worker image before you deploy it:

```bash
gh attestation verify oci://ghcr.io/opentypeai/challenge-worker@sha256:<digest> --repo OpentypeAI/challenge
```

### Phase 0: sizing before the first crown

1. Pause crowns (`PUT /v1/admin/crowns {"paused": true}`) and run worker jobs with
   `--once`. The evidence reports only aggregate `timings` (`serve_seconds`,
   `read_seconds`), so measure each track with a single-track `OPENTYPE_PLAN` (for example
   `{"longctx": {"weight": 1, "cases": 100}}`): decisions/s, seconds per longctx case, and
   ops, sql and paint episodes per hour.
2. From the verdict's `tracks` and `levels`, measure the base model's loss per track and
   per decisions level. Choose the ladder so the base scores 60–90 % on the entry levels.
   Drop from the plan any track the base cannot move at all.
3. Set `OPENTYPE_PLAN` so a whole duel finishes in a few hours and every harness track has
   well over 30 pairs (the per-track guard). Keep the weights where they reflect product
   priority. The composite SE is dominated by the smallest tracks
   ([mechanism.md](mechanism.md#9-power)).
4. With the teacher on, measure the keep rate and gateway spend of one full bank (the
   container log line `bank <kind>: kept N of target M, discarded {...}`), then set
   `OPENTYPE_BANK_TARGETS`.
5. Resume crowns.

## 4. Windows, banks and audits

A window's secret seeds every duel leased in it. While the window is open, only
`sha256(secret)` and its `bank_digest` are public.

Bank lifecycle:

1. The first window opens with the empty bank.
2. With a teacher, a background task builds the **next** bank. It is stored in the database
   but not yet sealed. `/v1/status` reports `teacher.state` as `configured`, `building` or
   `ready`. A failed build is logged and retried after 10 minutes.
3. Once the next bank is ready and the open window is at least `OPENTYPE_WINDOW_HOURS` old,
   the container rotates on its own: it closes the window, reveals its secret, jobs and
   bank, and opens a new window sealed with the ready bank. The builder then starts on the
   following bank.
4. A manual rotation works at any time. It seals the ready bank, or the empty bank when
   none is ready, which drops sealed and prose content from that window.

A bank is never swapped while its window is open. Rotating does not interrupt a leased job:
it keeps its seed and bank. Queued jobs are re-targeted to the new window at lease time.

```bash
curl -fsS -X POST -H "Authorization: Bearer $(cat admin.token)" \
  https://<cortex-master>/challenge/opentype/v1/admin/window/rotate
```

- Without a teacher, rotate on a fixed cadence (for example weekly), preferably when no job
  is leased.
- Rotation publishes the old secret with every job's digest, mix, plan, beacon, judge flag,
  case count and `cases_sha256`, plus the bank (`GET /v1/windows/<id>/bank`). Audit it, or
  let anyone audit it:

```bash
opentype-challenge audit --api https://<cortex-master>/challenge/opentype \
  --window <closed id> --window-secret <revealed hex>
```

  The audit checks the secret and the bank against their commitments, then prints one
  line per job with `ok`. The exit status is 1 on any mismatch.
- Spot-check exactness before a release: `uv run pytest tests/test_generator.py
  tests/test_longctx.py tests/test_ops.py tests/test_sqltask.py tests/test_paint.py`. These
  tests recompute read gold from the text and replay the reference oracles to loss 0.

## 5. Admin controls

| Action | Call |
| --- | --- |
| pause or resume crowns (anchors regress, or an incident) | `PUT /v1/admin/crowns {"paused": true}`. Scored winners wait. Resuming settles them in intake order |
| change the ladder | `PUT /v1/admin/ladder {"order": [3,4,5,6,7,8], "width": 2}` |
| re-run a job (worker incident, suspect evidence) | `POST /v1/admin/jobs/<job>/requeue`. Its results are discarded and a fresh job is queued. A crowned job cannot be re-queued |
| schedule the 75/25 split | `PUT /v1/admin/lanes {"epoch": n}`, once, past every persisted epoch. See §8 |
| publish or withdraw the runtime calibration | `PUT /v1/admin/runtime/calibration <object or null>`. See §8 |

All admin calls need `Authorization: Bearer <admin.token>`. See [api.md](api.md).

## 6. Images, channels and releases

| Image | Contents |
| --- | --- |
| `ghcr.io/opentypeai/challenge` | the challenge container (target `server`, `python:3.12-slim-bookworm` pinned by digest) |
| `ghcr.io/opentypeai/challenge-worker` | the duel worker (target `worker`, the vLLM nightly pinned by digest plus the pinned `structured_server.py`) |

| Tag | Moves | Built by |
| --- | --- | --- |
| `sha-<commit>` | never | `images.yml`, every push to `main` after `ci.yml` passes |
| `edge` | every push to `main` | the same build |
| `vX.Y.Z` | never | `release.yml` on an annotated tag, which aliases the tested `sha-` digest without a rebuild |
| `stable` | every release | the same alias |

Every pushed digest gets a GitHub build-provenance attestation (`actions/attest-build-provenance`,
pushed to the registry). Retagging keeps the digest, so the attestation still applies.

To release:

1. Bump `version` in `pyproject.toml` and `__version__` in `src/opentype_challenge/__init__.py`,
   and add a `## [X.Y.Z]` section to `CHANGELOG.md`.
2. Merge to `main` and wait for `images.yml` to publish `sha-<commit>`.
3. Run `git tag -a vX.Y.Z -m "vX.Y.Z" <commit> && git push origin vX.Y.Z`.

`release.yml` checks that the tag is annotated, on `main` and equal to the package version.
It moves `vX.Y.Z` and `stable` on both images and creates the GitHub release with both
digests.

## 7. Pins

`src/opentype_challenge/pins.py` pins the base model revision and the sha256 of every base
file. `Dockerfile` pins the Python, uv and vLLM images and the `structured_server.py`
source and sha256. Changing any pin is a release.

## 8. Runtime lane

Two lanes share the challenge's emission: **quality** 750 000 000 and **runtime**
250 000 000 units of every epoch (1e9 = one epoch-mass). `full_share_mass` stays 1.0; a lane
with nothing owed burns its share rather than giving it to the other.

### Activation

`PUT /v1/admin/lanes {"epoch": n}` with `n` past every epoch already served. Until `n` the
historical rule pays (one budget, quality only); every served epoch replays byte for byte.
From `n` on, quality debt, old debt included at its full amount, is repaid from 0.75 per
epoch. Check the Cortex emission mode before scheduling: nothing here changes the signed
emission configuration.

### Calibration (required before any runtime submission)

The runtime lane stays closed, and no timing is accepted, until you publish a calibration.
Every threshold in it comes from your own pilot; this repository ships none.

1. On the reference hardware (H200 first), run the worker with `--lane runtime` against a
   staging container, reference against reference (stock options on both sides), with an
   explicit GPU budget. Nothing here launches it for you.
2. Measure the B/B' drift and the block-to-block spread per cell; choose `max_drift`,
   `min_gain`, `blocks` and `bootstrap_resamples` so that a stock-vs-stock job is rejected.
3. Publish `{"version", "profile", "cells", "blocks", "max_drift", "min_gain",
   "latency_tolerance", "fidelity_loss_tolerance", "fidelity_accuracy_tolerance",
   "bootstrap_resamples", "credit_per_log_gain", "credit_cap"}`. `profile` must equal the
   pinned serving profile (`runtime.PROFILE_FIXED`) plus the `gpu` and `driver` strings the
   worker reads from `nvidia-smi` and the `vllm_version` it reads from the installed
   package. The worker builds its profile from what it runs: `vllm_image` from
   `/opt/opentype/build.json` (written by the Dockerfile's worker stage from its
   `VLLM_IMAGE` build argument), the installed vllm version, the reader's sha256 and its
   own dtype, canvas and length flags. It refuses a job (infrastructure retry) when any of
   them cannot be read or differs from the calibration.

   Trust limit: this is self-reported by a worker you operate, not an attestation. The
   manifest is only as good as the build that wrote it and the host that runs it; a
   modified worker can report anything. Run runtime workers only on hardware and images you
   control. Each cell is `{"track" (decisions, longctx, ops or sql),
   "cases", "concurrency", "slo_ms", "weight", "warm"}`; weights sum to 1.

Changing the calibration makes a running runtime job that is still on target (same signed
profile, kernel slot still open) duel again under the new one. A job the change takes off
target (a new profile, or its kernel slot closed) expires. Queued and judging jobs expire
at once, and their pending judgments are dropped. A leased job turns stale and expires when
its worker completes or releases it. The miner then submits again. Closing a kernel slot (removing it from
`kernel_slots`) therefore expires kernel submissions for that slot, and an incumbent with that slot's kernel stops being the reference: stock
serves as B again. A stored calibration that this build cannot parse counts as withdrawn. A
submission signs the profile digest: if the new calibration changes the profile, its
queued and running runtime work expires and miners sign again. Withdrawing the calibration
(`null`) parks runtime work in the queue until one is published again. Caps:
`blocks` <= 999, `bootstrap_resamples` <= 100 000, per cell `cases` <= 10 000 and
`concurrency` <= 1024.

### Runtime workers

`opentype-challenge worker --lane runtime` on dedicated, operator-owned hardware whose
profile matches the calibration. A runtime job leases only when no other job of this
challenge is leased, and nothing leases while it runs (ponytail: exclusion is per
challenge, not per GPU). While a runtime job waits and a runtime worker polled within two
minutes, quality leases pause after `Settings.runtime_every` (4) of them so the GPU drains;
a runtime lease resets the count, and without a polling runtime worker quality never
pauses. The worker:

- serves the champion's weights with stock flags (fidelity side `champion`), then with the
  candidate's flags (side `challenger`), one server at a time at the calibrated
  `gpu_memory_utilization`, exactly as in the timed blocks, for the fidelity cases, which
  the container scores;
- then, for each block, starts one `vllm serve` (plus the pinned reader) at a time for B,
  C and B', checks that every process exited and `nvidia-smi` lists no compute process
  before and after each, posts every timed task's raw output and latency to
  `/v1/worker/jobs/<id>/timings` and reports each run's `time.monotonic()` seconds; the
  container scores the outputs against gold. A failed check stops the job and the verdict
  is `NO_DECISION`;
- starts the stock (fidelity) or incumbent (B) server alone first: if it fails to start, the host
  is at fault (retry); if only the candidate then fails to start, its options are
  (rejected, no retry). Option combinations the pinned `SchedulerConfig` refuses
  (`max_num_batched_tokens < max_num_seqs`; chunked prefill off with
  `max_num_batched_tokens < max_model_len`) are refused at intake;
- passes only flags produced by `runtime.options_argv` from the allowlist, and an
  environment without any variable whose name looks like a credential.

Allowlisted options (each a parsed `vllm serve` flag at the pinned nightly `7f1a5398`;
their effect on DiffusionGemma is what the benchmark measures): `max_num_seqs`,
`max_num_batched_tokens`, `enable_chunked_prefill`, `enable_prefix_caching`. Anything else
is refused at intake.

### NVFP4 migration of the quality champion

The runtime lane measures NVFP4 weights on B300 only, so it stays closed while the champion
is BF16. `POST /v1/admin/champion/nvfp4` (no body) resets the quality lane to NVFP4, once,
and only while the base is champion: the pinned official export (`pins.NVFP4_REPO` at
`NVFP4_REVISION`, exactly `pins.NVFP4_FILES`) becomes the champion. A mined BF16 champion is
refused (409): nothing here could prove that an NVFP4 checkpoint derives from it.

- It is a prospective reset, not a claim that the export equals the BF16 champion. A new
  champion row is added with no hotkey and no entitlement. Earlier champions, entitlements,
  payments and served epochs are unchanged, and old debt keeps paying FIFO. There is no
  way back.
- Queued BF16 work expires at once. A judging duel also expires at once, and its pending
  judgments are dropped (the teacher judges nothing for it). A leased duel turns stale;
  its worker completes or fails it normally, then it expires before anything is judged. Nothing BF16 is re-duelled: a submission in another
  format than the champion's expires instead of being requeued.
- From then on intake takes only the NVFP4 config, with the weight index (a single
  `model.safetensors` is refused), and the worker checks the pinned tensor layout of both
  sides before serving. A later NVFP4 champion is always a crowned challenger that passed
  that check.
- Quality duels then serve NVFP4 on both sides, and quality jobs lease only to workers
  that declare `nvfp4`: the sandbox controller (`deploy/modal_controller.py`: the deployed `quality` Function, spawned, never `modal run`),
  one fresh B300 sandbox per side. Both sides must measure the runtime lane's pinned profile
  (image, reader, NVFP4 weights, flags, share, context, B300), which differs only by the
  versioned `runtime.QUALITY_SERVING` entry. Both sides must also measure the same
  vllm/driver/compute capability, or the job retries. The duel records both profiles and
  the GPU UUIDs. A sandbox that never starts retries and never rejects the challenger:
  each side runs on a fresh placement. An H200 worker leases nothing, so stop
  `opentype-worker` (`modal app stop opentype-worker`) at the migration, and validate one
  duel on the controller first. Quality vllm pins
  `--kv-cache-dtype bfloat16`, because `auto` would turn FP8 with unit scales under this
  config. Re-run Phase 0 sizing on the NVFP4 champion before trusting the quality margin:
  its error rates differ.

### Kernels: implemented, not enabled

`opentype_challenge.sandbox` runs miner kernels only in fresh Modal Sandboxes:
`gpu="B300"`, `block_network=True`, `secrets=[]`, no OIDC token, and the verified NVFP4
snapshot mounted read-only from the dedicated volume `opentype-nvfp4-snapshot` (never the
quality worker's). The bootstrap measures the GPU identity before any miner code exists
there. vllm, with the kernel, runs as uid 10001. The pinned reader runs as uid 10002 and
binds its port first. A reply counts only while every served process is alive. Build and
serve children are capped by rlimits and logs are tail-read. The kernel is compiled offline
in its own CPU sandbox first. The controller holds every token and gold answer. Requests
travel over the sandbox's exec stdio as numbered lines of 16 KiB or less, each frame
capped at 8 MiB, and a lost line fails the channel.

This is tested against hostile local processes. That is not a proof that no escape
exists. Kernels stay off in production until all of these hold:

1. The B300 controller (`deploy/modal_controller.py`) has run a real quality duel and a
   runtime calibration pilot. The controller is a CPU Function that holds the worker token.
   It drives `SandboxLauncher` over `ModalBackend(gpu="B300", commit=True)`, with one work
   volume per lane and one writer. It is not deployed or scheduled by this repository. The
   local-process launcher refuses every kernel.
2. The champion is migrated to NVFP4.
3. A B300 calibration listing `kernel_slots` is published from a representative pilot, not
   from the smokes.

Operational notes:

- A fresh Sandbox is not the same physical GPU as the previous one. The calibration
  measures that spread, and each run's `placements` record the GPU UUIDs.
- Stage the snapshot (`modal run deploy/modal_runtime.py::stage`) only while nothing
  serves from it. It commits once and must never be written concurrently.
- Relay privacy: the exec path does not mirror frames into Modal app logs. The
  `entrypoint` path of `relay_probe` does. Prompts are private benchmark inputs, so never
  publish raw Modal app logs. No gold and no token ever enters a sandbox.
