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

Tokens are read on every request, so you can rotate one by rewriting the file without a
restart.

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
OPENTYPE_DUEL_CASES = "40000"       # cases per duel (≈ 230 k decisions)
OPENTYPE_MAX_PENDING = "4"          # queued submissions across all hotkeys
# OPENTYPE_WINDOW_ENTITLEMENT_CAP = "20"   # optional cap, in epoch-masses per window
```

The image already sets `CHALLENGE_SLUG=opentype`, `CHALLENGE_STATE_DIR=/data` and the three
`CHALLENGE_*_TOKEN_FILE` paths under `/run/secrets/`. The supervisor verifies the labels
(`io.cortex.challenge.slug=opentype`, `io.cortex.challenge.contract=1`, source) and the
GitHub build-provenance attestation of the pulled digest. It then runs a canary without
secrets, which must answer `/version`, and replaces the container on the same `/data`
volume.

State is a single SQLite database, `/data/opentype.sqlite3` (WAL). It holds windows and
their secrets, submissions, jobs, results, champions, the ledger and every persisted epoch
body. Back it up with SQLite's online backup API (for example
`sqlite3.connect(src).backup(dst)`), never with a plain file copy while the container runs.

Burn-in: run with the id registered but absent from the trust root, check
`GET /challenge/opentype/version` and `/v1/status`, then activate it with a signed
algorithm-3 profile. `full_share_mass` is `1.0`, so unpaid mass burns.

## 3. Duel workers (B300)

One worker host needs 1× B300 (both BF16 models, about 52 GB each, at 0.45 of GPU memory
per side). It also needs about 250 GB of disk for the kept champion, one challenger and the
base support files, and outbound HTTPS to `huggingface.co` and the master.

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
- Use `--once` for a single job (useful for phase-0 measurement) and `--concurrency` for
  in-flight reads per side (default 64).

Verify a worker image before you deploy it:

```bash
gh attestation verify oci://ghcr.io/opentypeai/challenge-worker@sha256:<digest> --repo OpentypeAI/challenge
```

### Phase 0: before the first crown

1. Run one worker job with `--once` and read `timings` in the job evidence. Measure
   decisions per second at canvas 256.
2. Measure the base model's accuracy per level from the first duels (`/v1/status` levels).
   Choose the ladder so the base scores 60–90 % on the active levels
   (`PUT /v1/admin/ladder`).
3. Set `OPENTYPE_DUEL_CASES` from the measured throughput and the power table in
   [mechanism.md](mechanism.md#4-power). Below about 1,000 decisions/s, use 20,000 cases
   and rely more on the ladder.

## 4. Windows and audits

A window's secret seeds every duel leased in it. Only `sha256(secret)` is public while the
window is open.

```bash
curl -fsS -X POST -H "Authorization: Bearer $(cat admin.token)" \
  https://<cortex-master>/challenge/opentype/v1/admin/window/rotate
```

- Rotate on a fixed cadence (for example weekly), preferably when no job is leased. Queued
  jobs are re-targeted to the new window when they are leased.
- Rotation publishes the old secret with every job's digest, mix, case count and
  `cases_sha256`. Audit it, or let anyone audit it:

```bash
opentype-challenge audit --api https://<cortex-master>/challenge/opentype \
  --window <closed id> --window-secret <revealed hex>
```

  Each line reports a job and `ok`. The exit status is 1 on any mismatch.
- Spot-check generator exactness before a release: `uv run pytest tests/test_generator.py`
  recomputes the gold of thousands of cases from their text alone.

## 5. Admin controls

| Action | Call |
| --- | --- |
| pause or resume crowns (anchors regress, or an incident) | `PUT /v1/admin/crowns {"paused": true}`. Scored winners wait. Resuming settles them in intake order |
| change the ladder | `PUT /v1/admin/ladder {"order": [3,4,5,6,7,8], "width": 2}` |
| re-run a job (worker incident, suspect evidence) | `POST /v1/admin/jobs/<job>/requeue`. Its results are discarded and a fresh job is queued. A crowned job cannot be re-queued |

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
