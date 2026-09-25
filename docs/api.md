# API reference

Base URL from outside: `https://<cortex-master>/challenge/opentype`. The master proxies
every path except `/internal/*`, forwards only the `content-type`, `accept` and
`authorization` headers, limits request bodies to the registry `proxy_body_limit` (1 MiB by
default) and responses to 8 MiB, and times out after 30 s. Inside the `cortex-challenges`
network, the container is `http://cortex-challenge-opentype:8000`.

## Conventions

- Every body is JSON. Unknown fields are refused (`422`).
- An error is `{"detail": "<reason>"}`.
- Times are UTC `YYYY-MM-DDTHH:MM:SSZ`.
- Hotkeys are SS58 (network 42). A submission may give the hotkey as SS58 or 64-hex.

| Status | Meaning |
| --- | --- |
| `400` | a semantically invalid value (bad `exp`, bad hotkey, `case_index` out of range, bad ladder) |
| `401` | missing or wrong bearer, or an invalid signature |
| `403` | slug mismatch on `get_weights`, or a hotkey not registered on the subnet |
| `404` | unknown submission, job or window |
| `409` | nonce reuse, an open submission already exists, clone of the champion, lease mismatch, incomplete job, crowned job |
| `413` | body over the route limit |
| `422` | schema validation failed, or the manifest is refused |
| `429` | duel queue full |
| `503` | this route's token file is missing, or the metagraph is unavailable (retry) |

## Contract (Cortex v1)

### `GET /version`

Liveness. Needs no state and no secrets.

```json
{"slug": "opentype", "version": "2.0.0", "contract": 1, "capabilities": ["get_weights", "proxy_routes"]}
```

### `GET /health`

Readiness only: `200 {"ok": true}` when the state volume is writable and the internal
token is readable, otherwise `503 {"ok": false}`.

### `GET /internal/v1/get_weights?epoch=<u64>`

Headers: `Authorization: Bearer <internal.token>` (`401`) and
`X-Platform-Challenge-Slug: opentype` (`403`). The master can reach this route only on the
private network.

```json
{
  "challenge_slug": "opentype",
  "computed_at": "2026-09-24T17:40:30Z",
  "epoch": 25316,
  "full_share_mass": 1.0,
  "metadata": {
    "burned": 0.25,
    "payments": [{"champion": 2, "entitlement": 1, "hotkey": "5F...", "mass": 0.75}]
  },
  "weights": {"5F...": 0.75}
}
```

The first call for an epoch pays outstanding entitlements first in, first out, up to 1.0
and persists the body. Every later call returns the same bytes.

## Public

### `GET /v1/status`

Champion, queue, ladder, next duel mix, open window, plan, tracks, teacher, pause flag and
constants. New in 2.0: `window.bank_digest`, `plan`, `tracks`, `teacher`, and
`constants.duel_cases` now equal to the plan's total.

```json
{
  "champion": {"id": 1, "hotkey": null, "repo": "google/diffusiongemma-26B-A4B-it",
               "revision": "f7f5b7f5...", "digest": "3db7247e...", "g_lcb": null,
               "crowned_at": "..."},
  "queue": [{"id": "s_...", "hotkey": "5D...", "intake": 1, "state": "leased", "job": "j_..."}],
  "levels": [{"level": 1, "state": "active", "champion_accuracy": 0.83,
              "champion_accuracy_lcb99": 0.82, "determined": 41022}],
  "next_duel_mix": {"1": 0.5, "2": 0.5},
  "window": {"id": 1, "commitment": "<sha256(secret) hex>", "bank_digest": "<sha256 hex>",
             "opened_at": "..."},
  "plan": {"decisions": {"weight": 0.35, "cases": 4000}, "longctx": {"weight": 0.25, "cases": 800},
           "ops": {"weight": 0.15, "cases": 1000}, "paint": {"weight": 0.15, "cases": 600},
           "sql": {"weight": 0.1, "cases": 1000}},
  "tracks": {"decisions": {"weight": 0.35, "cases": 4000, "results": 5200}, "...": {}},
  "teacher": {"state": "ready", "judge": true, "judgments_pending": 0},
  "crowns_paused": false,
  "constants": {"g_min": 0.0513, "z": 2.326, "guard_max": 0.002, "duel_cases": 7400,
                "early_stop_decisions": 5000, "early_stop_se": 3.0, "retire_accuracy": 0.999,
                "max_pending": 4, "window_entitlement_cap": null,
                "base": {"repo": "...", "revision": "..."}}
}
```

- `plan` is the effective plan of the open window: it leaves out tracks that its bank and
  judge cannot build, and renormalizes the weights. `tracks[t].results` counts the stored
  side results of that track.
- `teacher.state` is `off`, `configured`, `building` (the next bank is being built) or
  `ready` (the next bank is waiting for rotation). `judge` tells whether `depict` is scored.

### `GET /v1/leaderboard`

`{"crowns": [...], "hotkeys": {...}}`. Each crown is the champion object plus
`entitlement`, `paid` and `outstanding` (in epoch-masses) once it has an entitlement.
`hotkeys` maps a hotkey to `{"entitlement", "paid"}` totals.

### `GET /v1/windows` and `GET /v1/windows/{id}`

`/v1/windows` returns
`{"windows": [{"id", "commitment", "bank_digest", "opened_at", "closed_at", "revealed"}]}`
and never includes secrets. `/v1/windows/{id}` of a closed window adds `"secret"` (hex) and
`"jobs"`:
`[{"id", "digest", "mix", "plan", "beacon", "judge", "cases", "state", "cases_sha256", "cases_fetched"}]`.
`beacon` is `{"round", "randomness"}` or `null`. `plan` is `{track: {"weight", "cases"}}`.
These are exactly the inputs that `opentype-challenge audit` needs.

### `GET /v1/windows/{id}/bank?offset=0&limit=1000`

New in 2.0. Published only for a closed window (`404` otherwise). `limit` is at most
10,000, and a page stays under 6 MiB.

```json
{"window": 3, "bank_digest": "<sha256 hex>", "total": 884, "offset": 0,
 "items": [["family", "<16 hex key>", {...payload...}], ["prose", "...", {...}]]}
```

Items are `[kind, key, payload]`, sorted by `(kind, key)`. `bank_digest` is sha256 over
the canonical JSON line of each item. Kinds: `family`, `prose`, `ops_story`, `depict` (see
[tracks.md](tracks.md#2-bank-teacher-content-per-window)).

### `POST /v1/submissions`

The body is at most 64 KiB. It returns `201` with the submission.

```json
{
  "manifest": {
    "repo": "you/model",
    "revision": "<40 lowercase hex>",
    "files": {"config.json": "<sha256>", "model-00001-of-00011.safetensors": "<sha256>", "...": "..."}
  },
  "hotkey": "5F...",
  "nonce": "<32 lowercase hex>",
  "exp": 1790000240,
  "signature": "<64-byte sr25519 signature hex>"
}
```

- `files`: 2 to 128 entries, names from `config.json`, `model.safetensors.index.json`,
  `model.safetensors` or `model-NNNNN-of-NNNNN.safetensors`. Digests are lowercase sha256.
  Shards need the index, and `config.json` must equal the base's.
- The signature covers `opentype-submit-v1|<public key hex>|<manifest digest>|<nonce>|<exp>`,
  where the manifest digest is `sha256` of
  `{"files": {sorted}, "repo": ..., "revision": ...}` as compact sorted-key JSON.
- `exp` must be in the future and at most 300 s ahead. A nonce is single-use.

### `GET /v1/submissions/{id}`

```json
{
  "id": "s_a1c580f2a43edaeb", "hotkey": "5D...", "repo": "you/model", "revision": "...",
  "digest": "...", "state": "queued", "reason": null, "intake": 1, "created_at": "...",
  "job": {"id": "j_...", "state": "leased", "judgments_pending": 0, "champion": 1, "window": 1,
          "cases": 7400, "paired": 2600, "attempts": 0, "verdict": null, "evidence": null,
          "reason": null}
}
```

- Submission states: `queued`, `crowned`, `rejected` or `failed`.
- Job states: `queued`, `leased`, `judging` (new: every case is answered and some depict
  sides wait for the judge), `scored`, `crowned`, `rejected`, `failed` or `superseded`
  (re-targeted after the champion changed, or re-queued by the operator; the submission
  then points at its new job).
- `verdict` holds `crown`, `early_stop`, `g`, `se` (the weighted composite), `g_lcb`,
  `g_lcb_halves`, `g_min`, `guard_ucb`, `guard_max`, `cases`, `decisions` and the
  decisions-track `levels` object with
  `{champion, challenger}: {determined, correct, accuracy, under, brier, loss}`. New in 2.0:
  - `tracks: {t: {g, se, pairs, champion_loss, challenger_loss, accuracy: {champion,
    challenger}, regressed}}`;
  - `track_guard: {pairs: 30, min: −ln 1.02}`;
  - `unjudged`, the number of cases dropped on both sides because a judge failed.

## Worker (`Authorization: Bearer <worker.token>`, bodies at most 1 MiB)

### `POST /v1/worker/lease`

Returns `204` when the queue is empty. Otherwise:

```json
{
  "job": "j_...", "lease": "<32 hex>", "lease_expires": "...", "cases": 7400,
  "plan": {"decisions": {"weight": 0.35, "cases": 4000}, "...": {}},
  "champion": {"repo": "...", "revision": "...", "files": {...}, "digest": "..."},
  "challenger": {"repo": "...", "revision": "...", "files": {...}, "digest": "..."},
  "base": {"repo": "google/diffusiongemma-26B-A4B-it", "revision": "f7f5b7f5..."}
}
```

The lease lasts 30 minutes. An expired lease is re-queued and counts as one of 3 attempts.

### `POST /v1/worker/jobs/{job}/heartbeat`

Body `{"lease"}`. Returns `{"lease_expires", "stale"}`. `stale` means the champion has
changed.

### `GET /v1/worker/jobs/{job}/cases?lease=&offset=0&limit=100`

`limit` is at most 200. Returns `{"cases": [{"index", "track", "body"}]}` (`track` is new).
A page is at most 6 MiB of JSON and always holds at least one case, so the worker pages by
the returned count until it has `cases`.

- Read tracks (`decisions`, `longctx`): `body` is the `/v1/systemone` request (`model`,
  `instructions`, `state`, `questions`, `samples`, `seed`). A longctx body is up to 600 kB.
- Harness tracks (`ops`, `sql`, `paint`): `body` is
  `{"harness", "version": 2, "task", "limits": {"turns", "max_tokens"}, "seed"}`. The
  worker plays the episode with `harness.run_episode`. The model sees only the system
  prompt and the observations, never `task`.

Gold, levels, expected actions and depict rubrics are never included.

### `POST /v1/worker/jobs/{job}/answers`

```json
{"lease": "...", "items": [
  {"case_index": 0, "side": "champion",
   "answers": {"<qid>": {"noul": 0.93} },
   "reads": {"<qid>": {"label_mass": 0.99, "argmax_is_label": true}}},
  {"case_index": 0, "side": "challenger", "error": "422: invalid output"},
  {"case_index": 7, "side": "champion", "transcript": ["{\"tool\": \"sql\", ...}", "..."]}
]}
```

1 to 2,000 items per request. There are three kinds of item:

- a read item carries `answers` and `reads`;
- a harness item carries `transcript`, the raw model outputs in order: at most 64 strings
  of at most 8,192 characters, and no more than the case's `limits.turns`. The container
  replays it and scores the final state;
- an `error` item (at most 500 characters) forfeits that case for that side.

A duplicate `(case_index, side)` is ignored. Returns
`{"accepted", "paired", "judging", "continue", "early_stop", "stale"}`. `judging` (new)
counts the items of this batch stored for the judge. Stop reading when `continue` is
false.

### `POST /v1/worker/jobs/{job}/complete`

Body `{"lease", "evidence": {...}}`. The call returns `409` unless every case is answered on
both sides, the job has early-stopped, or the job is stale. It returns the submission.
When depict sides are still waiting for the judge, the job moves to `judging` and the call
waits up to 20 s for them. The response may therefore show `judging`, which the container
settles in the background.

### `POST /v1/worker/jobs/{job}/fail`

Body `{"lease", "reason", "retry": true, "evidence": {...}}`. `retry: true` re-queues the job
(`failed` after 3 attempts). `retry: false` rejects the submission. Returns the submission.

## Admin (`Authorization: Bearer <admin.token>`)

| Route | Body | Result |
| --- | --- | --- |
| `POST /v1/admin/window/rotate` | none | `{"closed", "opened", "commitment", "bank_digest"}`; the closed window's secret, jobs and bank become public, and the new window is sealed with the next bank, or with the empty bank when none is ready |
| `PUT /v1/admin/ladder` | `{"order": [levels], "width": n}` | `{"order", "width", "retired"}`; `400` on unknown or duplicate levels, or a bad width |
| `PUT /v1/admin/crowns` | `{"paused": bool}` | `{"crowns_paused"}` |
| `POST /v1/admin/jobs/{job}/requeue` | none | the submission with a fresh queued job; `409` for a crowned job |
| `PUT /v1/admin/runtime/calibration` | the calibration object, or `null` | `{"calibration"}` (its public form); `422` on any missing, extra or out-of-range key, or a profile that differs from the pinned serving profile. `null` closes the runtime lane |
| `PUT /v1/admin/lanes` | `{"epoch": n}` | `{"lanes_from_epoch"}`; the 75/25 split pays from epoch `n` on. Once only; `409` when `n` is not past every persisted epoch |

## Runtime lane

### `GET /v1/runtime`

`{"open", "lanes_from_epoch", "budgets", "options", "kernels", "calibration", "target",
"incumbent", "queue", "crowns"}`. `target` is `{"champion", "digest"}` of the current quality
champion; `calibration.profile_digest` is what a submission signs. `kernels` says why miner
kernels are disabled.

### `POST /v1/runtime/submissions`

```json
{"target": {"champion": 3, "digest": "<64 hex>"}, "profile": "<64 hex>",
 "options": {"max_num_seqs": 128}, "hotkey": "<ss58>", "nonce": "<32 hex>", "exp": 1790000200,
 "signature": "<128 hex>"}
```

The signature is sr25519 over `opentype-runtime-v1|<pubkey hex>|<digest>|<nonce>|<exp>`,
where `digest` is the sha256 of the canonical JSON
`{"challenge", "lane": "runtime", "options", "profile", "target"}`. A weights signature never
verifies here. `options` holds only allowlisted keys (`GET /v1/runtime`); unknown keys and
any other field are `422`. `503` while the lane is closed; `409` for a stale target or
profile, a reused nonce, the incumbent's own options, or a second open runtime submission of
the hotkey (its quality slot is separate). `201` returns the submission.

### Worker

`POST /v1/worker/lease?lane=runtime` leases runtime jobs; without `lane` a worker gets quality
jobs only. A runtime lease is never handed out while any job is leased, and nothing is leased
while a runtime job is. Its body has `"lane": "runtime"`, `"challenger": null` and
`"runtime": {"calibration", "incumbent", "candidate", "seed"}`. Its cases are fidelity reads
(`champion` = stock, `challenger` = candidate, posted in two one-side passes) on every track a calibrated cell measures
(decisions always). Timed tasks go to `POST /v1/worker/jobs/<id>/timings`
`{"lease", "items": [{"block", "side": "B"|"C"|"B2", "cell", "case_index", "ms",
"answers"?, "reads"?, "transcript"?, "error"?}]}`: raw outputs only (an `ok` field is
`422`); the container rebuilds the case from the seed, scores it against gold and returns
`{"accepted", "ok"}`. `complete` carries
`evidence.runtime = {"profile", "blocks": [{"order": ["B","C","B2"], "seconds": {side:
{cell: s}}, "quiescent"}]}`. A runtime job whose signed target is no longer the champion is
expired, never requeued. While a runtime job waits, a quality lease can be `204` so the GPU
drains (bounded, see operator.md §8).
