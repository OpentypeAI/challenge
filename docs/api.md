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
{"slug": "opentype", "version": "1.0.0", "contract": 1, "capabilities": ["get_weights", "proxy_routes"]}
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

Champion, queue, ladder, next duel mix, open window, pause flag and constants.

```json
{
  "champion": {"id": 1, "hotkey": null, "repo": "google/diffusiongemma-26B-A4B-it",
               "revision": "f7f5b7f5...", "digest": "3db7247e...", "g_lcb": null,
               "crowned_at": "..."},
  "queue": [{"id": "s_...", "hotkey": "5D...", "intake": 1, "state": "leased", "job": "j_..."}],
  "levels": [{"level": 1, "state": "active", "champion_accuracy": 0.83,
              "champion_accuracy_lcb99": 0.82, "determined": 41022}],
  "next_duel_mix": {"1": 0.5, "2": 0.5},
  "window": {"id": 1, "commitment": "<sha256(secret) hex>", "opened_at": "..."},
  "crowns_paused": false,
  "constants": {"g_min": 0.0513, "z": 2.326, "guard_max": 0.002, "duel_cases": 40000,
                "early_stop_decisions": 5000, "early_stop_se": 3.0, "retire_accuracy": 0.999,
                "max_pending": 4, "window_entitlement_cap": null,
                "base": {"repo": "...", "revision": "..."}}
}
```

### `GET /v1/leaderboard`

`{"crowns": [...], "hotkeys": {...}}`. Each crown is the champion object plus
`entitlement`, `paid` and `outstanding` (in epoch-masses) once it has an entitlement.
`hotkeys` maps a hotkey to `{"entitlement", "paid"}` totals.

### `GET /v1/windows` and `GET /v1/windows/{id}`

`/v1/windows` returns `{"windows": [{"id", "commitment", "opened_at", "closed_at", "revealed"}]}`
and never includes secrets. `/v1/windows/{id}` of a closed window adds `"secret"` (hex) and
`"jobs"`:
`[{"id", "digest", "mix", "cases", "state", "cases_sha256", "cases_fetched"}]`.

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
  "job": {"id": "j_...", "state": "leased", "champion": 1, "window": 1, "cases": 40000,
          "paired": 12000, "attempts": 0, "verdict": null, "evidence": null, "reason": null}
}
```

- Submission states: `queued`, `crowned`, `rejected` or `failed`.
- Job states: `queued`, `leased`, `scored`, `crowned`, `rejected`, `failed` or `superseded`
  (re-targeted after the champion changed, or re-queued by the operator; the submission
  then points at its new job).
- `verdict` holds `crown`, `early_stop`, `g`, `se`, `g_lcb`, `g_lcb_halves`, `g_min`,
  `guard_ucb`, `guard_max`, `cases`, `decisions` and a per-level `levels` object with
  `{champion, challenger}: {determined, correct, accuracy, under, brier, loss}`.

## Worker (`Authorization: Bearer <worker.token>`, bodies at most 1 MiB)

### `POST /v1/worker/lease`

Returns `204` when the queue is empty. Otherwise:

```json
{
  "job": "j_...", "lease": "<32 hex>", "lease_expires": "...", "cases": 40000,
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

`limit` is at most 200. Returns `{"cases": [{"index", "body"}]}`, where `body` is the
`/v1/systemone` request (`model`, `instructions`, `state`, `questions`, `samples`, `seed`).
Gold and levels are never included.

### `POST /v1/worker/jobs/{job}/answers`

```json
{"lease": "...", "items": [
  {"case_index": 0, "side": "champion",
   "answers": {"<qid>": {"noul": 0.93} },
   "reads": {"<qid>": {"label_mass": 0.99, "argmax_is_label": true}}},
  {"case_index": 0, "side": "challenger", "error": "422: invalid output"}
]}
```

1 to 2,000 items per request. An `error` item forfeits that case for that side. A
duplicate `(case_index, side)` is ignored. Returns
`{"accepted", "paired", "continue", "early_stop", "stale"}`. Stop reading when `continue`
is false.

### `POST /v1/worker/jobs/{job}/complete`

Body `{"lease", "evidence": {...}}`. The call returns `409` unless every case is paired, the
job has early-stopped, or the job is stale. It returns the submission.

### `POST /v1/worker/jobs/{job}/fail`

Body `{"lease", "reason", "retry": true, "evidence": {...}}`. `retry: true` re-queues the job
(`failed` after 3 attempts). `retry: false` rejects the submission. Returns the submission.

## Admin (`Authorization: Bearer <admin.token>`)

| Route | Body | Result |
| --- | --- | --- |
| `POST /v1/admin/window/rotate` | none | `{"closed", "opened", "commitment"}`; the closed window's secret becomes public |
| `PUT /v1/admin/ladder` | `{"order": [levels], "width": n}` | `{"order", "width", "retired"}`; `400` on unknown or duplicate levels, or a bad width |
| `PUT /v1/admin/crowns` | `{"paused": bool}` | `{"crowns_paused"}` |
| `POST /v1/admin/jobs/{job}/requeue` | none | the submission with a fresh queued job; `409` for a crowned job |
