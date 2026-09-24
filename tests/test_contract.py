"""Cortex challenge container contract v1 and the auth of every non-public route."""

import json

import pytest

from opentype_challenge import __version__
from opentype_challenge.bank import EMPTY_BANK

from .conftest import ADMIN, INTERNAL, SLUG, WORKER, bearer


def weights(client, epoch, token=INTERNAL, slug=SLUG):
    headers = {"x-platform-challenge-slug": slug} if slug else {}
    if token:
        headers.update(bearer(token))
    return client.get(f"/internal/v1/get_weights?epoch={epoch}", headers=headers)


def test_version_and_health(client):
    assert client.get("/version").json() == {
        "slug": SLUG,
        "version": __version__,
        "contract": 1,
        "capabilities": ["get_weights", "proxy_routes"],
    }
    assert client.get("/health").json() == {"ok": True}


def test_get_weights_auth(client):
    assert weights(client, 1, token=None).status_code == 401
    assert weights(client, 1, token="wrong").status_code == 401
    assert weights(client, 1, slug=None).status_code == 403
    assert weights(client, 1, slug="bounty").status_code == 403
    response = client.get("/internal/v1/get_weights?epoch=1", headers={"authorization": INTERNAL})
    assert response.status_code == 401
    assert weights(client, -1).status_code == 422


def test_get_weights_body_and_byte_identical_replay(client, clock):
    first = weights(client, 25316)
    assert first.status_code == 200
    body = first.json()
    assert body["challenge_slug"] == SLUG and body["epoch"] == 25316
    assert body["weights"] == {} and body["full_share_mass"] == 1.0
    assert body["computed_at"].endswith("Z")
    clock.now += 3600
    assert weights(client, 25316).content == first.content
    assert weights(client, 25317).json()["computed_at"] != body["computed_at"]


def test_canary_without_secrets_starts_and_answers_version(tmp_path, clock, master):
    """The supervisor canary mounts no secret files: /version works, guarded routes 503."""
    import httpx
    from fastapi.testclient import TestClient

    from opentype_challenge.app import Config, create_app

    missing = tmp_path / "no-secrets"
    config = Config(
        SLUG,
        tmp_path / "canary",
        "http://master.test",
        missing / "internal.token",
        missing / "admin.token",
        missing / "worker.token",
    )
    with TestClient(create_app(config, clock, httpx.MockTransport(master.handler))) as canary:
        assert canary.get("/version").json()["contract"] == 1
        assert canary.get("/health").status_code == 503
        assert weights(canary, 1).status_code == 503
        assert canary.post("/v1/worker/lease", headers=bearer(WORKER)).status_code == 503
        assert canary.post("/v1/admin/window/rotate", headers=bearer(ADMIN)).status_code == 503
        assert canary.get("/v1/status").status_code == 200


def test_teacher_without_a_readable_token_is_off(tmp_path, clock, master, monkeypatch):
    """A configured gateway URL whose token file is missing leaves the teacher off."""
    import httpx
    from fastapi.testclient import TestClient

    from opentype_challenge.app import Config, create_app

    monkeypatch.setenv("OPENTYPE_TEACHER_URL", "https://gateway.test")
    monkeypatch.setenv("OPENTYPE_TEACHER_TOKEN_FILE", str(tmp_path / "missing.token"))
    config = Config(SLUG, tmp_path / "state", "http://master.test", None, None, None)
    app = create_app(config, clock, httpx.MockTransport(master.handler), beacon=lambda: None)
    with TestClient(app) as client:
        teacher = client.get("/v1/status").json()["teacher"]
    assert teacher == {"state": "off", "judge": False, "judgments_pending": 0}


def test_token_files_are_read_per_request(client, secrets_dir):
    (secrets_dir / "internal.token").write_text("rotated\n")
    assert weights(client, 1).status_code == 401
    assert weights(client, 1, token="rotated").status_code == 200
    (secrets_dir / "internal.token").unlink()
    assert weights(client, 2, token="rotated").status_code == 503
    assert client.get("/health").status_code == 503


def test_health_is_503_when_the_state_volume_is_read_only(client):
    store = client.app.state.store
    store._db.execute("PRAGMA query_only=ON")
    try:
        assert client.get("/health").status_code == 503
        assert client.get("/version").status_code == 200
    finally:
        store._db.execute("PRAGMA query_only=OFF")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/worker/lease"),
        ("get", "/v1/worker/jobs/j_x/cases?lease=x"),
        ("post", "/v1/worker/jobs/j_x/answers"),
        ("post", "/v1/worker/jobs/j_x/heartbeat"),
        ("post", "/v1/worker/jobs/j_x/complete"),
        ("post", "/v1/worker/jobs/j_x/fail"),
        ("post", "/v1/admin/window/rotate"),
        ("put", "/v1/admin/ladder"),
        ("put", "/v1/admin/crowns"),
        ("post", "/v1/admin/jobs/j_x/requeue"),
    ],
)
def test_worker_and_admin_routes_need_their_own_bearer(client, method, path):
    for token in (None, INTERNAL, "nope"):
        headers = bearer(token) if token else {}
        assert getattr(client, method)(path, headers=headers).status_code == 401
    other = ADMIN if path.startswith("/v1/worker") else WORKER
    assert getattr(client, method)(path, headers=bearer(other)).status_code == 401


def test_public_views(client):
    status = client.get("/v1/status").json()
    assert status["champion"]["hotkey"] is None
    assert status["champion"]["repo"] == "google/diffusiongemma-26B-A4B-it"
    assert [lv["state"] for lv in status["levels"][:3]] == ["active", "active", "pending"]
    assert sum(status["next_duel_mix"].values()) == pytest.approx(1)
    assert len(status["window"]["commitment"]) == 64
    assert status["window"]["bank_digest"] == EMPTY_BANK.digest
    assert status["plan"] == {"decisions": {"weight": 1.0, "cases": 40}}
    assert status["teacher"] == {"state": "off", "judge": False, "judgments_pending": 0}
    assert client.get("/v1/leaderboard").json()["crowns"][0]["id"] == 1
    assert client.get("/v1/submissions/s_missing").status_code == 404
    assert client.get("/v1/windows/99").status_code == 404


def test_window_secret_is_revealed_only_after_rotation(client):
    window = client.get("/v1/windows/1").json()
    assert "secret" not in window and window["revealed"] is False
    rotated = client.post("/v1/admin/window/rotate", headers=bearer(ADMIN)).json()
    assert rotated == {
        "closed": 1,
        "opened": 2,
        "commitment": rotated["commitment"],
        "bank_digest": EMPTY_BANK.digest,
    }
    assert client.get("/v1/windows/1/bank").json()["items"] == []
    assert client.get("/v1/windows/2/bank").status_code == 404  # open: sealed
    closed = client.get("/v1/windows/1").json()
    import hashlib

    assert hashlib.sha256(bytes.fromhex(closed["secret"])).hexdigest() == window["commitment"]
    assert "secret" not in client.get("/v1/windows/2").json()
    listed = client.get("/v1/windows").json()["windows"]
    assert [w["revealed"] for w in listed] == [True, False]
    assert all("secret" not in w for w in listed)


def test_admin_ladder_and_pause(client):
    headers = bearer(ADMIN)
    assert (
        client.put("/v1/admin/ladder", headers=headers, json={"order": [9], "width": 1}).status_code
        == 400
    )
    assert (
        client.put(
            "/v1/admin/ladder", headers=headers, json={"order": [2, 3], "width": 3}
        ).status_code
        == 400
    )
    ok = client.put("/v1/admin/ladder", headers=headers, json={"order": [3, 4, 5], "width": 1})
    assert ok.json() == {"order": [3, 4, 5], "width": 1, "retired": []}
    assert client.get("/v1/status").json()["next_duel_mix"] == {"3": 1.0}
    assert client.put("/v1/admin/crowns", headers=headers, json={"paused": True}).json() == {
        "crowns_paused": True
    }
    assert client.get("/v1/status").json()["crowns_paused"] is True


def test_worker_body_limit(client):
    big = json.dumps(
        {"lease": "x", "items": [{"case_index": 0, "side": "champion", "error": "e" * 400}] * 3000}
    )
    response = client.post(
        "/v1/worker/jobs/j_x/answers",
        content=big,
        headers={**bearer(WORKER), "content-type": "application/json"},
    )
    assert response.status_code == 413
