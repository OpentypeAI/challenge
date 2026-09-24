"""Signed manifest intake: signature, nonce, expiry, metagraph, limits, clones."""

import json
import secrets

import pytest

from opentype_challenge import pins
from opentype_challenge.crypto import CryptoError, decode_hotkey, encode_hotkey
from opentype_challenge.miner import signed_submission

from .conftest import Miner, submit, weights_manifest

# A known Bittensor hotkey (Alice's well-known dev key).
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def test_ss58_round_trip_and_checks():
    public = decode_hotkey(ALICE)
    assert public.hex() == "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
    assert encode_hotkey(public) == ALICE
    assert decode_hotkey("0x" + public.hex()) == public
    with pytest.raises(CryptoError):
        decode_hotkey(ALICE[:-1] + ("Z" if ALICE[-1] != "Z" else "Y"))
    with pytest.raises(CryptoError):
        decode_hotkey("0" * 48)


def test_accepted_submission(client, miner, clock):
    response = submit(client, miner, weights_manifest("a"), clock)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"] == "queued" and body["hotkey"] == miner.hotkey
    assert body["job"]["state"] == "queued" and body["job"]["cases"] == 40
    assert client.get(f"/v1/submissions/{body['id']}").json()["id"] == body["id"]
    assert client.get("/v1/status").json()["queue"][0]["id"] == body["id"]


def test_bad_signature_is_401(client, miner, clock):
    body = signed_submission(weights_manifest("a"), miner.signer, clock.now)
    body["manifest"]["files"]["model-00001-of-00011.safetensors"] = "0" * 64
    assert client.post("/v1/submissions", json=body).status_code == 401
    other = Miner(secrets.token_bytes(32))
    forged = signed_submission(weights_manifest("a"), other.signer, clock.now)
    forged["hotkey"] = miner.hotkey
    assert client.post("/v1/submissions", json=forged).status_code == 401


def test_unregistered_hotkey_is_403(client, clock):
    stranger = Miner(secrets.token_bytes(32))
    assert submit(client, stranger, weights_manifest("a"), clock).status_code == 403


def test_metagraph_outage_is_503_and_consumes_nothing(client, miner, master, clock):
    master.up = False
    body = signed_submission(weights_manifest("a"), miner.signer, clock.now)
    assert client.post("/v1/submissions", json=body).status_code == 503
    master.up = True
    assert client.post("/v1/submissions", json=body).status_code == 201  # same nonce still valid


def test_nonce_is_single_use(client, miner, clock):
    body = signed_submission(weights_manifest("a"), miner.signer, clock.now)
    assert client.post("/v1/submissions", json=body).status_code == 201
    client.app.state.store._db.execute("UPDATE submissions SET state='rejected'")
    assert client.post("/v1/submissions", json=body).status_code == 409


@pytest.mark.parametrize("seconds_left", [0, -1, 301])
def test_expiry_window(client, miner, clock, seconds_left):
    """exp must be in the future and at most five minutes ahead."""
    body = signed_submission(weights_manifest("a"), miner.signer, clock.now)
    clock.now = body["exp"] - seconds_left
    assert client.post("/v1/submissions", json=body).status_code == 400


def test_one_open_submission_per_hotkey_and_global_cap(make_client, miner, master, clock):
    client = make_client(max_pending=2)
    second = Miner(secrets.token_bytes(32))
    third = Miner(secrets.token_bytes(32))
    master.hotkeys.update({second.hotkey: 2, third.hotkey: 3})
    assert submit(client, miner, weights_manifest("a"), clock).status_code == 201
    assert submit(client, miner, weights_manifest("b"), clock).status_code == 409
    assert submit(client, second, weights_manifest("c"), clock).status_code == 201
    assert submit(client, third, weights_manifest("d"), clock).status_code == 429


def test_clone_of_the_champion_is_rejected_without_a_duel(client, miner, clock):
    clone = {"repo": "copy/cat", "revision": "a" * 40, "files": dict(pins.BASE_FILES)}
    response = submit(client, miner, clone, clock)
    assert response.status_code == 409 and "clone" in response.json()["detail"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["files"].update({"tokenizer.json": "a" * 64}),
        lambda m: m["files"].update({"modeling_evil.py": "a" * 64}),
        lambda m: m["files"].update({"config.json": "b" * 64}),
        lambda m: m["files"].pop("model.safetensors.index.json"),
        lambda m: m["files"].update({"model-00001-of-00011.safetensors": "XYZ"}),
        lambda m: m.update({"revision": "main"}),
        lambda m: m.update({"repo": "../etc"}),
    ],
)
def test_manifest_policy(client, miner, clock, mutate):
    manifest = weights_manifest("a")
    mutate(manifest)
    assert submit(client, miner, manifest, clock).status_code == 422


def test_body_limit_and_unknown_fields(client, miner, clock):
    body = signed_submission(weights_manifest("a"), miner.signer, clock.now)
    body["extra"] = 1
    assert client.post("/v1/submissions", json=body).status_code == 422
    huge = json.dumps({**body, "pad": "x" * 70_000})
    response = client.post(
        "/v1/submissions", content=huge, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
