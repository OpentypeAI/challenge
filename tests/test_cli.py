import json
import secrets
from types import SimpleNamespace

import pytest

from opentype_challenge import bank, cli, generator, miner, pins
from opentype_challenge.crypto import decode_hotkey, manifest_digest, submit_message, verify


def test_generate_writes_jsonl_with_exact_targets(tmp_path):
    out = tmp_path / "train.jsonl"
    cli.main(["generate", "--family", "agent_trace", "--level", "3", "--n", "4", "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4 and {row["family"] for row in rows} == {"agent_trace"}
    for row in rows:
        solved = generator.solve(row["request"])
        for qid, (_kind, _options, probs) in row["gold"].items():
            assert solved[qid] == pytest.approx(probs)


def test_audit_regenerates_cases(capsys):
    secret = secrets.token_bytes(32)
    mix = {"2": 0.5, "3": 0.5}
    seed = bank.job_seed(secret, "j_1", "d" * 64)
    cli.main(
        [
            "audit",
            "--window-secret",
            secret.hex(),
            "--commitment",
            bank.commitment(secret),
            "--job",
            "j_1",
            "--digest",
            "d" * 64,
            "--mix",
            json.dumps(mix),
            "--cases",
            "3",
        ]
    )
    line = json.loads(capsys.readouterr().out)
    expected = bank.cases_digest(bank.job_case(seed, mix, i).body for i in range(3))
    assert line["cases_sha256"] == expected and line["ok"]
    with pytest.raises(SystemExit):
        cli.main(["audit", "--window-secret", secret.hex(), "--commitment", "0" * 64])


def test_seed_signer_signature_verifies():
    signer = miner.seed_signer(secrets.token_hex(32))
    files = dict(pins.BASE_FILES)
    body = miner.signed_submission(
        {"repo": "a/b", "revision": "c" * 40, "files": files}, signer, 1000
    )
    assert body["exp"] == 1000 + miner.EXP_SECONDS
    public = decode_hotkey(body["hotkey"])
    digest = manifest_digest("a/b", "c" * 40, files)
    message = submit_message(public, digest, body["nonce"], body["exp"])
    assert verify(public, message, bytes.fromhex(body["signature"]))


def test_wallet_signer_matches_substrate_verification(tmp_path):
    bittensor_wallet = pytest.importorskip("bittensor_wallet")
    keypair = bittensor_wallet.Keypair.create_from_seed("0x" + "11" * 32)
    message = submit_message(keypair.public_key, "0" * 64, "1" * 32, 5)
    assert verify(keypair.public_key, message, keypair.sign(message))


def test_hf_manifest_uses_lfs_sha_and_hashes_small_files(monkeypatch, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    lfs = SimpleNamespace(sha256="a" * 64)
    siblings = [
        SimpleNamespace(rfilename="model.safetensors", lfs=lfs),
        SimpleNamespace(rfilename="config.json", lfs=None),
        SimpleNamespace(rfilename="tokenizer.json", lfs=None),
        SimpleNamespace(rfilename="README.md", lfs=None),
    ]
    info = SimpleNamespace(private=False, gated=False, sha="f" * 40, siblings=siblings)

    class Api:
        def model_info(self, *args, **kwargs):
            assert kwargs["files_metadata"] is True and kwargs["token"] is False
            return info

    def download(repo, name, revision, local_dir, token):
        path = tmp_path / name
        path.write_bytes(b"{}")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest = miner.hf_manifest("org/model", "f" * 40)
    assert manifest["files"] == {
        "model.safetensors": "a" * 64,
        "config.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    }
    info.private = True
    with pytest.raises(SystemExit):
        miner.hf_manifest("org/model", "f" * 40)
