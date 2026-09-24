"""Miner client: build a manifest from Hugging Face metadata, sign it, submit it."""

from __future__ import annotations

import hashlib
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .crypto import (
    allowed_file,
    encode_hotkey,
    manifest_digest,
    manifest_problem,
    sign_with_seed,
    submit_message,
)

EXP_SECONDS = 240


@dataclass(frozen=True)
class Signer:
    public: bytes
    sign: Callable[[bytes], bytes]


def hf_manifest(repo: str, revision: str) -> dict[str, Any]:
    """The manifest of a public repo at a commit: LFS sha256 from the hub metadata, and a
    local sha256 of small non-LFS files (config.json, the weight index)."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
    if info.private or info.gated:
        raise SystemExit(f"{repo} must be public and ungated")
    if info.sha != revision:
        raise SystemExit(f"{repo}: pass the full 40-hex commit sha, not {revision!r}")
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sibling in info.siblings or []:
            name = sibling.rfilename
            if not allowed_file(name):
                continue
            if sibling.lfs is not None:
                files[name] = sibling.lfs.sha256
            else:
                path = hf_hub_download(repo, name, revision=revision, local_dir=tmp, token=False)
                files[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    problem = manifest_problem(files)
    if problem:
        raise SystemExit(f"{repo}@{revision}: {problem}")
    return {"repo": repo, "revision": revision, "files": files}


def seed_signer(seed_hex: str) -> Signer:
    """A signer from a 32-byte hex sr25519 mini-secret (the hotkey seed)."""
    seed = bytes.fromhex(seed_hex.strip().removeprefix("0x"))
    public = sign_with_seed(seed, b"")[0]
    return Signer(public, lambda message: sign_with_seed(seed, message)[1])


def wallet_signer(name: str, hotkey: str, path: str | None = None) -> Signer:
    """A signer from a bittensor wallet hotkey (needs the `wallet` extra)."""
    from bittensor_wallet import Wallet

    keypair = Wallet(name=name, hotkey=hotkey, **({"path": path} if path else {})).hotkey
    if keypair.public_key is None:
        raise SystemExit(f"wallet {name}/{hotkey} has no hotkey")
    return Signer(keypair.public_key, lambda message: bytes(keypair.sign(message)))


def signed_submission(
    manifest: dict[str, Any], signer: Signer, now: float | None = None
) -> dict[str, Any]:
    digest = manifest_digest(manifest["repo"], manifest["revision"], manifest["files"])
    nonce = secrets.token_hex(16)
    exp = int(now if now is not None else time.time()) + EXP_SECONDS
    signature = signer.sign(submit_message(signer.public, digest, nonce, exp))
    return {
        "manifest": manifest,
        "hotkey": encode_hotkey(signer.public),
        "nonce": nonce,
        "exp": exp,
        "signature": signature.hex(),
    }


def post(api: str, body: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(api.rstrip("/") + "/v1/submissions", json=body, timeout=60)
    if response.status_code != 201:
        raise SystemExit(f"submission refused: {response.status_code} {response.text}")
    result: dict[str, Any] = response.json()
    return result


def status(api: str, submission_id: str) -> dict[str, Any]:
    response = httpx.get(api.rstrip("/") + f"/v1/submissions/{submission_id}", timeout=30)
    if response.status_code != 200:
        raise SystemExit(f"{response.status_code} {response.text}")
    result: dict[str, Any] = response.json()
    return result
