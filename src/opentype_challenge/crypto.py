"""Hotkeys, manifests and submission signatures."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping

import sr25519

_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
SUBMIT_DOMAIN = "opentype-submit-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
WEIGHT_FILE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$|^model\.safetensors$")
# Only weights and config: tokenizer, chat template and processor always come from the base.
MANIFEST_FILES = ("config.json", "model.safetensors.index.json")


class CryptoError(ValueError):
    pass


# decode_hotkey / encode_hotkey: copied from cortex/protocol/crypto.py (checked SS58, network 42).
def decode_hotkey(value: str) -> bytes:
    """Decode a raw/hex public key or checksummed Bittensor SS58 (network 42)."""
    raw = value.removeprefix("0x")
    if len(raw) == 64:
        try:
            return bytes.fromhex(raw)
        except ValueError as error:
            raise CryptoError("invalid hotkey hex") from error
    if not 46 <= len(value) <= 50:
        raise CryptoError("invalid SS58 hotkey")
    number = 0
    for char in value:
        index = _ALPHABET.find(char)
        if index < 0:
            raise CryptoError("invalid SS58 alphabet")
        number = number * 58 + index
    decoded = number.to_bytes((number.bit_length() + 7) // 8, "big")
    decoded = bytes(len(value) - len(value.lstrip("1"))) + decoded
    if len(decoded) != 35 or decoded[0] != 42:
        raise CryptoError("expected Bittensor SS58 network 42")
    checksum = hashlib.blake2b(b"SS58PRE" + decoded[:-2]).digest()[:2]
    if not hmac.compare_digest(checksum, decoded[-2:]):
        raise CryptoError("invalid SS58 checksum")
    return decoded[1:33]


def encode_hotkey(public: bytes) -> str:
    if len(public) != 32:
        raise CryptoError("public key must be 32 bytes")
    payload = b"\x2a" + public
    payload += hashlib.blake2b(b"SS58PRE" + payload).digest()[:2]
    number = int.from_bytes(payload, "big")
    result = ""
    while number:
        number, remainder = divmod(number, 58)
        result = _ALPHABET[remainder] + result
    return result


def allowed_file(name: str) -> bool:
    return name in MANIFEST_FILES or bool(WEIGHT_FILE.match(name))


def manifest_problem(files: Mapping[str, str]) -> str | None:
    """Why a manifest file set is refused, or None."""
    bad = sorted(n for n, digest in files.items() if not allowed_file(n) or not HEX64.match(digest))
    if bad:
        return f"files not allowed or digests not lowercase sha256 hex: {', '.join(bad[:5])}"
    if "config.json" not in files or not any(WEIGHT_FILE.match(n) for n in files):
        return "the manifest needs config.json and safetensors weights"
    shards = [n for n in files if n.startswith("model-")]
    if shards and "model.safetensors.index.json" not in files:
        return "sharded weights need model.safetensors.index.json"
    return None


def canonical_manifest(repo: str, revision: str, files: Mapping[str, str]) -> bytes:
    body = {"files": dict(sorted(files.items())), "repo": repo, "revision": revision}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def manifest_digest(repo: str, revision: str, files: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_manifest(repo, revision, files)).hexdigest()


def submit_message(public: bytes, digest: str, nonce: str, exp: int) -> bytes:
    return f"{SUBMIT_DOMAIN}|{public.hex()}|{digest}|{nonce}|{exp}".encode()


def sign_with_seed(seed: bytes, message: bytes) -> tuple[bytes, bytes]:
    """(public key, signature) with the Substrate sr25519 signing context."""
    if len(seed) != 32:
        raise CryptoError("seed must be 32 bytes")
    public, secret = sr25519.pair_from_seed(seed)
    return public, sr25519.sign((public, secret), message)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    try:
        return bool(sr25519.verify(signature, message, public))
    except ValueError:
        return False
