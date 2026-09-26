"""The B300 pilots (deploy/modal_pilot.py) on CPU: Modal stubbed, the bootstrap local, the
staged snapshot's real file inventory (two shards, the index, the base support files)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from opentype_challenge import pins, runtime, sandbox

from .test_e2e import _sandbox_launcher

DEPLOY = Path(__file__).parent.parent / "deploy"
SMALL = {
    track: {"track": track, "cases": 2, "concurrency": 2, "slo_ms": 120000.0}
    for track in runtime.CELL_TRACKS
}


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    """modal_pilot with Modal stubbed and a snapshot laid out as stage_nvfp4 writes it."""
    snap = tmp_path / "snap" / "nvfp4"
    snap.mkdir(parents=True)
    files = {
        "config.json": b"nvfp4 config",
        "model.safetensors.index.json": b'{"weight_map": {}}',
        "model-00001-of-00002.safetensors": b"exact 1",
        "model-00002-of-00002.safetensors": b"exact 2",
    }
    support = {n: f"support {n}".encode() for n in pins.BASE_SUPPORT_FILES}
    for name, data in {**files, **support}.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)
    sha = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    for name, value in (
        ("NVFP4_FILES", sha),
        ("NVFP4_CONFIG_SHA256", sha["config.json"]),
        ("NVFP4_SCHEMA_SHA256", "5" * 64),
        ("BASE_SUPPORT_FILES", {n: hashlib.sha256(b).hexdigest() for n, b in support.items()}),
    ):
        monkeypatch.setattr(pins, name, value)
    monkeypatch.setitem(runtime.PROFILE_FIXED, "weights_config_sha256", sha["config.json"])
    monkeypatch.setitem(runtime.PROFILE_FIXED, "weights_schema_sha256", "5" * 64)
    monkeypatch.setattr(sandbox, "tensor_schema", lambda model: "5" * 64)

    volume = types.SimpleNamespace(with_mount_options=lambda **k: None, commit=lambda: None)
    fake_app = types.SimpleNamespace(
        function=lambda **k: lambda f: f, local_entrypoint=lambda: lambda f: f
    )
    modal = types.SimpleNamespace(
        App=lambda *a, **k: fake_app,
        Volume=types.SimpleNamespace(from_name=lambda *a, **k: volume),
    )
    runtime_module = types.ModuleType("modal_runtime")
    stubs: dict[str, Any] = {"SNAPSHOT": "/snap", "image": None, "snapshot": volume}
    for name, value in {**stubs, "_directory": lambda: snap}.items():
        setattr(runtime_module, name, value)
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal_runtime", runtime_module)
    monkeypatch.syspath_prepend(str(DEPLOY))
    monkeypatch.delitem(sys.modules, "modal_pilot", raising=False)
    import modal_pilot  # type: ignore[import-not-found]

    monkeypatch.setattr(modal_pilot, "WORK", str(tmp_path / "work"))
    starts: list[str] = []
    launcher = _sandbox_launcher(tmp_path, starts)
    monkeypatch.setattr(sandbox, "ModalBackend", lambda *a, **k: None)
    monkeypatch.setattr(sandbox, "SandboxLauncher", lambda backend, ready_timeout: launcher)
    return modal_pilot, launcher, starts, snap


def test_the_quality_pilot_runs_one_production_duel_on_the_snapshot(pilot):
    modal_pilot, launcher, starts, _ = pilot
    out = modal_pilot.quality(cases="decisions=4,longctx=2,ops=2,sql=2")
    assert out["error"] is None and out["ran"] is True and out["quiescent"]
    submission = out["submission"]
    assert submission["repo"] == modal_pilot.PILOT_REPO
    assert submission["state"] in ("rejected", "crowned")  # whatever the container scored
    assert submission["job"]["paired"] == 10 and len(starts) == 2
    evidence = submission["job"]["evidence"]
    assert set(evidence["profiles"]) == {"champion", "challenger"}
    assert out["run_dir_kept"] is None


def test_the_pilot_fetch_serves_only_the_pinned_inventory(pilot, tmp_path):
    modal_pilot, _, _, _ = pilot
    with pytest.raises(RuntimeError, match="does not serve"):
        modal_pilot._snapshot_fetch(pins.BASE_REPO, "r", "model.safetensors", tmp_path)
    with pytest.raises(RuntimeError, match="does not serve"):
        modal_pilot._snapshot_fetch("other/repo", "r", "config.json", tmp_path)


def test_calibration_measures_every_track_and_zero_goodput_is_no_decision(pilot):
    modal_pilot, _, starts, snap = pilot
    out = modal_pilot.calibration(blocks=1, cells=json.dumps(SMALL))
    assert out["decision"] == "measured" and len(starts) == 3
    assert {r["side"] for r in out["runs"]} == set(runtime.SIDES)
    assert all(set(r["cells"]) == set(SMALL) for r in out["runs"])
    for shard in snap.glob("model-*.safetensors"):
        shard.write_text("broken")
    out = modal_pilot.calibration(blocks=1, cells=json.dumps(SMALL))
    assert out["decision"] == "no_decision" and "zero goodput" in out["reason"]


def test_calibration_refuses_missing_tracks_and_oversized_cells(pilot):
    modal_pilot, _, starts, _ = pilot
    without_sql = {k: v for k, v in SMALL.items() if k != "sql"}
    huge = {**SMALL, "sql": {**SMALL["sql"], "cases": 10_000}}
    for cells in (without_sql, huge):
        with pytest.raises(SystemExit):
            modal_pilot.calibration(blocks=1, cells=json.dumps(cells))
    assert starts == []  # refused before any sandbox


def test_an_off_profile_run_is_reported_not_measured(pilot, monkeypatch):
    """The sandbox reports another image than the pinned one: nothing is timed."""
    modal_pilot, _, _, _ = pilot
    monkeypatch.setitem(runtime.PROFILE_FIXED, "vllm_image", "vllm/pinned@sha256:" + "0" * 64)
    out = modal_pilot.calibration(blocks=1, cells=json.dumps(SMALL))
    assert out["decision"] == "incomplete" and "off the profile" in out["error"]
    assert out["runs"] == [] and out["quiescent"]


def test_a_failed_pilot_still_writes_its_report(pilot, tmp_path):
    modal_pilot, _, _, _ = pilot
    path = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="report kept"):
        modal_pilot._write_private(str(path), {"error": "boom", "runs": []})
    assert json.loads(path.read_text())["error"] == "boom"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        modal_pilot._write_private(str(path), {"error": None})


@pytest.fixture
def source(tmp_path, monkeypatch):
    """deploy/modal_runtime.source_sha256 over a copy of the uploaded tree (Modal mocked)."""
    from unittest import mock

    monkeypatch.setitem(sys.modules, "modal", mock.MagicMock())
    monkeypatch.syspath_prepend(str(DEPLOY))
    monkeypatch.delitem(sys.modules, "modal_runtime", raising=False)
    import modal_runtime  # type: ignore[import-not-found]

    for name in ("src/opentype_challenge/a.py", *modal_runtime.UPLOADED):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    return modal_runtime, tmp_path


def test_the_source_hash_is_stable_and_covers_the_controller(source):
    modal_runtime, root = source
    first = modal_runtime.source_sha256(root)
    assert modal_runtime.source_sha256(root) == first
    (root / "deploy/modal_controller.py").write_text("a modified controller")
    assert modal_runtime.source_sha256(root) != first


def test_the_source_hash_refuses_a_symlink(source):
    modal_runtime, root = source
    (root / "src/opentype_challenge/link.py").symlink_to(root / "README.md")
    with pytest.raises(SystemExit, match="symlink"):
        modal_runtime.source_sha256(root)


def test_the_real_checkout_hashes(source):
    modal_runtime, _ = source
    assert len(modal_runtime.source_sha256()) == 64
