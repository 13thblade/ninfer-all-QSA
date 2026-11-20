"""The graft reads the long-16k indexer by default, and the legacy v2 asset only when named.

The two assets are not interchangeable: the long-16k one carries the pooled-key indexer
(four roles, zero-centred norms) distilled from a .pt checkpoint, the legacy one the
kuraneko1 v2 five-role indexer copied out of a GGUF.  A missing default is a real state
(fresh checkout: ``*.safetensors`` is gitignored), and the only acceptable response is to
ask for the .pt donor -- silently substituting the legacy weights would change what the
model computes without saying so.
"""

from __future__ import annotations

import pytest

from tools.convert import qsa_gguf


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(qsa_gguf.INDEXER_ASSET_ENV, raising=False)
    monkeypatch.setattr(qsa_gguf, "_indexer_warned", [False])


def test_default_is_the_long_16k_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    (tmp_path / qsa_gguf.INDEXER_ASSET_NAME).write_bytes(b"long-16k")
    assert qsa_gguf.indexer_asset_path() == tmp_path / qsa_gguf.INDEXER_ASSET_NAME


def test_missing_default_asks_for_the_pt_donor_and_never_downgrades(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    # A legacy payload sitting beside the missing default must not be picked up.
    (tmp_path / qsa_gguf.LEGACY_INDEXER_ASSET_NAME).write_bytes(b"legacy")
    assert qsa_gguf.indexer_asset_path() is None
    err = capsys.readouterr().err
    assert "extract_long_16k_indexer.py" in err
    assert ".pt" in err
    assert "legacy" in err


def test_legacy_is_used_only_when_named(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    (tmp_path / qsa_gguf.LEGACY_INDEXER_ASSET_NAME).write_bytes(b"legacy")
    assert qsa_gguf.indexer_asset_path() is None
    for token in ("legacy", "legacy-v2", "LEGACY"):
        monkeypatch.setenv(qsa_gguf.INDEXER_ASSET_ENV, token)
        assert qsa_gguf.indexer_asset_path() == tmp_path / qsa_gguf.LEGACY_INDEXER_ASSET_NAME


def test_named_legacy_that_is_missing_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    monkeypatch.setenv(qsa_gguf.INDEXER_ASSET_ENV, "legacy")
    with pytest.raises(ValueError, match="extract_indexer_tensors.py"):
        qsa_gguf.indexer_asset_path()


def test_explicit_path_wins_over_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    (tmp_path / qsa_gguf.INDEXER_ASSET_NAME).write_bytes(b"long-16k")
    chosen = tmp_path / "chosen.safetensors"
    chosen.write_bytes(b"chosen")
    monkeypatch.setenv(qsa_gguf.INDEXER_ASSET_ENV, str(chosen))
    assert qsa_gguf.indexer_asset_path() == chosen


def test_off_disables_the_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(qsa_gguf, "_ASSETS_DIR", tmp_path)
    (tmp_path / qsa_gguf.INDEXER_ASSET_NAME).write_bytes(b"long-16k")
    for token in ("off", "none", "0"):
        monkeypatch.setenv(qsa_gguf.INDEXER_ASSET_ENV, token)
        assert qsa_gguf.indexer_asset_path() is None
