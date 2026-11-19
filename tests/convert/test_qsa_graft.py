"""The graft route resolves the indexer norms' direct format exactly as the official one.

A graft converts a stock quantisation and fills the 80 indexer tensors from the committed
asset (or a donor GGUF).  Its indexer parameters must land in the same format the official
``qwen3_8_27b_qsa_gguf`` recipe produces, so ``fp32_direct`` has to reach three calls:
``register_parameters``, and whichever of ``sources_from_asset`` / ``sources`` supplies the
weights.  A regression here is silent: the artifact still loads, it just carries BF16 norms
where the target asked for FP32.

The model and recipe are stubs — the point is the wiring, not the conversion, which
``test_qwen3_8_27b_qsa_gguf.py`` covers against a synthetic GGUF.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.convert import qsa_gguf, qsa_graft_recipe
from tools.convert.model import Model
from tools.convert.sources.gguf import GGUFFile
from tools.convert.ternary import HIDDEN, LAYERS


def _target() -> GGUFFile:
    """A GGUFFile that passes ``isinstance`` without a 20 GB file behind it."""

    gguf = object.__new__(GGUFFile)
    gguf.path = Path("target.gguf")
    gguf.kv = {"qwen35.block_count": LAYERS}
    return gguf


def _model(**config) -> Model:
    # "num_experts" must stay absent: its presence is what the recipe rejects.
    return Model(
        components={"text": {"config": {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, **config}}}
    )


def _graft(monkeypatch, model: Model) -> dict[str, object]:
    seen: dict[str, object] = {}

    monkeypatch.setattr(qsa_graft_recipe, "has_indexer", lambda gguf: False)
    monkeypatch.setattr(qsa_graft_recipe, "validate", lambda gguf: None)
    monkeypatch.setattr(qsa_graft_recipe, "qwen3_8_27b_gguf", lambda *a, **k: None)
    # No indexer parameters to place: this test only watches the format decision.
    monkeypatch.setattr(qsa_gguf, "qsa_layers", lambda: [])
    monkeypatch.setattr(qsa_gguf, "has_indexer_asset", lambda: True)

    def register(model, dtype=None):
        seen["register"] = dtype

    def sources_from_asset(path=None, dtype=None):
        seen["source"] = dtype
        return {}

    monkeypatch.setattr(qsa_gguf, "register_parameters", register)
    monkeypatch.setattr(qsa_gguf, "sources_from_asset", sources_from_asset)

    qsa_graft_recipe.graft_qsa_indexer(model, SimpleNamespace(), {"gguf": _target()})
    return seen


@pytest.mark.parametrize("flag", [True, False])
def test_graft_follows_fp32_direct(monkeypatch, flag):
    expected = "fp32" if flag else "bf16"
    seen = _graft(monkeypatch, _model(fp32_direct=flag))
    assert seen == {"register": expected, "source": expected}


def test_graft_defaults_to_bf16_when_the_flag_is_absent(monkeypatch):
    assert _graft(monkeypatch, _model()) == {"register": "bf16", "source": "bf16"}


@pytest.mark.parametrize("config, expected", [
    ({}, "bf16"),
    ({"fp32_direct": False}, "bf16"),
    ({"fp32_direct": True}, "fp32"),
])
def test_indexer_direct_format(config, expected):
    assert qsa_gguf.indexer_direct_format(config) == expected
