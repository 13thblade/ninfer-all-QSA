"""The graft route resolves the indexer's direct format and role set exactly as the official one.

A graft converts a stock quantisation and fills the indexer tensors from the committed
asset (or a donor GGUF).  Its indexer parameters must land in the same format the official
``qwen3_8_27b_qsa_gguf`` recipe produces, so ``fp32_direct`` has to reach three calls:
``register_parameters``, and whichever of ``sources_from_asset`` / ``sources`` supplies the
weights.  A regression here is silent: the artifact still loads, it just carries BF16 norms
where the target asked for FP32.

The role set is the second half: a donor without the indexer's per-token key norm
(Prefrontal-AI's long-16k-1M-heavy run) supplies four tensors per layer, and the artifact
must register exactly those — the missing binding is what selects the Op's pool-then-norm
path.

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


class _Recipe:
    """Records what the recipe asked of it; assigning and grouping do nothing."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.assigned: list[str] = []

    def add_parameter(self, name):
        self.added.append(name)

    def assign(self, name, **_kwargs):
        self.assigned.append(name)

    def group(self, _names):
        pass

    def separate(self, _names):
        pass


class _Handle:
    """The slice of ``safe_open`` the role/norm-form readers touch."""

    def __init__(self, keys, metadata=None) -> None:
        self._keys = set(keys)
        self._metadata = metadata

    def keys(self):
        return sorted(self._keys)

    def metadata(self):
        return self._metadata


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


def _graft(monkeypatch, roles, **config) -> tuple[dict, Model, _Recipe]:
    """Run the graft body against one layer and `roles`, returning what it did."""

    seen: dict[str, object] = {}

    monkeypatch.setattr(qsa_graft_recipe, "has_indexer", lambda gguf: False)
    monkeypatch.setattr(qsa_graft_recipe, "validate", lambda gguf: None)
    monkeypatch.setattr(qsa_graft_recipe, "qwen3_8_27b_gguf", lambda *a, **k: None)
    monkeypatch.setattr(qsa_gguf, "has_indexer_asset", lambda: True)
    # One layer keeps the registration's placeholder tensors small; the recipe is
    # layer-agnostic, so the role set is what this test is about.
    monkeypatch.setattr(qsa_gguf, "qsa_layers", lambda: [3])

    def sources_from_asset(path=None, dtype=None):
        seen["source"] = dtype
        return {qsa_gguf.parameter_name(3, role): None for role in roles}

    monkeypatch.setattr(qsa_gguf, "sources_from_asset", sources_from_asset)

    model = _model(**config)
    recipe = _Recipe()
    qsa_graft_recipe.graft_qsa_indexer(model, recipe, {"gguf": _target()})
    return seen, model, recipe


@pytest.mark.parametrize("roles, carries", [
    (qsa_gguf.ROLES, True),
    (qsa_gguf.ROLES_NO_PRE_NORM, False),
])
def test_graft_registers_the_roles_the_asset_carries(monkeypatch, roles, carries):
    seen, model, recipe = _graft(monkeypatch, roles)
    pre_norm = qsa_gguf.parameter_name(3, "key_pre_norm")
    assert (pre_norm in model.parameters) is carries
    assert len(model.parameters) == len(roles)
    assert sorted(recipe.added) == sorted(model.parameters)
    # The source call follows the same format decision as the registration.
    assert seen == {"source": "bf16"}


def test_graft_defaults_to_five_roles_for_a_gguf_donor(monkeypatch):
    _seen, _model_, recipe = _graft(monkeypatch, qsa_gguf.ROLES)
    assert len(recipe.added) == len(qsa_gguf.ROLES)


@pytest.mark.parametrize("flag, expected", [(True, "fp32"), (False, "bf16")])
def test_graft_follows_fp32_direct(monkeypatch, flag, expected):
    seen, model, _recipe = _graft(monkeypatch, qsa_gguf.ROLES, fp32_direct=flag)
    assert seen == {"source": expected}
    assert model.parameters[qsa_gguf.parameter_name(3, "query_norm")].direct_format == expected
    # The projections stay BF16 on both: the kernels read them so.
    assert model.parameters[qsa_gguf.parameter_name(3, "query")].direct_format == "bf16"


def test_graft_defaults_to_bf16_when_the_flag_is_absent(monkeypatch):
    seen, model, _recipe = _graft(monkeypatch, qsa_gguf.ROLES)
    assert seen == {"source": "bf16"}
    assert model.parameters[qsa_gguf.parameter_name(3, "query_norm")].direct_format == "bf16"


@pytest.mark.parametrize("config, expected", [
    ({}, "bf16"),
    ({"fp32_direct": False}, "bf16"),
    ({"fp32_direct": True}, "fp32"),
])
def test_indexer_direct_format(config, expected):
    assert qsa_gguf.indexer_direct_format(config) == expected


def _asset_keys(roles) -> list[str]:
    return [qsa_gguf.gguf_tensor(layer, role)
            for layer in qsa_gguf.qsa_layers() for role in roles]


def test_asset_roles_reads_five_or_four_from_the_keys():
    assert qsa_gguf._asset_roles(_Handle(_asset_keys(qsa_gguf.ROLES))) == qsa_gguf.ROLES
    assert (qsa_gguf._asset_roles(_Handle(_asset_keys(qsa_gguf.ROLES_NO_PRE_NORM)))
            == qsa_gguf.ROLES_NO_PRE_NORM)


def test_asset_roles_refuses_a_mixed_asset():
    keys = _asset_keys(qsa_gguf.ROLES_NO_PRE_NORM)
    keys.append(qsa_gguf.gguf_tensor(3, "key_pre_norm"))
    with pytest.raises(ValueError, match="some layers but not others|on 1 of"):
        qsa_gguf._asset_roles(_Handle(keys))


def test_asset_norm_form_defaults_to_one_plus():
    assert qsa_gguf._asset_norm_form(_Handle([], None)) == "one-plus"
    assert qsa_gguf._asset_norm_form(_Handle([], {"norm_form": "zero-centred"})) == "zero-centred"
