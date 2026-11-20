"""The long-16k-1M-heavy extraction keeps the checkpoint's values and records how to read them.

Prefrontal-AI's distillation stores four indexer tensors per layer, with the query heads'
rows first in one ``index_qk_proj`` and the norms already zero-centred.  A mistake in the
split or in the ``norm_form`` marker is silent — the artifact still loads, it just selects
different blocks — so this pins both against the checkpoint itself.

The checkpoint is a local 340 MB file with no public download in CI, so the test runs only
with ``NINFER_LONG_16K_PT`` set::

    NINFER_LONG_16K_PT=E:/LLMs/qwen38_27b__long-16k-1M-heavy-qsa.pt \
        pytest tests/convert/test_long_16k_asset.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tools.convert import qsa_gguf
from tools.convert import extract_long_16k_indexer as extractor

CHECKPOINT = os.environ.get("NINFER_LONG_16K_PT")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT or not Path(CHECKPOINT).is_file(),
    reason="set NINFER_LONG_16K_PT to the long-16k-1M-heavy checkpoint to run this",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    assets = tmp_path_factory.mktemp("long-16k-assets")
    assert extractor.extract(str(CHECKPOINT), assets) == 0
    asset = assets / extractor.ASSET_NAME
    manifest = json.loads((assets / extractor.MANIFEST_NAME).read_text(encoding="utf-8"))
    return asset, manifest


@pytest.fixture(scope="module")
def checkpoint():
    return torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)["maps"]


def test_asset_has_four_roles_per_layer(built):
    asset, manifest = built
    with safe_open(str(asset), framework="pt") as handle:
        keys = set(handle.keys())
    layers = qsa_gguf.qsa_layers()
    assert manifest["tensor_count"] == len(layers) * len(qsa_gguf.ROLES_NO_PRE_NORM)
    assert not any(key.endswith("k_norm.pre_weight") for key in keys)
    for layer in layers:
        for role in qsa_gguf.ROLES_NO_PRE_NORM:
            assert qsa_gguf.gguf_tensor(layer, role) in keys


def test_metadata_marks_the_norms_zero_centred(built, checkpoint):
    asset, manifest = built
    with safe_open(str(asset), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["norm_form"] == "zero-centred"
    assert manifest["norm_form"] == "zero-centred"
    stored = float(checkpoint[qsa_gguf.qsa_layers()[0]]["attention_scaling"])
    assert float(metadata["attention_scaling"]) == pytest.approx(stored, abs=1e-6)


def test_projection_splits_into_query_then_key(built, checkpoint):
    asset, _manifest = built
    heads = qsa_gguf.INDEXER_HEADS * qsa_gguf.INDEXER_DIM
    with safe_open(str(asset), framework="pt") as handle:
        for layer in (qsa_gguf.qsa_layers()[0], qsa_gguf.qsa_layers()[-1]):
            projection = checkpoint[layer]["index_qk_proj"].float()
            query = handle.get_tensor(qsa_gguf.gguf_tensor(layer, "query"))
            key = handle.get_tensor(qsa_gguf.gguf_tensor(layer, "key"))
            # Stored BF16, and the converter casts the projection to BF16 anyway: compare in
            # that representation, bit for bit.
            assert torch.equal(query, projection[:heads].to(torch.bfloat16))
            assert torch.equal(key, projection[heads:].to(torch.bfloat16))


def test_norms_are_the_zero_centred_weights(built, checkpoint):
    asset, _manifest = built
    with safe_open(str(asset), framework="pt") as handle:
        for layer in qsa_gguf.qsa_layers():
            for role, key in (("query_norm", "q_layernorm"), ("key_norm", "k_layernorm")):
                stored = handle.get_tensor(qsa_gguf.gguf_tensor(layer, role))
                assert torch.equal(stored, checkpoint[layer][key].float()), (layer, role)


def test_reader_skips_the_unfold(built, checkpoint):
    asset, _manifest = built
    sources = qsa_gguf.sources_from_asset(path=asset, dtype="fp32")
    assert len(sources) == len(qsa_gguf.qsa_layers()) * len(qsa_gguf.ROLES_NO_PRE_NORM)
    layer = qsa_gguf.qsa_layers()[0]
    # The zero-centred w must survive unchanged: subtracting one would shift every gain.
    for role, key in (("query_norm", "q_layernorm"), ("key_norm", "k_layernorm")):
        source = sources[qsa_gguf.parameter_name(layer, role)]
        assert torch.equal(source.values().reshape(source.shape),
                           checkpoint[layer][key].float()), role
    query = sources[qsa_gguf.parameter_name(layer, "query")]
    heads = qsa_gguf.INDEXER_HEADS * qsa_gguf.INDEXER_DIM
    assert torch.equal(query.rows(0, heads),
                       checkpoint[layer]["index_qk_proj"][:heads].to(torch.bfloat16).float())


def test_manifest_hash_matches_the_payload(built):
    asset, manifest = built
    assert manifest["asset_sha256"] == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert manifest["asset_bytes"] == asset.stat().st_size
    # Provenance only: the rotation the donor's model was served under at distillation time.
    # The engine gives the indexer the trunk's plan, so nothing reads this back as a setting.
    assert manifest["distillation_rope"]["factor"] == 40.0
    assert manifest["distillation_rope"]["original_max_position_embeddings"] == 262_144
