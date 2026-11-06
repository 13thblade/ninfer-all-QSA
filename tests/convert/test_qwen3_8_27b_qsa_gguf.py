"""Offline tests for the Qwen3.8-27B-QSA GGUF import path.

These tests build a synthetic GGUF header (shapes, types and metadata only) so they
run without the 16-30 GB model file.  The real-file preflight is
``scripts/verify_qsa_gguf.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import pytest
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
from tools.convert import gguf_blocks, qsa_gguf
from tools.convert.model import Model
from tools.convert.ternary import HIDDEN, LAYERS, full_attention

# Q8_0: a real ggml block type that GGUF_FORMATS_BY_TYPE accepts.
BLOCK_TYPE_ID = 8
BLOCK_TYPE_NAME = "Q8_0"


@dataclass
class FakeInfo:
    name: str
    shape: tuple[int, ...]
    type_name: str
    type_id: int
    offset: int = 0

    @property
    def nbytes(self) -> int:
        # Only used for the truncation check; the fake file has no data section.
        return prod(self.shape)


class FakeGGUF:
    def __init__(
        self,
        *,
        qsa: bool = True,
        mtp: bool = True,
        budget: int | None = None,
        bad_ratios: bool = False,
        drop: tuple[str, ...] = (),
    ) -> None:
        self.path = "fake-qsa-27b.gguf"
        self.kv = dict(gguf_blocks.EXPECTED_HEADER)
        self.kv["qwen35.block_count"] = LAYERS + (1 if mtp else 0)
        if qsa:
            self.kv.update(qsa_gguf.QSA_HEADER)
            if budget is not None:
                self.kv["qwen35.attention.indexer.top_k"] = budget
            ratios = [
                qsa_gguf.COMPRESS_RATIO if full_attention(layer) else 0
                for layer in range(LAYERS)
            ] + [0]
            if bad_ratios:
                ratios[qsa_gguf.qsa_layers()[0]] = 0
            self.kv[qsa_gguf.COMPRESS_RATIOS_KEY] = ratios

        spec = gguf_blocks.expected_tensors(mtp, qsa=qsa)
        self.tensors: dict[str, FakeInfo] = {}
        for name, (shape, kind) in spec.items():
            if name in drop:
                continue
            kinds = (kind,) if isinstance(kind, str) else kind
            if "blocks" in kinds:
                type_name, type_id = BLOCK_TYPE_NAME, BLOCK_TYPE_ID
            elif "F32" in kinds:
                type_name, type_id = "F32", 0
            elif "BF16" in kinds:
                type_name, type_id = "BF16", 30
            else:  # GDN controls stored as Q8_0
                type_name, type_id = BLOCK_TYPE_NAME, BLOCK_TYPE_ID
            self.tensors[name] = FakeInfo(name, shape, type_name, type_id)
        self.data_bytes_available = 1 << 40

    def read_direct(self, name: str) -> np.ndarray:
        # The fork stores every indexer tensor as F32 (norms as 1 + w).
        return np.full(self.tensors[name].shape, 3.0, dtype=np.float32)


def test_qsa_adds_exactly_80_f32_indexer_tensors():
    stock = gguf_blocks.expected_tensors(True)
    qsa = gguf_blocks.expected_tensors(True, qsa=True)
    extra = set(qsa) - set(stock)
    assert len(extra) == 80
    assert len(qsa_gguf.qsa_layers()) == 16
    assert all(qsa[name][1] == "F32" for name in extra)
    assert set(extra) == set(qsa_gguf.expected_tensors())
    assert qsa["blk.3.indexer.q_proj.weight"][0] == (512, HIDDEN)
    assert qsa["blk.3.indexer.k_proj.weight"][0] == (128, HIDDEN)
    assert all(qsa[f"blk.3.indexer.{n}"][0] == (128,) for n in
               ("q_norm.weight", "k_norm.weight", "k_norm.pre_weight"))


def test_validate_accepts_a_complete_qsa_gguf():
    gguf = FakeGGUF()
    gguf_blocks.validate(gguf, qsa=True)
    qsa_gguf.check_header(gguf)


def test_stock_recipe_still_validates_a_dense_gguf():
    """The qsa=False default must be byte-for-byte the old behaviour."""

    gguf_blocks.validate(FakeGGUF(qsa=False), qsa=False)
    assert set(gguf_blocks.expected_tensors(True)) == set(
        gguf_blocks.expected_tensors(True, qsa=False)
    )


def test_stock_validate_rejects_qsa_gguf_and_names_the_recipe():
    with pytest.raises(ValueError, match="qwen3_8_27b_qsa_gguf"):
        gguf_blocks.validate(FakeGGUF(), qsa=False)


def test_qsa_validate_rejects_a_stock_dense_gguf():
    with pytest.raises(ValueError, match="is this a QSA-27B GGUF"):
        gguf_blocks.validate(FakeGGUF(qsa=False), qsa=True)


def test_qsa_validate_rejects_wrong_budget_and_ratios():
    with pytest.raises(ValueError, match="indexer.top_k"):
        gguf_blocks.validate(FakeGGUF(budget=512), qsa=True)
    with pytest.raises(ValueError, match="compress_ratios"):
        gguf_blocks.validate(FakeGGUF(bad_ratios=True), qsa=True)


def test_qsa_validate_rejects_missing_indexer_tensor():
    missing = qsa_gguf.gguf_tensor(3, "key_pre_norm")
    with pytest.raises(ValueError, match="tensor set mismatch"):
        gguf_blocks.validate(FakeGGUF(drop=(missing,)), qsa=True)


def test_sources_undo_the_one_plus_weight_fold():
    gguf = FakeGGUF()
    result = qsa_gguf.sources(gguf)
    assert len(result) == 80
    for role in ("query", "key"):
        source = result[qsa_gguf.parameter_name(3, role)]
        assert source.shape == qsa_gguf._SHAPES[role]
        values = source.values(0, prod(source.shape))
        assert torch.all(values == 3.0), role
    for role in ("query_norm", "key_norm", "key_pre_norm"):
        source = result[qsa_gguf.parameter_name(3, role)]
        values = source.values(0, prod(source.shape))
        assert torch.all(values == 2.0), role  # stored 3.0 == 1 + w -> w = 2.0


def test_register_parameters_shapes_inputs_and_format():
    model = Model(components={"text": {"config": {}}})
    qsa_gguf.register_parameters(model)
    qsa_names = [n for n in model.parameters if "/indexer/" in n]
    assert len(qsa_names) == 80
    for layer in qsa_gguf.qsa_layers():
        for role in qsa_gguf.ROLES:
            p = model.parameters[qsa_gguf.parameter_name(layer, role)]
            assert p.shape == qsa_gguf._SHAPES[role]
            assert p.direct_format == qsa_gguf.DIRECT_FORMAT
            assert p.residency == "text"
            expected_inputs = (
                (f"text/layers/{layer}/mixer_input",) if role in ("query", "key") else ()
            )
            assert p.inputs == expected_inputs
    qsa_gguf.register_parameters(model)  # idempotent
    assert len([n for n in model.parameters if "/indexer/" in n]) == 80


def test_artifact_config_matches_the_engine_qsa_constants():
    assert qsa_gguf.artifact_config() == {
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_compress_ratio": 4,
        "indexer_budget": 2048,
    }
    assert qsa_gguf.INDEXER_BUDGET // qsa_gguf.COMPRESS_RATIO == 512
    assert BLOCK_TYPE_ID in GGUF_FORMATS_BY_TYPE


def test_qsa_recipe_is_registered():
    assert "qwen3_8_27b_qsa_gguf" in gguf_blocks.RECIPES
    assert callable(gguf_blocks.RECIPES["qwen3_8_27b_qsa_gguf"])
