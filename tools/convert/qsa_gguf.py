"""The learned QSA block indexer of the ``Qwen3.8-27B-QSA-GGUF`` releases.

The 27B QSA adaptation keeps the dense Qwen3.5/3.8-27B text tower (64 layers, 16
full-attention + 48 Gated DeltaNet) and adds a 4-head, 128-dim block indexer to each of
the 16 full-attention layers.  The published GGUFs therefore carry 80 extra F32
tensors on top of the 866 tensors of the stock model (``blk.N.indexer.*``), which the
stock ``qwen3_8_27b_gguf`` recipe rejects as unknown names.  See the fork's
root ``MODIFICATION.md`` and the model card at
huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF. The one-shot converter entry
point is ``tools/convert/convert_qsa27b.py``.

Indexer mathematics (identical to ninfer's existing QSA path, see
``include/ninfer/ops/qsa_indexer.h``)::

    K_b = rope(groupnorm(mean_{i<4} k[4b+i], key_norm), 4b)
    score_b = sum_{h<4} relu(q_h . K_b)       -> the best 2048/4 = 512 blocks

The fork's converter (``conversion/qwen.py``) folds every zero-centred norm as
``1 + w`` and forces all indexer tensors to F32.  This module undoes that offset and
registers the five logical parameters per QSA layer that the engine binds.

The GGUF metadata keys come from the fork's ``gguf-py/gguf/constants.py``:
``qwen35.attention.indexer.{head_count,key_length,top_k}`` plus
``qwen35.attention.compress_ratios`` (4 on full-attention layers, 0 elsewhere).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .model import Model, Parameter
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source
from .ternary import HIDDEN, LAYERS, full_attention

INDEXER_HEADS = 4
INDEXER_DIM = 128
#: Four positions per indexer block; also ``indexer_compress_ratio`` in the engine.
COMPRESS_RATIO = 4
#: Token budget. The engine selects ``INDEXER_BUDGET / COMPRESS_RATIO`` = 512 blocks.
INDEXER_BUDGET = 2048

# ninfer's qsa_pool_kernel reads BF16 norm weights and the engine's direct-parameter
# loader accepts BF16 matrices.  The source GGUF stores F32; casting to BF16 here is
# the same representation the Flash-Next QSA path uses.  Switch to "fp32" together
# with the engine-side direct format when FP32 indexer weights are required.
DIRECT_FORMAT = "bf16"

QSA_HEADER = {
    "qwen35.attention.indexer.head_count": INDEXER_HEADS,
    "qwen35.attention.indexer.key_length": INDEXER_DIM,
    "qwen35.attention.indexer.top_k": INDEXER_BUDGET,
}

COMPRESS_RATIOS_KEY = "qwen35.attention.compress_ratios"

#: The five tensors llama.cpp's QSA-27B converter writes for each full-attention layer.
ROLES = ("query", "key", "query_norm", "key_norm", "key_pre_norm")

_GGUF_SUFFIX = {
    "query": "q_proj.weight",
    "key": "k_proj.weight",
    "query_norm": "q_norm.weight",
    "key_norm": "k_norm.weight",           # pooled-key groupnorm (HF k_block_norm)
    "key_pre_norm": "k_norm.pre_weight",   # RMSNorm on the raw key before pooling
}

_SHAPES = {
    "query": (INDEXER_HEADS * INDEXER_DIM, HIDDEN),   # (512, 5120)
    "key": (INDEXER_DIM, HIDDEN),                     # (128, 5120)
    "query_norm": (INDEXER_DIM,),
    "key_norm": (INDEXER_DIM,),
    "key_pre_norm": (INDEXER_DIM,),
}

# All three norms are stored as ``1 + w`` by the fork's converter.
_NORM_ROLES = frozenset({"query_norm", "key_norm", "key_pre_norm"})


def qsa_layers() -> tuple[int, ...]:
    """The 16 full-attention layer indices of the Qwen3.8-27B text tower."""

    return tuple(layer for layer in range(LAYERS) if full_attention(layer))


def gguf_tensor(layer: int, role: str) -> str:
    return f"blk.{layer}.indexer.{_GGUF_SUFFIX[role]}"


def parameter_name(layer: int, role: str) -> str:
    return f"text/layers/{layer}/attention/indexer/{role}"


def expected_tensors() -> dict[str, tuple[tuple[int, ...], str]]:
    """The 16 x 5 F32 indexer tensors the QSA recipe adds to ``expected_tensors``."""

    return {
        gguf_tensor(layer, role): (_SHAPES[role], "F32")
        for layer in qsa_layers()
        for role in ROLES
    }


def check_header(gguf: GGUFFile) -> None:
    for key, expected in QSA_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}; "
                "is this a QSA-27B GGUF?"
            )
    ratios = gguf.kv.get(COMPRESS_RATIOS_KEY)
    if not isinstance(ratios, (list, tuple, np.ndarray)) or len(ratios) < LAYERS:
        raise ValueError(
            f"{gguf.path}: {COMPRESS_RATIOS_KEY} must be an array of at least {LAYERS} entries"
        )
    for layer in range(LAYERS):
        expected = COMPRESS_RATIO if full_attention(layer) else 0
        if int(ratios[layer]) != expected:
            raise ValueError(
                f"{gguf.path}: {COMPRESS_RATIOS_KEY}[{layer}] = {ratios[layer]!r}, "
                f"expected {expected}"
            )


def sources(gguf: GGUFFile) -> dict[str, LogicalSource]:
    """Read the 80 indexer tensors into logical sources, dropping the stored ``1 + w``.

    ``GGUFFile.read_direct`` returns F32/F16/BF16 words as float32 in row-major shape;
    the QSA release stores every indexer tensor as F32.
    """

    out: dict[str, LogicalSource] = {}
    for layer in qsa_layers():
        for role in ROLES:
            name = gguf_tensor(layer, role)
            words = gguf.read_direct(name).astype(np.float32)
            if role in _NORM_ROLES:
                words = words - 1.0
            out[parameter_name(layer, role)] = array_source(
                torch.from_numpy(np.ascontiguousarray(words)).to(torch.bfloat16), name
            )
    return out


# --- optional offline asset of the 80 F32 indexer tensors ------------------------------
# Route-B graft can read the indexer tensors from a compact asset safetensors built by
# extract_indexer_tensors.py instead of the full ~16 GB donor GGUF. The env var overrides:
#   unset            -> use the committed default asset if present, else require a donor
#   <path>           -> use that asset file
#   off|none|0       -> disable the asset, require a donor GGUF exactly as before
# The asset stores donor F32 values verbatim (norms still carry 1 + w), and sources_from_asset
# below applies the SAME -1.0 un-fold and BF16 cast as sources(), so an asset graft is
# bit-identical to a donor-GGUF graft.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INDEXER_ASSET_NAME = "indexer_tensors_fp32.safetensors"
INDEXER_ASSET_ENV = "NINFER_QSA_INDEXER_ASSET"


def indexer_asset_path() -> Path | None:
    raw = os.environ.get(INDEXER_ASSET_ENV)
    if raw is not None and raw.strip().lower() in ("off", "none", "0"):
        return None
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise ValueError(f"{INDEXER_ASSET_ENV}: {path} is not an indexer asset file")
        return path
    default = _ASSETS_DIR / INDEXER_ASSET_NAME
    return default if default.is_file() else None


def has_indexer_asset() -> bool:
    return indexer_asset_path() is not None


def sources_from_asset(path: Path | None = None) -> dict[str, LogicalSource]:
    """Read the 80 indexer tensors from the asset, converting exactly as sources() does."""
    path = path or indexer_asset_path()
    if path is None:
        raise ValueError("no indexer asset; set NINFER_QSA_INDEXER_ASSET or pass a donor GGUF")

    from safetensors import safe_open

    out: dict[str, LogicalSource] = {}
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        expected = {gguf_tensor(layer, role) for layer in qsa_layers() for role in ROLES}
        missing = sorted(expected - keys)
        if missing:
            raise ValueError(
                f"indexer asset {path} is missing {len(missing)} tensors ({missing[:3]}...)")
        for layer in qsa_layers():
            for role in ROLES:
                name = gguf_tensor(layer, role)
                words = handle.get_tensor(name).detach().to("cpu").numpy().astype(np.float32)
                if tuple(words.shape) != tuple(_SHAPES[role]):
                    raise ValueError(
                        f"{name}: asset shape {tuple(words.shape)} != expected {_SHAPES[role]}")
                if role in _NORM_ROLES:
                    words = words - 1.0
                out[parameter_name(layer, role)] = array_source(
                    torch.from_numpy(np.ascontiguousarray(words)).to(torch.bfloat16),
                    f"{name}[indexer-asset]",
                )
    return out


def register_parameters(model: Model) -> None:
    """Add the QSA logical parameters to a model built from the dense 27B geometry.

    The stock ``qwen3_5`` builder does not know about the indexer.  Registering the
    parameters here (with correctly shaped placeholder sources) lets the existing
    ``Recipe.assign`` / ``cast_direct`` pipeline write them into the artifact without
    changing the builder used by every other recipe.
    """

    for layer in qsa_layers():
        prefix = f"text/layers/{layer}/"
        for role in ROLES:
            name = parameter_name(layer, role)
            if name in model.parameters:
                continue
            shape = _SHAPES[role]
            placeholder = array_source(
                torch.zeros(shape, dtype=torch.bfloat16), name + ":placeholder"
            )
            model.add(
                Parameter(
                    name,
                    shape,
                    placeholder,
                    inputs=(prefix + "mixer_input",) if role in ("query", "key") else (),
                    direct_format=DIRECT_FORMAT,
                    residency="text",
                )
            )


def artifact_config() -> dict:
    """Fields the engine's qwen3_5 config needs to enable the QSA path."""

    return {
        "indexer_n_heads": INDEXER_HEADS,
        "indexer_head_dim": INDEXER_DIM,
        "indexer_compress_ratio": COMPRESS_RATIO,
        "indexer_budget": INDEXER_BUDGET,
    }


__all__ = [
    "COMPRESS_RATIO",
    "COMPRESS_RATIOS_KEY",
    "DIRECT_FORMAT",
    "INDEXER_BUDGET",
    "INDEXER_DIM",
    "INDEXER_HEADS",
    "QSA_HEADER",
    "ROLES",
    "artifact_config",
    "check_header",
    "expected_tensors",
    "gguf_tensor",
    "parameter_name",
    "qsa_layers",
    "register_parameters",
    "sources",
    "sources_from_asset",
    "indexer_asset_path",
    "has_indexer_asset",
    "INDEXER_ASSET_ENV",
    "INDEXER_ASSET_NAME",
]
