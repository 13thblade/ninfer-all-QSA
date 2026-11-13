"""Converter recipe: graft the QSA-27B block indexer while converting a GGUF.

This is the route-B form of the QSA transplant: the 866 base tensors are read from
the target GGUF and the 80 F32 ``blk.N.indexer.*`` tensors from a QSA donor GGUF in
the same conversion pass, so no grafted 18-22 GB GGUF is ever written.

    python -m tools.convert ^
      --model "E:\\LLMs\\ninfer-base\\Qwen3.8-27B" ^
      --recipe qsa_graft_recipe.py:graft_qsa_indexer ^
      --components text,vision,mtp,dflash2 ^
      --source gguf="E:\\LLMs\\Qwen3.8-27B-UD-Q6_K.gguf" ^
      --source indexer="E:\\LLMs\\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf" ^
      --source vision="E:\\LLMs\\mmproj-BF16.gguf" ^
      --source dflash2="E:\\LLMs\\ninfer-dflash2\\official" ^
      --out "E:\\LLMs\\Qwen3.8-27B-QSA-UD-Q6_K.ninfer" ^
      --device cpu

The recipe also accepts an already-QSA GGUF: when the target already carries its
indexer tensors the graft step is skipped and the file goes through the fork's
official ``qwen3_8_27b_qsa_gguf`` recipe unchanged.  The donor is then unused, so
``--source indexer=...`` may be omitted.

Why a byte-identical transplant is the official method
------------------------------------------------------
The published Qwen3.8-27B-QSA Unsloth variants are not retrained: they copy all
866 tensors of the stock Unsloth GGUF byte-for-byte and append the same 80 FP32
indexer tensors ("added without retraining or requantization" -- kuraneko1/
Qwen3.8-27B-QSA-llama.cpp).  ``tools/convert/convert_qsa27b.py`` drives this recipe.

The donor's three indexer norms are stored zero-centred as ``1 + w``; ``qsa_gguf.sources``
un-folds that offset on read, which is exactly the contract of the official recipe.
"""

from __future__ import annotations

import os
from pathlib import Path

from tools.convert import qsa_gguf
from tools.convert.gguf_blocks import (
    _group_same_format,
    has_mtp,
    qwen3_8_27b_gguf,
    qwen3_8_27b_qsa_gguf,
    validate,
)
from tools.convert.methods import cast_direct
from tools.convert.sources.gguf import GGUFFile
from tools.convert.ternary import HIDDEN, LAYERS

#: Environment fallback so the recipe can run without ``--source indexer``.
DONOR_ENV = "QSA_INDEXER_GGUF"

#: The only GGUF metadata key a donor may not share with an incompatible base.
INDEXER_TENSOR = ".indexer."


def has_indexer(gguf: GGUFFile) -> bool:
    """True when this GGUF already carries the 80 QSA indexer tensors."""

    return any(INDEXER_TENSOR in name for name in gguf.tensors)


def donor_path(sources) -> Path | None:
    """The donor GGUF named by ``--source indexer=`` or ``QSA_INDEXER_GGUF``."""

    source = None
    try:
        source = sources["indexer"]
    except (KeyError, ValueError):
        source = None
    path = getattr(source, "path", None) or os.environ.get(DONOR_ENV)
    return Path(path) if path else None


def open_donor(sources) -> GGUFFile:
    """Return the donor as a ``GGUFFile`` (re-opening it if it came in another wrapper)."""

    path = donor_path(sources)
    if path is None:
        raise ValueError(
            "this target GGUF has no QSA indexer tensors, so the graft needs a donor: "
            "--source indexer=<Qwen3.8-27B-QSA>.gguf (or set QSA_INDEXER_GGUF)"
        )
    source = sources["indexer"] if "indexer" in sources else None
    if isinstance(source, GGUFFile):
        return source
    return GGUFFile(path)


def check_donor(donor: GGUFFile, block_count: int | None = None) -> None:
    """Refuse any donor that is not a Qwen3.8-27B-QSA indexer carrier."""

    if donor.kv.get("general.architecture") != "qwen35":
        raise ValueError(
            f"{donor.path}: general.architecture is "
            f"{donor.kv.get('general.architecture')!r}, expected 'qwen35'"
        )
    # The indexer header keys and the 4-on-full-attention-layer ratio pattern.
    qsa_gguf.check_header(donor)
    if int(donor.kv.get("qwen35.embedding_length") or 0) != HIDDEN:
        raise ValueError(
            f"{donor.path}: embedding_length is "
            f"{donor.kv.get('qwen35.embedding_length')!r}, expected {HIDDEN}"
        )
    if block_count is not None and int(donor.kv.get("qwen35.block_count") or 0) != block_count:
        raise ValueError(
            f"{donor.path}: block_count is {donor.kv.get('qwen35.block_count')!r}, "
            f"the target has {block_count}"
        )

    expected = qsa_gguf.expected_tensors()
    missing = sorted(name for name in expected if name not in donor.tensors)
    if missing:
        raise ValueError(
            f"{donor.path}: not a QSA-27B donor, missing {len(missing)} indexer tensors "
            f"({missing[:3]}...)"
        )
    for name, (shape, kind) in expected.items():
        info = donor.tensors[name]
        if info.shape != shape or info.type_name != kind:
            raise ValueError(
                f"{donor.path}: {name} is {info.type_name} {info.shape}, "
                f"expected {kind} {shape}"
            )
    extra = sorted(
        name for name in donor.tensors if INDEXER_TENSOR in name and name not in expected
    )
    if extra:
        raise ValueError(f"{donor.path}: unexpected extra indexer tensors {extra[:3]}")


def graft_qsa_indexer(model, recipe, sources) -> None:
    """Target GGUF + QSA donor -> a QSA ``.ninfer``; already-QSA targets convert directly."""

    target = sources["gguf"]
    if not isinstance(target, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")

    if has_indexer(target):
        # Preflight said the file is already grafted: no donor, no rewrite.
        qwen3_8_27b_qsa_gguf(model, recipe, sources)
        return

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError("the QSA graft recipe requires the Qwen3.8-27B Dense geometry")
    validate(target)
    mtp = "mtp" in model.components
    if mtp and not has_mtp(target):
        raise ValueError(f"{target.path}: the MTP component needs an -mtp GGUF (blk.64.nextn.*)")

    # Indexer source: prefer the committed asset; fall back to a donor GGUF (legacy path).
    if qsa_gguf.has_indexer_asset():
        indexer_sources = qsa_gguf.sources_from_asset()
    else:
        donor = open_donor(sources)
        if donor.path.resolve() == target.path.resolve():
            raise ValueError(f"{donor.path}: the donor must not be the target file")
        check_donor(donor, block_count=target.kv.get("qwen35.block_count"))
        indexer_sources = qsa_gguf.sources(donor)

    # Register the indexer parameters and their artifact config before the stock
    # recipe body runs, exactly as qwen3_8_27b_qsa_gguf does.
    qsa_gguf.register_parameters(model)
    model.config.update(qsa_gguf.artifact_config())
    for layer in qsa_gguf.qsa_layers():
        for role in qsa_gguf.ROLES:
            recipe.add_parameter(qsa_gguf.parameter_name(layer, role))

    # Text tower, MTP block, vision and DFlash2 come from the target.
    qwen3_8_27b_gguf(model, recipe, sources)

    # ... and the 80 indexer parameters are filled from the asset or the donor. Both
    # un-fold the stored 1 + w norms, so the bytes match the official path.
    formats: dict[str, str] = {}
    for name, source in indexer_sources.items():
        parameter = model.parameters[name]
        recipe.assign(
            name, format=parameter.direct_format, method=cast_direct, source=source
        )
        formats[name] = parameter.direct_format
    for layer in qsa_gguf.qsa_layers():
        prefix = f"text/layers/{layer}/attention/indexer/"
        _group_same_format(recipe, [prefix + "query", prefix + "key"], formats)


configure = graft_qsa_indexer

__all__ = ["check_donor", "configure", "donor_path", "graft_qsa_indexer", "has_indexer"]
