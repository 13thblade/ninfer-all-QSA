"""Extract the QSA indexer of Prefrontal-AI's LSA distillation (vLSA) into the asset format
the graft reads, so a transplant no longer needs a kuraneko1 GGUF.

The checkpoint is a ``maps`` dict of sixteen layers (3, 7, ... 63), each holding
``index_qk_proj`` ``[(4 + 1) * 128, 5120]`` with the query heads' rows first, the two
zero-centred Gemma norms ``q_layernorm`` / ``k_layernorm``, and three runtime scalars
(``log_tau``, ``inverse_frequencies``, ``attention_scaling``) that are not weights.

Two differences from a kuraneko1 donor decide the shape of this asset:

* the indexer has **no per-token key norm**, so only four roles are written and the
  artifact never declares ``key_pre_norm`` -- which is what selects the Op's
  pool-then-norm path;
* the norms are already the additive gamma ``w`` of ``1 + w``, recorded as
  ``norm_form=zero-centred`` in the safetensors metadata so the reader skips its
  ``-1`` un-fold.

The runtime scalars are checked, not trusted: this checkpoint's ``inverse_frequencies`` is
``theta^(-2i/64)`` under YaRN factor 40 over a 262,144-token window, and ``attention_scaling``
is that factor's ``0.1 * ln(40) + 1``.  Extraction fails if either stops holding, so a
different distillation cannot silently inherit the wrong rope.

Per ``.gitignore`` (``*.safetensors``) the asset is NOT committed; it ships as a GitHub
Release attachment (verified against the manifest ``asset_sha256``), or is rebuilt here.
Run with the ninfer python env::

    <py> tools/convert/extract_lsa_indexer.py [CHECKPOINT.pt] [ASSET_DIR]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import qsa_gguf                      # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
ASSET_NAME = "indexer_tensors_lsa.safetensors"
MANIFEST_NAME = "indexer_tensors_lsa.manifest.json"

#: The indexer's rotation as this checkpoint trained it: YaRN factor 40 over the model's
#: 262,144-token native window, at the model's 1e7 theta, on 64 of the 128 head dims.
ROPE = {"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 262_144}
THETA = 1.0e7
ROTARY_PAIRS = 32
#: What the engine feeds ``rope_yarn_frequencies`` for the indexer (``2 * kRotaryPairs``).
ROPE_DIM = 2 * ROTARY_PAIRS


def _yarn_table(factor: float, native: int, theta: float) -> tuple[np.ndarray, float]:
    """``yarn_schedule`` of ``src/ops/launcher/rope.cu``, in float64 then float32.

    Same beta_fast=32 / beta_slow=1 correction range and linear ramp, so this is what the
    engine will compute for the same ``RopeYarn``.
    """

    log_base = math.log(theta)
    correction = lambda rotations: (ROPE_DIM * math.log(native / (rotations * 2.0 * math.pi))
                                    / (2.0 * log_base))
    low = max(math.floor(correction(32.0)), 0.0)
    high = min(math.ceil(correction(1.0)), ROPE_DIM - 1.0)
    if high == low:
        high += 0.001
    table = []
    for pair in range(ROTARY_PAIRS):
        frequency = theta ** (-2.0 * pair / ROPE_DIM)
        ramp = min(max((pair - low) / (high - low), 0.0), 1.0) if factor > 1.0 else 0.0
        table.append(frequency * ((1.0 - ramp) + ramp / factor))
    scaling = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return np.asarray(table, dtype=np.float32), float(scaling)


def _check_rope(layer: int, maps: dict) -> tuple[float, np.ndarray]:
    """Refuse a checkpoint whose recorded rope differs from ``ROPE``."""

    reference = maps[layer]
    stored_frequencies = reference["inverse_frequencies"].detach().to("cpu").float().numpy()
    stored_scaling = float(reference["attention_scaling"])
    table, scaling = _yarn_table(ROPE["factor"], ROPE["original_max_position_embeddings"], THETA)
    if stored_frequencies.shape != table.shape:
        raise SystemExit(
            f"layer {layer}: inverse_frequencies is {stored_frequencies.shape}, "
            f"expected {table.shape}")
    relative = np.abs(stored_frequencies - table) / np.maximum(np.abs(table), 1e-30)
    if float(relative.max()) > 1e-5:
        raise SystemExit(
            f"layer {layer}: inverse_frequencies do not match YaRN factor "
            f"{ROPE['factor']} over {ROPE['original_max_position_embeddings']} "
            f"(max relative difference {float(relative.max()):.3e}); check ROPE")
    if abs(stored_scaling - scaling) > 1e-6:
        raise SystemExit(
            f"layer {layer}: attention_scaling is {stored_scaling}, expected {scaling} "
            f"for factor {ROPE['factor']}")
    return stored_scaling, stored_frequencies


def extract(checkpoint: str, assets: Path = ASSETS) -> int:
    source = Path(checkpoint)
    if not source.is_file():
        raise SystemExit(f"checkpoint not found: {source}")
    try:
        obj = torch.load(str(source), map_location="cpu", weights_only=True)
    except Exception as error:                        # noqa: BLE001 - reported verbatim
        raise SystemExit(f"{source}: not a weights-only torch checkpoint ({error})")

    maps = obj.get("maps") if isinstance(obj, dict) else None
    if not isinstance(maps, dict) or not maps:
        raise SystemExit(f"{source}: no 'maps' dict; is this the QSA distillation checkpoint?")
    try:
        layers = sorted(int(layer) for layer in maps)
    except (TypeError, ValueError):
        raise SystemExit(f"{source}: 'maps' keys are not layer indices")
    expected_layers = list(qsa_gguf.qsa_layers())
    if layers != expected_layers:
        raise SystemExit(f"{source}: layers {layers[:4]}... != {expected_layers[:4]}...")

    roles = qsa_gguf.ROLES_NO_PRE_NORM
    query_rows = qsa_gguf.INDEXER_HEADS * qsa_gguf.INDEXER_DIM
    expected_projection = (query_rows + qsa_gguf.INDEXER_DIM, qsa_gguf.HIDDEN)
    weights: dict[str, torch.Tensor] = {}
    manifest_tensors: dict[str, dict] = {}
    scaling, frequencies = 0.0, np.zeros(ROTARY_PAIRS, dtype=np.float32)
    log_tau: dict[str, float] = {}

    for index, layer in enumerate(layers):
        entry = maps[layer]
        missing = sorted({"index_qk_proj", "q_layernorm", "k_layernorm"} - set(entry))
        if missing:
            raise SystemExit(f"layer {layer}: missing {missing}")
        # Every layer must carry the same rope; the check is cheap and catches a mixed export.
        layer_scaling, layer_frequencies = _check_rope(layer, maps)
        if index == 0:
            scaling, frequencies = layer_scaling, layer_frequencies
        projection = entry["index_qk_proj"].detach().to("cpu").float()
        if tuple(projection.shape) != expected_projection:
            raise SystemExit(
                f"layer {layer}: index_qk_proj is {tuple(projection.shape)}, "
                f"expected {expected_projection}")
        if not torch.isfinite(projection).all():
            raise SystemExit(f"layer {layer}: index_qk_proj has non-finite values")
        pieces = {
            # The query heads' rows first, then the single key head (model.py's split).
            "query": projection[:query_rows].contiguous(),
            "key": projection[query_rows:].contiguous(),
            "query_norm": entry["q_layernorm"].detach().to("cpu").float().contiguous(),
            "key_norm": entry["k_layernorm"].detach().to("cpu").float().contiguous(),
        }
        if tuple(pieces["key"].shape) != (qsa_gguf.INDEXER_DIM, expected_projection[1]):
            raise SystemExit(f"layer {layer}: key rows are {tuple(pieces['key'].shape)}")
        for role, tensor in pieces.items():
            if role in qsa_gguf._NORM_ROLES:
                if tuple(tensor.shape) != (qsa_gguf.INDEXER_DIM,):
                    raise SystemExit(f"layer {layer}: {role} is {tuple(tensor.shape)}")
                if not torch.isfinite(tensor).all():
                    raise SystemExit(f"layer {layer}: {role} has non-finite values")
            else:
                # The two projections are BF16 in every artifact; storing them so keeps the
                # asset small without losing anything the converter would keep.
                tensor = tensor.to(torch.bfloat16)
            name = qsa_gguf.gguf_tensor(layer, role)
            weights[name] = tensor
            manifest_tensors[name] = {
                "dtype": "F32" if role in qsa_gguf._NORM_ROLES else "BF16",
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                "stored": ("the checkpoint's zero-centred w, verbatim (gain 1 + w, no un-fold)"
                           if role in qsa_gguf._NORM_ROLES else
                           "checkpoint F32 rounded to BF16 (the converter casts these anyway)"),
            }
        log_tau[str(layer)] = float(entry["log_tau"]) if "log_tau" in entry else float("nan")

    assets.mkdir(parents=True, exist_ok=True)
    asset = assets / ASSET_NAME
    save_file(weights, str(asset), metadata={
        "format": "pt",
        "source": source.name,
        "donor": "vLSA",
        "norm_form": "zero-centred",
        "attention_scaling": f"{scaling:.9g}",
        "indexer_rope": f"yarn factor {ROPE['factor']:g} over "
                        f"{ROPE['original_max_position_embeddings']}",
    })

    data = asset.read_bytes()
    manifest = {
        "source_donor": source.name,
        "donor_family": "vLSA (Prefrontal-AI/latent-sparse-attention)",
        "tensor_count": len(manifest_tensors),
        "precision": "BF16 projections + F32 zero-centred norms (per-tensor dtype below)",
        "norm_form": "zero-centred",
        "roles": list(roles),
        "indexer_rope": ROPE,
        "attention_scaling": scaling,
        "inverse_frequencies": [float(v) for v in frequencies],
        "log_tau": log_tau,
        "asset_bytes": len(data),
        "asset_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": manifest_tensors,
        "git_note": ("*.safetensors is gitignored; ship as a GitHub Release attachment "
                     "verified against asset_sha256, or rebuild from the checkpoint"),
    }
    (assets / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {asset} ({len(data) / 1e6:.2f} MB, {len(weights)} tensors)")
    print(f"asset_sha256 = {manifest['asset_sha256']}")
    print(f"rope: factor {ROPE['factor']:g} over {ROPE['original_max_position_embeddings']}, "
          f"attention_scaling {scaling:.9f}")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NINFER_LSA_PT")
    if not path:
        raise SystemExit(
            "usage: extract_lsa_indexer.py CHECKPOINT.pt [ASSET_DIR] "
            "(or set NINFER_LSA_PT; there is no built-in path)")
    directory = Path(sys.argv[2]) if len(sys.argv) > 2 else ASSETS
    sys.exit(extract(path, directory))
