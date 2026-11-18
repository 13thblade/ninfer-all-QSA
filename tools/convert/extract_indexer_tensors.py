"""Extract the 80 F32 QSA-27B indexer tensors from a QSA donor GGUF into one asset
safetensors + manifest, so the route-B graft no longer needs the full ~16 GB donor tree.

The donor's raw F32 values are stored VERBATIM (the three norms still carry the fork's
`1 + w` offset). qsa_gguf.sources applies the un-fold (-1.0) and the BF16 cast, so a graft
built from this asset is bit-identical to one built from the donor GGUF.

Per .gitignore (`*.safetensors`) the asset is NOT committed to git; it ships as a GitHub
Release attachment (verified against the manifest asset_sha256), or is rebuilt here from a
donor GGUF. Run with the ninfer python env:
  <py> tools/convert/extract_indexer_tensors.py DONOR.gguf   (or set QSA_INDEXER_GGUF)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import qsa_gguf                      # noqa: E402
from tools.convert.sources.gguf import GGUFFile          # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
ASSET_NAME = "indexer_tensors.safetensors"
MANIFEST_NAME = "indexer_tensors.manifest.json"


def extract(donor: str) -> int:
    donor_path = Path(donor)
    if not donor_path.is_file():
        raise SystemExit(f"donor GGUF not found: {donor_path}")

    g = GGUFFile(str(donor_path))
    qsa_gguf.check_header(g)                     # refuse a non-QSA-27B donor
    expected = qsa_gguf.expected_tensors()        # 80 tensors: {blk.N.indexer.x: (shape, "F32")}
    missing = sorted(k for k in expected if k not in g.tensors)
    if missing:
        raise SystemExit(f"donor is missing {len(missing)} indexer tensors ({missing[:3]}...)")

    tensors: dict[str, torch.Tensor] = {}
    manifest_tensors: dict[str, dict] = {}
    for name, (shape, _kind) in expected.items():
        info = g.info(name)
        if info.type_name != "F32":
            raise SystemExit(f"{name}: expected F32 in donor, got {info.type_name}")
        arr = g.read_direct(name)                 # float32, row-major, VERBATIM (1 + w kept)
        if tuple(arr.shape) != tuple(shape):
            raise SystemExit(f"{name}: shape {tuple(arr.shape)} != expected {tuple(shape)}")
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name}: non-finite indexer weights")
        # The engine keeps these as BF16, and the two projections are cast straight to BF16 by
        # the converter (no arithmetic in between), so storing them as BF16 loses nothing and
        # halves the asset. The three norms must stay F32: the converter subtracts 1.0 (the
        # stored 1 + w) in F32 *before* the BF16 cast, and bf16(x) - 1 != bf16(x - 1).
        keep_f32 = any(name.endswith(suffix) for suffix in ("q_norm.weight", "k_norm.weight",
                                                            "k_norm.pre_weight"))
        t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
        if not keep_f32:
            t = t.to(torch.bfloat16)
        tensors[name] = t
        manifest_tensors[name] = {
            "dtype": "F32" if keep_f32 else "BF16",
            "shape": list(shape),
            "sha256": hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest(),
            "stored": ("verbatim donor value (norm, still carries 1 + w; F32 for the exact "
                       "un-fold before the BF16 cast)"
                       if keep_f32 else
                       "donor F32 rounded to BF16 (the converter casts these to BF16 anyway)"),
        }

    ASSETS.mkdir(exist_ok=True)
    asset = ASSETS / ASSET_NAME
    save_file(tensors, str(asset), metadata={"format": "pt", "source": donor_path.name})

    data = asset.read_bytes()
    manifest = {
        "source_donor": donor_path.name,
        "tensor_count": len(manifest_tensors),
        "precision": "BF16 projections + F32 norms (per-tensor dtype below)",
        "asset_bytes": len(data),
        "asset_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": manifest_tensors,
        "git_note": ("*.safetensors is gitignored; ship as a GitHub Release attachment "
                     "verified against asset_sha256, or rebuild from a donor GGUF"),
    }
    (ASSETS / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {asset} ({len(data)/1e6:.2f} MB, {len(tensors)} tensors)")
    print(f"asset_sha256 = {manifest['asset_sha256']}")
    return 0


if __name__ == "__main__":
    donor = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QSA_INDEXER_GGUF")
    if not donor:
        raise SystemExit(
            "usage: extract_indexer_tensors.py DONOR.gguf "
            "(or set QSA_INDEXER_GGUF; there is no built-in path)"
        )
    sys.exit(extract(donor))
