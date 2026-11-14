"""Extract the native BF16 GDN controls (in_proj_a / in_proj_b) from an FP8 HF release
into a compact, committed asset so the GGUF->ninfer converter can graft them without the
full FP8 tree.

Read-only on the FP8 source; writes two files under tools/convert/assets/:
  gdn_controls_fp8_bf16.safetensors   96 bf16 tensors, HF keys, grouped head order VERBATIM
  gdn_controls_fp8_bf16.manifest.json source id, per-tensor shape/dtype, asset sha256

The keys are kept as the official HF names
(`model.language_model.layers.N.linear_attn.in_proj_{a,b}.weight`) so the converter's
provider maps 1:1 with what it read from the FP8 tree during C1 verification.

Run with the ninfer python env:
  <py> tools/convert/extract_gdn_controls.py [FP8_DIR]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DEFAULT = r"E:\LLMs\Qwen3.8-27B-FP8"
ASSETS = Path(__file__).resolve().parent / "assets"
ASSET_NAME = "gdn_controls_fp8_bf16.safetensors"
MANIFEST_NAME = "gdn_controls_fp8_bf16.manifest.json"
EXPECT_ROWS, EXPECT_COLS = 48, 5120   # GDN_VALUE_HEADS, HIDDEN


def extract(src: str) -> int:
    src_path = Path(src)
    index = src_path / "model.safetensors.index.json"
    if not index.exists():
        raise SystemExit(f"{index}: not an HF release directory")
    weight_map = json.load(open(index, encoding="utf-8"))["weight_map"]

    keys = sorted(
        k for k in weight_map
        if k.endswith("linear_attn.in_proj_a.weight") or k.endswith("linear_attn.in_proj_b.weight")
    )
    if len(keys) != 96:
        raise SystemExit(f"expected 96 in_proj_a/b tensors, found {len(keys)}")

    tensors: dict[str, torch.Tensor] = {}
    manifest_tensors: dict[str, dict] = {}
    # Open each shard once, pull the keys it owns.
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(str(src_path / shard), framework="pt") as f:
            for k in shard_keys:
                t = f.get_tensor(k).detach().to("cpu")
                if t.dtype != torch.bfloat16:
                    raise SystemExit(f"{k}: expected bf16, got {t.dtype}")
                if tuple(t.shape) != (EXPECT_ROWS, EXPECT_COLS):
                    raise SystemExit(f"{k}: shape {tuple(t.shape)} != {(EXPECT_ROWS, EXPECT_COLS)}")
                tensors[k] = t.contiguous()
                manifest_tensors[k] = {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "sha256": hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest(),
                }

    ASSETS.mkdir(exist_ok=True)
    asset = ASSETS / ASSET_NAME
    save_file(tensors, str(asset), metadata={"format": "pt", "source": src})

    data = asset.read_bytes()
    manifest = {
        "source_id": os.path.basename(src.rstrip("\\/")),
        "tensor_count": len(manifest_tensors),
        "head_order": "grouped (native HF, VERBATIM - the converter must NOT re-untile)",
        "asset_bytes": len(data),
        "asset_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": manifest_tensors,
        "git_note": ("~45 MiB binary; committing to git history is irreversible repo "
                     "bloat - decide plain commit vs .gitignore vs LFS"),
    }
    (ASSETS / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {asset} ({len(data)/1e6:.2f} MB, {len(tensors)} tensors)")
    print(f"asset_sha256 = {manifest['asset_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(extract(sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT))
