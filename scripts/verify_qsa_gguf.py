#!/usr/bin/env python3
"""Preflight a downloaded Qwen3.8-27B-QSA GGUF before running the converter.

Offline structural check: metadata, the exact 946-tensor set (866 stock + 80 QSA
indexer, incl. the MTP block), F32 indexer types and the data-section extent.
It does not read tensor payloads, so it is fast on a 16-30 GB file.

    python scripts/verify_qsa_gguf.py E:/LLMs/Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf

Exit status 0 = the file is accepted by ``qwen3_8_27b_qsa_gguf``'s validator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.convert import gguf_blocks, qsa_gguf  # noqa: E402
from tools.convert.sources.gguf import GGUFFile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="path to the QSA-27B .gguf file")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()
    path = args.gguf
    if not path.is_file():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    with GGUFFile(path) as gguf:
        try:
            gguf_blocks.validate(gguf, qsa=True)
        except ValueError as error:
            print(f"REJECT: {error}", file=sys.stderr)
            return 1
        qsa_names = [n for n in gguf.tensors if ".indexer." in n]
        types = Counter(gguf.tensors[n].type_name for n in qsa_names)
        report = {
            "path": str(path),
            "file_bytes": gguf.file_bytes,
            "tensors": len(gguf.tensors),
            "qsa_indexer_tensors": len(qsa_names),
            "qsa_indexer_types": dict(types),
            "mtp": gguf_blocks.has_mtp(gguf),
            "block_count": gguf.kv.get("qwen35.block_count"),
            "indexer_budget": gguf.kv.get("qwen35.attention.indexer.top_k"),
            "indexer_heads": gguf.kv.get("qwen35.attention.indexer.head_count"),
            "indexer_key_length": gguf.kv.get("qwen35.attention.indexer.key_length"),
            "compress_ratio": qsa_gguf.COMPRESS_RATIO,
            "selected_blocks": qsa_gguf.INDEXER_BUDGET // qsa_gguf.COMPRESS_RATIO,
            "status": "accepted",
        }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:24s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
