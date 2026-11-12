"""One-shot converter: Qwen3.8-27B-QSA GGUF  ->  a single .ninfer artifact.

This is the entry point this fork used to produce
``E:\\LLMs\\Qwen3.8-27B-QSA.ninfer`` (1292 objects, 17.80 GB). It wraps the
repo's general converter (``python -m tools.convert``) with the
``qwen3_8_27b_qsa_gguf`` recipe, which is what teaches the converter about the
80 extra F32 ``blk.N.indexer.*`` tensors the QSA GGUF carries.

The recipe itself lives in ``tools/convert/qsa_gguf.py`` and is registered in
``tools/convert/gguf_blocks.py``. ``scripts/verify_qsa_gguf.py`` is the offline
preflight that checks the GGUF header and tensor set before converting.

Example (the exact command used for the verified artifact)::

    python tools/convert/convert_qsa27b.py ^
      --base-model E:\\LLMs\\ninfer-base\\Qwen3.8-27B ^
      --gguf E:\\LLMs\\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf ^
      --vision E:\\LLMs\\mmproj-BF16.gguf ^
      --dflash2 E:\\LLMs\\ninfer-dflash2\\official ^
      --out E:\\LLMs\\Qwen3.8-27B-QSA.ninfer

Model card: https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(argv: list[str]) -> int:
    print("+ " + " ".join(f'"{a}"' if " " in a else a for a in argv), flush=True)
    return subprocess.call(argv, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pack a Qwen3.8-27B-QSA GGUF into a .ninfer artifact")
    p.add_argument("--base-model", type=Path, required=True,
                   help="the stock Qwen3.8-27B model directory (config/tokenizer source)")
    p.add_argument("--gguf", type=Path, required=True, help="the QSA-27B Q4_K_M GGUF")
    p.add_argument("--vision", type=Path, default=None, help="mmproj-BF16.gguf")
    p.add_argument("--dflash2", type=Path, default=None,
                   help="the infer-dflash2 official weight directory")
    p.add_argument("--out", type=Path, required=True, help="output .ninfer path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--components", default="text,vision,mtp,dflash2")
    p.add_argument("--proposal", action="store_true",
                   help="pack the indexed proposal head (required for --lm-head-draft)")
    p.add_argument("--proposal-rows", type=int, default=131072,
                   help="shortlist size of the proposal head (default 131072)")
    p.add_argument("--ranking", type=Path, default=None,
                   help="token-frequency ranking fixture (converter default if unset)")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    if not args.gguf.is_file():
        p.error(f"GGUF not found: {args.gguf}")
    if not args.base_model.exists():
        p.error(f"base model not found: {args.base_model}")

    if not args.skip_preflight:
        print("== preflight ==", flush=True)
        code = run([sys.executable, "scripts/verify_qsa_gguf.py", str(args.gguf)])
        if code != 0:
            print("preflight failed; refusing to convert", file=sys.stderr)
            return code

    cmd = [sys.executable, "-m", "tools.convert",
           "--model", str(args.base_model),
           "--recipe", "qwen3_8_27b_qsa_gguf",
           "--components", args.components,
           "--source", f"gguf={args.gguf}",
           "--out", str(args.out),
           "--device", args.device]
    if args.vision is not None:
        cmd += ["--source", f"vision={args.vision}"]
    if args.dflash2 is not None:
        cmd += ["--source", f"dflash2={args.dflash2}"]
    if args.proposal:
        cmd += ["--proposal", "--proposal-rows", str(args.proposal_rows)]
        if args.ranking is not None:
            cmd += ["--ranking", str(args.ranking)]
    print("== convert ==", flush=True)
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
