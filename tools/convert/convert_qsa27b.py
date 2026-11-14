"""One-shot converter: any Qwen3.8-27B GGUF -> a single .ninfer artifact.

Handles both inputs, deciding from the file itself:

1. **already grafted** -- the GGUF carries its 80 F32 ``blk.N.indexer.*`` tensors.
   The graft step is skipped and the converted file goes through the fork's
   official ``qwen3_8_27b_qsa_gguf`` recipe, exactly as before.
2. **stock GGUF** -- the GGUF has the 866 dense tensors only.  The 80 indexer
   tensors are transplanted from a QSA donor *during* conversion
   (``tools/convert/qsa_graft_recipe.py``), so no intermediate grafted GGUF is
   written.  This is the published method: the QSA Unsloth releases copy the 866
   stock tensors byte-for-byte and append the same 80 FP32 indexer tensors
   without retraining or requantization.

Example (route B -- stock GGUF + donor, direct to .ninfer)::

    python tools/convert/convert_qsa27b.py ^
      --base-model E:\\LLMs\\ninfer-base\\Qwen3.8-27B ^
      --gguf E:\\LLMs\\Qwen3.8-27B-UD-Q6_K.gguf ^
      --donor E:\\LLMs\\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf ^
      --vision E:\\LLMs\\mmproj-BF16.gguf ^
      --dflash2 E:\\LLMs\\ninfer-dflash2\\official ^
      --out E:\\LLMs\\Qwen3.8-27B-QSA-UD-Q6_K.ninfer ^
      --proposal

Example (already-QSA GGUF, no donor)::

    python tools/convert/convert_qsa27b.py ^
      --base-model E:\\LLMs\\ninfer-base\\Qwen3.8-27B ^
      --gguf E:\\LLMs\\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf ^
      --out E:\\LLMs\\Qwen3.8-27B-QSA.ninfer

Model card: https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAFT_RECIPE = ROOT / "tools" / "convert" / "qsa_graft_recipe.py"
OFFICIAL_RECIPE = "qwen3_8_27b_qsa_gguf"

#: The donor this machine already has; only a fallback when --donor is omitted.
LOCAL_DONOR = Path("E:/LLMs/Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf")
DONOR_ENV = "QSA_INDEXER_GGUF"


def run(argv: list[str]) -> int:
    print("+ " + " ".join(f'"{a}"' if " " in a else a for a in argv), flush=True)
    return subprocess.call(argv, cwd=str(ROOT))


def resolve_donor(explicit: Path | None) -> Path | None:
    for candidate in (explicit, os.environ.get(DONOR_ENV), LOCAL_DONOR):
        if candidate is None:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
        if explicit is not None:
            raise SystemExit(f"error: donor GGUF not found: {path}")
    return None


def preflight(gguf_path: Path, donor_path: Path | None, *, check: bool) -> tuple[bool, bool]:
    """Report the header, return ``(already_qsa, mtp)``; validate unless ``check`` is off."""

    sys.path.insert(0, str(ROOT))
    try:
        from tools.convert import qsa_graft_recipe as graft
        from tools.convert.gguf_blocks import has_mtp, validate
        from tools.convert.sources.gguf import GGUFFile
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"error: {error.name!r} is not importable by {sys.executable}\n"
            "       run this script with the conversion environment, e.g.\n"
            r"       C:\Users\bili8\.workbuddy\binaries\python\envs\ninfer\Scripts\python.exe"
        ) from error

    with GGUFFile(gguf_path) as source:
        tensors = len(source.tensors)
        indexer = sum(1 for name in source.tensors if graft.INDEXER_TENSOR in name)
        already = indexer > 0
        print(f"source   : {gguf_path}")
        print(f"           {source.file_bytes:,} bytes, {tensors} tensors, "
              f"indexer tensors {indexer}")
        print(f"arch     : {source.kv.get('general.architecture')} "
              f"block_count={source.kv.get('qwen35.block_count')} "
              f"hidden={source.kv.get('qwen35.embedding_length')}")
        mtp = has_mtp(source)
        if not check:
            print("validate : skipped")
        elif already:
            validate(source, qsa=True)
            print("validate : PASS (already carries the QSA indexer)")
        else:
            validate(source)
            print("validate : PASS (stock Qwen3.8-27B text tower)")
        block_count = source.kv.get("qwen35.block_count")

    if not already and check:
        if donor_path is None:
            raise SystemExit(
                "error: this GGUF has no QSA indexer tensors and no donor was given;\n"
                "       pass --donor <Qwen3.8-27B-QSA>.gguf (or set QSA_INDEXER_GGUF),\n"
                "       or use an already-grafted GGUF"
            )
        with GGUFFile(donor_path) as donor:
            print(f"donor    : {donor_path}")
            print(f"           {donor.file_bytes:,} bytes, {len(donor.tensors)} tensors")
            graft.check_donor(donor, block_count=block_count)
            print("validate : PASS (QSA donor)")
    return already, mtp


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pack a Qwen3.8-27B GGUF into a .ninfer artifact, grafting the "
                    "QSA indexer from a donor when the GGUF does not carry one")
    p.add_argument("--base-model", type=Path, required=True,
                   help="the stock Qwen3.8-27B model directory (config/tokenizer source)")
    p.add_argument("--gguf", type=Path, required=True,
                   help="the source GGUF: stock (needs --donor) or already QSA-grafted")
    p.add_argument("--donor", type=Path, default=None,
                   help="Qwen3.8-27B-QSA GGUF that carries the 80 F32 indexer tensors "
                        f"(default: ${DONOR_ENV}, then {LOCAL_DONOR})")
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
    p.add_argument("--name", default=None, help="public instance name saved in metadata")
    p.add_argument("--force", action="store_true",
                   help="delete an existing --out and its .conversion.json first")
    p.add_argument("--check-only", action="store_true",
                   help="run the preflight, print the chosen recipe and stop")
    p.add_argument("--skip-preflight", action="store_true",
                   help="skip the shape/tensor-set gate (the QSA detection still runs)")
    args = p.parse_args()

    if not args.gguf.is_file():
        p.error(f"GGUF not found: {args.gguf}")
    if not args.base_model.exists():
        p.error(f"base model not found: {args.base_model}")
    components = tuple(part for part in args.components.split(",") if part)
    if "vision" in components and args.vision is None:
        p.error("--components includes vision but --vision was not given")
    if "dflash2" in components and args.dflash2 is None:
        p.error("--components includes dflash2 but --dflash2 was not given")

    print("== preflight ==", flush=True)
    already, mtp = preflight(args.gguf, args.donor, check=not args.skip_preflight)

    if already:
        print("plan     : target already carries the QSA indexer -> skip graft, "
              "convert it directly", flush=True)
        recipe = OFFICIAL_RECIPE
        donor = None
    else:
        donor = resolve_donor(args.donor)
        if donor is None:
            raise SystemExit(
                "error: this GGUF has no QSA indexer tensors and no donor was found;\n"
                f"       pass --donor <Qwen3.8-27B-QSA>.gguf or set {DONOR_ENV}"
            )
        print(f"plan     : graft 80 indexer tensors from {donor.name} during conversion "
              f"(no intermediate GGUF)", flush=True)
        recipe = f"{GRAFT_RECIPE}:graft_qsa_indexer"
    if "mtp" in components and not mtp:
        p.error(f"--components includes mtp but {args.gguf.name} has no blk.64.nextn.* tensors")

    out = args.out
    report = Path(str(out) + ".conversion.json")
    if not args.check_only:
        for path in (out, report):
            if path.exists():
                if not args.force:
                    p.error(f"{path} exists - rerun with --force to replace it")
                path.unlink()
                print(f"removed  : {path}", flush=True)

    cmd = [sys.executable, "-m", "tools.convert",
           "--model", str(args.base_model),
           "--recipe", recipe,
           "--components", args.components,
           "--source", f"gguf={args.gguf}",
           "--out", str(out),
           "--device", args.device]
    if donor is not None:
        cmd += ["--source", f"indexer={donor}"]
    if args.vision is not None:
        cmd += ["--source", f"vision={args.vision}"]
    if args.dflash2 is not None:
        cmd += ["--source", f"dflash2={args.dflash2}"]
    if args.name is not None:
        cmd += ["--name", args.name]
    if args.proposal:
        cmd += ["--proposal", "--proposal-rows", str(args.proposal_rows)]
        if args.ranking is not None:
            cmd += ["--ranking", str(args.ranking)]
    if args.check_only:
        print("\n" + " ".join(f'"{a}"' if " " in a else a for a in cmd))
        print("\n[check-only] preflight passed; nothing written")
        return 0
    print("== convert ==", flush=True)
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
