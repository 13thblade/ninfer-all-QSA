"""Pack QUASAR Qwen3.8-27B with the QSA indexer, Vision, MTP and proposal head.

    python -m tools.convert.quasar_nvfp4_qsa --model QUASAR_DIR --out MODEL.ninfer

The indexer defaults to the long-16k-1M-heavy asset under tools/convert/assets.
"""

from __future__ import annotations

import os
from pathlib import Path

from tools.convert import qsa_gguf, quasar_nvfp4


def main() -> None:
    parser = quasar_nvfp4.conversion_parser(__doc__, with_dflash2=False)
    parser.set_defaults(name="qwen3.8-27b-quasar-qsa", proposal=True)
    parser.add_argument(
        "--indexer-asset", type=Path,
        help="QSA indexer safetensors (default: long-16k-1M-heavy asset under tools/convert/assets)",
    )
    args = parser.parse_args()
    os.environ["NINFER_GDN_CONTROLS_BF16"] = "off"
    asset = args.indexer_asset or Path(__file__).with_name("assets") / qsa_gguf.INDEXER_ASSET_NAME
    if not asset.is_file():
        parser.error(
            f"missing indexer asset {asset}; supply --indexer-asset or build it "
            "with tools/convert/extract_long_16k_indexer.py"
        )
    quasar_nvfp4.convert(args, indexer_asset=asset)


if __name__ == "__main__":
    main()
