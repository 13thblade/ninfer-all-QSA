# Modifications

This repository (`ninfer-all-QSA`) is a fork of
[ninfer-all](https://github.com/iamwavecut/ninfer-all) that adds support for
**Qwen3.8-27B-QSA-GGUF**: a Qwen3.8-27B model grafted with the QSA (Query Sparse
Attention) mechanism from the Qwen4 series, with a KV cache that streams from system
memory into VRAM on demand.

## What this fork adds or changes

- **QSA attention.** A sparse-attention operator and a QSA indexer under
  `src/ops/sparse_attention/` and `src/ops/qsa_indexer/`, plus QSA-aware KV-cache
  append and streaming paths under `src/ops/kv_cache/` (the KV stream that keeps the
  cache in host memory and moves it to VRAM per decode).
- **`--kv-resident` option** (`src/product/kv_resident_options.h` and the planning /
  startup wiring) to control where the KV cache lives — GPU, host, disk, or `auto`
  from the upstream preset.
- **Conversion tooling** for the QSA GGUF under `tools/convert/` (with tests under
  `tests/convert/`).

Untouched files remain exactly as upstream ninfer-all; see
[CONTRIBUTORS.md](CONTRIBUTORS.md) and the git history for upstream attribution.

## License

This fork's additions and modifications are Copyright (c) 2026 13thblade, offered
under the same Apache-2.0 license as upstream. The QSA model weights and GGUF
conversion are by **higa2515**
(https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF); the QSA llama.cpp port is by
**kuraneko1** (https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp).
