# Modifications

This repository is a fork of NInfer-all that adds support for
**Qwen3.8-27B-QSA**. This file records the changes the fork carries relative to
upstream, as section 4(b) of the Apache License 2.0 requires ("You must cause any
modified files to carry prominent notices stating that You changed the files").
It is a record of what differs, by module; the git history remains the
authoritative record.

## Fork base and attribution

| Item | Value |
| --- | --- |
| Upstream project | NInfer-all, https://github.com/iamwavecut/ninfer-all |
| Upstream copyright | The NInfer-all contributors (see CONTRIBUTORS.md) |
| Upstream commit | `40d913db` |
| Fork repository | https://github.com/13thblade/ninfer-all-QSA |
| License | Apache License 2.0, see LICENSE (unchanged from upstream) |
| Scope | 16 files added, 74 modified; no upstream file deleted |

The model weights and GGUF conversion are by higa2515
(https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF); the QSA llama.cpp port
is by kuraneko1 (https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp).
The fork's additions and modifications are Copyright (c) 2026 13thblade,
offered under the same Apache-2.0 license.

## Changes by module

### `ops` — QSA sparse attention and the block indexer

- `src/ops/qsa_indexer/qsa_indexer.cu` scores the block indexer: keys pooled
  through a pre-key RMSNorm (`key_pre_norm`), interleaved three-axis (t/h/w)
  M-RoPE positions with YaRN and position interpolation, and per-lane indexer
  state so one program can score several sequences.
- `src/ops/sparse_attention/sparse_attention.cu` selects and attends with the
  model's 24-over-4 geometry. The selection budget is a call argument
  (`include/ninfer/ops/qsa_budget.h` holds the shared cap and geometry), and the
  split count, merge bound and partial-softmax workspace are derived from that
  call's budget (`split_count(top_blocks)`), so a narrow window does not pay for
  the widest accepted one.
- `src/ops/launcher/rope.cu` launches the interleaved three-axis rotation the
  indexer and the text attention share under QSA.

### `models/qwen3_5` — the QSA executor

- `--qwen38-27b-qsa` selects the sparse branch of every full-attention layer.
  `execution/text.cpp` (`run_qsa`) splits a call into one part per sequence lane
  — the rows of a decode/verify batch, the segments of a batched prefill, or the
  whole single-sequence call — and each part scores, appends and reads through
  its own lane's indexer state, row-bound KV view and stream pool.
- A prefill that already holds its layer's complete KV routes to the read-only
  dense causal attention (`causal_softmax_attention_cached`) instead of scoring
  and gathering a selection it has already copied; decode and speculative
  verification keep the sparse route, whose pool holds only the resolved blocks.
  `NINFER_QSA_DENSE_PREFILL=0` restores the sparse route for both.
- `--vision` is enabled under QSA: the indexer takes the prompt's 3-D M-RoPE
  positions, and a multimodal prefill runs the dense route.
- The artifact's own MTP drafts drive speculative decoding under QSA; only the
  MTP backend is wired to the indexer.
- `config.*`, `execution/parameters.*`, `load/text.cpp` and `weights.h` bind the
  indexer's configuration and weights, including the 80 F32 `blk.N.indexer.*`
  tensors a QSA GGUF carries.
- `program/planning/startup.cpp` sizes the selected matrix, the indexer and split
  workspaces, and the streamed-KV pool from the same selection budget, reserves
  the dense operator's transient capacity beside the sparse one, and admits up to
  eight concurrent lanes.

### `program/prefix` and `state` — the hybrid prefix cache over streamed KV

- `program/prefix/hybrid_cache.*`, `hybrid_program.cpp` and `hybrid_persist.cpp`
  back the hybrid prefix cache with the streamed host KV store: blocks, tails and
  images are snapshotted and restored from the pinned host copy, and the QSA
  indexer planes travel in the `StateImage` so a resume lands them back.
- `program/prefix/hybrid_host_layout.*` is the host-side layout that store reads
  and writes; `state/decoder_state.h` carries one indexer lane state and one
  `(host, pool, map)` stream state per (lane, layer), sharing the identity table
  and the staging layer.
- `state/state_image.cpp`, `program/storage/session_snapshot.cpp` and
  `program/transactions/commit.cpp` carry the QSA payload through a snapshot and
  re-pool an accepted prefix per lane on a speculative commit.

### `runtime` — capacity, limits and the prefix index

- `runtime/prefix_cache/prefix_index.*` indexes the streamed blocks: a logical
  page maps to a host-resident flat capacity stride, and the device layout is a
  zero-stride view of the same pages.
- `runtime/engine/kv_capacity.cpp` and `model_instance.cpp` resolve
  `--kv-capacity auto` against the streamed pool. Under QSA the checkpoint
  catalog (`--use-original-prefix-caching`) keeps exactly the one conversation its
  indexer scratch still owns — one Device slot, one private continuation, no Host
  or disk tier, no anchors, no shared prefixes — and names the flags to lower when
  a request asks for more. Without `--qwen38-27b-qsa` this normalization does not
  run and the catalog behaves exactly as upstream.

### `ops/kv_stream` — streamed KV

- `--kv-resident N|all|auto` (`src/product/kv_resident_options.h`) sets how many
  positions of each sparse-attention layer's KV stay in device memory; past them
  the layers stream from pinned host memory. `auto` follows the upstream preset
  (all with device experts, 32,768 with host or disk experts).
- `src/ops/kv_stream/kv_stream.cu` and `include/ninfer/ops/kv_stream.h` move the
  authoritative KV to pinned host memory behind one shared staging layer, pin the
  device pool to the logical page count, and size the pool floor from the
  attention window and the verification width
  (`(window blocks + 1) × width`), so a verify's disjoint selections cannot
  overflow it.

### `serve`, `apps/cli` — the flag surface

- `--qwen38-27b-qsa` is accepted by `ninfer-serve`, `ninfer` and
  `ninfer-perplexity`, and `--qsa-top-blocks N` by the same three, through the
  shared `src/product/qsa_top_blocks_options.h`. The count is in four-token
  blocks, 1 to 65,536; unset or `0` keeps the artifact's own 512 blocks, and the
  value is fixed at startup so a captured graph replays the grid it captured.
- `src/serve/serve_options.cpp` refuses `--qwen38-27b-qsa` together with
  `--use-original-prefix-caching`: the sparse branch runs on the hybrid prefix
  cache, and the checkpoint catalog has no candidate that could rebuild a QSA
  session. The refusal is an `OptionConflict`, so a layered configuration resolves
  it by layer precedence and one command line fails fast.
- `src/serve/operational_log.cpp` logs the effective history window with the
  source it came from.

### `tools/convert` — the converter and the weight graft

- `qsa_gguf.py` implements the `qwen3_8_27b_qsa_gguf` recipe registered in
  `gguf_blocks.py`; it teaches the converter the 80 extra F32 indexer tensors, and
  `convert_qsa27b.py` is the one-shot wrapper that writes a single `.ninfer`
  artifact.
- `qsa_graft_recipe.py` grafts the indexer tensors and the native BF16 GDN
  controls onto an artifact converted from another quantization; the donor is
  optional, and `extract_indexer_tensors.py` / `extract_gdn_controls.py` write the
  manifests under `tools/convert/assets/` that make a graft reproducible.
- `scripts/verify_qsa_gguf.py` is the offline preflight for a downloaded QSA GGUF.

### Tests

- Operator: `tests/ops/test_qsa_indexer.cpp` covers the 24-over-4 select, the
  three-axis M-RoPE with YaRN and interpolation, and the selection budget;
  `test_sparse_attention.cpp` and `test_kv_stream.cpp` cover the attention and
  streaming paths.
- Cache and layout: `tests/test_prefix_cache_index.cpp` covers the block index
  over the streamed store, `tests/models/qwen3_5/test_hybrid_stream_layout.cpp`
  the host/device layout, and `test_state_image_layout.cpp` the two-lane state
  image.
- Options: `tests/test_cli_options.cpp` and `tests/test_serve_options.cpp` cover
  the three flags and the hybrid-only constraint; `test_kv_capacity.cpp` and
  `test_context_cache_defaults.cpp` cover the capacity policy they select.
- Converter: `tests/convert/test_qwen3_8_27b_qsa_gguf.py` and
  `test_gguf_blocks.py`.

### Benchmarks and diagnostics

- `tools/bench/pcie_bandwidth_monitor.py` samples
  `nvmlDeviceGetPcieThroughput` through `ctypes` from a background thread and
  integrates the readings over request windows; the engine carries no PCIe
  counters of its own.
- `tools/bench/run_qsa27b_kvstream_bench.py` (`perf32k`, `pcie512k`) and
  `run_qsa27b_32k_ab.py` drive a server through those windows.

### Documentation

- `README.md` states the fork's attribution, the three flags, the streamed-KV
  floor and the QSA prefill/decode measurements.
- `docs/cli.md` and `docs/serving.md` list `--qwen38-27b-qsa`,
  `--qsa-top-blocks` and the Qwen3.8-27B reading of `--kv-resident`;
  `docs/maintainer/qwen4-exp-model.md` records where the sparse branch sits in the
  Qwen4 experimental model.

## Known limits

- QSA runs on one device without tensor parallelism, with the hybrid prefix
  cache, and with CUDA graphs off above one concurrent request (a per-row lane
  indirection inside the captured kernels is the follow-up).
- Under QSA the checkpoint catalog is restricted to one conversation's own next
  turn; the hybrid cache is the supported cache for the flag.

## Files

### Added (16)

Root (1)

- `MODIFICATIONS.md`

include/ninfer/ops (1)

- `include/ninfer/ops/qsa_budget.h`

scripts (1)

- `scripts/verify_qsa_gguf.py`

src/product (1)

- `src/product/qsa_top_blocks_options.h`

tests/convert (1)

- `tests/convert/test_qwen3_8_27b_qsa_gguf.py`

tests/models/qwen3_5 (1)

- `tests/models/qwen3_5/test_hybrid_stream_layout.cpp`

tools/bench (3)

- `tools/bench/pcie_bandwidth_monitor.py`
- `tools/bench/run_qsa27b_32k_ab.py`
- `tools/bench/run_qsa27b_kvstream_bench.py`

tools/convert/assets (2)

- `tools/convert/assets/gdn_controls_fp8_bf16.manifest.json`
- `tools/convert/assets/indexer_tensors_fp32.manifest.json`

tools/convert (5)

- `tools/convert/convert_qsa27b.py`
- `tools/convert/extract_gdn_controls.py`
- `tools/convert/extract_indexer_tensors.py`
- `tools/convert/qsa_gguf.py`
- `tools/convert/qsa_graft_recipe.py`

### Modified (74)

Root (1)

- `README.md`

apps/cli (3)

- `apps/cli/main.cpp`
- `apps/cli/options.cpp`
- `apps/cli/options.h`

apps/perplexity (1)

- `apps/perplexity/main.cpp`

docs (2)

- `docs/cli.md`
- `docs/serving.md`

docs/maintainer (1)

- `docs/maintainer/qwen4-exp-model.md`

include/ninfer (1)

- `include/ninfer/engine_options.h`

include/ninfer/observability (1)

- `include/ninfer/observability/resources.h`

include/ninfer/ops (4)

- `include/ninfer/ops/kv_stream.h`
- `include/ninfer/ops/qsa_indexer.h`
- `include/ninfer/ops/rope.h`
- `include/ninfer/ops/sparse_attention.h`

src/models/qwen3_5 (3)

- `src/models/qwen3_5/config.cpp`
- `src/models/qwen3_5/config.h`
- `src/models/qwen3_5/weights.h`

src/models/qwen3_5/execution (4)

- `src/models/qwen3_5/execution/parameters.cpp`
- `src/models/qwen3_5/execution/parameters.h`
- `src/models/qwen3_5/execution/text.cpp`
- `src/models/qwen3_5/execution/text.h`

src/models/qwen3_5/load (1)

- `src/models/qwen3_5/load/text.cpp`

src/models/qwen3_5/program (8)

- `src/models/qwen3_5/program/context.h`
- `src/models/qwen3_5/program/decode.cpp`
- `src/models/qwen3_5/program/graphs.cpp`
- `src/models/qwen3_5/program/prefill.cpp`
- `src/models/qwen3_5/program/program.cpp`
- `src/models/qwen3_5/program/program.h`
- `src/models/qwen3_5/program/program_impl.cpp`
- `src/models/qwen3_5/program/program_impl.h`

src/models/qwen3_5/program/planning (3)

- `src/models/qwen3_5/program/planning/request_plan.cpp`
- `src/models/qwen3_5/program/planning/startup.cpp`
- `src/models/qwen3_5/program/planning/startup.h`

src/models/qwen3_5/program/prefix (6)

- `src/models/qwen3_5/program/prefix/hybrid_cache.cpp`
- `src/models/qwen3_5/program/prefix/hybrid_cache.h`
- `src/models/qwen3_5/program/prefix/hybrid_host_layout.cpp`
- `src/models/qwen3_5/program/prefix/hybrid_host_layout.h`
- `src/models/qwen3_5/program/prefix/hybrid_persist.cpp`
- `src/models/qwen3_5/program/prefix/hybrid_program.cpp`

src/models/qwen3_5/program/speculative (1)

- `src/models/qwen3_5/program/speculative/mtp.cpp`

src/models/qwen3_5/program/storage (1)

- `src/models/qwen3_5/program/storage/session_snapshot.cpp`

src/models/qwen3_5/program/transactions (1)

- `src/models/qwen3_5/program/transactions/commit.cpp`

src/models/qwen3_5/state (3)

- `src/models/qwen3_5/state/decoder_state.h`
- `src/models/qwen3_5/state/state_image.cpp`
- `src/models/qwen3_5/state/state_image.h`

src/models/qwen4_exp (1)

- `src/models/qwen4_exp/executor.cpp`

src/ops/kv_stream (1)

- `src/ops/kv_stream/kv_stream.cu`

src/ops/launcher (1)

- `src/ops/launcher/rope.cu`

src/ops/qsa_indexer (1)

- `src/ops/qsa_indexer/qsa_indexer.cu`

src/ops/sparse_attention (1)

- `src/ops/sparse_attention/sparse_attention.cu`

src/product (1)

- `src/product/kv_resident_options.h`

src/runtime/contract (1)

- `src/runtime/contract/resources.h`

src/runtime/engine (2)

- `src/runtime/engine/kv_capacity.cpp`
- `src/runtime/engine/model_instance.cpp`

src/runtime/prefix_cache (2)

- `src/runtime/prefix_cache/prefix_index.cpp`
- `src/runtime/prefix_cache/prefix_index.h`

src/serve (5)

- `src/serve/generation_service.cpp`
- `src/serve/operational_log.cpp`
- `src/serve/request_log.cpp`
- `src/serve/serve_options.cpp`
- `src/serve/serve_options.h`

tests/convert (1)

- `tests/convert/test_gguf_blocks.py`

tests/models/qwen3_5 (2)

- `tests/models/qwen3_5/test_state_image_layout.cpp`
- `tests/models/qwen3_5/tests.cmake`

tests/ops (3)

- `tests/ops/test_kv_stream.cpp`
- `tests/ops/test_qsa_indexer.cpp`
- `tests/ops/test_sparse_attention.cpp`

tests (5)

- `tests/test_cli_options.cpp`
- `tests/test_context_cache_defaults.cpp`
- `tests/test_kv_capacity.cpp`
- `tests/test_prefix_cache_index.cpp`
- `tests/test_serve_options.cpp`

tools/convert (2)

- `tools/convert/__main__.py`
- `tools/convert/gguf_blocks.py`
