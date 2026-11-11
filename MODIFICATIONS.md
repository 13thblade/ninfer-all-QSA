# Modifications

This repository is a fork of NInfer-all that adds support for
**Qwen3.8-27B-QSA-GGUF**. This file records the changes relative to the fork
base, as required by section 4(b) of the Apache License 2.0 ("You must cause
any modified files to carry prominent notices stating that You changed the
files"). It is an index of the changes; the fork's git history remains the
authoritative record.

## Fork base

| Item | Value |
| --- | --- |
| Upstream project | NInfer-all, https://github.com/iamwavecut/ninfer-all |
| Upstream copyright | The NInfer-all contributors (see CONTRIBUTORS.md) |
| Fork baseline | `c1d98a69` imports upstream master at `8327e99c98b7` (2026-11-05) |
| Upstream sync | two merges: `cc415771` took `64bd1115` (137 commits), `ea30b646` took `2207ee04` |
| Fork repository | https://github.com/13thblade/ninfer-all-QSA |
| License | Apache License 2.0, see LICENSE (unchanged from upstream) |

The model weights and GGUF conversion are by higa2515
(https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF); the QSA llama.cpp port
is by kuraneko1 (https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp).
The fork's additions and modifications are Copyright (c) 2026 13thblade,
offered under the same Apache-2.0 license.

## Fork commits

| Commit | Date | Subject |
| --- | --- | --- |
| `c1d98a69` | 2026-11-06 | baseline: import ninfer-all master @ 8327e99c98b7 (2026-11-05) |
| `efae170c` | 2026-11-06 | convert: accept Qwen3.8-27B-QSA GGUF (80 F32 indexer tensors) |
| `e1947bac` | 2026-11-06 | engine: bind the QSA-27B indexer config and weights in qwen3_5 |
| `71b8aeb2` | 2026-11-06 | ops: declare the QSA-27B key_pre_norm indexer input (interface only) |
| `0e9ca66f` | 2026-11-06 | scripts: preflight a downloaded QSA-27B GGUF |
| `31874483` | 2026-11-06 | docs: add QSA-27B port status and continuation guide |
| `722567dd` | 2026-11-06 | ops: pool QSA-27B indexer keys through the pre-key RMSNorm |
| `47a9c7f1` | 2026-11-06 | ops: score sparse attention with the QSA-27B 24-over-4 geometry |
| `c99329f1` | 2026-11-06 | docs: record the verified QSA-27B operator stage |
| `cf3053de` | 2026-11-06 | convert: register the late-constructed QSA indexer parameters on the recipe |
| `75d30b9a` | 2026-11-06 | engine: run the QSA sparse branch of qwen3_5 behind --qwen38-27b-qsa |
| `08e539c5` | 2026-11-06 | docs: record the landed QSA executor and converter commits |
| `3127a610` | 2026-11-06 | engine: run QSA-27B with the artifact's own MTP speculative drafts |
| `8d103dae` | 2026-11-06 | docs: record QSA-27B MTP support, real-model verification and the 32K A/B |
| `eadd2282` | 2026-11-06 | docs: plan QSA-27B Vision (IMROPE) and streamed KV compatibility |
| `567f2cab` | 2026-11-06 | ops: add interleaved 3-axis M-RoPE to the QSA indexer (V1) |
| `d49f7f4c` | 2026-11-06 | tests: cover QSA indexer 3-axis M-RoPE, YaRN and interpolation (V2) |
| `d78fe1e2` | 2026-11-06 | engine: wire QSA-27B indexer to interleaved M-RoPE positions (V3) |
| `7e9906d8` | 2026-11-06 | engine: enable --vision under QSA-27B (V4) |
| `213085f1` | 2026-11-06 | docs: mark the V4 commit in the roadmap |
| `d7fdefd2` | 2026-11-06 | engine: plan and allocate QSA-27B streamed KV (K1) |
| `a07fc300` | 2026-11-06 | engine: run QSA-27B attention through the KV stream (K2) |
| `0fefb8af` | 2026-11-06 | engine: complete QSA-27B streamed KV (K3/K4) |
| `b04c5af2` | 2026-11-06 | docs: mark all K-stage commits in the roadmap |
| `8e1dc055` | 2026-11-11 | engine: stream QSA KV through one shared staging layer (L1-L3) |
| `d7218f70` | 2026-11-11 | tools: replace in-engine PCIe counters with an external NVML monitor |
| `dfc501c5` | 2026-11-11 | tools: add QSA-27B one-shot converter and the external PCIe benchmark driver |
| `b38838a2` | 2026-11-11 | engine: resolve QSA --kv-resident "auto" from the upstream preset |
| `99faf43e` | 2026-11-11 | docs: attribute the QSA fork in README and add MODIFICATIONS.md |

## Change summary

Relative to upstream master (`2207ee04`) this fork carries **38 modified
upstream files and 8 new files**, and deletes no upstream file.
`docs/qsa-port.md` was a fork-only document, removed by `d7218f70`, so it does
not appear in the lists below.

Because the fork was cut from an older upstream snapshot and later took two
upstream merges, `git diff --name-status upstream/master..HEAD` reports 192
modified and 8 added files. Most of that modified set (`.github/scripts/*`,
`scripts/pods/*`, `docs/*`, and other support files) is the version skew between
the import snapshot and current upstream, not work done by this fork. The lists
below contain only files that a fork commit after the baseline import actually
changed and that remain in the repository.

**QSA sparse attention.** The QSA mechanism, originally from the Qwen4 series,
is scored with the model's 24-over-4 geometry in
`src/ops/sparse_attention/sparse_attention.cu`. The block indexer in
`src/ops/qsa_indexer/qsa_indexer.cu` pools keys through a pre-key RMSNorm and
takes interleaved three-axis M-RoPE positions, with YaRN and position
interpolation covered. `src/ops/launcher/rope.cu` and the operator headers carry
the matching interfaces; `tests/ops/test_qsa_indexer.cpp` and
`tests/ops/test_sparse_attention.cpp` are the operator tests.

**QSA executor.** `--qwen38-27b-qsa` (`src/serve/serve_options.cpp`,
`apps/cli/options.cpp`) selects the QSA sparse branch of every full-attention
layer in `qwen3_5`, using the artifact's block indexer and one concurrent
request. The same path enables Vision and the model's own MTP speculative
drafts; only the MTP backend is wired to the QSA indexer. The executor binds
the QSA indexer config and weights, and the streamed-KV state lives in
`src/models/qwen3_5/state/decoder_state.h` and the program/transaction files.

**Streamed KV.** QSA needs only 2,048 historical tokens per token, so the KV
cache is kept in pinned host memory and streamed into VRAM on demand through
one shared staging layer. `--kv-resident N|all|auto`
(`src/product/kv_resident_options.h`) sets how many positions of each sparse
attention layer's KV stay on the GPU (`all` keeps the whole cache resident,
`auto` follows the upstream preset: all for device experts, 32768 for host/disk
experts). No change is made to the inference hot path for this option.

**Converter.** `tools/convert/` carries the `qwen3_8_27b_qsa_gguf` recipe,
registered in `tools/convert/gguf_blocks.py` and implemented in
`tools/convert/qsa_gguf.py`, which teaches the converter the 80 extra F32
`blk.N.indexer.*` tensors a QSA GGUF holds. `tools/convert/convert_qsa27b.py`
is a one-shot wrapper producing a single `.ninfer` artifact,
`scripts/verify_qsa_gguf.py` is the offline preflight, and
`tests/convert/test_qwen3_8_27b_qsa_gguf.py` is the converter test.

**External PCIe monitoring.** The per-request counters that instrumented
`kv_stream` inside the engine were removed; they saw only the blocks one
operator copied and forced a per-layer stream synchronisation at each request
boundary. `tools/bench/pcie_bandwidth_monitor.py` samples NVML's
`nvmlDeviceGetPcieThroughput` through `ctypes` (no pynvml dependency) from a
background thread, writes a CSV of TX/RX readings, and integrates them over
request windows to report peak/mean bandwidth and link utilisation from the
reported PCIe generation and width. `tools/bench/run_qsa27b_kvstream_bench.py`
(`perf32k` and `pcie512k` phases) and `tools/bench/run_qsa27b_32k_ab.py` are
the benchmark drivers that start a server and use those windows.

**Documentation.** `README.md` keeps the Chinese description, adds an English
counterpart, and states the fork's attribution in both languages. This file is
the record of the fork's changes; untouched files remain exactly as upstream,
credited in `CONTRIBUTORS.md` and the git history.

## Modified files (38)

### Root (1)

- `README.md`

### apps/cli (3)

- `apps/cli/main.cpp`
- `apps/cli/options.cpp`
- `apps/cli/options.h`

### apps/perplexity (1)

- `apps/perplexity/main.cpp`

### include/ninfer (4)

- `include/ninfer/engine_options.h`
- `include/ninfer/ops/qsa_indexer.h`
- `include/ninfer/ops/rope.h`
- `include/ninfer/ops/sparse_attention.h`

### src/models/qwen3_5 (19)

- `src/models/qwen3_5/config.cpp`
- `src/models/qwen3_5/config.h`
- `src/models/qwen3_5/execution/parameters.cpp`
- `src/models/qwen3_5/execution/parameters.h`
- `src/models/qwen3_5/execution/text.cpp`
- `src/models/qwen3_5/execution/text.h`
- `src/models/qwen3_5/load/text.cpp`
- `src/models/qwen3_5/program/context.h`
- `src/models/qwen3_5/program/decode.cpp`
- `src/models/qwen3_5/program/graphs.cpp`
- `src/models/qwen3_5/program/planning/startup.cpp`
- `src/models/qwen3_5/program/planning/startup.h`
- `src/models/qwen3_5/program/prefill.cpp`
- `src/models/qwen3_5/program/program_impl.cpp`
- `src/models/qwen3_5/program/program_impl.h`
- `src/models/qwen3_5/program/speculative/mtp.cpp`
- `src/models/qwen3_5/program/transactions/commit.cpp`
- `src/models/qwen3_5/state/decoder_state.h`
- `src/models/qwen3_5/weights.h`

### src/ops (3)

- `src/ops/launcher/rope.cu`
- `src/ops/qsa_indexer/qsa_indexer.cu`
- `src/ops/sparse_attention/sparse_attention.cu`

### src/product (1)

- `src/product/kv_resident_options.h`

### src/serve (3)

- `src/serve/generation_service.cpp`
- `src/serve/serve_options.cpp`
- `src/serve/serve_options.h`

### tests/ops (2)

- `tests/ops/test_qsa_indexer.cpp`
- `tests/ops/test_sparse_attention.cpp`

### tools/convert (1)

- `tools/convert/gguf_blocks.py`

## Added files (8)

### Root (1)

- `MODIFICATIONS.md`

### scripts (1)

- `scripts/verify_qsa_gguf.py`

### tests/convert (1)

- `tests/convert/test_qwen3_8_27b_qsa_gguf.py`

### tools/bench (3)

- `tools/bench/pcie_bandwidth_monitor.py`
- `tools/bench/run_qsa27b_32k_ab.py`
- `tools/bench/run_qsa27b_kvstream_bench.py`

### tools/convert (2)

- `tools/convert/convert_qsa27b.py`
- `tools/convert/qsa_gguf.py`
