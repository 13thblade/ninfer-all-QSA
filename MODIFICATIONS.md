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
| Fork baseline | `052d0380` imports upstream master at `8327e99c98b7` (2026-11-05) |
| Upstream sync | two merges: `01bcc2ef` took `d406f46d` (137 commits), `7afc29e3` took `2207ee04` |
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
| `052d0380` | 2026-11-06 | baseline: import ninfer-all master @ 8327e99c98b7 (2026-11-05) |
| `b9912805` | 2026-11-06 | convert: accept Qwen3.8-27B-QSA GGUF (80 F32 indexer tensors) |
| `251ea4bc` | 2026-11-06 | engine: bind the QSA-27B indexer config and weights in qwen3_5 |
| `a725a22b` | 2026-11-06 | ops: declare the QSA-27B key_pre_norm indexer input (interface only) |
| `11a092d3` | 2026-11-06 | scripts: preflight a downloaded QSA-27B GGUF |
| `4651f03f` | 2026-11-06 | docs: add QSA-27B port status and continuation guide |
| `474dd4cd` | 2026-11-06 | ops: pool QSA-27B indexer keys through the pre-key RMSNorm |
| `62036944` | 2026-11-06 | ops: score sparse attention with the QSA-27B 24-over-4 geometry |
| `b74c94b3` | 2026-11-06 | docs: record the verified QSA-27B operator stage |
| `7e287d78` | 2026-11-06 | convert: register the late-constructed QSA indexer parameters on the recipe |
| `e3fc8bb0` | 2026-11-06 | engine: run the QSA sparse branch of qwen3_5 behind --qwen38-27b-qsa |
| `a9b95b1a` | 2026-11-06 | docs: record the landed QSA executor and converter commits |
| `3bc6c49d` | 2026-11-06 | engine: run QSA-27B with the artifact's own MTP speculative drafts |
| `d2cb0fa6` | 2026-11-06 | docs: record QSA-27B MTP support, real-model verification and the 32K A/B |
| `00ec0c4a` | 2026-11-06 | docs: plan QSA-27B Vision (IMROPE) and streamed KV compatibility |
| `a4894964` | 2026-11-06 | ops: add interleaved 3-axis M-RoPE to the QSA indexer (V1) |
| `bd329e04` | 2026-11-06 | tests: cover QSA indexer 3-axis M-RoPE, YaRN and interpolation (V2) |
| `b4944a0e` | 2026-11-06 | engine: wire QSA-27B indexer to interleaved M-RoPE positions (V3) |
| `3918197c` | 2026-11-06 | engine: enable --vision under QSA-27B (V4) |
| `672c1e71` | 2026-11-06 | docs: mark the V4 commit in the roadmap |
| `b5af538f` | 2026-11-06 | engine: plan and allocate QSA-27B streamed KV (K1) |
| `423ce2be` | 2026-11-06 | engine: run QSA-27B attention through the KV stream (K2) |
| `36c636d4` | 2026-11-06 | engine: complete QSA-27B streamed KV (K3/K4) |
| `f0b35872` | 2026-11-06 | docs: mark all K-stage commits in the roadmap |
| `ae212f10` | 2026-11-11 | engine: stream QSA KV through one shared staging layer (L1-L3) |
| `3fb7f647` | 2026-11-11 | tools: replace in-engine PCIe counters with an external NVML monitor |
| `a5865c12` | 2026-11-11 | tools: add QSA-27B one-shot converter and the external PCIe benchmark driver |
| `878c9712` | 2026-11-11 | engine: resolve QSA --kv-resident "auto" from the upstream preset |
| `ca748d86` | 2026-11-11 | docs: attribute the QSA fork in README and add MODIFICATIONS.md |
| `67b8d0bd` | 2026-11-11 | docs: plan the QSA history window and record the current fork |
| `174c0a18` | 2026-11-11 | ops: take the QSA selection budget at run time |
| `becf3ce4` | 2026-11-11 | engine: apply --qsa-history-tokens to the QSA selection budget |
| `82fb2a9d` | 2026-11-11 | flags: expose --qsa-history-tokens in all three entry points |
| `ce2d266e` | 2026-11-11 | docs: document the QSA history window and mark the plan done |
| `1da2b6f6` | 2026-11-11 | engine: raise the QSA history cap to 32K tokens |
| `960990eb` | 2026-11-11 | serve: log the QSA history window and where it came from |
| `93564799` | 2026-11-11 | ops: size the QSA split grid from the call's budget, not the cap |
| `17105e33` | 2026-11-11 | flags: count the QSA history window in blocks |
| `2d4a1ffb` | 2026-11-11 | docs: record the QSA history window and tabulate its measurements |
| `190507b0` | 2026-11-11 | docs: keep the planning document out of the repository |

## Change summary

Relative to upstream master (`2207ee04`) this fork carries **48 modified
upstream files and 10 new files**, and deletes no upstream file.
`docs/qsa-port.md` was a fork-only document, removed by `3fb7f647`, so it does
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

**Configurable history window.** `--qsa-top-blocks N` (CLI, `ninfer-serve` and
`ninfer-perplexity`, parsed by the shared `src/product/qsa_top_blocks_options.h`)
sets how many four-token history blocks each QSA query may look back over, 1 to
65,536; unset or `0` keeps the artifact's own 512 blocks. The value is fixed at
startup, so a captured graph replays the grid it captured.
`include/ninfer/ops/qsa_budget.h` carries the shared geometry; the indexer
select, `sparse_softmax_attention` and the `kv_stream` family take the budget as
their last argument, and `src/ops/sparse_attention/sparse_attention.cu` derives
its split count, merge bound and partial-softmax workspace from that call's
budget (`split_count(top_blocks)`), so a narrow budget no longer pays for the
largest accepted one. The planner (`program/planning/startup.cpp`,
`qwen4_exp/executor.cpp`) sizes the selected matrix, the indexer and split
workspaces and the KV-stream pool from the same number, and
`src/serve/operational_log.cpp` logs the effective window with its source. The
measurements this knob was tuned against are in `README.md`.

**Documentation.** `README.md` keeps the Chinese description, adds an English
counterpart, and states the fork's attribution in both languages. This file is
the record of the fork's changes; untouched files remain exactly as upstream,
credited in `CONTRIBUTORS.md` and the git history.

## Modified files (48)

### Root (1)

- `README.md`

### apps/cli (3)

- `apps/cli/main.cpp`
- `apps/cli/options.cpp`
- `apps/cli/options.h`

### apps/perplexity (1)

- `apps/perplexity/main.cpp`

### include/ninfer (5)

- `include/ninfer/engine_options.h`
- `include/ninfer/ops/kv_stream.h`
- `include/ninfer/ops/qsa_indexer.h`
- `include/ninfer/ops/rope.h`
- `include/ninfer/ops/sparse_attention.h`

### src/models/qwen3_5 (21)

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
- `src/models/qwen3_5/program/program.cpp`
- `src/models/qwen3_5/program/program.h`
- `src/models/qwen3_5/program/program_impl.cpp`
- `src/models/qwen3_5/program/program_impl.h`
- `src/models/qwen3_5/program/speculative/mtp.cpp`
- `src/models/qwen3_5/program/transactions/commit.cpp`
- `src/models/qwen3_5/state/decoder_state.h`
- `src/models/qwen3_5/weights.h`

### src/models/qwen4_exp (1)

- `src/models/qwen4_exp/executor.cpp`

### src/ops (4)

- `src/ops/kv_stream/kv_stream.cu`
- `src/ops/launcher/rope.cu`
- `src/ops/qsa_indexer/qsa_indexer.cu`
- `src/ops/sparse_attention/sparse_attention.cu`

### src/product (1)

- `src/product/kv_resident_options.h`

### src/runtime/engine (1)

- `src/runtime/engine/model_instance.cpp`

### src/serve (4)

- `src/serve/generation_service.cpp`
- `src/serve/operational_log.cpp`
- `src/serve/serve_options.cpp`
- `src/serve/serve_options.h`

### tests (2)

- `tests/test_cli_options.cpp`
- `tests/test_serve_options.cpp`

### tests/ops (3)

- `tests/ops/test_kv_stream.cpp`
- `tests/ops/test_qsa_indexer.cpp`
- `tests/ops/test_sparse_attention.cpp`

### tools/convert (1)

- `tools/convert/gguf_blocks.py`

## Added files (10)

### Root (1)

- `MODIFICATIONS.md`

### include/ninfer/ops (1)

- `include/ninfer/ops/qsa_budget.h`

### src/product (1)

- `src/product/qsa_top_blocks_options.h`

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
