# QSA-27B port status

Working tree of `iamwavecut/ninfer-all` master @ `8327e99c98b7` (2026-11-05),
imported from a codeload zip because `github.com` is unreachable through the
local proxy. `git log c1d98a6..HEAD` shows every change made here; `c1d98a6` is
the untouched baseline.

The goal is to run `higa2515/Qwen3.8-27B-QSA-GGUF` on this engine, including its
QSA sparse attention and the engine's streamed sparse-attention KV. The QSA
operator set (block indexer, sparse softmax, KV stream) already exists for
`qwen4_exp`; the port connects it to the dense `qwen3_5` (Qwen3.8-27B) backbone
and teaches the converter about the QSA GGUF.

## Commits

| Commit | Subject | State |
|---|---|---|
| `c1d98a6` | baseline import | upstream, zero changes |
| `efae170` | `convert: accept Qwen3.8-27B-QSA GGUF (80 F32 indexer tensors)` | done, 11 offline tests |
| `e1947ba` | `engine: bind the QSA-27B indexer config and weights in qwen3_5` | done, MSVC syntax-checked |
| `71b8aeb` | `ops: declare the QSA-27B key_pre_norm indexer input (interface only)` | done, behaviour unchanged |
| `0e9ca66` | `scripts: preflight a downloaded QSA-27B GGUF` | done |
| `3187448` | `docs: add QSA-27B port status and continuation guide` | first revision of this document |
| `722567d` | `ops: pool QSA-27B indexer keys through the pre-key RMSNorm` | done, op test green |
| `47a9c7f` | `ops: score sparse attention with the QSA-27B 24-over-4 geometry` | done, op test green |
| (this commit) | `docs: record the verified QSA-27B operator stage` | this document |

## What works now

- `tools.convert`'s recipe `qwen3_8_27b_qsa_gguf` accepts a QSA-27B GGUF: it
  validates `qwen35.attention.indexer.{head_count=4,key_length=128,top_k=2048}`
  and `qwen35.attention.compress_ratios` (4 on the 16 full-attention layers, 0
  elsewhere), then reads the 80 F32 `blk.N.indexer.*` tensors, subtracts the
  stored `1 + w` offset from all three norms, and writes
  `text/layers/N/attention/indexer/{query,key,query_norm,key_norm,key_pre_norm}`
  as direct BF16 parameters. The stock `qwen3_8_27b_gguf` recipe is unchanged.
- The qwen3_5 loader accepts the four new optional config fields and binds the
  five indexer `WeightId`s on text full-attention layers (MTP excluded, since the
  QSA fork's MTP block has no trained indexer). Tensor-parallel ranks hold a
  replica, as `qwen4_exp` does.
- `scripts/verify_qsa_gguf.py` checks a real GGUF's header and tensor set without
  reading payloads.

## What does NOT work yet

**The engine still runs dense full attention.** Loading the QSA artifact will
succeed at the parameter layer, but no QSA indexer or sparse attention is
executed, and `--kv-resident` is still Flash-Next-only. Do not benchmark a
converted artifact as if it were QSA.

## Operator work (committed, verified)

Both QSA-27B operator adaptations are committed and their op tests pass:

- `src/ops/qsa_indexer/qsa_indexer.cu` (`722567d`): a `pre_norm_key<PreNorm>`
  helper; `qsa_pool_kernel<Yarn, PreNorm>` applies a zero-centred RMSNorm with
  `key_pre_norm` to every raw key *before* the 4-token mean (matching the fork's
  `qwen35.cpp:864-877`), and `qsa_tail_kernel<PreNorm>` does the same for the
  incomplete block's tail keys. The launcher dispatches over the four (yarn,
  pre_norm) variants; a null `key_pre_norm` keeps the Flash-Next raw-key path
  unchanged.
- `src/ops/sparse_attention/sparse_attention.cu` (`47a9c7f`): the geometry is a
  kernel template parameter `Group` (query heads per KV head): 12 for
  Flash-Next (24 q / 2 kv) and 6 for QSA-27B (24 q / 4 kv). `kThreads =
  Group * 32`, the merge dispatch covers query-head totals 6/12/24, and
  validation accepts group 6 or 12 with 1-24 KV heads. Storage dispatch runs
  over both groups.
- The op tests run in the `ninfer_tests` bundle as always
  (`ninfer_qsa_indexer_test`, `ninfer_sparse_attention_test`); the temporary
  STANDALONE targets used for local iteration are removed.

Remaining work, in dependency order:

1. **M-RoPE** — the QSA fork rotates the indexer with `ggml_rope_multi`'s 3D
   sections; `qsa_indexer.cu` uses a scalar position + YaRN. Text-only is
   equivalent; image/video needs section-aware positions.
2. **Executor wiring** — in `src/models/qwen3_5/program/` and
   `execution/attention.*`, follow `src/models/qwen4_exp/executor.cpp:3419-3442`:
   `qsa_indexer_append` -> `qsa_indexer_select` -> `kv_stream_resolve` ->
   `sparse_softmax_attention` -> `kv_stream_evict`. Also add `pooled` / `tail`
   state to `decoder_state` and the prefix-cache snapshot, and replay the
   indexer tail on a speculative commit. Dense fallback below
   `2048 + 3 = 2051` positions.
3. **`--kv-resident` on qwen3_5** — `src/runtime/engine/qwen4_exp_core.cpp:305`
   is currently the only place that forwards `kv_resident_positions` to an
   executor. Generalise `src/product/kv_resident_options.h` help text and the
   qwen3_5 core. Note `--tensor-parallel` and streaming are mutually exclusive
   on Flash-Next (executor.cpp:1043-1046).
4. **End-to-end** — convert the downloaded Q4_K_M GGUF, build sm_86, and compare
   greedy token sequences / logit KLD against the QSA fork's `llama-server` on
   the same prompt, cold restart, fixed seed, no KV-cache hits.

## Verification performed

```
# converter (offline, synthetic GGUF headers)
PYTHONPATH=F:/ninfer-all-QSA \
  C:/Users/bili8/.workbuddy/binaries/python/envs/ninfer/Scripts/python.exe \
  -m pytest tests/convert/ -q
# -> 301 passed, 1 failed: test_deepseek_v41.py symlink needs Windows
#    SeCreateSymbolicLinkPrivilege (WinError 1314), unrelated to this port.
```

C++ syntax (MSVC 19.44, CUDA 13.4 headers, `cl /Zs /std:c++20 /DNINFER_TIER_AMPERE`):
`src/models/qwen3_5/config.cpp` and `src/models/qwen3_5/load/text.cpp` compile
clean.

A full CMake build **is** now available. The earlier blocker (vcpkg toolchain
for FFmpeg/curl) is resolved by naming the prebuilt triplet tree, per
`cmake/Dependencies.cmake` and `scripts/build-native.bat`:

```bat
:: from a shell with the BuildTools vcvars64.bat environment
E:\dev\tools\cmake-4.4.3-windows-x86_64\bin\cmake -G Ninja -S F:\ninfer-all-QSA ^
  -B F:\ninfer-all-QSA\build-msvc -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 -DNINFER_VCPKG_TREE=E:/deps/vcpkg/installed/x64-windows
cmake --build F:\ninfer-all-QSA\build-msvc --parallel 12
```

Configure succeeded (FFmpeg + CURL found from `E:/deps/vcpkg/installed/x64-windows`).

Operator tests (sm_86, under the BuildTools `vcvars64.bat` environment). The
scoped build of both test targets finished 2026-11-06 13:39 (588/588 edges of
their dependency closure); both tests were verified as standalone executables
before the temporary targets were dropped, and both compile in the
`ninfer_tests` bundle:

```
tests\ninfer_qsa_indexer_standalone_test.exe
# -> PASS qsa_indexer
tests\ninfer_sparse_attention_standalone_test.exe
# -> PASS sparse_softmax_attention
```

The remaining ~602 edges of the full tree are the other tests, benchmarks and
apps; none of them blocks the operator work. Resume the full build with
`cmake --build F:\ninfer-all-QSA\build-msvc --parallel 12`.

## How to continue after an interruption

Order of work (the planning document carries the rolling checklist):

1. **Executor wiring** (`src/models/qwen3_5/`): bind `QsaIndexerParameters`
   in `execution/parameters.*`; plan per-layer `pooled`/`tail` indexer state
   in `state/decoder_state.*`; add the QSA branch in
   `execution/text.cpp::attn_mix` (`qsa_indexer_append` ->
   `qsa_indexer_select` -> `kv_cache_append` -> `sparse_softmax_attention`);
   gate QSA to single-device, `max_concurrency == 1`, non-batched, no
   speculative decoding, and auto-disable the context cache (the indexer
   state is not yet in `StateImage`).
2. **Build, convert, validate.** Full-tree build; convert with
   `python -m tools.convert --model E:/LLMs/ninfer-base/Qwen3.8-27B
   --recipe qwen3_8_27b_qsa_gguf --components text,vision,mtp,dflash2
   --source gguf=E:/LLMs/Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf
   --source vision=E:/LLMs/mmproj-BF16.gguf` (dflash2 source:
   `E:\LLMs\ninfer-dflash2\official`); run the engine with 12 CPU threads on
   the sm_86 card and compare greedy output against the published
   reference. Model card: `https://hf-mirror.com/higa2515/Qwen3.8-27B-QSA-GGUF`.

Already validated against the real file:
`E:\LLMs\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf` (16,674,185,792 bytes) passes
`scripts/verify_qsa_gguf.py` — 946 tensors, 80 F32 indexer, MTP present, 65
blocks, `indexer_budget=2048`, heads=4, key_length=128, compress_ratio=4,
selected_blocks=512, status `accepted`.

The matching planning document with a rolling checklist is
`D:\WorkBuddy\2026-11-06-10-13-06\QSA27B_ninfer_移植方案.md`.

## Known numerical caveat

The source GGUF stores all 80 indexer tensors as F32, while this port writes
them as BF16 direct parameters (`tools/convert/qsa_gguf.py:DIRECT_FORMAT`) to
match the existing `qsa_pool_kernel` BF16 norm reads and the Flash-Next path.
If the selection proves sensitive near tied radix scores, switch both the
recipe's `direct_format` and the engine's binding to `fp32` together.
