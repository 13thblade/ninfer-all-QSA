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
| (this commit) | `docs: add QSA-27B port status and continuation guide` | this document |

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

Remaining work, in dependency order:

1. **Pooling arithmetic** — `src/ops/qsa_indexer/qsa_indexer.cu`: apply
   `key_pre_norm` to each raw key before the 4-token mean in `qsa_pool_kernel`
   and to the keys written by `qsa_tail_kernel`. `QsaIndexerWeights::key_pre_norm`
   is declared but unused. This changes the numeric path; verify with
   `tests/ops/test_qsa_indexer.cpp` against the FP64 oracle plus an A/B against
   the QSA fork's BF16 reference.
2. **KV-head geometry** — `src/ops/sparse_attention/sparse_attention.cu:41-46`
   hardcodes `kQueryHeads=24, kKvHeads=2, kGroup=12`. QSA-27B has 4 KV heads
   (`kGroup=6`), which changes `kThreads`, the shared-memory query tile and the
   merge kernel. Parameterise as a template argument and re-run the operator
   tests on an sm_86 card.
3. **M-RoPE** — the QSA fork rotates the indexer with `ggml_rope_multi`'s 3D
   sections; `qsa_indexer.cu` uses a scalar position + YaRN. Text-only is
   equivalent; image/video needs section-aware positions.
4. **Executor wiring** — in `src/models/qwen3_5/program/` and
   `execution/attention.*`, follow `src/models/qwen4_exp/executor.cpp:3419-3442`:
   `qsa_indexer_append` -> `qsa_indexer_select` -> `kv_stream_resolve` ->
   `sparse_softmax_attention` -> `kv_stream_evict`. Also add `pooled` / `tail`
   state to `decoder_state` and the prefix-cache snapshot, and replay the
   indexer tail on a speculative commit. Dense fallback below
   `2048 + 3 = 2051` positions.
5. **`--kv-resident` on qwen3_5** — `src/runtime/engine/qwen4_exp_core.cpp:305`
   is currently the only place that forwards `kv_resident_positions` to an
   executor. Generalise `src/product/kv_resident_options.h` help text and the
   qwen3_5 core. Note `--tensor-parallel` and streaming are mutually exclusive
   on Flash-Next (executor.cpp:1043-1046).
6. **End-to-end** — convert the downloaded Q4_K_M GGUF, build sm_86, and compare
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
clean. A full CMake build is **not** available here: Windows configure requires a
vcpkg toolchain for FFmpeg/curl (`cmake/Dependencies.cmake:27`) and CMake's
registry probe calls `reg.exe`, which the sandbox blocks.

## How to continue after an interruption

```
cd F:\ninfer-all-QSA
git log --oneline c1d98a6..HEAD          # what has landed
python scripts/verify_qsa_gguf.py <downloaded.gguf>   # real-file preflight
python -m tools.convert --model E:/LLMs/ninfer-base/Qwen3.8-27B \
  --recipe qwen3_8_27b_qsa_gguf --components text,vision,mtp \
  --source gguf=<downloaded.gguf> \
  --source vision=E:/LLMs/mmproj-BF16.gguf \
  --out E:/LLMs/QSA27B-Q4KM.ninfer
```

The matching planning document with a rolling checklist is
`D:\WorkBuddy\2026-11-06-10-13-06\QSA27B_ninfer_移植方案.md`.

## Known numerical caveat

The source GGUF stores all 80 indexer tensors as F32, while this port writes
them as BF16 direct parameters (`tools/convert/qsa_gguf.py:DIRECT_FORMAT`) to
match the existing `qsa_pool_kernel` BF16 norm reads and the Flash-Next path.
If the selection proves sensitive near tied radix scores, switch both the
recipe's `direct_format` and the engine's binding to `fp32` together.
