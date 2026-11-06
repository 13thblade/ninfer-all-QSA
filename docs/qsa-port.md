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
| `cf3053d` | `convert: register the late-constructed QSA indexer parameters on the recipe` | done, 301 converter tests |
| `75d30b9` | `engine: run the QSA sparse branch of qwen3_5 behind --qwen38-27b-qsa` (+ docs) | done: full build, op tests, conversion, offline artifact inspection |

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

**The engine runs the QSA sparse branch, behind a flag.**
`--qwen38-27b-qsa` (CLI `ninfer`, `ninfer-perplexity`, `ninfer-serve`) sets
`EngineOptions::qwen38_27b_qsa`. A QSA artifact without the flag keeps its
native dense attention exactly as before; a QSA artifact with the flag runs,
on every one of its 16 full-attention layers,

```
indexer query/key linears of the normalized residual (Hadamard basis handled)
-> explicit kv_cache_append of the call's K/V rows
-> qsa_indexer_append (pool the completed 4-token blocks into the layer's
   FP32 pooled plane, keep the incomplete block's tail)
-> qsa_indexer_select (512 best blocks)
-> sparse_softmax_attention over the selected blocks + the call's own
   incomplete block
-> the dense branch's in-Op gate is replaced by the same sigmoid
   multiply (or Hadamard form) after the sparse call
```

The gates at startup (`validate_target_options`) require, with the flag: the
artifact's `config().text.qsa`, one device, no tensor parallelism,
`max_concurrency == 1`, no speculative backend or draft/ngram draft tokens;
the context cache is auto-disabled. The indexer state is one rank-0 FP32 copy
per full-attention layer (`pooled [128, capacity/4 + 1]` + `tail [128, 3]`
in one plane each); a new prefill repools from block zero, so no
initialization is needed while the context cache is off. Decode reads the
sequence through its row-bound execution view (`text_kv_view`); the CUDA
Graph decode is captured with the capture row bound, so replay reads the
row's published block table. The branch admits a prefill chunk, an ordinary
one-column decode, or one whole-pass segment of a batched prefill; anything
else throws.

## What does NOT work yet

- **Prefix-cache / continuation reuse under QSA** needs the pooled+tail
  planes in `StateImage`; the context cache is disabled instead.
- **Speculative decoding (MTP/DFlash2) under QSA** needs the indexer tail to
  replay on verify/commit; the gates reject speculative decoding instead.
- **M-RoPE (image/video) indexer positions**: the QSA fork rotates the
  indexer with 3D sections; this port uses the scalar position + YaRN,
  equivalent for text only.
- **`--kv-resident`** remains Flash-Next-only; the QSA path reads the
  sequence's full paged KV on the device (no host streaming).

Do not benchmark a converted artifact as if it were QSA unless
`--qwen38-27b-qsa` is set.

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
2. **Executor wiring** — done (this revision): `QsaIndexerParameters` bound in
   `execution/parameters.*`; per-layer `pooled`/`tail` planes in
   `state/decoder_state.*` and the startup plan; the QSA branch of
   `execution/text.cpp::attn_mix` (`run_qsa`) with the explicit
   `kv_cache_append`; startup gates, context-cache auto-disable, CLI/serve
   flag plumbing. The Flash-Next `kv_stream` (host-resident KV) is not used:
   QSA-27B reads its sequence's paged KV on the device.
3. **Prefix-cache state and speculative replay** — carry `pooled`/`tail` in
   `StateImage` and replay the indexer tail on a speculative commit; both are
   gated off for now (context cache disabled, speculative rejected).
4. **`--kv-resident` on qwen3_5** — `src/runtime/engine/qwen4_exp_core.cpp:305`
   is currently the only place that forwards `kv_resident_positions` to an
   executor. Generalise `src/product/kv_resident_options.h` help text and the
   qwen3_5 core.
5. **End-to-end** — convert the downloaded Q4_K_M GGUF, build sm_86, and compare
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

C++ (MSVC 14.44.35207, VS 2022 BuildTools, CUDA 13.4 headers, Ninja, Release,
arch 86): the **full tree** builds clean in `F:\ninfer-all-QSA\build-msvc`
(455/455 edges, then 36/36 incremental after the indexer basis fix); every app
and the `ninfer_tests` bundle link. Configure command as above
(`-DNINFER_VCPKG_TREE=E:/deps/vcpkg/installed/x64-windows`); build from a shell
with the BuildTools `vcvars64.bat` environment:

```bat
cmake --build F:\ninfer-all-QSA\build-msvc --parallel 12
```

Operator tests (sm_86, in the `ninfer_tests` bundle). This box's CUDA 13.4
install is tools-only (nvcc/ptxas/headers, no runtime DLLs), so the CUDA
runtime closure (cuBLAS/cuBLASLt/cuDART) resolves from the newest staged copy
on the machine — `F:\ninfer-all-workspace\ninfer-all-nvfp4\build-merge\apps`
(`cublas64_13.dll` 6.14.11.1370, `cudart64_13.dll` 6.14.11.13040; CUDA 13.0's
6.14.11.1300/13000 in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64`
also satisfies the same CUDA-13 ABI) — plus `zlib.dll` from that same
directory. Run:

```
tests\ninfer_tests.exe ninfer_qsa_indexer_test        # -> PASS qsa_indexer
tests\ninfer_tests.exe ninfer_sparse_attention_test   # -> PASS sparse_softmax_attention
```

Conversion (CPU, 2026-11-06; the RTX 3090 Ti's 24 GiB is held by a running
`ninfer-serve`, leaving ~0.8 GiB, so no GPU convert and no model load this
session):

```
python -m tools.convert --model E:\LLMs\ninfer-base\Qwen3.8-27B \
  --recipe qwen3_8_27b_qsa_gguf --components text,vision,mtp,dflash2 \
  --source gguf=E:\LLMs\Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf \
  --source vision=E:\LLMs\mmproj-BF16.gguf \
  --source dflash2=E:\LLMs\ninfer-dflash2\official \
  --out E:\LLMs\Qwen3.8-27B-QSA.ninfer --device cpu
# -> wrote E:\LLMs\Qwen3.8-27B-QSA.ninfer: 1292 objects, 1 files, 49.7s
```

Offline artifact inspection (`tools.artifact.reader.Artifact`, directory only,
no payloads): components `text, vision, mtp, dflash2`; 1292 objects / 1591
bindings / 874 uses in one 17.80 GB file; text config carries
`indexer_n_heads=4 / indexer_head_dim=128 / indexer_compress_ratio=4 /
indexer_budget=2048`; all 80 indexer parameters are bound — per full-attention
layer the query (512x5120) and key (128x5120) projections are stacked into one
bf16 (640, 5120) object and split by row range, plus the three (128,) bf16
norms — every object's encoding validated against its declared shape/format;
each indexer projection's use reads the layer's `mixer_input`; the MTP block
carries no indexer (as the fork's MTP block has none).

## How to continue after an interruption

Order of work (the planning document carries the rolling checklist):

1. **Executor wiring** (`src/models/qwen3_5/`) — DONE. `QsaIndexerParameters`
   bound in `execution/parameters.*`; per-layer `pooled`/`tail` indexer state
   planned in `state/decoder_state.*` and the startup plan; the QSA branch in
   `execution/text.cpp::attn_mix` (`run_qsa`: indexer linears ->
   `kv_cache_append` -> `qsa_indexer_append` -> `qsa_indexer_select` ->
   `sparse_softmax_attention`); gated behind `EngineOptions::qwen38_27b_qsa`
   (single device, `max_concurrency == 1`, no speculative, context cache
   auto-disabled) and wired through `ninfer`, `ninfer-perplexity` and
   `ninfer-serve` as `--qwen38-27b-qsa`.
2. **Convert + validate** — DONE (2026-11-06). The conversion above produced
   `E:\LLMs\Qwen3.8-27B-QSA.ninfer` (1292 objects, 1 file, 17.80 GB); the op
   oracles pass in the `ninfer_tests` bundle and the offline artifact
   inspection passes. What remains is end-to-end: start the 27B model once
   the RTX 3090 Ti's VRAM is free, run a prompt with `--qwen38-27b-qsa` and
   compare greedy output / logit KLD against the QSA fork's `llama-server`
   (cold restart, fixed seed, no KV-cache hits). Until then do not benchmark
   the artifact as if it were QSA. Model card:
   `https://hf-mirror.com/higa2515/Qwen3.8-27B-QSA-GGUF`.

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
