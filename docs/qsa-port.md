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
| `08e539c` | `docs: record the landed QSA executor and converter commits` | done |
| `3127a61` | `engine: run QSA-27B with the artifact's own MTP speculative drafts` | done: full build, op tests, cold-start smoke, greedy A/B, 32K benchmark |

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
artifact's `config().text.qsa`, one device, no tensor parallelism and
`max_concurrency == 1`. The only admitted speculative backend is the
artifact's own MTP head; DFlash/DFlash2/DSpark are rejected. `--vision` is
rejected (see below). The context cache stays off because the indexer state
is not in `StateImage` yet. The indexer state is one rank-0 FP32 copy per
full-attention layer (`pooled [128, capacity/4 + 1]` + `tail [128, 3]` +
`tail_scratch [128, 3]` in one plane); a new prefill repools from block zero,
so no initialization is needed while the context cache is off. Decode reads
the sequence through its row-bound execution view (`text_kv_view`); the CUDA
Graph decode is captured with the capture row bound, so replay reads the
row's published block table.

**MTP speculation** (`--spec mtp --draft-tokens N`, best with
`--ngram-draft-tokens 0`) is supported. A verification is one N-column target
pass: `qsa_indexer_append`, `qsa_indexer_select` and
`sparse_softmax_attention` are already N-token operators, so no kernel
changes were needed. `run_qsa` stages the verification's projection into a
persistent per-layer record plane (a stable address for the captured decode
graph) and pools into a per-layer scratch tail, leaving the live pooled/tail
planes at the pre-verification frontier. `resolve_pending_raw` re-runs
`qsa_indexer_append` over exactly the accepted columns of those records at
commit, so rejected drafts never enter the persistent state — this is the
same tail-replay contract as `qwen4_exp`'s executor and the fork's
`seq_rm`/`blk_rebuild` state machine. The branch admits a prefill chunk, an
ordinary one-column decode, one whole-pass batched-prefill segment, or one
N-column verification; anything else throws.

## What does NOT work yet

- **DFlash / DFlash2 speculation under QSA**: the masked draft backends keep
  their own KV pool and replay contract; only the artifact's own MTP head is
  wired to the QSA indexer. Deliberately deferred — the conflict surface is
  large and MTP is the model's native drafter.
- **Vision / M-RoPE under QSA**: the artifact does carry the Vision tower and
  the dense path fully supports `--vision` with 3-axis M-RoPE, but
  `qsa_indexer.cu` rotates with a scalar position + YaRN. An image/video
  prompt would make the indexer select the wrong blocks *silently*, so
  `--qwen38-27b-qsa --vision` is currently refused at startup.
  **Correction to an earlier note**: the fork does *not* use 4 contiguous
  `sections`. `LLM_ARCH_QWEN35` maps to `LLAMA_ROPE_TYPE_IMROPE`
  (`llama-model.cpp:3211-3217`), whose `sector % 3` interleaving is
  numerically identical to ninfer's `pair_axes`/`pair % 3`
  (`config.cpp:91-100`) for `sections = [11,11,10,0]`. So the fix is to push
  the engine's existing axis-major `[T,3]` positions down into the indexer
  kernels, not to add a second RoPE semantics. Pure text has three equal axes,
  which must reduce exactly to the current scalar path — a self-contained
  regression test that needs no external multimodal reference. See the
  roadmap below.
- **Prefix-cache / continuation reuse under QSA** needs the pooled+tail
  planes in `StateImage`; the context cache is disabled instead.
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

1. **Executor wiring** — DONE (`75d30b9`). `QsaIndexerParameters` bound in
   `execution/parameters.*`; per-layer `pooled`/`tail` planes in
   `state/decoder_state.*` and the startup plan; the QSA branch of
   `execution/text.cpp::attn_mix` (`run_qsa`) with the explicit
   `kv_cache_append`; startup gates, context-cache auto-disable, CLI/serve
   flag plumbing. The Flash-Next `kv_stream` (host-resident KV) is not used:
   QSA-27B reads its sequence's paged KV on the device.
2. **MTP speculative replay** — DONE (`3127a61`). Persistent verification
   records, a scratch tail, and the accepted-prefix re-pool in
   `resolve_pending_raw`. Verified by a 512-token greedy A/B at 32K context:
   `qsa_mtp` output is byte-identical to `qsa_plain`.
3. **M-RoPE / Vision under QSA** — the QSA fork rotates the indexer with
   `ggml_rope_multi`'s 3D sections; `qsa_indexer.cu` uses a scalar position +
   YaRN. Text-only is equivalent; image/video needs section-aware per-block
   positions. The combination is refused at startup rather than silently
   computing wrong selections.
4. **Prefix-cache state** — carry `pooled`/`tail` in `StateImage` so a
   continuation can restore the indexer; the context cache stays off until
   then.
5. **DFlash2 / `--kv-resident` on qwen3_5** — masked draft backends need
   their own replay contract; `--kv-resident` is still forwarded only by
   `src/runtime/engine/qwen4_exp_core.cpp:305`.
6. **Cross-engine numerical alignment** — compare greedy token sequences /
   logit KLD against the QSA fork's `llama-server` on the same prompt, cold
   restart, fixed seed, no KV-cache hits. Still outstanding; the 32K run
   below is an *in-engine* A/B, not a correctness proof against the fork.

## Roadmap: Vision and streamed KV (rolling checklist)

The detailed, interruptible plan lives in
`D:\WorkBuddy\2026-11-06-10-13-06\QSA27B_ninfer_移植方案.md` §9. Each stage is
verified, ticked there and in this table, then committed separately.

| Stage | Work | State |
|---|---|---|
| V0 | Source survey: IMROPE == `pair_axes`; locate the 3D position path | done |
| V1 | `qsa_indexer` append/select gain optional axis-major `[T,3]` positions; scalar path unchanged | done (`567f2ca`) |
| V2 | Operator test: 3D cases + "three equal axes == scalar" bit-for-bit regression, YaRN x 3D, interpolation x 3D | done |
| V3 | qwen3_5: persistent per-block/tail axis planes, `run_qsa` forwarding, commit re-pool with 3D | todo |
| V4 | Remove the `--vision` gate; real multimodal smoke | todo |
| K0 | Survey: qwen3_5 has no kv_stream wiring; `--kv-resident` silently ignored | done |
| K1 | Forward `kv_resident_positions`; plan per-layer host/pool/map planes; help text | todo |
| K2 | Split host identity / pool views in `run_qsa`; `kv_stream_resolve`/`evict`; wide-call staging | todo |
| K3 | Cap verify width at 16 when streaming; evict rejected draft blocks at commit; fixed graph addresses | todo |
| K4 | Full build, op regression, streamed-vs-resident greedy A/B at 32K | todo |

Constraints worth keeping in view: `kv_stream_resolve` accepts at most 16 query
columns with exactly `[512, tokens]` selected blocks (`kv_stream.cu:364-375`),
while qwen3_5 allows a 64-column verification window; the QSA-27B model has 16
full-attention layers against Flash-Next's 12, so the `verify_width * 513` slot
pool is larger; and `HostKVExtentStore` in `program_impl.h` is a context-cache
replica, not the identity-layout pinned host copy that kv_stream requires.

**Hard constraint: YaRN.** Every QSA/KV addition must stay compatible with the
model's YaRN. Verified: the indexer's `inverse_frequency[pair]` and
`attention_factor` come from the same `yarn_schedule()` as dense
(`launcher/rope.cu:192-225`), the same `RopeYarn` value reaches dense, MTP,
the indexer and the speculative commit re-pool, dense YaRN permits `axes == 3`
(`wrapper/rope.cpp:99-110`), and `kv_stream` does no rotation and has no
`kv_resident × rope_yarn` gate. "Model-supplied YaRN" here means the native
262,144-position window plus Qwen's YaRN extension resolved by the CLI
(`--rope-yarn` / `--rope-yarn-factor`, up to 4x); there is no `rope_scaling`
object in the artifact. V1 additionally removes the indexer's old rejection of
`--rope-scaling-factor`: linear interpolation now runs per axis with the same
threshold/scale as `rope.cuh`'s YaRN kernel, so the indexer no longer diverges
from dense. Stage acceptance (Y4–Y9 in the planning document §9.7) requires
YaRN x 3D operator cases, a bit-for-bit "three equal axes == scalar"
regression, and a streamed-vs-resident A/B with YaRN enabled.

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

### Real model load (2026-11-06, VRAM freed)

The 3090 Ti was idle (24,309 MiB free), so the converted artifact was finally
loaded. **This was the first execution of the QSA branch on real weights** and
it exposed two latent `Tensor::view` bugs (a reshape that changed the element
count): `call_positions.view({1})` and `index.view({query_rows, T})` both
threw `view element count mismatch` during CUDA Graph preparation. `3127a61`
fixes them — the first position is now a one-word Tensor over the same device
address, and the two indexer linears write contiguous planes that a 2D copy
assembles into the Op's `[640, tokens]` layout.

Full-tree rebuild after `3127a61`: 92/92 edges, all apps and `ninfer_tests`
link. Operator tests: `ninfer_qsa_indexer_test` -> `PASS qsa_indexer`,
`ninfer_sparse_attention_test` -> `PASS sparse_softmax_attention`.

Cold-start smoke, QSA + native MTP (`--spec mtp --draft-tokens 2
--ngram-draft-tokens 0 --qwen38-27b-qsa`, 2048 ctx, int8 KV, greedy):
`engine ready` in 8.4 s, CUDA graphs captured in 550 ms, 15 verification
rounds, 30 drafted / 20 accepted tokens, `prefix_cache_hit_tokens=0`, coherent
text. The request log confirms `speculative_backend=mtp`, `draft_window=2`,
`ngram_draft_window=0`, `context_cache.enabled=false`.

Four-arm greedy A/B (one cold server per arm, same prompt/seed, 12-token
prompt, 256 output tokens, `--no-prefix-reuse`, `cache_prompt:false`):

| arm | prefill tok/s | decode tok/s | MTP accepted |
|---|---|---|---|
| dense, no spec | 789.0 | 52.2 | - |
| dense + MTP | 847.7 | 94.0 | 148/213 |
| QSA, no spec | 735.4 | 50.9 | - |
| QSA + MTP | 739.9 | 89.1 | 146/216 |

`qsa_mtp` and `qsa_plain` produced **identical 256-token greedy text**, and the
same equality holds at 32K context with 512 output tokens: the accepted-prefix
re-pool keeps speculation from changing QSA's greedy stream.

32K benchmark (`tools/bench/run_qsa27b_32k_ab.py`; `--max-context 32768`,
`--kv-capacity 32768`, int8 KV, one 30,550-token deterministic prompt, 512
output tokens forced with `ignore_eos`, seed 57004, greedy, no prefix reuse,
`--draft-tokens 3 --ngram-draft-tokens 0`):

| arm | prefill tok/s | decode tok/s | TTFT | MTP accepted |
|---|---|---|---|---|
| dense, no spec | 1908.0 | 49.4 | 16014 ms | - |
| dense + MTP | 1891.7 | 106.4 | 16168 ms | 349/485 |
| QSA, no spec | 1568.9 | 49.5 | 19490 ms | - |
| QSA + MTP | 1564.4 | 97.6 | 19546 ms | 339/516 |

MTP more than doubles decode in both branches (dense 2.15x, QSA 1.98x). QSA
is 0.82x dense on prefill and parity on decode at this length: the two
indexer linears, pooling and radix selection still cost more than the KV reads
they save. First things to try next are a larger `--prefill-chunk` with
`--prefill-cublas`, the fork's dense fallback below 2,051 positions, and a
longer context where the 512-block budget actually binds. Note also that
`dense_plain != dense_mtp` on this engine (greedy diverges after ~172
characters); that is pre-existing dense/verify numerics, not introduced by the
QSA port, and needs its own logit comparison.

## How to continue after an interruption

Order of work (the planning document carries the rolling checklist):

1. **Executor wiring + native MTP** — DONE (`75d30b9`, `3127a61`).
   `QsaIndexerParameters` bound in `execution/parameters.*`; per-layer
   `pooled`/`tail`/`tail_scratch` planes plus verification records in
   `state/decoder_state.*` and the startup plan; the QSA branch in
   `execution/text.cpp::attn_mix` (`run_qsa`: indexer linears ->
   `kv_cache_append` -> `qsa_indexer_append` -> `qsa_indexer_select` ->
   `sparse_softmax_attention`); accepted-prefix re-pool in
   `program/prefill.cpp::resolve_pending_raw`. Gated behind
   `EngineOptions::qwen38_27b_qsa` (single device, `max_concurrency == 1`,
   MTP only, `--vision` refused, context cache auto-disabled) and wired
   through `ninfer`, `ninfer-perplexity` and `ninfer-serve`.
2. **Convert + validate + real load** — DONE (2026-11-06). The conversion
   produced `E:\LLMs\Qwen3.8-27B-QSA.ninfer` (1292 objects, 1 file,
   17.80 GB); op oracles, offline artifact inspection, full-tree build,
   cold-start QSA+MTP smoke, four-arm greedy A/B and the 32K benchmark above
   all pass. **Next**, in order: (a) cross-engine greedy / logit KLD against
   the QSA fork's `llama-server` — the in-engine A/B above is not a
   correctness proof against the fork; (b) M-RoPE section-aware indexer
   positions so `--vision` can be enabled under QSA; (c) `pooled`/`tail` in
   `StateImage` to restore the context cache; (d) DFlash2 replay and
   `--kv-resident`. Model card:
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

QSA and dense greedy streams diverge on long prompts in this port (the 32K A/B
above). Expected contributors: BF16 indexer norms, the sparse softmax
reduction, and the missing dense fallback below 2,051 positions. Speculation
itself is clean under QSA (`qsa_mtp == qsa_plain`), but on the *dense* path
`dense_mtp` diverges from `dense_plain` after ~172 characters — a pre-existing
verify-width numerics issue in the engine, unrelated to this port. Both need a
logit-level comparison against the fork before any quality claim.
