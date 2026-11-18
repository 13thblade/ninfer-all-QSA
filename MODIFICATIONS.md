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
| Scope | 27 files added, 82 modified; no upstream file deleted |

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
  its own lane's indexer state, row-bound KV view and stream pool. A
  verification stages its indexer projection in a persistent per-(lane, layer)
  record plane and its tail in a scratch copy, so commit() can replay only the
  accepted prefix.
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
  indexer's configuration and weights, including the 80 `blk.N.indexer.*` tensors
  a QSA GGUF carries (F32 in the source GGUF; the converter casts them to BF16,
  the format the kernels read).
- `program/planning/startup.cpp` sizes the selected matrix, the indexer and split
  workspaces, and the streamed-KV pool from the same selection budget, reserves
  the dense operator's transient capacity beside the sparse one, and admits up
  to the engine's eight concurrent lanes. Under streamed KV it lays out one
  indexer plane plus one tail-axis plane per (lane, layer), an MTP-only record
  plane and first-position word, one device stream pool and map per (lane,
  layer), the shared staging layer, and the pinned host planes. The device text
  and MTP page pools get one full-history page group per lane, the MTP pool plus
  its draft windows, because a verifier reads its complete history and never
  streams.
- A prefix-append prefill keeps the prompt's `rope_delta` in `io_.rope_delta` and
  rotates its continuation at `token index + rope_delta`, so a cached media prompt
  resumes at the same absolute M-RoPE position as a cold prefill of the whole
  prompt. `execution/rope_axes.h` holds the axis-count rule; `execution/text.cpp`
  applies it to new queries, new K/V rows and the first generated token.

### `state` — per-lane QSA state and the state image

- `state/decoder_state.h` carries one indexer lane state per lane (pooled block
  keys, live tail, tail M-RoPE axes, verification scratch) and the Program's
  streamed-KV state: a `host`, `pool` and residency-map triple per (lane, layer),
  the pinned host planes that hold the authoritative KV, and the shared staging
  layer with its identity table. A lane's tenure bumps the staging generation, so
  the next tenant cannot read bytes it did not write.
- `state/state_image.*` adds the QSA payload to the state image. The host image
  carries every layer's pooled keys, live tail and tail axes in one fixed order;
  the device layout binds one such plane set per lane and moves the "rest" part
  through the lane's planes. A state image without the QSA region, or without its
  Host replica, cannot restore a QSA lane; the admission falls back to a
  shallower source instead of resuming from stale planes.
- `program/storage/session_snapshot.cpp` and `program/transactions/commit.cpp`
  use those gates: under the Legacy catalog a QSA session is refused rather than
  saved without its indexer state, the QSA region is trusted only in its Host
  slab, an endpoint image lands it in the destination lane's planes, and a resume
  from a prefix is accepted only on the lane that still owns that state (owner and
  generation checked on commit).

### `program/prefix` — the streamed tier's continuation store and resume

- `program/prefix/qsa_continuation.h` holds the small pieces the recording, the
  store and the admission share: `HeldPrefix` (tokens, cumulative media key,
  position), `LanePrefixRecord` (the lane's `held` and `anchor` prefixes, the
  sequence's own per-64-token media keys, and the lane epoch), `qsa_prefix_reuse`
  (upstream's strict-prefix rule), `qsa_media_key_at` / `qsa_vision_key_at`,
  `qsa_reuse_trace` (`NINFER_DEBUG_REUSE`), the parametric image layout
  (`QsaImageGeometry`, `qsa_image_bytes`, `qsa_copy_image`,
  `qsa_kv_bytes_per_head_token` — the KV width follows `--kv-dtype`, never a
  hardcoded 1,024 bytes) and the plane chunk copy.
- `program/prefix/qsa_continuation.cpp` implements the layout arithmetic and
  `QsaContinuationStore`: 8 MiB page-aligned host pages, exact-key images, a
  budget with LRU eviction, a 128-token floor, longest-prefix lookup keyed at each
  image's own length, prefix dropping, and payload write/read. There is no disk
  tier.
- `program/prefix/qsa_image.*` maps a lane's authoritative KV to image order:
  the planes are kept whole (one memcpy per plane, exact round trip), because a
  page is the inner axis of the `[leading, 128, heads, pages]` tensor and one
  position's pages are strided.
- `program/prefix/qsa_continuation_program.cpp` is the admission itself:
  `qsa_quote` settles the three-way choice (the lane's live prefix, its
  prompt-end snapshot, the store's longest image); `qsa_reserve` starts the
  sequence, adopts the lane's live state or snapshot when one continues the
  prompt, or lands an image — its StateImage half through
  `copy_from_host_segments`, its planes through `qsa_copy_image`; `qsa_retire`
  writes an endpoint image before the lane is released and charges the real
  StateImage geometry, so a store failure never fails the request that just
  finished. Anchor capture runs where a prompt ends (`prefill.cpp`), and the
  lane's record is advanced in `program_impl.cpp` where the KV frontier moves.
- `program/prefix/hybrid_program.cpp` is upstream except for the pieces that are
  not reuse: the loud refusal to build the hybrid cache for the streamed tier,
  the two QSA StateImage gates (a snapshot's QSA state is only restorable from its
  Host replica), and `qsa_end_lane_tenure` / `qsa_clear_lane_prefix` in
  `hybrid_release_lane` (which therefore runs with or without a hybrid cache).
  `hybrid_cache.*`, `hybrid_host_layout.*`, `hybrid_persist.*` and
  `runtime/prefix_cache/prefix_index.*` are upstream verbatim again; the streamed
  tier no longer reuses blocks through the hybrid cache's Host slab.
- `program/planning/request_plan.cpp` binds a request to a lane: it reports the
  three-way scan under `NINFER_DEBUG_REUSE`, keeps the Legacy catalog's lane
  ownership rule, and lets the streamed tier choose its lane in the resource
  manager. The program plumbing carries the admission types
  (`QsaAdmissionQuote`, the pending publication, the lane-selection helpers) in
  `program.h`, `program_impl.*`, `program.cpp`, `context.h`, `decode.cpp`,
  `graphs.cpp`, `prefill.cpp`, `runtime_types.h` and `program_sources.cmake`.

### `runtime` — cache mode, admission and capacity

- `runtime/engine/context_cache/qsa_resource_manager.h` presents the same
  ResourceManager surface as `HybridResourceManager`, so scheduling, prefill,
  decode and commit orchestration stay shared. Its quote is taken against every
  free lane and the highest reuse wins; a tie goes to a lane that carries no
  other conversation's state, so serial root admissions spread over lanes instead
  of stacking on lane 0 and evicting its state. A reservation completes
  synchronously (the image's two halves are host memory), progress publishes the
  sequence it already started, and terminal settlement retires the lane's prefix
  into the store before the lane is released.
- `runtime/engine/engine.cpp` selects that core for `ContextCacheMode::Qsa`.
- `runtime/engine/model_instance.cpp` normalizes the mode: it rejects Legacy
  policies and shared/long-anchor catalogs, sizes the private catalog and the
  state-image pool for the lanes (two images per lane: the sequence in flight and
  its prompt-end snapshot), turns `--host-cache-mib auto` into "no store" (roots
  only), and refuses an explicit `--max-context` that cannot hold every lane with
  a hint that names `--kv-resident`. The Legacy-catalog normalization for a QSA
  branch built through the Engine API still keeps exactly one conversation.
- `runtime/engine/kv_capacity.cpp` accepts the host-resident curve's zero byte
  stride and caps the resolved page count at the curve's maximum.

### `ops/kv_stream` — streamed KV

- `--kv-resident N|all|auto` (`src/product/kv_resident_options.h`) sets how many
  positions of each sparse-attention layer's KV stay in device memory; past them
  the layers stream from pinned host memory. `auto` follows the upstream preset
  (all with device experts, 32,768 with host or disk experts).
- `src/ops/kv_stream/kv_stream.cu` and `include/ninfer/ops/kv_stream.h` move the
  authoritative KV to pinned host memory behind one shared staging layer and plan
  the device pool at two windows, column 0 plus one column of draft headroom
  (`2 * --qsa-top-blocks` blocks, 4096 tokens at the default 512;
  `planning/startup.cpp`). The query-block count is budget-parametric
  (`kKvStreamQueryBlocks`, `kv_stream_minimum_slots(tokens, query_blocks)`), and a
  committed call without draft columns refuses an overflowing pool (control[3] 2)
  instead of emptying a real column's shortlist. The pool's CLOCK sweep runs two
  passes: it first takes slots nothing reads this round, and only then reclaims a
  draft column's slot, which costs that column its KV and is reported as the
  recoverable overflow.
- An overflowing round degrades on the device, before acceptance:
  `resolve_kernel` writes the worst overflow code into a per-lane device flag and
  empties the draft columns' shortlists (their blocks are `-1` in the table),
  `kv_stream_degrade_verify_extents()` then withholds those lanes' drafts so the
  acceptance kernel itself licenses column 0's target token with consistent
  lengths, anchors and selected hidden state. `program/stream_fallback.h` folds
  the per-lane words for the code-2 refusal and the K/V-only pass gate; a
  streamed QSA layer never runs the last-layer K/V-only fast path, whose device
  text-KV index has only the shared staging layer. `request_done.speculative`
  reports `qsa_stream_overflow_rounds`, `qsa_stream_degraded_rounds` and
  `qsa_stream_max_overflow_code`.

### `serve`, `apps/cli` — the flag surface

- `--qwen38-27b-qsa` is accepted by `ninfer-serve`, `ninfer` and
  `ninfer-perplexity`, and `--qsa-top-blocks N` by the same three, through the
  shared `src/product/qsa_top_blocks_options.h`. The count is in four-token
  blocks, 1 to 65,536; unset or `0` keeps the artifact's own 512 blocks, and the
  value is fixed at startup so a captured graph replays the grid it captured.
- `src/serve/serve_options.cpp` selects `ContextCacheMode::Qsa` when
  `--qwen38-27b-qsa` runs with a numeric (or `all`) `--kv-resident`, and leaves
  `--host-cache-mib` as the store's budget. It refuses `--qwen38-27b-qsa` with
  `--use-original-prefix-caching`, because the checkpoint catalog has no
  candidate that could rebuild a QSA session.
- `src/serve/operational_log.cpp` logs the effective history window with the
  source it came from; `src/serve/request_log.cpp` carries the stream counters.
- Vision follows `--max-context`: `frontend/frontend.cpp` drops the 32,768-token
  aggregate ceiling (`include/ninfer/base.h` raises the media accounting ceiling
  to 12 GiB), and `generation_service.cpp` sizes the per-request host patch buffer
  from `--max-request-mib`. Device footprint is unchanged: the per-item bound
  still sizes the workspace.

### `tools/convert` — the converter and the weight graft

- `qsa_gguf.py` implements the `qwen3_8_27b_qsa_gguf` recipe registered in
  `gguf_blocks.py`; it teaches the converter the 80 extra `blk.N.indexer.*`
  tensors, and `convert_qsa27b.py` is the one-shot wrapper that writes a single
  `.ninfer` artifact. The converter un-folds the norms' `1 + w` offset and casts the
  donor's F32 indexer tensors to BF16, the kernels' format, and its parser
  (`build_parser`) pins the default that packs the indexed proposal head, with an
  explicit opt-out.
- `qsa_graft_recipe.py` grafts the indexer tensors and the native BF16 GDN
  controls onto an artifact converted from another quantization; the donor is
  optional, and `extract_indexer_tensors.py` / `extract_gdn_controls.py` write the
  manifests under `tools/convert/assets/` that make a graft reproducible. A
  missing asset warns loudly instead of falling back to a machine-local path.
- `scripts/verify_qsa_gguf.py` is the offline preflight for a downloaded QSA GGUF.

### Tests

- Operator: `tests/ops/test_qsa_indexer.cpp` covers the 24-over-4 select, the
  three-axis M-RoPE with YaRN and interpolation, and the selection budget;
  `test_sparse_attention.cpp` and `test_kv_stream.cpp` cover the attention and
  streaming paths.
- Streamed tier: `tests/models/qwen3_5/test_qsa_lane_prefix.cpp` pins the lane
  record (frontier follows writes, a stop cut truncates, the media key is the
  record's own, a text-only prefix keys zero, the anchor catches a prompt that
  drops the generation, clear drops everything);
  `test_qsa_continuation_store.cpp` covers the prefix rule, budget and LRU order,
  exact-key refresh, the 128-token floor, page alignment, longest prefix lookup,
  the KV-width table across precisions, image arithmetic at 13,312 / 67,136 /
  131,072 tokens, the plane and payload round trips with the short-payload
  refusal, and the three-way choice (live, anchor, store, an equal-length
  regenerate, a media mismatch, and a lane epoch change that leaves only the
  store).
- State and layout: `tests/models/qwen3_5/test_state_image_layout.cpp` covers the
  QSA regions of the image; `tests/models/qwen3_5/test_engine_hybrid_prefix_real.cpp`
  gates its readout/shared-read diagnostics behind `NINFER_TEST_READOUT_DEBUG` and
  `NINFER_TEST_SHARED_READS`, and serializes the reference engine in the
  stop-string scenario so one 27B model fits a 24 GiB card.
- Options: `tests/test_cli_options.cpp` and `tests/test_serve_options.cpp` cover
  the flags and the cache-mode selection; `test_kv_capacity.cpp` and
  `test_context_cache_defaults.cpp` cover the capacity policy; and
  `test_prefix_cache_index.cpp` is upstream's again.
- Positions: `tests/models/qwen3_5/test_prefill_rope_axes.cpp` pins the axis
  count of cold, prefix-append and multimodal prefills, and the post-media
  `index + rope_delta` equivalence a prefix-append prefill relies on.
- Overflow fallback: `tests/models/qwen3_5/test_qsa_stream_fallback.cpp` covers
  the per-lane overflow words, the streamed K/V-only gate and the device-side
  degradation rules; `tests/ops/test_kv_stream.cpp` pins the two-column pool
  floor, column 0's residency, the truncated draft shortlists and the round
  flag; `tests/test_prefill_pacing.cpp` covers the QSA step floor.
- Converter: `tests/convert/test_qwen3_8_27b_qsa_gguf.py` and
  `test_gguf_blocks.py`.

### Build

- `cmake/NinferTargets.cmake` stages the FFmpeg/curl runtime closure next to
  every binary. The vcpkg zlib port outputs `zlib.dll`, so the closure glob
  matches `zlib*.dll`; the previous `z.dll` name matched nothing and the staged
  binaries aborted at load time with an unresolved `zlib.dll` import.
- `src/models/qwen3_5/program_sources.cmake` adds the store, image and admission
  sources; `tests/models/qwen3_5/tests.cmake` registers their tests.

### Benchmarks and diagnostics

- `tools/bench/pcie_bandwidth_monitor.py` samples
  `nvmlDeviceGetPcieThroughput` through `ctypes` from a background thread and
  integrates the readings over request windows; the engine carries no PCIe
  counters of its own.
- `tools/bench/run_qsa27b_kvstream_bench.py` (`perf32k`, `pcie512k`) and
  `run_qsa27b_32k_ab.py` drive a server through those windows.
- `NINFER_DEBUG_REUSE` prints `[resume]` (per-lane live/anchored/stored and the
  chosen path), `[admit]` (the lane and the reuse a request actually took) and
  `[retire]` (the image written). `SpeculativeStats`
  (`include/ninfer/observability/execution.h`) carries the three
  `qsa_stream_*` counters, `observability/resources.h` reports the staging and
  pinned-host KV bytes beside the MTP pool, and the KV-frontier diagnostic in
  `program/storage/kv_store.h` names the frontier, the committed frontier and the
  page capacity when a commit is rejected.

### Documentation

- `README.md` states the fork's attribution, the three flags, the streamed-KV
  floor and the QSA prefill/decode measurements.
- `docs/cli.md` and `docs/serving.md` list `--qwen38-27b-qsa`,
  `--qsa-top-blocks`, the Qwen3.8-27B reading of `--kv-resident` and the media
  accounting ceiling; `docs/maintainer/qwen4-exp-model.md` records where the sparse
  branch sits in the Qwen4 experimental model, and that this fork makes its width
  runtime-selectable.

### Not carried here

The branch `experiment/qsa-hybrid-streamed` holds a separate line of work:
keeping a conversation's resume state inside the hybrid cache, with snapshot
ownership marks, live-lane prefix pages and block-level Host slabs. That line is
not part of this branch: the pieces it had put into `hybrid_cache.*`,
`hybrid_host_layout.*`, `hybrid_persist.*` and `prefix_index.*` were reverted,
because the streamed tier's block content lives in the pinned host planes while
the hybrid cache's Host layout describes dense Device pages that under
`--kv-resident` are bookkeeping placeholders carrying no KV, so a slab copied from
them restores nothing. The streamed tier resumes through its own store instead
(above), and the hybrid cache refuses to be built for it.

## Known limits

- A store resume is refused while MTP drafts run: an image carries no draft KV,
  so the admission waits (`Aborted`) rather than restoring a state the drafter
  did not follow. A live or anchor resume, and every root, is unaffected.
- `--host-cache-mib auto` has no streamed-tier model yet: without an explicit
  budget the store stays absent and the tier admits roots only. The default
  budget is 8 GiB; `--host-cache-mib 0` disables the store.
- Only a conversation's endpoint is written to the store. A turn closure resumes
  from the lane's retained prompt-end snapshot instead; an image is what a
  conversation resumes from once another session took its lane. An image keeps
  the lane's planes whole, so it costs the lane's full plane capacity.
- QSA runs on one device without tensor parallelism, with at most the engine's
  eight lanes, and with CUDA graphs off above one concurrent request. Upstream
  `qwen4_exp` (native QSA) also runs batches of several sequences eagerly, so a
  per-row lane indirection inside captured kernels has no upstream
  implementation to follow.
- `--qwen38-27b-qsa` cannot be combined with `--use-original-prefix-caching`: the
  checkpoint catalog is upstream-shaped and carries no QSA state. Built through
  the Engine API, a QSA Legacy branch is still restricted to one conversation's
  own next turn.
- Streaming needs the context to be larger than the pool floor (`2 *
  --qsa-top-blocks` blocks, or `--kv-resident` when that is larger); a context at
  or below it turns streaming off and keeps every full-attention layer resident
  (the startup warning names the levers).

## Files

### Added (27)

Root (1)

- `MODIFICATIONS.md`

include/ninfer/ops (1)

- `include/ninfer/ops/qsa_budget.h`

scripts (1)

- `scripts/verify_qsa_gguf.py`

src/models/qwen3_5/execution (1)

- `src/models/qwen3_5/execution/rope_axes.h`

src/models/qwen3_5/program (1)

- `src/models/qwen3_5/program/stream_fallback.h`

src/models/qwen3_5/program/prefix (5)

- `src/models/qwen3_5/program/prefix/qsa_continuation.cpp`
- `src/models/qwen3_5/program/prefix/qsa_continuation.h`
- `src/models/qwen3_5/program/prefix/qsa_continuation_program.cpp`
- `src/models/qwen3_5/program/prefix/qsa_image.cpp`
- `src/models/qwen3_5/program/prefix/qsa_image.h`

src/product (1)

- `src/product/qsa_top_blocks_options.h`

src/runtime/engine/context_cache (1)

- `src/runtime/engine/context_cache/qsa_resource_manager.h`

tests/convert (1)

- `tests/convert/test_qwen3_8_27b_qsa_gguf.py`

tests/models/qwen3_5 (4)

- `tests/models/qwen3_5/test_prefill_rope_axes.cpp`
- `tests/models/qwen3_5/test_qsa_continuation_store.cpp`
- `tests/models/qwen3_5/test_qsa_lane_prefix.cpp`
- `tests/models/qwen3_5/test_qsa_stream_fallback.cpp`

tools/bench (3)

- `tools/bench/pcie_bandwidth_monitor.py`
- `tools/bench/run_qsa27b_32k_ab.py`
- `tools/bench/run_qsa27b_kvstream_bench.py`

tools/convert/assets (2)

- `tools/convert/assets/gdn_controls_bf16.manifest.json`
- `tools/convert/assets/indexer_tensors.manifest.json`

tools/convert (5)

- `tools/convert/convert_qsa27b.py`
- `tools/convert/extract_gdn_controls.py`
- `tools/convert/extract_indexer_tensors.py`
- `tools/convert/qsa_gguf.py`
- `tools/convert/qsa_graft_recipe.py`

### Modified (82)

Root (1)

- `README.md`

apps/cli (3)

- `apps/cli/main.cpp`
- `apps/cli/options.cpp`
- `apps/cli/options.h`

apps/perplexity (1)

- `apps/perplexity/main.cpp`

cmake (1)

- `cmake/NinferTargets.cmake`

docs (2)

- `docs/cli.md`
- `docs/serving.md`

docs/maintainer (1)

- `docs/maintainer/qwen4-exp-model.md`

include/ninfer (2)

- `include/ninfer/base.h`
- `include/ninfer/engine_options.h`

include/ninfer/observability (2)

- `include/ninfer/observability/execution.h`
- `include/ninfer/observability/resources.h`

include/ninfer/ops (4)

- `include/ninfer/ops/kv_stream.h`
- `include/ninfer/ops/qsa_indexer.h`
- `include/ninfer/ops/rope.h`
- `include/ninfer/ops/sparse_attention.h`

src/models/qwen3_5 (4)

- `src/models/qwen3_5/config.cpp`
- `src/models/qwen3_5/config.h`
- `src/models/qwen3_5/program_sources.cmake`
- `src/models/qwen3_5/weights.h`

src/models/qwen3_5/execution (4)

- `src/models/qwen3_5/execution/parameters.cpp`
- `src/models/qwen3_5/execution/parameters.h`
- `src/models/qwen3_5/execution/text.cpp`
- `src/models/qwen3_5/execution/text.h`

src/models/qwen3_5/frontend (1)

- `src/models/qwen3_5/frontend/frontend.cpp`

src/models/qwen3_5/load (1)

- `src/models/qwen3_5/load/text.cpp`

src/models/qwen3_5/program (9)

- `src/models/qwen3_5/program/context.h`
- `src/models/qwen3_5/program/decode.cpp`
- `src/models/qwen3_5/program/graphs.cpp`
- `src/models/qwen3_5/program/prefill.cpp`
- `src/models/qwen3_5/program/program.cpp`
- `src/models/qwen3_5/program/program.h`
- `src/models/qwen3_5/program/program_impl.cpp`
- `src/models/qwen3_5/program/program_impl.h`
- `src/models/qwen3_5/program/runtime_types.h`

src/models/qwen3_5/program/planning (3)

- `src/models/qwen3_5/program/planning/request_plan.cpp`
- `src/models/qwen3_5/program/planning/startup.cpp`
- `src/models/qwen3_5/program/planning/startup.h`

src/models/qwen3_5/program/prefix (1)

- `src/models/qwen3_5/program/prefix/hybrid_program.cpp`

src/models/qwen3_5/program/speculative (2)

- `src/models/qwen3_5/program/speculative/mtp.cpp`
- `src/models/qwen3_5/program/speculative/target_verification.cpp`

src/models/qwen3_5/program/storage (3)

- `src/models/qwen3_5/program/storage/context.cpp`
- `src/models/qwen3_5/program/storage/kv_store.h`
- `src/models/qwen3_5/program/storage/session_snapshot.cpp`

src/models/qwen3_5/program/transactions (2)

- `src/models/qwen3_5/program/transactions/commit.cpp`
- `src/models/qwen3_5/program/transactions/materialization.cpp`

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

src/runtime/engine (5)

- `src/runtime/engine/engine.cpp`
- `src/runtime/engine/engine_core.h`
- `src/runtime/engine/kv_capacity.cpp`
- `src/runtime/engine/model_instance.cpp`
- `src/runtime/engine/prefill_pacing.h`

src/serve (6)

- `src/serve/generation_service.cpp`
- `src/serve/generation_service.h`
- `src/serve/operational_log.cpp`
- `src/serve/request_log.cpp`
- `src/serve/serve_options.cpp`
- `src/serve/serve_options.h`

tests (5)

- `tests/test_cli_options.cpp`
- `tests/test_context_cache_defaults.cpp`
- `tests/test_kv_capacity.cpp`
- `tests/test_prefill_pacing.cpp`
- `tests/test_serve_options.cpp`

tests/convert (1)

- `tests/convert/test_gguf_blocks.py`

tests/models/qwen3_5 (3)

- `tests/models/qwen3_5/test_engine_hybrid_prefix_real.cpp`
- `tests/models/qwen3_5/test_state_image_layout.cpp`
- `tests/models/qwen3_5/tests.cmake`

tests/ops (3)

- `tests/ops/test_kv_stream.cpp`
- `tests/ops/test_qsa_indexer.cpp`
- `tests/ops/test_sparse_attention.cpp`

tools/convert (2)

- `tools/convert/__main__.py`
- `tools/convert/gguf_blocks.py`
