#pragma once

#include "core/paged_kv_view.h"
#include "core/tensor.h"
#include "ninfer/ops/qsa_budget.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * A streamed sparse-attention layer of one sequence (Qwen3.8-Flash-Next QSA): its whole paged KV
 * lives in host memory that the device addresses directly (pinned, under unified addressing), and
 * the device holds a pool of `slots` pages of kKvStreamPageTokens positions each -- one indexer
 * block, every KV head, every plane of the storage -- with the residency map below in device
 * memory at fixed addresses, which captured graphs keep.
 *
 * The host copy is laid out as a resident layer with the identity block table: planes
 * [leading, 64, heads, pages], logical page i at page i, which kv_cache_append writes. The pool's
 * planes are [leading, 4, heads, slots] and the map's `table` is its block table: block b lives in
 * slot table[b], or nowhere (-1). sparse_softmax_attention reads the pool through that table.
 *
 * Before the attention of up to kKvStreamMaxTokens queries at consecutive positions
 * first .. first + tokens - 1, after the append that wrote those positions into the host copy,
 * kv_stream_resolve makes resident every block the attention reads: each query's selected blocks
 * and its own incomplete block. A block that is missing takes a slot no block of this call uses,
 * by a CLOCK sweep (second chance), and is copied from the host copy -- or, when the host copy
 * does not hold it (the lane's prefix under the streamed tier), from the prefix source the
 * caller names per 64-position block run. Blocks the append wrote
 * (first / 4 .. (first + tokens - 1) / 4) are copied again when the call reads them and evicted
 * when it does not, so no slot ever holds bytes older than the host copy. The map's counters
 * record the calls, the blocks looked up and the blocks copied. If the columns' union exceeds the
 * pool, column 0's blocks and the blocks the call wrote are placed first and the remaining draft
 * columns' blocks are dropped; control[3] reports this round's overflow (1 = only draft blocks
 * were dropped, the caller accepts column 0 alone; 2 = column 0 could not be kept complete, the
 * caller must refuse the round).
 *
 * A wide call (a prompt chunk) reads its layer from a fully resident staging copy instead, and
 * kv_stream_evict then drops the blocks it wrote from the pool.
 *
 * The residency scheme (the map, the claim of a missing block, the clock sweep that never takes a
 * slot of the current call) follows Strata's kv_stream (MIT licence, see CONTRIBUTORS.md).
 */
inline constexpr std::int32_t kKvStreamPageTokens = 4;
inline constexpr std::int32_t kKvStreamMaxTokens  = 16;
// The most blocks one query references: its selected blocks and its own incomplete block.
inline constexpr std::int32_t kKvStreamQueryBlocks = kQsaTopBlocksDefault + 1;

struct KvStreamMap {
    std::int32_t* table        = nullptr; // [blocks] block -> slot, or -1
    std::int32_t* slot_block   = nullptr; // [slots] slot -> block, or -1
    std::int32_t* slot_stamp   = nullptr; // [slots] the resolve that last used the slot
    std::int32_t* slot_ref     = nullptr; // [slots] the clock's reference bit
    std::int32_t* copy_block   = nullptr; // [slots + 8] a resolve's copies: block
    std::int32_t* copy_slot    = nullptr; // [slots + 8] and its slot
    std::int32_t* copy_primary = nullptr; // [slots + 8] 1 for column 0 or a written block
    std::int32_t* control      = nullptr; // [16] epoch/hand/copies/overflow, counters, diagnostics
    // Shared by every (lane, layer) map: one word per lane, zeroed before each decode round.
    // resolve writes the worst overflow code of its layer here; the MTP acceptance path reads it
    // on the device, before the host synchronizes, so the round degrades in phase.
    std::int32_t* round_flags = nullptr; // device I32 [lanes], owned by ProgramImpl
    std::uint32_t lane        = 0;
    std::int32_t blocks = 0, slots = 0;
};

// The slots a pool needs so that one call of `tokens` queries always fits, each query holding
// `query_blocks` blocks (its selection plus its own incomplete block).
[[nodiscard]] constexpr std::int32_t
kv_stream_minimum_slots(std::int32_t tokens,
                        std::int32_t query_blocks = kKvStreamQueryBlocks) noexcept {
    return tokens * query_blocks;
}

// The slots a QSA verification needs when only column 0's shortlist is guaranteed: one query's
// `stride` selected blocks plus the pages the call itself wrote. The planning floor and
// kv_stream_resolve's per-call check share this function so they cannot drift apart.
[[nodiscard]] constexpr std::int32_t
kv_stream_single_column_slots(std::int32_t stride, std::int32_t tokens) noexcept {
    return stride + (tokens + kKvStreamPageTokens - 1) / kKvStreamPageTokens;
}

// Device bytes of one map, and the map laid out in `memory` (256-byte aligned, that many bytes).
[[nodiscard]] std::size_t kv_stream_map_bytes(std::int32_t blocks, std::int32_t slots);
[[nodiscard]] KvStreamMap kv_stream_map(void* memory, std::int32_t blocks, std::int32_t slots);

// Every block evicted, every slot free, the counters zero. Capturable.
void kv_stream_reset(const KvStreamMap& map, cudaStream_t stream);

// One run of the lane's prefix blocks: the 64-position blocks
// [first_block, first_block + count) whose block records live in one pinned source
// (a writer's per-sequence stream image, or a snapshot's tail slab record), the first
// block's record at `base`. A block record packs the streamed layer pages in layer
// order; the record-to-record distance is the caller's `record_stride`, and each
// layer page's planes sit at `plane_offset[a]` within the record (exclusive of the
// plane's own within-page offset). The source's page layout is the host copy's: one
// 64-position page of [leading, 64, heads] per plane.
struct KvStreamPrefixSegment {
    std::uint64_t base        = 0; // pinned host memory, one block record at a time
    std::uint32_t first_block = 0;
    std::uint32_t count       = 0;
};

// Where the copy kernel takes the blocks the host copy does not hold (the lane's
// prefix, streamed tier): a device-readable table of segments and the record geometry.
// Every segment's source must stay valid for the lane's whole lifetime.
struct KvStreamPrefixSource {
    const KvStreamPrefixSegment* segments = nullptr;
    std::uint32_t segment_count           = 0;
    std::uint64_t record_stride           = 0;
    std::uint64_t plane_offset[4]         = {0, 0, 0, 0};
};

// See above. `selected` I32 [top_blocks, tokens], its width the indexer's budget, and `counts`
// I32 [tokens] are the indexer's blocks, `first_position` a device I32 word; `host` the host copy
// (identity layout), `pool` the device pool (its block_table the map's table). A block the host
// copy does not hold (the lane's prefix, streamed tier) names its source in `prefix` instead.
// Capturable: every size it launches with is fixed; the selection width is read from `selected`.
void kv_stream_resolve(const KvStreamMap& map, const Tensor& first_position,
                       const Tensor& selected, const Tensor& counts, const PagedKVLayerView& host,
                       const PagedKVLayerView& pool, const KvStreamPrefixSource& prefix,
                       cudaStream_t stream);
// The dense-tier form: every block the host copy does not hold is impossible, so the prefix
// source is empty.
inline void kv_stream_resolve(const KvStreamMap& map, const Tensor& first_position,
                              const Tensor& selected, const Tensor& counts,
                              const PagedKVLayerView& host, const PagedKVLayerView& pool,
                              cudaStream_t stream) {
    kv_stream_resolve(map, first_position, selected, counts, host, pool, {}, stream);
}

// Evicts the resident blocks among [first_block, end_block).
void kv_stream_evict(const KvStreamMap& map, std::int32_t first_block, std::int32_t end_block,
                     cudaStream_t stream);

// Before acceptance, withhold every draft of a lane whose pools overflowed this round. An
// acceptance with extent zero licenses column 0's target token and records accepted = 0 with the
// kernel's own lengths/anchors bookkeeping, which keeps the device-side hidden selection, scatter
// and next-round preparation in phase with the host. `round_flags` is I32 [lanes], `row_lanes`
// U32 [rows] maps batch rows to lanes, `current_extents` is I32 [rows].
void kv_stream_degrade_verify_extents(const std::int32_t* round_flags,
                                      const std::uint32_t* row_lanes,
                                      std::int32_t* current_extents, std::int32_t rows,
                                      cudaStream_t stream);

struct KvStreamCounters {
    std::uint64_t calls = 0, lookups = 0, copies = 0;
    bool overflow = false; // a call needed more slots than the pool has (never with the minimum)
};
// Waits for `stream`.
[[nodiscard]] KvStreamCounters kv_stream_counters(const KvStreamMap& map, cudaStream_t stream);

} // namespace ninfer::ops
