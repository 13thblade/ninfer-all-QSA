#pragma once

#include "core/host_kv_arena.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

// Host tier geometry of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §5.4). The whole Host budget is one pinned pool of equal slabs. One slab holds one KV block
// bundle: the Main text page packed with its Host page layout, then the backend page when a
// backend pool exists. A StateImage is split over `image_slabs` consecutive-in-image slabs and a
// snapshot tail bundle uses one more slab, so blocks and state images compete for the same bytes
// under the index's eviction policy instead of a fixed split.
// The streamed tier's block record: one packed 64-position page per streamed full-attention
// layer, in layer order, each with the layout of `text` (the layer's pinned host planes: K, V,
// K scale, V scale, skipping planes the storage has none of).
struct HybridStreamBlockLayout {
    std::uint32_t layers = 0;
};

struct HybridHostLayout {
    HostKVPageLayout text;
    std::optional<HostKVPageLayout> backend;
    std::size_t backend_offset = 0;
    std::size_t slab_bytes     = 0;
    // Bytes one block bundle actually transfers (the packed page records, without padding).
    std::size_t block_payload_bytes = 0;
    std::size_t image_bytes         = 0;
    std::uint32_t image_slabs       = 0;
    // Set when the block record is the per-layer streamed shape (HybridStreamBlockLayout);
    // empty for the ordinary single-Main-page record.
    std::optional<HybridStreamBlockLayout> stream;
};

[[nodiscard]] HybridHostLayout plan_hybrid_host_layout(const KVPageGeometry& text,
                                                       const KVPageGeometry* backend,
                                                       std::size_t image_bytes);
// The streamed variant of plan_hybrid_host_layout: `layers` streamed layers, each one
// 64-position page of `layer_page` (that layer's host planes), plus the backend page when
// present.
[[nodiscard]] HybridHostLayout plan_hybrid_stream_host_layout(std::uint32_t layers,
                                                              const KVPageGeometry& layer_page,
                                                              const KVPageGeometry* backend,
                                                              std::size_t image_bytes);

// Slabs a Host budget buys. A zero budget disables the Host tier. A nonzero budget must hold at
// least one complete snapshot (image and tail) and one block, or it is rejected: a smaller tier
// could never back a snapshot.
[[nodiscard]] std::uint32_t hybrid_host_slabs(const HybridHostLayout& layout,
                                              std::uint64_t budget_bytes);

} // namespace ninfer::models::qwen3_5::detail
