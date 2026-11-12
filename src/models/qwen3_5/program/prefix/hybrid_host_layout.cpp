#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Slab records start on page boundaries so every DMA run begins aligned.
constexpr std::size_t kSlabAlignment = 4096;

std::size_t align_slab(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - (kSlabAlignment - 1U)) {
        throw std::overflow_error("hybrid Host slab size overflows");
    }
    return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment;
}

} // namespace

HybridHostLayout plan_hybrid_host_layout(const KVPageGeometry& text, const KVPageGeometry* backend,
                                         std::size_t image_bytes) {
    if (image_bytes == 0) { throw std::invalid_argument("hybrid Host layout has no StateImage"); }
    HybridHostLayout out;
    out.text                = plan_host_kv_page_layout(text);
    std::size_t bytes       = out.text.page_stride;
    out.block_payload_bytes = out.text.page_stride;
    if (backend != nullptr) {
        out.backend        = plan_host_kv_page_layout(*backend);
        out.backend_offset = align_slab(bytes);
        bytes              = out.backend_offset + out.backend->page_stride;
        out.block_payload_bytes += out.backend->page_stride;
    }
    out.slab_bytes          = align_slab(bytes);
    out.image_bytes         = image_bytes;
    const std::size_t slabs = 1U + (image_bytes - 1U) / out.slab_bytes;
    if (slabs > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("hybrid StateImage slab count exceeds uint32");
    }
    out.image_slabs = static_cast<std::uint32_t>(slabs);
    return out;
}

// The streamed variant: one packed layer page per streamed layer, then the backend page when a
// backend pool exists. The layer pages are laid out consecutively in the record (layer 0 first),
// so a layer's page starts at layer * text.page_stride in its slab.
HybridHostLayout plan_hybrid_stream_host_layout(std::uint32_t layers, const KVPageGeometry& layer_page,
                                                const KVPageGeometry* backend,
                                                std::size_t image_bytes) {
    if (image_bytes == 0) { throw std::invalid_argument("hybrid Host layout has no StateImage"); }
    if (layers == 0 ||
        layer_page.page_tokens != static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("hybrid stream layer page geometry is invalid");
    }
    HybridHostLayout out;
    out.text = plan_host_kv_page_layout(layer_page);
    out.stream = HybridStreamBlockLayout{.layers = layers};
    std::size_t bytes = static_cast<std::size_t>(layers) * out.text.page_stride;
    out.block_payload_bytes = bytes;
    if (backend != nullptr) {
        out.backend        = plan_host_kv_page_layout(*backend);
        out.backend_offset = align_slab(bytes);
        bytes              = out.backend_offset + out.backend->page_stride;
        out.block_payload_bytes += out.backend->page_stride;
    }
    out.slab_bytes          = align_slab(bytes);
    out.image_bytes         = image_bytes;
    const std::size_t slabs = 1U + (image_bytes - 1U) / out.slab_bytes;
    if (slabs > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("hybrid StateImage slab count exceeds uint32");
    }
    out.image_slabs = static_cast<std::uint32_t>(slabs);
    return out;
}

std::uint32_t hybrid_host_slabs(const HybridHostLayout& layout, std::uint64_t budget_bytes) {
    if (budget_bytes == 0) { return 0; }
    if (layout.slab_bytes == 0) { throw std::logic_error("hybrid Host layout is empty"); }
    const std::uint64_t slabs   = budget_bytes / layout.slab_bytes;
    const std::uint64_t minimum = static_cast<std::uint64_t>(layout.image_slabs) + 2U;
    if (slabs < minimum) {
        const std::uint64_t minimum_mib = (minimum * layout.slab_bytes + (1ULL << 20U) - 1U) >> 20U;
        throw std::invalid_argument(
            "--host-cache-mib is too small for the hybrid prefix cache: one state snapshot needs " +
            std::to_string(minimum_mib) + " MiB; use at least that or 0 to disable the Host tier");
    }
    if (slabs > std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw std::overflow_error("hybrid Host slab count is not representable");
    }
    return static_cast<std::uint32_t>(slabs);
}

} // namespace ninfer::models::qwen3_5::detail
