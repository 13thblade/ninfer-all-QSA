// Host layout oracle for the streamed tier of the hybrid prefix cache
// (docs/maintainer/hybrid-prefix-cache-spec.md §5.4): one block record is one packed
// 64-position page per streamed layer, in layer order, then the backend page when a backend
// pool exists. Every expectation below is computed by hand from the packing rules: planes are
// 256-byte aligned inside a page, the layer pages run consecutively in the record, the slab is
// 4096-byte aligned, and a snapshot spans image_bytes / slab_bytes + 1 slabs.

#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::DType;
using ninfer::KVPageGeometry;
using ninfer::KVPlaneGeometry;
using ninfer::models::qwen3_5::detail::HybridHostLayout;
using ninfer::models::qwen3_5::detail::hybrid_host_slabs;
using ninfer::models::qwen3_5::detail::plan_hybrid_stream_host_layout;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // A BF16 layer page: two planes of 4 heads by 128 dims. One plane is 4 * 64 * 2 = 512 bytes
    // per head, 512 * 128 = 65536 bytes per page; K@0, V@65536, page stride 131072. Two streamed
    // layers with no backend make a slab of 2 * 131072 = 262144 bytes, already 4096-aligned, and
    // a 1 MiB image spans 1 + 1048575 / 262144 = 4 slabs.
    {
        KVPageGeometry layer;
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::BF16, .leading_extent = 4,
                                               .head_extent = 128});
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::BF16, .leading_extent = 4,
                                               .head_extent = 128});
        const HybridHostLayout layout = plan_hybrid_stream_host_layout(2, layer, nullptr, 1ULL << 20);
        failures += check(layout.stream.has_value() && layout.stream->layers == 2,
                          "the streamed tier did not record its layer count");
        failures += check(layout.text.planes.size() == 2 &&
                              layout.text.planes[0].offset == 0 &&
                              layout.text.planes[0].page_payload_bytes == 65536 &&
                              layout.text.planes[1].offset == 65536 &&
                              layout.text.planes[1].page_payload_bytes == 65536 &&
                              layout.text.page_stride == 131072,
                          "the layer page did not pack the two BF16 planes");
        failures += check(layout.backend == std::nullopt &&
                              layout.block_payload_bytes == 262144 && layout.slab_bytes == 262144 &&
                              layout.image_slabs == 4,
                          "the no-backend slab and image split did not match the hand arithmetic");
        failures += check(hybrid_host_slabs(layout, 0) == 0,
                          "a zero budget must disable the streamed Host tier");
        const std::uint64_t slab = layout.slab_bytes;
        failures += check(hybrid_host_slabs(layout, 6 * slab) == 6,
                          "the minimum streamed budget did not buy one snapshot and a block");
        failures += check(hybrid_host_slabs(layout, 7 * slab - 1) == 6,
                          "a streamed Host budget must buy whole slabs");
        bool rejected = false;
        try {
            (void)hybrid_host_slabs(layout, 6 * slab - 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "a streamed budget below one snapshot was accepted");
    }

    // An int8 layer page with FP32 scales: K and V are 8 heads by 128 dims (65536 bytes a page),
    // and each scale plane is 8 * 64 * 4 = 2048 bytes per head, 2048 * 2 = 4096 bytes a page.
    // K@0, V@65536, K scales@131072, V scales@135168, page stride 139264. Three layers with a
    // BF16 backend page (stride 131072): the layer pages end at 3 * 139264 = 417792, already
    // 4096-aligned, so the backend starts there and the slab is 417792 + 131072 = 548864 bytes;
    // a 2 MiB image spans 1 + 2097151 / 548864 = 4 slabs.
    {
        KVPageGeometry layer;
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::I8, .leading_extent = 8,
                                               .head_extent = 128});
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::I8, .leading_extent = 8,
                                               .head_extent = 128});
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::FP32, .leading_extent = 8,
                                               .head_extent = 2});
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::FP32, .leading_extent = 8,
                                               .head_extent = 2});
        KVPageGeometry backend;
        backend.planes.push_back(KVPlaneGeometry{.dtype = DType::BF16, .leading_extent = 4,
                                                 .head_extent = 128});
        backend.planes.push_back(KVPlaneGeometry{.dtype = DType::BF16, .leading_extent = 4,
                                                 .head_extent = 128});
        const HybridHostLayout layout = plan_hybrid_stream_host_layout(3, layer, &backend, 2ULL << 20);
        failures += check(layout.stream.has_value() && layout.stream->layers == 3,
                          "the streamed tier did not record its layer count");
        failures += check(layout.text.planes.size() == 4 &&
                              layout.text.planes[0].offset == 0 &&
                              layout.text.planes[0].page_payload_bytes == 65536 &&
                              layout.text.planes[1].offset == 65536 &&
                              layout.text.planes[1].page_payload_bytes == 65536 &&
                              layout.text.planes[2].offset == 131072 &&
                              layout.text.planes[2].page_payload_bytes == 4096 &&
                              layout.text.planes[3].offset == 135168 &&
                              layout.text.planes[3].page_payload_bytes == 4096 &&
                              layout.text.page_stride == 139264,
                          "the int8 layer page did not pack the four planes");
        failures += check(layout.backend.has_value() && layout.backend->page_stride == 131072 &&
                              layout.backend_offset == 417792 &&
                              layout.block_payload_bytes == 548864 && layout.slab_bytes == 548864 &&
                              layout.image_slabs == 4,
                          "the int8 slab with backend did not match the hand arithmetic");
    }

    // Geometries that break the streamed shape are rejected: zero layers, a page that is not a
    // 64-position block, and no StateImage.
    for (int variant = 0; variant < 3; ++variant) {
        KVPageGeometry layer;
        layer.planes.push_back(KVPlaneGeometry{.dtype = DType::BF16, .leading_extent = 1,
                                               .head_extent = 1});
        bool rejected = false;
        try {
            if (variant == 0) { (void)plan_hybrid_stream_host_layout(0, layer, nullptr, 1); }
            if (variant == 1) {
                layer.page_tokens = 128;
                (void)plan_hybrid_stream_host_layout(1, layer, nullptr, 1);
            }
            if (variant == 2) { (void)plan_hybrid_stream_host_layout(1, layer, nullptr, 0); }
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an invalid streamed geometry was accepted");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
