#include "models/qwen3_5/program/prefix/qsa_image.h"

#include "core/paged_kv_page.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

namespace {

constexpr std::size_t kAllPages = std::numeric_limits<std::size_t>::max();

// Bytes of one whole 64-position page of `plane`: the pages axis is the plane's outermost
// contiguous run, so its stride is exactly one page's payload and `pages` of them are the plane.
std::size_t plane_page_bytes(const Tensor& plane) {
    const std::size_t pages = static_cast<std::size_t>(plane.ne[3]);
    const std::size_t block = static_cast<std::size_t>(plane.nb[3]);
    if (pages == 0 || block == 0 || block > plane.bytes() || block * pages != plane.bytes()) {
        throw std::logic_error("qsa image: a host KV plane is not page-major");
    }
    return block;
}

// Appends the four planes of one streamed layer, in image order, each carrying `pages` whole
// pages (kAllPages = the plane's whole extent). Absent planes (a storage without scales) are
// skipped, which is what keeps an image's chunk order independent of the storage's plane count.
void append_layer_planes(const QsaStreamLayerState& state, std::size_t pages,
                         std::vector<ImageChunk>& out) {
    const Tensor* planes[4] = {&state.host.k_pages, &state.host.v_pages,
                               &state.host.k_scale_pages, &state.host.v_scale_pages};
    for (const Tensor* plane : planes) {
        if (plane->data == nullptr || plane->numel() == 0) { continue; }
        const std::size_t block = plane_page_bytes(*plane);
        const std::size_t take  = std::min(pages, static_cast<std::size_t>(plane->ne[3]));
        if (take == 0) { continue; }
        out.push_back(ImageChunk{static_cast<std::byte*>(plane->data), take * block});
    }
}

void collect(const QsaStreamState& stream, std::uint32_t lane, std::size_t pages,
             std::vector<ImageChunk>& out) {
    out.clear();
    if (pages == 0) { return; }
    if (lane >= stream.lanes) { throw std::out_of_range("qsa image lane is out of range"); }
    if (stream.layer_states.size() < static_cast<std::size_t>(stream.lanes) * stream.layers) {
        throw std::logic_error("qsa image: the lane states are not allocated");
    }
    out.reserve(static_cast<std::size_t>(stream.layers) * 4U);
    for (std::uint32_t layer = 0; layer < stream.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(lane) * stream.layers + layer;
        append_layer_planes(stream.layer_states[index], pages, out);
    }
}

} // namespace

void qsa_collect_plane_chunks(const QsaStreamState& stream, std::uint32_t lane,
                              std::vector<ImageChunk>& out) {
    collect(stream, lane, kAllPages, out);
}

void qsa_collect_plane_prefix_chunks(const QsaStreamState& stream, std::uint32_t lane,
                                     std::uint32_t position, std::vector<ImageChunk>& out) {
    if (position == 0) {
        out.clear();
        return;
    }
    const std::size_t used =
        (static_cast<std::size_t>(position) + ninfer::kPagedKVPageSize - 1) /
        ninfer::kPagedKVPageSize;
    collect(stream, lane, used, out);
}

std::uint64_t qsa_lane_plane_bytes(const QsaStreamState& stream, std::uint32_t lane) {
    std::vector<ImageChunk> chunks;
    qsa_collect_plane_chunks(stream, lane, chunks);
    return qsa_image_chunk_bytes(chunks);
}

} // namespace ninfer::models::qwen3_5::detail
