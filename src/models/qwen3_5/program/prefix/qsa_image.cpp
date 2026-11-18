#include "models/qwen3_5/program/prefix/qsa_image.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

namespace {

// The four planes of one streamed layer, in image order.
void append_layer_planes(const QsaStreamLayerState& state, std::vector<ImageChunk>& out) {
    const Tensor* planes[4] = {&state.host.k_pages, &state.host.v_pages,
                               &state.host.k_scale_pages, &state.host.v_scale_pages};
    for (const Tensor* plane : planes) {
        if (plane->data == nullptr || plane->numel() == 0) { continue; }
        out.push_back(ImageChunk{static_cast<std::byte*>(plane->data), plane->bytes()});
    }
}

} // namespace

void qsa_collect_plane_chunks(const QsaStreamState& stream, std::uint32_t lane,
                              std::vector<ImageChunk>& out) {
    if (lane >= stream.lanes) { throw std::out_of_range("qsa image lane is out of range"); }
    if (stream.layer_states.size() < static_cast<std::size_t>(stream.lanes) * stream.layers) {
        throw std::logic_error("qsa image: the lane states are not allocated");
    }
    out.clear();
    out.reserve(static_cast<std::size_t>(stream.layers) * 4U);
    for (std::uint32_t layer = 0; layer < stream.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(lane) * stream.layers + layer;
        append_layer_planes(stream.layer_states[index], out);
    }
}

std::uint64_t qsa_lane_plane_bytes(const QsaStreamState& stream, std::uint32_t lane) {
    std::vector<ImageChunk> chunks;
    qsa_collect_plane_chunks(stream, lane, chunks);
    return qsa_image_chunk_bytes(chunks);
}

} // namespace ninfer::models::qwen3_5::detail
