#pragma once

// The lane <-> image mapping for the QSA continuation store (R4).
//
// The streamed tier's KV authority is the lane's pinned host planes (`QsaStreamState.host`), one
// `[leading, 128, heads, pages]` tensor per plane per streamed layer. An image keeps them whole:
// the page is the tensors' innermost axis, so one position's pages are strided and cannot be
// sliced off with a single memcpy. A whole-plane copy is one memcpy per plane, round-trips
// exactly, and costs the lane's full capacity -- the budget arithmetic (qsa_image_bytes) stays a
// separate, position-parametric model of the *layout*, not of this snapshot.

#include "models/qwen3_5/program/prefix/qsa_continuation.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// One lane's authoritative KV as image chunks, in a fixed order: layer by layer, and within a
// layer the planes the storage has (K, V, K scale, V scale), skipping the absent ones. Saving and
// loading both walk this order, so an image built by one build restores byte-for-byte in the same
// build (a storage change is a new image, which the store's budget never mixes).
void qsa_collect_plane_chunks(const QsaStreamState& stream, std::uint32_t lane,
                              std::vector<ImageChunk>& out);

// Bytes one lane's planes occupy in an image.
[[nodiscard]] std::uint64_t qsa_lane_plane_bytes(const QsaStreamState& stream, std::uint32_t lane);

} // namespace ninfer::models::qwen3_5::detail
