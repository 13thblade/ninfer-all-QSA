#pragma once

// The lane <-> image mapping for the QSA continuation store (R4).
//
// The streamed tier's KV authority is the lane's pinned host planes (`QsaStreamState.host`), one
// `[leading, kPagedKVPageSize, heads, pages]` tensor per plane per streamed layer. `ne[0]` is the
// innermost axis here (core/tensor.cpp:50), so a plane's pages are its *outermost* run: page p is
// one contiguous block of `leading * 64 * heads * dtypesize` bytes, logical page p at block p
// (ops/kv_stream.h:21, and the copy kernel's host address `((page * heads + head) * 64 + within)`).
// An image therefore keeps a plain prefix of each plane -- the pages the key reached -- and needs
// no strided gather: at a full window the prefix is the whole plane.

#include "models/qwen3_5/program/prefix/qsa_continuation.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// One lane's authoritative KV as image chunks, in a fixed order: layer by layer, and within a
// layer the planes the storage has (K, V, K scale, V scale), skipping the absent ones. Saving and
// loading both walk this order, so an image built by one build restores byte-for-byte in the same
// build (a storage change is a new image, which the store's budget never mixes).
void qsa_collect_plane_chunks(const QsaStreamState& stream, std::uint32_t lane,
                              std::vector<ImageChunk>& out);

// The same, for an image whose key is `position` tokens long: every chunk holds only the
// `ceil(position / kPagedKVPageSize)` whole pages its plane reached, not the lane's whole
// capacity. `position` is the image's key length and never the lane's live frontier -- a restore
// reads the length the image was written at, and the lane's pages beyond it belong to whoever
// used the lane before. Saving and loading must be called with the same length, which is what
// keeps the two directions byte-symmetric.
//
// The last page is copied whole even when the key stops inside it: a page is the smallest unit a
// plane holds, and the bytes past the key in that page are never read back (text_kv_valid lands
// on the key). A position past the lane's capacity clamps to the whole plane.
void qsa_collect_plane_prefix_chunks(const QsaStreamState& stream, std::uint32_t lane,
                                     std::uint32_t position, std::vector<ImageChunk>& out);

// Bytes one lane's planes occupy in an image.
[[nodiscard]] std::uint64_t qsa_lane_plane_bytes(const QsaStreamState& stream, std::uint32_t lane);

} // namespace ninfer::models::qwen3_5::detail
