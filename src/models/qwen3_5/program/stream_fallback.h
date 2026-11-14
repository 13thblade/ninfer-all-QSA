#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// The (lane, layer) overflow words are staged into one bitmask of at most 32 lanes.
inline constexpr std::uint32_t kQsaStreamLaneMaskBits = 32;
// kv_stream's control[3] for the round just synchronized: 0 clean, 1 only draft columns' blocks
// were dropped, 2 column 0's shortlist could not be kept complete.
inline constexpr std::int32_t kQsaStreamOverflowDraftOnly = 1;
inline constexpr std::int32_t kQsaStreamOverflowUnsafe    = 2;

// A K/V-only pass appends the last layer's rows straight into the device text-KV view. Under
// --kv-resident the authoritative copy is the pinned host layer and the device text cache has a
// single shared staging layer, so the pass has neither the layers it indexes nor the right
// destination. The prefill planner turns it off at its only entry point.
[[nodiscard]] constexpr bool kv_only_pass_supported(bool qsa_stream_active) noexcept {
    return !qsa_stream_active;
}

// The device-side fallback: an overflowed lane offers no drafts to the acceptance kernel, so it
// commits column 0 alone with the kernel's own complete bookkeeping.
[[nodiscard]] constexpr std::int32_t
qsa_stream_degraded_extent(std::int32_t round_flag, std::int32_t extent) noexcept {
    return round_flag != 0 ? 0 : extent;
}

// Every (lane, layer) pool reports its own overflow word; a round keeps the worst one.
[[nodiscard]] constexpr std::int32_t
qsa_stream_worst_code(std::int32_t current, std::int32_t word) noexcept {
    return word > current ? word : current;
}

// Fold one lane's per-layer overflow words into the round mask. Returns false when a layer says
// column 0 could not be kept complete: no in-place degradation can recover from that.
[[nodiscard]] constexpr bool
fold_qsa_stream_lane(const std::int32_t* words, std::uint32_t layers, std::uint32_t lane,
                     std::uint32_t& mask) noexcept {
    bool safe     = true;
    bool degraded = false;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::int32_t word = words[layer];
        if (word >= kQsaStreamOverflowUnsafe) {
            safe = false;
        } else if (word != 0) {
            degraded = true;
        }
    }
    if (degraded) { mask |= std::uint32_t{1} << lane; }
    return safe;
}

} // namespace ninfer::models::qwen3_5::detail
