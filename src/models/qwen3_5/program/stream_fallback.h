#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// The (lane, layer) overflow words are staged into one bitmask of at most 32 lanes.
inline constexpr std::uint32_t kQsaStreamLaneMaskBits = 32;
// kv_stream's control[3] for the round just synchronized: 0 clean, 1 only draft columns' blocks
// were dropped, 2 column 0's shortlist could not be kept complete.
inline constexpr std::int32_t kQsaStreamOverflowDraftOnly = 1;
inline constexpr std::int32_t kQsaStreamOverflowUnsafe    = 2;

// Whether this lane's pools reported a recoverable draft-column overflow this round.
[[nodiscard]] constexpr bool
qsa_stream_lane_degraded(std::uint32_t mask, std::uint32_t lane) noexcept {
    return lane < kQsaStreamLaneMaskBits && (mask & (std::uint32_t{1} << lane)) != 0;
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

// The row metadata an MTP egress validator requires (`accepted + 1 == licensed`, and a next
// extent within the draft window). An overflowed round accepts no draft, licenses column 0's
// token alone and enters the next round without drafts.
struct MtpRowMetadata {
    std::int32_t accepted_drafts;
    std::int32_t licensed_counts;
    std::int32_t next_extents;
};

[[nodiscard]] constexpr MtpRowMetadata degrade_mtp_row() noexcept { return {0, 1, 0}; }

} // namespace ninfer::models::qwen3_5::detail
