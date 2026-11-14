// Host-only oracle for the QSA streamed-KV overflow fallback: the per-lane overflow words folded
// into a round mask, and the MTP row metadata a degraded round must publish. The kernel's
// column-0 placement is exercised separately by tests/ops/test_kv_stream.cpp.

#include "models/qwen3_5/program/stream_fallback.h"

#include <array>
#include <cstdint>
#include <iostream>

namespace {

using ninfer::models::qwen3_5::detail::degrade_mtp_row;
using ninfer::models::qwen3_5::detail::fold_qsa_stream_lane;
using ninfer::models::qwen3_5::detail::kQsaStreamOverflowDraftOnly;
using ninfer::models::qwen3_5::detail::kQsaStreamOverflowUnsafe;
using ninfer::models::qwen3_5::detail::qsa_stream_lane_degraded;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    std::uint32_t mask = 0;
    constexpr std::array<std::int32_t, 4> clean{0, 0, 0, 0};
    failures += check(fold_qsa_stream_lane(clean.data(), 4, 0, mask) && mask == 0,
                      "clean layers leave the round mask clear");

    constexpr std::array<std::int32_t, 4> draft_only{0, kQsaStreamOverflowDraftOnly, 0, 0};
    failures += check(fold_qsa_stream_lane(draft_only.data(), 4, 3, mask) &&
                          mask == (std::uint32_t{1} << 3),
                      "a draft-only overflow marks just its lane for degradation");
    failures += check(qsa_stream_lane_degraded(mask, 3), "lane 3 degrades");
    failures += check(!qsa_stream_lane_degraded(mask, 4), "lane 4 does not");

    // A dirty column 0 shortlist must never be folded into a degradable mask, even when another
    // layer of the same lane reported only a draft-column overflow.
    constexpr std::array<std::int32_t, 4> unsafe{kQsaStreamOverflowDraftOnly, 0, 0,
                                                 kQsaStreamOverflowUnsafe};
    std::uint32_t unsafe_mask = 0;
    failures += check(!fold_qsa_stream_lane(unsafe.data(), 4, 1, unsafe_mask),
                      "a dirty column 0 shortlist cannot be degraded in place");
    failures += check(!qsa_stream_lane_degraded(~std::uint32_t{0}, 32),
                      "a lane beyond 32 bits is never degraded");

    // The egress validator requires accepted + 1 == licensed and a next extent within the draft
    // window; a degraded round licenses column 0 alone and enters the next round without drafts.
    constexpr auto row = degrade_mtp_row();
    static_assert(row.accepted_drafts == 0 && row.licensed_counts == 1 &&
                  row.next_extents == 0);
    failures += check(row.accepted_drafts + 1 == row.licensed_counts &&
                          row.next_extents >= 0,
                      "the degraded row metadata satisfies the MTP egress validator");

    if (failures != 0) { std::cerr << failures << " check(s) failed\n"; }
    return failures != 0 ? 1 : 0;
}
