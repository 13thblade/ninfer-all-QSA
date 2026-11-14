// Host-only oracle for the QSA streamed-KV fallback: the per-lane overflow words folded into a
// round mask, the K/V-only gate under streaming, and the device-side degradation rules. The
// kernel's column-0 placement and shortlist truncation are exercised separately by
// tests/ops/test_kv_stream.cpp.

#include "models/qwen3_5/program/stream_fallback.h"

#include <array>
#include <cstdint>
#include <iostream>

namespace {

using ninfer::models::qwen3_5::detail::fold_qsa_stream_lane;
using ninfer::models::qwen3_5::detail::kQsaStreamOverflowDraftOnly;
using ninfer::models::qwen3_5::detail::kQsaStreamOverflowUnsafe;
using ninfer::models::qwen3_5::detail::kv_only_pass_supported;
using ninfer::models::qwen3_5::detail::qsa_stream_degraded_extent;
using ninfer::models::qwen3_5::detail::qsa_stream_worst_code;

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
                      "a draft-only overflow marks just its lane");
    failures += check((mask & (std::uint32_t{1} << 3)) != 0 &&
                          (mask & (std::uint32_t{1} << 4)) == 0,
                      "the mask names lane 3 and no other lane");

    // A dirty column 0 shortlist must never be folded into a degradable mask, even when another
    // layer of the same lane reported only a draft-column overflow.
    constexpr std::array<std::int32_t, 4> unsafe{kQsaStreamOverflowDraftOnly, 0, 0,
                                                 kQsaStreamOverflowUnsafe};
    std::uint32_t unsafe_mask = 0;
    failures += check(!fold_qsa_stream_lane(unsafe.data(), 4, 1, unsafe_mask),
                      "a dirty column 0 shortlist cannot be degraded in place");

    // The K/V-only fast path writes the last layer's rows into the device text-KV view. With the
    // authoritative copy on pinned host memory and a single shared staging layer, it is invalid.
    static_assert(kv_only_pass_supported(false));
    static_assert(!kv_only_pass_supported(true));

    // An overflowed lane offers no drafts: the acceptance kernel then licenses column 0's target
    // token with its own lengths/anchors/selected-hidden bookkeeping. Clean lanes keep their extent.
    static_assert(qsa_stream_degraded_extent(0, 3) == 3);
    static_assert(qsa_stream_degraded_extent(kQsaStreamOverflowDraftOnly, 3) == 0);
    static_assert(qsa_stream_degraded_extent(kQsaStreamOverflowUnsafe, 1) == 0);
    static_assert(qsa_stream_worst_code(0, 0) == 0);
    static_assert(qsa_stream_worst_code(kQsaStreamOverflowDraftOnly, 0) ==
                  kQsaStreamOverflowDraftOnly);
    static_assert(qsa_stream_worst_code(kQsaStreamOverflowDraftOnly,
                                        kQsaStreamOverflowUnsafe) ==
                  kQsaStreamOverflowUnsafe);
    failures += check(qsa_stream_worst_code(0, kQsaStreamOverflowDraftOnly) ==
                          kQsaStreamOverflowDraftOnly,
                      "a lane keeps the worst overflow code of its layers");

    if (failures != 0) { std::cerr << failures << " check(s) failed\n"; }
    return failures != 0 ? 1 : 0;
}
