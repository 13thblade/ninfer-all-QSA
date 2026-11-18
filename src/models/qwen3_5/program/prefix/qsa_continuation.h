#pragma once

// QSA session-internal KV continuation (aligned with upstream qwen4_exp's three-way resume).
//
// This header holds the small, dependency-free pieces shared by the recording, the image store
// (S2+) and the admission (S4+):
//   * HeldPrefix  - the exact token prefix a lane has already written into its KV, plus the media
//                   identity that went with those tokens. One per lane; a pure record.
//   * qsa_prefix_reuse - the upstream `prefix_reuse` semantics, copied rather than abstracted.
//   * qsa_reuse_trace  - NINFER_DEBUG_REUSE-gated stderr logging used to observe live/anchored.
//
// Nothing here changes admission yet: S1 only records and logs.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// What one lane currently holds in its authoritative (pinned host) KV. `tokens` are the prompt
// tokens written up to `position`; `vision_key` is the accumulated Vision key of the media items
// that fall inside that prefix (zero for a text-only prompt). This mirrors upstream's
// `slot.fed` / `slot.anchor` pair together with their `media` vector, condensed to a key because
// this fork's PreparedPrompt carries media identity as a cumulative per-block Vision key.
struct HeldPrefix {
    std::vector<TokenId> tokens;
    std::uint64_t vision_key = 0;
    std::uint32_t position   = 0; // == tokens.size()

    [[nodiscard]] bool empty() const noexcept { return tokens.empty(); }
    void clear() noexcept {
        tokens.clear();
        vision_key = 0;
        position   = 0;
    }
};

[[nodiscard]] inline bool qsa_reuse_trace_enabled() noexcept {
    return std::getenv("NINFER_DEBUG_REUSE") != nullptr;
}

// Copied 1:1 in semantics from upstream qwen4_exp_core.cpp:1213 (prefix_reuse):
// `held` can start a prompt only when it is a strict prefix of it (the prompt's last token is
// always fed, so it is not part of the reusable prefix), or the whole prompt for a branched read
// (`whole`), and when the media identity of the prefix matches. Returns held.size() or 0.
//
// The media check is a single accumulated key here; `prefix_matches()` on the full
// ResidentPrefixIdentity remains the authoritative check in the engine paths that have one.
[[nodiscard]] inline std::uint32_t
qsa_prefix_reuse(const HeldPrefix& held, std::span<const TokenId> prompt_tokens,
                 std::uint64_t prompt_vision_key, bool whole = false) {
    if (held.empty() || prompt_tokens.empty() ||
        held.tokens.size() > prompt_tokens.size() - (whole ? 0U : 1U) ||
        !std::equal(held.tokens.begin(), held.tokens.end(), prompt_tokens.begin())) {
        return 0;
    }
    if (held.vision_key != prompt_vision_key) { return 0; }
    return static_cast<std::uint32_t>(held.tokens.size());
}

// The cumulative Vision key a prompt carries at `position` tokens. With no Vision items this is
// zero, matching a text-only HeldPrefix. `block_extras` holds one cumulative key per full
// 64-token block (frontend/prepared_prompt.h:166); the key of the block containing `position`
// covers the media items up to that block's end, which is the identity of the prefix.
[[nodiscard]] inline std::uint64_t qsa_vision_key_at(const PreparedPromptData& prompt,
                                                     std::size_t position) {
    if (position == 0 || prompt.block_extras.empty() || prompt.vision_items.empty()) { return 0; }
    constexpr std::size_t kBlock = 64;
    const std::size_t block      = (position - 1) / kBlock;
    return block < prompt.block_extras.size() ? prompt.block_extras[block] : prompt.block_extras.back();
}

// One observability line per admission, gated by NINFER_DEBUG_REUSE. `chosen` names the resume
// path; S1 always records "root" because the store does not exist yet.
inline void qsa_reuse_trace(std::uint32_t lane, std::uint32_t live, std::uint32_t anchored,
                            std::uint32_t store, const char* chosen, std::uint32_t scanned = 0,
                            std::uint32_t max_held = 0) {
    if (!qsa_reuse_trace_enabled()) { return; }
    std::fprintf(stderr,
                 "[resume] lane=%u live=%u anchored=%u store=%u chosen=%s scanned=%u max_held=%u\n",
                 lane, live, anchored, store, chosen, scanned, max_held);
    std::fflush(stderr);
}

} // namespace ninfer::models::qwen3_5::detail
