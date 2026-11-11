#pragma once

// --qsa-history-tokens, which ninfer, ninfer-serve and ninfer-perplexity share: how much
// history each Qwen3.8-27B QSA query looks back over.

#include "ninfer/ops/qsa_budget.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kQsaHistoryTokensHelp =
    "  --qsa-history-tokens N        Qwen3.8-27B with --qwen38-27b-qsa: how many\n"
    "                                history tokens each query may look back over,\n"
    "                                a multiple of 4 up to 8192; wider selections need\n"
    "                                a --kv-resident pool of more blocks\n"
    "                                (default: the artifact's own 2048)\n";

// Applies `option` when it is --qsa-history-tokens, taking its argument from `value()`; false
// leaves it to the caller. The engine refuses the flag without --qwen38-27b-qsa.
template <class Value>
bool parse_qsa_history_tokens_option(std::string_view option, std::uint32_t& out, Value&& value) {
    if (option != "--qsa-history-tokens") { return false; }
    const std::string text(value());
    std::uint32_t tokens = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), tokens);
    constexpr std::uint32_t kMaxTokens =
        static_cast<std::uint32_t>(ops::kQsaTopBlocksMax) * static_cast<std::uint32_t>(ops::kQsaBlockTokens);
    if (error != std::errc{} || end != text.data() + text.size() ||
        tokens % static_cast<std::uint32_t>(ops::kQsaBlockTokens) != 0 || tokens > kMaxTokens) {
        throw std::invalid_argument(
            "--qsa-history-tokens takes a multiple of 4 tokens, at most 8192");
    }
    out = tokens;
    return true;
}

} // namespace ninfer::product
