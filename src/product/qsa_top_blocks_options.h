#pragma once

// --qsa-top-blocks, which ninfer, ninfer-serve and ninfer-perplexity share: how many history
// blocks (four tokens each) each Qwen3.8-27B QSA query may look back over. The flag counts
// blocks so that it speaks the indexer's own unit and needs no conversion at the boundary.

#include "ninfer/ops/qsa_budget.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kQsaTopBlocksHelp =
    "  --qsa-top-blocks N            Qwen3.8-27B with --qwen38-27b-qsa: how many\n"
    "                                history blocks (4 tokens each) each query may\n"
    "                                look back over, 1 to 65536; wider selections\n"
    "                                need a --kv-resident pool of more blocks\n"
    "                                (default: the artifact's own 512 blocks)\n";

// Applies `option` when it is --qsa-top-blocks, taking its argument from `value()`; false
// leaves it to the caller. The engine refuses the flag without --qwen38-27b-qsa.
template <class Value>
bool parse_qsa_top_blocks_option(std::string_view option, std::uint32_t& out, Value&& value) {
    if (option != "--qsa-top-blocks") { return false; }
    const std::string text(value());
    std::uint32_t blocks = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), blocks);
    constexpr std::uint32_t kMaxBlocks = static_cast<std::uint32_t>(ops::kQsaTopBlocksMax);
    if (error != std::errc{} || end != text.data() + text.size() || blocks > kMaxBlocks) {
        throw std::invalid_argument(
            "--qsa-top-blocks takes a block count, at most " + std::to_string(kMaxBlocks) +
            " (each block is 4 tokens)");
    }
    out = blocks;
    return true;
}

} // namespace ninfer::product
