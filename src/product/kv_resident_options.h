#pragma once

// --kv-resident, which ninfer, ninfer-serve and ninfer-perplexity share: how much of each
// Qwen3.8-Flash-Next sparse-attention layer's KV stays in device memory.

#include <charconv>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kKvResidentHelp =
    "  --kv-resident N|all|auto      Qwen3.8-Flash-Next, or Qwen3.8-27B with\n"
    "                                --qwen38-27b-qsa: positions of each sparse-\n"
    "                                attention layer's KV a sequence keeps on the GPU;\n"
    "                                a longer context keeps that KV in pinned RAM and\n"
    "                                the GPU the blocks it reads. auto follows the\n"
    "                                upstream preset (device experts: all; host/disk\n"
    "                                experts: 32768). QSA needs one device and a\n"
    "                                verification window <= 15 drafts\n";

// Applies `option` when it is --kv-resident, taking its argument from `value()`; false leaves it
// to the caller.
template <class Value>
bool parse_kv_resident_option(std::string_view option, std::optional<std::uint32_t>& out,
                              Value&& value) {
    if (option != "--kv-resident") { return false; }
    const std::string text(value());
    if (text == "all") {
        out = 0;
    } else if (text == "auto") {
        out.reset();
    } else {
        std::uint32_t positions = 0;
        const auto [end, error] =
            std::from_chars(text.data(), text.data() + text.size(), positions);
        if (error != std::errc{} || end != text.data() + text.size() || positions == 0) {
            throw std::invalid_argument("--kv-resident takes a positive position count, all or "
                                        "auto");
        }
        out = positions;
    }
    return true;
}

} // namespace ninfer::product
