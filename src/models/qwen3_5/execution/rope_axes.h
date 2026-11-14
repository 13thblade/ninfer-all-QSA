#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// Cold multimodal prefill reads the prompt's [3, T] M-RoPE table. A pure-text prefill starting
// at position 0 needs no offset tensor: the token index is the position on every axis. A
// prefix-append prefill resumes after cached media tokens, whose grid advanced the axes past
// the token indices; its continuation needs one axis holding `token index + io.rope_delta`.
constexpr std::int32_t prefill_rope_axes(bool multimodal, std::uint32_t base) noexcept {
    return multimodal ? 3 : (base != 0 ? 1 : 0);
}

} // namespace ninfer::models::qwen3_5::execution
