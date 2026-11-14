// A cached media prompt resumes its prefill after the visual block, where the M-RoPE axes run
// ahead of the token indices. This oracle pins the two facts the continuation relies on:
// the axis count chosen by prefill, and `position = index + rope_delta` for every token at or
// after the end of a media run (frontend/processor.cpp:1126-1186).

#include "models/qwen3_5/execution/rope_axes.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using ninfer::models::qwen3_5::execution::prefill_rope_axes;

static_assert(prefill_rope_axes(false, 0) == 0);
static_assert(prefill_rope_axes(false, 2057) == 1);
static_assert(prefill_rope_axes(true, 0) == 3);
static_assert(prefill_rope_axes(true, 2057) == 3);

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct PositionTable {
    std::vector<std::int32_t> axes; // [3, length], axis-major
    std::size_t length              = 0;
    std::size_t media_end           = 0;
    std::size_t media_begin         = 0;
    std::int32_t rope_delta         = 0;
    int grid_width                  = 0;
};

// One media run of gh x gw merged tokens between two text spans, built by the same rules as
// assign_positions: text advances all three axes by one per token, a visual block freezes t and
// lays h/w out on a grid, and t jumps by max(gh, gw) at the block boundary.
PositionTable make_table(std::size_t prefix, int gh, int gw, std::size_t suffix) {
    const std::size_t length = prefix + static_cast<std::size_t>(gh) * gw + suffix;
    PositionTable table;
    table.axes.resize(3 * length, 0);
    table.length      = length;
    table.media_begin = prefix;
    table.grid_width  = gw;
    const auto set = [&](int axis, std::size_t index, std::int32_t value) {
        table.axes[static_cast<std::size_t>(axis) * length + index] = value;
    };
    std::int32_t current = 0;
    std::int32_t maximum = 0;
    const auto assign_text = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const std::int32_t position = current + static_cast<std::int32_t>(index - begin);
            for (int axis = 0; axis < 3; ++axis) { set(axis, index, position); }
            maximum = std::max(maximum, position);
        }
        current += static_cast<std::int32_t>(end - begin);
    };

    assign_text(0, prefix);
    std::size_t index = prefix;
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x, ++index) {
            set(0, index, current);
            set(1, index, current + y);
            set(2, index, current + x);
            maximum = std::max({maximum, current, current + y, current + x});
        }
    }
    current += std::max(gh, gw);
    table.media_end = index;
    assign_text(table.media_end, length);
    table.rope_delta = maximum + 1 - static_cast<std::int32_t>(length);
    return table;
}

std::int32_t position_at(const PositionTable& table, int axis, std::size_t index) {
    if (index < table.length) {
        return table.axes[static_cast<std::size_t>(axis) * table.length + index];
    }
    // Generated / continuation tokens keep advancing all three axes by one.
    return table.axes[static_cast<std::size_t>(axis) * table.length + table.length - 1] +
           static_cast<std::int32_t>(index - (table.length - 1));
}

} // namespace

int main() {
    int failures = 0;

    const PositionTable table = make_table(9, 32, 64, 9);
    failures += check(table.rope_delta != 0,
                      "a media grid must produce a non-zero rope_delta");

    // Inside the visual block the axes differ, so a cached segment that ends before the block
    // still needs the three-axis table.
    const std::size_t inner = table.media_begin + static_cast<std::size_t>(table.grid_width);
    failures += check(position_at(table, 1, inner) != position_at(table, 0, inner),
                      "a visual block must separate the h axis from the t axis");

    // At and after the visual block all three axes equal `index + rope_delta`. A prefix-append
    // prefill that starts here can therefore use one axis offset by io.rope_delta and stay
    // bit-identical to a cold prefill of the whole prompt.
    bool equivalent = true;
    for (std::size_t index = table.media_end; index < table.length + 16; ++index) {
        const std::int32_t expected = static_cast<std::int32_t>(index) + table.rope_delta;
        for (int axis = 0; axis < 3; ++axis) {
            if (position_at(table, axis, index) != expected) { equivalent = false; }
        }
    }
    failures += check(equivalent,
                      "post-media positions must equal index + rope_delta on all three axes");

    // A pure-text prompt has zero delta: dropping the offset tensor changes nothing.
    const PositionTable text_only = make_table(0, 0, 0, 16);
    bool text_flat = true;
    for (std::size_t index = 0; index < text_only.length; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            if (position_at(text_only, axis, index) != static_cast<std::int32_t>(index)) {
                text_flat = false;
            }
        }
    }
    failures += check(text_flat && text_only.rope_delta == 0,
                      "a text-only prompt must keep rope_delta == 0");

    if (failures == 0) { std::cout << "ok (prefill rope axes)\n"; }
    return failures == 0 ? 0 : 1;
}
