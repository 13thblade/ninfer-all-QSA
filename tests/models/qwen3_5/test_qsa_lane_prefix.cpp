// Host-only checks for the QSA lane record (R1): the per-lane held/anchor pair a three-way resume
// compares against. No GPU, no model artifact -- the record is pure bookkeeping over a token
// history, mirroring upstream qwen4_exp_core.cpp's slot.fed / slot.anchor / media.

#include "models/qwen3_5/program/prefix/qsa_continuation.h"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::detail::LanePrefixRecord;
using ninfer::models::qwen3_5::detail::qsa_media_key_at;
using ninfer::models::qwen3_5::detail::qsa_prefix_reuse;
using ninfer::models::qwen3_5::detail::qsa_set_prefix_frontier;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<TokenId> tokens_of(int seed, std::size_t count) {
    std::vector<TokenId> tokens;
    tokens.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens.push_back(static_cast<TokenId>(seed * 100000 + i));
    }
    return tokens;
}

// A record as the Program builds one: the prompt's tokens as the anchor, its per-block media keys,
// and the held prefix at whatever the KV already covers.
LanePrefixRecord begin(int seed, std::size_t prompt_tokens, std::size_t held,
                       std::vector<std::uint64_t> media = {}) {
    LanePrefixRecord record;
    // The Program copies the prompt's ledger here; the test keeps both in step explicitly.
    const std::vector<TokenId> prompt = tokens_of(seed, prompt_tokens);
    record.media                      = std::move(media);
    record.anchor.tokens              = prompt;
    record.anchor.position            = static_cast<std::uint32_t>(prompt.size());
    record.anchor.vision_key          = qsa_media_key_at(record.media, prompt.size());
    qsa_set_prefix_frontier(record, prompt, static_cast<std::uint32_t>(held));
    return record;
}

// The lane's token history once generation has appended `generated` tokens past the prompt.
std::vector<TokenId> with_generation(int seed, std::size_t prompt_tokens, std::size_t generated) {
    std::vector<TokenId> tokens = tokens_of(seed, prompt_tokens);
    for (std::size_t i = 0; i < generated; ++i) {
        tokens.push_back(static_cast<TokenId>(seed * 100000 + prompt_tokens + i));
    }
    return tokens;
}

void test_frontier_follows_writes() {
    LanePrefixRecord record = begin(1, 8, 4);
    expect(record.held.position == 4, "held starts at the reused frontier");
    expect(record.anchor.position == 8, "anchor is the whole prompt");

    const std::vector<TokenId> ledger = with_generation(1, 8, 3);
    qsa_set_prefix_frontier(record, ledger, 11);
    expect(record.held.position == 11, "held follows a committed frontier");
    expect(record.held.tokens == ledger, "held is the ledger's prefix");

    // Idempotent: a commit that did not move the frontier changes nothing.
    qsa_set_prefix_frontier(record, ledger, 11);
    expect(record.held.position == 11, "an unchanged frontier keeps held");
}

void test_stop_cut_truncates() {
    LanePrefixRecord record = begin(2, 8, 8);
    const std::vector<TokenId> ledger = with_generation(2, 8, 5);
    qsa_set_prefix_frontier(record, ledger, 13);
    expect(record.held.position == 13, "held reaches the generated frontier");

    // A stop string cut the visible text back to 10: the record ends where the text does, so a
    // prompt that echoes the visible text resumes from it.
    qsa_set_prefix_frontier(record, ledger, 10);
    expect(record.held.position == 10, "a cut truncates held");
    expect(record.held.tokens == std::vector<TokenId>(ledger.begin(), ledger.begin() + 10),
           "the truncated held is the visible prefix");

    // A later step appends again from the same ledger.
    qsa_set_prefix_frontier(record, ledger, 12);
    expect(record.held.position == 12, "held grows again after a cut");
    expect(record.anchor.position == 8, "the anchor is untouched by a cut");
}

void test_media_identity_is_the_records_own() {
    // Media in the second 64-token block: keys 0x11 then 0x22.
    const std::vector<std::uint64_t> media{0x11ULL, 0x22ULL};
    LanePrefixRecord record = begin(3, 128, 64, media);
    expect(record.held.vision_key == 0x11ULL, "held keys on its own prefix's block");
    expect(record.anchor.vision_key == 0x22ULL, "anchor keys on the whole prompt");

    const std::vector<TokenId> ledger = with_generation(3, 128, 4);
    qsa_set_prefix_frontier(record, ledger, 132);
    expect(record.held.vision_key == 0x22ULL, "past the last keyed block the key holds");

    // A prompt continuing the lane's prefix: one token past it, since a prompt's last token is
    // always fed.
    std::vector<TokenId> next = ledger;
    next.push_back(static_cast<TokenId>(999));
    // The prompt's own key at the same length agrees; a different media identity does not.
    const std::uint32_t whole = qsa_prefix_reuse(record.held, next,
                                                 qsa_media_key_at(media, record.held.position));
    expect(whole == 132, "the lane's prefix matches a prompt with the same media");
    const std::uint32_t other = qsa_prefix_reuse(record.held, next, 0x33ULL);
    expect(other == 0, "a different media identity rejects the prefix");
}

void test_text_only_prefix_keys_zero() {
    LanePrefixRecord record = begin(4, 16, 16);
    expect(record.media.empty(), "a text-only prompt carries no media keys");
    expect(record.held.vision_key == 0, "a text-only prefix keys zero");
    const std::vector<TokenId> ledger = with_generation(4, 16, 2);
    qsa_set_prefix_frontier(record, ledger, 18);
    std::vector<TokenId> next = ledger;
    next.push_back(static_cast<TokenId>(999));
    expect(qsa_prefix_reuse(record.held, next, 0) == 18, "a text-only prompt reuse matches");
}

void test_anchor_picks_up_a_prompt_that_drops_the_generation() {
    LanePrefixRecord record = begin(5, 12, 12);
    const std::vector<TokenId> ledger = with_generation(5, 12, 6);
    qsa_set_prefix_frontier(record, ledger, 18);

    // A client that does not echo the generation: the next prompt continues the prompt end only.
    std::vector<TokenId> next = tokens_of(5, 12);
    const std::vector<TokenId> continuation = tokens_of(6, 3);
    next.insert(next.end(), continuation.begin(), continuation.end());
    expect(qsa_prefix_reuse(record.held, next, 0) == 0, "the live prefix does not continue it");
    expect(qsa_prefix_reuse(record.anchor, next, 0) == 12,
           "the turn-closure prefix does continue it");

    // A client that echoes the generation continues the live prefix instead.
    std::vector<TokenId> echoed = ledger;
    echoed.insert(echoed.end(), continuation.begin(), continuation.end());
    expect(qsa_prefix_reuse(record.held, echoed, 0) == 18, "the live prefix continues it");
    expect(qsa_prefix_reuse(record.anchor, echoed, 0) == 12,
           "the closure prefix is shorter than the live one");
}

void test_clear_drops_everything() {
    LanePrefixRecord record = begin(7, 8, 8, {0x44ULL});
    record.epoch            = 3;
    record.clear();
    expect(record.held.empty() && record.anchor.empty() && record.media.empty(),
           "clear drops the tokens and the media");
    expect(record.epoch == 0, "clear drops the epoch, so the next tenant cannot inherit it");
}

} // namespace

int main() {
    test_frontier_follows_writes();
    test_stop_cut_truncates();
    test_media_identity_is_the_records_own();
    test_text_only_prefix_keys_zero();
    test_anchor_picks_up_a_prompt_that_drops_the_generation();
    test_clear_drops_everything();
    if (failures != 0) {
        std::cerr << failures << " QSA lane-prefix checks failed\n";
        return 1;
    }
    std::cout << "QSA lane-prefix checks passed\n";
    return 0;
}
