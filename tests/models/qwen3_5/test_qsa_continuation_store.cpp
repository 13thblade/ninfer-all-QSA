// Host-only checks for the QSA continuation image store (S2). No GPU, no model artifact: the
// store's page allocator uses aligned host memory and its bookkeeping mirrors upstream
// qwen4_exp_core.cpp (Stored / take_pages / make_host_room / find_stored / store_image / retire).

#include "models/qwen3_5/program/prefix/qsa_continuation.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::detail::HeldPrefix;
using ninfer::models::qwen3_5::detail::QsaContinuationStore;
using ninfer::models::qwen3_5::detail::qsa_prefix_reuse;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

HeldPrefix key_of(int seed, std::size_t tokens, std::uint64_t vision = 0) {
    HeldPrefix key;
    key.tokens.reserve(tokens);
    for (std::size_t i = 0; i < tokens; ++i) {
        key.tokens.push_back(static_cast<TokenId>(seed * 100000 + i));
    }
    key.vision_key = vision;
    key.position   = static_cast<std::uint32_t>(tokens);
    return key;
}

void test_prefix_reuse_semantics() {
    // Copied from upstream: held is a strict prefix of the prompt, and the media key must agree.
    auto prompt = key_of(1, 10).tokens; // 10 tokens
    HeldPrefix held;
    held.tokens.assign(prompt.begin(), prompt.begin() + 9); // strict prefix (last token not fed)
    held.vision_key = 0;
    expect(qsa_prefix_reuse(held, prompt, 0) == 9, "a strict prefix should reuse held.size()");
    held.tokens = prompt; // now the whole prompt: not reusable unless whole
    expect(qsa_prefix_reuse(held, prompt, 0) == 0, "the whole prompt is not a strict-prefix reuse");
    expect(qsa_prefix_reuse(held, prompt, 0, true) == 10, "whole prompt reuses under whole=true");
    held.tokens.assign(prompt.begin(), prompt.begin() + 9);
    expect(qsa_prefix_reuse(held, prompt, 42) == 0, "a different media key must not reuse");
    HeldPrefix not_prefix = held;
    not_prefix.tokens[0] += 1000;
    expect(qsa_prefix_reuse(not_prefix, prompt, 0) == 0, "a non-prefix must not reuse");
}

void test_budget_and_lru_eviction() {
    // One image = one 8 MiB page (image_bytes below one page). Budget 2 pages -> the third store
    // evicts the least recently used (first) image.
    const std::uint64_t budget = 2 * QsaContinuationStore::kHostPageBytes;
    QsaContinuationStore store(budget);
    const std::uint64_t bytes = 1000;
    auto a = key_of(1, 200), b = key_of(2, 200), c = key_of(3, 200);
    expect(store.store_image(a, bytes, 0), "store a within budget");
    expect(store.store_image(b, bytes, 0), "store b within budget");
    expect(store.images().size() == 2, "two images held");
    expect(store.used_bytes() == 2 * QsaContinuationStore::kHostPageBytes,
           "used bytes charged in whole pages");
    // Touch a so b becomes the oldest.
    expect(store.restore_stored(a).has_value(), "restore a refreshes recency");
    expect(store.store_image(c, bytes, 0), "store c after evicting the LRU");
    expect(store.images().size() == 2, "budget still holds two images after eviction");
    expect(!store.stored_index(b).has_value(), "b (least recently used) was evicted");
    expect(store.stored_index(a).has_value() && store.stored_index(c).has_value(),
           "the recently used a and the new c survive");
}

void test_same_prefix_refreshes_last_used() {
    QsaContinuationStore store(4 * QsaContinuationStore::kHostPageBytes);
    auto key = key_of(7, 200);
    expect(store.store_image(key, 1000, 0), "first store of key");
    const auto first_clock  = store.clock();
    const auto first_bytes  = store.used_bytes();
    expect(!store.store_image(key, 1000, 0), "second store of the same key is not a new image");
    expect(store.images().size() == 1, "same prefix keeps a single image");
    expect(store.used_bytes() == first_bytes, "same prefix does not charge pages again");
    expect(store.clock() == first_clock + 1, "same prefix refreshes last_used once");
}

void test_short_prefix_not_stored() {
    QsaContinuationStore store(4 * QsaContinuationStore::kHostPageBytes);
    auto short_key = key_of(9, QsaContinuationStore::kMinStoredTokens - 1);
    expect(!store.store_image(short_key, 1000, 0), "prefixes under kMinStoredTokens are ignored");
    expect(store.images().empty() && store.used_bytes() == 0, "a short prefix allocates nothing");
}

void test_page_alignment_and_rounding() {
    QsaContinuationStore store(2 * QsaContinuationStore::kHostPageBytes);
    auto key = key_of(11, 200);
    expect(store.store_image(key, 1000, 0), "store one image for alignment check");
    const auto& image = store.images().front();
    expect(image.pages.size() == 1, "one small image takes exactly one page");
    const auto addr = reinterpret_cast<std::uintptr_t>(image.pages[0]);
    expect(addr % QsaContinuationStore::kHostPageBytes == 0, "page slice is 8 MiB aligned");
    expect(QsaContinuationStore::page_rounded(1, QsaContinuationStore::kHostPageBytes) ==
               QsaContinuationStore::kHostPageBytes,
           "page_rounded rounds up to a whole page");
    expect(QsaContinuationStore::page_rounded(QsaContinuationStore::kHostPageBytes,
                                              QsaContinuationStore::kHostPageBytes) ==
               QsaContinuationStore::kHostPageBytes,
           "page_rounded leaves a whole page unchanged");
}

void test_find_stored_longest_and_drop() {
    QsaContinuationStore store(4 * QsaContinuationStore::kHostPageBytes);
    auto a = key_of(21, 200), b = key_of(22, 300);
    expect(store.store_image(a, 1000, 0), "store a");
    expect(store.store_image(b, 1000, 1), "store b");
    // A prompt continuing b by 5 tokens: longest stored prefix is b (300).
    auto prompt = b.tokens;
    for (int i = 0; i < 5; ++i) { prompt.push_back(9000 + i); }
    const auto [index, length] = store.find_stored(prompt, b.vision_key);
    expect(index == store.stored_index(b).value() && length == 300,
           "find_stored returns the longest matching stored prefix");
    // A media-key mismatch rejects the match.
    expect(store.find_stored(prompt, b.vision_key + 7).second == 0,
           "find_stored rejects a media-key mismatch");
    // Drop every stored prefix of the prompt: a is from a different sequence, so only b goes.
    store.drop_stored_prefixes(prompt);
    expect(!store.stored_index(b).has_value(), "drop_stored_prefixes removes b");
}

} // namespace

int main() {
    test_prefix_reuse_semantics();
    test_budget_and_lru_eviction();
    test_same_prefix_refreshes_last_used();
    test_short_prefix_not_stored();
    test_page_alignment_and_rounding();
    test_find_stored_longest_and_drop();
    if (failures != 0) {
        std::cerr << failures << " QSA continuation-store checks failed\n";
        return 1;
    }
    std::cout << "QSA continuation-store checks passed\n";
    return 0;
}
