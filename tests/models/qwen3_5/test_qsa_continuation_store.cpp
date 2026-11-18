// Host-only checks for the QSA continuation image store (S2). No GPU, no model artifact: the
// store's page allocator uses aligned host memory and its bookkeeping mirrors upstream
// qwen4_exp_core.cpp (Stored / take_pages / make_host_room / find_stored / store_image / retire).

#include "models/qwen3_5/program/prefix/qsa_continuation.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

using ninfer::models::qwen3_5::detail::ImageChunk;
using ninfer::models::qwen3_5::detail::ImageRegion;
using ninfer::models::qwen3_5::detail::QsaImageGeometry;
using ninfer::models::qwen3_5::detail::qsa_copy_image;
using ninfer::models::qwen3_5::detail::qsa_copy_plane_chunks;
using ninfer::models::qwen3_5::detail::qsa_image_bytes;
using ninfer::models::qwen3_5::detail::qsa_image_chunk_bytes;
using ninfer::models::qwen3_5::detail::qsa_kv_bytes_per_head_token;

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::detail::HeldPrefix;
using ninfer::models::qwen3_5::detail::LanePrefixRecord;
using ninfer::models::qwen3_5::detail::QsaLaneRetentionView;
using ninfer::models::qwen3_5::detail::QsaContinuationStore;
using ninfer::models::qwen3_5::detail::qsa_choose_reuse;
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

void test_kv_bytes_follow_storage_precision() {
    // The per-head KV width must come from the upstream profile, not a hardcoded rk4v4 constant:
    // every supported --kv-dtype has a different plane layout, and bf16/int8 are the common ones.
    using ninfer::KvCacheStorage;
    const std::uint32_t bf16  = qsa_kv_bytes_per_head_token(KvCacheStorage::BFloat16);
    const std::uint32_t int8  = qsa_kv_bytes_per_head_token(KvCacheStorage::Int8Group64);
    const std::uint32_t rk4v4 = qsa_kv_bytes_per_head_token(KvCacheStorage::RotatedLloyd4KeyInt4Value);
    const std::uint32_t fp8   = qsa_kv_bytes_per_head_token(KvCacheStorage::Fp8E4M3Row256);
    const std::uint32_t nfp4  = qsa_kv_bytes_per_head_token(KvCacheStorage::Nvfp4Group16);
    // bf16: 256*2 (K) + 256*2 (V), no scales = 1024. int8: 256 + 256 + 4*2 + 4*2 = 528.
    // rk4v4: 128 + 128 + 4*2 + 8*2 = 280. All derived, none hardcoded in the geometry.
    expect(bf16 == 1024, "bf16 KV bytes/head/token should be 1024 (two BF16 planes)");
    expect(int8 == 528, "int8 KV bytes/head/token should be 528 (two INT8 + two G64 scales)");
    expect(rk4v4 == 280, "rk4v4 KV bytes/head/token should be 280 (2 int4 + 2 scale planes)");
    expect(fp8 > 0 && nfp4 > 0, "fp8/nvfp4 profiles must yield a positive width");
    expect(bf16 > int8 && int8 > rk4v4, "precision width ordering bf16 > int8 > rk4v4");
}

void test_image_bytes_arithmetic_all_positions() {
    // Geometry of the QSA-27B rk4v4 streamed build; kv width derived from the storage profile.
    QsaImageGeometry g;
    g.page_tokens             = 128;
    g.kv_heads                = 4;
    g.full_layers             = 16;
    g.kv_bytes_per_head_token = qsa_kv_bytes_per_head_token(ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value);
    g.indexer_head_dim        = 128;
    g.compress_ratio          = 4;
    g.recurrent_bytes         = 0;
    g.mtp_bytes               = 0;
    // Independent oracle with the same §4.3 layout, recomputed per position.
    const auto oracle = [&](std::uint32_t position) {
        const std::uint64_t used   = (std::uint64_t(position) + g.page_tokens - 1) / g.page_tokens;
        const std::uint64_t blocks = std::uint64_t(position) / g.compress_ratio;
        const std::uint64_t kv = used * g.page_tokens * g.kv_heads * g.kv_bytes_per_head_token;
        const std::uint64_t pooled = std::uint64_t(g.indexer_head_dim) * 4 * blocks;
        const std::uint64_t tail   = std::uint64_t(g.indexer_head_dim) * (g.compress_ratio - 1) * 4;
        const std::uint64_t axes   = 3ULL * 3 * 4;
        return (kv + pooled + tail + axes) * g.full_layers + g.recurrent_bytes + g.mtp_bytes;
    };
    for (const std::uint32_t position : {13312U, 67136U, 131072U}) {
        expect(qsa_image_bytes(position, g) == oracle(position),
               "image_bytes matches the §4.3 region arithmetic");
    }
    // A wider precision must produce a strictly larger image at the same position.
    QsaImageGeometry bf16 = g;
    bf16.kv_bytes_per_head_token = qsa_kv_bytes_per_head_token(ninfer::KvCacheStorage::BFloat16);
    expect(qsa_image_bytes(13312, bf16) > qsa_image_bytes(13312, g),
           "bf16 image is larger than rk4v4 at the same position");
    expect(qsa_image_bytes(0, g) == 0, "an empty position has a zero-byte image");
}

void test_image_save_load_roundtrip() {
    constexpr std::size_t kPage = QsaContinuationStore::kHostPageBytes;
    // Two regions of distinct pattern bytes, crossing a page boundary to exercise slicing.
    std::vector<std::byte> a(kPage + 123, std::byte{0x5A});
    std::vector<std::byte> b(4096, std::byte{0xC3});
    std::vector<std::byte> a_copy(a.size(), std::byte{0});
    std::vector<std::byte> b_copy(b.size(), std::byte{0});
    const std::vector<ImageRegion> src{{a.data(), a.size()}, {b.data(), b.size()}};
    const std::vector<ImageRegion> dst{{a_copy.data(), a_copy.size()}, {b_copy.data(), b_copy.size()}};
    const std::uint64_t total = a.size() + b.size();
    const std::size_t pages_needed = static_cast<std::size_t>((total + kPage - 1) / kPage);
    std::vector<std::vector<std::byte>> page_store(pages_needed, std::vector<std::byte>(kPage));
    std::vector<std::byte*> pages;
    pages.reserve(pages_needed);
    for (auto& page : page_store) { pages.push_back(page.data()); }
    qsa_copy_image(src, pages, kPage, true);   // regions -> pages
    qsa_copy_image(dst, pages, kPage, false);  // pages -> regions
    expect(a_copy == a && b_copy == b, "save->load round-trips every region byte-for-byte");
}

void test_plane_chunks_roundtrip() {
    // Three chunks of different sizes stand in for a lane's planes (K, V, and a scale plane).
    std::vector<std::byte> a(4096), b(1024), c(96);
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = std::byte(i & 0xFF); }
    for (std::size_t i = 0; i < b.size(); ++i) { b[i] = std::byte((i * 7) & 0xFF); }
    for (std::size_t i = 0; i < c.size(); ++i) { c[i] = std::byte((i * 13) & 0xFF); }
    const std::vector<std::byte> a_orig = a;
    const std::vector<std::byte> b_orig = b;
    const std::vector<std::byte> c_orig = c;
    const std::vector<ImageChunk> chunks{
        {a.data(), a.size()}, {b.data(), b.size()}, {c.data(), c.size()}};
    expect(qsa_image_chunk_bytes(chunks) == a.size() + b.size() + c.size(),
           "the chunk bytes are the plane bytes");
    std::vector<std::byte> image(qsa_image_chunk_bytes(chunks), std::byte{0});
    qsa_copy_plane_chunks(chunks, image, /*save=*/true);
    // The live planes move on (the next sequence writes them); the image must still hold the
    // prefix, and loading it back must restore every byte.
    std::fill(a.begin(), a.end(), std::byte{0xEE});
    std::fill(b.begin(), b.end(), std::byte{0xEE});
    std::fill(c.begin(), c.end(), std::byte{0xEE});
    qsa_copy_plane_chunks(chunks, image, /*save=*/false);
    expect(a == a_orig && b == b_orig && c == c_orig, "load restores every plane byte-for-byte");

    // A buffer that does not match the planes is refused rather than truncating a plane.
    bool threw = false;
    try {
        qsa_copy_plane_chunks(chunks, std::span<std::byte>(image.data(), image.size() - 1), true);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "a short image buffer is refused");
}

void test_image_payload_roundtrip() {
    constexpr std::size_t kPage = QsaContinuationStore::kHostPageBytes;
    QsaContinuationStore store(64ULL << 20);
    const HeldPrefix key = key_of(11, 4096);

    // Two planes inside one page, each a distinct pattern.
    std::vector<std::byte> plane_a((1U << 20) + 7, std::byte{0x11});
    std::vector<std::byte> plane_b(4096, std::byte{0x22});
    std::vector<ImageChunk> chunks{{plane_a.data(), plane_a.size()},
                                   {plane_b.data(), plane_b.size()}};
    const std::uint64_t payload = qsa_image_chunk_bytes(chunks);

    expect(store.store_image(key, payload, 0, true), "an endpoint image stores");
    const auto index = store.stored_index(key);
    expect(index.has_value(), "the stored image is found by its key");
    expect(store.images()[*index].whole, "an image stored at its frontier serves the whole prompt");
    expect(store.used_bytes() == kPage, "the budget charges the image's pages");

    store.write_payload(*index, chunks);
    // Mutate the live planes, then read the image back.
    std::fill(plane_a.begin(), plane_a.end(), std::byte{0x00});
    std::fill(plane_b.begin(), plane_b.end(), std::byte{0x00});
    store.read_payload(*index, chunks);
    expect(std::all_of(plane_a.begin(), plane_a.end(),
                       [](std::byte b) { return b == std::byte{0x11}; }),
           "the first plane round-trips through the image");
    expect(std::all_of(plane_b.begin(), plane_b.end(),
                       [](std::byte b) { return b == std::byte{0x22}; }),
           "the second plane round-trips through the image");

    // A payload that does not match the image is refused, not truncated.
    bool threw = false;
    try {
        std::vector<ImageChunk> short_chunks{{plane_a.data(), plane_a.size()}};
        store.write_payload(*index, short_chunks);
    } catch (const std::logic_error&) { threw = true; }
    expect(threw, "a short payload is refused");

    // Eviction releases the pages, so the budget returns to zero.
    expect(store.make_host_room(64ULL << 20), "making room evicts the image");
    expect(store.used_bytes() == 0, "the image's pages are released with it");
}

void test_three_way_choice() {
    QsaContinuationStore store(64ULL << 20);
    std::vector<TokenId> prompt(256);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = static_cast<TokenId>(i + 1);
    }
    LanePrefixRecord record;
    record.epoch = 7;
    record.held.tokens.assign(prompt.begin(), prompt.begin() + 200);
    record.held.position = 200;
    record.anchor.tokens.assign(prompt.begin(), prompt.begin() + 128);
    record.anchor.position = 128;
    constexpr std::uint64_t kEpoch = 7;
    QsaLaneRetentionView retention;
    retention.live               = true;
    retention.snapshot           = true;
    retention.snapshot_drafter_follows = true;

    // The prompt grows past the lane's live state: the endpoint resumes 200.
    auto choice = qsa_choose_reuse(record, kEpoch, retention, true, &store, prompt, {});
    expect(choice.tokens == 200 && choice.path_kind == 0 && !choice.from_store,
           "the live state wins when it is the longest");
    expect(choice.live == 200 && choice.anchored == 128,
           "the log reports what each source offered");

    // A regenerate of the anchor's prompt: equal length, so only `whole` can resume it, and the
    // live state (200 > 127) is rejected outright.
    const std::vector<TokenId> same(prompt.begin(), prompt.begin() + 128);
    choice = qsa_choose_reuse(record, kEpoch, retention, true, &store, same, {});
    expect(choice.tokens == 128 && choice.path_kind == 1,
           "an equal-length prompt resumes at the lane's prompt-end snapshot");
    expect(choice.whole, "the snapshot covers the whole prompt");

    // A stored endpoint image of the whole prompt beats both (the lane moved on).
    HeldPrefix whole_key;
    whole_key.tokens   = prompt;
    whole_key.position = static_cast<std::uint32_t>(prompt.size());
    expect(store.store_image(whole_key, 4096, 0, true), "store an endpoint image of the prompt");
    choice = qsa_choose_reuse(record, kEpoch, retention, true, &store, prompt, {});
    expect(choice.from_store && choice.tokens == 256 && choice.whole,
           "the stored image serves the whole prompt");

    // An anchor image (state only) never serves a store resume: it has no KV to land.
    expect(store.store_image(key_of(3, 300), 4096, 1, true), "an anchor image stores");
    choice = qsa_choose_reuse(record, kEpoch + 1, retention, true, &store, prompt, {});
    expect(choice.tokens == 256 && choice.from_store,
           "a lane that changed tenants keeps only the endpoint image");

    // The same image read with a different media identity does not match.
    const std::vector<std::uint64_t> media{0xAAULL};
    choice = qsa_choose_reuse(record, kEpoch, retention, true, &store, prompt, media);
    expect(!choice.from_store, "a media mismatch rejects the stored image");

    // An image carries no draft KV: a drafting request cannot resume from one.
    choice = qsa_choose_reuse(record, kEpoch + 1, retention, false, &store, prompt, {});
    expect(choice.tokens == 0, "a store resume is refused while a drafter runs");

    // A lane that retains nothing offers nothing.
    choice = qsa_choose_reuse(record, kEpoch, {}, false, nullptr, prompt, {});
    expect(choice.tokens == 0, "an empty lane and an absent store admit a root");
}

} // namespace

int main() {
    test_prefix_reuse_semantics();
    test_budget_and_lru_eviction();
    test_same_prefix_refreshes_last_used();
    test_short_prefix_not_stored();
    test_page_alignment_and_rounding();
    test_find_stored_longest_and_drop();
    test_kv_bytes_follow_storage_precision();
    test_image_bytes_arithmetic_all_positions();
    test_image_save_load_roundtrip();
    test_plane_chunks_roundtrip();
    test_image_payload_roundtrip();
    test_three_way_choice();
    if (failures != 0) {
        std::cerr << failures << " QSA continuation-store checks failed\n";
        return 1;
    }
    std::cout << "QSA continuation-store checks passed\n";
    return 0;
}
