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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <utility>
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

// ---- The image store (S2) --------------------------------------------------------------------
// Upstream qwen4_exp_core.cpp:743-767, 1446-1527 translated to a host-only, GPU-free component.
// An image is a token-exact prefix whose bytes live in a run of kHostPageBytes pinned pages. This
// tier keeps the whole image (KV + QSA pooled/tail/axes + recurrent + MTP); the byte layout itself
// is filled in by image_regions() (S3), so here a payload is opaque bytes of `image_bytes`.

struct StoredImage {
    HeldPrefix key;
    std::vector<std::byte*> pages; // 8 MiB slices; empty while only on disk
    std::uint64_t bytes     = 0;   // exact image bytes (<= pages.size() * page_bytes)
    std::uint64_t last_used = 0;
    std::uint32_t path_kind = 0;   // PrefixReusePath semantics (0 = endpoint, 1 = turn closure)
};

class QsaContinuationStore {
public:
    static constexpr std::size_t kHostPageBytes = std::size_t(8) << 20;
    static constexpr std::size_t kMinStoredTokens = 128;

    QsaContinuationStore(std::uint64_t host_budget, std::size_t page_bytes = kHostPageBytes);
    ~QsaContinuationStore();
    QsaContinuationStore(const QsaContinuationStore&)            = delete;
    QsaContinuationStore& operator=(const QsaContinuationStore&) = delete;

    [[nodiscard]] static std::uint64_t page_rounded(std::uint64_t bytes, std::size_t page_bytes);

    // Allocates page_rounded(bytes)/page_bytes fresh pages and charges host_used. Upstream
    // take_pages (:1446): returns empty when the budget cannot be met (never over-commit).
    [[nodiscard]] std::vector<std::byte*> take_pages(std::uint64_t bytes);
    void give_back(StoredImage& image) noexcept;
    // Upstream make_host_room (:1470): evicts least-recently-used images until `bytes` more fit.
    // This tier has no disk in the host-only configuration, so an evicted image is dropped.
    [[nodiscard]] bool make_host_room(std::uint64_t bytes);

    // Longest stored prefix of `prompt` (upstream find_stored :1336): index and length.
    [[nodiscard]] std::pair<std::size_t, std::uint32_t>
    find_stored(std::span<const TokenId> prompt_tokens, std::uint64_t prompt_vision_key) const;
    [[nodiscard]] std::optional<std::size_t>
    stored_index(const HeldPrefix& key) const;

    // Upstream store_image (:1500): exact-key images refresh last_used; short prefixes and a
    // zero budget are ignored; a full store evicts before taking pages.
    bool store_image(const HeldPrefix& key, std::uint64_t image_bytes, std::uint32_t path_kind);
    // Upstream restore_stored (:1551) returns the image index; the caller copies the payload via
    // load_image (S3). Restoring refreshes last_used.
    [[nodiscard]] std::optional<std::size_t> restore_stored(const HeldPrefix& key);

    // Upstream drop_stored_prefixes (:1158): drop every image whose key is a prefix of `prompt`.
    void drop_stored_prefixes(std::span<const TokenId> prompt_tokens);
    void drop_dead() noexcept;

    [[nodiscard]] const std::vector<StoredImage>& images() const noexcept { return stored_; }
    [[nodiscard]] std::uint64_t used_bytes() const noexcept { return host_used_; }
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return host_budget_; }
    [[nodiscard]] std::uint64_t clock() const noexcept { return use_clock_; }
    // Test hooks: the pages handed out are 8 MiB-aligned; expose the free count for assertions.
    [[nodiscard]] std::size_t free_page_count() const noexcept { return free_pages_.size(); }
    std::byte* writable_page(StoredImage& image, std::size_t page) const noexcept {
        return page < image.pages.size() ? image.pages[page] : nullptr;
    }

private:
    // A stateful deleter keeps the page alignment so ::operator delete matches the aligned new.
    struct AlignedDeleter {
        std::size_t page_bytes = 0;
        void operator()(std::byte* p) const noexcept; // defined in qsa_continuation.cpp
    };
    struct Chunk {
        std::unique_ptr<std::byte[], AlignedDeleter> data;
    };
    void pin_pages(std::size_t count);

    std::uint64_t host_budget_;
    std::uint64_t host_used_  = 0;
    std::uint64_t use_clock_  = 0;
    std::size_t page_bytes_;
    std::vector<Chunk> chunks_;
    std::vector<std::byte*> free_pages_;
    std::vector<StoredImage> stored_;
};

} // namespace ninfer::models::qwen3_5::detail
