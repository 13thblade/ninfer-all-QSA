#include "models/qwen3_5/program/prefix/qsa_continuation.h"

#include "core/dtype.h"
#include "ops/kv_cache/d256_profile.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

namespace {

// page_bytes-aligned allocation. Each chunk is a run of whole pages, so every page slice
// (base + i*page_bytes) keeps the page alignment. S2 uses aligned host memory (pageable) so the
// store is testable without a GPU; S4 swaps the allocator for pinned memory without touching the
// bookkeeping below.
std::byte* allocate_pages(std::size_t bytes, std::size_t page_bytes) {
    return static_cast<std::byte*>(::operator new(bytes, std::align_val_t(page_bytes)));
}

} // namespace

void QsaContinuationStore::AlignedDeleter::operator()(std::byte* p) const noexcept {
    ::operator delete(p, std::align_val_t(page_bytes));
}

std::uint32_t qsa_kv_bytes_per_head_token(ninfer::KvCacheStorage storage) {
    // Same four planes the engine sizes its streamed host copy with (program_impl.cpp:772-807):
    // key codes, value codes, key scale, value scale. A zero leading extent means no scale plane.
    const auto p = ninfer::ops::d256_kv_cache_profile(storage);
    std::uint32_t bytes = 0;
    bytes += static_cast<std::uint32_t>(p.key_leading_extent) *
             static_cast<std::uint32_t>(ninfer::dtype_size(p.key_code_dtype));
    bytes += static_cast<std::uint32_t>(p.value_leading_extent) *
             static_cast<std::uint32_t>(ninfer::dtype_size(p.value_code_dtype));
    bytes += static_cast<std::uint32_t>(p.scale_leading_extent) *
             static_cast<std::uint32_t>(ninfer::dtype_size(p.key_scale_dtype));
    bytes += static_cast<std::uint32_t>(p.value_scale_leading_extent) *
             static_cast<std::uint32_t>(ninfer::dtype_size(p.value_scale_dtype));
    return bytes;
}

std::uint64_t qsa_image_bytes(std::uint32_t position, const QsaImageGeometry& g) {
    if (position == 0) { return 0; }
    const std::uint64_t used = (std::uint64_t(position) + g.page_tokens - 1) / g.page_tokens;
    const std::uint64_t blocks = std::uint64_t(position) / g.compress_ratio;
    // One layer's KV: all 4 planes of every head over `used` whole pages.
    const std::uint64_t kv_per_layer =
        std::uint64_t(used) * g.page_tokens * g.kv_heads * g.kv_bytes_per_head_token;
    const std::uint64_t pooled_per_layer =
        std::uint64_t(g.indexer_head_dim) * 4 * blocks;
    const std::uint64_t tail_per_layer =
        std::uint64_t(g.indexer_head_dim) * (g.compress_ratio - 1) * 4;
    const std::uint64_t axes_per_layer = 3ULL * 3 * 4;
    const std::uint64_t per_layer = kv_per_layer + pooled_per_layer + tail_per_layer + axes_per_layer;
    return per_layer * g.full_layers + g.recurrent_bytes + g.mtp_bytes;
}

void qsa_copy_plane_chunks(std::span<const ImageChunk> chunks, std::span<std::byte> image,
                           bool save) {
    const std::size_t total = qsa_image_chunk_bytes(chunks);
    if (image.size() != total) {
        throw std::invalid_argument("qsa image: the buffer does not match the lane's planes");
    }
    std::size_t at = 0;
    for (const ImageChunk& chunk : chunks) {
        if (chunk.bytes == 0) { continue; }
        if (save) { std::memcpy(image.data() + at, chunk.data, chunk.bytes); }
        else { std::memcpy(chunk.data, image.data() + at, chunk.bytes); }
        at += chunk.bytes;
    }
}

void qsa_copy_image(std::span<const ImageRegion> regions, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save) {
    std::uint64_t total = 0;
    for (const ImageRegion& region : regions) { total += region.bytes; }
    if (page_bytes == 0 ||
        pages.size() < (total + page_bytes - 1) / page_bytes) {
        throw std::invalid_argument("qsa image: the pages do not hold the image");
    }
    std::uint64_t at = 0;
    for (const ImageRegion& region : regions) {
        std::size_t left = region.bytes;
        auto* cursor = const_cast<std::byte*>(region.data);
        while (left > 0) {
            const std::size_t in_page = std::min<std::size_t>(left, page_bytes - at % page_bytes);
            std::byte* host = pages[at / page_bytes] + at % page_bytes;
            if (save) { std::memcpy(host, cursor, in_page); }
            else { std::memcpy(cursor, host, in_page); }
            cursor += in_page;
            left -= in_page;
            at += in_page;
        }
    }
}

std::uint64_t QsaContinuationStore::page_rounded(std::uint64_t bytes, std::size_t page_bytes) {
    if (page_bytes == 0) { return bytes; }
    return (bytes + page_bytes - 1) / page_bytes * page_bytes;
}

QsaContinuationStore::QsaContinuationStore(std::uint64_t host_budget, std::size_t page_bytes)
    : host_budget_(host_budget), page_bytes_(page_bytes == 0 ? kHostPageBytes : page_bytes) {}

QsaContinuationStore::~QsaContinuationStore() {
    // Release the page slices before the owning aligned chunks (members die after the body).
    stored_.clear();
    free_pages_.clear();
    chunks_.clear();
}

void QsaContinuationStore::pin_pages(std::size_t count) {
    if (count == 0) { return; }
    const std::size_t bytes = count * page_bytes_;
    std::byte* base         = allocate_pages(bytes, page_bytes_);
    Chunk chunk{std::unique_ptr<std::byte[], AlignedDeleter>(
        base, AlignedDeleter{page_bytes_})};
    for (std::size_t i = count; i-- > 0;) { free_pages_.push_back(base + i * page_bytes_); }
    chunks_.push_back(std::move(chunk));
}

std::vector<std::byte*> QsaContinuationStore::take_pages(std::uint64_t bytes) {
    const std::size_t count =
        static_cast<std::size_t>(page_rounded(bytes, page_bytes_) / page_bytes_);
    if (count == 0) { return {}; }
    if (free_pages_.size() < count) {
        // Upstream take_pages pins more when the free list falls short; it never exceeds the
        // budget because the caller already made room.
        if (host_used_ + static_cast<std::uint64_t>(count - free_pages_.size()) * page_bytes_ >
            host_budget_) {
            return {};
        }
        pin_pages(count - free_pages_.size());
    }
    std::vector<std::byte*> out(free_pages_.end() - std::ptrdiff_t(count), free_pages_.end());
    free_pages_.resize(free_pages_.size() - count);
    host_used_ += static_cast<std::uint64_t>(count) * page_bytes_;
    return out;
}

void QsaContinuationStore::give_back(StoredImage& image) noexcept {
    const std::uint64_t released = static_cast<std::uint64_t>(image.pages.size()) * page_bytes_ +
                                   image.hidden_bytes;
    host_used_ -= released > host_used_ ? host_used_ : released;
    free_pages_.insert(free_pages_.end(), image.pages.begin(), image.pages.end());
    image.pages.clear();
    image.hidden.clear();
    image.hidden_bytes = 0;
}

void QsaContinuationStore::drop_dead() noexcept {
    std::erase_if(stored_, [](const StoredImage& image) { return image.pages.empty(); });
}

bool QsaContinuationStore::make_host_room(std::uint64_t bytes) {
    if (bytes > host_budget_) { return false; }
    while (host_used_ + bytes > host_budget_) {
        std::optional<std::size_t> oldest;
        for (std::size_t i = 0; i < stored_.size(); ++i) {
            if (!stored_[i].pages.empty() &&
                (!oldest || stored_[i].last_used < stored_[*oldest].last_used)) {
                oldest = i;
            }
        }
        if (!oldest) { return false; }
        // No disk tier in this host-only configuration: dropping the oldest is the eviction.
        give_back(stored_[*oldest]);
        drop_dead();
    }
    drop_dead();
    return true;
}

std::optional<std::size_t>
QsaContinuationStore::stored_index(const HeldPrefix& key) const {
    for (std::size_t i = 0; i < stored_.size(); ++i) {
        if (stored_[i].key.tokens == key.tokens &&
            stored_[i].key.vision_key == key.vision_key) {
            return i;
        }
    }
    return std::nullopt;
}

std::pair<std::size_t, std::uint32_t>
QsaContinuationStore::find_stored(std::span<const TokenId> prompt_tokens,
                                  std::uint64_t prompt_vision_key) const {
    std::pair<std::size_t, std::uint32_t> best{0, 0};
    for (std::size_t i = 0; i < stored_.size(); ++i) {
        if (stored_[i].pages.empty()) { continue; }
        const StoredImage& image = stored_[i];
        // Same token-prefix rule as upstream find_stored (:1336), with the stored image's own
        // media key checked against the prompt's (upstream compares stored.media to prompt.media).
        if (image.key.vision_key != prompt_vision_key) { continue; }
        HeldPrefix prefix{image.key.tokens, image.key.vision_key,
                          static_cast<std::uint32_t>(image.key.tokens.size())};
        const std::uint32_t length =
            qsa_prefix_reuse(prefix, prompt_tokens, prompt_vision_key);
        if (length > best.second) { best = {i, length}; }
    }
    return best;
}

std::pair<std::size_t, std::uint32_t>
QsaContinuationStore::find_stored(std::span<const TokenId> prompt_tokens,
                                  std::span<const std::uint64_t> prompt_media,
                                  bool whole) const {
    std::pair<std::size_t, std::uint32_t> best{0, 0};
    if (prompt_tokens.empty()) { return best; }
    for (std::size_t i = 0; i < stored_.size(); ++i) {
        if (stored_[i].pages.empty()) { continue; }
        const StoredImage& image = stored_[i];
        const std::size_t held   = image.key.tokens.size();
        // Strict prefix, or the whole prompt when the image carries the continuation hidden a
        // regenerate samples from (upstream `whole` for a branched read; here also regenerate).
        if (held == 0 || held > prompt_tokens.size() - (whole ? 0U : 1U)) { continue; }
        if (!std::equal(image.key.tokens.begin(), image.key.tokens.end(), prompt_tokens.begin())) {
            continue;
        }
        // Keyed at the image's own length: the prompt's cumulative Vision key at `held` must be
        // the one the image was written with.
        if (image.key.vision_key != qsa_media_key_at(prompt_media, held)) { continue; }
        if (held > best.second) { best = {i, static_cast<std::uint32_t>(held)}; }
    }
    return best;
}

bool QsaContinuationStore::store_image(const HeldPrefix& key, std::uint64_t image_bytes,
                                       std::uint32_t path_kind, std::uint64_t hidden_bytes) {
    if (key.tokens.size() < kMinStoredTokens || host_budget_ == 0) { return false; }
    if (const auto known = stored_index(key)) {
        stored_[*known].last_used = ++use_clock_;
        return false; // upstream: an already-stored prefix only refreshes last_used
    }
    if (image_bytes > host_budget_ || hidden_bytes > host_budget_ - image_bytes) { return false; }
    if (!make_host_room(page_rounded(image_bytes, page_bytes_) + hidden_bytes)) { return false; }
    StoredImage image;
    image.key          = key;
    image.path_kind    = path_kind;
    image.bytes        = image_bytes;
    image.hidden_bytes = hidden_bytes;
    image.pages        = take_pages(image_bytes);
    if (image.pages.empty()) { return false; }
    if (hidden_bytes != 0) {
        // The hidden lives beside the pages, not in them: it is one BF16 row, and rounding it to
        // an 8 MiB page would cost more than the KV it accompanies on short prompts.
        host_used_ += hidden_bytes;
        image.hidden.resize(static_cast<std::size_t>(hidden_bytes));
    }
    image.last_used = ++use_clock_;
    stored_.push_back(std::move(image));
    return true;
}

void QsaContinuationStore::write_payload(std::size_t index, std::span<const ImageChunk> chunks,
                                         std::span<const std::byte> hidden) {
    if (index >= stored_.size()) { throw std::out_of_range("qsa store: image index"); }
    StoredImage& image = stored_[index];
    const std::uint64_t payload = qsa_image_chunk_bytes(chunks);
    if (payload != image.bytes || hidden.size() != image.hidden_bytes) {
        throw std::logic_error("qsa store: the payload does not match the image");
    }
    std::vector<ImageRegion> regions;
    regions.reserve(chunks.size());
    for (const ImageChunk& chunk : chunks) { regions.push_back(ImageRegion{chunk.data, chunk.bytes}); }
    qsa_copy_image(regions, image.pages, page_bytes_, true);
    if (!hidden.empty()) {
        std::copy(hidden.begin(), hidden.end(), image.hidden.begin());
    }
}

void QsaContinuationStore::read_payload(std::size_t index, std::span<const ImageChunk> chunks,
                                        std::span<std::byte> hidden) const {
    if (index >= stored_.size()) { throw std::out_of_range("qsa store: image index"); }
    const StoredImage& image = stored_[index];
    const std::uint64_t payload = qsa_image_chunk_bytes(chunks);
    if (payload != image.bytes || hidden.size() != image.hidden_bytes) {
        throw std::logic_error("qsa store: the payload does not match the image");
    }
    std::vector<ImageRegion> regions;
    regions.reserve(chunks.size());
    for (const ImageChunk& chunk : chunks) { regions.push_back(ImageRegion{chunk.data, chunk.bytes}); }
    qsa_copy_image(regions, image.pages, page_bytes_, false);
    if (!hidden.empty()) {
        std::copy(image.hidden.begin(), image.hidden.end(), hidden.begin());
    }
}

std::optional<std::size_t> QsaContinuationStore::restore_stored(const HeldPrefix& key) {
    auto index = stored_index(key);
    if (!index) { return std::nullopt; }
    if (stored_[*index].pages.empty()) { return std::nullopt; } // no disk tier in S2
    stored_[*index].last_used = ++use_clock_;
    return index;
}

void QsaContinuationStore::drop_stored_prefixes(std::span<const TokenId> prompt_tokens) {
    for (std::size_t i = stored_.size(); i-- > 0;) {
        const std::vector<TokenId>& tokens = stored_[i].key.tokens;
        if (tokens.size() > prompt_tokens.size() ||
            !std::equal(tokens.begin(), tokens.end(), prompt_tokens.begin())) {
            continue;
        }
        give_back(stored_[i]);
        stored_.erase(stored_.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

} // namespace ninfer::models::qwen3_5::detail
