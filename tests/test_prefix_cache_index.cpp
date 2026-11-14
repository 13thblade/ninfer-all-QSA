#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}


class RecordingBackend final : public PrefixIndexBackend {
public:
    void release_device_block(std::uint32_t id) noexcept override {
        released.push_back(id);
        live.erase(id);
    }

    std::uint32_t allocate() {
        const std::uint32_t id = next++;
        live.insert(id);
        return id;
    }

    std::optional<std::uint32_t> allocate_stream_image(std::uint64_t bytes) noexcept override {
        if (fail_next_stream_image) { fail_next_stream_image = false; return std::nullopt; }
        allocated_stream_images.push_back(bytes);
        return next_stream_image++;
    }

    void release_stream_image(std::uint32_t image) noexcept override {
        released_stream_images.push_back(image);
    }

    std::vector<std::uint32_t> released;
    std::set<std::uint32_t> live;
    std::uint32_t next = 1;
    std::vector<std::uint64_t> allocated_stream_images;
    std::vector<std::uint32_t> released_stream_images;
    std::uint32_t next_stream_image = 100;
    bool fail_next_stream_image      = false;
};

PrefixIndexConfig small_config(std::uint32_t host_slabs = 64, std::uint32_t slots = 2) {
    PrefixIndexConfig config;
    config.max_nodes                 = 256;
    config.max_snapshots             = 32;
    config.host_slabs                = host_slabs;
    config.image_slabs               = 4;
    config.device_snapshot_slots     = slots;
    config.block_bytes               = 1U << 20U;
    config.image_bytes               = 4U << 20U;
    config.cost.token_seconds        = 1.0e-4;
    config.cost.h2d_bytes_per_second = 50.0e9;
    return config;
}

std::vector<TokenId> make_tokens(std::uint32_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> tokens(count);
    for (TokenId& token : tokens) { token = static_cast<TokenId>(rng() % 50000U); }
    return tokens;
}

// Inserts every full block of `tokens` below the root as one sequence would, returning its path.
std::vector<NodeRef> insert_sequence(PrefixCacheIndex& index, RecordingBackend& backend,
                                     const std::vector<TokenId>& tokens) {
    const auto hashes = block_lookup_hashes(tokens, {});
    std::vector<NodeRef> path;
    NodeRef parent;
    for (std::size_t b = 0; b < hashes.size(); ++b) {
        const std::span<const TokenId> block(tokens.data() + b * kBlockTokens, kBlockTokens);
        const std::uint32_t id    = backend.allocate();
        const InsertResult result = index.insert_block(parent, hashes[b], block, 0, id);
        require(result.inserted == result.device_attached || !result.inserted,
                "a new node must own its device id");
        if (!result.device_attached) { backend.release_device_block(id); }
        path.push_back(result.node);
        parent = result.node;
    }
    index.check_invariants();
    return path;
}

SnapshotRef publish_tap(PrefixCacheIndex& index, NodeRef anchor) {
    const auto slot = index.acquire_device_slot();
    require(slot.has_value(), "no device slot for a tap");
    const std::uint32_t frontier = (index.node(anchor).depth + 1U) * kBlockTokens;
    const PublishResult result =
        index.publish_snapshot(anchor, frontier, {}, std::nullopt, *slot, SnapshotKind::Tap);
    require(result.created, "tap snapshot not created");
    index.check_invariants();
    return result.snapshot;
}

// Publishes a snapshot at `frontier` of `tokens`, whose full blocks are `path`; a frontier inside
// a block takes its tail from that block's tokens.
SnapshotRef publish_at(PrefixCacheIndex& index, RecordingBackend& backend,
                       const std::vector<NodeRef>& path, const std::vector<TokenId>& tokens,
                       std::uint32_t frontier, SnapshotKind kind) {
    const std::uint32_t full = frontier / kBlockTokens;
    const std::uint32_t tail = frontier % kBlockTokens;
    const auto slot          = index.acquire_device_slot();
    require(slot.has_value(), "no device slot for a snapshot");
    std::optional<std::uint32_t> tail_id;
    if (tail != 0) { tail_id = backend.allocate(); }
    const PublishResult result = index.publish_snapshot(
        full == 0 ? NodeRef{} : path[full - 1], frontier,
        std::span<const TokenId>(tokens.data() + static_cast<std::size_t>(full) * kBlockTokens,
                                 tail),
        tail_id, *slot, kind);
    require(result.created, "snapshot not created");
    index.check_invariants();
    return result.snapshot;
}

void backup_node(PrefixCacheIndex& index, NodeRef node) {
    index.pin_node(node);
    const auto slab = index.begin_host_fill(node);
    require(slab.has_value(), "no slab for a block backup");
    index.complete_host_fill(node);
    index.unpin_node(node);
    index.check_invariants();
}

void backup_snapshot(PrefixCacheIndex& index, SnapshotRef snapshot) {
    index.pin_snapshot(snapshot);
    require(index.begin_snapshot_host_fill(snapshot), "no slabs for a snapshot backup");
    index.complete_snapshot_host_fill(snapshot);
    index.unpin_snapshot(snapshot);
    index.check_invariants();
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

void test_hashes() {
    const auto tokens = make_tokens(256, 1);
    const auto a      = block_lookup_hashes(tokens, {});
    const auto b      = block_lookup_hashes(tokens, {});
    require(a.size() == 4 && a == b, "block hashes are not deterministic");
    auto changed = tokens;
    changed[130] ^= 1;
    const auto c = block_lookup_hashes(changed, {});
    require(c[0] == a[0] && c[1] == a[1] && c[2] != a[2] && c[3] != a[3],
            "a token change must change its block hash and every later chained hash");
    const std::vector<std::uint64_t> extras{0, 7, 0, 0};
    const auto d = block_lookup_hashes(tokens, extras);
    require(d[0] == a[0] && d[1] != a[1], "the extra key must change the block hash");
    require_throws([&] { (void)block_lookup_hashes(tokens, std::vector<std::uint64_t>{1}); },
                   "short extras");
}

void test_match_and_choice() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto tokens = make_tokens(64 * 8 + 10, 2);
    const std::vector<TokenId> first(tokens.begin(), tokens.begin() + 64 * 6);
    const auto path        = insert_sequence(index, backend, first);
    const SnapshotRef tap2 = publish_tap(index, path[1]); // frontier 128
    const SnapshotRef tap5 = publish_tap(index, path[4]); // frontier 320
    index.release_path(path);
    index.check_invariants();

    const auto hashes = block_lookup_hashes(tokens, {});
    const MatchResult match =
        index.match(tokens, hashes, {}, static_cast<std::uint32_t>(tokens.size()));
    require(match.path.size() == 6, "match did not follow every cached block");
    require(match.candidates.size() == 2 && match.candidates[0].snapshot == tap5 &&
                match.candidates[1].snapshot == tap2,
            "candidates are not deepest first");
    require(match.candidates[0].restore_bytes == 0 && match.candidates[0].image_on_device,
            "device-resident candidate should need no restore");
    const AdmissionChoice choice = index.choose(match, static_cast<std::uint32_t>(tokens.size()));
    require(choice.candidate == 0U, "the deepest device-resident snapshot should be chosen");

    // A prompt that ends exactly at the tap frontier cannot use it (at least one token prefills).
    const MatchResult exact = index.match(tokens, hashes, {}, 320);
    require(exact.candidates.size() == 1 && exact.candidates[0].snapshot == tap2,
            "a snapshot at the prompt end must not be offered");
    require(index.match(tokens, hashes, {}, 321).candidates[0].snapshot == tap5,
            "a snapshot one token before the prompt end must be offered");

    // A different token in block 3 stops the path at block 2 and hides tap5.
    auto edited = tokens;
    edited[64 * 3 + 5] += 1;
    const auto edited_hashes = block_lookup_hashes(edited, {});
    const MatchResult partial =
        index.match(edited, edited_hashes, {}, static_cast<std::uint32_t>(edited.size()));
    require(partial.path.size() == 3 && partial.candidates.size() == 1 &&
                partial.candidates[0].snapshot == tap2,
            "an edit must fall back to the snapshot before it");

    // The extra key (media identity) distinguishes identical tokens.
    std::vector<std::uint64_t> extras(hashes.size(), 0);
    extras[0]               = 99;
    const auto media_hashes = block_lookup_hashes(tokens, extras);
    require(index.match(tokens, media_hashes, extras, static_cast<std::uint32_t>(tokens.size()))
                .path.empty(),
            "different media must not match");
}

void test_collisions_and_tails() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto a = make_tokens(64, 3);
    const auto b = make_tokens(64, 4);
    // Force the same lookup hash for two different blocks: exact comparison must separate them.
    const InsertResult first  = index.insert_block({}, 42, a, 0, backend.allocate());
    const InsertResult second = index.insert_block({}, 42, b, 0, backend.allocate());
    require(first.inserted && second.inserted && !(first.node == second.node),
            "colliding blocks must be distinct nodes");
    require(index.find_child({}, 42, a, 0) == first.node &&
                index.find_child({}, 42, b, 0) == second.node,
            "collision lookup returned the wrong node");
    require(!index.find_child({}, 42, a, 1).has_value(), "extra key mismatch must miss");
    index.check_invariants();

    // Endpoint with a tail: frontier 64 + 10 anchored at `first`.
    std::vector<TokenId> prompt = a;
    const auto tail_tokens      = make_tokens(10, 5);
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    prompt.push_back(8);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(first.node, 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint snapshot not created");
    index.check_invariants();

    std::vector<std::uint64_t> hashes{42};
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].frontier == 74 &&
                match.candidates[0].tail && match.candidates[0].tail_on_device,
            "tail snapshot did not match its exact tail");
    auto other_tail = prompt;
    other_tail[70] += 1;
    require(index.match(other_tail, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                .candidates.empty(),
            "a different tail token must not match");

    // Duplicate publication frees the new slot and tail and returns the existing snapshot.
    const auto slot2              = index.acquire_device_slot();
    const std::uint32_t tail_copy = backend.allocate();
    const PublishResult duplicate = index.publish_snapshot(first.node, 74, tail_tokens, tail_copy,
                                                           *slot2, SnapshotKind::Endpoint);
    require(!duplicate.created && duplicate.snapshot == endpoint.snapshot,
            "duplicate snapshot publication was not deduplicated");
    require(backend.released.back() == tail_copy, "duplicate tail was not released");
    require(index.stats().free_device_slots == 1, "duplicate staging slot was not freed");
    require_throws(
        [&] {
            const auto s = index.acquire_device_slot();
            (void)index.publish_snapshot(first.node, 75, tail_tokens, backend.allocate(), *s,
                                         SnapshotKind::Tap);
        },
        "frontier mismatch");
    index.check_invariants();
}

void test_device_lru_order() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0), backend);
    const auto tokens = make_tokens(64 * 4, 6);
    const auto path   = insert_sequence(index, backend, tokens);
    (void)publish_tap(index, path[3]);
    require(index.device_evictable_blocks() == 0, "pinned blocks must not be evictable");
    index.release_path(path);
    index.check_invariants();
    require(index.device_evictable_blocks() == 4, "released blocks must become evictable");

    // Unbacked: evicting the deepest block first loses only that node and its snapshot.
    const std::uint32_t deepest = index.node(path[3]).device_id;
    require(index.evict_device_blocks(1) == 1, "one block should be released");
    require(backend.released.back() == deepest, "deepest block must be evicted first");
    require(!index.valid(path[3]) && index.valid(path[2]), "only the deepest node is lost");
    require(index.stats().snapshots == 0, "the snapshot on a lost node must be removed");
    index.check_invariants();

    // Re-pinning a path protects it.
    const std::vector<NodeRef> prefix(path.begin(), path.begin() + 3);
    index.acquire_path(prefix);
    require(index.evict_device_blocks(10) == 0, "pinned path must not be evicted");
    index.release_path(prefix);
    require(index.evict_device_blocks(10) == 3, "whole released path must be evictable");
    require(index.stats().nodes == 0 && backend.live.empty(), "every page must be released");
    index.check_invariants();
}

void test_backed_before_unbacked() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64), backend);
    const auto a             = make_tokens(64 * 2, 7);
    const auto b             = make_tokens(64 * 2, 8);
    const auto path_a        = insert_sequence(index, backend, a);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_a = publish_tap(index, path_a[1]);
    const SnapshotRef snap_b = publish_tap(index, path_b[1]);
    backup_node(index, path_b[0]);
    backup_node(index, path_b[1]);
    backup_snapshot(index, snap_b);
    index.release_path(path_a); // older: unbacked
    index.release_path(path_b); // newer: backed
    index.check_invariants();
    require(index.evict_device_blocks(2) == 2, "two backed blocks should be released");
    require(index.valid(path_b[0]) && index.node(path_b[0]).device == CopyState::Absent &&
                index.node(path_b[0]).host == CopyState::Resident,
            "backed block must survive as host-only");
    require(index.node(path_a[0]).device == CopyState::Resident,
            "unbacked blocks must be evicted only after backed ones");
    require(index.valid(snap_a) && index.valid(snap_b), "no snapshot should be lost");

    // The host-only path is still a candidate and reports its restore bytes.
    const auto hashes = block_lookup_hashes(b, {});
    auto prompt       = b;
    prompt.push_back(1);
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].host_only_blocks == 2,
            "host-only path blocks must be counted");
    require(match.candidates[0].restore_bytes == 2U * (1U << 20U),
            "restore bytes must cover host-only blocks (image still on device)");

    // Restore one block to device.
    index.acquire_path(match.path);
    const std::uint32_t id = backend.allocate();
    index.begin_device_fill(match.path[0], id);
    require(index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                    .candidates[0]
                    .filling_blocks == 1,
            "a filling block must be reported");
    index.complete_device_fill(match.path[0]);
    index.release_path(match.path);
    index.check_invariants();
}

void test_host_dead_and_gdsf() {
    RecordingBackend backend;
    // 12 slabs: images need 4 slabs each.
    PrefixCacheIndex index(small_config(12, 3), backend);
    const auto dead_tokens = make_tokens(64 * 4, 9);
    const auto dead_path   = insert_sequence(index, backend, dead_tokens);
    for (const NodeRef node : dead_path) { backup_node(index, node); }
    index.release_path(dead_path); // no snapshot: dead KV
    index.check_invariants();

    const auto a             = make_tokens(64 * 1, 10);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    // Allocating 4 slabs (8 free) needs no reclaim; the next ones reclaim dead blocks first.
    backup_snapshot(index, snap_a);
    require(index.stats().host_dead_reclaims == 0, "no reclaim needed yet");
    index.release_path(path_a);

    const auto b             = make_tokens(64 * 1, 11);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    backup_snapshot(index, snap_b);
    require(index.stats().host_dead_reclaims == 0, "4 slabs were still free");
    backup_node(index, path_b[0]);
    require(index.stats().host_dead_reclaims == 1, "dead KV must be reclaimed before snapshots");
    require(index.valid(snap_a) && index.valid(snap_b),
            "no snapshot should be evicted for dead KV");
    index.release_path(path_b);

    // Hit b so it is more valuable; a new snapshot must evict a (the GDSF minimum).
    index.note_hit(snap_b);
    const auto c             = make_tokens(64 * 1, 12);
    const auto path_c        = insert_sequence(index, backend, c);
    const SnapshotRef snap_c = publish_tap(index, path_c[0]);
    const double before      = index.stats().gdsf_inflation;
    // Consume the remaining dead nodes first, then a snapshot's host copy must go.
    backup_snapshot(index, snap_c);
    require(index.stats().host_dead_reclaims == 4, "all dead nodes must go before snapshots");
    require(index.stats().host_snapshot_evictions == 1, "one snapshot host copy must be evicted");
    // a is the GDSF minimum; it is still complete on the Device, so only its host copy goes.
    require(index.valid(snap_a) && index.snapshot(snap_a).host == CopyState::Absent &&
                index.snapshot(snap_a).device_slot != kNoId,
            "the least valuable snapshot must yield its host copy and keep its device image");
    require(index.valid(snap_b) && index.snapshot(snap_b).host == CopyState::Resident &&
                index.valid(snap_c) && index.snapshot(snap_c).host == CopyState::Resident,
            "more valuable snapshots keep their host copies");
    require(index.stats().gdsf_inflation >= before, "GDSF inflation must be monotonic");

    // Host pressure left a Device-only: it is lost when its slot is taken, so it goes before a
    // backed owner gives up its Device copy, for a snapshot worth as much.
    const auto slot = index.acquire_device_slot();
    require(slot.has_value() && !index.valid(snap_a) && index.valid(snap_b) &&
                index.snapshot(snap_b).device_slot != kNoId &&
                index.snapshot(snap_c).device_slot != kNoId,
            "the Device-only snapshot must lose its slot before a backed one");
    // With no free or Device-only slot left, the least recently hit backed owner keeps only its
    // Host copy.
    const auto backed_slot = index.acquire_device_slot(0.0);
    require(backed_slot.has_value() && index.valid(snap_b) && index.valid(snap_c) &&
                (index.snapshot(snap_b).device_slot == kNoId ||
                 index.snapshot(snap_c).device_slot == kNoId),
            "a backed device slot must be reclaimable whatever the claim");
    index.release_device_slot(*slot);
    index.release_device_slot(*backed_slot);
    index.release_path(path_c);
    index.check_invariants();
}

void test_host_only_reattach() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens      = make_tokens(64 * 3, 30);
    const auto path        = insert_sequence(index, backend, tokens);
    const SnapshotRef snap = publish_tap(index, path[2]);
    for (const NodeRef node : path) { backup_node(index, node); }
    backup_snapshot(index, snap);
    index.release_path(path);
    require(index.evict_device_blocks(3) == 3, "backed blocks must be released");
    for (const NodeRef node : path) {
        require(index.node(node).device == CopyState::Absent, "block must be host-only");
    }
    const std::size_t live_before = backend.live.size();

    // A sequence recomputing the same blocks re-establishes their Device copies.
    const auto again = insert_sequence(index, backend, tokens);
    require(again == path, "recomputed blocks must resolve to the existing nodes");
    require(backend.live.size() == live_before + 3, "the index must keep the recomputed pages");
    for (const NodeRef node : path) {
        const NodeView view = index.node(node);
        require(view.device == CopyState::Resident && view.host == CopyState::Resident,
                "a reattached block is resident on both tiers");
    }
    // Device-resident duplicates stay private to the inserting sequence.
    const auto twice = insert_sequence(index, backend, tokens);
    require(backend.live.size() == live_before + 3, "a resident duplicate must not be retained");
    index.release_path(twice);
    index.release_path(again);
    index.check_invariants();
    require(index.evict_device_blocks(3) == 3, "reattached backed blocks are evictable again");
    index.check_invariants();
}

// A saved Host tier keeps exactly the Host-backed snapshots whose paths are Host-resident, and a
// fresh index rebuilt from it matches the same prompts through host-only blocks.
void test_persistence_roundtrip() {
    RecordingBackend backend;
    PrefixCacheIndex source(small_config(64, 2), backend);
    const auto tokens = make_tokens(64 * 3, 40);
    const auto path   = insert_sequence(source, backend, tokens);
    for (const NodeRef node : path) { backup_node(source, node); }
    const SnapshotRef backed = publish_tap(source, path[1]);
    backup_snapshot(source, backed);
    // A branch point stays one across a restart.
    source.supersede(backed, Supersession::NextTap);
    source.note_hit(backed);
    const auto tail_tokens      = make_tokens(10, 41);
    const auto slot             = source.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        source.publish_snapshot(path[2], 202, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(source, endpoint.snapshot);
    // A Device-only snapshot on an unbacked path is not persistable.
    const auto other      = make_tokens(64, 42);
    const auto other_path = insert_sequence(source, backend, other);
    const auto other_slot = source.acquire_device_slot();
    require(other_slot.has_value(), "no slot for an unbacked snapshot");
    (void)source.publish_snapshot(other_path[0], 64, {}, std::nullopt, *other_slot,
                                  SnapshotKind::Tap);
    source.release_path(path);
    source.release_path(other_path);

    std::vector<NodeRef> nodes;
    std::vector<SnapshotRef> snapshots;
    source.collect_persistable(nodes, snapshots);
    require(nodes.size() == 3 && snapshots.size() == 2,
            "only Host-backed snapshots and their paths are persistable");

    PrefixCacheIndex restored(small_config(64, 2), backend);
    std::vector<std::pair<NodeRef, NodeRef>> mapping;
    const auto mapped = [&](NodeRef ref) {
        for (const auto& [from, to] : mapping) {
            if (from == ref) { return to; }
        }
        return NodeRef{};
    };
    for (const NodeRef node : nodes) {
        const BlockIdentity identity = source.block_identity(node);
        const auto block             = restored.restore_host_block(
            mapped(identity.parent), identity.lookup_hash, identity.tokens, identity.extra);
        require(block.has_value(), "a persisted block did not restore");
        mapping.emplace_back(node, block->node);
    }
    for (const SnapshotRef snapshot : snapshots) {
        const SnapshotView view = source.snapshot(snapshot);
        const auto copy         = restored.restore_host_snapshot(
            mapped(view.anchor), view.frontier, view.tail, view.kind, view.hits, view.branch_point,
            view.periodic);
        require(copy.has_value(), "a persisted snapshot did not restore");
        require(restored.snapshot(*copy).branch_point == (snapshot == backed),
                "a restored snapshot must keep its branch point flag");
        restored.supersede(*copy, Supersession::Resumed);
        require(restored.snapshot(*copy).superseded == (snapshot != backed),
                "a restored branch point must not be superseded");
    }
    restored.check_invariants();

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(3);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        restored.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 2 && match.candidates[0].frontier == 202 &&
                match.candidates[0].host_only_blocks == 3 && !match.candidates[0].image_on_device,
            "a restored Host tier must match through host-only blocks and images");

    // A full Host tier restores what fits and nothing else.
    PrefixCacheIndex tiny(small_config(1, 2), backend);
    const BlockIdentity first = source.block_identity(nodes.front());
    require(tiny.restore_host_block(NodeRef{}, first.lookup_hash, first.tokens, first.extra)
                .has_value(),
            "one slab must hold one block");
    const BlockIdentity second = source.block_identity(nodes[1]);
    require(!tiny.restore_host_block(NodeRef{}, second.lookup_hash, second.tokens, second.extra)
                 .has_value(),
            "restoring must not evict to make room");
    tiny.check_invariants();
}

// The version 6 file keeps image-backed blocks in per-sequence stream images: save collects the
// images' kept runs, restore rebuilds them pinned, and the plan fits both budgets.
void test_stream_persistence_roundtrip() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config();
    config.stream_image_budget_bytes = 8U << 20U; // eight image blocks
    const std::uint64_t block       = 1U << 20U;
    PrefixCacheIndex source(config, backend);
    const auto tokens = make_tokens(64 * 4, 60); // a four-block document
    const auto path   = insert_sequence(source, backend, tokens);
    // A second sequence resumes from block 1 and adds two blocks of its own: its match pins
    // the shared prefix (as the engine's admission does), so its lane path is prefix plus run,
    // and its block hashes chain from the prefix, as the engine's prompt hashes do.
    const auto branch_tokens = make_tokens(64 * 2, 61);
    std::vector<TokenId> branch_prompt(tokens.begin(), tokens.begin() + 2 * kBlockTokens);
    branch_prompt.insert(branch_prompt.end(), branch_tokens.begin(), branch_tokens.end());
    const auto branch_hashes = block_lookup_hashes(branch_prompt, {});
    source.acquire_path(std::span<const NodeRef>(path.data(), 2));
    std::vector<NodeRef> branch;
    std::vector<NodeRef> branch_lane(path.begin(), path.begin() + 2); // shared prefix
    NodeRef parent = path[1];
    for (std::size_t b = 0; b < branch_hashes.size(); ++b) {
        const std::span<const TokenId> block_tokens(
            branch_tokens.data() + b * kBlockTokens, kBlockTokens);
        const std::uint32_t id    = backend.allocate();
        const InsertResult result =
            source.insert_block(parent, branch_hashes[b + 2], block_tokens, 0, id);
        if (!result.device_attached) { backend.release_device_block(id); }
        branch.push_back(result.node);
        branch_lane.push_back(result.node);
        parent = result.node;
    }
    // The lanes release: each unpins its whole lane path, shared prefix plus its own run.
    const auto image_a =
        source.register_stream_image(std::span<const NodeRef>(path.data(), 4), 0, 4, 4 * block);
    require(image_a.has_value(), "no image for the document run");
    const auto image_b =
        source.register_stream_image(std::span<const NodeRef>(branch.data(), 2), 2, 2, 2 * block);
    require(image_b.has_value(), "no image for the branch run");
    // A snapshot on each path; the Host image halves stay in slabs.
    const SnapshotRef tap_a = publish_tap(source, path[3]);
    backup_snapshot(source, tap_a);
    const SnapshotRef tap_b = publish_tap(source, branch[1]);
    backup_snapshot(source, tap_b);
    source.release_path(path);
    source.release_path(branch_lane);
    source.check_invariants();

    std::vector<NodeRef> nodes;
    std::vector<SnapshotRef> snapshots;
    std::vector<PersistableImage> images;
    source.collect_persistable(nodes, snapshots, &images);
    require(nodes.size() == 6 && snapshots.size() == 2 && images.size() == 2,
            "the backed paths and their two images are persistable");
    require(images[0].run.size() == 4 && images[1].run.size() == 2,
            "each image's kept run is its whole run here");

    PrefixCacheIndex restored(config, backend);
    std::vector<std::pair<NodeRef, NodeRef>> mapping;
    const auto mapped = [&](NodeRef ref) {
        for (const auto& [from, to] : mapping) {
            if (from == ref) { return to; }
        }
        return NodeRef{};
    };
    const auto restore_image = [&](std::span<const NodeRef> run, NodeRef parent) {
        std::vector<SavedStreamBlock> saved(run.size());
        for (std::size_t i = 0; i < run.size(); ++i) {
            const BlockIdentity identity = source.block_identity(run[i]);
            saved[i] = {.lookup_hash = identity.lookup_hash,
                        .tokens      = identity.tokens,
                        .extra       = identity.extra};
        }
        const std::optional<RestoredStreamImage> image = restored.restore_stream_image(
            parent, saved, source.node(run.front()).depth, run.size() * block);
        require(image.has_value(), "a persisted image did not restore");
        for (std::size_t i = 0; i < run.size(); ++i) { mapping.emplace_back(run[i], image->nodes[i]); }
    };
    // The slab-backed blocks come first (none here), then the images, shallowest run first.
    restore_image(std::span<const NodeRef>(path.data(), 4), NodeRef{});
    std::vector<NodeRef> branch_run{branch[0], branch[1]};
    restore_image(branch_run, mapped(path[1]));
    for (const SnapshotRef snapshot : snapshots) {
        const SnapshotView view = source.snapshot(snapshot);
        const auto copy         = restored.restore_host_snapshot(
            mapped(view.anchor), view.frontier, view.tail, view.kind, view.hits, view.branch_point,
            view.periodic);
        require(copy.has_value(), "a persisted snapshot did not restore");
    }
    restored.check_invariants();
    require(restored.stats().stream_images == 2 && restored.stats().stream_image_bytes == 6 * block,
            "the restored tier must hold both images in its budget");

    // The restored Host tier matches through host-only blocks and images.
    auto prompt_a = tokens;
    prompt_a.push_back(7);
    const auto hashes_a = block_lookup_hashes(prompt_a, {});
    const MatchResult match_a = restored.match(
        prompt_a, hashes_a, {}, static_cast<std::uint32_t>(prompt_a.size()));
    require(match_a.candidates.size() == 1 &&
                match_a.candidates[0].frontier == 4U * kBlockTokens &&
                match_a.candidates[0].host_only_blocks == 4U &&
                !match_a.candidates[0].image_on_device,
            "the restored tier must match the document through host-only blocks");
    std::vector<TokenId> prompt_b(prompt_a.begin(), prompt_a.begin() + 2 * kBlockTokens);
    prompt_b.insert(prompt_b.end(), branch_tokens.begin(), branch_tokens.end());
    prompt_b.push_back(7);
    const auto hashes_b = block_lookup_hashes(prompt_b, {});
    const MatchResult match_b = restored.match(
        prompt_b, hashes_b, {}, static_cast<std::uint32_t>(prompt_b.size()));
    require(match_b.candidates.size() == 1 &&
                match_b.candidates[0].frontier == 4U * kBlockTokens &&
                match_b.candidates[0].host_only_blocks == 4U &&
                !match_b.candidates[0].image_on_device,
            "the restored tier must match the branch through host-only blocks");

    // The version 6 plan fits both budgets, and degrades by GDSF density against both.
    const auto chosen_blocks = [](const HostRestorePlan& plan) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t index = 0; index < plan.blocks.size(); ++index) {
            if (plan.blocks[index]) { out.push_back(index); }
        }
        return out;
    };
    const std::vector<std::int32_t> parents = {-1, 0, 1, 2, 1, 4};
    const std::vector<std::uint8_t> image_backed = {1, 1, 1, 1, 1, 1};
    const std::vector<SavedSnapshotShape> shapes = {
        {.anchor = 3, .frontier = 4U * 64U, .slabs = 4, .hits = 0},
        {.anchor = 5, .frontier = 4U * 64U, .slabs = 4, .hits = 1},
    };
    const HostRestorePlan all = plan_host_restore_v6(parents, image_backed, shapes, 64,
                                                     8U << 20U, block, config);
    require(all.slabs == 8 && all.image_bytes == 6U * block &&
                std::all_of(all.blocks.begin(), all.blocks.end(), [](bool b) { return b; }) &&
                std::all_of(all.snapshots.begin(), all.snapshots.end(), [](bool b) { return b; }),
            "a tier that holds the whole file must restore all of it");
    const HostRestorePlan one = plan_host_restore_v6(parents, image_backed, shapes, 64,
                                                     4U << 20U, block, config);
    require(!one.snapshots[0] && one.snapshots[1] && one.slabs == 4 && one.image_bytes == 4U * block,
            "a small tier must keep the most used snapshot");
    require(chosen_blocks(one) == std::vector<std::uint32_t>{0, 1, 4, 5},
            "a small tier must restore exactly the kept snapshot's path");
    const HostRestorePlan none = plan_host_restore_v6(parents, image_backed, shapes, 64,
                                                      3U << 20U, block, config);
    require(!none.snapshots[0] && !none.snapshots[1] && none.slabs == 0 && none.image_bytes == 0,
            "a tier that cannot hold a path must restore nothing");
}

void test_tail_device_fill() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens = make_tokens(64, 31);
    const auto path   = insert_sequence(index, backend, tokens);
    backup_node(index, path[0]);
    const auto tail_tokens      = make_tokens(10, 32);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint not created");
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 5,
            "a host-backed tail needs its own slab");
    index.release_path(path);
    // While its image holds a Device slot, the snapshot keeps its tail on the Device: only the
    // block goes.
    require(index.evict_device_blocks(2) == 1 && backend.live.count(tail_id) == 1 &&
                index.snapshot(endpoint.snapshot).tail_device_copy == CopyState::Resident,
            "a snapshot holding a Device slot must keep its tail on the Device");
    // Its slot goes to another snapshot once the free one is taken (it keeps its Host copy); then
    // the tail can go too, and the host-backed snapshot survives.
    const auto free_slot = index.acquire_device_slot();
    const auto taken     = index.acquire_device_slot();
    require(free_slot.has_value() && taken.has_value() &&
                index.snapshot(endpoint.snapshot).device_slot == kNoId,
            "a backed owner gives up its Device slot");
    index.release_device_slot(*free_slot);
    index.release_device_slot(*taken);
    require(index.evict_device_blocks(1) == 1, "the tail must be evictable once the slot is gone");
    require(backend.live.count(tail_id) == 0, "the tail's device page must be released");
    const SnapshotView evicted = index.snapshot(endpoint.snapshot);
    require(index.valid(endpoint.snapshot) && evicted.tail_device_copy == CopyState::Absent,
            "a backed tail survives host-only");

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].tail &&
                !match.candidates[0].tail_on_device,
            "a host-only tail must be reported");

    // Abort then complete a restore of the tail.
    index.pin_snapshot(endpoint.snapshot);
    require_throws([&] { index.complete_tail_device_fill(endpoint.snapshot); }, "no fill open");
    const std::uint32_t first = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, first);
    index.abort_tail_device_fill(endpoint.snapshot);
    require(backend.live.count(first) == 0, "an aborted tail fill must return its page");
    const std::uint32_t second = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, second);
    index.complete_tail_device_fill(endpoint.snapshot);
    index.unpin_snapshot(endpoint.snapshot);
    index.check_invariants();
    const SnapshotView restored = index.snapshot(endpoint.snapshot);
    require(restored.tail_device_copy == CopyState::Resident && restored.tail_device == second,
            "a completed tail fill is resident");
    require(index.evict_device_blocks(1) == 1 && backend.live.count(second) == 0,
            "a restored tail is evictable again");
    index.check_invariants();
}

void test_device_slots() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 1), backend);
    const auto a             = make_tokens(64, 13);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    require(!index.acquire_device_slot(0.0).has_value(),
            "an unbacked slot must not be taken for a snapshot worth less");
    backup_snapshot(index, snap_a);
    const auto slot = index.acquire_device_slot();
    require(slot.has_value(), "a backed slot must be reusable");
    require(index.valid(snap_a) && index.snapshot(snap_a).device_slot == kNoId,
            "the evicted slot's snapshot must survive host-only");
    index.release_device_slot(*slot);

    // Unbacked with permission: the owner is lost.
    const auto b             = make_tokens(64, 14);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    const auto forced        = index.acquire_device_slot();
    require(forced.has_value() && !index.valid(snap_b), "unbacked slot eviction loses its owner");
    index.release_device_slot(*forced);
    index.release_path(path_a);
    index.release_path(path_b);
    index.check_invariants();
}

void test_supersession() {
    RecordingBackend backend;
    // 20 slabs: a tail-less image takes 4, an image with a tail 5. Five Device slots, one per
    // snapshot below.
    PrefixCacheIndex index(small_config(20, 5), backend);
    const auto tokens = make_tokens(64 * 12 + 1, 30);
    const auto path   = insert_sequence(index, backend, tokens);
    const SnapshotRef shared =
        publish_at(index, backend, path, tokens, 64 * 2, SnapshotKind::Boundary);
    const SnapshotRef first =
        publish_at(index, backend, path, tokens, 64 * 3 + 10, SnapshotKind::Endpoint);
    const SnapshotRef second =
        publish_at(index, backend, path, tokens, 64 * 7 + 20, SnapshotKind::Endpoint);
    // A second branch sharing five blocks continues past the boundary differently: from here on
    // it serves two conversations.
    std::vector<TokenId> other(tokens.begin(), tokens.begin() + 64 * 5);
    const auto suffix = make_tokens(64 * 4 + 1, 31);
    other.insert(other.end(), suffix.begin(), suffix.end());
    const auto other_path = insert_sequence(index, backend, other);
    index.supersede(shared, Supersession::Resumed);
    require(!index.snapshot(shared).superseded,
            "a boundary another conversation continued from must not be superseded");
    index.supersede(first, Supersession::Resumed);
    require(index.snapshot(first).superseded, "the resumed snapshot must be superseded");
    index.note_hit(first);
    require(!index.snapshot(first).superseded && !index.snapshot(first).branch_point,
            "a hit must retain a superseded snapshot again, until its new lineage moves on");
    // Hits make `first` the most valuable snapshot by GDSF; superseding it must still put it first.
    for (int hit = 0; hit < 4; ++hit) { index.note_hit(first); }
    index.supersede(first, Supersession::Resumed);
    require(index.snapshot(first).superseded, "the new lineage moving on must supersede it again");
    index.check_invariants();

    // The second branch hangs below `first`, whose tail lies in a shared block.
    const SnapshotRef branch =
        publish_at(index, backend, other_path, other, 64 * 6 + 3, SnapshotKind::Endpoint);

    for (const SnapshotRef snapshot : {shared, first, second, branch}) {
        backup_snapshot(index, snapshot);
    }
    require(index.stats().host_free_slabs == 1, "the Host tier must be nearly full");
    const SnapshotRef tip = publish_at(index, backend, path, tokens, 64 * 11, SnapshotKind::Tap);
    backup_snapshot(index, tip);
    require(index.valid(first) && index.snapshot(first).host == CopyState::Absent &&
                index.snapshot(first).device_slot != kNoId,
            "the superseded snapshot must give up its Host copy first, and only that while it is "
            "complete on the Device");
    require(index.stats().superseded_evictions == 1 && index.stats().host_snapshot_evictions == 1,
            "exactly one superseded eviction");
    for (const SnapshotRef snapshot : {shared, second, branch, tip}) {
        require(index.valid(snapshot) && index.snapshot(snapshot).host == CopyState::Resident,
                "retained snapshots must keep their Host copies");
    }
    // A new snapshot takes the superseded owner's Device slot first, and without a Host copy the
    // superseded snapshot is gone.
    const auto slot = index.acquire_device_slot();
    require(slot.has_value() && !index.valid(first),
            "the superseded snapshot must give up its Device slot first");
    index.release_device_slot(*slot);

    // A tap that only its own prompt's next tap moved past, hit by a request that diverged from
    // that prompt after it, is a branch point: nothing supersedes it again.
    index.supersede(tip, Supersession::NextTap);
    index.note_hit(tip);
    require(index.snapshot(tip).branch_point && !index.snapshot(tip).superseded,
            "a hit on a tap its prompt's next tap superseded must make it a branch point");
    index.supersede(tip, Supersession::Resumed);
    index.supersede(tip, Supersession::NextTap);
    require(!index.snapshot(tip).superseded, "a branch point must never be superseded");
    index.release_path(path);
    index.release_path(other_path);
    index.check_invariants();
}

// Without a Host tier a slot's owner is lost when the slot is taken, so value decides (§9.2): a
// short request's snapshot must not push out the one a long conversation resumes from, and a
// snapshot worth less than every owner takes no slot.
void test_unbacked_slot_values() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0, 2), backend);
    const auto long_tokens = make_tokens(64 * 20, 50);
    const auto long_path   = insert_sequence(index, backend, long_tokens);
    const SnapshotRef long_end =
        publish_at(index, backend, long_path, long_tokens, 64 * 20, SnapshotKind::Endpoint);
    const auto short_tokens = make_tokens(64 * 2, 51);
    const auto short_path   = insert_sequence(index, backend, short_tokens);
    const SnapshotRef short_end =
        publish_at(index, backend, short_path, short_tokens, 64 * 2, SnapshotKind::Endpoint);

    // Another short request's endpoint may take the short endpoint's slot, never the long one's.
    const auto other      = make_tokens(64 * 2, 52);
    const auto other_path = insert_sequence(index, backend, other);
    const auto slot       = index.acquire_device_slot(index.estimate_priority(0, 128, false));
    require(slot.has_value() && index.valid(long_end) && !index.valid(short_end),
            "the least valuable unbacked snapshot must lose its slot first");
    const SnapshotRef other_end =
        index.publish_snapshot(other_path[1], 128, {}, std::nullopt, *slot, SnapshotKind::Endpoint)
            .snapshot;
    require(!index.acquire_device_slot(0.0).has_value() && index.valid(long_end) &&
                index.valid(other_end),
            "a snapshot worth less than every owner must not displace one");
    index.release_path(long_path);
    index.release_path(short_path);
    index.release_path(other_path);
    index.check_invariants();
}

// A refused snapshot advances GDSF time as if it had been inserted and evicted (§9.2, §9.3):
// owners that nothing hits again age, and a later snapshot of the same value displaces the least
// valuable of them instead of every new snapshot being refused for good. Decision reads about new
// states hit each state's snapshot two or three times and then never again.
void test_refusals_age_owners() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0, 2), backend);
    std::vector<std::vector<NodeRef>> paths;
    std::vector<SnapshotRef> owners;
    for (std::uint32_t seed = 60; seed < 62; ++seed) {
        const auto tokens = make_tokens(64 * 2, seed);
        paths.push_back(insert_sequence(index, backend, tokens));
        owners.push_back(
            publish_at(index, backend, paths.back(), tokens, 64 * 2, SnapshotKind::Boundary));
        index.note_hit(owners.back());
        index.note_hit(owners.back());
    }
    const auto tokens = make_tokens(64 * 2, 62);
    paths.push_back(insert_sequence(index, backend, tokens));
    const double before = index.stats().gdsf_inflation;
    require(!index.acquire_device_slot(index.estimate_priority(0, 128, false)).has_value(),
            "a snapshot worth less than every owner must not displace one");
    require(index.stats().gdsf_inflation > before, "a refusal must advance GDSF time");
    std::optional<std::uint32_t> slot;
    int attempts = 1;
    while (!slot && attempts < 6) {
        slot = index.acquire_device_slot(index.estimate_priority(0, 128, false));
        ++attempts;
    }
    require(slot.has_value() && attempts >= 3,
            "owners hit twice must yield to a snapshot of their value after two refusals");
    require(index.valid(owners[0]) != index.valid(owners[1]),
            "exactly one aged owner must lose its slot");
    const SnapshotRef fresh =
        index.publish_snapshot(paths.back()[1], 128, {}, std::nullopt, *slot, SnapshotKind::Boundary)
            .snapshot;
    require(index.valid(fresh), "the new snapshot must be published");
    for (const auto& path : paths) { index.release_path(path); }
    index.check_invariants();
}

// A snapshot's Host write only displaces snapshots worth no more than itself (§9.3): a tap just
// past a conversation's boundary must not push another conversation's boundary out of the Host
// tier. Rejected, it stays Device-only.
void test_host_write_admission() {
    RecordingBackend backend;
    // 4 slabs: one tail-less image.
    PrefixCacheIndex index(small_config(4, 4), backend);
    const auto a      = make_tokens(64 * 8, 56);
    const auto path_a = insert_sequence(index, backend, a);
    const SnapshotRef valuable =
        publish_at(index, backend, path_a, a, 64 * 8, SnapshotKind::Boundary);
    backup_snapshot(index, valuable);
    const auto b            = make_tokens(64 * 1, 57);
    const auto path_b       = insert_sequence(index, backend, b);
    const SnapshotRef cheap = publish_tap(index, path_b[0]);
    index.pin_snapshot(cheap);
    require(!index.begin_snapshot_host_fill(cheap),
            "a snapshot worth less must not displace a Host snapshot worth more");
    index.unpin_snapshot(cheap);
    require(index.snapshot(valuable).host == CopyState::Resident &&
                index.snapshot(cheap).host == CopyState::Absent,
            "the rejected snapshot stays Device-only and the valuable one keeps its Host copy");
    index.release_path(path_a);
    index.release_path(path_b);
    index.check_invariants();
}

// A boundary no other conversation continued from is just its lineage's snapshot: superseded
// when the lineage moves past it, and retained again by a later hit.
void test_unshared_boundary() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(20, 4), backend);
    const auto tokens = make_tokens(64 * 6, 58);
    const auto path   = insert_sequence(index, backend, tokens);
    const SnapshotRef boundary =
        publish_at(index, backend, path, tokens, 64 * 2 + 5, SnapshotKind::Boundary);
    index.supersede(boundary, Supersession::Resumed);
    require(index.snapshot(boundary).superseded,
            "a boundary no other conversation shares must be superseded like any snapshot");
    index.note_hit(boundary);
    require(!index.snapshot(boundary).superseded, "a hit must retain it again");
    index.release_path(path);
    index.check_invariants();
}

// Hits count only while they are recent (§9.3): once a third of the block capacity (Device pages
// plus Host slabs) has been published since a snapshot's last hit, half of them count, and after
// two thirds GDSF values it as never hit. A document asked about nine times and then left must not
// outrank the conversations that follow it for good (issue #222), while one still being asked
// about keeps its rank even when a round of conversations that missed passes one window between
// two of its questions (issue #248).
void test_hits_expire() {
    enum class Age { Recent, OneWindow, TwoWindows };
    // Plain: a conversation's own snapshot hit right after its publication. Burst: a branch point
    // asked about in one burst right after its publication. Periodic: a branch point first asked
    // about after a pause. Returning: a conversation's own snapshot first hit after a pause.
    enum class Kind { Plain, Burst, Periodic, Returning };
    for (const Kind kind : {Kind::Plain, Kind::Burst, Kind::Periodic, Kind::Returning}) {
        const bool paused = kind == Kind::Periodic || kind == Kind::Returning;
        const bool branch = kind == Kind::Burst || kind == Kind::Periodic;
        for (const Age age : {Age::Recent, Age::OneWindow, Age::TwoWindows}) {
            RecordingBackend backend;
            // 8 slabs: two tail-less images. max_nodes 256: the window is 85 published blocks.
            PrefixCacheIndex index(small_config(8, 4), backend);
            // 48 blocks of other content, published before the document or between its
            // publication and its first hit: either way the hits happen at clock 56.
            const auto pause = make_tokens(64 * 48, 79);
            std::vector<NodeRef> pause_path;
            if (!paused) { pause_path = insert_sequence(index, backend, pause); }
            const auto doc      = make_tokens(64 * 4, 80);
            const auto doc_path = insert_sequence(index, backend, doc);
            const SnapshotRef document =
                publish_at(index, backend, doc_path, doc, 64 * 4, SnapshotKind::Tap);
            if (paused) { pause_path = insert_sequence(index, backend, pause); }
            if (branch) {
                // Its own prompt moved past it, then another request resumed from it: the questions
                // about a document branch here.
                index.supersede(document, Supersession::NextTap);
            }
            for (int hit = 0; hit < 9; ++hit) { index.note_hit(document); }
            require(index.snapshot(document).hits == 9 &&
                        index.snapshot(document).branch_point == branch &&
                        index.snapshot(document).periodic == paused,
                    "hits count while recent; a hit after half a window marks a periodic one");
            const auto chat      = make_tokens(64 * 4, 81);
            const auto chat_path = insert_sequence(index, backend, chat);
            const SnapshotRef conversation =
                publish_at(index, backend, chat_path, chat, 64 * 4, SnapshotKind::Endpoint);
            backup_snapshot(index, document);
            backup_snapshot(index, conversation);
            // A periodic snapshot keeps half its hits for a second window, a branch point or not;
            // one hit in a burst keeps none past the first, so it does not hold the cache longer
            // than the hits that earned it.
            const std::uint32_t half = paused ? 4U : 0U;
            if (age != Age::Recent) {
                // The document's hits, counted at clock 56, count fully until 141 (56 + 85) and a
                // periodic snapshot's half until 226. The conversation brings the clock to 64;
                // 72 blocks bring it to 136, 157 to 221; the next snapshot adds 8.
                const std::uint32_t blocks = age == Age::OneWindow ? 72U : 157U;
                const auto other           = make_tokens(64 * blocks, 82);
                const auto other_path      = insert_sequence(index, backend, other);
                require(index.snapshot(document).hits == (age == Age::OneWindow ? 9U : half),
                        "the window has not passed yet, or a periodic snapshot's half counts");
                index.release_path(other_path);
            }
            const auto next      = make_tokens(64 * 4, 83);
            const auto next_path = insert_sequence(index, backend, next);
            const SnapshotRef fresh =
                publish_at(index, backend, next_path, next, 64 * 4, SnapshotKind::Endpoint);
            const std::uint32_t counted = age == Age::Recent ? 9U : age == Age::OneWindow ? half : 0U;
            require(index.snapshot(document).hits == counted,
                    "all hits count within one window, a periodic snapshot's half within two, "
                    "none after");
            backup_snapshot(index, fresh);
            // Equal C / Z: the counted hits decide while there are any, and recency once they expired.
            const bool kept = counted > 0;
            require(index.snapshot(document).host == (kept ? CopyState::Resident : CopyState::Absent) &&
                        index.snapshot(conversation).host ==
                            (kept ? CopyState::Absent : CopyState::Resident),
                    kept ? "a document asked about within its counted hits must keep its Host copy"
                         : "a snapshot whose hits expired must yield its Host copy");
            index.note_hit(document);
            require(index.snapshot(document).hits == counted + 1U,
                    "a hit counts on top of the hits still counted");
            require(index.snapshot(document).periodic ==
                        (paused || age != Age::Recent),
                    "a hit after a window's pause marks a document asked about periodically");
            index.release_path(pause_path);
            index.release_path(doc_path);
            index.release_path(chat_path);
            index.release_path(next_path);
            index.check_invariants();
        }
    }
}

// A conversation is resumed once per turn, and each turn publishes new snapshots: they carry the
// hits the snapshot they superseded still counted, so the lineage keeps its rank turn after turn
// (§9.3). Only an endpoint gives way to an endpoint: a client that re-renders the reply resumes
// its next turn from the tap again.
void test_lineage_hits() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 6), backend);
    const auto tokens = make_tokens(64 * 12 + 1, 90);
    const auto path   = insert_sequence(index, backend, tokens);
    SnapshotRef resume = publish_at(index, backend, path, tokens, 64 * 2, SnapshotKind::Tap);
    for (std::uint32_t turn = 1; turn <= 3; ++turn) {
        index.note_hit(resume);
        const std::uint32_t carried = index.supersede(resume, Supersession::Resumed);
        require(carried == turn, "a superseded snapshot passes on the hits it counted");
        require(index.estimate_priority(64 * 2 * turn, 64 * 2 * (turn + 1), false, carried) >
                    index.estimate_priority(64 * 2 * turn, 64 * 2 * (turn + 1), false),
                "a lineage's hits raise the claim of its next snapshot");
        const auto slot = index.acquire_device_slot();
        require(slot.has_value(), "no slot for the next turn's tap");
        resume = index
                     .publish_snapshot(path[2 * (turn + 1) - 1], 64 * 2 * (turn + 1), {},
                                       std::nullopt, *slot, SnapshotKind::Tap, carried)
                     .snapshot;
        require(index.snapshot(resume).hits == turn, "the next turn's tap starts with them");
    }
    // A turn that publishes only its endpoint leaves the tap it resumed from retained.
    index.note_hit(resume);
    const SnapshotRef endpoint =
        publish_at(index, backend, path, tokens, 64 * 9 + 7, SnapshotKind::Endpoint);
    require(index.supersede(resume, Supersession::Endpoint) == 0 &&
                !index.snapshot(resume).superseded,
            "an endpoint must not supersede the tap a re-rendering client resumes from");
    // An echoing client resumes from the endpoint, and its next endpoint moves past it.
    index.note_hit(endpoint);
    require(index.supersede(endpoint, Supersession::Endpoint) == 1 &&
                index.snapshot(endpoint).superseded,
            "an endpoint must supersede the endpoint its lineage resumed from");
    require(index.supersede(endpoint, Supersession::Endpoint) == 0,
            "a superseded snapshot passes on nothing again");
    index.release_path(path);
    index.check_invariants();
}

// The production failure: a conversation resuming turn after turn under Host pressure, next to a
// deep stale conversation. Each turn resumes from the previous turn's endpoint and publishes a
// deeper one; the new endpoint must always survive to serve the next turn.
void test_lineage_under_pressure() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(16, 4), backend);
    const auto stale_tokens = make_tokens(64 * 40 + 1, 40);
    const auto stale_path   = insert_sequence(index, backend, stale_tokens);
    const SnapshotRef stale =
        publish_at(index, backend, stale_path, stale_tokens, 64 * 40 - 7, SnapshotKind::Endpoint);
    backup_snapshot(index, stale);
    index.release_path(stale_path);

    const auto tokens = make_tokens(64 * 30 + 1, 41);
    const auto path   = insert_sequence(index, backend, tokens);
    // An aligned tap early in the conversation (the Legacy valuation anchored every later
    // endpoint to it and kept evicting them).
    SnapshotRef resume = publish_at(index, backend, path, tokens, 64 * 4, SnapshotKind::Tap);
    backup_snapshot(index, resume);
    for (std::uint32_t turn = 1; turn <= 8; ++turn) {
        const std::uint32_t frontier = 64 * 4 + turn * 200 + 13;
        index.note_hit(resume);
        const SnapshotRef tip =
            publish_at(index, backend, path, tokens, frontier, SnapshotKind::Endpoint);
        index.supersede(resume, Supersession::Resumed);
        backup_snapshot(index, tip);
        // Other traffic writes to the Host tier before this conversation's next turn arrives (a
        // short new conversation here): that allocation used to evict the newest endpoint.
        const auto noise_tokens = make_tokens(64 * 2 + 1, 100 + turn);
        const auto noise_path   = insert_sequence(index, backend, noise_tokens);
        backup_snapshot(index, publish_at(index, backend, noise_path, noise_tokens, 64 * 2 - 5,
                                          SnapshotKind::Endpoint));
        index.release_path(noise_path);
        require(index.valid(tip) && index.snapshot(tip).host == CopyState::Resident,
                "the newest endpoint must keep its Host copy");
        const std::vector<TokenId> next(tokens.begin(), tokens.begin() + frontier + 50);
        const auto hashes = block_lookup_hashes(next, {});
        const MatchResult match =
            index.match(next, hashes, {}, static_cast<std::uint32_t>(next.size()));
        require(!match.candidates.empty() && match.candidates.front().snapshot == tip,
                "the next turn must resume from the newest endpoint");
        resume = tip;
    }
    require(index.stats().superseded_evictions >= 6, "superseded endpoints must make the room");
    index.release_path(path);
    index.check_invariants();
}

// Questions about one long document (issue #222): every request sends the document with a
// different short question, one request at a time. The first request's deepest ladder tap serves
// every later question; each question's own tap holds the question in its tail and serves nobody
// else. The tap must survive the first request's block write-through into a Host tier too small
// for its blocks, and then stay a branch point however many questions resume from it.
void test_questions_about_one_document() {
    RecordingBackend backend;
    // 20 slabs hold four tail-less images and one with a tail (21) only after one is evicted, and
    // the 14 blocks need more than is left. Three Device slots.
    PrefixCacheIndex index(small_config(20, 3), backend);
    const auto document                     = make_tokens(64 * 14 + 10, 70);
    const std::vector<std::uint32_t> ladder = {64 * 4, 64 * 8, 64 * 10, 64 * 12};
    const std::uint32_t question_tap        = 64 * 14 + 20;
    std::vector<SnapshotRef> first_taps;
    // One request as the hybrid program serves it: resume from the chosen snapshot, capture the
    // planned taps past it (each superseding the snapshot it resumed from and the lane's previous
    // tap), write each image through to the Host tier when it fits, then at release the lane's
    // blocks, and unpin its path. Returns the tokens reused.
    const auto serve = [&](std::uint32_t question) {
        auto prompt      = document;
        const auto words = make_tokens(30, 1000 + question);
        prompt.insert(prompt.end(), words.begin(), words.end());
        const auto hashes            = block_lookup_hashes(prompt, {});
        const auto n                 = static_cast<std::uint32_t>(prompt.size());
        const MatchResult match      = index.match(prompt, hashes, {}, n);
        const AdmissionChoice choice = index.choose(match, n);
        SnapshotRef resume;
        std::uint32_t reused = 0;
        if (choice.candidate) {
            resume = match.candidates[*choice.candidate].snapshot;
            reused = match.candidates[*choice.candidate].frontier;
            index.note_hit(resume);
        }
        const auto path = insert_sequence(index, backend, prompt);
        SnapshotRef previous;
        std::uint32_t base                   = reused;
        std::vector<std::uint32_t> frontiers = ladder;
        frontiers.push_back(question_tap);
        for (const std::uint32_t frontier : frontiers) {
            if (frontier <= reused) { continue; }
            if (resume.valid()) {
                index.supersede(resume, Supersession::Resumed);
                resume = {};
            }
            if (previous.valid()) { index.supersede(previous, Supersession::NextTap); }
            const std::uint32_t tail = frontier % kBlockTokens;
            const auto slot =
                index.acquire_device_slot(index.estimate_priority(base, frontier, tail != 0));
            if (!slot) { continue; }
            std::optional<std::uint32_t> tail_id;
            if (tail != 0) { tail_id = backend.allocate(); }
            const PublishResult published = index.publish_snapshot(
                path[frontier / kBlockTokens - 1], frontier,
                std::span<const TokenId>(prompt.data() + frontier - tail, tail), tail_id, *slot,
                SnapshotKind::Tap);
            require(published.created, "a tap of a new question must be new");
            index.pin_snapshot(published.snapshot);
            if (index.begin_snapshot_host_fill(published.snapshot)) {
                index.complete_snapshot_host_fill(published.snapshot);
            }
            index.unpin_snapshot(published.snapshot);
            if (question == 0) { first_taps.push_back(published.snapshot); }
            previous = published.snapshot;
            base     = frontier;
        }
        for (const NodeRef node : path) {
            if (index.node(node).host != CopyState::Absent) { continue; }
            index.pin_node(node);
            const bool written = index.begin_host_fill(node).has_value();
            if (written) { index.complete_host_fill(node); }
            index.unpin_node(node);
            if (!written) { break; }
        }
        index.release_path(path);
        index.check_invariants();
        return reused;
    };
    require(serve(0) == 0, "the first question has nothing to resume from");
    require(index.stats().host_free_slabs < 4, "the blocks must fill the Host tier");
    require(!index.valid(first_taps[0]) && !index.valid(first_taps[1]),
            "superseded taps without a Device copy must make the room");
    require(index.valid(first_taps[3]) && index.snapshot(first_taps[3]).device_slot != kNoId,
            "the deepest ladder tap keeps its Device copy");
    for (std::uint32_t question = 1; question <= 8; ++question) {
        require(serve(question) == 64 * 12,
                "question " + std::to_string(question) + " must resume from the ladder tap");
        require(!index.snapshot(first_taps[3]).superseded,
                "a tap that later questions resume from must stay retained");
    }
}

// The first question about a new document when other conversations hold the Device slots (here
// one slot): each tap of the prompt takes the slot of the tap before it, which keeps only its Host
// copy. The lane's block write-through at release evicts superseded taps first, but not the newest
// one: the next question resumes from it, and the blocks being written are its own (§9.3).
void test_write_through_keeps_superseded_tap() {
    RecordingBackend backend;
    // 20 slabs: three images and the question tap's tail (13) leave 7 for the 14 blocks.
    PrefixCacheIndex index(small_config(20, 1), backend);
    const auto document = make_tokens(64 * 14 + 10, 71);
    SnapshotRef early_tap;
    SnapshotRef ladder_tap;
    const auto serve = [&](std::uint32_t question) {
        auto prompt      = document;
        const auto words = make_tokens(30, 2000 + question);
        prompt.insert(prompt.end(), words.begin(), words.end());
        const auto hashes            = block_lookup_hashes(prompt, {});
        const auto n                 = static_cast<std::uint32_t>(prompt.size());
        const MatchResult match      = index.match(prompt, hashes, {}, n);
        const AdmissionChoice choice = index.choose(match, n);
        SnapshotRef resume;
        std::uint32_t reused = 0;
        if (choice.candidate) {
            resume = match.candidates[*choice.candidate].snapshot;
            reused = match.candidates[*choice.candidate].frontier;
            index.note_hit(resume);
        }
        const auto path = insert_sequence(index, backend, prompt);
        SnapshotRef previous;
        SnapshotRef superseded;
        for (const std::uint32_t frontier : {64U * 8U, 64U * 12U, 64U * 14U + 20U}) {
            if (frontier <= reused) { continue; }
            if (resume.valid()) {
                index.supersede(resume, Supersession::Resumed);
                resume = {};
            }
            if (previous.valid()) {
                index.supersede(previous, Supersession::NextTap);
                superseded = previous;
            }
            const std::uint32_t tail = frontier % kBlockTokens;
            const auto slot          = index.acquire_device_slot();
            require(slot.has_value(), "a tap takes the slot of the tap before it");
            std::optional<std::uint32_t> tail_id;
            if (tail != 0) { tail_id = backend.allocate(); }
            const SnapshotRef tap =
                index
                    .publish_snapshot(
                        path[frontier / kBlockTokens - 1], frontier,
                        std::span<const TokenId>(prompt.data() + frontier - tail, tail), tail_id,
                        *slot, SnapshotKind::Tap)
                    .snapshot;
            index.pin_snapshot(tap);
            const bool backed = index.begin_snapshot_host_fill(tap);
            if (backed) { index.complete_snapshot_host_fill(tap); }
            index.unpin_snapshot(tap);
            require(backed || question != 0, "the first question's taps fit the Host tier");
            if (question == 0 && frontier == 64U * 8U) { early_tap = tap; }
            if (question == 0 && frontier == 64U * 12U) { ladder_tap = tap; }
            previous = tap;
        }
        for (const NodeRef node : path) {
            if (index.node(node).host != CopyState::Absent) { continue; }
            index.pin_node(node);
            const bool written = index.begin_host_fill(node, superseded).has_value();
            if (written) { index.complete_host_fill(node); }
            index.unpin_node(node);
            if (!written) { break; }
        }
        index.release_path(path);
        index.check_invariants();
        return reused;
    };
    require(serve(0) == 0, "the first question has nothing to resume from");
    require(!index.valid(early_tap), "an older superseded tap must make room for the blocks");
    require(index.valid(ladder_tap) && index.snapshot(ladder_tap).device_slot == kNoId &&
                index.snapshot(ladder_tap).host == CopyState::Resident,
            "the write-through must keep the Host copy of the lane's newest superseded tap");
    require(serve(1) == 64 * 12, "the next question must resume from the ladder tap");
}

// A snapshot whose image holds a Device slot keeps its tail on the Device, so a Host eviction
// takes only its Host copy. Before, Device pressure evicted the tail first, and a lane's block
// write-through then removed the snapshot whole: another conversation's resume point, whose image
// was still on the Device. A 4 GiB Host tier served issue #222's rotation less than 2 GiB did.
void test_slot_snapshot_keeps_tail() {
    RecordingBackend backend;
    // 5 slabs: one image with its tail.
    PrefixCacheIndex index(small_config(5, 2), backend);
    const auto chat      = make_tokens(64 * 2 + 10, 90);
    const auto chat_path = insert_sequence(index, backend, chat);
    for (const NodeRef node : chat_path) { index.pin_node(node); }
    const SnapshotRef resume =
        publish_at(index, backend, chat_path, chat, 64 * 2 + 10, SnapshotKind::Endpoint);
    for (const NodeRef node : chat_path) { index.unpin_node(node); }
    backup_snapshot(index, resume);
    index.release_path(chat_path);
    // Device pressure takes the unbacked blocks' only copies last; the tail is not offered.
    require(index.device_evictable_blocks() == 2,
            "a Device-slot snapshot's tail must not be a Device eviction candidate");
    const auto other      = make_tokens(64, 91);
    const auto other_path = insert_sequence(index, backend, other);
    index.pin_node(other_path[0]);
    const auto slab = index.begin_host_fill(other_path[0]);
    require(slab.has_value(), "the write-through takes the Host copy of a snapshot whole on the Device");
    index.complete_host_fill(other_path[0]);
    index.unpin_node(other_path[0]);
    const SnapshotView view = index.snapshot(resume);
    require(index.valid(resume) && view.host == CopyState::Absent && view.device_slot != kNoId &&
                view.tail_device_copy == CopyState::Resident,
            "the snapshot must stay whole on the Device after its Host copy yields");
    index.release_path(other_path);
    index.check_invariants();
}

// A lane's blocks serve the snapshots it published: their write-through displaces no retained
// snapshot worth more than the most valuable of those.
void test_write_through_claim() {
    RecordingBackend backend;
    // 4 slabs: one tail-less image.
    PrefixCacheIndex index(small_config(4, 4), backend);
    const auto a      = make_tokens(64 * 8, 92);
    const auto path_a = insert_sequence(index, backend, a);
    const SnapshotRef valuable =
        publish_at(index, backend, path_a, a, 64 * 8, SnapshotKind::Boundary);
    backup_snapshot(index, valuable);
    index.release_path(path_a);
    const auto b            = make_tokens(64 * 2, 93);
    const auto path_b       = insert_sequence(index, backend, b);
    const SnapshotRef cheap = publish_tap(index, path_b[1]);
    const SnapshotRef published[] = {cheap, valuable};
    require(index.retained_priority(std::span<const SnapshotRef>(published, 1)) <
                index.retained_priority(std::span<const SnapshotRef>(published, 2)),
            "the claim is the most valuable retained snapshot");
    require(index.retained_priority({}) == -std::numeric_limits<double>::infinity(),
            "a lane that published nothing claims nothing");
    const double claim = index.retained_priority(std::span<const SnapshotRef>(published, 1));
    index.pin_node(path_b[0]);
    require(!index.begin_host_fill(path_b[0], {}, claim).has_value(),
            "a lane's blocks must not displace a retained snapshot worth more than its own");
    require(index.snapshot(valuable).host == CopyState::Resident,
            "the more valuable snapshot keeps its Host copy");
    index.supersede(cheap, Supersession::NextTap);
    require(index.retained_priority(std::span<const SnapshotRef>(published, 1)) ==
                -std::numeric_limits<double>::infinity(),
            "a superseded snapshot claims nothing");
    const auto slab = index.begin_host_fill(path_b[0]);
    require(slab.has_value() && index.snapshot(valuable).host == CopyState::Absent,
            "without a claim the write-through takes the Host copy of a snapshot whole on the Device");
    index.complete_host_fill(path_b[0]);
    index.unpin_node(path_b[0]);
    index.release_path(path_b);
    index.check_invariants();
}

// The image a lane writes to the Host never displaces the lane's newest superseded tap, as its
// block write-through does not: the next question about the same document resumes from it.
void test_image_write_keeps_superseded_tap() {
    for (const bool keep : {true, false}) {
        RecordingBackend backend;
        // 8 slabs: two tail-less images.
        PrefixCacheIndex index(small_config(8, 4), backend);
        const auto other      = make_tokens(64 * 8, 94);
        const auto other_path = insert_sequence(index, backend, other);
        const SnapshotRef valuable =
            publish_at(index, backend, other_path, other, 64 * 8, SnapshotKind::Boundary);
        backup_snapshot(index, valuable);
        index.release_path(other_path);
        const auto doc      = make_tokens(64 * 3, 95);
        const auto doc_path = insert_sequence(index, backend, doc);
        const SnapshotRef ladder = publish_tap(index, doc_path[1]);
        backup_snapshot(index, ladder);
        const SnapshotRef opener = publish_tap(index, doc_path[2]);
        index.supersede(ladder, Supersession::NextTap);
        index.pin_snapshot(opener);
        const bool written = index.begin_snapshot_host_fill(opener, keep ? ladder : SnapshotRef{});
        if (written) { index.complete_snapshot_host_fill(opener); }
        index.unpin_snapshot(opener);
        if (keep) {
            require(!written && index.valid(ladder) &&
                        index.snapshot(ladder).host == CopyState::Resident,
                    "the lane's own image must not displace its newest superseded tap");
        } else {
            require(written && (!index.valid(ladder) || index.snapshot(ladder).host ==
                                                            CopyState::Absent),
                    "without the keep a superseded tap yields its slabs first");
        }
        require(index.snapshot(valuable).host == CopyState::Resident,
                "the more valuable snapshot keeps its Host copy");
        index.release_path(doc_path);
        index.check_invariants();
    }
}

void test_image_only_host_tier() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config(8, 1);
    config.image_slabs       = 1;
    config.host_blocks       = false;
    PrefixCacheIndex index(config, backend);
    const auto a      = make_tokens(64, 15);
    const auto path_a = insert_sequence(index, backend, a);
    index.pin_node(path_a[0]);
    require(!index.begin_host_fill(path_a[0]).has_value(),
            "an image-only host tier must not back KV blocks");
    index.unpin_node(path_a[0]);

    // Endpoint with a tail; its host copy holds the image only.
    const auto tail_tokens      = make_tokens(10, 16);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path_a[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 1,
            "an image-only host copy must not reserve a tail slab");
    index.release_path(path_a);
    index.check_invariants();

    // The tail is unbacked: once its image leaves the Device slot, evicting the tail loses the
    // snapshot even though its image is on host.
    const auto taken = index.acquire_device_slot();
    require(taken.has_value() && index.valid(endpoint.snapshot) &&
                index.snapshot(endpoint.snapshot).device_slot == kNoId,
            "an image-backed owner gives up its Device slot");
    index.release_device_slot(*taken);
    std::uint32_t released = 0;
    while (index.valid(endpoint.snapshot) && released < 4) {
        released += index.evict_device_blocks(1);
        index.check_invariants();
    }
    require(!index.valid(endpoint.snapshot), "a snapshot without its tail KV must be removed");
    require(index.stats().host_free_slabs == 8, "the removed snapshot's image slab was leaked");
}

// Streamed-tier per-sequence stream images: registration, first-writer-wins, budget eviction
// (oldest unpinned first, pinned ones skipped), direct pins, and clean refusals.
//
// The images live on sibling branches off one shared prefix: pin_node pins a node's whole
// root chain (the handoff's write pin), so an image on one branch is never pinned by a
// registration on another, and the eviction order stays observable.
void test_stream_images() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config();
    config.stream_image_budget_bytes = 4U << 20U; // two two-block images
    PrefixCacheIndex index(config, backend);
    const auto prefix_tokens = make_tokens(64 * 2, 200);
    const std::uint64_t two_blocks = 2U * 1U << 20U;
    std::optional<std::uint32_t> b, c, d, e;

    const auto insert_branch = [&](std::uint32_t seed, std::uint32_t extra_blocks) {
        std::vector<TokenId> tokens = prefix_tokens;
        const auto tail             = make_tokens(64 * extra_blocks, seed);
        tokens.insert(tokens.end(), tail.begin(), tail.end());
        return insert_sequence(index, backend, tokens);
    };
    const auto path_a = insert_branch(201, 3); // [prefix, A2, A3, A4]
    const auto path_b = insert_branch(202, 2);
    const auto path_c = insert_branch(203, 2);
    const auto path_d = insert_branch(204, 2);
    const auto path_e = insert_branch(205, 2);

    // One image larger than the whole budget is refused up front.
    require(!index.register_stream_image(std::span<const NodeRef>(path_a.data() + 2, 2), 2, 2,
                                         5U * 1U << 20U)
                 .has_value(),
            "a single image larger than the budget must be refused");

    const auto a =
        index.register_stream_image(std::span<const NodeRef>(path_a.data() + 2, 2), 2, 2, two_blocks);
    require(a.has_value(), "no image for the first run");
    require(index.node(path_a[2]).host_image == *a && index.node(path_a[3]).host_image == *a &&
                index.node(path_a[4]).host_image == kNoId,
            "the image must bind exactly its run");
    require(index.stats().stream_images == 1 && index.stats().stream_image_bytes == two_blocks,
            "the image must count toward the budget");
    index.check_invariants();

    // Re-handing the same run adds nothing (the first writer's image keeps the copy).
    require(!index.register_stream_image(std::span<const NodeRef>(path_a.data() + 2, 2), 2, 2,
                                         two_blocks)
                 .has_value(),
            "an already-bound run must not register again");

    // A partially-bound run is a caller bug.
    require_throws(
        [&] { (void)index.register_stream_image(std::span<const NodeRef>(path_a.data() + 3, 2), 3, 2,
                                                two_blocks); },
        "a partially-bound run");

    // The insertions pinned every node once; drop it, so the images start unpinned.
    for (const auto& branch : {path_a, path_b, path_c, path_d, path_e}) { index.release_path(branch); }
    index.check_invariants();

    {
        const std::span<const NodeRef> run(path_b.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        b = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        require(b.has_value() && index.stats().stream_image_bytes == 2 * two_blocks,
                "the second image must fit the budget");
    }

    // The third evicts the oldest unpinned image (a).
    {
        const std::span<const NodeRef> run(path_c.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        c = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        require(c.has_value(), "the third image must fit after evicting the oldest");
        require(backend.released_stream_images.back() == *a, "the oldest image must be released");
        require(index.node(path_a[2]).host_image == kNoId && index.node(path_c[2]).host_image == *c,
                "eviction must unbind the old run and bind the new one");
        require(index.stats().stream_images == 2 &&
                    index.stats().stream_image_bytes == 2 * two_blocks &&
                    index.stats().stream_image_evictions == 1,
                "the eviction must count");
    }
    index.check_invariants();

    // A pinned node keeps its image out of the eviction order.
    {
        index.pin_node(path_b[2]); // holds b's image
        const std::span<const NodeRef> run(path_d.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        d = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        index.unpin_node(path_b[2]);
        require(d.has_value(), "the fourth image must fit after evicting the unpinned one");
        require(backend.released_stream_images.back() == *c, "the pinned image must be skipped");
        require(index.node(path_d[2]).host_image == *d && index.node(path_c[2]).host_image == kNoId,
                "the eviction must take the other run");
    }
    index.check_invariants();

    // A direct image pin (a resume lane holding its prefix) is skipped too.
    {
        index.pin_stream_image(*b);
        const std::span<const NodeRef> run(path_e.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        e = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        index.unpin_stream_image(*b);
        require(e.has_value() && backend.released_stream_images.back() == *d,
                "the direct pin must skip b and evict d");
    }
    index.check_invariants();

    // With every live image pinned, a new one is refused (a's old run is unbound again).
    {
        index.pin_stream_image(*b);
        index.pin_stream_image(*e);
        const std::span<const NodeRef> run(path_a.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        const auto f = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        index.unpin_stream_image(*b);
        index.unpin_stream_image(*e);
        require(!f.has_value(), "a budget full of pinned images must be refused");
    }
    index.check_invariants();

    // A backend allocation failure (after any eviction) is a clean refusal.
    {
        backend.fail_next_stream_image = true;
        const std::span<const NodeRef> run(path_a.data() + 2, 2);
        for (const NodeRef node : run) { index.pin_node(node); }
        const auto f = index.register_stream_image(run, 2, 2, two_blocks);
        for (const NodeRef node : run) { index.unpin_node(node); }
        require(!f.has_value(), "a failed allocation must refuse the image");
    }
    index.check_invariants();
}

// An image whose last bound node loses its storage is orphaned: the dead-node reclaim drops
// the block, and the image returns its buffer with it.
void test_stream_image_orphan() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config(5, 1); // four image slabs plus the tail slab
    config.stream_image_budget_bytes = 1U << 20U;
    PrefixCacheIndex index(config, backend);
    const auto a    = make_tokens(64, 22);
    const auto path = insert_sequence(index, backend, a); // one node, pinned
    const auto image =
        index.register_stream_image(std::span<const NodeRef>(path.data(), 1), 0, 1, 1U << 20U);
    require(image.has_value(), "no image for the block");
    backup_node(index, path[0]); // device copy plus the only slab
    index.release_path(path);
    index.check_invariants();
    require(index.evict_backed_device_blocks(1) == 1,
            "the slab-backed copy must drop its device half");
    // The block is dead now (slab, no device copy, nothing below). A root tap's image fill
    // needs slabs, reclaims the dead block, drops it, and orphans the image.
    const auto tail_tokens = make_tokens(10, 23);
    const auto slot        = index.acquire_device_slot();
    require(slot.has_value(), "no device slot");
    const auto tail_id = backend.allocate();
    const auto tap     = index.publish_snapshot(
        NodeRef{}, 10, std::span<const TokenId>(tail_tokens.data(), 10), tail_id, *slot,
        SnapshotKind::Tap);
    require(tap.created, "root tap not created");
    index.pin_snapshot(tap.snapshot);
    const bool filled = index.begin_snapshot_host_fill(tap.snapshot);
    (void)filled; // the fill reclaims the dead block, which orphans the image
    index.unpin_snapshot(tap.snapshot);
    require(!index.valid(path[0]), "the dead block must be reclaimed");
    require(index.stats().stream_images == 0 && index.stats().stream_image_bytes == 0,
            "the orphaned image must return its bytes");
    require(backend.released_stream_images.back() == *image,
            "the orphaned image must be released");
    index.check_invariants();
}

// Multi-session isolation (dense tier): sessions share only an identical leading prefix (a
// system prompt), and nothing past it. Simulates retained sessions (one dispatch keeps an
// interrupted session's KV for a later resume) and concurrent sessions (two lanes active at
// once), and checks that a session's match never reaches another session's blocks, that a
// partial prefix shares only its identical leading blocks, and that eviction pressure on one
// session cannot drop another session's blocks or the shared prefix.
void test_multi_session_isolation() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config(); // 64 host slabs, two device slots, 1MiB blocks
    PrefixCacheIndex index(config, backend);

    const auto system_tokens = make_tokens(64 * 3, 700); // the shared "system prompt"
    const auto a_tail        = make_tokens(64 * 3, 701);
    const auto b_tail        = make_tokens(64 * 3, 702);
    std::vector<TokenId> a_tokens = system_tokens;
    a_tokens.insert(a_tokens.end(), a_tail.begin(), a_tail.end());
    std::vector<TokenId> b_tokens = system_tokens;
    b_tokens.insert(b_tokens.end(), b_tail.begin(), b_tail.end());

    // Session A starts from the root; session B starts from the root with the same system
    // prompt and then diverges. The first three blocks must be A's identical nodes (shared),
    // and each session's own tail must be new.
    const auto path_a = insert_sequence(index, backend, a_tokens); // [S1,S2,S3,A4,A5,A6]
    const auto path_b = insert_sequence(index, backend, b_tokens); // [S1,S2,S3,B4,B5,B6]
    require(path_a.size() == 6 && path_b.size() == 6, "each session owns six blocks");
    require(path_b[0] == path_a[0] && path_b[1] == path_a[1] && path_b[2] == path_a[2],
            "the identical leading prefix must be shared, not duplicated");
    require(!(path_b[3] == path_a[3]) && !(path_b[4] == path_a[4]) && !(path_b[5] == path_a[5]),
            "past the shared prefix the sessions must be disjoint");
    index.check_invariants();

    // Match isolation: each session's match follows its own lineage only.
    const auto a_hashes = block_lookup_hashes(a_tokens, {});
    const auto b_hashes = block_lookup_hashes(b_tokens, {});
    const MatchResult match_a =
        index.match(a_tokens, a_hashes, {}, static_cast<std::uint32_t>(a_tokens.size()));
    const MatchResult match_b =
        index.match(b_tokens, b_hashes, {}, static_cast<std::uint32_t>(b_tokens.size()));
    // A 384-token prompt admits a frontier of at most 383, so at most five blocks match.
    require(match_a.path.size() == 5 && match_a.path[3] == path_a[3] && match_a.path[4] == path_a[4],
            "A's match must follow A's own blocks");
    require(match_b.path.size() == 5 && match_b.path[3] == path_b[3] && match_b.path[4] == path_b[4],
            "B's match must follow B's own blocks");
    for (const NodeRef node : match_a.path) {
        require(node != path_b[3] && node != path_b[4] && node != path_b[5],
                "A's match must not reach B's blocks");
    }
    for (const NodeRef node : match_b.path) {
        require(node != path_a[3] && node != path_a[4] && node != path_a[5],
                "B's match must not reach A's blocks");
    }

    // A prompt that shares only the first two system blocks matches exactly those two.
    std::vector<TokenId> partial_tokens = system_tokens;
    partial_tokens.resize(64 * 2);
    const auto partial_tail = make_tokens(64, 703);
    partial_tokens.insert(partial_tokens.end(), partial_tail.begin(), partial_tail.end());
    const auto partial_hashes = block_lookup_hashes(partial_tokens, {});
    const MatchResult match_p = index.match(partial_tokens, partial_hashes, {},
                                            static_cast<std::uint32_t>(partial_tokens.size()));
    require(match_p.path.size() == 2 && match_p.path[0] == path_a[0] && match_p.path[1] == path_a[1],
            "a partial prefix must share exactly its identical leading blocks");
    index.check_invariants();

    // Each session publishes a tap on its own tail (past the shared prefix).
    const SnapshotRef tap_a = publish_tap(index, path_a[3]); // A4, frontier 256
    const SnapshotRef tap_b = publish_tap(index, path_b[3]); // B4, frontier 256
    index.check_invariants();

    // Concurrent sessions plus an interrupted one: a third session C shares the prefix, then is
    // interrupted (its own blocks unpinned). Eviction pressure must take C's blocks first and
    // never A's or B's or the shared prefix.
    const auto c_tail   = make_tokens(64 * 2, 704);
    std::vector<TokenId> c_tokens = system_tokens;
    c_tokens.insert(c_tokens.end(), c_tail.begin(), c_tail.end()); // [S1,S2,S3,C4,C5]
    const auto path_c  = insert_sequence(index, backend, c_tokens);
    index.release_path(path_c); // C is interrupted; its own blocks are now the only unpinned ones
    index.check_invariants();
    require(index.device_evictable_blocks() == 2,
            "only the interrupted session's own blocks may be evictable");
    require(index.evict_device_blocks(2) == 2, "the two evictable blocks must go");
    require(!index.valid(path_c[3]) && !index.valid(path_c[4]), "C's own blocks must be lost");
    require(index.valid(path_a[5]) && index.node(path_a[5]).pins > 0 &&
                index.valid(path_b[5]) && index.node(path_b[5]).pins > 0,
            "the active sessions' blocks must survive the pressure");
    require(index.valid(path_a[0]) && index.node(path_a[0]).pins >= 2,
            "the shared prefix must stay pinned by both active sessions");
    index.check_invariants();

    // Retained-session resume: A is interrupted (B stays active). A's resume must offer A's own
    // tap, never B's.
    index.release_path(path_a);
    index.check_invariants();
    const MatchResult resume_a =
        index.match(a_tokens, a_hashes, {}, static_cast<std::uint32_t>(a_tokens.size()));
    require(resume_a.path.size() == 5, "A's resume must still follow A's own blocks");
    const MatchCandidate* own = nullptr;
    for (const MatchCandidate& candidate : resume_a.candidates) {
        if (candidate.snapshot == tap_a) { own = &candidate; }
        require(candidate.snapshot != tap_b, "B's tap must not be offered to A's resume");
    }
    require(own != nullptr && own->frontier == 4 * kBlockTokens,
            "A must resume from its own tap at its own frontier");
    index.check_invariants();
}

// Streamed tier: each session's compacted host copy lives in its own stream image, holding
// only that session's own blocks. The shared leading prefix stays out of every per-session
// image (bound at most once, first-writer-wins), so one session's stream image can never be
// read as another session's KV.
void test_streamed_multi_session_isolation() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config();
    config.stream_image_budget_bytes = 8U << 20U; // two three-block images (3MiB each)
    PrefixCacheIndex index(config, backend);
    const std::uint64_t three_blocks = 3U * 1U << 20U;

    const auto system_tokens = make_tokens(64 * 3, 710); // shared leading prefix
    const auto a_tail        = make_tokens(64 * 3, 711);
    const auto b_tail        = make_tokens(64 * 3, 712);

    // The shared prefix is built once; both sessions resume from it.
    const auto path_sp = insert_sequence(index, backend, system_tokens); // [S1,S2,S3]
    index.release_path(path_sp); // the prefix is now cached (unpinned)
    index.check_invariants();

    std::vector<TokenId> a_tokens = system_tokens;
    a_tokens.insert(a_tokens.end(), a_tail.begin(), a_tail.end());
    std::vector<TokenId> b_tokens = system_tokens;
    b_tokens.insert(b_tokens.end(), b_tail.begin(), b_tail.end());
    const auto path_a = insert_sequence(index, backend, a_tokens); // [S1,S2,S3,A4,A5,A6]
    const auto path_b = insert_sequence(index, backend, b_tokens); // [S1,S2,S3,B4,B5,B6]
    require(path_a[0] == path_b[0] && path_a[1] == path_b[1] && path_a[2] == path_b[2],
            "the shared prefix must be identical nodes");
    require(!(path_a[3] == path_b[3]), "the sessions must diverge past the prefix");

    // A's handoff compacts its own run into A's image; B's into B's image.
    const std::span<const NodeRef> run_a(path_a.data() + 3, 3);
    const auto image_a = index.register_stream_image(run_a, 3, 3, three_blocks);
    require(image_a.has_value(), "no image for A's own run");
    const std::span<const NodeRef> run_b(path_b.data() + 3, 3);
    const auto image_b = index.register_stream_image(run_b, 3, 3, three_blocks);
    require(image_b.has_value() && *image_a != *image_b,
            "each session's own run must get its own image");

    // Each image binds exactly its own run; the shared prefix is in neither.
    require(index.node(path_a[3]).host_image == *image_a && index.node(path_a[4]).host_image == *image_a &&
                index.node(path_a[5]).host_image == *image_a,
            "A's image must bind exactly A's own run");
    require(index.node(path_b[3]).host_image == *image_b && index.node(path_b[4]).host_image == *image_b &&
                index.node(path_b[5]).host_image == *image_b,
            "B's image must bind exactly B's own run");
    require(index.node(path_a[0]).host_image == kNoId && index.node(path_a[1]).host_image == kNoId &&
                index.node(path_a[2]).host_image == kNoId,
            "the shared prefix must not live in any per-session image");
    require(index.node(path_a[3]).host_image != *image_b && index.node(path_b[3]).host_image != *image_a,
            "one session's image must not bind the other session's blocks");

    // First-writer-wins: re-handing A's run is refused (its image already keeps the copy).
    require(!index.register_stream_image(run_a, 3, 3, three_blocks).has_value(),
            "A's already-bound run must not register again");

    // Both sessions' images coexist in the budget.
    require(index.stats().stream_images == 2 && index.stats().stream_image_bytes == 2 * three_blocks,
            "both sessions' images must coexist");
    index.check_invariants();
}

// A pinned snapshot keeps its anchor path: evicting an unbacked path block would lose the node
// and every snapshot anchored below it, the pinned one included.
void test_snapshot_pin_holds_anchor_path() {
    {
        // A snapshot Host write in flight after its sequence released the path.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64), backend);
        const auto tokens     = make_tokens(64 * 4, 11);
        const auto path       = insert_sequence(index, backend, tokens);
        const SnapshotRef tap = publish_tap(index, path[3]);
        index.release_path(path);
        index.pin_snapshot(tap);
        index.check_invariants();
        require(index.device_evictable_blocks() == 0, "a pinned snapshot's path is not evictable");
        require(index.begin_snapshot_host_fill(tap), "no slabs for the snapshot write");
        require(index.evict_device_blocks(10) == 0, "eviction took a pinned snapshot's path");
        require(index.valid(tap), "the pinned snapshot was lost");
        index.complete_snapshot_host_fill(tap);
        index.unpin_snapshot(tap);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 4, "an unpinned path is evictable again");
        require(!index.valid(tap), "a snapshot whose path lost its last copy is removed");
        index.check_invariants();
    }
    {
        // An aborted write returns the path to the LRU.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64), backend);
        const auto path       = insert_sequence(index, backend, make_tokens(64 * 2, 12));
        const SnapshotRef tap = publish_tap(index, path[1]);
        index.release_path(path);
        index.pin_snapshot(tap);
        require(index.begin_snapshot_host_fill(tap), "no slabs for the snapshot write");
        index.abort_snapshot_host_fill(tap);
        index.unpin_snapshot(tap);
        index.check_invariants();
        require(index.device_evictable_blocks() == 2, "an aborted write keeps no pins");
    }
    {
        // Pins on snapshots sharing ancestors add up, and so do repeated pins of one snapshot.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64, 3), backend);
        const auto path         = insert_sequence(index, backend, make_tokens(64 * 4, 13));
        const SnapshotRef upper = publish_tap(index, path[1]);
        const SnapshotRef lower = publish_tap(index, path[3]);
        index.release_path(path);
        index.pin_snapshot(upper);
        index.pin_snapshot(lower);
        index.pin_snapshot(lower);
        index.check_invariants();
        index.unpin_snapshot(lower);
        require(index.evict_device_blocks(10) == 0, "a twice pinned snapshot lost its path");
        index.unpin_snapshot(lower);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 2, "only the blocks below the pinned one go");
        require(index.valid(upper) && !index.valid(lower), "the pinned snapshot must survive");
        index.unpin_snapshot(upper);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 2 && !index.valid(upper),
                "the rest of the path is evictable once unpinned");
    }
    {
        // A root-anchored snapshot pins no node.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64), backend);
        const auto tokens = make_tokens(64 * 2, 14);
        const auto path   = insert_sequence(index, backend, tokens);
        const SnapshotRef rooted =
            publish_at(index, backend, path, tokens, 10, SnapshotKind::Endpoint);
        index.release_path(path);
        index.pin_snapshot(rooted);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 2, "a root snapshot pins no block");
        require(index.valid(rooted), "a pinned root snapshot must survive");
        index.unpin_snapshot(rooted);
        index.check_invariants();
    }
    {
        // A Host restore in flight pins its restored blocks and its snapshot; the Device-only
        // anchor below the restored block stays with the snapshot.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64), backend);
        const auto tokens     = make_tokens(64 * 3, 15);
        const auto path       = insert_sequence(index, backend, tokens);
        const SnapshotRef tap = publish_tap(index, path[2]);
        backup_node(index, path[1]);
        backup_snapshot(index, tap);
        index.release_path(path);
        require(index.evict_backed_device_blocks(10) == 1, "only the backed block is evicted");
        require(index.node(path[1]).device == CopyState::Absent, "the backed block is Host-only");
        const std::uint32_t page = backend.allocate();
        index.pin_node(path[1]);
        index.begin_device_fill(path[1], page);
        index.pin_snapshot(tap);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 0,
                "eviction took the restoring snapshot's anchor");
        require(index.valid(tap), "the restoring snapshot was lost");
        index.complete_device_fill(path[1]);
        index.unpin_node(path[1]);
        index.unpin_snapshot(tap);
        index.check_invariants();
        require(index.evict_device_blocks(10) == 3, "the restored path is evictable again");
        index.check_invariants();
    }
}

void test_random_stress() {
    std::mt19937 rng(1234);
    for (int round = 0; round < 20; ++round) {
        RecordingBackend backend;
        PrefixIndexConfig config = small_config(40, 3);
        config.max_nodes         = 128;
        config.max_snapshots     = 16;
        PrefixCacheIndex index(config, backend);
        // A small vocabulary of shared prefixes so paths overlap.
        std::vector<std::vector<TokenId>> prompts;
        for (int p = 0; p < 6; ++p) {
            auto tokens = make_tokens(64 * (2 + p % 4), 100 + round);
            auto suffix = make_tokens(64 * 3, 200 + p + round * 10);
            tokens.insert(tokens.end(), suffix.begin(), suffix.end());
            tokens.push_back(1);
            prompts.push_back(std::move(tokens));
        }
        std::vector<std::vector<NodeRef>> held;
        std::vector<SnapshotRef> published_snapshots;
        // Snapshot Host writes still in flight: pinned across later releases and evictions.
        std::vector<SnapshotRef> writing;
        const auto finish_write = [&](std::size_t which) {
            const SnapshotRef snapshot = writing[which];
            if (rng() % 4U == 0) {
                index.abort_snapshot_host_fill(snapshot);
            } else {
                index.complete_snapshot_host_fill(snapshot);
            }
            index.unpin_snapshot(snapshot);
            writing.erase(writing.begin() + static_cast<std::ptrdiff_t>(which));
        };
        for (int step = 0; step < 300; ++step) {
            const int op = static_cast<int>(rng() % 6U);
            if (op <= 1 && held.size() < 3) {
                const auto& prompt           = prompts[rng() % prompts.size()];
                const auto hashes            = block_lookup_hashes(prompt, {});
                const auto n                 = static_cast<std::uint32_t>(prompt.size());
                MatchResult match            = index.match(prompt, hashes, {}, n);
                const AdmissionChoice choice = index.choose(match, n);
                std::uint32_t reuse_blocks   = 0;
                if (choice.candidate) {
                    const MatchCandidate& candidate = match.candidates[*choice.candidate];
                    reuse_blocks                    = candidate.path_blocks;
                    index.note_hit(candidate.snapshot);
                }
                std::vector<NodeRef> path(match.path.begin(), match.path.begin() + reuse_blocks);
                index.acquire_path(path);
                for (const NodeRef node : path) {
                    if (index.node(node).device == CopyState::Absent) {
                        index.begin_device_fill(node, backend.allocate());
                        index.complete_device_fill(node);
                    }
                }
                NodeRef parent = path.empty() ? NodeRef{} : path.back();
                for (std::uint32_t b = reuse_blocks; b < hashes.size(); ++b) {
                    if (index.stats().nodes + 2 >= config.max_nodes) {
                        (void)index.evict_device_blocks(4);
                    }
                    const std::span<const TokenId> block(prompt.data() + b * kBlockTokens,
                                                         kBlockTokens);
                    const std::uint32_t id    = backend.allocate();
                    const InsertResult result = index.insert_block(parent, hashes[b], block, 0, id);
                    if (!result.device_attached) { backend.release_device_block(id); }
                    path.push_back(result.node);
                    parent = result.node;
                    if (index.node(result.node).host == CopyState::Absent &&
                        index.node(result.node).device == CopyState::Resident && rng() % 2U) {
                        index.pin_node(result.node);
                        if (index.begin_host_fill(result.node)) {
                            index.complete_host_fill(result.node);
                        }
                        index.unpin_node(result.node);
                    }
                    if (rng() % 3U == 0) {
                        if (const auto slot = index.acquire_device_slot(
                                rng() % 2U ? std::numeric_limits<double>::infinity() : 0.0)) {
                            const PublishResult published = index.publish_snapshot(
                                result.node, (index.node(result.node).depth + 1U) * kBlockTokens,
                                {}, std::nullopt, *slot,
                                rng() % 4U == 0 ? SnapshotKind::Boundary : SnapshotKind::Tap);
                            if (published.created) {
                                published_snapshots.push_back(published.snapshot);
                            }
                            if (published.created && rng() % 2U) {
                                index.pin_snapshot(published.snapshot);
                                if (!index.begin_snapshot_host_fill(published.snapshot)) {
                                    index.unpin_snapshot(published.snapshot);
                                } else if (writing.size() < 3 && rng() % 2U) {
                                    writing.push_back(published.snapshot);
                                } else {
                                    index.complete_snapshot_host_fill(published.snapshot);
                                    index.unpin_snapshot(published.snapshot);
                                }
                            }
                        }
                    }
                    index.check_invariants();
                }
                held.push_back(std::move(path));
            } else if (op <= 3 && !held.empty()) {
                const std::size_t which = rng() % held.size();
                index.release_path(held[which]);
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(which));
            } else if (op == 4 && !published_snapshots.empty()) {
                const SnapshotRef snapshot =
                    published_snapshots[rng() % published_snapshots.size()];
                if (index.valid(snapshot)) {
                    if (rng() % 2U) {
                        index.supersede(snapshot,
                                        rng() % 2U ? Supersession::Resumed : Supersession::NextTap);
                    } else {
                        index.note_hit(snapshot);
                    }
                }
            } else {
                (void)index.evict_device_blocks(1 + rng() % 8U);
                for (const SnapshotRef snapshot : writing) {
                    require(index.valid(snapshot), "a snapshot was lost during its Host write");
                }
                if (!writing.empty() && rng() % 2U) { finish_write(rng() % writing.size()); }
            }
            index.check_invariants();
        }
        while (!writing.empty()) { finish_write(writing.size() - 1); }
        for (const auto& path : held) { index.release_path(path); }
        (void)index.evict_device_blocks(1000);
        index.check_invariants();
        // Every device page not held by an index node was released exactly once.
        std::uint32_t resident = 0;
        for (std::uint32_t id : backend.live) {
            (void)id;
            ++resident;
        }
        require(resident == index.stats().device_resident_blocks +
                                0U /* tails are never used in this test */,
                "device pages leaked or double-released");
    }
}

void test_tap_planner() {
    TapPlannerConfig config;
    config.max_new_taps   = 8;
    config.ladder_tokens  = 4096;
    config.min_gap_tokens = 1024;
    const std::vector<TapHint> hints{
        {100, TapHintKind::Structural},        {5000, TapHintKind::MessageBoundary},
        {9000, TapHintKind::MessageBoundary},  {21800, TapHintKind::MessageBoundary},
        {29990, TapHintKind::MessageBoundary}, {30010, TapHintKind::GenerationOpener},
        {12345, TapHintKind::Explicit},
    };
    const auto taps = plan_taps(30020, 0, hints, {}, {}, config);
    require(std::is_sorted(
                taps.begin(), taps.end(),
                [](const PlannedTap& a, const PlannedTap& b) { return a.position < b.position; }),
            "taps must be sorted");
    for (const PlannedTap& tap : taps) {
        require(tap.position > 0 && tap.position <= 30019, "tap outside the legal domain");
    }
    auto find = [&](const std::vector<PlannedTap>& planned,
                    std::uint32_t position) -> std::optional<PlannedTap> {
        for (const PlannedTap& tap : planned) {
            if (tap.position == position) { return tap; }
        }
        return std::nullopt;
    };
    require(find(taps, 12345) && find(taps, 12345)->placement == TapPlacement::Exact,
            "an explicit hint must be tapped exactly");
    require(find(taps, 30010) && find(taps, 30010)->placement == TapPlacement::Exact,
            "the generation opener must be tapped exactly");
    require(find(taps, 100) && find(taps, 100)->placement == TapPlacement::Exact,
            "a structural hint must be tapped exactly");
    require(find(taps, 100)->boundary && find(taps, 12345)->boundary &&
                !find(taps, 30010)->boundary,
            "structural and explicit taps are boundaries, the generation opener is not");
    require(!find(taps, 30019),
            "the prompt tail next to the generation opener covers nothing and must be dropped");
    // A protocol-automatic marker is planned like a structural boundary but marks the latest turn,
    // so its snapshot is supersedable; clustered with a shared boundary it stays shared.
    const std::vector<TapHint> automatic{{3000, TapHintKind::Automatic},
                                         {5000, TapHintKind::Automatic},
                                         {5030, TapHintKind::Structural}};
    const auto automatic_taps = plan_taps(8000, 0, automatic, {}, {}, config);
    require(find(automatic_taps, 3000) &&
                find(automatic_taps, 3000)->placement == TapPlacement::Exact &&
                !find(automatic_taps, 3000)->boundary,
            "an automatic marker must be tapped exactly and stay supersedable");
    require(find(automatic_taps, 5000) && find(automatic_taps, 5000)->boundary &&
                !find(automatic_taps, 5030),
            "a cluster keeps its earliest position, shared when any member is");
    // n - 8192 = 21828: the boundary 28 tokens below is within the gap, so the ladder snaps.
    require(find(taps, 21800) && find(taps, 21800)->placement == TapPlacement::Flexible,
            "the ladder must snap to a nearby message boundary and stay flexible");
    // n - 4096 = 25924 has no boundary within the gap: the ladder keeps its target.
    require(find(taps, 25924) && find(taps, 25924)->placement == TapPlacement::Flexible,
            "the ladder must keep its target without a nearby boundary");

    // A short user turn puts the system-block end and the generation opener in one cluster: the
    // earlier boundary is kept because a new conversation sharing the system block diverges
    // there, and the prompt tail next to it is dropped.
    const std::vector<TapHint> short_turn{{1085, TapHintKind::Structural},
                                          {1098, TapHintKind::GenerationOpener}};
    const auto clustered = plan_taps(1104, 0, short_turn, {}, {}, config);
    require(clustered.size() == 1 && clustered[0].position == 1085 &&
                clustered[0].placement == TapPlacement::Exact,
            "a semantic cluster must keep its earliest boundary alone");

    // An explicit breakpoint right after the system-block end does not replace it: only the
    // earlier snapshot serves a new conversation diverging between the two.
    const std::vector<TapHint> marked{{2109, TapHintKind::Structural},
                                      {2115, TapHintKind::Explicit},
                                      {2117, TapHintKind::GenerationOpener}};
    const auto both = plan_taps(2124, 0, marked, {}, {}, config);
    require(both.size() == 2 && both[0].position == 2109 && both[1].position == 2115,
            "an explicit breakpoint must not displace an earlier semantic boundary");

    // Without an opener near the end the prompt tail is a flexible tap.
    const std::vector<TapHint> plain{{100, TapHintKind::Structural}};
    const auto tail = plan_taps(3000, 0, plain, {}, {}, config);
    require(find(tail, 2999) && find(tail, 2999)->placement == TapPlacement::Flexible,
            "the prompt tail must be a flexible tap");

    // Resume base and existing snapshots suppress taps at or near them.
    const std::vector<std::uint32_t> existing{12345};
    const auto resumed = plan_taps(30020, 10000, hints, existing, {}, config);
    for (const PlannedTap& tap : resumed) {
        require(tap.position > 10000 && tap.position != 12345,
                "tap below base or duplicating an existing snapshot");
    }
    const std::vector<std::uint32_t> near_opener{30000};
    require(!find(plan_taps(30020, 0, hints, near_opener, {}, config), 30010),
            "a tap within the minimum separation of an existing snapshot must be dropped");

    // Exclusions move taps to the start of the excluded span.
    const std::vector<TapExclusion> exclusions{{12000, 13000}};
    const auto excluded = plan_taps(30020, 0, hints, {}, exclusions, config);
    for (const PlannedTap& tap : excluded) {
        require(!(tap.position > 12000 && tap.position < 13000), "tap inside a Vision span");
    }
    require(find(excluded, 12000).has_value(), "an excluded explicit tap moves to the span start");

    // A media item's exclusion reaches the end of the block the item begins in: a frontier inside
    // that block would have the item in no block key. Items that end past it keep their own end.
    require(media_exclusion(970, 990).begin == 970 && media_exclusion(970, 990).end == 1024,
            "a media item inside one block excludes the rest of that block");
    require(media_exclusion(960, 1024).end == 1024 && media_exclusion(1000, 1300).end == 1300 &&
                media_exclusion(1023, 1024).end == 1024 && media_exclusion(0, 4).end == 64,
            "a media item's exclusion ends at its own end or its first block's end");
    const std::vector<TapExclusion> media{media_exclusion(970, 990)};
    require(!inside_exclusion(970, media) && inside_exclusion(971, media) &&
                inside_exclusion(990, media) && inside_exclusion(1023, media) &&
                !inside_exclusion(1024, media) && !inside_exclusion(960, media),
            "a frontier past a media item waits for the end of the item's first block");
    // The generation opener 20 tokens past a 20-token item moves to the item's start.
    const std::vector<TapHint> after_media{{1010, TapHintKind::GenerationOpener}};
    const auto short_media = plan_taps(1013, 0, after_media, {}, media, config);
    for (const PlannedTap& tap : short_media) {
        require(!inside_exclusion(tap.position, media), "tap in a media item's first block");
    }
    require(find(short_media, 970).has_value(), "the opener past a short media item moves before it");

    // Budget: the highest-priority tap survives.
    TapPlannerConfig one = config;
    one.max_new_taps     = 1;
    const auto single    = plan_taps(30020, 0, hints, {}, {}, one);
    require(single.size() == 1 && single[0].position == 12345, "tap budget or priority ignored");

    // Ladder size is logarithmic for a long prompt without hints.
    const auto ladder = plan_taps(240000, 0, {}, {}, {}, config);
    require(ladder.size() <= 7, "ladder must be logarithmic");
    require(plan_taps(1, 0, hints, {}, {}, config).empty(), "no taps for a 1-token prompt");
}

// A Host tier smaller than a saved one restores whole snapshots with their paths, most valuable
// first, and never a block no chosen snapshot resumes through.
void test_restore_plan() {
    const PrefixIndexConfig config = small_config();
    // Two lineages sharing blocks 0-1: 0-1-2-3-4 and 0-1-5-6-7-8-9.
    const std::vector<std::int32_t> parents         = {-1, 0, 1, 2, 3, 1, 5, 6, 7, 8};
    const std::vector<SavedSnapshotShape> snapshots = {
        {.anchor = 4, .frontier = 5 * 64, .slabs = 4, .hits = 0},
        {.anchor = 9, .frontier = 7 * 64 + 10, .slabs = 5, .hits = 5},
        {.anchor = 1, .frontier = 2 * 64, .slabs = 4, .hits = 0},
    };
    const auto chosen_blocks = [](const HostRestorePlan& plan) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t index = 0; index < plan.blocks.size(); ++index) {
            if (plan.blocks[index]) { out.push_back(index); }
        }
        return out;
    };

    const HostRestorePlan all = plan_host_restore(parents, snapshots, 23, config);
    require(all.slabs == 23 && chosen_blocks(all).size() == parents.size() &&
                std::all_of(all.snapshots.begin(), all.snapshots.end(), [](bool b) { return b; }),
            "a tier that holds the whole file must restore all of it");

    // The blocks alone would fill 10 of these 12 slabs and leave no room for any image: the plan
    // instead takes the most used snapshot and exactly its path.
    const HostRestorePlan one = plan_host_restore(parents, snapshots, 12, config);
    require(!one.snapshots[0] && one.snapshots[1] && !one.snapshots[2] && one.slabs == 12,
            "a small tier must keep the most valuable snapshot");
    require(chosen_blocks(one) == std::vector<std::uint32_t>{0, 1, 5, 6, 7, 8, 9},
            "a small tier must restore exactly the kept snapshot's path");

    // Shared path blocks are paid for once, and a snapshot that does not fit is passed over for a
    // later one that does.
    const HostRestorePlan two = plan_host_restore(parents, snapshots, 16, config);
    require(!two.snapshots[0] && two.snapshots[1] && two.snapshots[2] && two.slabs == 16,
            "a snapshot sharing a kept path must cost only its own slabs");

    // Between otherwise equal snapshots, use decides.
    const std::vector<std::int32_t> roots        = {-1, -1};
    const std::vector<SavedSnapshotShape> equals = {
        {.anchor = 0, .frontier = 64, .slabs = 4, .hits = 0},
        {.anchor = 1, .frontier = 64, .slabs = 4, .hits = 3},
    };
    const HostRestorePlan used = plan_host_restore(roots, equals, 5, config);
    require(!used.snapshots[0] && used.snapshots[1] && !used.blocks[0] && used.blocks[1],
            "the more used of two equal snapshots must be kept");

    require_throws([&] { (void)plan_host_restore(std::vector<std::int32_t>{0}, {}, 8, config); },
                   "a block that is its own parent must be rejected");
}

} // namespace

int main() {
    try {
        test_hashes();
        test_match_and_choice();
        test_collisions_and_tails();
        test_device_lru_order();
        test_backed_before_unbacked();
        test_host_dead_and_gdsf();
        test_device_slots();
        test_supersession();
        test_lineage_under_pressure();
        test_questions_about_one_document();
        test_write_through_keeps_superseded_tap();
        test_unbacked_slot_values();
        test_refusals_age_owners();
        test_host_write_admission();
        test_unshared_boundary();
        test_hits_expire();
        test_lineage_hits();
        test_host_only_reattach();
        test_tail_device_fill();
        test_snapshot_pin_holds_anchor_path();
        test_persistence_roundtrip();
        test_stream_persistence_roundtrip();
        test_restore_plan();
        test_image_only_host_tier();
        test_stream_images();
        test_stream_image_orphan();
        test_multi_session_isolation();
        test_streamed_multi_session_isolation();
        test_slot_snapshot_keeps_tail();
        test_write_through_claim();
        test_image_write_keeps_superseded_tap();
        test_random_stress();
        test_tap_planner();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "prefix cache index tests passed\n";
    return 0;
}
