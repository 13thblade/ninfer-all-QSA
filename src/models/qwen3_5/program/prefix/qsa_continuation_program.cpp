// QSA streamed-tier reuse (ContextCacheMode::Qsa): quote, reserve and terminal settlement.
//
// The reuse itself is upstream qwen4_exp's (qwen4_exp_core.cpp:1648-1700): a request resumes from
// the longest of three things -- the lane's live state, the snapshot at the end of the lane's last
// prompt, and the store's longest prefix -- and only the third copies KV. The first two are the
// lane's own, which is what makes a same-conversation turn cheap.
//
// The states are qwen3_5's: the GDN conv/recurrent state and the continuation hidden live in a
// StateImage slot, so the lane keeps two of them (its live state and its prompt-end snapshot) the
// way the Legacy catalog keeps checkpoints -- frozen slots outside the catalog, like
// `stop_holds_`. The KV planes are the lane's pinned host planes; the store image is the fallback
// for a lane some other conversation took, and holds the same StateImage bytes beside them:
//
//   [0, state_bytes)              the StateImage host image (GDN, continuation hidden, draft state)
//   [state_bytes, + plane_bytes)  the lane's KV planes (K, V, K-scale, V-scale per layer)

#include "models/qwen3_5/program/prefix/qsa_continuation.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/prefix/qsa_image.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// ---- geometry --------------------------------------------------------------------------------

std::uint64_t ProgramImpl::qsa_state_image_bytes() const {
    if (state_images == nullptr) { return 0; }
    return QsaContinuationStore::page_rounded(state_images->host_layout().image_bytes,
                                              QsaContinuationStore::kHostPageBytes);
}

std::uint64_t ProgramImpl::qsa_plane_image_bytes(std::uint32_t lane) const {
    if (!qsa_stream_state.has_value() || lane >= qsa_stream_state->lanes) { return 0; }
    return qsa_lane_plane_bytes(*qsa_stream_state, lane);
}

// ---- the lane's retained states ---------------------------------------------------------------

QsaLaneRetentionView ProgramImpl::qsa_retention_view(std::uint32_t lane) const {
    QsaLaneRetentionView view;
    if (lane >= max_concurrency) { return view; }
    const QsaLaneRetention& retention = qsa_lane_retention_[lane];
    if (retention.epoch == 0 || retention.epoch != lane_epochs[lane]) { return view; }
    view.live                     = retention.live.has_value();
    view.snapshot                 = retention.snapshot.has_value();
    view.snapshot_drafter_follows = retention.snapshot_drafter_follows;
    return view;
}

void ProgramImpl::qsa_end_lane_tenure(std::uint32_t lane) noexcept {
    if (!qsa_stream_state.has_value() || lane >= max_concurrency) { return; }
    // The staging copy is keyed on the lane's tenure: bump the generation so the next tenant
    // cannot reuse the staging bytes this one left, and drop the cached (lane, layer, frontier)
    // triple with it. This is carrying, not reuse, so it runs whether or not anything is retained.
    auto& generation = qsa_stream_state->lane_generation[lane];
    ++generation;
    if (generation == 0) { ++generation; }
    if (qsa_stream_state->staged_lane == static_cast<std::int32_t>(lane)) {
        qsa_stream_state->staged_lane  = -1;
        qsa_stream_state->staged_layer = -1;
        qsa_stream_state->staged_end   = 0;
    }
}

void ProgramImpl::qsa_drop_lane_retention(std::uint32_t lane, bool save_image) noexcept {
    if (lane >= max_concurrency || state_store == nullptr) { return; }
    QsaLaneRetention& retention = qsa_lane_retention_[lane];
    try {
        // Upstream takes the image when a slot is taken away (qwen4_exp's eviction path), not at
        // the end of a turn. The lane's planes are still intact here and are about to be
        // overwritten, so this is the only point that needs the H2H copy. A store resume passes
        // save_image=false: its own image is already in the store, and a fresh write into a full
        // budget could evict the very image this admission is about to restore.
        if (save_image) { qsa_store_lane_image(lane); }
        if (retention.live && state_store->can_release(*retention.live)) {
            (void)state_store->release(*retention.live);
        }
        if (retention.snapshot && state_store->can_release(*retention.snapshot)) {
            (void)state_store->release(*retention.snapshot);
        }
        if (retention.backend && backend_kv_addresses != nullptr &&
            backend_kv_addresses->active(*retention.backend)) {
            backend_kv_addresses->deactivate(*retention.backend);
            (void)backend_kv_addresses->release(*retention.backend);
        }
    } catch (...) {}
    retention = {};
}

void ProgramImpl::qsa_capture_anchor(SequenceState& sequence) noexcept {
    if (state_store == nullptr || !qsa_stream_state.has_value() ||
        sequence.lane >= max_concurrency) {
        return;
    }
    QsaLaneRetention& retention = qsa_lane_retention_[sequence.lane];
    try {
        // Upstream takes the snapshot where the prompt ends, before the generation advances the
        // state (qwen4_exp_core.cpp:2219). At this point the live slot holds exactly that state
        // and `sequence.tail_hidden` is its continuation-hidden view, so the snapshot carries both
        // without touching either: one Device-to-Device copy.
        if (!state_store->valid(sequence.state.write) ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            return;
        }
        std::optional<StateImageHandle> snapshot = state_store->reserve_destination();
        if (!snapshot && retention.snapshot && state_store->can_release(*retention.snapshot)) {
            // The pool is sized C + C: every lane holds its live state (as the sequence's active
            // slot) and its last prompt's snapshot, so a full pool is the steady state, not an
            // accident. This prompt's snapshot supersedes that one, so giving it back first is
            // what makes room for the copy; a capture that still finds nothing leaves the lane
            // without a snapshot, which costs the next turn a re-prefill, not correctness.
            (void)state_store->release(*retention.snapshot);
            retention.snapshot.reset();
            snapshot = state_store->reserve_destination();
        }
        if (!snapshot) { return; }
        try {
            state_images->copy_slot(state_store->physical_slot(sequence.state.write),
                                    state_store->physical_slot(*snapshot), compute_streams);
            state_store->publish_copied_checkpoint(*snapshot);
        } catch (...) {
            (void)state_store->release(*snapshot);
            return;
        }
        // This prompt's snapshot supersedes the last one's.
        if (retention.snapshot && state_store->can_release(*retention.snapshot)) {
            (void)state_store->release(*retention.snapshot);
        }
        retention.snapshot = *snapshot;
        retention.epoch    = lane_epochs[sequence.lane];
        // A drafting request may only resume from a snapshot its drafter followed: the draft KV
        // covers [0, mtp_kv_valid) and must reach the anchor (upstream's `drafter_follows`).
        retention.snapshot_drafter_follows =
            speculative_backend != SpeculativeBackend::Mtp ||
            (sequence.kv && sequence.kv->backend &&
             sequence.mtp_kv_valid + 1U >= sequence.text_kv_valid);
    } catch (...) {}
}

// ---- quote -----------------------------------------------------------------------------------

QsaAdmissionQuote ProgramImpl::qsa_quote(const PreparedPromptData& prompt,
                                         const RequestBasePlan& base,
                                         runtime::LaneId destination) {
    if (base.impl_ == nullptr) {
        throw std::logic_error("QSA admission requires a base plan");
    }
    if (destination.value >= max_concurrency) {
        throw std::out_of_range("QSA admission lane is out of range");
    }
    const RequestBasePlanImpl& plan = *base.impl_;
    const auto n                    = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (n == 0 || n != plan.summary.prompt_tokens) {
        throw std::logic_error("QSA admission prompt does not match its base plan");
    }

    auto quote                        = std::make_shared<QsaQuoteImpl>();
    quote->summary                    = plan.summary;
    quote->sampling                   = plan.sampling;
    quote->grammar                    = plan.grammar;
    quote->logprobs                   = plan.logprobs;
    quote->readout_tokens             = plan.readout_tokens;
    quote->branches                   = plan.branches;
    quote->drafting                   = plan.drafting;
    quote->text_kv_page_entitlement    = plan.text_kv_page_entitlement;
    quote->backend_kv_page_entitlement = plan.backend_kv_page_entitlement;
    quote->vision_control_plan         = plan.vision_control_plan;
    quote->root_rebuild_work           = plan.root_rebuild_work;
    quote->root_rebuild_tail_begin     = plan.root_rebuild_tail_begin;
    quote->destination                 = destination.value;
    quote->destination_epoch           = lane_epochs[destination.value];

    QsaAdmissionQuote out;
    out.destination = destination;
    out.summary     = plan.summary;
    out.impl        = std::move(quote);

    const auto lane_free = [&]() {
        return active_continuations[destination.value] >= continuation_capacity &&
               !requests[destination.value].prefill.has_value();
    };
    if (!plan.allow_prefix_reuse) {
        // Reuse off (serve warmup, --no-prefix-reuse, the streamed-tier guard): root only.
        out.summary.reusable_prompt_tokens = 0;
        out.summary.prefix_reuse_path      = PrefixReusePath::Root;
        out.readiness =
            lane_free() ? runtime::Readiness::Ready : runtime::Readiness::TemporarilyBlocked;
        return out;
    }

    const LanePrefixRecord& record =
        qsa_lane_prefix.size() == max_concurrency ? qsa_lane_prefix[destination.value]
                                                  : LanePrefixRecord{};
    // A store resume is refused while a drafter runs: an image carries no draft KV, and handing
    // the drafter KV it never wrote is what upstream's `drafter_follows` bit forbids.
    const bool store_allowed = speculative_backend != SpeculativeBackend::Mtp;
    const QsaReuseChoice choice =
        qsa_choose_reuse(record, lane_epochs[destination.value],
                         qsa_retention_view(destination.value), store_allowed, qsa_store_.get(),
                         prompt.token_ids, prompt.block_extras);
    out.reuse      = choice.tokens;
    out.image      = choice.image;
    out.from_store = choice.from_store;
    out.whole      = choice.whole;
    out.path_kind  = choice.path_kind;
    // This lane still holds a prefix the prompt did not continue. Admission must prefer a lane
    // with no such owner: a root request that lands here drops the record and its retained state,
    // evicting a conversation that may return on its next turn. Under serial prefill every
    // arrival would otherwise take the lowest-index lane and evict its predecessor, so no session
    // ever keeps a lane to resume from.
    out.lane_carries_foreign_state =
        choice.tokens == 0 && record.epoch == lane_epochs[destination.value] && !record.held.empty();
    // The reservation builds the prefill state from the impl's copy of the summary and the
    // runtime reports that copy back as the Begin summary, while the Engine committed the one
    // this quote publishes at admission: both have to carry the same reuse, or the completion
    // is rejected as a mismatch.
    const PrefixReusePath path =
        choice.tokens == 0
            ? PrefixReusePath::Root
            : (choice.path_kind == 1 ? PrefixReusePath::PrivateTurnClosure
                                     : PrefixReusePath::PrivateEndpoint);
    for (runtime::RequestPlanSummary* summary : {&out.summary, &out.impl->summary}) {
        summary->reusable_prompt_tokens = choice.tokens;
        summary->prefix_reuse_path      = path;
    }
    // Scheduler service units, the shape the hybrid quote publishes: one per nominal prefill
    // chunk, one per Vision item the suffix still has to encode, one per further generated
    // token. Without it the request would hold the units a cold prompt costs.
    const std::uint32_t suffix = n - choice.tokens;
    std::uint64_t units        = suffix == 0 ? 1ULL : 1ULL + (suffix - 1ULL) / prefill_chunk;
    if (plan.vision_control_plan) {
        for (const auto& item : plan.vision_control_plan->items) {
            if (item.token_end > choice.tokens) { ++units; }
        }
    }
    if (out.summary.effective_output_tokens != 0) {
        units += out.summary.effective_output_tokens - 1ULL;
    }
    units += branch_service_work(plan.branches.get(), prefill_chunk);
    out.impl->summary.service_work_quanta = units;
    out.readiness = lane_free() ? runtime::Readiness::Ready
                                : runtime::Readiness::TemporarilyBlocked;
    return out;
}

// ---- reserve ---------------------------------------------------------------------------------

runtime::ContextTransactionReserveStatus ProgramImpl::qsa_reserve_materialization(
    QsaAdmissionQuote&& quote, const PreparedPromptData& prompt,
    const std::function<PreparedPromptData()>& take_prompt,
    runtime::CancellationFlagView cancellation) {
    if (quote.impl == nullptr || !qsa_live_continuation() || state_store == nullptr) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (quote.path_kind == 2 && qsa_store_ == nullptr) {
        // The quote named a stored image but the store is gone: wait for the next admission.
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const std::uint32_t lane = quote.destination.value;
    const QsaQuoteImpl& plan = *quote.impl;
    const auto n             = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (lane >= max_concurrency) {
        throw std::out_of_range("QSA reserve lane is out of range");
    }
    if (active_continuations[lane] < continuation_capacity ||
        requests[lane].prefill.has_value()) {
        // The lane filled up between the quote and the reservation: the request waits.
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }

    const std::uint32_t reuse = std::min(quote.reuse, n);
    const bool from_store     = quote.from_store && reuse != 0;

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    std::optional<KVAddressSpaceHandle> adopted_backend;
    std::uint32_t adopted_backend_pages = 0;
    bool continuation_bound             = false;
    const std::uint32_t continuation_index = active_continuations[lane];
    try {
        const std::optional<std::uint32_t> continuation = allocate_continuation_slot();
        if (!continuation) { return runtime::ContextTransactionReserveStatus::Aborted; }
        active_continuations[lane] = *continuation;
        continuation_bound         = true;
        SequenceState& sequence    = continuation_states[*continuation];
        sequence.lane              = lane;

        // State: the lane's own live slot when the prompt continues it, a copy of the lane's
        // prompt-end snapshot when the prompt continues that, else the image's state half. The
        // order is upstream's begin() (qwen4_exp_core.cpp:1655-1690).
        QsaLaneRetention& retention = qsa_lane_retention_[lane];
        bool state_whole            = false;
        const bool consumes_live     = reuse != 0 && quote.path_kind == 0;
        const bool consumes_snapshot = reuse != 0 && quote.path_kind == 1;
        if (!consumes_live && !consumes_snapshot) {
            // Root or store: the lane's retained states belong to a prefix this prompt does not
            // continue, and their planes are about to be overwritten. Drop them first, so the
            // destination reservation below never meets a full pool of the lane's own making.
            qsa_drop_lane_retention(lane, !from_store);
        }
        if (reuse == 0) {
            state = state_store->reserve_destination();
            if (!state) { throw std::runtime_error("QSA admission has no StateImage destination"); }
            sequence.state = ActiveStateBinding{.read = *state, .write = *state};
            // The reset zeroes the slot and makes it mutable; `ordered_reset` then requires
            // exactly that role, so the order is not interchangeable.
            state_store->activate_reset(*state, compute_streams);
            ordered_reset(sequence);
        } else if (consumes_live) {
            // The lane's live state *is* the prefix: adopt it without a copy.
            if (!retention.live) {
                throw std::runtime_error("QSA admission lost the lane's live state");
            }
            state_store->move_checkpoint_to_active(*retention.live);
            state = retention.live;
            retention.live.reset();
            sequence.state = ActiveStateBinding{.read = *state, .write = *state};
        } else if (quote.path_kind == 1) {
            // The prompt continues the lane's last prompt, not its generation: resume from the
            // snapshot taken where that prompt ended, and let the suffix prefill overwrite the
            // rest. The lane owns that snapshot and no other conversation can name it, so adopt
            // its slot instead of copying: the pool is C + C (a live and a snapshot per lane) and
            // with every lane occupied there is no spare destination for a 147 MiB copy. The
            // next prompt-end capture re-establishes the snapshot in a free slot.
            if (!retention.snapshot) {
                throw std::runtime_error("QSA admission lost the lane's prompt snapshot");
            }
            state_store->move_checkpoint_to_active(*retention.snapshot);
            state = retention.snapshot;
            retention.snapshot.reset();
            // The old endpoint is the history this prompt did *not* continue (that is why the
            // snapshot won), and keeping it would put two frozen slots beside this lane's active
            // one with no room left for the next prompt-end capture.
            if (retention.live && state_store->can_release(*retention.live)) {
                (void)state_store->release(*retention.live);
            }
            retention.live.reset();
            sequence.state = ActiveStateBinding{.read = *state, .write = *state};
            state_whole    = quote.whole;
        } else {
            // Some other conversation owns the lane: the image is the only source left, and it
            // carries both halves.
            if (quote.image >= qsa_store_->images().size()) {
                throw std::runtime_error("QSA admission image is out of range");
            }
            const HeldPrefix key = qsa_store_->images()[quote.image].key;
            const std::optional<std::size_t> image = qsa_store_->stored_index(key);
            if (!image) { throw std::runtime_error("QSA admission lost its image"); }
            const StoredImage& stored            = qsa_store_->images()[*image];
            const std::vector<std::byte*>& pages = stored.pages;
            const std::uint64_t state_bytes      = qsa_state_image_bytes();
            if (state_bytes == 0 || stored.bytes < state_bytes) {
                throw std::runtime_error("QSA admission image has no StateImage part");
            }
            state = state_store->reserve_destination();
            if (!state) { throw std::runtime_error("QSA admission has no StateImage destination"); }
            // The StateImage half: GDN, the continuation hidden, any draft state.
            // `refresh_state_views` below re-binds `sequence.tail_hidden` to the destination's
            // continuation-hidden region, the same view a Legacy checkpoint restore reads.
            state_images->copy_from_host_segments(
                pages, QsaContinuationStore::kHostPageBytes,
                state_store->physical_slot(*state), compute_streams, nullptr, lane);
            state_store->activate_copied(*state);
            sequence.state = ActiveStateBinding{.read = *state, .write = *state};
            std::vector<ImageChunk> chunks;
            qsa_collect_plane_chunks(*qsa_stream_state, lane, chunks);
            if (stored.bytes < state_bytes + qsa_image_chunk_bytes(chunks)) {
                throw std::runtime_error("QSA admission image has no KV part");
            }
            std::vector<ImageRegion> regions;
            regions.reserve(chunks.size());
            for (const ImageChunk& chunk : chunks) {
                regions.push_back(ImageRegion{chunk.data, chunk.bytes});
            }
            qsa_copy_image(regions, pages, QsaContinuationStore::kHostPageBytes, false,
                           state_bytes / QsaContinuationStore::kHostPageBytes);
            state_whole = quote.whole;
        }

        // The device text-KV pages are bookkeeping under --kv-resident: the authority is the
        // lane's pinned planes, so the address space only has to cover the entitlement.
        text_address = text_kv_addresses->create_inactive();
        if (!text_address) { throw std::runtime_error("QSA admission has no text KV address space"); }
        text_kv_addresses->activate(*text_address, plan.text_kv_page_entitlement,
                                    static_cast<std::int32_t>(lane), compute_streams);
        SequenceKVBundle bundle{.text = *text_address};
        const bool backend_pool = backend_kv_addresses != nullptr &&
                                  plan.backend_kv_page_entitlement != 0;
        if (backend_pool) {
            // The draft KV never streams, so a live endpoint keeps the address space that holds
            // it: it covers [0, the endpoint) and adopting it hands the drafter exactly the KV it
            // wrote. A turn-closure resume is different: its anchor sits at the *prompt* end,
            // while the lane's backend space was committed on to the later endpoint (the generated
            // tokens). Adopting it here leaves its committed frontier ahead of this request's
            // mtp_kv_valid, so the next commit would have to walk the frontier backwards and
            // throws. Regeneration discards those generated tokens anyway, so give it a fresh
            // backend space whose frontier starts at zero. A different entitlement starts from
            // scratch for the same reason.
            if (consumes_live && retention.backend &&
                retention.backend_pages == plan.backend_kv_page_entitlement) {
                adopted_backend       = *retention.backend;
                adopted_backend_pages = retention.backend_pages;
                retention.backend.reset();
                retention.backend_pages = 0;
                bundle.backend          = *adopted_backend;
            } else {
                // The lane's retained space covers a different extent than this prompt's (an
                // entitlement follows the prompt, so a growing conversation never matches), and
                // `activate` takes the lane's row exclusively: the old space has to go back to
                // the pool first, or the row is still bound when the new one asks for it. The
                // draft KV the bridge needs is re-established at the resume frontier below.
                if (retention.backend && backend_kv_addresses->active(*retention.backend)) {
                    backend_kv_addresses->deactivate(*retention.backend);
                    (void)backend_kv_addresses->release(*retention.backend);
                }
                retention.backend.reset();
                retention.backend_pages = 0;
                backend_address = backend_kv_addresses->create_inactive();
                if (!backend_address) {
                    throw std::runtime_error("QSA admission has no backend KV address space");
                }
                backend_kv_addresses->activate(*backend_address,
                                               plan.backend_kv_page_entitlement,
                                               static_cast<std::int32_t>(lane), compute_streams);
                bundle.backend = *backend_address;
            }
        }
        sequence.kv.emplace(bundle);

        const bool prepare_mtp = speculative_backend == SpeculativeBackend::Mtp;
        refresh_state_views(sequence);
        sequence.text_kv_valid = reuse;
        sequence.mtp_kv_valid  = prepare_mtp && reuse != 0 ? reuse - 1U : 0U;
        sequence.dflash_context_frontier =
            is_masked_draft_backend(speculative_backend) ? reuse : 0U;
        // A prompt that ends exactly at the resume is sampled from the state's own continuation
        // hidden; `refresh_state_views` bound the view to it, so nothing has to be copied.
        sequence.tail_hidden_valid = state_whole;
        sequence.endpoint_valid    = false;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt);
        sequence.prefix_digests.assign(prompt);
        sequence.rebuild_work       = plan.root_rebuild_work;
        sequence.rebuild_tail_begin = plan.root_rebuild_tail_begin;
        bind_sequence_kv(sequence);
        write_lane_scales(sequence, prompt);

        const std::uint32_t initial_mtp_extent =
            prepare_mtp ? std::min({draft_window,
                                    plan.summary.effective_output_tokens > 1
                                        ? plan.summary.effective_output_tokens - 2
                                        : 0U,
                                    capacity - n > 0 ? capacity - n - 1 : 0U})
                        : 0U;
        const std::uint32_t backend_materialized =
            prepare_mtp
                ? std::min(capacity, n + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
                : (speculative_backend == SpeculativeBackend::DFlash ? n : 0U);
        ensure_sequence_kv_mapped(sequence, n, backend_materialized);

        RequestControl& request = requests[lane];
        request.grammar         = plan.grammar;
        request.logprobs        = plan.logprobs;
        request.readout_tokens  = plan.readout_tokens;
        request.branches        = plan.branches;
        install_sampling(sequence, request, plan.sampling);
        sequence.mtp_draft_count    = 0;
        sequence.mtp_drafts_pending = false;
        sequence.rope_delta         = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);
        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = plan.summary.publish_continuation && plan.drafting &&
                                       !plan.branches;
        request.lease_ceiling =
            std::min(capacity, n + std::max(plan.summary.effective_output_tokens == 0
                                                ? 0U
                                                : plan.summary.effective_output_tokens - 1U,
                                            branch_extent(plan.branches.get())));
        request.active_resources = detail::PhysicalResources{
            .device = {.active_lanes     = 1,
                       .state_slots      = 1,
                       .main_kv_pages    = plan.text_kv_page_entitlement,
                       .backend_kv_pages = backend_pool ? plan.backend_kv_page_entitlement : 0U}};
        request.optional_resources = {};

        // Vision: the same suffix plan the hybrid reservation builds (hybrid_program.cpp), and
        // for the same reason -- the image tokens a resuming prompt still has to write are the
        // ones past `reuse`, and each of them is a prefill split where the encoder's handoff is
        // spliced into the text hidden state. A store image carries the media it was written with
        // (`qsa_choose_reuse` compares `block_extras`), so an item wholly inside the resumed
        // prefix is already in the lane's planes and must not be encoded again.
        PreparedPromptData staged_prompt = prompt;
        std::optional<VisionPrefillPlan> vision;
        if (plan.vision_control_plan) {
            VisionPrefillPlan vision_plan;
            vision_plan.control_plan = plan.vision_control_plan;
            std::size_t max_merged   = 0;
            for (std::size_t index_item = 0; index_item < plan.vision_control_plan->items.size();
                 ++index_item) {
                const qwen3_5::VisionItemControlPlan& item =
                    plan.vision_control_plan->items[index_item];
                if (item.token_end <= reuse) { continue; }
                const std::uint32_t begin =
                    prepare_mtp && item.token_begin != 0 ? item.token_begin - 1 : item.token_begin;
                vision_plan.uses.push_back(VisionUseSpan{
                    .begin               = begin,
                    .end                 = item.token_end,
                    .prepared_item_index = static_cast<std::uint32_t>(index_item),
                });
                max_merged = std::max(max_merged, item.merged_count);
            }
            if (!vision_plan.uses.empty()) {
                vision_plan.max_merged_count = max_merged;
                std::vector<bool> used(staged_prompt.media_payloads.size(), false);
                for (const VisionUseSpan& use : vision_plan.uses) {
                    used[use.prepared_item_index] = true;
                }
                for (std::size_t item = 0; item < used.size(); ++item) {
                    if (!used[item]) { staged_prompt.media_payloads[item].reset(); }
                }
                const std::uint32_t first_item = vision_plan.uses.front().prepared_item_index;
                auto control                   = std::make_shared<qwen3_5::VisionControl>(
                    qwen3_5::build_vision_control(staged_prompt, *vision_plan.control_plan,
                                                  first_item));
                for (VisionUseSpan& use : vision_plan.uses) {
                    use.control_index = use.prepared_item_index - first_item;
                }
                vision_plan.control = std::move(control);
                vision_plan.control_plan.reset();
                vision = std::move(vision_plan);
            }
        }
        if (staged_prompt.has_media() && !vision) { staged_prompt.release_all_media_payloads(); }

        RequestControl::Prefill prefill{
            .prompt             = std::move(staged_prompt),
            .vision_plan        = std::move(vision),
            .vision             = nullptr,
            .capture_groups     = {},
            .base               = reuse,
            .cursor             = reuse,
            .prompt_tokens      = n,
            .initial_mtp_extent = initial_mtp_extent,
            .drafter_frontier   = std::nullopt,
            .elapsed_seconds    = 0.0,
            .prepare_mtp        = prepare_mtp,
            .reuse              = plan.summary.prefix_reuse_path,
            // A resume with a suffix bridges the state into the tokens about to be prefilled;
            // a whole-prompt match (reuse == n, a regenerate) has no suffix and bridges after the
            // exact hit instead (request_plan.cpp:1000,1007). Leaving either mode unset makes
            // prefill throw on a zero-suffix MTP reuse.
            .mtp_bridge = prepare_mtp && reuse != 0
                              ? (reuse < n ? MtpBridgeMode::BeforeSuffix
                                           : MtpBridgeMode::AfterExactHit)
                              : MtpBridgeMode::None,
        };
        request.prefill.emplace(std::move(prefill));
        if (request.prefill->vision_plan) {
            if (!workspace_plan.vision) {
                throw std::logic_error("Vision prefill has no startup workspace plan");
            }
            if (parameters.model.cpu_vision()) {
                request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                    device, parameters, *workspace_plan.vision, request.prefill->prompt,
                    *request.prefill->vision_plan, vision_handoff_peak_bytes,
                    DeviceSpan{static_cast<std::byte*>(workspace_storage.base()) +
                                   workspace_plan.vision_bridge_offset,
                               workspace_plan.vision_bridge_bytes});
                // The first item starts encoding on CPU threads while other lanes decode.
                request.prefill->vision->submit_next_item();
            } else if (vision_broker) {
                request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                    device, parameters, *workspace_plan.vision, request.prefill->prompt,
                    *request.prefill->vision_plan, vision_handoff_peak_bytes, *vision_broker,
                    vision_results->acquire(),
                    DeviceSpan{static_cast<std::byte*>(workspace_storage.base()) +
                                   workspace_plan.vision_bridge_offset,
                               workspace_plan.vision_bridge_bytes});
                request.prefill->vision->submit_next_item();
            } else {
                request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                    device, parameters,
                    DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                    *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                    vision_handoff_peak_bytes);
            }
        }
        request.lifecycle = Lifecycle::Prefilling;

        // The lane's tenure starts here, so the epoch is bumped before the record is stamped:
        // stamping first would leave every record one epoch behind its lane and the admission
        // would silently see no live prefix at all.
        invalidate_lane(lane);
        qsa_begin_lane_prefix(lane, prompt, sequence);
        advance_resource_revision();
        qsa_pending_published_ = StartResult{
            .sequence = ContractAccess::make_sequence(this, runtime::LaneId{lane},
                                                      lane_epochs[lane])};
        // One line per admitted request, at the point the lane is actually bound: unlike the
        // [resume] line (which reports the best-scoring lane from the trace scan, not the lane
        // chosen), this names the destination the request will run on. The 6s4c harness reads it
        // to map each session's prompt size to its lane. Gated by the same switch as [resume].
        if (qsa_reuse_trace_enabled()) {
            std::fprintf(stderr, "[admit] lane=%u prompt=%u reuse=%u kind=%u\n", lane, n, reuse,
                         quote.path_kind);
            std::fflush(stderr);
        }
        (void)take_prompt;
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        if (continuation_bound) {
            SequenceState& failed = continuation_states[active_continuations[lane]];
            failed.state          = {};
            if (adopted_backend) {
                qsa_lane_retention_[lane].backend       = adopted_backend;
                qsa_lane_retention_[lane].backend_pages = adopted_backend_pages;
                if (failed.kv) { failed.kv->backend.reset(); }
            }
            failed.kv.reset();
        }
        if (text_address) {
            try {
                if (text_kv_addresses->active(*text_address)) {
                    text_kv_addresses->deactivate(*text_address);
                }
                (void)text_kv_addresses->release(*text_address);
            } catch (...) {}
        }
        if (backend_address) {
            try {
                if (backend_kv_addresses->active(*backend_address)) {
                    backend_kv_addresses->deactivate(*backend_address);
                }
                (void)backend_kv_addresses->release(*backend_address);
            } catch (...) {}
        }
        if (state) {
            try { (void)state_store->release(*state); } catch (...) {}
        }
        if (continuation_bound) { active_continuations[lane] = continuation_index; }
        throw;
    }
}

// ---- terminal settlement ---------------------------------------------------------------------

bool ProgramImpl::qsa_finish_lane(SequenceState& sequence, RequestControl& request,
                                  std::uint32_t lane, bool consistent) noexcept {
    if (!qsa_live_continuation() || state_store == nullptr || lane >= max_concurrency) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity) { return false; }
    // A lane whose state may not describe its frontier is released the ordinary way: the caller
    // falls back to abort, which drops the record and the retained slots with the tenure.
    if (!consistent) { return false; }
    const std::uint32_t continuation = active_continuations[lane];
    // Nothing is copied aside here. The lane keeps its live state and its prompt-end snapshot as
    // the two resume sources; the image is written only when a later tenant evicts that state.
    try {
        if (sequence.reserved_state) {
            if (!state_store->release(*sequence.reserved_state)) { return false; }
            sequence.reserved_state.reset();
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
            sequence.rewrite_state.reset();
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }
        sequence.long_anchors.clear();

        QsaLaneRetention& retention = qsa_lane_retention_[lane];
        // The live state this turn ends in becomes the lane's own: the next prompt that continues
        // it resumes with no copy at all (upstream's `live` branch). It supersedes the last
        // turn's, whose slot is released here.
        if (retention.live && state_store->can_release(*retention.live)) {
            (void)state_store->release(*retention.live);
        }
        retention.live.reset();
        retention.epoch = lane_epochs[lane];
        if (state_store->valid(sequence.state.read) && !sequence.state.fork_pending &&
            state_store->role(sequence.state.read) == StateImageRole::ActiveMutable) {
            state_store->freeze(sequence.state.read);
            retention.live = sequence.state.read;
            // Remember whether the endpoint state sits exactly at the end of its prefix key;
            // a stop-string cut leaves it ahead of the key, so the eventual image cannot be
            // used for a whole-prompt resume.
            retention.live_whole =
                sequence.lane < qsa_lane_prefix.size() &&
                qsa_lane_prefix[sequence.lane].held.position == sequence.text_kv_valid;
            sequence.state = {};
        } else {
            return false; // not retainable: the caller aborts and releases everything
        }
        sequence.tail_hidden       = {};
        sequence.tail_hidden_valid = false;

        // The draft KV stays with the lane: it covers a prefix of every later prompt, so keeping
        // its address space active is what lets a drafting request resume at all.
        if (speculative_backend == SpeculativeBackend::Mtp && sequence.kv && sequence.kv->backend &&
            backend_kv_addresses != nullptr && backend_kv_addresses->active(*sequence.kv->backend)) {
            retention.backend       = *sequence.kv->backend;
            retention.backend_pages = backend_kv_addresses->entitlement(*sequence.kv->backend);
            // Not deactivated: an inactive address space returns its pages to the pool.
            sequence.kv->backend.reset();
        }
        release_active_shared_references_strict(sequence);
        if (sequence.kv) {
            if (!text_kv_addresses->can_release_after_deactivate(sequence.kv->text)) {
                return false;
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
            if (!text_kv_addresses->release(sequence.kv->text)) { return false; }
            sequence.kv.reset();
        }
        if (host_kv_extents != nullptr) { (void)host_kv_extents->release_unreferenced(); }
    } catch (...) {
        return false;
    }
    request.active_resources   = {};
    request.optional_resources = {};
    request.prefill.reset();
    request.lifecycle = Lifecycle::Empty;
    request.pending   = {};
    // Frees the continuation slot, clears the lane binding and bumps the slot generation.
    retire_continuation_slot(continuation);
    // The staging copy is keyed on the tenure, and the prefix record survives on purpose: the
    // next prompt that continues this lane resumes from what it retained.
    qsa_end_lane_tenure(lane);
    advance_resource_revision();
    return true;
}

// ---- images for the store tier -----------------------------------------------------------------

void ProgramImpl::qsa_forget(SequenceState& sequence) noexcept {
    if (sequence.lane < qsa_lane_prefix.size()) { qsa_lane_prefix[sequence.lane].clear(); }
}

void ProgramImpl::qsa_retire_handle(SequenceHandle sequence) noexcept {
    if (!valid_sequence(sequence)) { return; }
    qsa_retire(active_sequence(ContractAccess::lane(sequence).value));
}

void ProgramImpl::qsa_forget_handle(SequenceHandle sequence) noexcept {
    if (!valid_sequence(sequence)) { return; }
    qsa_forget(active_sequence(ContractAccess::lane(sequence).value));
}

void ProgramImpl::qsa_write_lane_image(std::uint32_t lane, StateImageHandle state,
                                       bool whole) noexcept {
    if (qsa_store_ == nullptr || !qsa_stream_state.has_value() || lane >= max_concurrency ||
        qsa_lane_prefix.size() != max_concurrency || !state.valid()) {
        return;
    }
    LanePrefixRecord& record = qsa_lane_prefix[lane];
    if (record.epoch == 0 || record.epoch != lane_epochs[lane] || record.held.empty()) { return; }
    try {
        const std::uint64_t state_bytes = qsa_state_image_bytes();
        const std::uint64_t plane_bytes = qsa_plane_image_bytes(lane);
        if (state_bytes == 0 || plane_bytes == 0) { return; }
        std::vector<ImageChunk> chunks;
        qsa_collect_plane_chunks(*qsa_stream_state, lane, chunks);
        const std::uint64_t payload = state_bytes + plane_bytes;
        std::vector<ImageRegion> regions;
        regions.reserve(chunks.size());
        for (const ImageChunk& chunk : chunks) {
            regions.push_back(ImageRegion{chunk.data, chunk.bytes});
        }
        const std::uint32_t physical = static_cast<std::uint32_t>(state_store->physical_slot(state));
        // Only the endpoint goes to the store: a turn closure resumes from the lane's retained
        // snapshot, and storing a second full image per eviction would halve how many lanes the
        // budget holds. A lane some other conversation took therefore resumes an endpoint
        // continuation, not a rollback to its last prompt.
        if (record.held.tokens.size() >= QsaContinuationStore::kMinStoredTokens &&
            qsa_store_->store_image(record.held, payload, 0, whole)) {
            const std::optional<std::size_t> index = qsa_store_->stored_index(record.held);
            if (index) {
                const std::vector<std::byte*>& pages = qsa_store_->images()[*index].pages;
                // The StateImage half first, then the planes at their page offset.
                state_images->copy_to_host_segments(physical, pages,
                                                    QsaContinuationStore::kHostPageBytes,
                                                    compute_streams, false, nullptr, lane);
                qsa_copy_image(regions, pages, QsaContinuationStore::kHostPageBytes, true,
                               state_bytes / QsaContinuationStore::kHostPageBytes);
                // The StateImage D2H is queued on the device streams; the image is only visible
                // to a later restore once it has landed.
                device.synchronize();
            }
        }
        if (qsa_reuse_trace_enabled()) {
            std::fprintf(stderr,
                         "[retire] lane=%u anchor=%u held=%u whole=%d store_used=%llu "
                         "store_budget=%llu\n",
                         lane, record.anchor.position, record.held.position, whole ? 1 : 0,
                         static_cast<unsigned long long>(qsa_store_->used_bytes()),
                         static_cast<unsigned long long>(qsa_store_->budget_bytes()));
            std::fflush(stderr);
        }
    } catch (...) {
        // A store failure never fails the request that caused the eviction.
    }
}

void ProgramImpl::qsa_store_lane_image(std::uint32_t lane) noexcept {
    if (qsa_store_ == nullptr || lane >= max_concurrency) { return; }
    const QsaLaneRetention& retention = qsa_lane_retention_[lane];
    if (retention.epoch == 0 || retention.epoch != lane_epochs[lane] || !retention.live ||
        state_store == nullptr || !state_store->valid(*retention.live)) {
        return;
    }
    qsa_write_lane_image(lane, *retention.live, retention.live_whole);
}

void ProgramImpl::qsa_retire(SequenceState& sequence) noexcept {
    if (sequence.lane >= max_concurrency || qsa_lane_prefix.size() != max_concurrency) { return; }
    const LanePrefixRecord& record = qsa_lane_prefix[sequence.lane];
    qsa_write_lane_image(sequence.lane, sequence.state.read,
                         record.held.position == sequence.text_kv_valid);
}

} // namespace ninfer::models::qwen3_5::detail
