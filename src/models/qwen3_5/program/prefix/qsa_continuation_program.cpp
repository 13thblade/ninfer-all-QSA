// QSA streamed-tier reuse (ContextCacheMode::Qsa): quote, reserve and retirement.
//
// The streamed tier keeps its KV in each lane's pinned host planes and its GDN conv/recurrent
// state plus the continuation hidden in the sequence's StateImage slot. An image therefore holds
// both halves, in one byte order:
//
//   [0, state_bytes)                 the StateImage host image (GDN conv/recurrent, hidden, draft)
//   [state_bytes, + plane_bytes)     the lane's KV planes, layer by layer (K, V, K-scale, V-scale)
//
// `state_bytes` is page-rounded, so the plane half starts on a page and each half reads and writes
// its own view of the store's pages. Restoring an image lands the StateImage half into the
// destination slot (device-to-host at save, host-to-device at load) and the plane half into the
// lane's pinned planes (host-to-host): after a restore the lane holds exactly what it held when
// the image was taken, which is what lets a store resume carry GDN state a lane no longer has.

#include "models/qwen3_5/program/prefix/qsa_continuation.h"
#include "models/qwen3_5/program/prefix/qsa_image.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

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

    auto quote                         = std::make_shared<QsaQuoteImpl>();
    quote->summary                     = plan.summary;
    quote->sampling                    = plan.sampling;
    quote->grammar                      = plan.grammar;
    quote->logprobs                     = plan.logprobs;
    quote->readout_tokens               = plan.readout_tokens;
    quote->branches                     = plan.branches;
    quote->drafting                     = plan.drafting;
    quote->text_kv_page_entitlement     = plan.text_kv_page_entitlement;
    quote->backend_kv_page_entitlement  = plan.backend_kv_page_entitlement;
    quote->vision_control_plan          = plan.vision_control_plan;
    quote->root_rebuild_work            = plan.root_rebuild_work;
    quote->root_rebuild_tail_begin      = plan.root_rebuild_tail_begin;
    quote->destination                  = destination.value;
    quote->destination_epoch            = lane_epochs[destination.value];

    QsaAdmissionQuote out;
    out.destination = destination;
    out.summary     = plan.summary;
    out.impl        = std::move(quote);

    const auto lane_free = [&]() {
        return active_continuations[destination.value] >= continuation_capacity &&
               !requests[destination.value].prefill.has_value();
    };

    if (qsa_store_ == nullptr || !plan.allow_prefix_reuse) {
        // No host budget, or reuse off (serve warmup, --no-prefix-reuse): root only -- the
        // streamed tier's pre-R4 shape.
        out.summary.reusable_prompt_tokens = 0;
        out.summary.prefix_reuse_path      = PrefixReusePath::Root;
        out.readiness =
            lane_free() ? runtime::Readiness::Ready : runtime::Readiness::TemporarilyBlocked;
        return out;
    }

    const QsaReuseChoice choice =
        qsa_choose_reuse(qsa_lane_prefix, lane_epochs, qsa_store_.get(), prompt.token_ids,
                         prompt.block_extras);
    out.reuse      = choice.tokens;
    out.lane       = choice.lane;
    out.image      = choice.image;
    out.from_store = choice.from_store;
    out.whole      = choice.whole;
    out.path_kind  = choice.path_kind;
    out.summary.reusable_prompt_tokens = choice.tokens;
    out.summary.prefix_reuse_path =
        choice.tokens == 0
            ? PrefixReusePath::Root
            : (choice.path_kind == 1 ? PrefixReusePath::PrivateTurnClosure
                                     : PrefixReusePath::PrivateEndpoint);
    out.readiness = lane_free() ? runtime::Readiness::Ready
                                : runtime::Readiness::TemporarilyBlocked;
    return out;
}

// ---- reserve ---------------------------------------------------------------------------------

runtime::ContextTransactionReserveStatus ProgramImpl::qsa_reserve_materialization(
    QsaAdmissionQuote&& quote, const PreparedPromptData& prompt,
    const std::function<PreparedPromptData()>& take_prompt,
    runtime::CancellationFlagView cancellation) {
    if (quote.impl == nullptr || !qsa_stream_state.has_value()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (qsa_store_ == nullptr && (quote.from_store || quote.reuse != 0)) {
        // Without a store the tier admits roots only; a stale reused quote waits for the next
        // admission instead of starting the prompt from the wrong baseline.
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
    // A drafting request resumes only from a state its drafter followed. An image does not carry
    // the MTP block's KV yet, so a store resume stays out of reach while MTP runs; the lane's own
    // state (root) does not need it.
    if (from_store && speculative_backend == SpeculativeBackend::Mtp) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    bool continuation_bound = false;
    const std::uint32_t continuation_index = active_continuations[lane];
    try {
        const std::optional<std::uint32_t> continuation = allocate_continuation_slot();
        if (!continuation) { return runtime::ContextTransactionReserveStatus::Aborted; }
        active_continuations[lane] = *continuation;
        continuation_bound         = true;
        SequenceState& sequence    = continuation_states[*continuation];
        sequence.lane              = lane;

        state = state_store->reserve_destination();
        if (!state) { throw std::runtime_error("QSA admission has no StateImage destination"); }
        sequence.state = ActiveStateBinding{.read = *state, .write = *state};

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
            backend_address = backend_kv_addresses->create_inactive();
            if (!backend_address) {
                throw std::runtime_error("QSA admission has no backend KV address space");
            }
            backend_kv_addresses->activate(*backend_address, plan.backend_kv_page_entitlement,
                                           static_cast<std::int32_t>(lane), compute_streams);
            bundle.backend = *backend_address;
        }
        sequence.kv.emplace(bundle);

        const bool prepare_mtp = speculative_backend == SpeculativeBackend::Mtp;
        if (reuse == 0) {
            ordered_reset(sequence);
            state_store->activate_reset(*state, compute_streams);
        } else if (from_store) {
            // `image` is the quote's store index; restore by its key so recency is refreshed.
            if (quote.image >= qsa_store_->images().size()) {
                throw std::runtime_error("QSA admission image is out of range");
            }
            const HeldPrefix stored_key = qsa_store_->images()[quote.image].key;
            const auto image            = qsa_store_->restore_stored(stored_key);
            if (!image) { throw std::runtime_error("QSA admission lost its stored image"); }
            const StoredImage& stored = qsa_store_->images()[*image];
            const std::vector<std::byte*>& pages = stored.pages;
            const std::uint64_t state_bytes       = qsa_state_image_bytes();
            if (state_bytes == 0 || stored.bytes < state_bytes) {
                throw std::runtime_error("QSA admission image has no StateImage part");
            }
            state_images->copy_from_host_segments(
                pages, QsaContinuationStore::kHostPageBytes,
                state_store->physical_slot(*state), compute_streams, nullptr, lane);
            state_store->activate_copied(*state);
            std::vector<ImageChunk> chunks;
            qsa_collect_plane_chunks(*qsa_stream_state, lane, chunks);
            std::vector<ImageRegion> regions;
            regions.reserve(chunks.size());
            for (const ImageChunk& chunk : chunks) {
                regions.push_back(ImageRegion{chunk.data, chunk.bytes});
            }
            qsa_copy_image(regions, pages, QsaContinuationStore::kHostPageBytes, false,
                           state_bytes / QsaContinuationStore::kHostPageBytes);
            sequence.tail_hidden_valid = true;
        } else {
            // A live or anchor resume: the lane's own state is already where it belongs.
            state_store->activate_copied(*state);
            sequence.tail_hidden_valid = quote.whole;
        }

        refresh_state_views(sequence);
        sequence.text_kv_valid = reuse;
        sequence.mtp_kv_valid  = prepare_mtp && reuse != 0 ? reuse - 1U : 0U;
        sequence.dflash_context_frontier =
            is_masked_draft_backend(speculative_backend) ? reuse : 0U;
        sequence.endpoint_valid = false;
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

        RequestControl::Prefill prefill{
            .prompt             = prompt,
            .vision_plan        = std::nullopt,
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
            .mtp_bridge = prepare_mtp && reuse != 0 ? MtpBridgeMode::BeforeSuffix
                                                    : MtpBridgeMode::None,
        };
        request.prefill.emplace(std::move(prefill));
        request.lifecycle = Lifecycle::Prefilling;

        qsa_begin_lane_prefix(lane, prompt, sequence);
        invalidate_lane(lane);
        advance_resource_revision();
        qsa_pending_published_ = StartResult{
            .sequence = ContractAccess::make_sequence(this, runtime::LaneId{lane},
                                                      lane_epochs[lane])};
        (void)take_prompt;
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        if (text_address) {
            if (text_kv_addresses->active(*text_address)) {
                text_kv_addresses->deactivate(*text_address);
            }
            (void)text_kv_addresses->release(*text_address);
        }
        if (backend_address) {
            if (backend_kv_addresses->active(*backend_address)) {
                backend_kv_addresses->deactivate(*backend_address);
            }
            (void)backend_kv_addresses->release(*backend_address);
        }
        if (state) { (void)state_store->release(*state); }
        if (continuation_bound) { active_continuations[lane] = continuation_index; }
        throw;
    }
}

// ---- retirement ------------------------------------------------------------------------------

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

void ProgramImpl::qsa_retire(SequenceState& sequence) noexcept {
    if (qsa_store_ == nullptr || !qsa_stream_state.has_value() ||
        sequence.lane >= max_concurrency || qsa_lane_prefix.size() != max_concurrency) {
        return;
    }
    LanePrefixRecord& record = qsa_lane_prefix[sequence.lane];
    if (record.epoch == 0 || record.epoch != lane_epochs[sequence.lane] ||
        record.held.empty() || !sequence.state.read.valid()) {
        return;
    }
    try {
        const std::uint64_t state_bytes = qsa_state_image_bytes();
        const std::uint64_t plane_bytes = qsa_plane_image_bytes(sequence.lane);
        if (state_bytes == 0 || plane_bytes == 0) { return; }
        std::vector<ImageChunk> chunks;
        qsa_collect_plane_chunks(*qsa_stream_state, sequence.lane, chunks);
        const std::uint64_t payload = state_bytes + plane_bytes;
        std::vector<ImageRegion> regions;
        regions.reserve(chunks.size());
        for (const ImageChunk& chunk : chunks) {
            regions.push_back(ImageRegion{chunk.data, chunk.bytes});
        }
        const std::uint32_t physical = static_cast<std::uint32_t>(
            state_store->physical_slot(sequence.state.read));
        const auto store_one = [&](const HeldPrefix& key, std::uint32_t path_kind) {
            if (key.tokens.size() < QsaContinuationStore::kMinStoredTokens) { return; }
            if (!qsa_store_->store_image(key, payload, path_kind)) { return; }
            const std::optional<std::size_t> index = qsa_store_->stored_index(key);
            if (!index) { return; }
            const std::vector<std::byte*>& pages = qsa_store_->images()[*index].pages;
            // The StateImage half first, then the planes at their page offset.
            state_images->copy_to_host_segments(physical, pages,
                                                QsaContinuationStore::kHostPageBytes,
                                                compute_streams, false, nullptr, sequence.lane);
            qsa_copy_image(regions, pages, QsaContinuationStore::kHostPageBytes, true,
                           state_bytes / QsaContinuationStore::kHostPageBytes);
        };
        store_one(record.anchor, 1);
        if (record.held.position > record.anchor.position) { store_one(record.held, 0); }
        if (qsa_reuse_trace_enabled()) {
            std::fprintf(stderr,
                         "[retire] lane=%u anchor=%u held=%u store_used=%llu store_budget=%llu\n",
                         sequence.lane, record.anchor.position, record.held.position,
                         static_cast<unsigned long long>(qsa_store_->used_bytes()),
                         static_cast<unsigned long long>(qsa_store_->budget_bytes()));
            std::fflush(stderr);
        }
    } catch (...) {
        // A store failure never fails the request that just finished.
    }
}

} // namespace ninfer::models::qwen3_5::detail
