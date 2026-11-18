#pragma once

#include "ninfer/base.h"
#include "ninfer/observability/execution.h"
#include "ninfer/observability/snapshot.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/resource_manager.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::runtime {

// Engine-side driver of the QSA streamed-tier continuation store (docs: the live-continuation
// plan, R4). It presents the same ResourceManager surface HybridResourceManager does, so the
// Engine's scheduling, prefill, decode and commit orchestration are shared; what differs is where
// the resume comes from and how it lands:
//
//   * the quote is the three-way choice between a lane's live prefix, its anchor and a stored
//     image, settled by the Program from records advanced where the KV frontier moves;
//   * the reservation completes synchronously -- the image's two halves are the StateImage host
//     image (GDN conv/recurrent, continuation hidden) and the lane's pinned KV planes, so there is
//     no staged device restore for the Engine to progress. `progress_context_transaction`
//     publishes the sequence the reservation already started;
//   * terminal settlement retires the lane's prefix into the store before the lane is released.
template <class ModelContract>
class QsaResourceManager {
public:
    using Program                      = typename ModelContract::Program;
    using PreparedPrompt               = typename ModelContract::PreparedPrompt;
    using RequestBasePlan              = typename ModelContract::RequestBasePlan;
    using PersistentBackfillProof      = typename ModelContract::PersistentBackfillProof;
    using SequenceHandle               = typename ModelContract::SequenceHandle;
    using CaptureOffer                 = typename ModelContract::CaptureOffer;
    using StartResult                  = typename ModelContract::StartResult;
    using FinishResult                 = typename ModelContract::FinishResult;
    using AbortResult                  = typename ModelContract::AbortResult;
    using Quote                        = typename ModelContract::QsaAdmissionQuote;

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return quote_.summary; }
        [[nodiscard]] LaneId destination() const noexcept { return quote_.destination; }
        [[nodiscard]] bool needs_transfer() const noexcept { return false; }

    private:
        explicit Choice(Quote&& quote) : quote_(std::move(quote)) {}

        Quote quote_;

        friend class QsaResourceManager;
    };

    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_) {}

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

    private:
        PublishedActivation(QsaResourceManager& owner, StartResult&& result, LaneId destination)
            : owner_(&owner), result_(std::move(result)), destination_(destination) {}

        QsaResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};

        friend class QsaResourceManager;
    };

    struct MaterializationOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
        std::optional<PublishedActivation> activation;
        MaterializationDiagnostics diagnostics;
        std::string failure;
    };

    struct ActiveCaptureOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    };

    using ContextTransactionOutcome =
        std::variant<ContextTransactionInProgress, MaterializationOutcome, ActiveCaptureOutcome>;

    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    enum class MaterializationReserveResult : std::uint8_t {
        Reserved,
        Stale,
        Aborted,
    };

    enum class ActiveCaptureReserveResult : std::uint8_t {
        Reserved,
        Skipped,
    };

    explicit QsaResourceManager(std::uint32_t lane_count) : lane_count_(lane_count) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency) {
            throw std::invalid_argument("QSA resource-manager lane count is invalid");
        }
    }

    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base, std::uint64_t publication_order,
                                     PlanningAllowance /*allowance*/ = {}) {
        if (open_lane_ || program.has_context_transaction()) {
            return {.readiness = Readiness::TemporarilyBlocked};
        }
        if (publication_order == 0) {
            throw std::invalid_argument("request publication order is zero");
        }
        if (!program.isolated_request_feasible(base)) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (lanes_[lane] == Lane::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }
        Quote quote = program.qsa_quote(prompt, base, *destination);
        if (quote.readiness != Readiness::Ready) { return {.readiness = quote.readiness}; }
        return {.readiness = Readiness::Ready, .choice = Choice(std::move(quote))};
    }

    // No persistent backfill proof: the streamed tier resumes from whole images, not from a
    // Device page a later admission could overtake.
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(Program&, const RequestBasePlan&, const Choice&,
                              std::span<const SequenceHandle>) const {
        return std::nullopt;
    }

    void prefetch_blocked_head(Program&, const PreparedPrompt&, const RequestBasePlan&,
                               std::uint64_t) {}

    [[nodiscard]] MaterializationReserveResult
    reserve_materialization(Program& program, Choice&& choice, PreparedPrompt&& prompt,
                            CancellationFlagView cancellation) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("QSA admission overlaps an open resource transaction");
        }
        const LaneId lane = choice.destination();
        require_lane(lane, Lane::Free);
        const ContextTransactionReserveStatus status = program.qsa_reserve_materialization(
            std::move(choice.quote_), std::move(prompt), cancellation);
        if (status == ContextTransactionReserveStatus::Aborted) {
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        lanes_[lane.value] = Lane::Materializing;
        open_lane_         = lane;
        return MaterializationReserveResult::Reserved;
    }

    [[nodiscard]] std::optional<ContextTransactionKind> context_transaction_kind() const noexcept {
        if (open_lane_) { return ContextTransactionKind::Materialization; }
        return std::nullopt;
    }

    // The reservation already started the sequence: publish it here instead of driving a Legacy
    // materialization progress step.
    [[nodiscard]] ContextTransactionOutcome
    progress_context_transaction(Program& program, CancellationFlagView /*cancellation*/) {
        if (!open_lane_) { throw std::logic_error("QSA manager has no progressable admission"); }
        std::optional<StartResult> published = program.take_qsa_published();
        if (!published) { throw std::logic_error("QSA admission published no sequence"); }
        MaterializationOutcome outcome;
        outcome.status = ContextTransactionStatus::Published;
        outcome.activation.emplace(
            PublishedActivation(*this, std::move(*published), *open_lane_));
        return outcome;
    }

    void adopt(Program& /*program*/, PublishedActivation&& activation) noexcept {
        if (activation.owner_ != this || !activation.result_ || !open_lane_ ||
            activation.destination_.value != open_lane_->value ||
            lanes_[activation.destination_.value] != Lane::Materializing) {
            std::terminate();
        }
        lanes_[activation.destination_.value] = Lane::Active;
        activation.result_.reset();
        activation.owner_ = nullptr;
        open_lane_.reset();
    }

    [[nodiscard]] ActiveCaptureReserveResult reserve_active_capture(Program& program, LaneId,
                                                                    CaptureOffer&& offer,
                                                                    std::uint32_t,
                                                                    CancellationFlagView) {
        program.skip_capture(std::move(offer));
        return ActiveCaptureReserveResult::Skipped;
    }

    // Device pages are bookkeeping under --kv-resident: nothing to reclaim from the store.
    std::uint32_t reclaim_device_kv_for_lease(Program&, std::uint32_t, std::uint32_t) {
        return 0;
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        require_lane(lane, Lane::TerminalPending);
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal finish overlaps an open resource transaction");
        }
        FinishResult result = program.finish(sequence);
        if (result.status == ConsumeStatus::Consumed) {
            lanes_[lane.value] = Lane::Free;
            return result;
        }
        AbortResult discarded = program.abort(sequence);
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program could neither publish nor discard a sequence");
        }
        lanes_[lane.value] = Lane::Free;
        FinishResult released;
        released.status      = ConsumeStatus::Consumed;
        released.disposition = FinishDisposition::Released;
        released.timings     = discarded.timings;
        released.speculative = std::move(discarded.speculative);
        return released;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal abort overlaps an open resource transaction");
        }
        if (lanes_.at(lane.value) != Lane::Active &&
            lanes_.at(lane.value) != Lane::TerminalPending) {
            throw std::logic_error("aborted lane has no active owner");
        }
        AbortResult result = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program did not consume aborted sequence");
        }
        lanes_[lane.value] = Lane::Free;
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes,
                      const typename ModelContract::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const LaneId lane = lanes[row];
            require_lane(lane, Lane::Active);
            switch (result.rows[row].disposition) {
            case CommitDisposition::Active:
                break;
            case CommitDisposition::Finishable:
                lanes_[lane.value] = Lane::TerminalPending;
                break;
            case CommitDisposition::CancelledReleased:
                lanes_[lane.value] = Lane::Free;
                break;
            }
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename ModelContract::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (const LaneId lane : lanes) { release_lane(lane); }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value < lane_count_ && (lanes_[lane.value] == Lane::Active ||
                                             lanes_[lane.value] == Lane::TerminalPending)) {
                lanes_[lane.value] = Lane::Free;
            }
        }
    }

    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.host_state_occupied_slots        = usage.host_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        out.device_main_kv_lease_pages       = usage.device_main_kv_lease_pages;
        out.device_backend_kv_lease_pages    = usage.device_backend_kv_lease_pages;
        out.host_kv_occupied_bytes           = usage.host_kv_bytes;
    }

    [[nodiscard]] LogicalLaneState lane_state(LaneId lane) const noexcept {
        return lane.value < lane_count_ ? lanes_[lane.value] : LogicalLaneState::Free;
    }

    void clear_after_program_cleanup() noexcept {
        open_lane_.reset();
        lanes_.fill(Lane::Free);
    }

private:
    using Lane = LogicalLaneState;

    void require_lane(LaneId lane, Lane expected) const {
        if (lane.value >= lane_count_ || lanes_[lane.value] != expected) {
            throw std::logic_error("QSA lane is not in its expected ownership state");
        }
    }

    void release_lane(LaneId lane) {
        if (lane.value >= lane_count_ ||
            (lanes_[lane.value] != Lane::Active && lanes_[lane.value] != Lane::TerminalPending)) {
            throw std::logic_error("released lane has no active owner");
        }
        lanes_[lane.value] = Lane::Free;
    }

    std::uint32_t lane_count_ = 0;
    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::optional<LaneId> open_lane_;
};

} // namespace ninfer::runtime
