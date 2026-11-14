#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/structured_round.h"
#include "ninfer/ops/kv_stream.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    // The structured lanes' masks, computed while the verification ran.
    if (execution.io.structured != nullptr) { execution.io.structured->wait(execution.device.stream); }
    // Before acceptance adds this round's tokens to the penalty counts.
    gather_logprobs(execution, frame.target_logits, frame.sampling, &frame.drafts);
    if (frame.qsa_round_flags != nullptr) {
        // --kv-resident: an overflowed lane's draft columns lost their KV blocks. Withhold the
        // drafts *before* acceptance so the kernel licenses column 0's target token with its own
        // lengths/anchors/selected-hidden bookkeeping, in phase with the device-side state. A
        // host-side patch after device.synchronize() would be too late: licensed_tokens[0] of an
        // accepted draft is a draft token, and select_accepted_hidden/scatter already ran.
        ops::kv_stream_degrade_verify_extents(
            frame.qsa_round_flags, frame.qsa_row_lanes,
            static_cast<std::int32_t*>(frame.current_extents.data),
            static_cast<std::int32_t>(frame.current_extents.ne[0]), execution.device.stream);
    }
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_coupled_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::models::qwen3_5::execution
