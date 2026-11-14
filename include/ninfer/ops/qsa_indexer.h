#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/qsa_budget.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The block indexer of Qwen3.8-Flash-Next's sparse attention (transformers
 * Qwen4ExpTextQSAIndexer), for one sequence. A query at position p sees the complete 4-token
 * blocks b with 4b + 3 <= p; each block has a pooled key
 *
 *   K_b = rope(groupnorm(mean_{i<4} k[4b + i], key_norm), 4b)            FP32 [128]
 *
 * from the raw indexer keys k (projection rows 512..639), and the query selects
 *
 *   score_b = sum_{h<4} relu(q_h . K_b),   q_h = rope(groupnorm(q[h], query_norm), p)
 *
 * the min(top_blocks, blocks) best blocks, ties to the lower block index, where `top_blocks`
 * defaults to the artifact's 512 (kQsaTopBlocksDefault) and may be raised to kQsaTopBlocksMax.
 * groupnorm is the zero-centred
 * RMSNorm x / sqrt(mean x^2 + eps) * (1 + w) over 128 values; rope rotates the first 64 dims
 * rotate-half at angle fl32(fl32(position) * f_i), f_i = fl32(1e7^(-2i/64)), i < 32, and passes
 * dims 64..127 through. Whenever `yarn.active()` (a YaRN factor above one OR linear position
 * interpolation), f_i is the pair's YaRN inverse frequency, rope_yarn_frequencies(64, 32, 1e7,
 * yarn), and cos and sin both carry its attention factor. Linear position interpolation is
 * applied to the position of every axis before the angle, exactly as the dense YaRN kernel
 * (rope.cuh's rope_yarn_kernel) does. This keeps the indexer's rotation on the same schedule
 * as the model's dense attention, the MTP block and Flash-Next's QSA.
 * The indexer's rotation carries YaRN's attention factor as the main attention's does: the
 * reference indexer (Hugging Face's `Qwen4ExpTextQSAIndexer`) rotates with the layer's shared
 * `position_embeddings`, whose cos and sin carry `attention_scaling`.
 * With QsaIndexerPositions (interleaved 3-axis M-RoPE), the angle of pair i uses axis
 * `pair_axes[i]` (the engine's native route is i % 3, matching RopeConfig::pair_axes for
 * sections [11,11,10]). Three equal axes reduce exactly to the scalar rotation above, so
 * YaRN, interpolation and a text-only sequence all keep their original results.
 * The query's own incomplete block (positions 4 floor((p+1)/4) .. p) is attended by every query
 * and is not part of the selection.
 *
 * `projection` is BF16 [640, tokens] (4 query heads of 128, then the key) for the consecutive
 * positions `first_position` .. `first_position + tokens - 1`; `first_position` is a device I32
 * word, so a call captured in a graph replays at whatever position the word holds. `pooled` is
 * FP32 [128, capacity], the sequence's pooled keys by block, sized by the caller for every
 * position it will reach (blocks at or past `capacity` are neither pooled nor selected); `tail`
 * FP32 [128, 3] the raw keys of the incomplete block before `first_position` (slot = position
 * mod 4). The oracle evaluates in FP64 from the
 * represented inputs (angles as above); pooled keys are compared as FP32, and a selection must
 * equal the oracle's up to blocks whose FP64 scores tie the top_blocks-th within 1e-5 relative.
 */
struct QsaIndexerWeights {
    const Tensor* query_norm = nullptr; // BF16 [128]
    const Tensor* key_norm   = nullptr; // BF16 [128]
    // QSA-27B runs a zero-centred RMSNorm on each raw indexer key before the
    // 4-token pooling (the fork's blk.N.indexer.k_norm.pre_weight). Flash-Next
    // has no such norm, so nullptr selects its existing path unchanged.
    const Tensor* key_pre_norm = nullptr;
};

/**
 * Optional interleaved 3-axis M-RoPE positions for an image/video prompt.
 *
 * Qwen3.5/3.8 rotates the indexer with the same interleaved MRoPE as the dense
 * attention: rotary pair i uses axis `pair_axes[i]` (the engine's native route
 * is i % 3, matching `RopeConfig::pair_axes` for sections [11,11,10]). Three
 * equal axes reduce exactly to the scalar rotation below, so a text-only call
 * may leave `rope_positions` null and takes the original path unchanged.
 *
 * A completed block is rotated at the 3-axis position of its FIRST token
 * (4 * block), which can lie in the previous call's incomplete-block tail. The
 * persistent `tail_axes` carries those positions across calls; it is written
 * by append and read by the next append before that append overwrites it.
 */
struct QsaIndexerPositions {
    // I32 [tokens, 3] axis-major (axis * tokens + t), the same layout as
    // ops::rope's Text M-RoPE positions. Null: all three axes equal the scalar
    // cache position.
    const Tensor* rope_positions = nullptr;
    // Persistent I32 [3, 3] (slot, axis) axis positions of the incomplete
    // block's three tail slots, axis * 3 + slot. Null on the scalar path.
    Tensor* tail_axes = nullptr;
    // Optional host array of kRotaryPairs=32 entries, each 0..2. Null selects
    // the engine's interleaved mapping i % 3.
    const std::uint8_t* pair_axes = nullptr;
};

// Pools every block that completes in [first_position, first_position + tokens) into `pooled`
// and leaves the incomplete block's raw keys in `tail`. The scalar-position overload keeps the
// original symbol for callers that have no M-RoPE positions (Flash-Next, text-only QSA-27B).
void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        Tensor& pooled, Tensor& tail, cudaStream_t stream,
                        const QsaIndexerPositions& positions);
void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        Tensor& pooled, Tensor& tail, cudaStream_t stream);

[[nodiscard]] std::size_t qsa_indexer_select_workspace_bytes(
    std::int32_t tokens, std::int32_t capacity,
    std::int32_t top_blocks = kQsaTopBlocksDefault);

// Writes each query's selected blocks in increasing order into `selected` I32 [top_blocks, tokens]
// and their number into `counts` I32 [tokens]. Run after qsa_indexer_append for the same tokens.
// `top_blocks` (1..kQsaTopBlocksMax) is fixed for the process so a captured graph replays it.
void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        const Tensor& pooled, WorkspaceArena& workspace, Tensor& selected,
                        Tensor& counts, cudaStream_t stream,
                        const QsaIndexerPositions& positions,
                        std::int32_t top_blocks = kQsaTopBlocksDefault);
void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        const Tensor& pooled, WorkspaceArena& workspace, Tensor& selected,
                        Tensor& counts, cudaStream_t stream,
                        std::int32_t top_blocks = kQsaTopBlocksDefault);

} // namespace ninfer::ops
