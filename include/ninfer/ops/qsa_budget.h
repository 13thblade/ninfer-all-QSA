#pragma once

#include <cstdint>

namespace ninfer::ops {

// The QSA block indexer's geometry, shared by qsa_indexer, sparse_attention and kv_stream.
//
// A block is four tokens; a query attends to the blocks its indexer selects plus its own
// incomplete block. The artifact's own budget is 512 blocks (2048 tokens), which is what every
// caller that does not pass a budget gets. A caller may raise it up to kQsaTopBlocksMax: the
// sparse attention's split count and partial-softmax workspace are sized for that cap at compile
// time, so a runtime budget stays graph-safe. A budget is fixed for the life of the process.
// The cap is 8192 blocks (32,768 tokens); every block of it costs a split of the partial-softmax
// workspace, which is ~51 MiB at 24 query heads and only allocated for calls of 8 tokens or fewer.
inline constexpr std::int32_t kQsaBlockTokens      = 4;
inline constexpr std::int32_t kQsaTopBlocksDefault = 512;
inline constexpr std::int32_t kQsaTopBlocksMax     = 8192;

} // namespace ninfer::ops
