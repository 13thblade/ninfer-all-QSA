#pragma once

#include <cstdint>

namespace ninfer::ops {

// The QSA block indexer's geometry, shared by qsa_indexer, sparse_attention and kv_stream.
//
// A block is four tokens; a query attends to the blocks its indexer selects plus its own
// incomplete block. The artifact's own budget is 512 blocks (2048 tokens), which is what every
// caller that does not pass a budget gets. A caller may raise it up to kQsaTopBlocksMax, and the
// sparse attention's split count and partial-softmax workspace follow that call's budget at run
// time, so a wide cap costs a narrow budget nothing. A budget is fixed for the life of the
// process, which keeps a captured graph's grid valid on replay.
inline constexpr std::int32_t kQsaBlockTokens      = 4;
inline constexpr std::int32_t kQsaTopBlocksDefault = 512;
inline constexpr std::int32_t kQsaTopBlocksMax     = 65536;

} // namespace ninfer::ops
