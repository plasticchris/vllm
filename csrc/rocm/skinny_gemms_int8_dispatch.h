// INT8 token-count dispatch is sharded to parallelize expensive HIP codegen.
// Keep the public entry in shard 0; other shards export host helpers only.
#pragma once
#ifndef WV_INT8_DISPATCH_SHARD
#define WV_INT8_DISPATCH_SHARD 0
#endif

#define WV_INT8_DECLARE(name) \
  torch::Tensor name(const at::Tensor&, const at::Tensor&, const at::Tensor&, \
                    const std::optional<at::Tensor>&, int64_t, int64_t);
WV_INT8_DECLARE(wvSplitK_int8_g_3_4)
WV_INT8_DECLARE(wvSplitK_int8_g_5_6)
WV_INT8_DECLARE(wvSplitK_int8_g_7_8)
WV_INT8_DECLARE(wvSplitK_int8_g_9_10)
WV_INT8_DECLARE(wvSplitK_int8_g_11_12)
#undef WV_INT8_DECLARE

#define WV_INT8_CASE(n) case n: WVSPLIT_INT8G_TILE(sYT, n) break;
#if WV_INT8_DISPATCH_SHARD == 0
#define WV_INT8_ENTRY wvSplitK_int8_g
#define WV_INT8_CASES WV_INT8_CASE(1) WV_INT8_CASE(2)
#elif WV_INT8_DISPATCH_SHARD == 1
#define WV_INT8_ENTRY wvSplitK_int8_g_3_4
#define WV_INT8_CASES WV_INT8_CASE(3) WV_INT8_CASE(4)
#elif WV_INT8_DISPATCH_SHARD == 2
#define WV_INT8_ENTRY wvSplitK_int8_g_5_6
#define WV_INT8_CASES WV_INT8_CASE(5) WV_INT8_CASE(6)
#elif WV_INT8_DISPATCH_SHARD == 3
#define WV_INT8_ENTRY wvSplitK_int8_g_7_8
#define WV_INT8_CASES WV_INT8_CASE(7) WV_INT8_CASE(8)
#elif WV_INT8_DISPATCH_SHARD == 4
#define WV_INT8_ENTRY wvSplitK_int8_g_9_10
#define WV_INT8_CASES WV_INT8_CASE(9) WV_INT8_CASE(10)
#elif WV_INT8_DISPATCH_SHARD == 5
#define WV_INT8_ENTRY wvSplitK_int8_g_11_12
#define WV_INT8_CASES WV_INT8_CASE(11) WV_INT8_CASE(12)
#else
#error Invalid INT8 dispatch shard
#endif
