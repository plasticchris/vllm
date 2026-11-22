// SPDX-License-Identifier: Apache-2.0
// Exact small BF16 TP4 peer-write reduction. Dedicated IPC arenas, gfx1100.
// Build with tools/asgard/build_gfx1100_peer_push.sh (no torch C++ ABI dependency).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#ifndef VLLM_PP_SOURCE_SHA256
#error "Build via tools/asgard/build_gfx1100_peer_push.sh"
#endif

namespace {
constexpr int kRanks = 4;
constexpr int kBlocks = 32;
constexpr int kThreads = 128;
constexpr int kHidden = 2304;
constexpr int kWords = 32 * kHidden / 8;
constexpr int kTimeoutMs = 250;
static_assert(sizeof(uint4) == 16);

struct alignas(64) ShapeArena {
  uint32_t epoch[kBlocks];
  uint32_t ready[2][kBlocks][kRanks];
  alignas(64) uint4 words[2][kRanks][kWords];
};
struct alignas(64) Arena {
  // One sticky flag per source GPU, shared across all shapes. No reset/retry.
  uint32_t failed[kRanks];
  ShapeArena shapes[3];
};
struct PeerPlan {
  Arena* peers[kRanks];
  int rank;
  uint64_t wait_ticks;
};
struct Bf16x8 { __hip_bfloat16 val[8]; };
static_assert(sizeof(Bf16x8) == sizeof(uint4));

__device__ __forceinline__ uint4 ordered_sum(const uint4 (&vectors)[kRanks]) {
  Bf16x8 data[kRanks];
  #pragma unroll
  for (int r = 0; r < kRanks; ++r)
    data[r] = __builtin_bit_cast(Bf16x8, vectors[r]);
  Bf16x8 result;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    float sum = __bfloat162float(data[0].val[j]);
    #pragma unroll
    for (int r = 1; r < kRanks; ++r) sum += __bfloat162float(data[r].val[j]);
    result.val[j] = __float2bfloat16(sum);
  }
  return __builtin_bit_cast(uint4, result);
}

__device__ __forceinline__ uint32_t load_system(uint32_t* p) {
  return __scoped_atomic_load_n(p, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t any_failure(Arena* a) {
  uint32_t status = 0;
  #pragma unroll
  for (int i = 0; i < kRanks; ++i) status |= load_system(&a->failed[i]);
  return status;
}

template <int Rows, int Shape>
__global__ void gfx1100_peer_push(PeerPlan p, const uint4* __restrict__ in,
                                             uint4* __restrict__ out) {
  constexpr int kCount = Rows * kHidden / 8;
  constexpr int kGrid = (kCount + kThreads - 1) / kThreads < kBlocks
                            ? (kCount + kThreads - 1) / kThreads : kBlocks;
  constexpr int kTile = (kCount + kGrid - 1) / kGrid;
  Arena* local = p.peers[p.rank];
  auto* mine = &local->shapes[Shape];
  const int b = blockIdx.x, t = threadIdx.x;
  const int lo = b * kTile, hi = min(lo + kTile, kCount);
  __shared__ uint32_t epoch, failed;
  if (t == 0) {
    epoch = ++mine->epoch[b];
    failed = any_failure(local);
  }
  __syncthreads();
  const int slot = epoch & 1;
  if (!failed) {
    for (int j = lo + t; j < hi; j += blockDim.x) {
      const uint4 v = in[j];
      #pragma unroll
      for (int n = 1; n < kRanks; ++n)
        p.peers[(p.rank + n) & 3]->shapes[Shape].words[slot][p.rank][j] = v;
    }
    // Each writer drains its system writes before publishing the ready flag.
    __threadfence_system();
    __syncthreads();
    if (t < kRanks && t != p.rank) {
      __scoped_atomic_store_n(&p.peers[t]->shapes[Shape].ready[slot][b][p.rank],
                              epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      const uint64_t start = wall_clock64();
      uint32_t i = 0;
      while (__scoped_atomic_load_n(&mine->ready[slot][b][t], __ATOMIC_ACQUIRE,
                                    __MEMORY_SCOPE_SYSTEM) != epoch) {
        if (++i >= (1u << 24) || ((i & 127) == 0 &&
            (wall_clock64() - start >= p.wait_ticks || any_failure(local)))) {
          atomicExch(&failed, 1u);
          break;
        }
      }
    }
    __syncthreads();
  }
  if (failed) {
    // Broadcast a sticky fault even to a slow peer that hasn't entered yet.
    // The normal sampling D2H event gates host output on this fault page.
    if (t < kRanks) {
      constexpr uint32_t kCodeBase = Rows << 8;
      __scoped_atomic_store_n(&p.peers[t]->failed[p.rank], kCodeBase | uint32_t(b + 1),
                              __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      __threadfence_system();
    }
    // Defined placeholder; never expose as a successful model result.
    for (int j = lo + t; j < hi; j += blockDim.x) out[j] = uint4{};
    return;
  }
  // Single execution stream per GPU, separate state per shape. A sender
  // can't enter e+2 until it has seen this peer's e+1 flag; the peer's e+1
  // launch is ordered after its read of e. Thus two slots suffice on replay.
  for (int j = lo + t; j < hi; j += blockDim.x) {
    uint4 vectors[kRanks];
    #pragma unroll
    for (int r = 0; r < kRanks; ++r)
      vectors[r] = r == p.rank ? in[j] : mine->words[slot][r][j];
    out[j] = ordered_sum(vectors);
  }
}

template <int Rows, int Shape>
void launch(PeerPlan plan, const void* in, void* out, hipStream_t stream) {
  constexpr int words = Rows * kHidden / 8;
  constexpr int blocks = (words + kThreads - 1) / kThreads < kBlocks
                            ? (words + kThreads - 1) / kThreads : kBlocks;
  gfx1100_peer_push<Rows, Shape><<<blocks, kThreads, 0, stream>>>(
      plan, static_cast<const uint4*>(in), static_cast<uint4*>(out));
}
}  // namespace

extern "C" int vllm_pp_abi_version() { return 1; }
extern "C" const char* vllm_pp_source_sha256() { return VLLM_PP_SOURCE_SHA256; }
extern "C" size_t vllm_pp_handle_bytes() { return sizeof(hipIpcMemHandle_t); }

extern "C" int vllm_pp_probe(char* bdf, int bdf_len, int device) {
  if (device < 0 || device >= kRanks || bdf_len < 16) return hipErrorInvalidValue;
  int count = 0;
  auto err = hipGetDeviceCount(&count);
  if (err != hipSuccess) return err;
  if (count != kRanks) return hipErrorInvalidDevice;
  hipDeviceProp_t props;
  err = hipGetDeviceProperties(&props, device);
  if (err != hipSuccess) return err;
  if (std::strncmp(props.gcnArchName, "gfx1100", 7) ||
      (props.gcnArchName[7] != '\0' && props.gcnArchName[7] != ':'))
    return hipErrorInvalidDevice;
  for (int p = 0; p < kRanks; ++p) {
    if (p == device) continue;
    int accessible = 0;
    err = hipDeviceCanAccessPeer(&accessible, device, p);
    if (err != hipSuccess) return err;
    if (!accessible) return hipErrorPeerAccessUnsupported;
  }
  return hipDeviceGetPCIBusId(bdf, bdf_len, device);
}

extern "C" int vllm_pp_alloc(void** out, void* handle_bytes) {
  void* buf = nullptr;
  auto err = hipExtMallocWithFlags(&buf, sizeof(Arena), hipDeviceMallocUncached);
  if (err != hipSuccess) return err;
  hipIpcMemHandle_t handle;
  if ((err = hipMemset(buf, 0, sizeof(Arena))) == hipSuccess &&
      (err = hipDeviceSynchronize()) == hipSuccess)
    err = hipIpcGetMemHandle(&handle, buf);
  if (err != hipSuccess) { (void)hipFree(buf); return err; }
  std::memcpy(handle_bytes, &handle, sizeof(handle));
  *out = buf;
  return hipSuccess;
}

extern "C" int vllm_pp_free(void* local) { return hipFree(local); }

extern "C" int vllm_pp_connect(void** out, void* local, const void* all_handles, int rank) {
  if (!local || !all_handles || rank < 0 || rank >= kRanks) return hipErrorInvalidValue;
  auto* p = new (std::nothrow) PeerPlan{};
  if (!p) return hipErrorOutOfMemory;
  p->peers[rank] = static_cast<Arena*>(local);
  p->rank = rank;
  int device, khz = 0;
  auto err = hipGetDevice(&device);
  if (err == hipSuccess) err = hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device);
  if (err == hipSuccess && khz <= 0) err = hipErrorInvalidDevice;
  if (err == hipSuccess) {
    p->wait_ticks = uint64_t(khz) * kTimeoutMs;
    for (int r = 0; r < kRanks && err == hipSuccess; ++r) {
      if (r == rank) continue;
      hipIpcMemHandle_t h;
      std::memcpy(&h, static_cast<const char*>(all_handles) + r * sizeof(h), sizeof(h));
      void* ptr = nullptr;
      err = hipIpcOpenMemHandle(&ptr, h, hipIpcMemLazyEnablePeerAccess);
      if (err == hipSuccess) p->peers[r] = static_cast<Arena*>(ptr);
    }
  }
  if (err != hipSuccess) {
    for (int r = 0; r < kRanks; ++r)
      if (r != rank && p->peers[r]) (void)hipIpcCloseMemHandle(p->peers[r]);
    delete p;
    return err;
  }
  *out = p;
  return hipSuccess;
}

extern "C" int vllm_pp_disconnect(void* ctx) {
  auto* p = static_cast<PeerPlan*>(ctx);
  hipError_t err = hipSuccess;
  for (int r = 0; r < kRanks; ++r) {
    if (r == p->rank) continue;
    auto close = hipIpcCloseMemHandle(p->peers[r]);
    if (err == hipSuccess) err = close;
  }
  delete p;
  return err;
}

extern "C" int vllm_pp_reduce(void* ctx, const void* in, void* out, int rows,
                             uintptr_t stream_ptr) {
  if (!ctx || !in || !out || in == out || ((uintptr_t(in) | uintptr_t(out)) & 15))
    return hipErrorInvalidValue;
  auto p = *static_cast<PeerPlan*>(ctx);
  auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
  switch (rows) {
    case 1: launch<1, 0>(p, in, out, stream); break;
    case 8: launch<8, 1>(p, in, out, stream); break;
    case 32: launch<32, 2>(p, in, out, stream); break;
    default: return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

extern "C" int vllm_pp_snapshot(void* ctx, uint32_t* device_out, uintptr_t stream_ptr) {
  if (!ctx || !device_out) return hipErrorInvalidValue;
  auto* p = static_cast<PeerPlan*>(ctx);
  return hipMemcpyAsync(device_out, p->peers[p->rank]->failed, kRanks * sizeof(*device_out),
                        hipMemcpyDeviceToDevice, reinterpret_cast<hipStream_t>(stream_ptr));
}
