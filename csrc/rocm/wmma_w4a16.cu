// Fused int4-dequant WMMA GEMM for gfx1100 (RX 7900 XTX), batched decode M=8..64.
// C[M,N] = A[M,K] @ dequant(W).T. W is ExLlama-shuffle-packed int4 stored as
// int32 [N, K/8]; per-group fp16/bf16 scale + raw zero-point [N, K/group_size].
// Dequant: (nibble - zp) * scale (asymmetric); symmetric callers pass zp of 8.
//
// Design (from the gfx1100 W4A16 WMMA reference in ~/gfx1100-docs, FINDINGS.md):
//   * Raw __builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32 (v16 in, v8fp32 acc).
//   * Weights (B) dequanted cooperatively into a double-buffered LDS tile,
//     shared by all waves in the block (dequant once, reuse across M-waves).
//   * Activations (A) loaded DIRECT into the fragment via a 128-bit vectorized
//     copy (2x global_load_b128) - no LDS staging for A.
//   * fp16 magic-number dequant (0x6400 exllamav2 trick); our ExLlama shuffle
//     already yields consecutive-K half2 pairs, matching the trick.
//   * WM waves cover M (wave w -> rows [16w..16w+15]); each wave register-blocks
//     NB adjacent 16-col N-tiles (NB independent accumulators => WMMA pipelining).
//   * Split-K over grid.z -> fp32 atomic accumulate, then a convert pass.
//
// RDNA3 wave32 layout (probed + confirmed by ISA/GPUOpen):
//   A frag: lane l -> row l%16, full 16 K in slots; input replicated across
//           half-waves (lanes 0-15 == 16-31). B frag: lane l -> col l%16, 16 K.
//   C acc:  lane l -> col l%16; acc[i] -> row (2*i + l/16).
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace {

using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

template <typename T> struct Nat;
template <> struct Nat<half> { using E = _Float16; using V = v16fp16; };
template <> struct Nat<__hip_bfloat16> { using E = __bf16; using V = v16bf16; };

template <typename E, typename T> __device__ __forceinline__ E bc(T x) {
  E r; __builtin_memcpy(&r, &x, sizeof(E)); return r;
}
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__hip_bfloat16 v) { return __bfloat162float(v); }

// fp16 magic-number dequant: our zp is a raw fp16 (integer-valued) zero-point.
//   low pairs  q=nibble+1024 : dq = q*scale + scale*(-1024 - zp)
//   high pairs q=nibble*16+1024, y=scale/16 : dq = q*(scale/16) + scale*(-64 - zp)
__device__ __forceinline__ void prep_fp16(float sc, float zp, half2& z, half2& y) {
  z = __float2half2_rn(1024.0f + zp);  // exact in fp16 for integer zp in [0,15]
  y = __float2half2_rn(sc);
}
__device__ __forceinline__ void dequant8_fp16(uint32_t qa, half2 (&dq)[4],
                                              half2 z, half2 y) {
  const uint32_t c0 = 0x64006400;  // half2(1024,1024)
  union { uint32_t u; half2 h; } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;         // (1024+k0, 1024+k1)
  q1.u = ((qa >> 4) & 0x000F000F) | c0;  // (1024+k2, 1024+k3)
  q2.u = ((qa >> 8) & 0x000F000F) | c0;  // (1024+k4, 1024+k5)
  q3.u = ((qa >> 12) & 0x000F000F) | c0; // (1024+k6, 1024+k7)
  dq[0] = __hmul2(__hsub2(q0.h, z), y);
  dq[1] = __hmul2(__hsub2(q1.h, z), y);
  dq[2] = __hmul2(__hsub2(q2.h, z), y);
  dq[3] = __hmul2(__hsub2(q3.h, z), y);
}
__device__ __forceinline__ void prep_bf16(float sc, float zp, __hip_bfloat162& z, __hip_bfloat162& y) {
  z = __bfloat162bfloat162(__float2bfloat16(128.0f + zp));  // exact in bf16 for int zp
  y = __bfloat162bfloat162(__float2bfloat16(sc));
}
__device__ __forceinline__ void dequant8_bf16(uint32_t qa, __hip_bfloat162 (&dq)[4],
                                              __hip_bfloat162 z, __hip_bfloat162 y) {
  const uint32_t c0 = 0x43004300;  // bf162(128,128)
  union { uint32_t u; __hip_bfloat162 b; } q0, q1, q2, q3;
  q0.u = ((qa >> 0) & 0x000F000F) | c0;
  q1.u = ((qa >> 4) & 0x000F000F) | c0;
  q2.u = ((qa >> 8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;
  dq[0] = __hmul2(__hsub2(q0.b, z), y);
  dq[1] = __hmul2(__hsub2(q1.b, z), y);
  dq[2] = __hmul2(__hsub2(q2.b, z), y);
  dq[3] = __hmul2(__hsub2(q3.b, z), y);
}

template <typename T, int WM, int WN, int NB>
__global__ void __launch_bounds__(WM * WN * 32)
    wmma_w4a16_kernel(const int* __restrict__ Wpk, const T* __restrict__ A,
                      const T* __restrict__ scale, const T* __restrict__ zp,
                      float* __restrict__ Cf32, int M, int N, int K, int gs, int nksteps) {
  using E = typename Nat<T>::E;
  using V = typename Nat<T>::V;
  constexpr int BN = WN * NB * 16;
  constexpr int NTHREADS = WM * WN * 32;
  const int tileM = blockIdx.y * (WM * 16);
  const int tileN = blockIdx.x * BN;
  const int kstart = blockIdx.z * nksteps * 64;
  int kend = kstart + nksteps * 64;
  if (kend > K) kend = K;
  if (kstart >= kend) return;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int wm = wave / WN, wn = wave % WN;
  const int lane_lo = lane & 15, lane_hi = lane >> 4;
  const int Kp = K / 8, ng = K / gs;

  const int gm = tileM + wm * 16 + lane_lo;  // this lane's A row
  const T* Arow = A + (size_t)gm * K;
  const bool m_ok = gm < M;

  __shared__ E b_lds[2][64][BN];  // 64-K stage per round (two 32-K groups)
  constexpr int MAXC = (BN + NTHREADS - 1) / NTHREADS;  // columns per thread

  // Load phase: each thread streams 64 K (two uint4 = 8 int32) of ONE column,
  // contiguous -> full-cacheline coalesced global loads.
  auto load_regs = [&](int kt, uint4 (&pk)[MAXC][2]) {
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
      const int col = tid + c * NTHREADS;
      const int gn = tileN + col;
      if (col < BN && gn < N) {
        const uint4* p = (const uint4*)(Wpk + gn * Kp + (kt >> 3));
        pk[c][0] = p[0];
        pk[c][1] = p[1];
      }
    }
  };
  // Store phase: dequant the register-held words into LDS. Two 32-K halves,
  // each its own group (handles group_size=32 within a 64-K stage).
  auto dq_store = [&](int buf, int kt, const uint4 (&pk)[MAXC][2]) {
#pragma unroll
    for (int c = 0; c < MAXC; c++) {
      const int col = tid + c * NTHREADS;
      const int gn = tileN + col;
      if (col >= BN || gn >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint32_t w[4] = {pk[c][h].x, pk[c][h].y, pk[c][h].z, pk[c][h].w};
        const int g = (kt + h * 32) / gs;
        const float sc = to_f(scale[gn * ng + g]);
        const float z = zp ? to_f(zp[gn * ng + g]) : 8.0f;
        if constexpr (std::is_same_v<T, half>) {
          half2 zz, yy; prep_fp16(sc, z, zz, yy);
#pragma unroll
          for (int m = 0; m < 4; m++) {
            half2 d[4]; dequant8_fp16(w[m], d, zz, yy);
            E* cl = &b_lds[buf][h * 32 + m * 8][col];
            cl[0 * BN] = bc<E>(__low2half(d[0]));  cl[1 * BN] = bc<E>(__high2half(d[0]));
            cl[2 * BN] = bc<E>(__low2half(d[1]));  cl[3 * BN] = bc<E>(__high2half(d[1]));
            cl[4 * BN] = bc<E>(__low2half(d[2]));  cl[5 * BN] = bc<E>(__high2half(d[2]));
            cl[6 * BN] = bc<E>(__low2half(d[3]));  cl[7 * BN] = bc<E>(__high2half(d[3]));
          }
        } else {
          __hip_bfloat162 z2, y2; prep_bf16(sc, z, z2, y2);
#pragma unroll
          for (int m = 0; m < 4; m++) {
            __hip_bfloat162 d[4]; dequant8_bf16(w[m], d, z2, y2);
            E* cl = &b_lds[buf][h * 32 + m * 8][col];
            cl[0 * BN] = bc<E>(d[0].x); cl[1 * BN] = bc<E>(d[0].y);
            cl[2 * BN] = bc<E>(d[1].x); cl[3 * BN] = bc<E>(d[1].y);
            cl[4 * BN] = bc<E>(d[2].x); cl[5 * BN] = bc<E>(d[2].y);
            cl[6 * BN] = bc<E>(d[3].x); cl[7 * BN] = bc<E>(d[3].y);
          }
        }
      }
    }
  };

  v8fp32 acc[NB];
#pragma unroll
  for (int j = 0; j < NB; j++)
    for (int i = 0; i < 8; i++) acc[j][i] = 0.0f;

  uint4 pk0[MAXC][2];
  load_regs(kstart, pk0); dq_store(0, kstart, pk0);
  __syncthreads();
  int buf = 0;
  for (int kt = kstart; kt < kend; kt += 64) {
    const int nb = buf ^ 1;
    const bool more = kt + 64 < kend;
    uint4 pk[MAXC][2];
    if (more) load_regs(kt + 64, pk);  // issue next-stage loads
#pragma unroll
    for (int ks = 0; ks < 64; ks += 16) {
      V a_frag;
      if (m_ok) __builtin_memcpy(&a_frag, Arow + kt + ks, sizeof(a_frag));
      else
#pragma unroll
        for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
#pragma unroll
      for (int j = 0; j < NB; j++) {
        V b_frag;
#pragma unroll
        for (int i = 0; i < 16; i++) b_frag[i] = b_lds[buf][ks + i][(wn * NB + j) * 16 + lane_lo];
        acc[j] = wmma_mma(a_frag, b_frag, acc[j]);
      }
    }
    if (more) dq_store(nb, kt + 64, pk);  // dequant after WMMA
    __syncthreads();
    buf = nb;
  }

#pragma unroll
  for (int j = 0; j < NB; j++) {
    const int gn = tileN + (wn * NB + j) * 16 + lane_lo;
    if (gn >= N) continue;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int gmo = tileM + wm * 16 + 2 * i + lane_hi;
      if (gmo < M) atomicAdd(&Cf32[(size_t)gmo * N + gn], acc[j][i]);
    }
  }
}

template <typename T>
__global__ void wmma_w4a16_convert(const float* __restrict__ Cf32,
                                   const T* __restrict__ bias, T* __restrict__ C,
                                   int MN, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float v = Cf32[i];
  if (bias) v += to_f(bias[i % N]);
  C[i] = (T)v;
}

}  // namespace

torch::Tensor wvSplitK_int4_wmma(const at::Tensor& weight, const at::Tensor& activation,
                                 const at::Tensor& scale,
                                 const std::optional<at::Tensor>& zero_points,
                                 const std::optional<at::Tensor>& bias,
                                 const int64_t group_size) {
  const int64_t M = activation.size(0);
  const int64_t K = activation.size(1);
  const int64_t N = weight.size(0);
  TORCH_CHECK(weight.scalar_type() == at::kInt, "weight must be int32 [N, K/8]");
  TORCH_CHECK(weight.size(1) == K / 8, "weight second dim must be K/8");
  TORCH_CHECK(activation.dtype() == torch::kFloat16 || activation.dtype() == torch::kBFloat16,
              "activation must be float16 or bfloat16");
  TORCH_CHECK(scale.dtype() == activation.dtype(), "scale dtype must match activation");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "group_size must be 32/64/128");
  TORCH_CHECK(K % group_size == 0 && K % 16 == 0, "K must be divisible by group_size and 16");
  const int64_t ng = K / group_size;
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == N && scale.size(1) == ng,
              "scale must be [N, K/group_size]");
  if (zero_points.has_value())
    TORCH_CHECK(zero_points->dim() == 2 && zero_points->size(0) == N &&
                    zero_points->size(1) == ng,
                "zero_points must be [N, K/group_size]");

  auto out = torch::empty({M, N},
                          torch::TensorOptions().dtype(activation.dtype()).device(activation.device()));
  auto acc32 = torch::zeros({M, N},
                            torch::TensorOptions().dtype(torch::kFloat32).device(activation.device()));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(activation));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int* wp = weight.data_ptr<int>();
  const bool has_zp = zero_points.has_value();
  const bool has_bias = bias.has_value() && bias->numel() > 0;
  float* cf32 = acc32.data_ptr<float>();

  // 2D wave grid: WM waves cover M-rows, WN waves cover N (keeps ~128 threads
  // per block loading weights even when M is small). NB N-tiles per wave.
  int WM, WN, NBdef;
  if (M <= 16) { WM = 1; WN = 4; NBdef = 2; }
  else if (M <= 32) { WM = 2; WN = 2; NBdef = 2; }
  else { WM = 4; WN = 1; NBdef = 4; }
  const char* nbe = getenv("VLLM_WMMA_NB");
  const int NB = nbe ? atoi(nbe) : NBdef;
  const int BN = WN * NB * 16;
  const int nkt_total = (K + 63) / 64;  // number of 64-K stages
  const int gx = (N + BN - 1) / BN, gy = (M + WM * 16 - 1) / (WM * 16);
  static int sk_target = []() { const char* e = getenv("VLLM_WMMA_SKTARGET"); return e ? atoi(e) : 320; }();
  int SK = sk_target / (gx * gy);
  if (SK < 1) SK = 1;
  if (SK > nkt_total) SK = nkt_total;
  const int nksteps = (nkt_total + SK - 1) / SK;
  SK = (nkt_total + nksteps - 1) / nksteps;
  dim3 grid(gx, gy, SK);

#define LAUNCH(T, _WM, _WN, _NB)                                               \
  do {                                                                         \
    wmma_w4a16_kernel<T, _WM, _WN, _NB><<<grid, _WM * _WN * 32, 0, stream>>>(   \
        wp, (const T*)activation.data_ptr(), (const T*)scale.data_ptr(),      \
        has_zp ? (const T*)zero_points->data_ptr() : nullptr, cf32, M, N, K,  \
        group_size, nksteps);                                          \
    wmma_w4a16_convert<T><<<(M * N + 255) / 256, 256, 0, stream>>>(           \
        cf32, has_bias ? (const T*)bias->data_ptr() : nullptr,               \
        (T*)out.data_ptr(), M* N, N);                                         \
  } while (0)
#define DISP_NB(T, _WM, _WN)                                                   \
  do {                                                                        \
    if (NB == 1) LAUNCH(T, _WM, _WN, 1);                                       \
    else if (NB == 2) LAUNCH(T, _WM, _WN, 2);                                  \
    else LAUNCH(T, _WM, _WN, 4);                                               \
  } while (0)
#define DISP_WMWN(T)                                                          \
  do {                                                                        \
    if (WM == 1) DISP_NB(T, 1, 4);                                             \
    else if (WM == 2) DISP_NB(T, 2, 2);                                        \
    else DISP_NB(T, 4, 1);                                                     \
  } while (0)

  if (activation.dtype() == torch::kFloat16) {
    DISP_WMWN(half);
  } else {
    DISP_WMWN(__hip_bfloat16);
  }
#undef LAUNCH
#undef DISP_NB
#undef DISP_WMWN
  return out;
}
