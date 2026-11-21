"""Packed NVFP4/MXFP8 expert GEMMs, BF16 activations (Hopper-like contract).

Weights stay in their checkpoint encoding. Dequantization is tile-local;
there is no model-sized BF16 copy and no cross-token atomic accumulation.
"""

import os
import torch
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from .tuning import OPTIMIZED, expert_launch_config

INTEGER_FP4 = os.environ.get("INKLING_INTEGER_FP4", "0") == "1"


@triton.jit
def _e2m1(x):
    mag = x & 7
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((x & 8) != 0, -value, value)


@triton.jit
def _e4m3(bits):
    exp = (bits >> 3) & 15
    mant = bits & 7
    value = tl.where(
        exp == 0,
        mant.to(tl.float32) * (2.0**-9),
        (1.0 + mant.to(tl.float32) / 8.0) * tl.exp2(exp.to(tl.float32) - 7.0),
    )
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _e2m1_bits(x):
    bits = x.to(tl.uint32)
    exponent = (bits >> 1) & 3
    mantissa = bits & 1
    normal = (((exponent + 126) << 23) | (mantissa << 22) | ((bits & 8) << 28)).to(
        tl.float32, bitcast=True
    )
    sub = mantissa.to(tl.float32) * 0.5
    sub = tl.where((bits & 8) != 0, -sub, sub)
    return tl.where(exponent == 0, sub, normal)


@triton.jit
def _e4m3_bits(x):
    bits = x.to(tl.uint32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20) | ((bits & 128) << 24)).to(
        tl.float32, bitcast=True
    )
    sub = mantissa.to(tl.float32) * (2.0**-9)
    sub = tl.where((bits & 128) != 0, -sub, sub)
    value = tl.where(exponent == 0, sub, normal)
    return tl.where((bits & 127) == 127, float("nan"), value)


@triton.jit
def _e8m0_bits(x):
    bits = x.to(tl.uint32)
    encoded = tl.where(bits == 0, 0x00400000, bits << 23)
    encoded = tl.where(bits == 255, 0x7FC00000, encoded)
    return encoded.to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_bf16_bits(x):
    bits = x.to(tl.uint32)
    magnitude = bits & 7
    encoded = (magnitude + 252) << 6
    encoded = tl.where(magnitude == 1, 0x3F00, encoded)
    encoded = tl.where(magnitude == 0, 0, encoded)
    encoded = encoded | ((bits & 8) << 12)
    return encoded.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _expert_gemm(
    A,
    W,
    S,
    Global,
    Sorted,
    Experts,
    Padded,
    Out,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    INPUT_DIV: tl.constexpr,
    NVFP4: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FAST_DECODE: tl.constexpr = False,
    INTEGER_DECODE: tl.constexpr = False,
):
    tile = tl.program_id(0)
    if tile * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + tile)
    if expert < 0:
        return
    rows = tl.load(Sorted + tile * BM + tl.arange(0, BM))
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    GROUP: tl.constexpr = 16 if NVFP4 else 32
    PACK: tl.constexpr = 2 if NVFP4 else 1
    glob = tl.load(Global + expert) if NVFP4 else 1.0
    for first in range(tl.cdiv(K, BK)):
        k = first * BK + ks
        av = tl.load(
            A + (rows // INPUT_DIV)[:, None] * K + k[None, :],
            (rows[:, None] < ROWS) & (k[None, :] < K),
            other=0,
        )
        raw = tl.load(
            W.to(tl.pointer_type(tl.uint8))
            + (expert * N + cols[None, :]) * (K // PACK)
            + k[:, None] // PACK,
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        scales = tl.load(
            S.to(tl.pointer_type(tl.uint8))
            + (expert * N + cols[None, :]) * (K // GROUP)
            + k[:, None] // GROUP,
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        if NVFP4:
            code = (raw >> ((k[:, None] % 2) * 4)) & 15
            if INTEGER_DECODE:
                vals = _e2m1_bf16_bits(code)
            else:
                vals = _e2m1_bits(code) if FAST_DECODE else _e2m1(code)
            scale = (_e4m3_bits(scales) if FAST_DECODE else _e4m3(scales)) * glob
        else:
            vals = _e4m3_bits(raw) if FAST_DECODE else _e4m3(raw)
            scale = (
                _e8m0_bits(scales)
                if FAST_DECODE
                else tl.exp2(scales.to(tl.float32) - 127.0)
            )
        bv = (vals * scale).to(av.dtype)
        acc = tl.dot(av, bv, acc)
    tl.store(
        Out + rows[:, None] * N + cols[None, :],
        acc.to(Out.dtype.element_ty),
        (rows[:, None] < ROWS) & (cols[None, :] < N),
    )


def expert_gemm(
    x,
    weight,
    scale,
    global_scale,
    ids,
    input_div,
    nvfp4,
    layout=None,
    launch_config=None,
    fast_decode=False,
):
    if launch_config is None and OPTIMIZED:
        k = weight.shape[2] * (2 if nvfp4 else 1)
        launch_config, fast_decode = expert_launch_config(ids.shape[0], k, nvfp4)
    bm, bn, bk = (16, 32, 64) if launch_config is None else launch_config
    assert bm in (16, 32, 64) and bn in (32, 64) and bk in (64, 128)
    rows = ids.numel()
    if layout is None:
        layout = moe_align_block_size(
            ids, bm, weight.shape[0], ignore_invalid_experts=True
        )
    sorted_ids, experts, padded = layout
    n = weight.shape[1]
    k = weight.shape[2] * (2 if nvfp4 else 1)
    out = torch.zeros((rows, n), dtype=x.dtype, device=x.device)
    if rows:
        _expert_gemm[(triton.cdiv(sorted_ids.numel(), bm), triton.cdiv(n, bn))](
            x,
            weight,
            scale,
            global_scale,
            sorted_ids,
            experts,
            padded,
            out,
            rows,
            n,
            k,
            input_div,
            nvfp4,
            bm,
            bn,
            bk,
            FAST_DECODE=fast_decode,
            INTEGER_DECODE=INTEGER_FP4,
            num_warps=4,
            num_stages=1,
        )
    return out, layout


@triton.jit
def _situ(
    X,
    Gamma,
    Out,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    GATE: tl.constexpr,
    UP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * N) + 2 * col, col < N, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * N) + 2 * col + 1, col < N, other=0).to(tl.float32)
    # Sigmoid uses the UNcapped gate, as in the released model.
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    gate = GATE * (2.0 / (1.0 + tl.exp(-2.0 * gate / GATE)) - 1.0)
    up = UP * (2.0 / (1.0 + tl.exp(-2.0 * up / UP)) - 1.0)
    result = ((gate * sigmoid) * up) * tl.load(Gamma + row)
    tl.store(Out + row * N + col, result.to(Out.dtype.element_ty), col < N)


def situ(x, gammas, gate_cap, up_cap):
    rows, twice_n = x.shape
    n = twice_n // 2
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if rows:
        _situ[(rows, triton.cdiv(n, 128))](
            x, gammas, out, n, rows, gate_cap, up_cap, 128, num_warps=4
        )
    return out
