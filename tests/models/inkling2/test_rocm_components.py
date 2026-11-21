"""Run with TRITON_INTERPRET=1 on CPU, or INKLING_TEST_DEVICE=cuda:0.
Numerical references are unfused torch implementations, not ported kernels.
"""

import os
import unittest
from unittest.mock import patch
import torch
from vllm.models.inkling2.amd.ops.norm import rms_norm
from vllm.models.inkling2.amd.ops.attn_res import attn_res
from vllm.models.inkling2.amd.ops.loop import pair_rms_norm, blend_rms_norm
from vllm.models.inkling2.amd.ops.linear import linear_k_groups
from vllm.models.inkling2.amd.ops.moe import _expert_gemm, situ
from vllm.models.inkling2.amd.moe import select_experts

DEVICE = os.environ.get("INKLING_TEST_DEVICE", "cpu")


def rms(x, eps=1e-6):
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)


@unittest.skipUnless(
    os.environ.get("INKLING_TEST_DEVICE", "").startswith("cuda:"),
    "GPU tests require INKLING_TEST_DEVICE=cuda:N and the external memory guard",
)
class Components(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8711)

    def rand(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, device=DEVICE, dtype=dtype)

    def close(self, actual, expected, atol=0.02, rtol=0.02):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )

    def test_norm(self):
        for t in (1, 8, 128):
            x, w = (
                self.rand(t, 2304, dtype=torch.bfloat16),
                self.rand(2304, dtype=torch.bfloat16),
            )
            self.close(rms_norm(x, w, 1e-6), (rms(x) * w.float()).bfloat16())

    def test_controller(self):
        for t in (1, 8):
            msg = self.rand(3, t, 2304)
            dest = self.rand(1, t, 2304).expand_as(msg)
            actual = pair_rms_norm(dest, msg)
            expected = torch.cat((rms(dest), rms(msg)), -1).bfloat16()
            self.close(actual, expected)
            gf, nw = self.rand(3, t, 4608), self.rand(3, 2304, dtype=torch.bfloat16)
            g, f = gf.chunk(2, -1)
            blended = (1 - f.tanh()) * dest + g.tanh() * msg
            expected = (rms(blended) * nw[:, None].float()).bfloat16()
            self.close(blend_rms_norm(dest, msg, gf, nw, 1e-6), expected)

    def test_attn_res(self):
        for t in (1, 8):
            stack = self.rand(2, t, 2304, dtype=torch.bfloat16)
            last, add = (
                self.rand(t, 2304, dtype=torch.bfloat16),
                self.rand(t, 2304, dtype=torch.bfloat16),
            )
            q = self.rand(3, 2304) / 2304**0.5
            sources = torch.cat((stack.float(), (last.float() + add.float())[None]))
            normalized = rms(sources)
            logits = torch.einsum("std,qd->qst", normalized, q)
            mix = torch.einsum("qst,std->qtd", logits.softmax(1), normalized)
            self.close(
                attn_res(stack, last, add, q, None, 0.0, True), mix, 0.002, 0.002
            )

    def test_linear(self):
        x, w = (
            self.rand(3, 2304, dtype=torch.bfloat16),
            self.rand(64, 2304, dtype=torch.bfloat16),
        )
        expected = sum(
            x[:, i : i + 1024].float() @ w[:, i : i + 1024].float().t()
            for i in range(0, 2304, 1024)
        )
        self.close(linear_k_groups(x, w), expected, 0.005, 0.005)

    def test_routing(self):
        logits = torch.zeros(2, 128, device=DEVICE)
        bias = torch.zeros(128, device=DEVICE)
        ids, weights = select_experts(logits, bias, 8, 8.0)
        self.assertEqual(ids.tolist(), [list(range(8))] * 2)
        self.close(weights, torch.ones_like(weights), 0.00001, 0.00001)

    def test_situ(self):
        x = self.rand(8, 256, dtype=torch.bfloat16) * 10
        weights = self.rand(8)
        g, u = x[:, 0::2].float(), x[:, 1::2].float()
        ref = (
            ((4 * (g / 4).tanh()) * g.sigmoid())
            * (25 * (u / 25).tanh())
            * weights[:, None]
        )
        actual = situ(x, weights, 4.0, 25.0)
        self.close(actual, ref.bfloat16(), 0.03, 0.02)
        # Negative control: capping before sigmoid MUST change this case.
        wrong = (
            ((4 * (g / 4).tanh()) * (4 * (g / 4).tanh()).sigmoid())
            * (25 * (u / 25).tanh())
            * weights[:, None]
        )
        self.assertGreater((actual.float() - wrong).abs().max().item(), 0.1)

    def test_attention(self):
        from vllm.models.inkling2.amd.ops.attention_sm90 import (
            rel_attention,
            MODE_GATED,
        )

        for window in (0, 8):
            for t, past in ((8, 0), (1, 23), (4, 17)):
                h, kv, d, extent, page = 6, 2, 128, 16, 32
                length = past + t
                q = self.rand(t, h, d, dtype=torch.bfloat16)
                k = self.rand(1, page, kv, d, dtype=torch.bfloat16)
                v = self.rand(1, page, kv, d, dtype=torch.bfloat16)
                gate, sink = (
                    self.rand(t, h * d, dtype=torch.bfloat16),
                    self.rand(t, h, dtype=torch.bfloat16),
                )
                bias = self.rand(t, h, extent, dtype=torch.bfloat16)
                rel = torch.zeros(
                    t + 128, h, extent + 256, device=DEVICE, dtype=torch.bfloat16
                )
                base = ((past + t + 127) // 128) * 128 - (extent + 256)
                for token in range(t):
                    for dist in range(extent):
                        key = past + token - dist
                        col = key - base
                        if 0 <= col < extent + 256:
                            rel[token, :, col] = bias[token, :, dist]
                out = torch.empty(t, h * d, device=DEVICE, dtype=torch.bfloat16)
                rel_attention(
                    q,
                    k,
                    v,
                    out,
                    None,
                    rel,
                    gate,
                    sink,
                    block_table=torch.zeros(1, 1, device=DEVICE, dtype=torch.int32),
                    cache_seqlens=torch.tensor(
                        [length], device=DEVICE, dtype=torch.int32
                    ),
                    cu_seqlens_q=torch.tensor([0, t], device=DEVICE, dtype=torch.int32),
                    max_query_len=t,
                    softmax_scale=d**-0.5,
                    window=window,
                    num_splits=1,
                    mode=MODE_GATED,
                )
                kk = k[0, :length].float().repeat_interleave(h // kv, 1)
                vv = v[0, :length].float().repeat_interleave(h // kv, 1)
                ref = []
                for token in range(t):
                    scores = torch.einsum("hd,khd->hk", q[token].float(), kk) * d**-0.5
                    for key in range(length):
                        dist = past + token - key
                        if dist < 0 or (window and dist >= window):
                            scores[:, key] = -float("inf")
                        elif dist < extent:
                            scores[:, key] += bias[token, :, dist].float()
                    mixed = torch.einsum("hk,khd->hd", scores.softmax(-1), vv)
                    mixed *= gate[token].view(h, d).float().sigmoid()
                    mixed *= (scores.logsumexp(-1) - sink[token].float()).sigmoid()[
                        :, None
                    ]
                    ref.append(mixed.flatten())
                self.close(out, torch.stack(ref).bfloat16(), 0.025, 0.025)
                from vllm.models.inkling2.amd.ops.attention_dispatch import (
                    run_attention,
                )

                with (
                    patch(
                        "vllm.models.inkling2.amd.ops.attention_dispatch.OPTIMIZED",
                        True,
                    ),
                    patch(
                        "vllm.models.inkling2.amd.ops.attention_dispatch.attention_launch_config",
                        return_value=(4, (32, 32, 4)),
                    ),
                ):
                    run_attention(
                        q,
                        k,
                        v,
                        out,
                        rel,
                        gate,
                        sink,
                        block_table=torch.zeros(1, 1, device=DEVICE, dtype=torch.int32),
                        cache_seqlens=torch.tensor(
                            [length], device=DEVICE, dtype=torch.int32
                        ),
                        cu_seqlens_q=torch.tensor(
                            [0, t], device=DEVICE, dtype=torch.int32
                        ),
                        max_query_len=t,
                        max_seq_len=length,
                        softmax_scale=d**-0.5,
                        window=window,
                        num_actual_tokens=t,
                    )
                self.close(out, torch.stack(ref).bfloat16(), 0.025, 0.025)

    def test_short_conv(self):
        from vllm.models.inkling2.amd.ops.sconv import short_conv

        t, c, w = 8, 32, 4
        x = self.rand(t, c, dtype=torch.bfloat16)
        weights = self.rand(c, w, dtype=torch.bfloat16)
        cache = torch.zeros(1, 1, 32, c, device=DEVICE, dtype=torch.bfloat16)
        bt = torch.zeros(1, 1, device=DEVICE, dtype=torch.int32)
        seq = torch.zeros(t, device=DEVICE, dtype=torch.int32)
        pos = torch.arange(t, device=DEVICE)

        def call(xx, pp, qs):
            return short_conv(xx, weights, cache, pp, bt, seq[: len(pp)], pp, qs, 0, c)

        first = call(x[:5], pos[:5], seq[:5])
        second = call(x[5:], pos[5:], seq[:3])
        ref = []
        for i in range(t):
            val = x[i].float()
            for j in range(w):
                src = i - (w - 1 - j)
                if src >= 0:
                    val = val + x[src].float() * weights[:, j].float()
            ref.append((val / (w + 1) ** 0.5).bfloat16())
        self.close(torch.cat((first, second)), torch.stack(ref), 0.025, 0.025)

    def test_owned_logits_softcap(self):
        from vllm.models.inkling2.amd.ops.logits import softcap_owned_logits

        for dtype in (torch.float32, torch.bfloat16):
            parent = torch.randn(17, 272, device=DEVICE, dtype=dtype) * 40
            work = parent[:, :257]
            original = work.clone()
            padding = parent[:, 257:].clone()
            expected = torch.tanh(original / 30.0) * 30.0
            result = softcap_owned_logits(work, 30.0)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            self.assertEqual(result.data_ptr(), work.data_ptr())
            self.assertTrue(torch.equal(parent[:, 257:], padding))

    def test_packed_experts(self):
        # Every E2M1 code (including signed zero), distinct expert scales,
        # nontrivial token routing, both projections' input-index contracts.
        e, n, k, rows, bm = 2, 32, 64, 6, 16
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device=DEVICE,
        )
        code = torch.arange(e * n * k, device=DEVICE).reshape(e, n, k) % 16
        packed = (code[:, :, 0::2] | (code[:, :, 1::2] << 4)).to(torch.uint8)
        scale = torch.full((e, n, k // 16), 0.5, device=DEVICE).to(torch.float8_e4m3fn)
        glob = torch.tensor([0.25, 2.0], device=DEVICE)
        decoded = (
            lut[code] * scale.float().repeat_interleave(16, -1) * glob[:, None, None]
        )
        sorted_ids = torch.tensor(
            [0, 2, 4] + [rows] * 13 + [1, 3, 5] + [rows] * 13,
            device=DEVICE,
            dtype=torch.int32,
        )
        experts = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32)
        padded = torch.tensor([32], device=DEVICE, dtype=torch.int32)
        for div in (1, 2):
            x = self.rand(rows // div, k, dtype=torch.bfloat16)
            out = torch.zeros(rows, n, device=DEVICE, dtype=torch.bfloat16)
            _expert_gemm[(2, 1)](
                x,
                packed,
                scale,
                glob,
                sorted_ids,
                experts,
                padded,
                out,
                rows,
                n,
                k,
                div,
                True,
                bm,
                32,
                64,
                num_warps=4,
                num_stages=1,
            )
            ref = torch.stack(
                [
                    x[r // div].float() @ decoded[r % 2].bfloat16().float().t()
                    for r in range(rows)
                ]
            ).bfloat16()
            self.close(out, ref, 0.03, 0.01)
            from vllm.models.inkling2.amd.ops.moe import expert_gemm

            ids = (torch.arange(rows, device=DEVICE) % 2).reshape(rows // div, div)
            with patch("vllm.models.inkling2.amd.ops.moe.OPTIMIZED", True):
                tuned, _ = expert_gemm(x, packed, scale, glob, ids, div, True)
            self.close(tuned, ref, 0.03, 0.01)
        fp8 = self.rand(e, n, k).to(torch.float8_e4m3fn)
        exponents = (
            torch.arange(e * n * (k // 32), device=DEVICE).reshape(e, n, k // 32) % 5
            + 125
        )
        scales = exponents.to(torch.uint8)
        decoded = fp8.float() * torch.exp2(exponents.float() - 127).repeat_interleave(
            32, -1
        )
        x = self.rand(rows, k, dtype=torch.bfloat16)
        out = torch.zeros(rows, n, device=DEVICE, dtype=torch.bfloat16)
        _expert_gemm[(2, 1)](
            x,
            fp8,
            scales,
            None,
            sorted_ids,
            experts,
            padded,
            out,
            rows,
            n,
            k,
            1,
            False,
            bm,
            32,
            64,
            num_warps=4,
            num_stages=1,
        )
        ref = torch.stack(
            [x[r].float() @ decoded[r % 2].bfloat16().float().t() for r in range(rows)]
        ).bfloat16()
        self.close(out, ref, 0.03, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
