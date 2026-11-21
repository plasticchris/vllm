import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from vllm.models.inkling2.amd.cache_specs import (
    RelativeFullAttentionSpec,
    RelativeSlidingWindowSpec,
    RotaryFullAttentionSpec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.core.leyline_relative import (
    is_relative_topology,
    prepare_relative_splice,
    has_relative_cache_groups,
)
from vllm.v1.core.leyline_splice import commit_splice
from vllm.v1.core.kv_cache_utils import KVCacheBlockSplice
from vllm.v1.attention.ops.leyline_relative import (
    apply_relative_splices,
    splice_jump_request_ids,
)


class RelativeSplice(unittest.TestCase):
    def config(self):
        def spec(cls, **kw):
            return cls(
                block_size=4, num_kv_heads=1, head_size=2, dtype=torch.bfloat16, **kw
            )

        return NS(
            num_blocks=16,
            has_mamba_layers=False,
            kv_cache_groups=[
                NS(kv_cache_spec=spec(RelativeFullAttentionSpec), layer_names=["fa"]),
                NS(
                    kv_cache_spec=spec(RelativeSlidingWindowSpec, sliding_window=4),
                    layer_names=["conv"],
                ),
            ],
        )

    def test_counter_jump_without_mamba(self):
        sp = KVCacheBlockSplice(1, 2, 0, 4, 0, 4, -4, 0, "relative-request")
        output = NS(mamba_state_transplants=[], kv_cache_block_splices=[sp])
        self.assertEqual(splice_jump_request_ids(output), {"relative-request"})
        # Negative control: the historical Mamba-only selector misses it.
        self.assertNotEqual(
            {op.req_id for op in output.mamba_state_transplants},
            splice_jump_request_ids(output),
        )
        self.assertEqual(splice_jump_request_ids(NS()), set())

    def test_native_convolution_page(self):
        from vllm.models.inkling2.amd import sconv_swa_attn as module

        cfg = NS(
            model_config=NS(dtype=torch.bfloat16),
            compilation_config=NS(static_forward_context={}),
        )
        with (
            patch.object(module, "get_current_vllm_config", return_value=cfg),
            patch.object(
                module, "get_tensor_model_parallel_world_size", return_value=4
            ),
        ):
            owner = module.InklingConvState(
                num_kv_heads=8,
                head_dim=128,
                hidden_size=2304,
                kernel_size=4,
                prefix="test.conv",
            )
            spec = owner.get_kv_cache_spec(cfg)
        self.assertEqual((spec.sliding_window, spec.block_size), (4, 64))
        self.assertEqual(spec.page_size_bytes, 1024**2)
        self.assertEqual(spec.leyline_position_encoding, "relative")

    def test_trimmed_convolution_page(self):
        from vllm.models.inkling2.amd import sconv_swa_attn as module

        cfg = NS(
            model_config=NS(dtype=torch.bfloat16),
            compilation_config=NS(static_forward_context={}),
        )
        with (
            patch.object(module, "get_current_vllm_config", return_value=cfg),
            patch.object(
                module, "get_tensor_model_parallel_world_size", return_value=4
            ),
            patch.dict("os.environ", {"INKLING_TRIM_CONV_PADDING": "1"}),
        ):
            owner = module.InklingConvState(
                num_kv_heads=8,
                head_dim=128,
                hidden_size=2304,
                kernel_size=4,
                prefix="test.conv",
            )
            spec = owner.get_kv_cache_spec(cfg)
        self.assertEqual(
            (spec.sliding_window, spec.block_size, spec.head_size), (4, 64, 2560)
        )
        self.assertEqual(spec.page_size_bytes, 655360)

    def test_mixed_rotary_group(self):
        c = self.config()
        c.kv_cache_groups[1].kv_cache_spec = RotaryFullAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=2, dtype=torch.bfloat16
        )
        self.assertTrue(is_relative_topology(c))
        raw = torch.arange(16 * 1 * 4 * 4).reshape(16, 1, 4, 4).to(torch.bfloat16)
        target = raw.clone()
        ctx = {
            "fa": NS(kv_cache=target),
            "conv": NS(
                kv_cache=raw.clone(), head_size=2, leyline_rope_inv_freq=torch.ones(1)
            ),
        }
        sp = KVCacheBlockSplice(1, 9, 0, 4, 0, 4, -4, 1, "request")
        with patch(
            "vllm.v1.attention.ops.leyline_rope.rotate_key_cache_block_inplace"
        ) as rotate:
            apply_relative_splices(ctx, c, [sp])
        self.assertEqual(rotate.call_count, 1)
        self.assertEqual(rotate.call_args.kwargs["delta_pos"], -4)
        self.assertEqual(
            tuple(rotate.call_args.kwargs["key_cache"].shape), (16, 4, 1, 2)
        )
        self.assertTrue(torch.equal(target, raw))
        self.assertTrue(torch.equal(ctx["conv"].kv_cache[9], raw[1]))

    def test_explicit_contract(self):
        c = self.config()
        self.assertTrue(is_relative_topology(c))
        c.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=2, dtype=torch.bfloat16
        )
        self.assertFalse(is_relative_topology(c))
        self.assertTrue(has_relative_cache_groups(c))

    def test_group_routed_copy(self):
        c = self.config()
        fa = torch.arange(16 * 1 * 4 * 2).reshape(16, 1, 4, 2).clone()
        conv = 1000 + fa.clone()
        before = conv.clone()
        ctx = {"fa": NS(kv_cache=fa), "conv": NS(kv_cache=conv)}
        sp = KVCacheBlockSplice(1, 9, 1, 4, 0, 3, -5, 0)
        expected = fa[1, :, 1:4].clone()
        apply_relative_splices(ctx, c, [sp])
        self.assertTrue(torch.equal(fa[9, :, :3], expected))
        self.assertTrue(torch.equal(conv, before))
        # Inject a bad group into a second op: fail before the first writes.
        original = fa.clone()
        bad = KVCacheBlockSplice(1, 10, 0, 4, 0, 4, -5, 9)
        with self.assertRaises(ValueError):
            apply_relative_splices(
                ctx, c, [KVCacheBlockSplice(2, 9, 0, 4, 0, 4, -5, 0), bad]
            )
        self.assertTrue(torch.equal(fa, original))

    def test_prepare_commit_and_eviction(self):
        c = self.config()
        blocks = {}
        for gid in range(2):
            for i in range(8):
                blocks[(gid, i)] = NS(
                    block_id=gid * 8 + i, is_null=False, block_hash=(gid, i), ref_cnt=0
                )
        pins = []
        pool = NS(
            hash_block_size=4,
            touch=lambda xs: pins.extend(xs),
            get_cached_block=lambda h, gids: (
                [blocks[(gids[0], h)]] if (gids[0], h) in blocks else None
            ),
        )
        managers = [
            NS(block_size=4, cached_blocks_this_step=set(), req_to_blocks={})
            for _ in range(2)
        ]
        # Real SlidingWindowManager has no Mamba publication-tracking set.
        del managers[1].cached_blocks_this_step
        manager = NS(
            kv_cache_config=c,
            coordinator=NS(single_type_managers=managers),
            block_pool=pool,
            _pending_kv_cache_block_splices=[],
            _pending_mamba_state_transplants=[],
            _pending_kv_cache_block_splice_retained=[],
        )
        req = NS(request_id="test", donor_block_hashes=list(range(8)))
        with patch(
            "vllm.v1.core.leyline_relative.resolve_block_hashes",
            lambda hashes, *args: hashes,
        ):
            plan = prepare_relative_splice(manager, req, 0, 8, 2, 20)
        self.assertIsNotNone(plan)
        self.assertEqual((plan.middle_end, plan.target_end), (2, 18))
        self.assertEqual(len(plan.states), 0)
        for gid, mgr in enumerate(managers):
            mgr.req_to_blocks["test"] = [
                NS(block_id=30 + gid * 8 + i, is_null=False, block_hash=None, ref_cnt=1)
                for i in range(8)
            ]
        self.assertTrue(commit_splice(manager, req, 1))
        self.assertEqual(req.kv_splice_stats["skipped_tokens"], 16)
        self.assertEqual(
            {sp.req_id for sp in manager._pending_kv_cache_block_splices}, {"test"}
        )
        self.assertEqual(
            {sp.kv_cache_group_id for sp in manager._pending_kv_cache_block_splices},
            {0, 1},
        )
        del blocks[(1, 5)]
        req = NS(request_id="evicted", donor_block_hashes=list(range(8)))
        touched = len(pins)
        with patch(
            "vllm.v1.core.leyline_relative.resolve_block_hashes",
            lambda hashes, *args: hashes,
        ):
            plan = prepare_relative_splice(manager, req, 0, 8, 2, 20)
        self.assertIsNone(plan)
        self.assertEqual(len(pins), touched)
        self.assertEqual(req.kv_splice_stats["skipped_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
