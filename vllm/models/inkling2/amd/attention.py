"""Inkling2 ROCm attention, conservative unfused gated path.

Cache owners use the Asgard get_kv_cache_spec API. Each loop pass owns a
separate attention cache and a separate four-stream convolution cache.
"""

from typing import cast
import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from .cache_specs import (
    RelativeFullAttentionSpec as FullAttentionSpec,
    RelativeSlidingWindowSpec as SlidingWindowSpec,
)
from ..configs import Inkling2ModelConfig
from .sconv_swa_attn import InklingConvState
from .sconv import Inkling2ShortConv, K_STREAM, V_STREAM, conv_metadata
from .ops.linear import linear
from .ops.sconv import attention_prologue, shift_rel_proj
from .ops.attention_dispatch import run_attention
from .ops.tuning import OPTIMIZED

Inkling2AttentionBackend = FlashAttentionBackend


def _kv_cache_spec(vllm_config, num_kv_heads, head_dim, sliding_window):
    fields = dict(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        ),
    )
    return (
        FullAttentionSpec(**fields)
        if sliding_window is None
        else SlidingWindowSpec(**fields, sliding_window=sliding_window)
    )


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`
    (each pass of the loop core attends over its own keys and values)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Read by the metadata builders of the layer's attention group.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r. Local layers
    given separate `x_k` / `x_v` (loop passes after the first) project them
    as full layers do.

    A loop core layer runs once per loop pass, each pass with its own KV and
    conv-state caches: `conv_states[p]` and pass `p`'s KV cache (this layer's
    for pass 0, `pass_caches[p - 1]` after). With `loop_share_cache`, every
    pass reads and writes this layer's KV cache (later tokens see the last
    pass's K/V).
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window_size = (config.sliding_window_size - 1, 0) if is_local else (-1, -1)
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for cuBLAS (the extra columns,
        # from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("ROCm Inkling2 currently requires BF16 KV")
        self.fp8_kv = False
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        """The attention metadata key and KV cache of loop pass `pass_idx`."""
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        # ``linear`` on the weight bypasses the layer (and a LoRA wrapper's
        # forward), so the LoRA deltas are added here (no-op without LoRA).
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(self.q_side_splits)])
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp8_kv:
            # Every loop pass owns byte storage containing unit-scale E4M3 KV.
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's bf16 `(T, hidden)` partial output (to be
        all-reduced over TP), written to `out` if given.

        Local layers read q/k/v from `x_q` (or `x_k` / `x_v` if given); full
        layers take `x_k`, `x_v` and the fp32 log-scaling `tau` for q/r.
        `pass_idx` picks the loop pass's caches.
        """
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        out = linear(output, self.wo_ud.weight, out)
        return out

    @eager_break_during_capture
    def _attention(
        self,
        positions,
        q,
        r,
        gate,
        sinks,
        k,
        v,
        k_dynamic,
        v_dynamic,
        tau,
        output,
        pass_idx=0,
    ):
        num_tokens = q.shape[0]
        metadata = get_forward_context().attn_metadata
        name, cache = self._pass_cache(pass_idx)
        if not isinstance(metadata, dict) or cache.numel() == 0:
            output.zero_()
            return
        md = cast(FlashAttentionMetadata, metadata[name])
        owner = self.conv_states[pass_idx]
        conv_md = conv_metadata(owner)
        assert conv_md is not None
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        clipped_q = torch.empty_like(q)
        rel = q.new_empty((num_tokens + 128, self.num_heads, self.rel_extent + 256))
        key_cache, value_cache = self._split_kv_cache(cache)
        attention_prologue(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            k,
            v,
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            owner.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
        )
        if not OPTIMIZED:
            output.zero_()
        nt = md.num_actual_tokens
        run_attention(
            clipped_q,
            key_cache,
            value_cache,
            output,
            rel,
            gate,
            sinks,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_query_len=md.max_query_len,
            softmax_scale=self.scaling,
            window=self.sliding_window if self.is_local else 0,
            max_seq_len=md.max_seq_len,
            num_actual_tokens=nt,
        )
        if nt < num_tokens:
            output[nt:].zero_()
