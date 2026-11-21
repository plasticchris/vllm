"""Inkling2 experts on ROCm, checkpoint-packed weights and BF16 activations.

This follows the upstream Hopper weight-only path, not native Blackwell
activation quantization. No precision mode is silently relabeled.
"""

import os
from typing import Literal
import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from .ops.linear import linear_k_groups
from .ops.moe import expert_gemm, situ
from .ops.combine import combine_experts
from .ops.staged_experts import staged_expert_gemm

FUSED_COMBINE = os.environ.get("INKLING_FUSED_COMBINE", "0") == "1"
STAGE_EXPERTS = os.environ.get("INKLING_STAGE_EXPERTS", "off")

QuantFormat = Literal["nvfp4", "mxfp8"]


def select_experts(logits, bias, topk, route_scale):
    # Stable descending sort implements the published lower-expert-ID tie break.
    selection = torch.sigmoid(logits) + bias
    ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :topk]
    chosen = logits.gather(-1, ids)
    log_probs = torch.nn.functional.logsigmoid(chosen)
    weights = torch.softmax(log_probs, dim=-1) * route_scale
    return ids.to(torch.int32), weights


class Inkling2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        for p in (self.weight, self.bias):
            set_weight_attrs(p, {"weight_loader": default_weight_loader})


class Inkling2Experts(nn.Module):
    def __init__(self, config, quant_format):
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.nvfp4 = quant_format == "nvfp4"
        self.topk = config.num_experts_per_tok
        self.gate_cap, self.up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        h, i = config.hidden_size, config.intermediate_size
        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
            self._register(
                name + "_weight",
                (self.num_experts, n, k // pack),
                torch.uint8 if self.nvfp4 else torch.float8_e4m3fn,
            )
            self._register(
                name + "_weight_scale",
                (self.num_experts, n, k // group),
                torch.float8_e4m3fn if self.nvfp4 else torch.uint8,
            )
            if self.nvfp4:
                self._register(
                    name + "_weight_amax", (self.num_experts,), torch.float32
                )
                self._register(
                    name + "_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )
        self.global_scales = {}

    def _register(self, name, shape, dtype, per_expert=True):
        p = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(
            p, {"weight_loader": self._load_expert if per_expert else self._load_full}
        )
        self.register_parameter(name, p)

    @staticmethod
    def _load_full(param, loaded):
        if param.shape != loaded.shape or param.dtype != loaded.dtype:
            raise ValueError(
                f"Expert tensor mismatch {param.shape}/{param.dtype} vs {loaded.shape}/{loaded.dtype}"
            )
        param.data.copy_(loaded)

    def _load_expert(self, param, loaded):
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    def process_weights_after_loading(self):
        if self.nvfp4:
            for name in ("w13", "w2"):
                amax = getattr(self, name + "_weight_amax")
                denominator = getattr(self, name + "_weight_outer_scale_denominator")
                self.global_scales[name] = amax / denominator

    def forward(self, x, ids, weights):
        local = ids - self.ep_rank * self.num_experts
        local = torch.where((local >= 0) & (local < self.num_experts), local, -1)

        def project(inp, name, divisor, layout=None):
            stage = x.shape[0] >= 2048 and (
                STAGE_EXPERTS == "all" or (STAGE_EXPERTS == "mx" and not self.nvfp4)
            )
            implementation = staged_expert_gemm if stage else expert_gemm
            return implementation(
                inp,
                getattr(self, name + "_weight"),
                getattr(self, name + "_weight_scale"),
                self.global_scales.get(name),
                local,
                divisor,
                self.nvfp4,
                layout,
            )

        up, layout = project(x, "w13", self.topk)
        activated = situ(up, weights.reshape(-1), self.gate_cap, self.up_cap)
        down, _ = project(activated, "w2", 1, layout)
        outputs = down.view(x.shape[0], self.topk, x.shape[-1])
        if (
            FUSED_COMBINE
            and self.topk == 8
            and x.shape[-1] == 2304
            and x.dtype == torch.bfloat16
        ):
            return combine_experts(outputs)
        return outputs.float().sum(1).to(x.dtype)


class Inkling2MoE(nn.Module):
    def __init__(self, config, quant_format, visit_ids=(0,)):
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)

    def forward(self, x, reduce=True, push=None, lora=None, pass_idx=0):
        if push is not None or lora is not None or not reduce:
            raise NotImplementedError(
                "ROCm Inkling2 supports the ordinary reduced path only"
            )
        logits = linear_k_groups(x.contiguous(), self.gate.weight)
        ids, weights = select_experts(
            logits, self.gate.bias, self.gate.topk, self.gate.route_scale
        )
        out = self.experts(x, ids, weights)
        if self.experts.ep_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
