# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    normalize_and_validate_gptq_desc_act,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    INT4_DTYPE,
    INT8_DTYPE,
    QuantKey,
    kInt4Static32GroupScale,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform


class MoeWNA16Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None,
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        # Avoid circular import
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        if self.linear_quant_method == "gptq":
            normalize_and_validate_gptq_desc_act(
                full_config.get("desc_act", False),
                group_size,
                full_config.get("dynamic") or {},
            )
        elif self.linear_quant_method in ("awq", "awq_marlin"):
            capability_tuple = current_platform.get_device_capability()
            device_capability = (
                -1 if capability_tuple is None else capability_tuple.to_int()
            )
            awq_min_capability = AutoAWQConfig.get_min_capability()
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "moe_wna16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method in ("awq", "awq_marlin"):
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        return cls(
            linear_quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        can_convert = cls.is_moe_wna16_compatible(hf_quant_cfg)
        if can_convert and user_quant == "moe_wna16":
            return cls.get_name()
        return None

    @classmethod
    def is_moe_wna16_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act", False)
        try:
            normalize_and_validate_gptq_desc_act(
                desc_act,
                quant_config.get("group_size"),
                quant_config.get("dynamic") or {},
            )
            supported_act_order = True
        except ValueError:
            supported_act_order = False

        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )
        # Avoid circular import
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_min_capability = AutoAWQConfig.get_min_capability()

        gptq_compatible = (
            quant_method == "gptq" and supported_act_order and num_bits in [4, 8]
        )
        awq_compatible = (
            quant_method == "awq"
            and num_bits == 4
            and device_capability >= awq_min_capability
        )

        return gptq_compatible or awq_compatible

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
            from vllm.model_executor.layers.quantization.auto_gptq import (
                AutoGPTQConfig,
            )

            linear_config: QuantizationConfig
            if self.linear_quant_method == "gptq":
                linear_config = AutoGPTQConfig.from_config(self.full_config)
            elif self.linear_quant_method in ("awq", "awq_marlin"):
                linear_config = AutoAWQConfig.from_config(self.full_config)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")

            # The delegate is rebuilt from the raw HF quantization dict, which
            # lists shard names and never fused ones. Without the mapping it
            # resolves fused layers (qkv_proj, gate_up_proj) to
            # UnquantizedLinearMethod and the checkpoint's qweight has nowhere
            # to load.
            linear_config.packed_modules_mapping = self.packed_modules_mapping
            return linear_config.get_quant_method(layer, prefix)
        elif isinstance(layer, RoutedExperts):
            return MoeWNA16Method(self, layer.moe_config)
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeWNA16Method(FusedMoEMethodBase):
    """Linear method for MOE WNA16 (W8A16/W4A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.

    """

    def __init__(self, quant_config: MoeWNA16Config, moe: "FusedMoEConfig") -> None:
        super().__init__(moe)
        self.quant_config = quant_config

        num_bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size

        if num_bits == 4:
            quant_type = INT4_DTYPE
            if group_size == 32:
                scale = kInt4Static32GroupScale
            else:
                scale = kInt4StaticGroupScale
        elif num_bits == 8:
            quant_type = INT8_DTYPE
            scale = kInt8StaticGroupScale
        else:
            raise ValueError("MoeWNA16Method only supports int4 and int8 now.")

        weight_key = QuantKey(quant_type, scale)

        # Select WNA16 MoE backend via oracle.
        # handle ZP?
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            quant_config=self.quant_config,
            may_have_zp=self.quant_config.has_zp,
            may_have_bias=False,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size divisible by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = MoeWNA16Method.get_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards
                    * intermediate_size_per_partition
                    // bit8_pack_factor,
                    hidden_size // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // bit8_pack_factor,
                    intermediate_size_per_partition // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if self.quant_config.linear_quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys: list[str] = []
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int32), requires_grad=False
                )
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        if self.wna16_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming import (
                get_humming_moe_quant_config,
            )

            return get_humming_moe_quant_config(
                layer,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )

        has_zp = self.quant_config.has_zp
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=layer.w13_qzeros if has_zp else None,
            w2_zp=layer.w2_qzeros if has_zp else None,
            group_size=layer.group_size,
            num_bits=self.quant_config.weight_bits,
        )

    def _setup_kernel(self, layer: RoutedExperts):
        assert self.experts_cls is not None
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_backend,
            routing_tables=layer._expert_routing_tables(),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if os.environ.get("VLLM_ROCM_RDNA3_NATIVE_MOE", "0") in (
            "1",
            "true",
            "yes",
        ):
            self._process_rdna3_native_weights(layer)
            return

        has_zp = self.quant_config.has_zp
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.quant_config,
            input_dtype=None,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_qzeros=layer.w13_qzeros if has_zp else None,
            w2_qzeros=layer.w2_qzeros if has_zp else None,
        )

        if converted is None:
            # Backend rewrote the layer's params in place (e.g. Humming).
            self._setup_kernel(layer)
            return

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            _,  # w13_bias
            _,  # w2_bias
        ) = converted

        # Replace common parameters
        replace_parameter(layer, "w13_qweight", w13_qweight)
        replace_parameter(layer, "w2_qweight", w2_qweight)
        replace_parameter(layer, "w13_scales", w13_scales)
        replace_parameter(layer, "w2_scales", w2_scales)
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        if has_zp:
            assert w13_qzeros is not None and w2_qzeros is not None
            replace_parameter(layer, "w13_qzeros", w13_qzeros)
            replace_parameter(layer, "w2_qzeros", w2_qzeros)

        # Marlin-specific parameters (not needed for Flashinfer)
        if self.wna16_backend != WNA16MoEBackend.FLASHINFER_TRTLLM:
            # Register input global scales if present
            if w13_input_global_scale is not None:
                layer.register_parameter(
                    "w13_input_global_scale",
                    torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
                )
            if w2_input_global_scale is not None:
                layer.register_parameter(
                    "w2_input_global_scale",
                    torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
                )

        self._setup_kernel(layer)

    def _process_rdna3_native_weights(self, layer: RoutedExperts) -> None:
        if (
            self.quant_config.weight_bits != 4
            or self.quant_config.has_zp
            or not current_platform.is_rocm()
        ):
            raise ValueError(
                "VLLM_ROCM_RDNA3_NATIVE_MOE requires symmetric INT4 on ROCm"
            )
        from vllm import _custom_ops as ops
        from vllm import _rocm_C  # noqa: F401

        if not hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3"):
            raise ValueError("native gfx1100 MoE kernel is not built")
        # AutoRound/GPTQ stores uint8 [E, N, K/2]. The native gfx1100 kernel
        # consumes shuffled int32 [E, K/8, N]. Repack without changing nibbles.
        def repack(weight: torch.Tensor) -> torch.Tensor:
            assert weight.dtype == torch.uint8 and weight.is_contiguous()
            packed = weight.view(torch.int32).transpose(1, 2).contiguous()
            for expert in range(packed.shape[0]):
                # Upstream's gptq_shuffle is 2-arg (no g_idx); the identity
                # shuffle (empty perm) is what the native kernel wants.
                ops.gptq_shuffle(packed[expert], 4)
            return packed

        w13_qweight = repack(layer.w13_qweight.data)
        w2_qweight = repack(layer.w2_qweight.data)
        w13_scales = layer.w13_scales.transpose(1, 2).contiguous()
        w2_scales = layer.w2_scales.transpose(1, 2).contiguous()
        replace_parameter(
            layer,
            "w13_qweight",
            torch.nn.Parameter(w13_qweight, requires_grad=False),
        )
        replace_parameter(
            layer,
            "w2_qweight",
            torch.nn.Parameter(w2_qweight, requires_grad=False),
        )
        replace_parameter(
            layer,
            "w13_scales",
            torch.nn.Parameter(w13_scales, requires_grad=False),
        )
        replace_parameter(
            layer,
            "w2_scales",
            torch.nn.Parameter(w2_scales, requires_grad=False),
        )

        # Alias the parameter names expected by the native CompressedTensors
        # runner without registering duplicate state-dict entries.
        object.__setattr__(layer, "w13_weight_packed", layer.w13_qweight)
        object.__setattr__(layer, "w2_weight_packed", layer.w2_qweight)
        object.__setattr__(layer, "w13_weight_scale", layer.w13_scales)
        object.__setattr__(layer, "w2_weight_scale", layer.w2_scales)
        # Empty zero-point tensors select the symmetric INT4 fast path in the
        # native kernel (implicit dequantization zero point 8).
        empty_qzeros = torch.empty(0, dtype=torch.int32, device=w13_qweight.device)
        object.__setattr__(layer, "w13_qzeros", empty_qzeros)
        object.__setattr__(layer, "w2_qzeros", empty_qzeros)

        max_decode_tokens = 16
        max_topk = max(10, self.moe.experts_per_token)
        rows = max_decode_tokens * max_topk
        gate_up = w13_qweight.shape[2]
        intermediate = gate_up // 2
        hidden = w2_qweight.shape[2]
        layer.rdna3_w1_buf = torch.zeros(
            rows, gate_up, dtype=w13_scales.dtype, device=w13_scales.device
        )
        layer.rdna3_act_buf = torch.empty(
            rows, intermediate, dtype=w13_scales.dtype, device=w13_scales.device
        )
        layer.rdna3_out_buf = torch.zeros(
            max_decode_tokens, hidden, dtype=w13_scales.dtype, device=w13_scales.device
        )
        layer.rdna3_empty_tw = torch.empty(0, device=w13_scales.device)
        self._rdna3_native = True
        self.moe_kernel = None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        if getattr(self, "_rdna3_native", False):
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna3 import (  # noqa: E501
                _rdna3_fused_moe,
            )

            return _rdna3_fused_moe(
                x,
                topk_weights,
                topk_ids,
                layer=layer,
                activation=(
                    layer.activation
                    if isinstance(layer.activation, MoEActivation)
                    else MoEActivation.from_str(layer.activation)
                ),
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
            )
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
        )

    @staticmethod
    def get_weight_loader(layer, weight_loader):
        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ):
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return False if return_success else None

            device = get_tp_group().device
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            # convert gptq and awq weight to a standard format
            # awq_marlin uses the same weight format as awq
            if layer.quant_config.linear_quant_method in ("awq", "awq_marlin"):
                assert layer.quant_config.weight_bits == 4
                if "weight" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qweight")
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    loaded_weight = loaded_weight.T
            elif layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [4, 8]
                if "weight" in weight_name:
                    loaded_weight = loaded_weight.T.contiguous().view(torch.uint8)
                elif "zeros" in weight_name:
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T

            # repeat the qzeros/scales to fit new group size
            if (
                layer.group_size_div_factor > 1
                and "qzeros" in weight_name
                or "scales" in weight_name
            ):
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1
                )

            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_config.tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                # qzeros are packed along the intermediate dim by
                # bit8_pack_factor, so the w1/w3 boundary is in packed units
                # (equals shard_size // 2 only for 4-bit).
                shard_zp_size = shard_size // layer.quant_config.bit8_pack_factor
                if shard_id == "w1":
                    param.data[expert_id, :shard_zp_size] = tensor
                else:
                    param.data[expert_id, shard_zp_size:] = tensor
                return True if return_success else None
            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_config.tp_size, -1
                )[:, tp_rank]
                return True if return_success else None
            else:
                # Delegate to the original loader, passing return_success
                return weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success=return_success,
                )

        return moe_wna16_weight_loader
