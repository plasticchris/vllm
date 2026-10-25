# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer


def _mock_tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


# NOTE: The three PLE-FP8 unit tests from asgard's pre-rebase qwen4_exp (test_amd_ple_declared_fp8_dtype_creates_scaled_embedding, test_amd_ple_fp8_output_uses_global_scale, test_amd_ple_fp8_ngram_forward_preserves_storage_dtype) were removed: they exercise asgard's old model's Qwen4ExpNGramEmbedding (max_num_reqs kwarg, ReplicatedLinear key_proj), which the rebase does not carry — the rebase uses upstream's newer Qwen4ExpPLELayer + common.ngram_embedding Qwen4ExpPLEFp8EmbeddingMethod. The amd/ple_layer Qwen4ExpPLEFp8EmbeddingMethod class the delta added is dead code on this model and is only kept so the delta cherry-pick stays minimal.


def test_complete_amd_ple_boundary_writes_output(monkeypatch) -> None:
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    expected = torch.randn(2, 4)
    monkeypatch.setattr(
        Qwen4ExpPLELayer,
        "forward",
        lambda self, hidden_states, input_ids, query_start_loc, ngram_context: expected,
    )
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"model.ple": layer}),
    )
    output = torch.empty_like(expected)

    ple_layer_module.qwen4_exp_amd_ple(
        torch.randn_like(expected),
        torch.tensor([1, 2]),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([[0]]),
        output,
        "model.ple",
    )

    torch.testing.assert_close(output, expected)

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires a ROCm GPU",
)
@pytest.mark.parametrize("shape", [(17,), (37, 4), (1, 4)])
def test_pipelined_cpu_embedding_matches_regular_lookup(shape) -> None:
    ple_layer_module._PLE_CPU_PIPELINE_STAGING.clear()
    weight = torch.randn(1000, 32, dtype=torch.float16).pin_memory()
    input_ids = torch.randint(0, weight.shape[0], shape, device="cuda")

    output = ple_layer_module._pipelined_cpu_embedding(input_ids, weight, 8)
    expected = torch.nn.functional.embedding(input_ids.cpu(), weight).cuda()

    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
