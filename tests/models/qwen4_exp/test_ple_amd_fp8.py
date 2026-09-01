# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLELayer,
)


def _mock_tp1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def test_amd_ple_declared_fp8_dtype_creates_scaled_embedding(monkeypatch) -> None:
    _mock_tp1(monkeypatch)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        eos_token_id=0,
        vocab_size=8,
        split_ngram_parts=2,
        seed=1234,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=1,
        ple_embedding_dtype="float8_e4m3fn",
    )

    module = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2,
        ple_dense_layer_id=0,
        max_total_tokens=4,
        max_num_reqs=1,
        prefix="ple",
        layer_name="layer",
        params_dtype=torch.bfloat16,
    )

    assert isinstance(module.ngram_embedding.quant_method, Qwen4ExpPLEFp8EmbeddingMethod)
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert module.ngram_embedding.weight_scale.dtype == torch.bfloat16


def test_amd_ple_fp8_output_uses_global_scale(monkeypatch) -> None:
    _mock_tp1(monkeypatch)
    embedding = embedding_module.VocabParallelEmbedding(
        3,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=Qwen4ExpPLEFp8EmbeddingMethod(),
    )
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    embedding.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    embedding.weight_scale.data.copy_(torch.tensor([0.25], dtype=torch.bfloat16))
    quantized = embedding(torch.tensor([2, 0]))

    ple = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple)
    ple.ple_embedding = nn.Module()
    ple.ple_embedding.ngram_embedding = embedding

    output = ple._dequantize_embeddings(quantized, torch.bfloat16)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, (weight[[2, 0]] * 0.25).bfloat16())


def test_amd_ple_fp8_ngram_forward_preserves_storage_dtype(monkeypatch) -> None:
    _mock_tp1(monkeypatch)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        eos_token_id=0,
        vocab_size=8,
        split_ngram_parts=2,
        seed=1234,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=1,
        ple_embedding_dtype="float8_e4m3fn",
    )
    module = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2,
        ple_dense_layer_id=0,
        max_total_tokens=4,
        max_num_reqs=1,
        prefix="ple",
        layer_name="layer",
        params_dtype=torch.bfloat16,
    )
    observed_dtypes = []

    def fake_lookup(ngram_ids, output, layer_name) -> None:
        del ngram_ids, layer_name
        observed_dtypes.append(output.dtype)
        output.zero_()

    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen4_exp_amd_ple_ngram_embedding",
        fake_lookup,
    )
    output = module(
        torch.tensor([1, 2]),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([[0]]),
    )

    assert observed_dtypes == [torch.float8_e4m3fn]
    assert output.dtype == torch.float8_e4m3fn
