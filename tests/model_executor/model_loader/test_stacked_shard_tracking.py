# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stacked parameter that received only some of its shards must be reported.

`track_weights_loading` diffs parameter names, so `qkv_proj.weight` counts as
loaded once any one shard is written. These tests pin the shard-level check
that closes that gap for `QKVParallelLinear` and `MergedColumnParallelLinear`.
"""

import pytest
import torch
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.amd.hyperconnection import GatedResidual
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

HIDDEN = 8
HEAD = 2

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("dist_init")]


def shard(shape: tuple[int, ...], shard_id) -> torch.Tensor:
    weight = torch.ones(shape)
    weight.shard_id = shard_id
    return weight


def load(layer: nn.Module, *weights: tuple[str, torch.Tensor]) -> set[str]:
    return set(layer.load_weights(weights))


def make_qkv(total_num_heads: int = 2) -> QKVParallelLinear:
    return QKVParallelLinear(HIDDEN, HEAD, total_num_heads, total_num_kv_heads=1)


Q = ("weight", shard((2 * HEAD, HIDDEN), "q"))
K = ("weight", shard((HEAD, HIDDEN), "k"))
V = ("weight", shard((HEAD, HIDDEN), "v"))


def test_qkv_reports_shards_the_checkpoint_never_filled():
    layer = make_qkv()
    load(layer, Q)
    assert layer.pop_missing_shards() == ["weight.k", "weight.v"]

    load(layer, Q, K, V)
    assert layer.pop_missing_shards() == []


def test_qkv_stacked_on_disk_fills_every_shard():
    layer = make_qkv()
    load(layer, ("weight", torch.ones(4 * HEAD, HIDDEN)))
    assert layer.pop_missing_shards() == []


def test_record_is_per_load_so_a_reload_cannot_coast_on_the_first():
    layer = make_qkv()
    load(layer, Q, K, V)
    assert layer.pop_missing_shards() == []

    load(layer, K, V)
    assert layer.pop_missing_shards() == ["weight.q"]


def test_each_parameter_is_checked_on_its_own():
    """A complete weight must not hide a bias that is missing a shard."""
    layer = make_qkv()
    load(layer, Q, K, V, ("bias", shard((2 * HEAD,), "q")))
    assert layer.pop_missing_shards() == ["bias.k", "bias.v"]


def test_shard_with_no_rows_is_never_expected():
    """Cross-attention kv_proj is a QKV layer with zero query heads."""
    layer = make_qkv(total_num_heads=0)
    load(layer, K, V)
    assert layer.pop_missing_shards() == []


def test_unknown_shard_id_is_rejected():
    with pytest.raises(ValueError, match="should be one of"):
        make_qkv().validate_shard_id("x")


def test_merged_column_tuple_shard_is_one_span():
    layer = MergedColumnParallelLinear(HIDDEN, [4, 4, 4], bias=False)
    load(layer, ("weight", shard((8, HIDDEN), (0, 1))))
    assert layer.pop_missing_shards() == ["weight.2"]

    load(
        layer, ("weight", shard((8, HIDDEN), (0, 1))), ("weight", shard((4, HIDDEN), 2))
    )
    assert layer.pop_missing_shards() == []


@pytest.mark.parametrize("missing_injection", [False, True])
def test_hc_padding_is_initialized_without_hiding_real_missing_weights(
    missing_injection,
):
    cfg = HyperConnectionConfig(
        hidden_size=4,
        hc_count=4,
        hc_lowrank=8,
        params_dtype=torch.float32,
        hc_per_branch_norm=True,
    )
    layer = GatedResidual(cfg)
    projection = layer.input_mix_weight_down_block_inject
    projection.weight.data.fill_(float("nan"))
    weights = [
        ("hc_norm.weight", torch.zeros(16)),
        ("input_mix_weight_up.weight", torch.ones(16, 8)),
        ("input_mix_weight_down_block_inject.weight", shard((8, 16), 0)),
    ]
    if not missing_injection:
        weights.append(("input_mix_weight_down_block_inject.weight", shard((4, 16), 1)))
    loaded = AutoWeightsLoader(layer).load_weights(iter(weights))
    tracker = DefaultModelLoader(LoadConfig())
    if missing_injection:
        with pytest.raises(ValueError, match=r"weight\.1"):
            tracker.track_weights_loading(layer, loaded, quantized=True)
    else:
        tracker.track_weights_loading(layer, loaded, quantized=True)
        assert torch.equal(projection.weight[:12], torch.ones(12, 16))
        assert torch.count_nonzero(projection.weight[12:]) == 0
        # A second load must initialize padding again, not inherit the first
        # load's tracking record or an accidentally overwritten padding row.
        projection.weight.data[12:].fill_(float("nan"))
        loaded = AutoWeightsLoader(layer).load_weights(iter(weights))
        tracker.track_weights_loading(layer, loaded, quantized=True)
        assert torch.count_nonzero(projection.weight[12:]) == 0


def test_minimax_m3_qkv_with_indexer_loads_all_shards():
    layer = MinimaxM3QKVParallelLinearWithIndexer(
        HIDDEN,
        HEAD,
        total_num_heads=2,
        total_num_kv_heads=1,
        total_num_index_heads=1,
        index_head_size=HEAD,
    )
    index_q = ("weight", shard((HEAD, HIDDEN), "index_q"))
    index_k = ("weight", shard((HEAD, HIDDEN), "index_k"))
    load(layer, Q, K, V, index_q, index_k)
    assert layer.pop_missing_shards() == []


def test_loader_lists_missing_shards_as_weights_not_loaded():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.qkv_proj = make_qkv()
    loaded = {f"layer.qkv_proj.{name}" for name in load(model.layer.qkv_proj, Q)}

    loader = DefaultModelLoader(LoadConfig())
    with pytest.raises(ValueError, match="not initialized from checkpoint") as exc:
        loader.track_weights_loading(model, loaded)
    assert "'layer.qkv_proj.weight.k'" in str(exc.value)
    assert "'layer.qkv_proj.weight.v'" in str(exc.value)
