# SPDX-License-Identifier: Apache-2.0
# CPU regression for the INT8 Qwen3.8-27B MTP K=2 graph dispatch ladder.
# The original eager-only diagnosis was retracted: live run-mode/replay logs
# confirmed FULL replay. This checks descriptor selection, not GPU execution.
#
# Run:  /mnt/hoard/vllm/venv/bin/python -m pytest \
#          tests/v1/worker/test_int8_decode_cudagraph_dispatch.py -x -v
#
# Backend support classes: the Qwen3.5/3.8 GDN hybrid uses GDNAttentionBackend
# (UNIFORM_BATCH) for its linear-attention layers; rocm FA is ALWAYS. Both must
# keep FULL decode. A linear_attn-class backend (UNIFORM_SINGLE_TOKEN_DECODE)
# legitimately demotes spec-decode FULL -> PIECEWISE, so that case asserts the
# demotion instead.
#
# ngram, not mtp: cudagraph dispatch keys only off num_speculative_tokens and
# uses_dynamic_speculative_decoding(), never the method. (prod: no
# num_speculative_tokens_per_batch_size => dynamic SD is False, flat K=2.)

import types

import pytest
import torch

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport

MODEL = "/mnt/fastscratch/qwen3.8-27b-int8-w8a16-local"
CAPTURE_SIZES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 21, 24, 27, 30, 32, 33, 36]
MAXSEQS = 4
NUM_SPEC = 2  # prod SPEC={"method":"mtp","num_speculative_tokens":2}
DQL = NUM_SPEC + 1  # uniform_decode_query_len = 3


def build_cfg() -> VllmConfig:
    mc = ModelConfig(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        max_model_len=262144,
        trust_remote_code=True,
    )
    pc = ParallelConfig(tensor_parallel_size=4)
    sc = SchedulerConfig(
        max_num_seqs=MAXSEQS, max_model_len=262144, is_encoder_decoder=False
    )
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=NUM_SPEC)
    cc = CompilationConfig(
        mode=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=list(CAPTURE_SIZES),
    )
    return VllmConfig(
        model_config=mc,
        parallel_config=pc,
        scheduler_config=sc,
        speculative_config=spec,
        compilation_config=cc,
        cache_config=CacheConfig(cache_dtype="bfloat16"),
    )


@pytest.fixture
def cpu_manager(monkeypatch):
    """CudaGraphManager on CPU: stub the PP group + global graph pool."""

    import vllm.v1.worker.gpu.cudagraph_utils as cu

    pp = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(cu, "get_pp_group", lambda: pp)
    monkeypatch.setattr(
        cu.current_platform,
        "get_global_graph_pool",
        classmethod(lambda cls: None).__get__(type(cu.current_platform)),
    )

    def make(min_cg_support: AttentionCGSupport):
        cfg = build_cfg()
        resolved = cfg.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            "GDNAttentionMetadata",
            uniform_decode_query_len=DQL,
            max_num_reqs=MAXSEQS,
        )
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        mgr = ModelCudaGraphManager(
            cfg, torch.device("cpu"), resolved, decode_query_lens={DQL: MAXSEQS}
        )
        # capture() normally flips this; dispatch() consults it.
        mgr._graphs_captured = True
        return resolved, mgr

    return make


@pytest.mark.parametrize(
    "min_cg_support, expect_decode",
    [
        (AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.FULL),
        (AttentionCGSupport.ALWAYS, CUDAGraphMode.FULL),
        (AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, CUDAGraphMode.PIECEWISE),
    ],
)
def test_resolved_mode(cpu_manager, min_cg_support, expect_decode):
    resolved, _ = cpu_manager(min_cg_support)
    assert resolved.decode_mode() == expect_decode, (
        f"{min_cg_support} resolved decode to {resolved.decode_mode()} "
        f"(expected {expect_decode})"
    )


def test_manager_captures_full_decode_ladder_and_dispatches(cpu_manager):
    resolved, mgr = cpu_manager(AttentionCGSupport.UNIFORM_BATCH)
    assert resolved.decode_mode() == CUDAGraphMode.FULL

    full = [
        d
        for mode, ds in mgr._capture_descs.items()
        if mode == CUDAGraphMode.FULL
        for d in ds
    ]
    sizes = sorted({d.num_tokens for d in full})
    assert sizes == [3, 6, 9, 12], (
        f"expected FULL decode ladder [3,6,9,12] for MAXSEQS={MAXSEQS} "
        f"K={NUM_SPEC}, got {sizes} (descriptors: {full})"
    )

    # Live shapes: 1..4 decode reqs x 3 tokens, uniform decode, no prefill.
    for reqs in (1, 2, 3, 4):
        d = mgr.dispatch(reqs, reqs * 3, DQL, 0, max_query_len=DQL)
        assert d.cg_mode == CUDAGraphMode.FULL, (
            f"dispatch(reqs={reqs}, toks={reqs * 3}, uni={DQL}) -> {d.cg_mode} "
            f"(candidates @ key: {mgr._candidates.get((reqs * 3, 0))})"
        )
        assert d.num_tokens == reqs * 3

    # A prefill-containing batch must NOT match the uniform decode graphs.
    d = mgr.dispatch(2, 100, None, 0, max_query_len=None)
    assert d.cg_mode != CUDAGraphMode.FULL, (
        "prefill batch must not replay a uniform decode graph"
    )
