# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leyline must not bypass upstream's cache-read/prompt-scoring limits."""

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.leyline_splice import prepare_splice
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("limit", [0, 16, 48])
def test_splice_rejects_bounded_prefix_cache_reads(limit):
    request = Request(
        request_id="bounded",
        prompt_token_ids=list(range(64)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_splice_mode": "amortize",
                "kv_splice_directives": [(16, 32, 0, 32)],
                "kv_splice_donor_tokens": list(range(80)),
            },
        ),
        pooling_params=None,
    )
    request.prefix_cache_hit_limit = limit
    manager = SimpleNamespace(
        enable_caching=True,
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
    )
    assert prepare_splice(manager, request) is None
    assert request.kv_splice_stats["fallback"] == "caching_disabled"
    assert request.leyline_splice_plan is None


def test_unbounded_request_continues_to_donor_validation():
    request = Request(
        request_id="unbounded",
        prompt_token_ids=list(range(64)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_splice_mode": "amortize",
                "kv_splice_directives": [(16, 32, 0, 32)],
            },
        ),
        pooling_params=None,
    )
    manager = SimpleNamespace(
        enable_caching=True,
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
    )
    assert request.prefix_cache_hit_limit is None
    assert prepare_splice(manager, request) is None
    assert request.kv_splice_stats["fallback"] == "no_donor"
