# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of gfx1100 large-prefill launch selection."""
import pytest
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w8a16 import (
    _w8a16_prefill_config,
)


@pytest.mark.parametrize('m,n,k,gfx,expected', [
    (1, 128, 256, True, (64, 64, 32, 4, 2)),
    (128, 256, 128, True, (64, 128, 32, 4, 2)),
    (129, 256, 128, True, (64, 128, 32, 8, 3)),
    (1024, 256, 128, True, (64, 128, 32, 8, 3)),
    (1025, 256, 128, True, (64, 128, 32, 4, 2)),
    (4800, 4352, 5120, True, (64, 128, 32, 4, 2)),
    (4800, 4352, 5120, False, (128, 128, 32, 4, 2)),
])
def test_w8a16_prefill_config(m,n,k,gfx,expected):
    assert _w8a16_prefill_config(m,n,k,gfx) == expected
