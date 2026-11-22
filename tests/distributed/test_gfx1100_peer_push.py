# SPDX-License-Identifier: Apache-2.0
"""CPU contract, independent from the four-rank hardware check."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators.gfx1100_peer_push import (
    Gfx1100PeerPush,
    _load_native,
    check_peer_push_fault,
)
from vllm.v1.worker.gpu.async_utils import AsyncOutput


@pytest.mark.parametrize("rows,accepted8,accepted32", [(1, True, True), (8, True, True),
                                                       (32, False, True), (7, False, False)])
def test_shape_gate(rows, accepted8, accepted32):
    peer = object.__new__(Gfx1100PeerPush)
    peer.device = torch.device("cpu")
    x = torch.empty(rows, 2304, dtype=torch.bfloat16)
    peer.max_rows = 8
    assert peer.accepts(x) == accepted8
    peer.max_rows = 32
    assert peer.accepts(x) == accepted32
    assert not peer.accepts(x.float())
    assert not peer.accepts(x.view(-1))
    assert not peer.accepts(x[:, ::2])
    assert not peer.accepts(torch.empty(rows, 4608, dtype=torch.bfloat16)[:, ::2])


def test_missing_native_refuses(tmp_path: Path):
    with pytest.raises(RuntimeError, match="build_gfx1100_peer_push"):
        _load_native(tmp_path / "absent.so")


def test_fault_gate_waits_before_touching_result():
    check_peer_push_fault(torch.zeros(4, dtype=torch.int32))
    # No sampled-token fields: the sticky code must reject this output after
    # its copy event but before touching any potentially corrupted data.
    out = object.__new__(AsyncOutput)
    out.copy_event = Mock()
    out._inkling_peer_fault = torch.tensor([0, 0, 2049, 0], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Fault detected in Inkling gfx1100 TP4 peer push"):
        out.get_output()
    out.copy_event.synchronize.assert_called_once_with()
