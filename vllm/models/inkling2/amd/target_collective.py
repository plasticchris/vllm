# SPDX-License-Identifier: Apache-2.0
"""Small target reductions only; embedding/draft use the existing TP route."""
import torch

from vllm import envs
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce


def _peer_or_regular(x: torch.Tensor) -> torch.Tensor:
    tp = get_tp_group()
    peer = tp.inkling_peer_push
    if peer is not None and peer.accepts(x):
        return peer.reduce(x)
    return tp.all_reduce(x)


# No extra Python dispatch on the default-off path. As with all launcher
# settings, changing the flag requires a new engine and new graphs.
target_all_reduce = (
    _peer_or_regular if envs.VLLM_ROCM_INKLING_PEER_PUSH_MAX_ROWS
    else tensor_model_parallel_all_reduce
)
