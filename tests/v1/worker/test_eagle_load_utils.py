# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.config.load import LoadConfig
from vllm.v1.worker.gpu.spec_decode.eagle import utils


class _DraftLoadObserved(Exception):
    pass


def test_load_eagle_model_uses_draft_load_config(monkeypatch) -> None:
    draft_model_config = object()
    draft_load_config = LoadConfig(load_format="auto")
    speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
        draft_load_config=draft_load_config,
        kv_cache_dtype=None,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    def observe_get_model(**kwargs):
        assert kwargs["vllm_config"] is vllm_config
        assert kwargs["model_config"] is draft_model_config
        assert kwargs["load_config"] is draft_load_config
        raise _DraftLoadObserved

    monkeypatch.setattr(utils, "get_model", observe_get_model)

    with pytest.raises(_DraftLoadObserved):
        utils.load_eagle_model(nn.Module(), vllm_config)
