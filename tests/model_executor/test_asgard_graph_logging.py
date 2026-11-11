# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real logging formatter without constructing a GPU runner."""

import ast
import logging
from pathlib import Path

import pytest


@pytest.mark.parametrize("requests", [None, 0, 1, 4])
def test_token_only_graph_descriptor_can_be_logged(requests):
    source = Path(__file__).resolve().parents[2] / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(source.read_text())
    formats = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("[asgard-cg] run step")
    ]
    assert len(formats) == 1
    # PIECEWISE descriptors can carry tokens without a fixed request count.
    # Python's logging emits a traceback instead of the diagnostic if its
    # format string requires an integer for that None.
    record = logging.LogRecord(
        "runner",
        logging.INFO,
        str(source),
        0,
        formats[0],
        (1, "PIECEWISE", 3, requests, 3, False, False),
        None,
    )
    assert f"reqs={requests} " in record.getMessage()
