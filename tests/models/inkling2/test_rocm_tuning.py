"""CPU-only launch-policy tests; no GPU initialization needed."""

from vllm.models.inkling2.amd.ops.tuning import (
    attention_launch_config,
    expert_launch_config,
)


def test_small_and_local_attention_stay_unsplit():
    for q, n, window in ((8, 1024, 0), (256, 1048576, 0), (8, 1048576, 1024)):
        assert attention_launch_config(q, 1, q, n, 6, 2, 128, window) == (1, None)


def test_large_prefill_uses_measured_narrow_k_tile():
    for window in (0, 1024):
        assert attention_launch_config(3072, 1, 3072, 1048576, 6, 2, 128, window) == (
            1,
            (64, 16, 4),
        )


def test_long_decode_geometry_and_budget():
    for q, n, expected in (
        (1, 131072, 64),
        (1, 1048576, 128),
        (8, 131072, 128),
        (8, 1048576, 128),
    ):
        s, cfg = attention_launch_config(q, 1, q, n, 6, 2, 128, 0)
        assert s == expected
        assert cfg == (16 if q == 1 else 32, 32, 4)
        assert s * q * 6 * 129 * 4 <= 32 * 2**20
    assert attention_launch_config(2048, 256, 8, 1048576, 6, 2, 128, 0) == (1, None)


def test_expert_up_down_alignment_policy():
    for t in (1, 8, 128, 511, 512, 2048):
        for nv in (False, True):
            up, fast_up = expert_launch_config(t, 2304, nv)
            down, fast_down = expert_launch_config(t, 1152, nv)
            assert up[0] == down[0], "shared EP alignment must use the same BM"
            assert fast_up == fast_down == nv
            assert up[1] == down[1] == 64
            assert up[0] == (64 if nv and t >= 512 else 32 if t >= 512 else 16)
