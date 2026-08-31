# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from types import SimpleNamespace

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(log_stats: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
    )


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_adaptive_prefill_control_reacts_once_per_prefill_observation():
    engine = SimpleNamespace(
        prefill_schedule_adaptive_target_ms=250.0,
        prefill_schedule_adaptive_update_interval=1,
        prefill_schedule_adaptive_max_interval=128,
        prefill_schedule_adaptive_min_token_budget=816,
        prefill_schedule_high_load_interval=24,
        _adaptive_prefill_interval=24,
        _adaptive_prefill_max_budget=2448,
        _adaptive_prefill_budget=2448,
        _adaptive_prefill_latencies=deque(maxlen=128),
        _adaptive_decode_latencies=deque([30.0] * 8, maxlen=128),
        _adaptive_prefill_observations=0,
        _adaptive_prefill_healthy_windows=0,
        _adaptive_prefill_p99_ms=None,
        _adaptive_decode_p99_ms=30.0,
        _adaptive_prefill_penalty_p99_ms=None,
        _last_prefill_service_time=0.0,
    )
    output = SimpleNamespace(
        high_prefill_load=True,
        scheduled_timestamp=time.monotonic() - 0.6,
        scheduled_prefill_tokens=816,
        model_step_elapsed_ms=0.0,
    )
    EngineCore._observe_prefill_control(engine, output)

    assert engine._adaptive_prefill_p99_ms >= 500
    assert engine._adaptive_prefill_penalty_p99_ms >= 470
    assert engine._adaptive_prefill_interval == 36
    assert engine._adaptive_prefill_budget == 816

    decode_output = SimpleNamespace(
        high_prefill_load=True,
        scheduled_timestamp=time.monotonic() - 0.03,
        scheduled_prefill_tokens=0,
        model_step_elapsed_ms=0.0,
    )
    for _ in range(16):
        EngineCore._observe_prefill_control(engine, decode_output)
    assert engine._adaptive_prefill_interval == 36


def test_adaptive_prefill_control_forces_aged_service():
    engine = SimpleNamespace(
        prefill_schedule_interval=4,
        prefill_schedule_high_load_interval=128,
        prefill_schedule_adaptive_target_ms=250.0,
        prefill_schedule_adaptive_max_wait_seconds=10.0,
        _adaptive_prefill_interval=256,
        _adaptive_high_load=True,
        _last_prefill_service_time=time.monotonic() - 11.0,
        _prefill_schedule_step=7,
    )
    assert not EngineCore._should_throttle_prefills(engine, high_load=True)
    assert time.monotonic() - engine._last_prefill_service_time < 1.0


def test_attach_kv_cache_stats_without_request_output():
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            get_kv_cache_block_counts=lambda: (20, 5, 75, 100)
        ),
        _adaptive_prefill_interval=48,
        _adaptive_prefill_budget=816,
        _adaptive_prefill_p99_ms=410.0,
    )
    outputs = {}
    EngineCore._attach_kv_cache_stats(engine, outputs)

    assert outputs[0].kv_cache_usage == 0.75
    assert outputs[0].kv_cache_free_blocks == 25
    assert outputs[0].kv_cache_immediate_free_blocks == 20
    assert outputs[0].kv_cache_evictable_blocks == 5
    assert outputs[0].kv_cache_pinned_blocks == 75
    assert outputs[0].kv_cache_total_blocks == 100
    assert outputs[0].prefill_control_interval == 48
