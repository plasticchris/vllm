# P5-3 regression tests: eb mode must be immune to the overlap-adaptive
# budget pin (quest eb-prefill-between-turns.md, WATCH PART 2).
#
# Era D live bug (2026-10-21): when the overlap-mode adaptive controller
# pins `_adaptive_prefill_budget` to its 408 floor
# (PREFILL_ADAPTIVE_MIN_TOKEN_BUDGET, at 18:13:43 under a cold
# 100-170k-tok turn storm), the engine's `_current_prefill_token_budget()`
# returned 408 EVEN IN EB MODE — the adaptive controller and the budget
# path had no eb gate. `step()`/`step_with_batch_queue()` handed 408 into
# `schedule()`; a forced prefill chunk capped at 408 < 800 (mamba block)
# starting at a block-aligned computed position hit the FIRST
# `_mamba_block_aligned_split` call (subblock_tokens=None — the
# decode-active 100 sub-block applies only on the SECOND call), which
# floors the chunk at the current 800 boundary -> 0 -> the request is
# skipped -> force-serve steps scheduled ZERO prefill tokens while decode
# ran at ~42 ms/step, and the oldest deferred prefill aged unbounded
# (44 live first-serve waits of 21-190 s, 18:17 -> ongoing; first-serve
# only landed in an idle window). P5-1's "forced id MUST be scheduled
# this step" was bypassed by this adjacent mechanism.
#
# Fix: `_current_prefill_token_budget` returns the CONFIGURED
# `decode_active_prefill_token_budget` (3072) in eb mode — the launcher
# contract already scopes the adaptive knobs to overlap ("in eb mode
# cadence/budget control is out of scope; aging is the only cadence").
# The overlap path is byte-identical (second test pins it).
#
# The pre-P5-3 signature (pf=0 for 40+ force-serve steps) is reproduced
# HERE through the REAL engine budget path: the harness pins
# `_adaptive_prefill_budget = 408` in `_observe_prefill_control`-style
# state and calls `core._current_prefill_token_budget(...)` exactly the
# way `step()` does. (P5-1's engine stub hardcoded 3072 in its
# schedule() calls, which is why the P5-1 suite missed this.) Pre-fix,
# `test_eb_force_serve_with_pinned_408_budget` FAILS ("cold0 never
# first-served in 40 force-serve steps"); post-fix it passes with the
# 100-tok trickle per force-serve step.
import os
import time

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


def _make_eb_pinned_core(scheduler, mode):
    """Engine stub in the era-D 18:13:43 pinned state: the overlap
    adaptive controller has driven _adaptive_prefill_budget down to its
    408 floor (min token budget) and it stays there (no relaxation
    window under the turn storm). `mode` selects eb vs overlap."""
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    # Configured decode-active budget (EngineCore.__init__ normally copies
    # this from scheduler_config; the stub bypasses __init__).
    core.decode_active_prefill_token_budget = (
        scheduler.scheduler_config.decode_active_prefill_token_budget
    )
    core.prefill_schedule_mode = mode
    core.prefill_schedule_eb_age_seconds = 10.0
    core.prefill_schedule_eb_idle_subblock = 400
    core.prefill_schedule_high_load_interval = None
    core.decode_active_prefill_high_load_token_budget = None
    core._prefill_schedule_step = 0
    core._eb_forced_prefill_id = None
    core._adaptive_oldest_prefill_wait_seconds = 0.0
    core._adaptive_deferred_prefill_count = None
    core._adaptive_aging_interval = None
    core._adaptive_aging_token_budget = None
    core._adaptive_high_load = False
    core._adaptive_control_tier = None
    # The pin (what _observe_prefill_control leaves behind):
    core.prefill_schedule_adaptive_target_ms = 2000.0
    core.prefill_schedule_adaptive_min_token_budget = 408
    core._adaptive_prefill_max_budget = 3072
    core._adaptive_prefill_budget = 408  # pinned at the 408 floor
    return core


def _make_live_shape_scheduler():
    """Live box shape (era D): MTP spec-2, 6144 batched tokens, 4896/816
    long-prefill thresholds, 3072 decode-active budget, 800 block,
    prefix cache, eagle flags as in the live server."""
    scheduler = create_scheduler(
        max_num_seqs=12,
        max_num_batched_tokens=6144,
        long_prefill_token_threshold=4896,
        decode_active_long_prefill_token_threshold=816,
        decode_active_prefill_token_budget=3072,
        block_size=800,
        max_model_len=32768,
        num_speculative_tokens=2,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.use_eagle = True
    scheduler.num_prefill_lookahead = 2
    (decode_req,) = create_requests(
        num_requests=1, num_tokens=25000, req_ids=["dec0"], max_tokens=512,
        ignore_eos=True, block_size=800,
    )
    scheduler.add_request(decode_req)
    # Prefill dec0 fully, then one decode step -> RUNNING decode.
    while decode_req.is_prefill_chunk or decode_req.num_output_tokens < 1:
        out = scheduler.schedule(False, 3072, 100, None)
        assert out.num_scheduled_tokens, "dec0 never scheduled"
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=list(out.num_scheduled_tokens),
                req_id_to_index={r: i for i, r in
                                 enumerate(out.num_scheduled_tokens)},
                sampled_token_ids=[[7] for _ in out.num_scheduled_tokens],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    assert not decode_req.is_prefill_chunk
    # cold0: same session's next turn: 24800 shared (block-aligned prefix
    # hit) + 4800 new tokens, already aged past k=10 at add time.
    (cold,) = create_requests(
        num_requests=1, num_tokens=20000, req_ids=["cold0"], max_tokens=8,
        block_size=800,
    )
    cold.prompt_token_ids = [0] * 24800 + [1] * 4800
    return scheduler, cold


def test_eb_force_serve_with_pinned_408_budget():
    """P5-3: eb force-serve survives the overlap adaptive pin. The engine
    budget path must yield the CONFIGURED 3072 (not the pinned 408) in
    eb mode, so the forced prefill trickles in 100-tok sub-blocks (the
    decode-active base sub-block) instead of the era-D 0-tok no-op."""
    scheduler, cold = _make_live_shape_scheduler()
    now0 = time.time()
    cold.arrival_time = now0 - 20.0  # already past k
    scheduler.add_request(cold)

    core = _make_eb_pinned_core(scheduler, mode="eb")
    core._eb_idle_subblock_enabled = True  # EngineCore (non-DP) process

    first_serve_step = None
    for s in range(1, 41):
        cold.arrival_time = now0 - 20.0 - s  # age 1 s per step
        # Exactly the step() call sequence:
        high_load = False
        overlap = scheduler.has_decode_prefill_overlap()
        throttle = core._should_throttle_prefills(high_load, overlap)
        forced = core._eb_forced_prefill_id
        # The REAL engine budget path (pre-P5-3 this returned the pinned
        # 408 in eb mode; that is the bug under test).
        budget = core._current_prefill_token_budget(high_load, overlap)
        assert budget == 3072, (
            "eb mode must use the configured 3072 budget, not the "
            "overlap-adaptive pinned %r (era D 18:13:43: 4/3072 -> 14/408)"
            % (budget,))
        out = scheduler.schedule(throttle, budget, 100, forced)
        if not throttle:
            # P5-1 + P5-3 combined: EVERY force-serve step makes progress
            # — the era-D no-op signature is pf=0 with a forced id (or a
            # running forced prefill) stuck at a block boundary.
            assert out.scheduled_prefill_tokens >= 100, (
                "force-serve step %d scheduled %d prefill tokens (era D "
                "live: pf=0 for 40+ steps while decode ran at 42 ms/step)"
                % (s, out.scheduled_prefill_tokens))
        if cold.request_id in out.num_scheduled_tokens and \
                first_serve_step is None:
            first_serve_step = (s, out.num_scheduled_tokens[cold.request_id])
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=list(out.num_scheduled_tokens),
                req_id_to_index={r: i for i, r in
                                 enumerate(out.num_scheduled_tokens)},
                sampled_token_ids=[[7] for _ in out.num_scheduled_tokens],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if first_serve_step is not None and s > first_serve_step[0] + 3:
            break
    assert first_serve_step is not None, (
        "cold0 never first-served in 40 force-serve steps — the P5-3 "
        "no-op (era D live: 143.8 s, kv==0 throughout)")
    s, toks = first_serve_step
    assert s <= 12, (
        "first serve at step %d (wait ~%d s) — beyond k+2 (era D live: "
        "143.8 s)" % (s, s - 1))
    assert toks >= 100, (
        "first-serve chunk %d < one decode-active sub-block (100)" % toks)


def test_overlap_budget_pin_unchanged():
    """Pin: the fix must not leak into overlap mode (byte-identical
    regression contract) or into the eb-inert DP proc path. With the
    same pinned state, overlap receives the adaptive 408 (overlap=True)
    or falls back to the configured 3072 (overlap=False); eb mode with
    the DP-inert flag (_eb_idle_subblock_enabled=False, as set on
    DPEngineCoreProc) keeps the pre-P5-3 overlap behavior."""
    scheduler, _cold = _make_live_shape_scheduler()

    overlap_core = _make_eb_pinned_core(scheduler, mode="overlap")
    assert scheduler.scheduler_config.decode_active_prefill_token_budget == 3072
    # Overlap, adaptive on, decode-active (overlap flag True): the PINNED
    # adaptive budget — exactly what pre-P5-3 overlap always got.
    assert overlap_core._current_prefill_token_budget(False, True) == 408
    # Overlap, no overlap: the configured budget (pre-P5-3 fallback).
    assert overlap_core._current_prefill_token_budget(False, False) == 3072

    dp_core = _make_eb_pinned_core(scheduler, mode="eb")
    dp_core._eb_idle_subblock_enabled = False  # DPEngineCoreProc
    # The DP proc is eb-inert (rank-aligned throttle override): it keeps
    # the overlap budget behavior, so the pin still applies there.
    assert dp_core._current_prefill_token_budget(False, True) == 408
