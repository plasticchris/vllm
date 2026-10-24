# SPDX-License-Identifier: Apache-2.0
"""Positional telemetry contract between the LLMEngine and the serving layer.

The in-process serving endpoint (rukhm) reads a handful of per-step /
per-request telemetry attributes off the engine every step and turns them
into /status + /advisory JSON. Those attributes were historically plain
``tuple``s whose field order was documented by hand in *both* trees (a
position-9 comment in llm_engine.py, a matching 13/21-int comment in the
serving layer) -- a cross-boundary contract that silently corrupts whenever
either side reorders a field.

This module is the single source of truth. ``LLMEngine`` now publishes these
as ``NamedTuple`` instances; the serving layer consumes them by *field name*
(``step_stats.gen_tokens``), so a reordering becomes a type error / attribute
error instead of a silent mislabel. The field order below is fixed and
backwards-compatible with the old positional tuples.

The serving layer imports these classes (cheap: no torch, only stdlib + typing)
to construct fixtures and assert the field set. Construct with keywords at the
engine side; read attributes at the serving side.
"""

from typing import NamedTuple


class StepStats(NamedTuple):
    """Per-scheduler-step breakdown (13 ints, fixed order).

    Every field defaults to 0 so a scheduled-but-no-outputs step (e.g. a
    KV-pressure stall) can be published with only the scheduler's own
    bookkeeping and the rest zeroed.
    """

    gen_tokens: int = 0
    prefill_sched_tokens: int = 0
    scheduled_total_tokens: int = 0
    prompt_computed: int = 0
    prompt_cached: int = 0
    preempted_reqs: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    kv_evictions: int = 0
    spec_draft_tokens: int = 0
    spec_accepted_tokens: int = 0
    running_reqs: int = 0
    waiting_reqs: int = 0


class KVCacheStats(NamedTuple):
    """KV-cache capacity snapshot (1 float + 8 ints, fixed order).

    Position 0 is the 0.0-1.0 usage fraction; positions 4/8 are the total
    block / total token counts used to derive the scheduler block size and
    the prefix-cache index cap.
    """

    usage: float
    immediate_free_blocks: int
    evictable_blocks: int
    pinned_blocks: int
    total_blocks: int
    immediate_free_tokens: int
    evictable_tokens: int
    pinned_tokens: int
    total_tokens: int


class PrefillControl(NamedTuple):
    """Prefill-control policy telemetry (21 fields, fixed order).

    Position 9 (``deferred_prefill_count``) is the EB(0) deferred-prefill
    count. ``decode_visible_p99_ms`` is the SLO p99 (decode-visible latency),
    carried on the wire as ``prefill_control_slo_p99_ms``.
    """

    interval: int | None = None
    token_budget: int | None = None
    prefill_step_p99_ms: float | None = None
    decode_step_p99_ms: float | None = None
    prefill_penalty_p99_ms: float | None = None
    drain_p99_ms: float | None = None
    drain_penalty_p99_ms: float | None = None
    decode_visible_p99_ms: float | None = None
    oldest_prefill_wait_seconds: float | None = None
    deferred_prefill_count: int | None = None
    aging_interval: int | None = None
    aging_token_budget: int | None = None
    high_load: bool | None = None
    cold_start: bool | None = None
    observations: int | None = None
    reset_count: int | None = None
    quiet_seconds: float | None = None
    tier: str | None = None
    mamba_prefill_subblock_tokens: int | None = None
    prefill_ms_per_token: float | None = None
    decode_gap_p99_ms: float | None = None
