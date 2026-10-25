# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.outputs import RequestError
from vllm.utils.math_utils import cdiv
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.leyline_splice import cancel_splice, commit_splice, prepare_splice
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    KV_FETCH_COMPLETED_WAITING,
    KV_FETCH_IN_PROGRESS,
    KV_FETCH_STAGES,
    KV_FETCH_WAITING_TO_START,
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.utils import strip_speculative_padding
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


def clamp_mamba_subblock_to_block(block_size: int, requested: int) -> int:
    """Largest divisor of ``block_size // 2`` that is <= ``requested``.

    The Mamba cache block size shifts with the speculative depth: the
    mamba conv state grows one row per draft token and the attention
    block is padded to whole attention pages to hold the mamba page, so
    a base tuned for one depth (e.g. 408 for the 816-token block at
    K=4) stops dividing the block when the depth changes (the 800-token
    block at flat K=2). Clamping to a divisor of block/2 keeps the base
    and both adaptive tiers (base/2, base*2) dividing the cache block.
    """
    half = block_size // 2
    for candidate in range(min(requested, half), 0, -1):
        if half % candidate == 0:
            return candidate
    return 1


class _TsvTrace:
    def __init__(self, path: str | None, columns: str, flush_interval: float = 1):
        self.path = path
        self.columns = columns
        self.file = None
        self.flush_interval = flush_interval
        self.last_flush = time.monotonic()

    def write(self, *values: object) -> None:
        if self.path is None:
            return
        if self.file is None:
            self.file = open(self.path, "a")
            self.file.write(f"# start={time.time():.6f}\t{self.columns}\n")
        self.file.write(f"{time.time():.6f}\t" + "\t".join(map(str, values)) + "\n")
        now = time.monotonic()
        if now - self.last_flush >= self.flush_interval:
            self.file.flush()
            self.last_flush = now

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


_spec_trace = _TsvTrace(
    os.environ.get("VLLM_SPEC_TRACE"),
    "timestamp\tstep\tbatch_size\trequest_id\tdrafted\taccepted",
)
_prefix_trace = _TsvTrace(
    os.environ.get("VLLM_PREFIX_TRACE"),
    "timestamp\tstep\trequest_id\tqueried\thits\tpreempted",
)
atexit.register(_spec_trace.close)
atexit.register(_prefix_trace.close)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.model_uses_mrope = vllm_config.model_config.uses_mrope
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.spec_decode_metrics_level = (
            self.observability_config.per_request_spec_decode_metrics
        )
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # Admission into RUNNING. Separate from max_num_running_reqs, which
        # is the model-runner slot count.
        self.max_num_active_reqs = (
            self.scheduler_config.max_num_active_seqs
            if self.scheduler_config.max_num_active_seqs is not None
            else self.max_num_running_reqs
        )
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.adaptive_long_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold_adaptive
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Pooling and diffusion models do not append a sampled token after
        # prefill, so they should not reserve one context slot while scheduling.
        self.num_sampled_tokens_per_step = (
            0
            if (
                vllm_config.model_config.runner_type == "pooling"
                or vllm_config.model_config.is_diffusion
            )
            else 1
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # Waiting requests that hold KV blocks are always drained before
        # self.waiting: a block-holder must never sit behind a request whose
        # failed allocation would stop the scheduling scan.
        self.kv_holding_waiting = create_request_queue(self.policy)
        # Waiting requests that are deferred, i.e. were skipped by the scheduling
        # scan or enqueued in a blocked status.
        self.deferred_waiting: set[Request] = set()
        self.running: list[Request] = []
        self._decode_completion_times: dict[str, float] = {}
        self._decode_gap_samples: deque[tuple[float, float]] = deque(maxlen=4096)
        self._decode_gap_window_seconds = max(
            self.scheduler_config.prefill_schedule_adaptive_reset_seconds or 30.0,
            30.0,
        )

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: dict[str, RequestError] = {}
        self.encoder_cache_mismatch_reqs: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = vllm_config.model_config.supports_multimodal_inputs
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is None:
            manager_cls_obj = (
                EncoderDecoderCacheManager
                if self.is_encoder_decoder
                else EncoderCacheManager
            )
        self.encoder_cache_manager = manager_cls_obj.create_manager(
            cache_size=encoder_cache_size, vllm_config=vllm_config
        )
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        # DSV41 SWA bounded replay: groups that declare a replay window are rebuilt
        # after a prefix hit by recomputing its trailing tokens. One window
        # for all such groups, so the rewind matches every group's allocation.
        self.prefix_replay_group_ids = tuple(
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_replay_tokens > 0
        )
        replay_windows = {
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.prefix_replay_tokens
            for group_id in self.prefix_replay_group_ids
        }
        assert len(replay_windows) <= 1, (
            f"Prefix replay windows should agree: {sorted(replay_windows)}"
        )
        self.prefix_replay_tokens = replay_windows.pop() if replay_windows else 0
        self.num_prefill_lookahead = vllm_config.num_prefill_lookahead_tokens
        # The V2 model runner skips drafting in steps where no scheduled request
        # can use the drafts. Only drafters with their own forward pass reserve
        # lookahead slots.
        self.can_skip_drafts = (
            self.num_lookahead_tokens > 0 and vllm_config.use_v2_model_runner
        )
        self.dynamic_sd_lookup: list[int] | None = None
        self.long_context_threshold: int | None = None
        self.long_context_num_spec_tokens: int | None = None
        self.greedy_num_spec_tokens: int | None = None
        self.profitability_aware_min_batch_size: int | None = None
        self.profitability_min_samples = 0
        self.profitability_exploration_interval = 32
        self.profitability_hysteresis = 0.0
        self.profitability_acceptance_threshold: float | None = None
        self._spec_acceptance_history: deque[float] = deque(maxlen=8)
        self._spec_acceptance_request_ids: frozenset[str] = frozenset()
        self._spec_request_profitability_history: dict[int, deque[float]] = {
            2: deque(maxlen=16),
            3: deque(maxlen=16),
        }
        self._last_spec_profitability_acceptance_gated = False
        self._spec_profitability_window = 16
        self._spec_profitability_history: dict[
            tuple[int, int], deque[float]
        ] = {}
        self._spec_profitability_steps: defaultdict[int, int] = defaultdict(int)
        self._last_spec_profitability_batch_size: int | None = None
        self._last_spec_profitability_selected_k: int | None = None
        self._last_spec_profitability_exploration = False
        self._last_spec_profitability_warm_start_batches: list[int] = []
        self.profitability_state_path: str | None = None
        self.profitability_state_save_interval = 16
        self.profitability_state_ttl_seconds = 86400.0
        self.profitability_state_half_life_seconds = 21600.0
        self.profitability_neighbor_batch_radius = 2
        self._profitability_state_fingerprint = ""
        self._spec_profitability_unsaved = 0
        self._spec_profitability_loaded_samples = 0
        self._spec_profitability_state_age_seconds: float | None = None
        if speculative_config is not None:
            self.long_context_threshold = speculative_config.long_context_threshold
            self.long_context_num_spec_tokens = (
                speculative_config.long_context_num_speculative_tokens
            )
            self.greedy_num_spec_tokens = speculative_config.greedy_num_speculative_tokens
            self.profitability_aware_min_batch_size = (
                speculative_config.profitability_aware_min_batch_size
            )
            self.profitability_min_samples = (
                speculative_config.profitability_min_samples
            )
            self.profitability_exploration_interval = (
                speculative_config.profitability_exploration_interval
            )
            self.profitability_hysteresis = (
                speculative_config.profitability_hysteresis
            )
            self.profitability_acceptance_threshold = (
                speculative_config.profitability_acceptance_threshold
            )
            self._spec_acceptance_history = deque(
                maxlen=speculative_config.profitability_acceptance_window
            )
            self._spec_profitability_window = speculative_config.profitability_window
            self._spec_request_profitability_history = {
                2: deque(maxlen=self._spec_profitability_window),
                3: deque(maxlen=self._spec_profitability_window),
            }
            self.profitability_state_path = speculative_config.profitability_state_path
            self.profitability_state_save_interval = (
                speculative_config.profitability_state_save_interval
            )
            self.profitability_state_ttl_seconds = (
                speculative_config.profitability_state_ttl_seconds
            )
            self.profitability_state_half_life_seconds = (
                speculative_config.profitability_state_half_life_seconds
            )
            self.profitability_neighbor_batch_radius = (
                speculative_config.profitability_neighbor_batch_radius
            )
            self._profitability_state_fingerprint = (
                self._build_spec_profitability_fingerprint(
                    vllm_config, speculative_config
                )
            )
            self._load_spec_profitability_state()
            if self.profitability_state_path:
                atexit.register(self._save_spec_profitability_state)
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            self.use_eagle = speculative_config.use_eagle()
            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
            if (
                speculative_config.needs_eagle_block_drop()
                and speculative_config.disable_eagle_block_drop
            ):
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle_block_drop,
            num_prefill_lookahead=self.num_prefill_lookahead,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
            enable_mamba_shared_prefix_checkpoint=(
                self.cache_config.enable_mamba_shared_prefix_checkpoint
            ),
        )
        # Mamba managers already account for their speculative state blocks.
        num_spec_decode_tokens = (
            1 + self.num_spec_tokens + self.num_lookahead_tokens
            if self.num_spec_tokens
            else 0
        )
        self._spec_decode_step_blocks = sum(
            cdiv(num_spec_decode_tokens, manager.block_size)
            for manager in self.kv_cache_manager.coordinator.single_type_managers
            if not isinstance(manager.kv_cache_spec, MambaSpec)
        )
        # Bind after construction so connectors can access the cache manager.
        if self.connector is not None:
            self.connector.bind_kv_cache_manager(self.kv_cache_manager)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # Flag to track whether the last cadence-aligned prefill batch exhausted
        # the token budget before draining the waiting queue. Prefill throttling
        # backs off in this case so sustained admission can keep up.
        self.prefill_capacity_bound = False
        # The DP engine core drives this cadence from a counter shared across
        # ranks; an engine without one never throttles, so apply the interval
        # locally, timed from the last prefill-carrying step. Three regimes:
        #   1. no DP -- local cadence applies;
        #   2. dense DP rank -- `reconfigure_for_independent_dp_rank` already
        #      set data_parallel_size=1, so each rank gets its own cadence.
        #      Intended, but a behaviour change: this used to be a no-op;
        #   3. MoE DP rank (size > 1) -- the engine core drives it, no-op here.
        self.local_prefill_interval = (
            self.scheduler_config.prefill_schedule_interval
            if self.parallel_config.data_parallel_size == 1
            else 1
        )
        if self.local_prefill_interval > 1:
            # Neither the hold nor its release is otherwise visible in logs or
            # stats, which makes a misapplied cadence hard to diagnose.
            logger.info_once(
                "Prefill cadence: applying prefill_schedule_interval=%d "
                "locally (this engine is not DP-coordinated); prefill is held "
                "until %d executed steps have passed since the last "
                "prefill-carrying step.",
                self.local_prefill_interval,
                self.local_prefill_interval,
            )
        # A full interval back, so the first step is never gated.
        self.last_prefill_step = -self.local_prefill_interval
        # Steps that actually scheduled tokens. `current_step` counts every
        # schedule() call including empty ones, so it would expire the hold
        # early; it keeps its own meaning (next_decode_eligible_step).
        self.num_executed_steps = 0
        self._prefill_rr_cursor = 0
        self._idle_prefill_rr_cursor = 0
        self._short_prefill_priority_streak = 0
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # TODO: Support models with multiple Mamba specs that require different
        # prefill checkpoint alignments instead of selecting the first one.
        self.mamba_prefill_checkpoint_alignment = next(
            (
                group.kv_cache_spec.prefill_checkpoint_alignment
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            None,
        )
        requested_subblock = self.scheduler_config.mamba_prefill_subblock_tokens
        if (
            requested_subblock is not None
            and self.block_size % requested_subblock != 0
        ):
            # The block size shifts with the speculative depth (see
            # clamp_mamba_subblock_to_block), so a launcher-tuned base can
            # stop dividing it. Clamp instead of dead-ending startup.
            self.mamba_prefill_subblock_tokens = clamp_mamba_subblock_to_block(
                self.block_size, requested_subblock
            )
            logger.warning(
                "mamba_prefill_subblock_tokens=%d does not divide the Mamba "
                "cache block %d (it shifts with the speculative depth); "
                "clamping to %d",
                requested_subblock,
                self.block_size,
                self.mamba_prefill_subblock_tokens,
            )
        else:
            self.mamba_prefill_subblock_tokens = requested_subblock
        self.mamba_has_prefill_checkpoint_blocks = (
            self.has_mamba_layers
            and not self.use_eagle
            and all(
                not isinstance(group.kv_cache_spec, MambaSpec)
                or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
                for group in kv_cache_config.kv_cache_groups
            )
        )
        # Fine-grained hits reuse the prompt's hash-aligned Mamba checkpoint.
        # Split there only if the backend cannot save it within the final chunk.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
        # Opt-in: also stop at the junction, where an eagle sibling resumes. The
        # manager decides whether it can check-point there (per-group eagle bit,
        # no MTP re-prefill tail); splitting for a stop it would refuse costs a
        # forward pass and displaces the block-boundary stop.
        self.mamba_shared_prefix_checkpoint = (
            self.mamba_partial_cache_hit
            and self.kv_cache_manager.mamba_shared_prefix_checkpoint
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.aux_output_connector = (
            AuxOutputSchedulerConnector(self.kv_cache_manager, self.connector)
            if vllm_config.aux_output_config.enabled
            else None
        )
        self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        # Requests with an async KV load, by stage (KV_FETCH_STAGES).
        # None when not tracked (stats disabled or no KV connector).
        self._kv_fetch_stages: dict[Request, str] | None = (
            {} if self.log_stats and self.connector is not None else None
        )
        self._kv_fetch_counts: dict[str, int] = dict.fromkeys(KV_FETCH_STAGES, 0)

    def _drafts_unused(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> bool:
        """Whether the drafts proposed for `request` after this step are sure to
        go unused.

        Drafts proposed after a step are verified in the request's next decode
        step. They go unused when no such step follows, in two cases:

        1. This step is a non-final prefill chunk. It samples no token, and the
           scheduler drops drafts proposed after a prefill chunk.
        2. This step samples the request's last output token: even if every
           draft is rejected, the request reaches `max_tokens` and finishes.

        Stop tokens and stop strings also end requests, but they cannot be known
        in advance. This returns True only when it is sure.

        The model runner proposes no drafts (K=0) in a step where this is True
        for every scheduled request on every DP rank.

        Args:
            request: A request scheduled in this step.
            num_computed_tokens: Tokens computed before this step, including
                prefix-cache and KV-connector hits for a newly admitted request.
            num_new_tokens: Tokens scheduled for this step.

        """
        # Case 1: a non-final prefill chunk stops before the request's last
        # known token, so this step samples nothing.
        if num_computed_tokens + num_new_tokens < request.num_tokens:
            return True

        # Case 2: this step samples the request's last output token. Drafts may
        # all be rejected, so count only the output tokens the request is sure
        # to have once this step finishes: the verified ones, plus one from each
        # step that samples. This step samples one.
        min_num_output_tokens = request.num_output_tokens + 1
        if request.num_output_placeholders > 0:
            # Async scheduling: a step that samples is still in flight, and it
            # adds one more.
            min_num_output_tokens += 1
        # Reaching max_tokens finishes the request, so no step verifies the drafts.
        return min_num_output_tokens >= request.max_tokens

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        subblock_tokens: int | None = None,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" mode, reusable states are materialized at block boundaries
        and mandatory prefix junctions. Internal checkpoints permit arbitrary
        chunks; an explicit sub-block size retains private running state while
        preserving aligned publication boundaries.
        """
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        block_size = self.block_size
        plan = getattr(request, "leyline_splice_plan", None)
        if plan is not None and start < plan.middle_end:
            # Advance up to middle_end in normal chunks while still stopping at
            # any full Mamba block boundary inside [start, middle_end) so APC
            # prefix states are cached on the way to the splice point.
            next_b = (start // block_size + 1) * block_size
            if start < next_b < plan.middle_end:
                return min(num_new_tokens, next_b - start)
            return min(num_new_tokens, plan.middle_end - start)
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        ckpt_block_size = (
            self.hash_block_size if self.mamba_partial_cache_hit else self.block_size
        )
        last_cache_position = get_mamba_prefill_checkpoint_position(
            request.num_tokens,
            block_size,
            self.use_eagle_block_drop,
        )

        end = start + num_new_tokens
        checkpoint_position = get_mamba_prefill_checkpoint_position(
            prefill_end,
            ckpt_block_size,
            drop_eagle_block=self.use_eagle_block_drop,
        )
        use_internal_checkpoint = (
            self.mamba_has_prefill_checkpoint_blocks
            and end >= prefill_end
            and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=checkpoint_position,
                hash_block_size=self.hash_block_size,
                mamba_block_size=block_size,
                checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
            )
        )
        if use_internal_checkpoint:
            last_cache_position = 0

        subblock_end = start + min(subblock_tokens or num_new_tokens, num_new_tokens)
        use_partial_state = (
            subblock_tokens is not None
            and subblock_end < prefill_end
            and subblock_end % block_size != 0
        )
        if use_partial_state:
            end = subblock_end
        elif end < prefill_end and not use_internal_checkpoint:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            get_mamba_prefill_checkpoint_position(
                request.num_prompt_tokens,
                self.hash_block_size,
                self.use_eagle_block_drop,
            )
            if self.mamba_partial_cache_hit and not use_internal_checkpoint
            else 0
        )
        junction = request.shared_prefix_boundary
        # Block-floored: a sub-block junction's state is not separately cacheable.
        block_floored = start + (junction - start) // block_size * block_size
        # Past the prompt the manager writes nothing, so fall back to the
        # block-floored stop rather than dropping it: a resumed request replaying
        # output tokens can still observe a junction there.
        junction_stop = (
            junction
            if self.mamba_shared_prefix_checkpoint
            and junction <= request.num_prompt_tokens
            else block_floored
        )
        stops = (
            # A chunk starting mid-block must stop at the next boundary.
            next_block_boundary if start % block_size != 0 else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Stop at the prompt checkpoint unless the backend saves it internally.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction: cache its state so sibling
            # requests sharing the prefix can reuse it.
            junction_stop if start < junction < end else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    def _fair_prefill_selection(
        self,
        candidate_ids: list[str],
        budget: int,
        minimum_chunk: int,
        priority_id: str | None = None,
    ) -> tuple[set[str], int]:
        max_selected = max(1, budget // minimum_chunk)
        count = min(len(candidate_ids), max_selected)
        if priority_id is not None and priority_id in candidate_ids:
            other_ids = [item for item in candidate_ids if item != priority_id]
            selected = {priority_id}
            if other_ids and count > 1:
                start = self._prefill_rr_cursor % len(other_ids)
                selected.update(
                    other_ids[(start + offset) % len(other_ids)]
                    for offset in range(count - 1)
                )
                self._prefill_rr_cursor = (start + count - 1) % len(other_ids)
        else:
            start = self._prefill_rr_cursor % len(candidate_ids)
            selected = {
                candidate_ids[(start + offset) % len(candidate_ids)]
                for offset in range(count)
            }
            self._prefill_rr_cursor = (start + count) % len(candidate_ids)
        return selected, max(1, budget // count)

    def _idle_prefill_block_budgets(
        self,
        candidate_ids: list[str],
        token_budget: int,
        block_size: int,
        max_chunk: int,
        priority_id: str | None = None,
    ) -> dict[str, int]:
        """Distribute whole idle-prefill blocks without wasting the remainder."""
        if not candidate_ids or token_budget < block_size:
            return {}
        total_blocks = token_budget // block_size
        max_blocks = max(1, max_chunk // block_size)
        if priority_id is not None and priority_id in candidate_ids:
            other_ids = [item for item in candidate_ids if item != priority_id]
            start = self._idle_prefill_rr_cursor % max(len(other_ids), 1)
            ordered = [priority_id] + [
                other_ids[(start + offset) % len(other_ids)]
                for offset in range(len(other_ids))
            ]
        else:
            start = self._idle_prefill_rr_cursor % len(candidate_ids)
            ordered = [
                candidate_ids[(start + offset) % len(candidate_ids)]
                for offset in range(len(candidate_ids))
            ]
        selected = ordered[: min(len(ordered), total_blocks)]
        budgets = {request_id: block_size for request_id in selected}
        remaining = total_blocks - len(selected)
        extra_blocks = 0
        while remaining > 0:
            made_progress = False
            for request_id in selected:
                if budgets[request_id] // block_size >= max_blocks:
                    continue
                budgets[request_id] += block_size
                remaining -= 1
                extra_blocks += 1
                made_progress = True
                if remaining == 0:
                    break
            if not made_progress:
                break
        advance = extra_blocks if extra_blocks else len(selected)
        rotation_size = (
            len(candidate_ids) - 1
            if priority_id is not None and priority_id in candidate_ids
            else len(candidate_ids)
        )
        if rotation_size:
            self._idle_prefill_rr_cursor = (
                start + max(advance - int(priority_id is not None), 1)
            ) % rotation_size
        return budgets

    def _estimated_uncached_prefill_tokens(self, request: Request) -> int:
        if request.num_computed_tokens > 0:
            return max(request.num_prompt_tokens - request.num_computed_tokens, 0)
        if (
            not self.kv_cache_manager.enable_caching
            or request.skip_reading_prefix_cache
        ):
            return request.num_prompt_tokens
        try:
            _, cached_tokens, _ = (
                self.kv_cache_manager.coordinator.find_longest_cache_hit(
                    request.block_hashes, request.num_tokens - 1
                )
            )
        except Exception:
            cached_tokens = 0
        return max(request.num_prompt_tokens - cached_tokens, 0)

    def _select_short_waiting_prefill(self) -> str | None:
        threshold = self.scheduler_config.short_prefill_priority_token_threshold
        if threshold is None:
            return None
        candidates: list[tuple[Request, RequestQueue]] = []
        long_prefills: list[Request] = [
            request
            for request in self.running
            if request.is_prefill_chunk
            and self._estimated_uncached_prefill_tokens(request) > threshold
        ]
        for request_queue in (self.skipped_waiting, self.waiting):
            for request in request_queue:
                is_prefill = request.num_computed_tokens < request.num_tokens - 1
                if not is_prefill or request.status != RequestStatus.WAITING:
                    continue
                if self._estimated_uncached_prefill_tokens(request) <= threshold:
                    candidates.append((request, request_queue))
                else:
                    long_prefills.append(request)
        if not candidates:
            self._short_prefill_priority_streak = 0
            return None

        max_burst = self.scheduler_config.short_prefill_priority_max_burst
        oldest_long_age = max(
            (time.time() - request.arrival_time for request in long_prefills),
            default=0.0,
        )
        if oldest_long_age >= self.scheduler_config.short_prefill_priority_aging_seconds:
            max_burst = 1
        if long_prefills and self._short_prefill_priority_streak >= max_burst:
            self._short_prefill_priority_streak = 0
            return None

        request, request_queue = min(
            candidates,
            key=lambda item: (
                self._estimated_uncached_prefill_tokens(item[0]),
                item[0].arrival_time,
            ),
        )
        request_queue.remove_request(request)
        request_queue.prepend_request(request)
        self._short_prefill_priority_streak += 1
        return request.request_id

    @staticmethod
    def _build_spec_profitability_fingerprint(
        vllm_config: VllmConfig, speculative_config: Any
    ) -> str:
        try:
            from vllm.platforms import current_platform

            device_name = current_platform.get_device_name()
        except Exception:
            device_name = "unknown"
        tunable_path = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
        tunable_digest = None
        if tunable_path:
            path = Path(tunable_path)
            candidates = [path] if path.is_file() else sorted(
                path.parent.glob(f"{path.stem}[0-9]*{path.suffix}")
            )
            if candidates:
                digest = hashlib.sha256()
                for candidate in candidates:
                    digest.update(candidate.name.encode())
                    digest.update(candidate.read_bytes())
                tunable_digest = digest.hexdigest()
        identity = {
            "model": vllm_config.model_config.compute_hash(),
            "device": device_name,
            "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
            "num_speculative_tokens": speculative_config.num_speculative_tokens,
            "batch_schedule": speculative_config.num_speculative_tokens_per_batch_size,
            "profitability_window": speculative_config.profitability_window,
            "profitability_acceptance_threshold": (
                speculative_config.profitability_acceptance_threshold
            ),
            "profitability_acceptance_window": (
                speculative_config.profitability_acceptance_window
            ),
            "mamba_prefill_subblock_tokens": (
                vllm_config.scheduler_config.mamba_prefill_subblock_tokens
            ),
            "tunable_ops": tunable_digest,
        }
        payload = json.dumps(identity, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def _load_spec_profitability_state(self) -> None:
        path = self.profitability_state_path
        if not path:
            return
        try:
            with open(path) as state_file:
                state = json.load(state_file)
            if state.get("version") != 2:
                raise ValueError("unsupported state version")
            if state.get("fingerprint") != self._profitability_state_fingerprint:
                raise ValueError("state fingerprint does not match this runtime")
            saved_at = float(state["saved_at"])
            state_age = max(time.time() - saved_at, 0.0)
            if state_age > self.profitability_state_ttl_seconds:
                raise ValueError("state has expired")
            loaded_history: dict[tuple[int, int], deque[float]] = {}
            loaded_steps: dict[int, int] = {}
            loaded_samples = 0
            for key, values in state.get("history", {}).items():
                batch_size, k = (int(part) for part in key.split(":"))
                valid = [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float)) and value > 0
                ]
                if valid:
                    retention = 0.5 ** (
                        state_age / self.profitability_state_half_life_seconds
                    )
                    keep = max(1, math.ceil(len(valid) * retention))
                    history = deque(
                        valid[-keep:], maxlen=self._spec_profitability_window
                    )
                    loaded_history[(batch_size, k)] = history
                    loaded_samples += len(history)
            for key, value in state.get("steps", {}).items():
                if isinstance(value, int) and value >= 0:
                    loaded_steps[int(key)] = value
            self._spec_profitability_history.update(loaded_history)
            self._spec_profitability_steps.update(loaded_steps)
            self._spec_profitability_loaded_samples = loaded_samples
            self._spec_profitability_state_age_seconds = state_age
            logger.info(
                "Loaded %d speculative profitability samples from %s",
                loaded_samples,
                path,
            )
        except FileNotFoundError:
            return
        except Exception:
            logger.warning(
                "Ignoring invalid speculative profitability state at %s",
                path,
                exc_info=True,
            )

    def _save_spec_profitability_state(self) -> None:
        path = self.profitability_state_path
        if not path or not self._spec_profitability_history:
            return
        state = {
            "version": 2,
            "fingerprint": self._profitability_state_fingerprint,
            "saved_at": time.time(),
            "history": {
                f"{batch_size}:{k}": list(history)
                for (batch_size, k), history in sorted(
                    self._spec_profitability_history.items()
                )
            },
            "steps": {
                str(batch_size): steps
                for batch_size, steps in sorted(
                    self._spec_profitability_steps.items()
                )
            },
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w") as state_file:
                json.dump(state, state_file, sort_keys=True)
                state_file.write("\n")
            os.replace(temporary, path)
            self._spec_profitability_unsaved = 0
        except Exception:
            logger.warning(
                "Could not persist speculative profitability state to %s",
                path,
                exc_info=True,
            )
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _profitability_history_for_batch(
        self, batch_size: int, k: int
    ) -> tuple[list[float], list[int]]:
        exact = list(self._spec_profitability_history.get((batch_size, k), ()))
        if len(exact) >= self.profitability_min_samples:
            return exact, []
        radius = self.profitability_neighbor_batch_radius
        neighbors = sorted(
            (
                (abs(other_batch - batch_size), other_batch, list(history))
                for (other_batch, other_k), history in (
                    self._spec_profitability_history.items()
                )
                if other_k == k
                and other_batch != batch_size
                and abs(other_batch - batch_size) <= radius
                and history
            ),
            key=lambda item: (item[0], item[1]),
        )
        combined = exact.copy()
        sources: list[int] = []
        for _, other_batch, history in neighbors:
            combined.extend(history)
            sources.append(other_batch)
            if len(combined) >= self.profitability_min_samples:
                break
        return combined[-self._spec_profitability_window :], sources

    def _profitability_adjusted_spec_tokens(
        self,
        batch_size: int,
        base_tokens: int,
        request_ids: Iterable[str] = (),
    ) -> int:
        selected = base_tokens
        exploration = False
        acceptance_gated = False
        request_signature = frozenset(request_ids)
        if request_signature != self._spec_acceptance_request_ids:
            self._spec_acceptance_request_ids = request_signature
            self._spec_acceptance_history.clear()
            for history in self._spec_request_profitability_history.values():
                history.clear()

        minimum_batch = self.profitability_aware_min_batch_size
        if (
            minimum_batch is not None
            and batch_size >= minimum_batch
            and base_tokens == 3
        ):
            lower_tokens = 2
            self._spec_profitability_steps[batch_size] += 1
            step = self._spec_profitability_steps[batch_size]
            if request_signature:
                # Generated content strongly affects draft acceptance. Re-sample
                # both depths for each active request instead of inheriting a
                # decision made by unrelated content. Global/persisted history
                # remains the warm-start path for callers without request IDs.
                base_history = list(
                    self._spec_request_profitability_history[base_tokens]
                )
                lower_history = list(
                    self._spec_request_profitability_history[lower_tokens]
                )
                base_sources = lower_sources = []
            else:
                base_history, base_sources = self._profitability_history_for_batch(
                    batch_size, base_tokens
                )
                lower_history, lower_sources = self._profitability_history_for_batch(
                    batch_size, lower_tokens
                )
            self._last_spec_profitability_warm_start_batches = sorted(
                set(base_sources + lower_sources)
            )
            if len(lower_history) < self.profitability_min_samples:
                selected = lower_tokens
            elif len(base_history) < self.profitability_min_samples:
                selected = base_tokens
            else:
                base_rate = sum(base_history) / len(base_history)
                lower_rate = sum(lower_history) / len(lower_history)
                selected = (
                    lower_tokens
                    if lower_rate
                    > base_rate * (1.0 + self.profitability_hysteresis)
                    else base_tokens
                )

            threshold = self.profitability_acceptance_threshold
            if (
                threshold is not None
                and len(self._spec_acceptance_history)
                >= self.profitability_min_samples
                and sum(self._spec_acceptance_history)
                / len(self._spec_acceptance_history)
                < threshold
            ):
                selected = lower_tokens
                acceptance_gated = True

            # Low-acceptance K3 probes create visible latency spikes because
            # they run an extra draft pass precisely when most draft work is
            # discarded. Probe less often while acceptance-gated, but retain a
            # bounded recovery path if the generated content becomes easier.
            exploration_interval = self.profitability_exploration_interval
            if acceptance_gated:
                exploration_interval *= 32
            if step % exploration_interval == 0:
                selected = base_tokens if selected == lower_tokens else lower_tokens
                exploration = True
                acceptance_gated = False
        self._last_spec_profitability_batch_size = batch_size
        self._last_spec_profitability_selected_k = selected
        self._last_spec_profitability_exploration = exploration
        self._last_spec_profitability_acceptance_gated = acceptance_gated
        return selected

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        connector = self.connector
        if connector is not None and connector.supports_divergent_local_hybrid_hits:
            return self.kv_cache_manager.get_computed_blocks_for_connector(
                request, max_cache_hit_length
            )

        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request, max_cache_hit_length)
        )
        return blocks, num_local, shared_prefix_boundary, False

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Never end a prefill chunk within num_prefill_lookahead of the
        prefill end.

        At a chunked-prefill boundary, the multi-module MTP drafter consumes
        the next num_prefill_lookahead known prefill tokens as draft inputs. A
        boundary closer to the end than that would make it fall back to
        sampled drafts, permanently polluting the trailing modules' KV caches.
        Either finish the prefill or leave at least num_prefill_lookahead for
        the next chunk. No-op for eagle-family drafters (lookahead 1).
        """
        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)

    def schedule(
        self,
        throttle_prefills: bool = False,
        decode_active_prefill_token_budget: int | None = None,
        mamba_prefill_subblock_tokens: int | None = None,
        eb_forced_prefill_id: str | None = None,
    ) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether a request scheduled this step can use the drafts proposed after
        # it. Stays set if that request is preempted later in the step.
        drafts_needed = not self.can_skip_drafts
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether a waiting request was admitted with prompt left to compute.
        # Separate from `prefill_scheduled`, which covers only the running loop.
        prefill_admitted = False
        scheduled_prefill_tokens = 0
        # Whether any scheduled request has a synchronous connector KV load.
        has_sync_kv_loads = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # Protect active decodes with both cadence and an aggregate prefill cap.
        active_decode = any(not r.is_prefill_chunk for r in self.running)
        active_mamba_subblock = (
            mamba_prefill_subblock_tokens
            if mamba_prefill_subblock_tokens is not None
            else self.mamba_prefill_subblock_tokens
        )
        if (
            active_mamba_subblock is not None
            and self.block_size % active_mamba_subblock != 0
        ):
            raise ValueError(
                "dynamic Mamba prefill subblock must divide cache block"
            )
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound and active_decode
        )
        priority_prefill_id = (
            None if defer_prefills else self._select_short_waiting_prefill()
        )
        # P5-1 (eb force-serve progress guarantee): the engine passes the
        # aged prefill id whose wait reached k on a force-serve step. It
        # MUST be scheduled this step: it is selected unconditionally
        # (overriding the fairness priority, which is a nudge) and the
        # waiting loop skips past unselected prefills ahead of it instead
        # of breaking the whole waiting phase (the P3-4 era-A stall: an
        # unselected head-of-line cold prefill kept breaking the loop, so
        # a force-serve step scheduled nothing and the wait grew
        # unbounded past k). It is deliberately NOT routed through
        # ``priority_prefill_id`` itself: that var also drives the
        # running-chunk short-priority clause (whose invariant is
        # "set implies threshold configured"), and the forced id is a
        # waiting request, not a running one. Only applies when prefills
        # are admitted this step (never while deferring); overlap never
        # passes one (its attr is always None, so every condition below
        # degenerates to the pre-P5-1 form).
        forced_prefill_pending = False
        if eb_forced_prefill_id is not None and not defer_prefills:
            forced_prefill_pending = True
        configured_prefill_budget = (
            self.scheduler_config.decode_active_prefill_token_budget
            if decode_active_prefill_token_budget is None
            else decode_active_prefill_token_budget
        )
        prefill_token_budget = configured_prefill_budget if active_decode else None
        selected_prefills: set[str] | None = None
        prefill_request_budget: int | None = None
        if prefill_token_budget is not None and not defer_prefills:
            candidate_ids = (
                [priority_prefill_id] if priority_prefill_id is not None else []
            )
            if forced_prefill_pending and (
                eb_forced_prefill_id not in candidate_ids
            ):
                # P5-1: the forced id must be selectable even if the
                # admission-slot slice of the waiting queue left it out.
                candidate_ids.insert(0, eb_forced_prefill_id)
            candidate_ids.extend(
                request.request_id
                for request in self.running
                if request.is_prefill_chunk
                and request.request_id != priority_prefill_id
            )
            available_slots = max(
                self.max_num_running_reqs
                - len(self.running)
                - self.num_waiting_for_streaming_input,
                0,
            )
            waiting = list(self.skipped_waiting) + list(self.waiting)
            if self.policy == SchedulingPolicy.PRIORITY:
                waiting.sort()
            candidate_ids.extend(
                request.request_id
                for request in waiting[:available_slots]
                if request.num_computed_tokens < request.num_tokens - 1
                and request.request_id != priority_prefill_id
            )
            if candidate_ids:
                minimum_chunk = (
                    self.cache_config.block_size
                    if self.need_mamba_block_aligned_split
                    else 1
                )
                # P5-1: the forced id is the unconditional selection
                # priority on a force-serve step (it wins over the
                # short-waiting fairness priority, which only nudges).
                selection_priority = (
                    eb_forced_prefill_id
                    if forced_prefill_pending
                    else priority_prefill_id
                )
                selected_prefills, prefill_request_budget = (
                    self._fair_prefill_selection(
                        candidate_ids,
                        prefill_token_budget,
                        minimum_chunk,
                        priority_id=selection_priority,
                    )
                )

        long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
        idle_prefill_budgets: dict[str, int] | None = None
        if (
            active_decode
            and self.scheduler_config.decode_active_long_prefill_token_threshold
            is not None
        ):
            long_prefill_threshold = (
                self.scheduler_config.decode_active_long_prefill_token_threshold
            )
        elif (
            not active_decode
            and long_prefill_threshold > 0
            and self.need_mamba_block_aligned_split
        ):
            available_slots = max(
                self.max_num_running_reqs
                - len(self.running)
                - self.num_waiting_for_streaming_input,
                0,
            )
            waiting = list(self.skipped_waiting) + list(self.waiting)
            candidate_ids = (
                [priority_prefill_id] if priority_prefill_id is not None else []
            )
            candidate_ids.extend(
                request.request_id
                for request in self.running
                if request.is_prefill_chunk
                and request.request_id != priority_prefill_id
            )
            candidate_ids.extend(
                request.request_id
                for request in waiting[:available_slots]
                if request.num_computed_tokens < request.num_tokens - 1
                and request.request_id != priority_prefill_id
            )
            idle_prefill_budgets = self._idle_prefill_block_budgets(
                candidate_ids,
                token_budget,
                self.block_size,
                long_prefill_threshold,
                priority_id=priority_prefill_id,
            )

        # `long_prefill_token_threshold` exists to stop a long prefill from
        # starving other requests of the token budget. When it is the only
        # request there is nobody to starve, so let it use the whole budget.
        num_running, num_waiting = self.get_request_counts()
        num_eligible_reqs = num_running + num_waiting
        long_prefill_token_threshold = (
            self.scheduler_config.long_prefill_token_threshold
            if num_eligible_reqs > 1
            else 0
        )
        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
            # Floor the cap at a fair share of the input budget so it never
            # cuts a request below max_num_batched_tokens / num requests.
            long_prefill_token_threshold = max(
                long_prefill_token_threshold, input_budget // num_eligible_reqs
            )

        # Local prefill cadence: hold prefill until the interval since the last
        # prefill-carrying step has elapsed, so decode runs in longer
        # uninterrupted stretches. Mirrors the DP gate above, including its
        # capacity escape: once a release step cannot drain the waiting queue,
        # throttling stops (#44558, test_throttle_capacity_bound_guard_admits).
        # Without that escape a backed-up engine holds both admissions and
        # in-flight chunks for interval-1 of every interval steps, capping
        # prefill at roughly one budget per interval and growing the waiting
        # queue without bound.
        if (
            not defer_prefills
            and self.local_prefill_interval > 1
            and not self.prefill_capacity_bound
            # This call becomes executed step num_executed_steps + 1 if it
            # schedules anything, and last_prefill_step is stamped on the same
            # convention below.
            and (self.num_executed_steps + 1) - self.last_prefill_step
            < self.local_prefill_interval
            and any(not r.is_prefill_chunk for r in self.running)
        ):
            defer_prefills = True

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if input_budget <= draft_slots:
                break

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if request.is_prefill_chunk and (
                (
                    priority_prefill_id is not None
                    and self._estimated_uncached_prefill_tokens(request)
                    > self.scheduler_config.short_prefill_priority_token_threshold
                )
                or defer_prefills
                or (
                    selected_prefills is not None
                    and request.request_id not in selected_prefills
                )
                or (
                    idle_prefill_budgets is not None
                    and request.request_id not in idle_prefill_budgets
                )
            ):
                req_index += 1
                continue

            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            ):
                req_index += 1
                continue
            splice_plan = getattr(request, "leyline_splice_plan", None)
            splice_jump = (
                splice_plan is not None
                and request.num_computed_tokens == splice_plan.middle_end
            )
            if splice_jump:
                request.num_computed_tokens = splice_plan.target_end

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = long_prefill_token_threshold
            if request.is_prefill_chunk and prefill_token_budget is not None:
                num_new_tokens = min(
                    num_new_tokens,
                    prefill_token_budget,
                    prefill_request_budget or prefill_token_budget,
                )
            if 0 < long_prefill_threshold < num_new_tokens:
                num_new_tokens = long_prefill_threshold
            if request.is_prefill_chunk and idle_prefill_budgets is not None:
                num_new_tokens = min(
                    num_new_tokens, idle_prefill_budgets[request.request_id]
                )
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            duplicate_encoder_inputs: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                    duplicate_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            # Keep multi-module MTP prefill inputs valid, then re-apply Mamba
            # alignment after any encoder-budget clipping.
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    subblock_tokens=(
                        active_mamba_subblock if active_decode else None
                    ),
                )

            if splice_plan is not None and not splice_jump:
                num_new_tokens = min(
                    num_new_tokens,
                    max(0, splice_plan.middle_end - request.num_computed_tokens),
                )
            if num_new_tokens == 0:
                if splice_jump:
                    request.num_computed_tokens = splice_plan.middle_end
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 5. Insufficient budget to keep a multi-module MTP prefill
                #    chunk out of the prefill-lookahead window.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    if (
                        self.connector is not None
                        and self.connector.has_pending_block_frees()
                    ):
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                    else:
                        preempted_req = self.running[-1]

                    # A deferred free will not help with immediate allocation.
                    if not self._request_blocks_can_be_freed(preempted_req):
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored_tokens = num_scheduled_tokens.pop(
                                preempted_req_id
                            )
                            token_budget += restored_tokens
                            input_budget += restored_tokens + draft_slots
                            if (
                                prefill_token_budget is not None
                                and preempted_req.is_prefill_chunk
                            ):
                                prefill_token_budget += restored_tokens
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            if splice_jump and not commit_splice(
                self.kv_cache_manager, request, num_new_tokens
            ):
                self.running.remove(request)
                self._preempt_request(request, scheduled_timestamp)
                preempted_reqs.append(request)
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            if request.is_prefill_chunk:
                scheduled_prefill_tokens += num_new_tokens
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if request.is_prefill_chunk and prefill_token_budget is not None:
                prefill_token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1
            drafts_needed = drafts_needed or not self._drafts_unused(
                request, request.num_computed_tokens, num_new_tokens
            )

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                encoder_compute_budget = new_encoder_compute_budget
            if encoder_inputs_to_schedule or external_load_encoder_input:
                self._allocate_encoder_inputs(
                    request,
                    (encoder_inputs_to_schedule or []) + external_load_encoder_input,
                    duplicate_encoder_inputs,
                )

        # Record the LoRAs of all running requests, including those not
        # scheduled in this step, which can join a later batch.
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in self.running
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:
            step_skipped_waiting: deque[Request] = deque()
            step_skipped_kv_holding: deque[Request] = deque()

            def skip_request(from_queue: RequestQueue) -> None:
                request_to_skip = from_queue.pop_request()
                self.deferred_waiting.add(request_to_skip)
                if self._holds_kv_blocks(request_to_skip):
                    step_skipped_kv_holding.appendleft(request_to_skip)
                else:
                    step_skipped_waiting.appendleft(request_to_skip)

            while token_budget > 0:
                # Requests holding KV blocks are always drained first, and are
                # the only ones admitted under PAUSED_NEW (e.g. async KV loads).
                request_queue = self.kv_holding_waiting
                if not request_queue and self._pause_state == PauseState.UNPAUSED:
                    request_queue = self.waiting
                if not request_queue or input_budget <= draft_slots:
                    break
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_active_reqs:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

                ready_to_schedule = self._handle_blocked_waiting_request(request)
                if not ready_to_schedule:
                    skip_request(request_queue)
                    continue

                if request.num_stale_output_tokens and not request.drop_stale_output:
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    skip_request(request_queue)
                    continue

                # Check that adding the request still respects the max_loras constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    skip_request(request_queue)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    max_cache_hit_length = (
                        self.aux_output_connector.max_cache_hit(request)
                        if self.aux_output_connector is not None
                        else None
                    )
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request, max_cache_hit_length)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )
                        if request.skip_reading_prefix_cache:
                            ext_tokens, load_kv_async = 0, False

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            skip_request(request_queue)
                            continue

                        if self.prefix_replay_tokens and not (
                            ext_tokens and self._load_restores_replay_window(request)
                        ):
                            # DSV41 SWA bounded replay recomputes the hit's last
                            # window, from the block holding its first
                            # token; the sliding-window groups retire
                            # whole blocks below the window of the hit's
                            # next token. Hits end on a block boundary, or
                            # a hit ending one token short of one would
                            # retire the block the replay starts in. A load
                            # that carries the window replays nothing.
                            ext_tokens -= ext_tokens % self.block_size
                            load_kv_async = load_kv_async and ext_tokens > 0

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if self.aux_output_connector is not None:
                            num_external_computed_tokens = (
                                self.aux_output_connector.cap_external_hit(
                                    request,
                                    num_new_local_computed_tokens,
                                    num_external_computed_tokens,
                                )
                            )
                            load_kv_async &= num_external_computed_tokens > 0

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(
                                request, max_cache_hit_length
                            )

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                    # DSV41 SWA bounded replay: a KV load that carries the
                    # window is taken as is; any other hit no longer than the
                    # replayed window would be recomputed in full and save
                    # nothing.
                    window_loaded = bool(
                        num_external_computed_tokens
                    ) and self._load_restores_replay_window(request)
                    if (
                        not window_loaded
                        and 0 < num_computed_tokens <= self.prefix_replay_tokens
                    ):
                        new_computed_blocks = (
                            self.kv_cache_manager.empty_kv_cache_blocks
                        )
                        num_new_local_computed_tokens = 0
                        num_external_computed_tokens = 0
                        connector_prefix_cache_hits = 0
                        num_computed_tokens = 0
                        load_kv_async = False

                    # Include encoder inputs needed to replay the prefix tail.
                    ec_computed_tokens = num_computed_tokens
                    if not window_loaded:
                        ec_computed_tokens = max(
                            0, num_computed_tokens - self.prefix_replay_tokens
                        )
                    if self._ec_transfer_pending(request, ec_computed_tokens):
                        skip_request(request_queue)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                    self._set_kv_fetch_stage(
                        request, KV_FETCH_WAITING_TO_START if load_kv_async else None
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed. A streaming-input
                    # session resumes here too, carrying whatever media its
                    # latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                duplicate_encoder_inputs = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False
                is_waiting_prefill = num_computed_tokens < request.num_tokens - 1

                # SWA bounded replay: recompute the tail of the hit without
                # rewriting its cached KV. An async load replays once the KV
                # has arrived (_update_waiting_for_remote_kv).
                num_replay_tokens = 0

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif is_waiting_prefill and (
                    defer_prefills
                    or (
                        selected_prefills is not None
                        and request_id not in selected_prefills
                    )
                    or (
                        idle_prefill_budgets is not None
                        and request_id not in idle_prefill_budgets
                    )
                ):
                    if defer_prefills or not forced_prefill_pending:
                        # Everything deferred this step, or no forced
                        # prefill to reach: pre-P5-1 head-of-line break
                        # (unchanged overlap/eb-defer semantics).
                        break
                    # The forced (aged) prefill is behind this unselected
                    # one in the queue: skip past it — re-queued at the
                    # front for the next step — instead of stalling the
                    # whole waiting phase on it (P3-4 era-A: 154-k cold
                    # head-of-line held the forced waiter 117–140 s at
                    # kvpct ~30% because every force-serve step broke
                    # here and scheduled nothing).
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
                else:
                    hybrid_advance = 0
                    if (
                        self.use_v2_model_runner
                        and not self.use_pp
                        and self.dcp_world_size == self.pcp_world_size == 1
                        and not load_kv_async
                        and num_external_computed_tokens == 0
                    ):
                        plan = prepare_splice(self.kv_cache_manager, request)
                        if plan is not None and num_computed_tokens > plan.middle_end:
                            cancel_splice(self.kv_cache_manager, request)
                            plan = None
                        if plan is not None and num_computed_tokens == plan.middle_end:
                            hybrid_advance = plan.target_end - num_computed_tokens
                            num_computed_tokens = plan.target_end
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    if did_prefix_cache_lookup:
                        # Fresh admission: local and/or sync external hit.
                        num_replay_tokens = self._mark_prefix_replay(
                            request, num_computed_tokens, window_loaded
                        )
                        num_computed_tokens -= num_replay_tokens
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
                        padded_num_tokens = 1 + self.num_spec_tokens
                        # Pad only when there is room for the sampled token(s).
                        if (
                            num_computed_tokens
                            + padded_num_tokens
                            + self.num_sampled_tokens_per_step
                            <= self.max_model_len
                        ):
                            if padded_num_tokens > request_token_budget:
                                # Prefer to not schedule than schedule un-padded.
                                break
                            num_new_tokens = padded_num_tokens
                            pad_spec_decode = True

                    if 0 < long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = long_prefill_token_threshold
                    if 0 < long_prefill_threshold < num_new_tokens:
                        num_new_tokens = long_prefill_threshold
                    if is_waiting_prefill and idle_prefill_budgets is not None:
                        num_new_tokens = min(
                            num_new_tokens, idle_prefill_budgets[request_id]
                        )
                    if is_waiting_prefill and prefill_token_budget is not None:
                        num_new_tokens = min(
                            num_new_tokens,
                            prefill_token_budget,
                            prefill_request_budget or prefill_token_budget,
                        )
                        if num_new_tokens == 0:
                            break

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > request_token_budget
                        and not self.is_mm_encoder_only
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, request_token_budget)
                    assert num_new_tokens > 0

                    # Apply Mamba alignment before encoder caps.
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens + hybrid_advance,
                        )
                        if num_new_tokens == 0:
                            break
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
                            # Alignment clipped the placeholder rows. The split
                            # aligns prefill chunks, but the padded tail rows are
                            # speculative positions, not prefill tokens. A padded
                            # request must keep all 1 + num_spec rows or the
                            # sampler's row count stops matching its query rows,
                            # so drop the padding instead of shortening it.
                            num_new_tokens = 1
                            pad_spec_decode = False

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                            duplicate_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                        )

                    # Keep multi-module MTP prefill inputs valid, then apply
                    # the fork's decode-active Mamba sub-block policy.
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, num_computed_tokens, num_new_tokens
                    )
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens + hybrid_advance,
                            subblock_tokens=(
                                active_mamba_subblock if active_decode else None
                            ),
                        )

                    plan = getattr(request, "leyline_splice_plan", None)
                    if plan is not None and not hybrid_advance:
                        num_new_tokens = min(
                            num_new_tokens,
                            max(0, plan.middle_end - num_computed_tokens),
                        )
                    if num_new_tokens == 0:
                        # The request cannot be scheduled. Requeue it so it does
                        # not stall the requests behind it.
                        skip_request(request_queue)
                        continue

                if load_kv_async:
                    hybrid_advance = 0
                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations)
                    # plus its own spec decode step blocks, to avoid deadlock and
                    # predictable preemptions.
                    reserved_blocks = (
                        self._inflight_prefill_reserved_blocks()
                        + self._spec_decode_step_blocks
                    )

                # Replayed tokens are already counted in the adopted hit; a
                # chunk that ends inside the replayed range needs no new slots.
                num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_tokens_past_hit,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=(
                        num_external_computed_tokens + hybrid_advance
                    ),
                    delay_cache_blocks=load_kv_async,
                    skip_zeroing_group_ids=(
                        self.connector.get_loaded_kv_cache_group_ids(request)
                        if load_kv_async and self.connector is not None
                        else ()
                    ),
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if hybrid_advance and not commit_splice(
                    self.kv_cache_manager, request, num_new_tokens
                ):
                    self.kv_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        preempted = request.num_preemptions > 0
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=preempted,
                            # The connector knows the tier only after the load
                            # plan above exists.
                            hits_by_source=(
                                self.connector.get_external_cache_hit_sources(
                                    request, connector_prefix_cache_hits
                                )
                                if connector_prefix_cache_hits and not preempted
                                else None
                            ),
                        )

                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                request = request_queue.pop_request()
                _prefix_trace.write(
                    self.current_step,
                    request_id,
                    request.num_tokens,
                    num_new_local_computed_tokens,
                    int(request.num_preemptions > 0),
                )
                if load_kv_async:
                    # This path deliberately does not reach `prefill_admitted`
                    # below, so it does not restart the prefill cadence:
                    # `num_new_tokens` was set to 0 above, so the step carries
                    # no prefill compute for this request. The interval is
                    # timed from steps that actually prefilled; refreshing it
                    # here would start the cadence from a step that did none.
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    if self.aux_output_connector is not None:
                        self.aux_output_connector.request_loading(
                            request, num_computed_tokens
                        )
                    self._inflight_prefills.add(request)
                    self._set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
                    skip_request(request_queue)
                    continue

                request = request_queue.pop_request()
                self.deferred_waiting.discard(request)
                self.running.append(request)
                if num_external_computed_tokens > 0:
                    # load_kv_async is False here
                    has_sync_kv_loads = True
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                spliced_advance = self.kv_cache_manager.maybe_apply_leyline_splice(
                    request, num_computed_tokens
                )
                if spliced_advance > 0 and not pad_spec_decode:
                    advance = min(spliced_advance, max(0, num_new_tokens - 1))
                    if advance > 0:
                        num_computed_tokens += advance
                        num_new_tokens -= advance
                        # Budgets have not been charged yet: charge only the
                        # actual forwarded tail below, with no double refund.
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if is_waiting_prefill:
                    scheduled_prefill_tokens += num_new_tokens
                if forced_prefill_pending and request_id == eb_forced_prefill_id:
                    # The aged prefill was scheduled this step: the
                    # guarantee is met; resume the pre-P5-1 break
                    # semantics for the (unselected) requests behind it.
                    forced_prefill_pending = False
                if is_waiting_prefill and prefill_token_budget is not None:
                    prefill_token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                self._set_kv_fetch_stage(request, None)
                drafts_needed = drafts_needed or not self._drafts_unused(
                    request, num_computed_tokens, num_new_tokens
                )
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # Must use the SAME predicate as the deferral gate in the
                # waiting loop (`num_computed_tokens < num_tokens - 1`).
                # Stamping on `< num_prompt_tokens` instead let a request with
                # exactly one token left pass the gate on a held step and still
                # restart the interval -- which is every full prefix-cache hit,
                # and every remote-KV resume, since those set
                # num_computed_tokens = num_tokens - 1.
                prefill_admitted |= num_computed_tokens < request.num_tokens - 1
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    encoder_compute_budget = new_encoder_compute_budget
                if encoder_inputs_to_schedule or external_load_encoder_input:
                    self._allocate_encoder_inputs(
                        request,
                        (encoder_inputs_to_schedule or [])
                        + external_load_encoder_input,
                        duplicate_encoder_inputs,
                    )

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_kv_holding:
                self.kv_holding_waiting.prepend_requests(step_skipped_kv_holding)
            if step_skipped_waiting:
                self.waiting.prepend_requests(step_skipped_waiting)

            # A waiting queue alone may only mean that max_num_seqs is full.
            # Treat the release as capacity-bound only when its token budget was
            # exhausted; otherwise the throttle remains useful for active decodes.
            if not defer_prefills:
                _, num_waiting = self.get_request_counts()
                self.prefill_capacity_bound = num_waiting > len(self.deferred_waiting)
                self.prefill_capacity_bound = bool(self.waiting) and token_budget == 0

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        # Count this step before stamping the cadence below, so that
        # last_prefill_step includes the step that carried the prefill.
        if total_num_scheduled_tokens > 0:
            self.num_executed_steps += 1

        # Restart the cadence interval from any step that carried prefill,
        # whether a new admission or a running chunk.
        if self.local_prefill_interval > 1 and (prefill_scheduled or prefill_admitted):
            self.last_prefill_step = self.num_executed_steps

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    uses_mrope=self.model_uses_mrope,
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Mamba "align" boundary states must be handed off with exact block ids;
        # they cannot be reconstructed from a connector's append-only block
        # table. Drained every step so stale offers cannot accumulate.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

        kv_connector_block_state = None
        if self.connector is not None:
            # Any request scheduled this step can become a connector job now,
            # not only the ones that were allocated blocks: a store save lands
            # on the step that fills a block, which allocated none.
            block_state_req_ids = set(num_scheduled_tokens)
            block_state_req_ids.update(
                req_id for req_id in boundary_state_offloads if req_id in self.requests
            )
            kv_connector_block_state = KVConnectorBlockState(
                req_ids=block_state_req_ids,
                resolve_block_ids=self.kv_cache_manager.get_block_ids,
                boundary_state_offloads=boundary_state_offloads,
            )

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None
        pending_kv_cache_rope_rotations = (
            self.kv_cache_manager.take_kv_cache_rope_rotations() or None
        )
        # Leyline hybrid §2b: fused copy+rotate splices and Mamba boundary-state
        # transplants. The splice ops run with this step's execution and touch
        # donor blocks retained for the step (see _apply_leyline_hybrid_splice);
        # their correctness is guaranteed once the step's forward has read them.
        # The donor (src) and target (dst) FA blocks and the donor Mamba block
        # are pinned for the step and released at step completion (mirrors the
        # pure-attention CoW pin/release), so no splice pin leaks (GAP 2).
        pending_kv_cache_block_splices, splice_retained = (
            self.kv_cache_manager.take_kv_cache_block_splices()
        )
        if splice_retained:
            self._free_cow_retained_blocks(splice_retained, self.sched_step_seq + 1)
        pending_mamba_state_transplants, transplant_retained = (
            self.kv_cache_manager.take_mamba_state_transplants()
        )
        if transplant_retained:
            self._free_cow_retained_blocks(
                transplant_retained, self.sched_step_seq + 1
            )
        pending_kv_cache_block_splices = pending_kv_cache_block_splices or None
        pending_mamba_state_transplants = pending_mamba_state_transplants or None

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
            if self.long_context_threshold is not None and all(
                self.requests[req_id].num_computed_tokens
                >= self.requests[req_id].num_prompt_tokens
                for req_id in num_scheduled_tokens
            ):
                max_context = max(
                    self.requests[req_id].num_computed_tokens
                    for req_id in num_scheduled_tokens
                )
                if max_context >= self.long_context_threshold:
                    assert self.long_context_num_spec_tokens is not None
                    num_spec_tokens_to_schedule = min(
                        num_spec_tokens_to_schedule,
                        self.long_context_num_spec_tokens,
                    )
            if self.greedy_num_spec_tokens is not None and all(
                self.requests[req_id].sampling_params.temperature == 0
                for req_id in num_scheduled_tokens
            ):
                num_spec_tokens_to_schedule = min(
                    num_spec_tokens_to_schedule,
                    self.greedy_num_spec_tokens,
                )
            num_spec_tokens_to_schedule = self._profitability_adjusted_spec_tokens(
                len(num_scheduled_tokens),
                num_spec_tokens_to_schedule,
                num_scheduled_tokens,
            )

        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_cache_rope_rotations=pending_kv_cache_rope_rotations,
            kv_cache_block_splices=pending_kv_cache_block_splices,
            mamba_state_transplants=pending_mamba_state_transplants,
            scheduled_prefill_tokens=scheduled_prefill_tokens,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            skip_drafts=not drafts_needed,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.aux_output_connector is not None:
            scheduler_output.aux_output_connector_metadata = (
                self.aux_output_connector.build_connector_meta(
                    scheduler_output, self.requests
                )
            )

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Connector-only block state must not be dispatched to workers.
        scheduler_output.kv_connector_block_state = None

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None
        return new_block_ids_to_zero or None

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        if self.aux_output_connector is not None:
            self.aux_output_connector.request_finished(request)
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        now_wall = time.time()
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            was_prefill = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            if was_prefill and num_scheduled_token > 0:
                request._eb_last_prefill_service_time = now_wall
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """
        # Keep all output tokens but the final sampled one.
        keep_end = session.num_tokens - 1
        assert keep_end >= session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : keep_end
        ]
        del session._all_token_ids[keep_end:]
        del session.block_hashes[keep_end // self.hash_block_size :]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.max_tokens = update.max_tokens
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _reject_on_encoder_cache_embed_mismatch(
        self,
        request: Request,
        input_id: int,
        identifier: str,
        cached_num_encoder_embeds: int,
        requested_num_encoder_embeds: int,
    ) -> bool:
        if cached_num_encoder_embeds == requested_num_encoder_embeds:
            return False
        logger.warning(
            "Encoder cache entry for multimodal identifier %r holds %d "
            "embeddings, but request %s input %d expects %d. A client-provided "
            "multimodal UUID may have been reused for different content.",
            identifier,
            cached_num_encoder_embeds,
            request.request_id,
            input_id,
            requested_num_encoder_embeds,
        )
        self.encoder_cache_mismatch_reqs.add(request.request_id)
        return True

    def _allocate_encoder_inputs(
        self,
        request: Request,
        input_ids: list[int],
        duplicate_input_ids: list[int],
    ) -> None:
        for input_id in input_ids:
            self.encoder_cache_manager.allocate(request, input_id)
            if self.ec_connector is not None:
                self.ec_connector.update_state_after_alloc(request, input_id)

        for input_id in duplicate_input_ids:
            self.encoder_cache_manager.check_and_update_cache(request, input_id)

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, [], []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []
        duplicate_encoder_inputs = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule: dict[str, int] = {}
        num_embeds_to_schedule = 0

        encoder_window_end = (
            num_computed_tokens + num_new_tokens + shift_computed_tokens
        )
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and not self.is_encoder_decoder
                # `<=`, not `<`: a chunk that starts exactly at the item still
                # splits it when it cannot cover the whole item.
                and num_computed_tokens <= start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    scheduled_num_encoder_embeds = mm_hashes_to_schedule[
                        item_identifier
                    ]
                    if self._reject_on_encoder_cache_embed_mismatch(
                        request,
                        i,
                        item_identifier,
                        scheduled_num_encoder_embeds,
                        num_encoder_embeds,
                    ):
                        return [], 0, encoder_compute_budget, [], []
                    # The same encoder input has already been scheduled in the
                    # current step.
                    duplicate_encoder_inputs.append(i)
                    continue

                cached_num_encoder_embeds = (
                    self.encoder_cache_manager.get_cached_num_encoder_embeds(request, i)
                )
                if cached_num_encoder_embeds is not None and (
                    self._reject_on_encoder_cache_embed_mismatch(
                        request,
                        i,
                        item_identifier,
                        cached_num_encoder_embeds,
                        num_encoder_embeds,
                    )
                ):
                    return [], 0, encoder_compute_budget, [], []

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input. No drafter shift here: its read-ahead into
                    # this unencoded input falls back to token embeddings.
                    num_new_tokens = min(
                        num_new_tokens, start_pos - num_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule[item_identifier] = num_encoder_embeds
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule[item_identifier] = num_encoder_embeds
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
            duplicate_encoder_inputs,
        )

    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()

        for req_id, input_ids in scheduled_encoder_inputs.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            for input_id in input_ids:
                mm_feature = request.mm_features[input_id]
                stats.num_inputs += 1
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()

        return stats if stats.num_inputs else None

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_acceptable_drafts = [
            len(strip_speculative_padding(spec_tokens.get(req_id, [])))
            for req_id in structured_output_request_ids
        ]
        return GrammarOutput(
            structured_output_request_ids, bitmask, num_acceptable_drafts
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        prompt_token_id_logprobs_dict = (
            model_runner_output.prompt_token_id_logprobs_dict
        )
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        ec_connector_output = model_runner_output.ec_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids: set[str] = set()
        if kv_connector_output and kv_connector_output.failed_recving:
            failed_kv_load_req_ids.update(
                self._handle_failed_recving(kv_connector_output.failed_recving)
            )
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids.update(
                self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids,
                    num_scheduled_tokens,
                )
            )
        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        # Stopped resumable requests to re-enqueue, in output order.
        continuing_reqs: list[Request] = []
        completion_timestamp = time.monotonic()
        decode_completion_times = getattr(self, "_decode_completion_times", None)
        if decode_completion_times is None:
            decode_completion_times = self._decode_completion_times = {}
        decode_gap_samples = getattr(self, "_decode_gap_samples", None)
        if decode_gap_samples is None:
            decode_gap_samples = self._decode_gap_samples = deque(maxlen=4096)
        spec_step_committed_tokens = 0
        spec_step_num_tokens: int | None = None
        spec_step_requests = 0
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                if not output_is_stale:
                    spec_step_committed_tokens += len(generated_token_ids)
                    spec_step_num_tokens = num_draft_tokens
                    spec_step_requests += 1
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                    _spec_trace.write(
                        self.current_step,
                        len(scheduler_output.num_scheduled_tokens),
                        req_id,
                        num_draft_tokens,
                        num_accepted,
                    )
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                if request.spec_decode_metrics is not None:
                    # Exclude grammar-invalidated drafts from the proposed
                    # count, mirroring make_spec_decoding_stats; the accepted
                    # bucket (j) is unaffected.
                    adj_draft_tokens = num_draft_tokens
                    if scheduler_output.num_invalid_spec_tokens:
                        adj_draft_tokens -= (
                            scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                        )
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=adj_draft_tokens,
                        num_accepted=num_accepted,
                        detailed=self.spec_decode_metrics_level == "detailed",
                    )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_sampling_mask = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
                if new_token_ids:
                    previous = decode_completion_times.get(req_id)
                    if previous is not None and num_output_tokens_before > 0:
                        decode_gap_samples.append(
                            (
                                completion_timestamp,
                                (completion_timestamp - previous) * 1000.0,
                            )
                        )
                    decode_completion_times[req_id] = completion_timestamp
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_mm_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and not self.structured_output_manager.accept_tokens(
                request, new_token_ids
            ):
                logger.error(
                    "Unexpected: grammar rejected tokens %s for request %s. "
                    "Terminating request.",
                    new_token_ids,
                    req_id,
                )
                request.status = RequestStatus.FINISHED_ERROR
                request.resumable = False
                stopped = True

            routed_experts = None
            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            if self.aux_output_connector is not None and should_emit_output:
                routed_experts = self.aux_output_connector.take_output(
                    request, model_runner_output.aux_output_connector_output
                )
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    skipped = int(request.kv_splice_stats.get("skipped_tokens", 0) or 0)
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request) + skipped
                    )
                    if skipped:
                        prefill_stats.num_local_cached_tokens += skipped

            # Extract sample logprobs before stop handling can replace the
            # sampling parameters for a streaming continuation.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)
                    if (
                        prefill_stats is not None
                        and kv_transfer_params is not None
                        and kv_transfer_params.get("do_remote_prefill")
                    ):
                        # P-side cache hits, so D can report them in
                        # prompt_tokens_details instead of its own (~100%)
                        # hit rate from the KV transfer.
                        kv_transfer_params["remote_prefill_cached_tokens"] = (
                            prefill_stats.num_cached_tokens
                        )
                else:
                    continuing_reqs.append(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if self.return_sampling_mask:
                sampling_masks = model_runner_output.sampling_masks
                if new_token_ids and sampling_masks is not None:
                    new_sampling_mask = sampling_masks.slice_request(
                        req_index, len(new_token_ids)
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            prompt_token_id_logprobs = prompt_token_id_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_sampling_mask=new_sampling_mask,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        prompt_token_id_logprobs=prompt_token_id_logprobs,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        spec_decode_metrics=(
                            request.spec_decode_metrics
                            if finish_reason is not None
                            else None
                        ),
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors
                assert prompt_token_id_logprobs is None

        # Remove stopped requests from their queues, then enqueue the ones
        # that continue, so each ends up in exactly one queue.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.kv_holding_waiting.remove_requests(stopped_preempted_reqs)
            self.deferred_waiting.difference_update(stopped_preempted_reqs)
        for request in continuing_reqs:
            self._enqueue_waiting_request(request)

        request_errors = self.grammar_compile_error_reqs
        self.grammar_compile_error_reqs = {}
        error_req_ids = set(request_errors)
        error_req_ids.update(self.encoder_cache_mismatch_reqs)
        self.encoder_cache_mismatch_reqs.clear()
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)
        if self.ec_connector is not None:
            # An encoder input the connector can no longer obtain. Failing is
            # retryable: re-issuing the request re-runs the encode.
            error_req_ids.update(self.ec_connector.take_unavailable_requests())

        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        error=request_errors.get(request.request_id),
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # EC Connector: update state from worker-side EC connector output.
        if self.ec_connector is not None and ec_connector_output:
            self.ec_connector.update_connector_output(ec_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # Worker-side EC connector stats from the model runner output.
        ec_connector_stats: ECConnectorStats | None = (
            ec_connector_output.ec_connector_stats if ec_connector_output else None
        )
        if self.ec_connector:
            # Scheduler-side EC connector stats collected after connector update.
            scheduler_ec_connector_stats = self.ec_connector.get_ec_connector_stats()
            if (
                scheduler_ec_connector_stats is not None
                and not scheduler_ec_connector_stats.is_empty()
            ):
                ec_connector_stats = (
                    ec_connector_stats.aggregate(scheduler_ec_connector_stats)
                    if ec_connector_stats is not None
                    else scheduler_ec_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        batch_size = len(scheduler_output.num_scheduled_tokens)
        minimum_batch = getattr(self, "profitability_aware_min_batch_size", None)
        if (
            minimum_batch is not None
            and batch_size >= minimum_batch
            and scheduler_output.scheduled_prefill_tokens == 0
            and scheduler_output.model_step_elapsed_ms > 0
            and spec_step_num_tokens in (2, 3)
            and spec_step_requests > 0
            and spec_step_committed_tokens > 0
        ):
            key = (batch_size, spec_step_num_tokens)
            history = self._spec_profitability_history.get(key)
            if history is None:
                history = deque(maxlen=self._spec_profitability_window)
                self._spec_profitability_history[key] = history
            observed_rate = (
                spec_step_committed_tokens / scheduler_output.model_step_elapsed_ms
            )
            history.append(observed_rate)
            if self._spec_acceptance_request_ids:
                self._spec_request_profitability_history[
                    spec_step_num_tokens
                ].append(observed_rate)
            if spec_step_num_tokens == 3:
                accepted = max(
                    spec_step_committed_tokens
                    - spec_step_requests * self.num_sampled_tokens_per_step,
                    0,
                )
                self._spec_acceptance_history.append(
                    accepted / (spec_step_requests * spec_step_num_tokens)
                )
            self._spec_profitability_unsaved += 1
            if (
                self._spec_profitability_unsaved
                >= self.profitability_state_save_interval
            ):
                self._save_spec_profitability_state()

        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
                ec_connector_stats=ec_connector_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _ec_transfer_pending(self, request: Request, num_computed_tokens: int) -> bool:
        """Whether an encoder input this request needs is still in transit."""
        return (
            self.ec_connector is not None
            and bool(request.mm_features)
            and not self.ec_connector.ensure_cache_available(
                request, num_computed_tokens
            )
        )

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    @staticmethod
    def _holds_kv_blocks(request: Request) -> bool:
        # Whether a request currently has kv blocks allocated.
        # While always the case when status is WAITING_FOR_REMOTE_KVS
        # or WAITING_FOR_STREAMING_REQ, there are such cases where it's
        # WAITING or PREEMPTED.
        return (
            request.num_computed_tokens > 0
            # num_computed_tokens is reset to 0 if kv transfer fails for requests in
            # WAITING_FOR_REMOTE_KVS state, before their allocated blocks are freed.
            or request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.deferred_waiting.add(request)
        if self._holds_kv_blocks(request):
            self.kv_holding_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        # Drop the finished turn's in-flight work and resume from the
        # materialized frontier.
        safe_frontier = request.num_computed_tokens - request.num_output_placeholders
        assert safe_frontier >= 0
        request.num_computed_tokens = safe_frontier
        request.spec_token_ids = []
        request.drop_stale_output = True
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                # num_computed_tokens covers all new tokens but the last one.
                # Keep that true after trimming: resumable requests resume from it.
                if not is_stale:
                    request.num_computed_tokens -= len(new_token_ids) - num_new
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self._free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and past the
                # placeholder range with no pending rejection that could roll
                # back into it. The drafter only reads ahead of
                # num_computed_tokens, so it cannot reference this input either.
                self._free_encoder_input(request, input_id)

    def _free_encoder_input(self, request: Request, input_id: int) -> None:
        self.encoder_cache_manager.free_encoder_input(request, input_id)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_free(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            request.spec_token_ids = self.structured_output_manager.validate_tokens(
                request, spec_token_ids
            )

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            spec_token_ids = self.structured_output_manager.validate_tokens(
                request, spec_token_ids
            )
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_max_decode_tokens(self) -> int:
        return max(
            (
                request.num_output_tokens
                for request in self.running
                if not request.is_prefill_chunk
            ),
            default=0,
        )

    def get_decode_prefill_arrival_gap(self) -> float:
        decode_arrivals = [
            request.arrival_time
            for request in self.running
            if not request.is_prefill_chunk
        ]
        prefill_arrivals = [
            request.arrival_time
            for request in self.running
            if request.is_prefill_chunk
        ]
        prefill_arrivals.extend(
            request.arrival_time
            for request in (*self.waiting, *self.skipped_waiting)
            if request.num_output_tokens == 0
        )
        if not decode_arrivals or not prefill_arrivals:
            return 0.0
        return max(min(prefill_arrivals) - min(decode_arrivals), 0.0)

    def get_kv_cache_block_counts(self) -> tuple[int, int, int, int]:
        block_pool = self.kv_cache_manager.block_pool
        immediate, evictable = block_pool.get_free_block_counts()
        # The permanently allocated null block is not usable capacity.
        total = max(block_pool.num_gpu_blocks - 1, 0)
        pinned = max(total - immediate - evictable, 0)
        return immediate, evictable, pinned, total

    def get_kv_cache_token_counts(self) -> tuple[int, int, int, int]:
        immediate, evictable, pinned, total = self.get_kv_cache_block_counts()
        return (
            immediate * self.block_size,
            evictable * self.block_size,
            pinned * self.block_size,
            total * self.block_size,
        )

    def get_request_kv_token_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for request_id in self.requests:
            block_ids = self.kv_cache_manager.get_block_ids(request_id)
            allocated_blocks = max((len(group) for group in block_ids), default=0)
            counts[request_id] = allocated_blocks * self.block_size
        return counts

    def get_decode_gap_p99_ms(self) -> float | None:
        cutoff = time.monotonic() - self._decode_gap_window_seconds
        while self._decode_gap_samples and self._decode_gap_samples[0][0] < cutoff:
            self._decode_gap_samples.popleft()
        if not self._decode_gap_samples:
            return None
        values = sorted(gap_ms for _, gap_ms in self._decode_gap_samples)
        return values[math.ceil(len(values) * 0.99) - 1]

    def has_decode_prefill_overlap(self) -> bool:
        has_decode = any(not request.is_prefill_chunk for request in self.running)
        if not has_decode:
            return False
        return any(request.is_prefill_chunk for request in self.running) or any(
            request.num_output_tokens == 0
            for request in (*self.waiting, *self.skipped_waiting)
        )

    def get_oldest_prefill_wait_seconds(self) -> float:
        arrivals = [
            request.arrival_time
            for request in self.running
            if request.is_prefill_chunk
        ]
        arrivals.extend(
            request.arrival_time
            for request in (*self.waiting, *self.skipped_waiting)
            if request.num_output_tokens == 0
        )
        return max(time.time() - min(arrivals), 0.0) if arrivals else 0.0

    def get_oldest_eb_prefill_wait_seconds(
        self, now: float | None = None
    ) -> float:
        """Oldest prefill wait for EB(0) phase separation:
        - waiting/skipped_waiting requests with zero output tokens use
          ``request.arrival_time``;
        - running ``is_prefill_chunk`` requests use
          ``max(arrival_time, _eb_last_prefill_service_time)`` so a running
          chunk that was just force-served does not stay permanently >= k and
          latch ``force_prefill_service=True`` into a 100-token trickle on
          every subsequent decode step.
        """
        t = time.time() if now is None else now
        waits: list[float] = []
        for request in self.running:
            if not request.is_prefill_chunk:
                continue
            ref = max(
                request.arrival_time,
                getattr(request, "_eb_last_prefill_service_time", 0.0),
            )
            waits.append(max(t - ref, 0.0))
        for request in (*self.waiting, *self.skipped_waiting):
            if request.num_output_tokens == 0:
                waits.append(max(t - request.arrival_time, 0.0))
        return max(waits) if waits else 0.0

    def get_deferred_prefill_count(self) -> int:
        """Prefills the prefill throttle is holding back (EB(0) deferral):
        waiting with zero output tokens. A prefill that is RUNNING is being
        served this step (throttling never preempts it), so it does not
        count. Same population as ``get_oldest_prefill_wait_seconds``."""
        return sum(
            1
            for request in (*self.waiting, *self.skipped_waiting)
            if request.num_output_tokens == 0
        )

    def get_oldest_deferred_prefill_id(self) -> str | None:
        """The deferred prefill with the oldest arrival (P5-1): the waiting
        request with zero output tokens and the minimum arrival_time — the
        same deferred population as ``get_deferred_prefill_count`` (running
        prefill chunks are already being served, so they are not
        candidates). The eb force-serve step hands this id to
        ``schedule(eb_forced_prefill_id=...)`` so it is guaranteed to be
        scheduled that step. None when the deferred population is empty."""
        oldest: Request | None = None
        for request in (*self.waiting, *self.skipped_waiting):
            if request.num_output_tokens != 0:
                continue
            if oldest is None or request.arrival_time < oldest.arrival_time:
                oldest = request
        if oldest is not None:
            return oldest.request_id
        # If no waiting request is deferred, fall back to the running prefill
        # chunk whose effective EB wait (since arrival or last service) is
        # oldest, so an aged running chunk is also guaranteed selection on its
        # force-serve step.
        oldest_running: Request | None = None
        oldest_ref = float("inf")
        for request in self.running:
            if not request.is_prefill_chunk:
                continue
            ref = max(
                request.arrival_time,
                getattr(request, "_eb_last_prefill_service_time", 0.0),
            )
            if ref < oldest_ref:
                oldest_ref = ref
                oldest_running = request
        return oldest_running.request_id if oldest_running is not None else None

    def get_spec_profitability_stats(self) -> dict[str, object] | None:
        if self.profitability_aware_min_batch_size is None:
            return None
        batches: dict[str, dict[str, float | int | None]] = {}
        for (batch_size, k), history in sorted(
            self._spec_profitability_history.items()
        ):
            batch = batches.setdefault(str(batch_size), {})
            batch[f"k{k}_samples"] = len(history)
            batch[f"k{k}_tokens_per_ms"] = (
                sum(history) / len(history) if history else None
            )
            batch["steps"] = self._spec_profitability_steps[batch_size]
        return {
            "last_batch_size": self._last_spec_profitability_batch_size,
            "last_selected_k": self._last_spec_profitability_selected_k,
            "last_exploration": self._last_spec_profitability_exploration,
            "last_acceptance_gated": (
                self._last_spec_profitability_acceptance_gated
            ),
            "recent_k3_acceptance": (
                sum(self._spec_acceptance_history)
                / len(self._spec_acceptance_history)
                if self._spec_acceptance_history
                else None
            ),
            "recent_k3_acceptance_samples": len(self._spec_acceptance_history),
            "request_k2_tokens_per_ms": (
                sum(self._spec_request_profitability_history[2])
                / len(self._spec_request_profitability_history[2])
                if self._spec_request_profitability_history[2]
                else None
            ),
            "request_k3_tokens_per_ms": (
                sum(self._spec_request_profitability_history[3])
                / len(self._spec_request_profitability_history[3])
                if self._spec_request_profitability_history[3]
                else None
            ),
            "warm_start_batches": self._last_spec_profitability_warm_start_batches,
            "loaded_samples": self._spec_profitability_loaded_samples,
            "state_age_seconds": self._spec_profitability_state_age_seconds,
            "state_fingerprint": self._profitability_state_fingerprint,
            "state_path": self.profitability_state_path,
            "batches": batches,
        }

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.kv_holding_waiting)

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return self.kv_cache_manager.usage

    def has_request(self, request_id: str) -> bool:
        return request_id in self.requests
    def _apply_leyline_splice_cache_cap(self, request: Request) -> None:
        """Set ``request._leyline_max_cacheable_tokens`` (design §2c) at
        admission, BEFORE any ``allocate_slots`` runs.

        A spliced request's target-suffix KV blocks are a function of the
        DONOR's context (they are CoW-copied + RoPE-rotated from the donor),
        so caching them under the TARGET tokens' chained APC hashes would let
        a later, unrelated request sharing the target suffix falsely hit
        donor-contaminated KV (wrong output + cross-session pollution). The
        cap clamps APC registration to ``s + r`` (the clean target prefix plus
        any freshly-decoded replacement middle); every full block at or past
        that boundary is donor-contaminated and must stay un-registered.

        This must run at admission (not in ``maybe_apply_leyline_splice``,
        which fires after ``allocate_slots`` has already cached the suffix).
        No-op unless the request carries amortize-mode splice directives and
        the model has a full-attention group (a pure-Mamba model's splice is a
        no-op, so its Mamba checkpoints stay uncapped).
        """
        sp = getattr(request, "sampling_params", None)
        if sp is None or getattr(sp, "extra_args", None) is None:
            return
        if sp.extra_args.get("kv_splice_mode", "off") != "amortize":
            return
        directives = getattr(request, "kv_splice_directives", None)
        if not directives:
            return
        # Only full-attention blocks are CoW-copied/rotated from the donor; a
        # model without a full-attention group applies no splice, so leave its
        # (Mamba) cache registration uncapped.
        if not any(
            not isinstance(g.kv_cache_spec, MambaSpec)
            for g in self.kv_cache_config.kv_cache_groups
        ):
            return
        max_cacheable = None
        for item in directives:
            if isinstance(item, dict):
                s = int(item.get("span_start", item.get("s", 0)))
                r = int(item.get("replacement_len", item.get("r", 0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                s, r = int(item[0]), int(item[2])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                s, r = int(item[0]), int(item[1])
            else:
                continue
            boundary = s + r
            max_cacheable = boundary if max_cacheable is None else min(
                max_cacheable, boundary
            )
        if max_cacheable is not None:
            if not self.kv_cache_config.has_mamba_layers:
                request._leyline_max_cacheable_tokens = max_cacheable

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            # §2c: clamp APC cache registration for amortize-mode splice
            # requests BEFORE the request is admitted to the waiting queue (and
            # therefore before any allocate_slots caches its suffix). No-op for
            # non-splice / mode-off requests.
            self._apply_leyline_splice_cache_cap(request)
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.spec_decode_metrics_level != "none":
                request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                    self.num_spec_tokens
                )
            if self.connector is not None:
                self.connector.on_new_request(request)
                kv_transfer_params = request.kv_transfer_params
                if kv_transfer_params and kv_transfer_params.get("do_remote_prefill"):
                    # Count P/D requests on arrival: connector matching is
                    # skipped while all run slots are taken.
                    self._set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.

        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.kv_holding_waiting.remove_requests(waiting_requests_to_remove)
            self.deferred_waiting.difference_update(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        if self.aux_output_connector is not None:
            self.aux_output_connector.request_finished(request)
        self._inflight_prefills.discard(request)
        self._set_kv_fetch_stage(request, None)
        self._decode_completion_times.pop(request.request_id, None)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        logger.info("setting pause state to %s", pause_state.name)
        self._pause_state = pause_state

    def _request_blocks_can_be_freed(self, request: Request) -> bool:
        # We must defer freeing blocks if an async kv connector may
        # write to them immediately (not ordered with GPU stream).
        return not self.defer_block_free or (
            request.last_sched_seq <= self.processed_step_seq
        )

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if self._request_blocks_can_be_freed(request):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        num_running, num_waiting = self.get_request_counts()
        if self._pause_state == PauseState.PAUSED_NEW:
            # Requests holding KV blocks still drain; queued ones stay queued.
            num_waiting = len(self.kv_holding_waiting)
        num_waiting -= self.num_waiting_for_streaming_input
        return num_waiting + num_running

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_running, num_waiting = self.get_request_counts()
        return len(self.requests) > (num_running + num_waiting)

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        if reset_successful and self.aux_output_connector is not None:
            self.aux_output_connector.reset()

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
        ec_connector_stats: ECConnectorStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.to_dict() if kv_connector_stats else None
        )
        ec_connector_stats_payload = (
            ec_connector_stats.data if ec_connector_stats else None
        )
        num_running, num_waiting = self.get_request_counts()
        num_deferred = len(self.deferred_waiting)
        return SchedulerStats(
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting - num_deferred,
            num_skipped_waiting_reqs=num_deferred,
            num_kv_fetch_reqs_by_stage=self._kv_fetch_counts.copy(),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
            ec_connector_stats=ec_connector_stats_payload,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        finished_partial_tails: list[tuple[int, int, int]] = []
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.is_kv_producer:
            finished_partial_tails = (
                self.kv_cache_manager.finalize_partial_tail_offloads(request)
            )

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )
        partial_tail_delay = False
        if finished_partial_tails:
            partial_tail_delay = self.connector.register_finished_partial_tail(
                request,
                block_ids,
                finished_partial_tails,
            )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            delay_free, kv_xfer_params = self.connector.request_finished(
                request, block_ids[0]
            )
        else:
            delay_free, kv_xfer_params = self.connector.request_finished_all_groups(
                request, block_ids
            )
        return delay_free or partial_tail_delay, kv_xfer_params

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to hold its full sequence and be promoted."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_tokens = full_num_tokens
        if self.num_spec_tokens:
            num_tokens += 1 + self.num_spec_tokens + self.num_lookahead_tokens
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )

    def _set_kv_fetch_stage(self, request: Request, stage: str | None) -> None:
        kv_fetch_stages = self._kv_fetch_stages
        if kv_fetch_stages is not None:
            prev_stage = kv_fetch_stages.pop(request, None)
            if prev_stage is not None:
                self._kv_fetch_counts[prev_stage] -= 1
            if stage is not None:
                kv_fetch_stages[request] = stage
                self._kv_fetch_counts[stage] += 1

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _load_restores_replay_window(self, request: Request) -> bool:
        """Whether the KV connector's load for ``request`` restores every DSV41
        SWA bounded-replay group, as a P/D transfer of the request's own blocks
        does; a prefix-cache store holds no window."""
        assert self.connector is not None
        return bool(self.prefix_replay_group_ids) and set(
            self.prefix_replay_group_ids
        ).issubset(self.connector.get_loaded_kv_cache_group_ids(request))

    def _mark_prefix_replay(
        self, request: Request, num_hit_tokens: int, window_loaded: bool
    ) -> int:
        """Record where a prefix hit's replay starts and return the number of
        hit tokens to recompute; the caller rewinds the computed count by it.

        The replayed tokens are the hit's last window (see
        ``Request.replay_start``): the worker rebuilds their sliding-window KV
        and leaves their cached KV alone. A hit whose KV load carries the
        window replays nothing.
        """
        if not self.prefix_replay_tokens:
            return 0
        if window_loaded:
            request.replay_start = 0
            return 0
        num_replay_tokens = min(self.prefix_replay_tokens, num_hit_tokens)
        request.replay_start = num_hit_tokens - num_replay_tokens
        return num_replay_tokens

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        load_failed = request.request_id in self.failed_recving_kv_req_ids
        if load_failed:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

        # DSV41 SWA bounded replay recomputes the tail of a hit loaded without
        # its window, or of a failed load, which may have lost it; the replay
        # covers the last token. Otherwise a full prompt hit re-computes that
        # token so the next one can be sampled.
        num_replay_tokens = self._mark_prefix_replay(
            request,
            request.num_computed_tokens,
            window_loaded=not load_failed
            and self._load_restores_replay_window(request),
        )
        if num_replay_tokens > 0:
            request.num_computed_tokens -= num_replay_tokens
        elif request.num_computed_tokens == request.num_tokens:
            request.num_computed_tokens = request.num_tokens - 1
        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        """Returns True if scheduling of this request should proceed, False if
        it should be skipped for this step."""
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # Check whether async loading requests are finished.
            request_id = request.request_id
            if request_id not in self.finished_recving_kv_req_ids:
                logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.", request_id)
                return False
            # Promote request to waiting or preempted.
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING

        elif request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            # Check whether grammar compilation has finished.
            struct_output_req = request.structured_output_request
            if not struct_output_req or struct_output_req.grammar is None:
                return False
            if isinstance(struct_output_req.grammar, Exception):
                self.grammar_compile_error_reqs[request.request_id] = RequestError(
                    code="validation_error",
                    message=f"Invalid structured outputs: {struct_output_req.grammar}",
                )
                return False
            # Grammar is ready, promote to waiting state.
            request.status = RequestStatus.WAITING

        elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            # Streaming input request waiting for more input.
            assert not request.streaming_queue
            return False

        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
                self._set_kv_fetch_stage(req, KV_FETCH_COMPLETED_WAITING)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.

        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # One block-id list per KV cache group; single-group layouts are
            # enforced by _handle_invalid_blocks.
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_failed_recving(self, failed_req_ids: set[str]) -> set[str]:
        """Fail closed for layouts whose block IDs are not globally unique."""
        affected_req_ids: set[str] = set()
        for req_id in failed_req_ids:
            request = self.requests.get(req_id)
            if (
                request is not None
                and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            ):
                affected_req_ids.add(req_id)
                if self.recompute_kv_load_failures:
                    request.num_computed_tokens = 0
                    self.failed_recving_kv_req_ids.add(req_id)
        return set() if self.recompute_kv_load_failures else affected_req_ids

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.

        """
        if len(self.kv_cache_config.kv_cache_groups) > 1:
            # Block IDs are only unique within a group, so a flat set of
            # invalid block IDs cannot be mapped back to requests.
            raise RuntimeError(
                "A KV connector reported block-level load failures "
                "(invalid_block_ids) which is not supported for models "
                "with multiple KV cache groups. "
                "Connectors must report failed requests via "
                "KVConnectorTransferResults.failed_recving instead."
            )

        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.kv_holding_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
