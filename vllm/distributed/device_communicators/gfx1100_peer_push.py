# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact peer writes for the Inkling target on same-host gfx1100 TP4.

Only an audited worker owns these arenas. Do not use this for arbitrary
multi-stream collectives, draft kernels or tensor parallel groups.
"""

import ctypes as C
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)
_WORLD = 4
_HIDDEN = 2304
_SO = Path(__file__).resolve().parents[2] / "_gfx1100_peer_push.so"


def _load_native(path: Path) -> tuple[C.CDLL, str]:
    """Require the built ABI; never fall back rank-locally on setup failure."""
    if not path.is_file():
        raise RuntimeError(f"{path} missing; run tools/asgard/build_gfx1100_peer_push.sh")
    lib = C.CDLL(str(path))
    p, i, size = C.c_void_p, C.c_int, C.c_size_t
    spec = {
        "abi_version": ([], i),
        "source_sha256": ([], C.c_char_p),
        "handle_bytes": ([], size),
        "probe": ([p, i, i], i),
        "alloc": ([C.POINTER(p), p], i),
        "free": ([p], i),
        "connect": ([C.POINTER(p), p, p, i], i),
        "disconnect": ([p], i),
        "reduce": ([p, p, p, i, size], i),
        "snapshot": ([p, p, size], i),
    }
    for name, (args, result) in spec.items():
        fn = getattr(lib, f"vllm_pp_{name}")
        fn.argtypes = args
        fn.restype = result
    if lib.vllm_pp_abi_version() != 1:
        raise RuntimeError("wrong gfx1100 peer-push ABI; rebuild the native module")
    digest = lib.vllm_pp_source_sha256().decode("ascii")
    source = path.parent.parent / "csrc/rocm/gfx1100_peer_push.hip"
    # If run from a source checkout, verify that it hasn't been rebuilt from a
    # different tree. Package installations can still validate ABI + consensus.
    if len(digest) != 64 or (source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != digest):
        raise RuntimeError("stale gfx1100 peer-push native module; rebuild it from this tree")
    return lib, digest


def _checked(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"gfx1100 peer push {what} failed: HIP error {rc}")


def _gather(group: dist.ProcessGroup, value):
    results = [None] * _WORLD
    # Gloo object collectives allocate mutable CPU tensors.
    with torch.inference_mode(False):
        dist.all_gather_object(results, value, group=group)
    return results


def _setup_error(stage: str, outcomes) -> RuntimeError:
    return RuntimeError(f"gfx1100 peer push TP4 {stage} rejected: {outcomes}")


def check_peer_push_fault(ready_cpu_snapshot: torch.Tensor) -> None:
    """Call only after the sampling copy event; never return poisoned output."""
    codes = ready_cpu_snapshot.tolist()
    if any(codes):
        raise RuntimeError(
            "Fault detected in Inkling gfx1100 TP4 peer push: sticky rank codes "
            f"{codes}. Discard this engine; the collective cannot be retried."
        )


class Gfx1100PeerPush:
    def __init__(self, group: dist.ProcessGroup, device: torch.device, max_rows: int):
        self.group = group
        self.device = device
        self.max_rows = max_rows
        self.rank = dist.get_rank(group=group)
        self.local, self.ctx = C.c_void_p(), C.c_void_p()
        self.capture_stream: int | None = None
        self.closed = False
        if dist.get_world_size(group) != _WORLD or dist.get_backend(group) != "gloo":
            raise ValueError("gfx1100 peer push requires a four-rank CPU Gloo group")
        from vllm.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group, source_rank=0)):
            raise RuntimeError("gfx1100 peer push does not support remote GPU ranks")

        error = None
        bdf, digest, handle_len = "", "", 0
        try:
            if max_rows not in (8, 32) or device.index is None:
                raise ValueError("expected explicit device and max_rows 8 or 32")
            if torch.cuda.current_device() != device.index:
                raise ValueError("expected the worker's current GPU device")
            self.main_stream = torch.cuda.current_stream(device).cuda_stream
            self.lib, digest = _load_native(_SO)
            raw_bdf = C.create_string_buffer(32)
            _checked(self.lib.vllm_pp_probe(raw_bdf, len(raw_bdf), device.index), "P2P probe")
            bdf = raw_bdf.value.decode("ascii")
            handle_len = self.lib.vllm_pp_handle_bytes()
            if not bdf or not 0 < handle_len <= 256:
                raise ValueError("invalid HIP PCI identity or IPC handle length")
        except Exception as exc:
            error = str(exc)
        results = _gather(group, (error, bdf, digest, handle_len, max_rows))
        if (any(row[0] is not None for row in results)
                or len({row[1] for row in results}) != _WORLD
                or len({tuple(row[2:]) for row in results}) != 1):
            raise _setup_error("topology/configuration", results)

        handle = C.create_string_buffer(handle_len)
        rc = self.lib.vllm_pp_alloc(C.byref(self.local), handle)
        allocation = _gather(group, (rc, handle.raw if rc == 0 else b""))
        if any(code for code, _ in allocation):
            if self.local:
                self.lib.vllm_pp_free(self.local)
            raise _setup_error("allocation", [code for code, _ in allocation])
        handles = C.create_string_buffer(b"".join(h for _, h in allocation))
        rc = self.lib.vllm_pp_connect(C.byref(self.ctx), self.local, handles, self.rank)
        connected = _gather(group, rc)
        if any(connected):
            self.close()
            raise _setup_error("IPC import", connected)
        if self.rank == 0:
            logger.info("Inkling gfx1100 peer push: max_rows=%d, logical-rank PCI devices=%s, build=%s",
                        max_rows, [r[1] for r in results], digest)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream) -> Iterator[None]:
        # vLLM's coordinated capture stream waits on the main stream. Replay
        # runs only on the same worker's ordered main compute stream.
        if self.capture_stream is not None:
            raise RuntimeError("nested peer-push capture is not supported")
        self.capture_stream = stream.cuda_stream
        try:
            yield
        finally:
            self.capture_stream = None

    def accepts(self, x: torch.Tensor) -> bool:
        return (
            x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == _HIDDEN
            and (x.shape[0] in (1, 8) or (self.max_rows == 32 and x.shape[0] == 32))
            and x.is_contiguous() and x.device == self.device
        )

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self.closed or stream not in (self.main_stream, self.capture_stream):
            raise RuntimeError("gfx1100 peer push requires the ordered worker stream")
        if x.data_ptr() % 16:
            # A local alignment fix, not a rank-dependent backend switch.
            x = x.clone()
        y = torch.empty_like(x)
        _checked(self.lib.vllm_pp_reduce(self.ctx, x.data_ptr(), y.data_ptr(), x.shape[0], stream),
                 "reduce launch")
        return y

    def snapshot_async(self) -> torch.Tensor:
        """Enqueue on the caller's copy stream, which already waits on target work."""
        # Use a PyTorch-owned device temp and PyTorch's nonblocking D2H copy
        # so that pinned memory is tracked even on ranks whose result is dropped.
        stream = torch.cuda.current_stream(self.device).cuda_stream
        device_status = torch.empty((_WORLD,), dtype=torch.int32, device=self.device)
        _checked(self.lib.vllm_pp_snapshot(self.ctx, device_status.data_ptr(), stream), "fault copy")
        return device_status.to("cpu", non_blocking=True)

    def close(self) -> None:
        """Called after graphs are destroyed and local work has drained."""
        if self.closed:
            return
        self.closed = True
        try:
            # Never free an export while any peer might be using its mapping.
            # If a peer has already exited, native process exit owns reclamation.
            how_long = timedelta(seconds=4)
            dist.monitored_barrier(self.group, timeout=how_long, wait_all_ranks=True)
            if self.ctx:
                _checked(self.lib.vllm_pp_disconnect(self.ctx), "IPC close")
                self.ctx = C.c_void_p()
            dist.monitored_barrier(self.group, timeout=how_long, wait_all_ranks=True)
            if self.local:
                _checked(self.lib.vllm_pp_free(self.local), "free")
                self.local = C.c_void_p()
        except Exception as exc:
            logger.warning("Leaving peer-push arenas to GPU worker process exit: %s", exc)


def create_inkling_peer_push(config: "VllmConfig", tp: "GroupCoordinator",
                             device: torch.device, max_rows: int) -> Gfx1100PeerPush:
    from vllm import envs
    from vllm.config.compilation import CompilationMode
    from vllm.models.inkling2.configs import INKLING2_ARCHITECTURES
    from vllm.platforms import current_platform

    p = config.parallel_config
    m = config.model_config
    ca = getattr(tp.device_communicator, "ca_comm", None)
    if not (
        current_platform.is_rocm() and INKLING2_ARCHITECTURES.intersection(m.architectures)
        and m.runner_type == "generate" and not m.enable_sleep_mode
        and p.world_size == p.tensor_parallel_size == _WORLD
        and p.data_parallel_size == p.pipeline_parallel_size == 1
        and p.prefill_context_parallel_size == p.decode_context_parallel_size == 1
        and not p.use_ubatching and not p.enable_elastic_ep
        and not envs.VLLM_BATCH_INVARIANT and not config.use_cumem_cudagraph_pool
        and config.compilation_config.mode == CompilationMode.NONE
        and ca is not None and not ca.disabled
    ):
        raise RuntimeError(
            "Inkling peer push requires ROCm Inkling generate, single-host TP4, "
            "ordinary custom AR, compilation mode 0, no PP/DP/CP/DBO/sleep or batch-invariance"
        )
    return Gfx1100PeerPush(tp.cpu_group, device, max_rows)
