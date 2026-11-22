#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Guarded four-rank numerical, reuse, timing and missing-rank controls.

Run with a memory guard and torchrun --standalone --nproc-per-node=4, only on
idle Asgard. A poisoned communicator is discarded and never reused for work.
"""
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.gfx1100_peer_push import (
    Gfx1100PeerPush,
    check_peer_push_fault,
)


def gather(item):
    out = [None] * 4
    dist.all_gather_object(out, item)
    return out


def consensus(ok, label):
    values = gather(bool(ok))
    if not all(values):
        raise AssertionError((label, values))


def report(rank, **entry):
    if rank == 0:
        print(json.dumps(entry, sort_keys=True), flush=True)


def bits_match(x, y):
    return torch.equal(x.view(torch.int16), y.view(torch.int16))


def failure_control(device, rank):
    plan = Gfx1100PeerPush(dist.group.WORLD, device, 32)
    x = torch.ones((8, 2304), dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize(); dist.barrier()
    t = time.monotonic()
    y = plan.reduce(x) if rank != 3 else None
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t
    dist.barrier()
    status = plan.snapshot_async()
    torch.cuda.synchronize()
    consensus(bool(status.count_nonzero()) and elapsed < 5, "sticky fault visible on all four GPUs")
    consensus(y is None or bool((y == 0).all()), "defined placeholder for missing rank")
    try:
        check_peer_push_fault(status)
    except RuntimeError as exc:
        if "Discard this engine" not in str(exc):
            raise
    else:
        raise AssertionError("missing-rank result was not rejected")
    report(rank, kind="missing_rank", receipts=gather({"elapsed_s": elapsed, "status": status.tolist()}))
    # Poisoned arena must return a defined result without waiting on peers.
    # Deliberately use a different shape to prove that the latch isn't per-shape.
    t = time.monotonic()
    y = plan.reduce(x[:1]) if rank == 0 else None
    torch.cuda.synchronize()
    consensus((y is None or bool((y == 0).all())) and time.monotonic() - t < .2, "latch across shapes")
    plan.close()


def graph(plan, fn, ca=None, steps=1):
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    dist.barrier()
    ctx = ca.capture() if ca is not None else plan.capture(s) if plan else nullcontext()
    with ctx, torch.cuda.graph(g, stream=s):
        for _ in range(steps):
            fn()
    dist.barrier()
    return g


def gpu_check(device, rank):
    plan = Gfx1100PeerPush(dist.group.WORLD, device, 32)
    ca = CustomAllreduce(dist.group.WORLD, device, max_size=1 << 20)
    consensus(not ca.disabled, "native CustomAllreduce required as reference")
    try:
        bufs = {r: torch.empty((r, 2304), dtype=torch.bfloat16, device=device) for r in (1, 8, 32)}
        push, pull, out_pair = {}, {}, {}
        torch.manual_seed(711 + rank)
        for rows, x in bufs.items():
            x.normal_()
            torch.cuda.synchronize(); dist.barrier()
            p = plan.reduce(x); c = ca.all_reduce(x)
            consensus(bits_match(p, c), "eager bitwise")
            vals = []
            def fn_p(xx=x):
                vals.append(plan.reduce(xx))
            def fn_c(xx=x):
                vals.append(ca.all_reduce(xx))
            push[rows] = graph(plan, fn_p)
            pull[rows] = graph(None, fn_c, ca)
            out_pair[rows] = vals
        # Reuse the same per-shape arena across non-monotone graph calls and
        # constantly change the payload to catch stale flags or peer reads.
        for i in range(36):
            rows = [8, 1, 32, 8, 1, 32][i % 6]
            x = bufs[rows]; x.normal_()
            if i % 2 == 0:
                x[:, ::3].mul_(1024 if rank in (0, 2) else .001953125)
                x[:, 1::31].fill_(-0.)
            push[rows].replay(); pull[rows].replay()
            torch.cuda.synchronize()
            consensus(bits_match(*out_pair[rows]), ("same bits across mixed shapes/replays", i))
        # NaN/Inf and negative zeros, compared to the deployed GPU arithmetic.
        x = bufs[8]
        x[:, :4] = torch.tensor([float("nan"), float("inf"), -0., float("-inf")], device=device)
        x[:, 4:8].fill_(float("-inf") if rank == 0 else float("inf"))
        p = plan.reduce(x); q = ca.all_reduce(x)
        consensus(bits_match(p, q), "same GPU special-value reduction")
        status = plan.snapshot_async(); torch.cuda.synchronize()
        consensus(not bool(status.count_nonzero()), "healthy graph copies leave fault clear")
        report(rank, kind="mixed-shape-and-special", changing_payload_replays=36, passed=True)

        for rows, x in bufs.items():
            x.normal_()
            for g in (pull[rows], push[rows]):
                for _ in range(6):
                    g.replay()
                torch.cuda.synchronize()
            # Compare the actual production API cost on graphs; remember the
            # cumulative dispatches carry no host barrier inside the graph.
            pg = graph(plan, lambda: plan.reduce(x), steps=16)
            cg = graph(None, lambda: ca.all_reduce(x), ca, steps=16)
            ms = [[], []]
            for repeat in range(10):
                for n in ((0, 1) if not repeat % 2 else (1, 0)):
                    a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    dist.barrier()
                    a.record(); (cg, pg)[n].replay(); b.record(); b.synchronize()
                    ms[n].append(max(gather(a.elapsed_time(b))) * 1000 / 16)
            report(rank, kind="timing", rows=rows, order=["copy_pull", "peer_push"],
                   median_max_rank_us=[statistics.median(m) for m in ms])
            del pg, cg
    finally:
        # Ensure all native readers are done before tearing down imports/exports.
        push.clear(); pull.clear(); out_pair.clear()
        torch.cuda.synchronize()
        plan.close()
        dist.barrier(); ca.close(); dist.barrier()


def main():
    # Test cannot be used as an unguarded build smoke. The caller also must
    # require the 96 GiB/no-swap lifetime guard and startup free check.
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0,1,2,3" or torch.cuda.device_count() != 4:
        raise RuntimeError("use exactly the four dedicated XTXs")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("requires four ranks")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 22 * 2**30:
        raise RuntimeError("GPU not idle")
    torch.cuda.set_per_process_memory_fraction(.15, device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=75))
    try:
        failure_control(device, rank)
        gpu_check(device, rank)
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print("GFX1100 PEER PUSH NATIVE CONTRACT PASS", flush=True)


if __name__ == "__main__":
    main()
