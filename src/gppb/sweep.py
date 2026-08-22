"""Concurrency sweep. Emits partial results after every level so a spot
preemption at level 5 of 9 still yields five usable points."""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import Awaitable, Callable

import httpx

from gppb.client import stream_one
from gppb.corpus import build_corpus
from gppb.models import Stats, StepResult, Workload

# Consecutive falling levels that count as "the peak is behind us".
MAX_CONSECUTIVE_DECLINES = 2


def stats_from(values: list[float]) -> Stats:
    if not values:
        raise ValueError("cannot compute stats over an empty sample")
    ordered = sorted(values)
    return Stats(
        p50=statistics.median(ordered),
        p90=ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)],
        min=ordered[0],
        max=ordered[-1],
    )


async def run_step(
    base_url: str,
    model: str,
    concurrency: int,
    workload: Workload,
    api_key: str | None = None,
    extra_body: dict | None = None,
    requests_per_step: int = 0,
) -> StepResult:
    """Saturate the server at one concurrency level and measure steady state."""
    total_requests = requests_per_step or concurrency * 4
    prompts = build_corpus(total_requests, workload.input_tokens)
    semaphore = asyncio.Semaphore(concurrency)

    # httpx defaults to 100 connections. Every level above that silently
    # measured the client's pool instead of the server: requests queued on the
    # laptop, TTFT at concurrency 128 rose from 671ms to 11829ms, and the
    # resulting fall in throughput read as a hardware saturation knee.
    limits = httpx.Limits(
        max_connections=concurrency + 16,
        max_keepalive_connections=concurrency + 16,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        async def one(prompt: str):
            async with semaphore:
                return await stream_one(
                    client, base_url, model, prompt, workload, api_key, extra_body
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(one(p) for p in prompts))
        wall_seconds = time.perf_counter() - started

    ok = [r for r in results if r.error is None]
    failed = len(results) - len(ok)
    output_tokens_total = sum(r.output_tokens for r in ok)

    # A level where everything failed still gets recorded — a zeroed row is
    # data, and deleting it would silently flatter the provider.
    ttfts = [r.ttft_ms for r in ok] or [0.0]
    tpots = [r.tpot_ms for r in ok] or [0.0]

    return StepResult(
        concurrency=concurrency,
        requests_completed=len(ok),
        requests_failed=failed,
        wall_seconds=wall_seconds,
        output_tokens_total=output_tokens_total,
        output_tokens_per_sec=output_tokens_total / wall_seconds if wall_seconds else 0.0,
        ttft_ms=stats_from(ttfts),
        tpot_ms=stats_from(tpots),
    )


async def run_sweep(
    base_url: str,
    model: str,
    levels: list[int],
    workload: Workload,
    on_step: Callable[[list[StepResult]], Awaitable[None]],
    api_key: str | None = None,
    extra_body: dict | None = None,
    before_step: Callable[[int], Awaitable[None]] | None = None,
) -> list[StepResult]:
    """`before_step` runs ahead of a level's requests — budget guards belong
    there, where refusing still prevents the spend."""
    steps: list[StepResult] = []
    declines = 0
    for level in levels:
        if before_step is not None:
            await before_step(level)
        steps.append(
            await run_step(base_url, model, level, workload, api_key, extra_body)
        )
        await on_step(steps)

        # Past the peak there is nothing left to find, and the levels above it
        # cost the most: level 512 submits 2048 requests, and on a card whose
        # KV cache holds ~62 at a time the rest only queue. Two consecutive
        # declines are required so one noisy level cannot end the sweep early.
        if len(steps) >= 2:
            if steps[-1].output_tokens_per_sec < steps[-2].output_tokens_per_sec:
                declines += 1
                if declines >= MAX_CONSECUTIVE_DECLINES:
                    break
            else:
                declines = 0
    return steps
