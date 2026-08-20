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

    async with httpx.AsyncClient() as client:
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
) -> list[StepResult]:
    steps: list[StepResult] = []
    for level in levels:
        steps.append(
            await run_step(base_url, model, level, workload, api_key, extra_body)
        )
        await on_step(steps)
    return steps
