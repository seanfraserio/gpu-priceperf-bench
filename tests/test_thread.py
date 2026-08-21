"""The X thread is generated, never typed.

A benchmark thread is the part of this project most likely to be read and least
likely to be checked, so the only defence against a flattering typo is that no
human ever writes a number into it. These tests exist to keep that true."""
from __future__ import annotations

import re

import pytest

from gppb.models import BenchResult, Hardware, Pricing, StepResult, Stats, Target, Timings, Workload
from report.thread import build_thread, render_thread


def _stats(v: float) -> Stats:
    return Stats(p50=v, p90=v, min=v, max=v)


def _step(concurrency: int, tps: float) -> StepResult:
    return StepResult(
        concurrency=concurrency, requests_completed=concurrency * 4,
        requests_failed=0, wall_seconds=10.0,
        output_tokens_total=int(tps * 10), output_tokens_per_sec=tps,
        ttft_ms=_stats(120.0), tpot_ms=_stats(9.0),
    )


def _selfhost(gpu: str, rate: float, tps: float) -> BenchResult:
    return BenchResult(
        run_id=f"vllm-{gpu}-tp1-{abs(hash(gpu)) % 10**8:08d}",
        target=Target(kind="vllm", model="Qwen/Qwen3-8B", precision="bfloat16", tp_size=1),
        hardware=Hardware(gpu_name=gpu, gpu_count=1, vllm_version="0.27.1"),
        pricing=Pricing(hourly_rate_usd=rate),
        timings=Timings(download_seconds=120.0, boot_seconds=300.0),
        workload=Workload(), run_index=1, partial=False,
        steps=[_step(1, tps / 4), _step(8, tps)],
    )


def _api(provider: str, out_per_mtok: float) -> BenchResult:
    return BenchResult(
        run_id=f"or-{provider}-0001",
        target=Target(kind="openrouter", model="Qwen/Qwen3-8B", provider=provider),
        hardware=Hardware(gpu_name="n/a", gpu_count=0),
        pricing=Pricing(input_per_mtok_usd=out_per_mtok / 4, output_per_mtok_usd=out_per_mtok),
        timings=Timings(), workload=Workload(), run_index=1, partial=False,
        steps=[_step(8, 200.0)],
    )


def test_thread_refuses_to_render_without_results():
    """Better no thread than a thread with invented numbers in it."""
    with pytest.raises(ValueError):
        build_thread([])


def test_every_number_in_the_thread_comes_from_a_result():
    """The guard against hand-typing: each rendered figure must be traceable to
    a value the harness actually measured or computed from measurements."""
    results = [_selfhost("RTX 5090", 0.34, 400.0), _api("together", 0.30)]
    thread = build_thread(results)
    rendered = "\n".join(post.text for post in thread)

    permitted = set()
    for row in thread:
        permitted.update(row.sources)
    for token in re.findall(r"\d+\.\d+", rendered):
        assert token in permitted, f"{token} appears in the thread but not in any result"


def test_thread_names_the_cheapest_and_says_by_how_much():
    """The headline claim is a ratio, and a ratio is exactly the kind of number
    that gets rounded in the author's favour when typed by hand."""
    cheap = _selfhost("RTX 5090", 0.34, 800.0)
    dear = _api("together", 3.00)
    thread = build_thread([cheap, dear])
    head = thread[0].text
    assert "RTX 5090" in head
    assert re.search(r"\d+(\.\d+)?x", head), "headline must quantify the gap"


def test_posts_fit_the_platform_limit():
    results = [_selfhost("RTX 5090", 0.34, 400.0), _api("together", 0.30)]
    for post in build_thread(results):
        assert len(post.text) <= 280, f"post is {len(post.text)} chars:\n{post.text}"


def test_rendered_thread_is_numbered_and_carries_the_method_note():
    """A thread without the caveats is a marketing claim, not a measurement."""
    results = [_selfhost("RTX 5090", 0.34, 400.0), _api("together", 0.30)]
    out = render_thread(build_thread(results))
    assert out.startswith("1/")
    assert "single run" in out.lower() or "run-to-run" in out.lower()
