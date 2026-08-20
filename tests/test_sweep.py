import pytest
from gppb.sweep import stats_from, run_sweep, run_step
from gppb.models import Workload


def test_stats_from_computes_percentiles():
    s = stats_from([10.0, 20.0, 30.0, 40.0])
    assert s.min == 10.0
    assert s.max == 40.0
    assert 20.0 <= s.p50 <= 30.0
    assert s.p90 >= s.p50


def test_stats_from_handles_single_value():
    s = stats_from([7.0])
    assert s.p50 == s.p90 == s.min == s.max == 7.0


def test_stats_from_rejects_empty():
    with pytest.raises(ValueError):
        stats_from([])


async def test_run_step_aggregates_throughput(monkeypatch):
    from gppb import sweep
    from gppb.client import RequestMetrics

    async def fake_stream_one(client, base_url, model, prompt, workload, api_key=None, extra_body=None):
        return RequestMetrics(ttft_ms=100.0, tpot_ms=10.0, output_tokens=256, total_ms=2660.0)

    monkeypatch.setattr(sweep, "stream_one", fake_stream_one)
    step = await run_step("http://x", "m", concurrency=4, workload=Workload(), requests_per_step=8)

    assert step.concurrency == 4
    assert step.requests_completed == 8
    assert step.requests_failed == 0
    assert step.output_tokens_total == 8 * 256
    assert step.output_tokens_per_sec > 0


async def test_run_step_counts_failures_separately(monkeypatch):
    from gppb import sweep
    from gppb.client import RequestMetrics
    calls = {"n": 0}

    async def flaky(client, base_url, model, prompt, workload, api_key=None, extra_body=None):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return RequestMetrics(0.0, 0.0, 0, 5.0, error="HTTP 503")
        return RequestMetrics(100.0, 10.0, 256, 2660.0)

    monkeypatch.setattr(sweep, "stream_one", flaky)
    step = await run_step("http://x", "m", concurrency=2, workload=Workload(), requests_per_step=4)

    assert step.requests_completed == 2
    assert step.requests_failed == 2
    assert step.output_tokens_total == 2 * 256


async def test_run_sweep_invokes_callback_after_every_level(monkeypatch):
    from gppb import sweep
    from gppb.client import RequestMetrics

    async def fake_stream_one(client, base_url, model, prompt, workload, api_key=None, extra_body=None):
        return RequestMetrics(100.0, 10.0, 256, 2660.0)

    monkeypatch.setattr(sweep, "stream_one", fake_stream_one)
    seen: list[int] = []

    async def on_step(steps):
        seen.append(len(steps))

    steps = await run_sweep("http://x", "m", [1, 2, 4], Workload(), on_step)
    assert len(steps) == 3
    assert seen == [1, 2, 3], "partial results must be emitted after each level"


async def test_run_sweep_calls_before_step_ahead_of_each_level(monkeypatch):
    """Budget guards need a hook that runs before the requests, not after."""
    from gppb import sweep
    from gppb.client import RequestMetrics

    order: list[str] = []

    async def fake_stream_one(client, base_url, model, prompt, workload, api_key=None, extra_body=None):
        order.append("request")
        return RequestMetrics(100.0, 10.0, 256, 2660.0)

    monkeypatch.setattr(sweep, "stream_one", fake_stream_one)

    async def before_step(level):
        order.append(f"before-{level}")

    async def on_step(steps):
        order.append("after")

    await run_sweep("http://x", "m", [1], Workload(), on_step, before_step=before_step)
    assert order[0] == "before-1"
    assert order[-1] == "after"
