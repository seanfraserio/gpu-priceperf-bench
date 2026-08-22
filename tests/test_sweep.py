import httpx
import pytest
from gppb.sweep import stats_from, run_sweep, run_step
from gppb.models import StepResult, Workload


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


async def test_the_client_pool_never_caps_the_concurrency_under_test():
    """httpx.AsyncClient defaults to max_connections=100, so levels above that
    measured the client's connection pool rather than the server.

    This is how it showed up in real data: TTFT at concurrency 128 jumped from
    671ms to 11829ms and throughput fell, which read as a textbook saturation
    knee. It was requests queueing for a socket on the laptop."""
    seen: list[int] = []

    class _Recorder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, text="data: [DONE]\n\n")

    original = httpx.AsyncClient.__init__

    ABSENT = object()

    def spy(self, *args, **kwargs):
        # Absent limits and limits=None are not the same thing: absent means
        # httpx's default of 100, which is the bug. Only an explicit setting
        # counts as configured.
        seen.append(kwargs.get("limits", ABSENT))
        return original(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = spy
    try:
        await run_step(
            "http://test", "m", concurrency=256, workload=Workload(),
            requests_per_step=1,
        )
    except Exception:
        pass
    finally:
        httpx.AsyncClient.__init__ = original

    assert seen, "run_step must build a client"
    assert seen[0] is not ABSENT, (
        "run_step used httpx's default pool of 100 connections while measuring "
        "concurrency 256"
    )
    cap = getattr(seen[0], "max_connections", None)
    assert cap is None or cap >= 256, (
        f"pool capped at {cap} while measuring concurrency 256"
    )


async def test_sweep_stops_climbing_once_throughput_has_turned_over(monkeypatch):
    """Past the peak there is nothing left to find, and the levels above it are
    the most expensive to run: level 512 submits 2048 requests, and on a card
    whose KV cache holds ~62 at a time the rest simply queue.

    Two consecutive declines are required so that one noisy level cannot end
    the sweep early."""
    curve = {1: 100.0, 2: 200.0, 4: 400.0, 8: 800.0, 16: 780.0, 32: 700.0,
             64: 650.0, 128: 600.0}
    seen: list[int] = []

    async def fake_step(base_url, model, concurrency, workload, api_key=None,
                        extra_body=None, requests_per_step=0):
        seen.append(concurrency)
        return StepResult(
            concurrency=concurrency, requests_completed=4, requests_failed=0,
            wall_seconds=1.0, output_tokens_total=100,
            output_tokens_per_sec=curve[concurrency],
            ttft_ms=stats_from([1.0]), tpot_ms=stats_from([1.0]),
        )

    monkeypatch.setattr("gppb.sweep.run_step", fake_step)

    async def on_step(steps):
        return None

    steps = await run_sweep(
        "http://test", "m", [1, 2, 4, 8, 16, 32, 64, 128], Workload(), on_step,
    )
    assert seen == [1, 2, 4, 8, 16, 32], "should stop after two declines"
    assert len(steps) == 6


async def test_a_single_dip_does_not_end_the_sweep(monkeypatch):
    """Run-to-run noise at one level must not be mistaken for the peak."""
    curve = {1: 100.0, 2: 90.0, 4: 400.0, 8: 800.0}
    seen: list[int] = []

    async def fake_step(base_url, model, concurrency, workload, api_key=None,
                        extra_body=None, requests_per_step=0):
        seen.append(concurrency)
        return StepResult(
            concurrency=concurrency, requests_completed=4, requests_failed=0,
            wall_seconds=1.0, output_tokens_total=100,
            output_tokens_per_sec=curve[concurrency],
            ttft_ms=stats_from([1.0]), tpot_ms=stats_from([1.0]),
        )

    monkeypatch.setattr("gppb.sweep.run_step", fake_step)

    async def on_step(steps):
        return None

    await run_sweep("http://test", "m", [1, 2, 4, 8], Workload(), on_step)
    assert seen == [1, 2, 4, 8]
