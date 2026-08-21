"""Runs that swept different concurrency levels are not the same measurement.

The sweep uploads a result after every level, so a preempted run leaves a short
result in the sink beside the complete ones. Collapsing those to a median by
label alone silently mixes a run that stopped at concurrency 4 with one that
reached 256 — and since the headline $/1M is quoted at peak throughput, the
short run drags the number in the direction that makes self-hosting look worse
than it measured. The blend has to be refused, not averaged."""
from __future__ import annotations

from gppb.cost import saturated
from gppb.models import BenchResult, Hardware, Pricing, Stats, StepResult, Target, Timings, Workload
from report.generate import cost_rows, median_rows


def _stats(v: float) -> Stats:
    return Stats(p50=v, p90=v, min=v, max=v)


def _step(concurrency: int, tps: float) -> StepResult:
    return StepResult(
        concurrency=concurrency, requests_completed=concurrency * 4,
        requests_failed=0, wall_seconds=10.0,
        output_tokens_total=int(tps * 10), output_tokens_per_sec=tps,
        ttft_ms=_stats(100.0), tpot_ms=_stats(10.0),
    )


def _result(gpu: str, steps: list[StepResult], run_id: str) -> BenchResult:
    return BenchResult(
        run_id=run_id,
        target=Target(kind="vllm", model="Qwen/Qwen3-8B", precision="bfloat16", tp_size=1),
        hardware=Hardware(gpu_name=gpu, gpu_count=1, vllm_version="0.27.1"),
        pricing=Pricing(hourly_rate_usd=0.35),
        timings=Timings(download_seconds=400.0, boot_seconds=340.0),
        workload=Workload(), run_index=1, partial=False, steps=steps,
    )


SHORT = [_step(1, 94.0), _step(2, 180.0), _step(4, 326.0)]
FULL = [_step(1, 94.0), _step(2, 180.0), _step(4, 326.0), _step(8, 575.0),
        _step(16, 900.0), _step(32, 1200.0), _step(64, 1150.0)]


def test_a_sweep_still_climbing_at_its_last_level_is_not_saturated():
    """Its peak is a floor on the hardware, not the hardware's ceiling, so the
    cost derived from it is an upper bound rather than a measurement."""
    assert saturated(SHORT) is False


def test_a_sweep_that_turned_over_is_saturated():
    """Throughput fell at 64 after peaking at 32 — the ceiling was found."""
    assert saturated(FULL) is True


def test_short_and_full_sweeps_are_not_medianed_together():
    rows = median_rows(cost_rows([
        _result("NVIDIA GeForce RTX 5090", SHORT, "vllm-a"),
        _result("NVIDIA GeForce RTX 5090", FULL, "vllm-b"),
        _result("NVIDIA GeForce RTX 5090", FULL, "vllm-c"),
    ]))
    assert len(rows) == 2, "a short sweep must not be blended into the full ones"
    by_saturation = {r.saturated: r for r in rows}
    # The full-sweep row must report the full sweep's peak, unpolluted.
    assert by_saturation[True].tokens_per_sec == 1200.0


def test_an_unsaturated_row_is_labelled_as_such():
    """A reader must be able to see which numbers are upper bounds."""
    rows = median_rows(cost_rows([
        _result("NVIDIA GeForce RTX 5090", SHORT, "vllm-a"),
        _result("NVIDIA GeForce RTX 5090", FULL, "vllm-b"),
    ]))
    unsaturated = [r for r in rows if not r.saturated]
    assert unsaturated and "≤4" in unsaturated[0].label


def test_repeat_runs_of_the_same_sweep_still_collapse_to_one_row():
    """The split must not defeat the reason median_rows exists."""
    rows = median_rows(cost_rows([
        _result("NVIDIA GeForce RTX 5090", FULL, f"vllm-{i}") for i in range(3)
    ]))
    assert len(rows) == 1
