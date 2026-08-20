import json
import pytest
from pathlib import Path
from report.generate import load_results, cost_rows, median_rows, render_markdown_table
from gppb.models import (
    BenchResult, Target, Hardware, Pricing, Timings, Workload, StepResult, Stats,
)


def _step(tps: float) -> StepResult:
    s = Stats(p50=1.0, p90=1.0, min=1.0, max=1.0)
    return StepResult(
        concurrency=64, requests_completed=256, requests_failed=0, wall_seconds=10.0,
        output_tokens_total=65536, output_tokens_per_sec=tps, ttft_ms=s, tpot_ms=s,
    )


def _vllm(run_id, tps, run_index=1, valid=True) -> BenchResult:
    return BenchResult(
        run_id=run_id, valid=valid,
        target=Target(kind="vllm", model="Qwen/Qwen3.8-27B", precision="fp8", tp_size=1),
        hardware=Hardware(gpu_name="NVIDIA H100 80GB HBM3"),
        pricing=Pricing(hourly_rate_usd=1.90),
        timings=Timings(boot_seconds=78.1), workload=Workload(),
        run_index=run_index, steps=[_step(tps)],
    )


def _api(provider) -> BenchResult:
    return BenchResult(
        run_id=f"or-{provider}",
        target=Target(kind="openrouter", model="qwen/qwen3.8-27b", provider=provider),
        hardware=Hardware(gpu_name="managed", gpu_count=0),
        pricing=Pricing(input_per_mtok_usd=0.40, output_per_mtok_usd=3.00),
        timings=Timings(), workload=Workload(), steps=[_step(120.0)],
    )


def test_load_results_skips_invalid_runs(tmp_path):
    (tmp_path / "a.json").write_text(_vllm("a", 1000.0).model_dump_json())
    (tmp_path / "b.json").write_text(_vllm("b", 1000.0, valid=False).model_dump_json())
    loaded = load_results(tmp_path)
    assert [r.run_id for r in loaded] == ["a"]


def test_cost_rows_uses_the_throughput_knee():
    rows = cost_rows([_vllm("a", 1000.0)])
    assert rows[0].tokens_per_sec == 1000.0
    assert rows[0].usd_per_mtok == pytest.approx(0.527777, rel=1e-5)


def test_cost_rows_computes_coldstart_for_selfhost_only():
    rows = cost_rows([_vllm("a", 1000.0), _api("chutes")])
    by_label = {r.label: r for r in rows}
    assert by_label["NVIDIA H100 80GB HBM3 TP1"].coldstart_usd > 0
    assert by_label["chutes"].coldstart_usd is None


def test_api_rows_use_blended_pricing():
    rows = cost_rows([_api("chutes")])
    assert rows[0].usd_per_mtok == pytest.approx(4.6, rel=1e-6)


def test_median_rows_collapses_repeat_runs():
    rows = cost_rows([
        _vllm("a", 900.0, run_index=1),
        _vllm("b", 1000.0, run_index=2),
        _vllm("c", 1100.0, run_index=3),
    ])
    collapsed = median_rows(rows)
    assert len(collapsed) == 1
    assert collapsed[0].tokens_per_sec == 1000.0


def test_markdown_table_renders_every_row():
    table = render_markdown_table(median_rows(cost_rows([_vllm("a", 1000.0), _api("chutes")])))
    assert "NVIDIA H100 80GB HBM3 TP1" in table
    assert "chutes" in table
    assert "$/1M" in table
