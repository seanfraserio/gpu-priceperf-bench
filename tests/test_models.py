import json
from gppb.models import BenchResult, StepResult, Stats, Target, Hardware, Pricing, Timings, Workload


def _minimal_result() -> BenchResult:
    return BenchResult(
        run_id="test-run-1",
        target=Target(kind="vllm", model="Qwen/Qwen3.8-27B", precision="fp8", tp_size=1),
        hardware=Hardware(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=1),
        pricing=Pricing(hourly_rate_usd=1.90),
        timings=Timings(download_seconds=500.0, boot_seconds=78.1),
        workload=Workload(),
        run_index=1,
        steps=[],
    )


def test_defaults_encode_the_global_constraints():
    w = Workload()
    assert w.input_tokens == 1024
    assert w.output_tokens == 256
    assert w.temperature == 0.0
    assert w.ignore_eos is True
    assert w.max_model_len == 32768


def test_result_roundtrips_through_json():
    original = _minimal_result()
    restored = BenchResult.model_validate_json(original.model_dump_json())
    assert restored == original


def test_result_defaults_to_valid_and_complete():
    r = _minimal_result()
    assert r.valid is True
    assert r.partial is False
    assert r.invalid_reason is None


def test_openrouter_target_carries_a_provider():
    t = Target(kind="openrouter", model="qwen/qwen3.8-27b", provider="chutes")
    assert t.provider == "chutes"
    assert t.tp_size is None


def test_step_result_holds_latency_stats():
    step = StepResult(
        concurrency=8,
        requests_completed=40,
        requests_failed=0,
        wall_seconds=33.2,
        output_tokens_total=10240,
        output_tokens_per_sec=308.4,
        ttft_ms=Stats(p50=120.0, p90=180.0, min=95.0, max=210.0),
        tpot_ms=Stats(p50=12.0, p90=15.0, min=10.0, max=22.0),
    )
    assert step.output_tokens_total == 40 * 256
