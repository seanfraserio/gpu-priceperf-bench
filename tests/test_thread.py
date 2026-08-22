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
        steps=[_step(1, tps / 4), _step(8, tps), _step(16, tps * 0.95)],
    )


def _api(provider: str, out_per_mtok: float) -> BenchResult:
    return BenchResult(
        run_id=f"or-{provider}-0001",
        target=Target(kind="openrouter", model="Qwen/Qwen3-8B", provider=provider),
        hardware=Hardware(gpu_name="n/a", gpu_count=0),
        pricing=Pricing(input_per_mtok_usd=out_per_mtok / 4, output_per_mtok_usd=out_per_mtok),
        timings=Timings(), workload=Workload(), run_index=1, partial=False,
        steps=[_step(4, 150.0), _step(8, 200.0), _step(16, 190.0)],
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


def _short(gpu: str, rate: float, tps: float) -> BenchResult:
    """A run that stopped while throughput was still climbing."""
    r = _selfhost(gpu, rate, tps)
    r.steps = [_step(1, tps / 4), _step(4, tps)]
    return r


def test_an_upper_bound_never_becomes_a_headline_claim():
    """A run that never found its ceiling is not a measurement of that GPU. On
    real data the generator compared a 5090 against the same 5090 swept less
    deeply and announced a 5.0x gap — a number about the sweep, not the
    hardware."""
    full = _selfhost("RTX 5090", 0.35, 1568.0)
    short = _short("RTX 5090", 0.35, 325.0)
    thread = build_thread([full, short])
    rendered = "\n".join(p.text for p in thread)
    assert "≤" not in rendered, "unsaturated rows must not reach the thread"
    assert "x more" not in rendered, "no gap claim against an upper bound"


def test_a_thread_with_no_saturated_run_refuses_to_render():
    """Every run stopped while still climbing: there is no ceiling to quote."""
    with pytest.raises(ValueError):
        build_thread([_short("RTX 5090", 0.35, 325.0)])


def _two_model_results():
    """Six tiers across two models — the shape the real results directory has."""
    from gppb.models import Target
    out = []
    for gpu, model, tps in [
        ("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", 1563.0),
        ("NVIDIA H100 80GB HBM3", "Qwen/Qwen3-8B", 5313.0),
        ("NVIDIA A100-SXM4-80GB", "Qwen/Qwen3-8B", 1890.0),
        ("NVIDIA L40S", "Qwen/Qwen3-8B", 1273.0),
        ("NVIDIA H100 80GB HBM3", "Qwen/Qwen3.8-27B", 2031.0),
        ("NVIDIA A100-SXM4-80GB", "Qwen/Qwen3.8-27B", 515.0),
    ]:
        r = _selfhost(gpu, 1.0, tps)
        r.target = Target(kind="vllm", model=model, precision="fp8", tp_size=1)
        out.append(r)
    return out


def test_the_thread_survives_a_matrix_with_two_models():
    """Naming the model in each row pushed the table post to 322 characters and
    it stopped rendering at all. The labels have to be compact enough that the
    real matrix fits the platform limit."""
    from report.thread import build_thread, MAX_POST_CHARS

    for post in build_thread(_two_model_results()):
        assert len(post.text) <= MAX_POST_CHARS, post.text


def test_the_table_does_not_claim_one_model_for_rows_of_two():
    """The header read "$/1M output tokens, Qwen/Qwen3.8-27B:" above rows that
    were mostly 8B — true of the first result, false of the table."""
    from report.thread import build_thread

    table = build_thread(_two_model_results())[1].text
    header = table.splitlines()[0]
    assert "Qwen3.8-27B" not in header, header
    assert "Qwen3-8B" in table and "Qwen3.8-27B" in table
