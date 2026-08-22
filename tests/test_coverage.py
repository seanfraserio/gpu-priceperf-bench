"""Which matrix cells actually have usable results, and which still don't.

Runs fail. Two of the first four hosts never pulled the image, and the sweep
does not go back for them — so "21 runs planned" is not "21 results". Counting
coverage by hand across four tiers and two models is exactly the kind of thing
that quietly ends up with a two-sample median presented as three."""
from __future__ import annotations

import pytest

from gppb.models import BenchResult, Hardware, Pricing, Stats, StepResult, Target, Timings, Workload
from launch.coverage import coverage, missing, tier_for


def _stats(v: float) -> Stats:
    return Stats(p50=v, p90=v, min=v, max=v)


def _step(c: int, tps: float) -> StepResult:
    return StepResult(
        concurrency=c, requests_completed=c * 4, requests_failed=0,
        wall_seconds=10.0, output_tokens_total=int(tps * 10),
        output_tokens_per_sec=tps, ttft_ms=_stats(1.0), tpot_ms=_stats(1.0),
    )


SATURATED = [_step(1, 94.0), _step(8, 1568.0), _step(16, 1400.0)]
CLIMBING = [_step(1, 94.0), _step(4, 326.0)]


def _result(gpu: str, model: str, steps, partial: bool = False) -> BenchResult:
    return BenchResult(
        run_id=f"vllm-{gpu}-{len(steps)}-{partial}",
        target=Target(kind="vllm", model=model, precision="bfloat16", tp_size=1),
        hardware=Hardware(gpu_name=gpu, gpu_count=1),
        pricing=Pricing(hourly_rate_usd=0.35),
        timings=Timings(), workload=Workload(), run_index=1,
        partial=partial, steps=steps,
    )


@pytest.mark.parametrize("gpu_name,expected", [
    ("NVIDIA GeForce RTX 5090", "RTX_5090"),
    ("NVIDIA A100-SXM4-80GB", "A100_SXM4"),
    ("NVIDIA L40S", "L40S"),
    ("NVIDIA H100 80GB HBM3", "H100_SXM"),
])
def test_nvidia_smi_names_map_to_tiers(gpu_name, expected):
    """The result records what nvidia-smi reports, which is not the string Vast
    was asked for."""
    assert tier_for(gpu_name) == expected


def test_a_pcie_h100_is_not_counted_as_an_sxm_one():
    """Different interconnect, different throughput. Guessing here would put a
    PCIe card's numbers under an SXM heading."""
    assert tier_for("NVIDIA H100 PCIe") is None


def test_an_unknown_card_is_reported_not_guessed():
    assert tier_for("NVIDIA GeForce RTX 4090") is None


def test_only_saturated_complete_runs_count_toward_coverage():
    results = [
        _result("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", SATURATED),
        _result("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", CLIMBING),
        _result("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", SATURATED, partial=True),
    ]
    assert coverage(results) == {("RTX_5090", "anchor"): 1}


def test_missing_asks_only_for_the_runs_still_needed():
    results = [_result("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", SATURATED)]
    gaps = missing(results, runs_per_config=3)
    fives = [r for r in gaps if r.tier_key == "RTX_5090"]
    assert len(fives) == 2, "one sample banked, two still owed"
    assert all(r.model_key == "anchor" for r in fives)


def test_a_fully_covered_matrix_asks_for_nothing():
    results = [
        _result(gpu, model, SATURATED)
        for gpu, model in [
            ("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B"),
            ("NVIDIA A100-SXM4-80GB", "Qwen/Qwen3-8B"),
            ("NVIDIA A100-SXM4-80GB", "Qwen/Qwen3.8-27B"),
            ("NVIDIA L40S", "Qwen/Qwen3-8B"),
            ("NVIDIA L40S", "Qwen/Qwen3.8-27B"),
            ("NVIDIA H100 80GB HBM3", "Qwen/Qwen3-8B"),
            ("NVIDIA H100 80GB HBM3", "Qwen/Qwen3.8-27B"),
        ]
        for _ in range(3)
    ]
    assert missing(results, runs_per_config=3) == []


def test_a_40gb_a100_is_not_filed_under_the_80gb_tier():
    """Half the VRAM is a different machine for this purpose: it changes what
    fits and what the KV cache can hold. The first live A100 run rented one."""
    assert tier_for("NVIDIA A100-SXM4-40GB") is None
    assert tier_for("NVIDIA A100-SXM4-80GB") == "A100_SXM4"


def test_a_resumed_run_is_numbered_after_the_ones_already_banked():
    """missing() reused the matrix's own indices, so with one RTX 5090 anchor
    result already banked it planned another 'run 1'. Two results for the same
    cell would then both claim to be the first, and the record would no longer
    say which three runs the median came from."""
    from launch.coverage import missing

    banked = [_result("NVIDIA GeForce RTX 5090", "Qwen/Qwen3-8B", SATURATED)]
    owed = missing(banked)
    cell = [r for r in owed if r.tier_key == "RTX_5090" and r.model_key == "anchor"]
    assert [r.run_index for r in cell] == [2, 3]


def test_an_untouched_cell_still_starts_at_one():
    from launch.coverage import missing

    owed = missing([])
    cell = [r for r in owed if r.tier_key == "RTX_5090" and r.model_key == "anchor"]
    assert [r.run_index for r in cell] == [1, 2, 3]
