"""The two stop-clocks must measure the same interval.

Run 1 of the first live sweep was force-destroyed at 75 minutes having
uploaded nothing, and the post-mortem was not a broken kill switch: the
container's TTL starts when onstart runs, which is *after* the 15GB image
pull, while the orchestrator's deadline started at launch. A slow pull pushed
the TTL past the orchestrator's deadline, so a run that was still working got
killed and recorded as a timeout. These tests pin the ordering that makes the
two clocks comparable."""
from __future__ import annotations

import launch.orchestrate as orch


def test_container_ttl_fires_before_the_orchestrator_gives_up():
    """The instance stops itself; the orchestrator is only the backstop.

    If this inverts, every slow run is killed from outside and reported as a
    timeout even though it was making progress."""
    assert orch.CONTAINER_TTL_MINUTES < orch.RUN_TIMEOUT_MINUTES


def test_container_backstop_is_the_last_resort():
    """The in-container backstop exists for a laptop that dies mid-sweep, so it
    must outlive the orchestrator rather than pre-empt it."""
    assert orch.RUN_TIMEOUT_MINUTES < orch.CONTAINER_BACKSTOP_MINUTES


def test_pull_time_is_excluded_from_the_run_deadline():
    """The run deadline is measured from 'running', not from launch, so the
    image pull is budgeted separately and never eats the run's own clock."""
    calls: list[str] = []

    def fake_instances():
        calls.append("poll")
        # Loading for two polls, then running.
        status = "loading" if len(calls) < 3 else "running"
        return [{"id": 7, "actual_status": status}]

    elapsed = orch.await_running(
        7, instances=fake_instances, sleep=lambda _s: None,
        timeout_minutes=orch.PULL_TIMEOUT_MINUTES, poll_seconds=0,
    )
    assert elapsed is True
    assert len(calls) == 3


def test_an_instance_that_never_boots_is_reported_not_waited_on_forever():
    """A host that never finishes pulling bills the whole time. Give up and let
    the caller destroy it rather than blocking the sweep."""
    assert orch.await_running(
        7, instances=lambda: [{"id": 7, "actual_status": "loading"}],
        sleep=lambda _s: None, timeout_minutes=0, poll_seconds=0,
    ) is False


def test_backstop_reaches_the_container():
    """A backstop the orchestrator sets but never sends is not a backstop."""
    from launch.vast import build_env

    env = build_env(
        model="m", precision="fp8", tp_size=1, run_index=1, hourly_usd=1.0,
        ttl_minutes=orch.CONTAINER_TTL_MINUTES, sink_url=None,
        backstop_minutes=orch.CONTAINER_BACKSTOP_MINUTES,
    )
    assert env["TTL_MINUTES"] == str(orch.CONTAINER_TTL_MINUTES)
    assert env["TTL_BACKSTOP_MINUTES"] == str(orch.CONTAINER_BACKSTOP_MINUTES)


def test_skip_resumes_a_sweep_without_repaying_for_finished_runs():
    """A multi-hour paid sweep will be interrupted — this one already was. If
    resuming means starting from run 1, the interruption costs the whole bill
    again, so the plan has to be resumable from a known offset."""
    full = orch.build_matrix()
    resumed = orch.run_matrix(skip=2, dry_run=True)
    assert resumed == full[2:]


def test_skip_and_limit_compose():
    """--limit counts the runs actually attempted, not the ones skipped."""
    assert len(orch.run_matrix(skip=2, limit=3, dry_run=True)) == 3


def test_pull_budget_is_far_shorter_than_the_run_itself():
    """A stalled pull uploads nothing and bills the whole time. Two of the
    first four hosts stalled, so the pull budget has to be impatient relative
    to the run it precedes."""
    assert orch.PULL_TIMEOUT_MINUTES < orch.CONTAINER_TTL_MINUTES


def test_resume_runs_what_coverage_says_is_owed(tmp_path, monkeypatch):
    """After two interruptions and two bad hosts, "where was I" is a question
    about results, not about an offset into the plan."""
    from gppb.models import (
        BenchResult, Hardware, Pricing, Stats, StepResult, Target, Timings, Workload,
    )

    def _stats(v):
        return Stats(p50=v, p90=v, min=v, max=v)

    def _step(c, tps):
        return StepResult(
            concurrency=c, requests_completed=c * 4, requests_failed=0,
            wall_seconds=10.0, output_tokens_total=int(tps * 10),
            output_tokens_per_sec=tps, ttft_ms=_stats(1.0), tpot_ms=_stats(1.0),
        )

    banked = BenchResult(
        run_id="vllm-5090-1",
        target=Target(kind="vllm", model="Qwen/Qwen3-8B", precision="bfloat16", tp_size=1),
        hardware=Hardware(gpu_name="NVIDIA GeForce RTX 5090", gpu_count=1),
        pricing=Pricing(hourly_rate_usd=0.34),
        timings=Timings(), workload=Workload(), run_index=1, partial=False,
        steps=[_step(1, 94.0), _step(8, 1568.0), _step(16, 1400.0)],
    )
    (tmp_path / "banked.json").write_text(banked.model_dump_json())

    owed = orch.run_matrix(dry_run=True, resume=True, results_dir=tmp_path)
    fives = [r for r in owed if r.tier_key == "RTX_5090"]
    assert len(fives) == 2, "one 5090 sample banked, two still owed"
    assert len(owed) == len(orch.build_matrix()) - 1
