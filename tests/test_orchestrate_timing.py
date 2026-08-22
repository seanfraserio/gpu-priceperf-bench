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


def test_timers_scale_with_the_expected_run_length():
    """The headline model is a 55.6GB download against the anchor's 17GB, on
    top of a slower fp8 load and a hybrid KV cache. A fixed 60-minute TTL
    would kill it mid-sweep, and since sync discards partial results the run
    would cost money and yield nothing."""
    short = orch.timers_for(25.0)
    long = orch.timers_for(40.0)
    assert long.ttl > short.ttl
    assert long.run_timeout > short.run_timeout


def test_the_ordering_invariant_holds_at_every_run_length():
    """Whatever the scaling produces, the instance must still stop itself
    before the orchestrator gives up, and the container backstop must outlive
    the orchestrator."""
    for minutes in (10.0, 25.0, 40.0, 90.0):
        t = orch.timers_for(minutes)
        assert t.ttl < t.run_timeout < t.backstop


def test_the_anchor_keeps_the_timings_that_are_known_to_work():
    """Run 2 completed a full nine-level sweep inside these."""
    t = orch.timers_for(25.0)
    assert (t.ttl, t.run_timeout, t.backstop) == (60, 75, 90)


def test_preflight_reports_a_tier_that_can_rent_nothing():
    """Raising a floor is how a tier silently stops being rentable — declaring
    the L40S at 48GB when hosts report 45 would have failed all six of its
    runs one at a time, mid-sweep. Better to know before spending."""
    from launch.vast import Offer

    def offers_for(gpu_name, num_gpus):
        if gpu_name == "L40S":
            return [Offer(id=1, gpu_name="L40S", num_gpus=1, hourly_usd=0.80,
                          inet_down_mbps=200.0, reliability=0.99, vram_gb=45.0)]
        return [Offer(id=2, gpu_name=gpu_name.replace("_", " "), num_gpus=1,
                      hourly_usd=0.30, inet_down_mbps=2000.0, reliability=0.99,
                      vram_gb=999.0)]

    unsatisfiable = orch.preflight(search=offers_for)
    assert "L40S" in unsatisfiable
    assert len(unsatisfiable) == 1


def test_preflight_is_quiet_when_every_tier_can_rent():
    from launch.vast import Offer

    def offers_for(gpu_name, num_gpus):
        return [Offer(id=2, gpu_name=gpu_name.replace("_", " "), num_gpus=1,
                      hourly_usd=0.30, inet_down_mbps=2000.0, reliability=0.99,
                      vram_gb=999.0)]

    assert orch.preflight(search=offers_for) == []
