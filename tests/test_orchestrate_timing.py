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
