"""Drive the serving sweep: rent, run, collect, repeat, and stop before the
credit does.

Runs are sequential on purpose. Parallel rentals would finish sooner but
multiply the number of instances that can be stranded if this process dies,
and a stranded GPU is the failure mode this whole project guards against."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from launch.matrix import (
    MODELS, TIERS, BudgetExhausted, BudgetGate, Run, build_matrix,
    estimate_run_usd,
)
from launch.reap import reap
from launch.vast import (
    build_env, launch_instance, onstart_script, search_offers, select_offer,
    vastai_bin,
)

SINK_URL = os.environ.get("SINK_URL", "https://gppb-sink.sfraser.workers.dev")
TOKEN_FILE = Path.home() / ".gppb-sink-token"

# Hosts slower than this spend more on pulling the image than they save on
# hourly rate — measured, not guessed: a 90Mbps host burned 23 minutes.
MIN_INET_DOWN_MBPS = 600.0
MIN_RELIABILITY = 0.97

# Three stop-clocks, and the ordering between them is the whole point.
#
# The container's timers start when onstart runs, which is *after* Vast has
# pulled a ~15GB image; the orchestrator's used to start at launch. The first
# live sweep proved what that costs: a slow pull pushed the container TTL past
# the orchestrator's deadline, so a run that was still working was killed from
# outside and recorded as a timeout with nothing uploaded.
#
# The fix is to measure the run from "running", so both clocks cover the same
# interval, and budget the pull separately.
#
#   CONTAINER_TTL  <  RUN_TIMEOUT  <  CONTAINER_BACKSTOP
#
# The instance stops itself (TTL). The orchestrator is the backstop for an
# instance that cannot. The container backstop is the backstop for a dead
# orchestrator, so it must outlive it rather than pre-empt it.
CONTAINER_TTL_MINUTES = 60
RUN_TIMEOUT_MINUTES = 75
CONTAINER_BACKSTOP_MINUTES = 90

# Pull plus scheduling, billed but not part of the run.
#
# A host past the 600Mbps floor should pull the ~15GB image in about four
# minutes; run 2's host was serving in three. Two of the first four hosts
# instead sat in "loading" indefinitely and uploaded nothing, so patience here
# is billed for nothing. 25 minutes is roughly six times the expected pull and
# still well short of the 40 that cost a wasted A100 slot.
PULL_TIMEOUT_MINUTES = 25

POLL_SECONDS = 60


def await_running(
    instance_id: int,
    instances: Callable[[], list[dict]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout_minutes: float = PULL_TIMEOUT_MINUTES,
    poll_seconds: float = POLL_SECONDS,
) -> bool:
    """Block until the instance is actually running. False if it never is.

    Returning False rather than raising keeps the decision with the caller,
    which is the only place that knows to destroy the instance first."""
    poll = instances or _instances
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        for item in poll():
            if item.get("id") == instance_id:
                if item.get("actual_status") == "running":
                    return True
                break
        sleep(poll_seconds)
    return False


def current_credit() -> float:
    raw = subprocess.run(
        [vastai_bin(), "show", "user", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    payload = json.loads(raw)
    user = payload[0] if isinstance(payload, list) else payload
    return float(user["credit"])


def _instances() -> list[dict]:
    raw = subprocess.run(
        [vastai_bin(), "show", "instances", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(raw)


def run_one(run: Run, gppb_ref: str, ttl_minutes: int = CONTAINER_TTL_MINUTES) -> str:
    """Rent one instance, wait for it to finish, and report how it ended.

    The instance destroys itself when the sweep completes, so its disappearance
    is the completion signal."""
    model = MODELS[run.model_key]
    tier = TIERS[run.tier_key]

    offer = select_offer(
        search_offers(tier.key, 1), tier.key, 1,
        max_hourly=tier.typical_hourly_usd * 2.0,
        min_inet_down_mbps=MIN_INET_DOWN_MBPS,
        min_reliability=MIN_RELIABILITY,
    )
    env = build_env(
        model=model.hf_id, precision=model.precision, tp_size=1,
        run_index=run.run_index, hourly_usd=round(offer.hourly_usd, 4),
        ttl_minutes=ttl_minutes, sink_url=SINK_URL, gppb_ref=gppb_ref,
        sink_token=TOKEN_FILE.read_text().strip(),
        backstop_minutes=CONTAINER_BACKSTOP_MINUTES,
    )
    created = launch_instance(offer, env, onstart_script(), disk_gb=120)
    instance_id = created.get("new_contract")
    print(f"    instance {instance_id} @ ${offer.hourly_usd:.4f}/hr")

    def destroy() -> None:
        subprocess.run(
            [vastai_bin(), "destroy", "instance", str(instance_id), "-y"],
            capture_output=True, check=False,
        )

    # The pull is billed but is not the run. Time it separately so a slow host
    # is reported as a slow host, not as a benchmark that hung.
    pull_started = time.time()
    if not await_running(instance_id):
        destroy()
        return "never-started"
    print(f"    running after {(time.time() - pull_started) / 60:.1f} min pull")

    deadline = time.time() + RUN_TIMEOUT_MINUTES * 60
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        if not any(i.get("id") == instance_id for i in _instances()):
            return "completed"
    # Past the timeout the instance is hung and its own TTL has already failed
    # to stop it, so kill it here rather than trusting that timer twice.
    destroy()
    return "timeout"


def run_matrix(
    gppb_ref: str = "main",
    limit: int | None = None,
    dry_run: bool = True,
    reserve_usd: float = 1.00,
    skip: int = 0,
) -> list[Run]:
    """Work the matrix until it is done or the budget says stop.

    `skip` drops the first N planned runs. A multi-hour paid sweep will be
    interrupted — this one was — and resuming from the start would buy the
    same results twice. The matrix is deterministic and ordered, so an offset
    is enough to resume from."""
    planned = build_matrix()[skip:]
    if limit is not None:
        planned = planned[:limit]
    if dry_run:
        return planned

    stranded = reap()
    if stranded:
        print(f"reaped before starting: {stranded}")

    gate = BudgetGate(current_credit(), reserve_usd=reserve_usd)
    print(f"credit ${gate.remaining:.2f}, reserve ${reserve_usd:.2f}")

    done: list[Run] = []
    for position, run in enumerate(planned, start=1):
        model, tier = MODELS[run.model_key], TIERS[run.tier_key]
        estimate = estimate_run_usd(model, tier)
        try:
            gate.check(estimate)
        except BudgetExhausted as exc:
            print(f"stopping after {len(done)} runs: {exc}")
            break
        print(f"[{position}/{len(planned)}] {tier.key} {model.hf_id} "
              f"run {run.run_index} (~${estimate:.2f})")
        outcome = run_one(run, gppb_ref)
        print(f"    {outcome}")
        done.append(run)

    leftover = reap()
    if leftover:
        print(f"reaped after finishing: {leftover}")
    return done


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="main", help="harness revision to run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0,
                        help="drop the first N planned runs (resume a sweep)")
    parser.add_argument("--go", action="store_true", help="actually spend money")
    args = parser.parse_args()

    result = run_matrix(gppb_ref=args.ref, limit=args.limit,
                        dry_run=not args.go, skip=args.skip)
    if not args.go:
        total = sum(
            estimate_run_usd(MODELS[r.model_key], TIERS[r.tier_key]) for r in result
        )
        print(f"planned {len(result)} runs, ~${total:.2f} — pass --go to run")
