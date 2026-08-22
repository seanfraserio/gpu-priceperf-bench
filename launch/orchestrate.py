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
from typing import Callable, NamedTuple

import httpx

from launch.matrix import (
    BASE_RUN_MINUTES, MODELS, TIERS, BudgetExhausted, BudgetGate, Run,
    build_matrix, estimate_run_usd,
)
from launch.coverage import missing
from launch.reap import reap
from launch.blocklist import Blocklist
from launch.sync import FAILURE_PREFIX
from launch.vast import (
    build_env, launch_instance, onstart_script, search_offers, select_offer,
    vastai_bin,
)

SINK_URL = os.environ.get("SINK_URL", "https://gppb-sink.sfraser.workers.dev")
TOKEN_FILE = Path.home() / ".gppb-sink-token"

# Hosts slower than this spend more on pulling the image than they save on
# hourly rate — measured, not guessed: a 90Mbps host burned 23 minutes.
#
# Raised from 600 after three of the first seven hosts sat in "loading" until
# the pull budget ran out. Every one of them cleared the old floor, so the
# advertised figure is a weak predictor of how fast a 15GB image actually
# arrives; the hosts just above the floor are the ones that stall. A stall
# costs a full pull budget at the run's hourly rate, while the next host up
# the ladder costs about 8% more per hour — the trade is not close.
# Lowered from 900 once the L40S pool shrank to three machines, exactly one of
# which is otherwise eligible and advertises 853. The stalls that motivated the
# floor were at 600-700; 900 was a round number carrying margin above the
# observed cliff, not a measured threshold. Bandwidth only affects the pull,
# which is timed separately and excluded from the throughput knee, so this
# cannot move a price/perf number — and the pull timeout still bounds the loss.
MIN_INET_DOWN_MBPS = 850.0
MIN_RELIABILITY = 0.97

# The image is a cu128 build, so a driver that tops out below 12.8 cannot run
# it: one A100 host advertises 12.2, and a run there pays for the pull and the
# weights before dying with CUDA error 803.
#
# Deliberately not set higher. 803 is also what a host with a broken driver
# install reports, and one observed failure is not evidence of a version
# threshold — a floor of 13.0 would have emptied the L40S pool entirely on the
# strength of a guess. Broken hosts are the blocklist's job.
MIN_CUDA = 12.8

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

# Those three are the anchor's timings, and run 2 completed a full nine-level
# sweep inside them. They do not generalise: the headline model is a 55.6GB
# download against the anchor's 17GB, on top of a slower fp8 load and a hybrid
# KV cache. A fixed TTL would kill it mid-sweep, and because sync discards
# partial results that run would cost money and yield nothing. So the timers
# derive from how long the run is expected to take, keeping the ordering.
TTL_SAFETY_FACTOR = CONTAINER_TTL_MINUTES / BASE_RUN_MINUTES


class Timers(NamedTuple):
    ttl: int
    run_timeout: int
    backstop: int


def timers_for(expected_minutes: float) -> Timers:
    """Stop-clocks sized to the run, preserving TTL < timeout < backstop."""
    ttl = max(CONTAINER_TTL_MINUTES, round(expected_minutes * TTL_SAFETY_FACTOR))
    return Timers(ttl=ttl, run_timeout=round(ttl * 1.25), backstop=round(ttl * 1.5))

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


def preflight(search: Callable[[str, int], list] | None = None) -> list[str]:
    """Tiers that cannot currently rent anything under the configured floors.

    Raising a floor is how a tier silently stops being rentable: declaring the
    L40S at 48GB when every host reports 45 would have failed all six of its
    runs one at a time, mid-sweep, each after a full pull budget. Cheap to
    check up front, and it costs nothing to be wrong about."""
    lookup = search or search_offers
    blocked: list[str] = []
    for key, tier in TIERS.items():
        try:
            select_offer(
                lookup(key, 1), key, 1,
                max_hourly=tier.typical_hourly_usd * 2.0,
                min_inet_down_mbps=MIN_INET_DOWN_MBPS,
                min_reliability=MIN_RELIABILITY,
                min_vram_gb=tier.vram_gb * 0.95,
                min_cuda=MIN_CUDA,
                blocked=BLOCKLIST.machines(),
            )
        except LookupError:
            blocked.append(key)
    return blocked


# Every run in the matrix executes the same harness on the same kind of host.
# A third consecutive failure is a bug, not bad luck, and the remaining budget
# would only buy the same failure eighteen more times.
MAX_CONSECUTIVE_FAILURES = 3

# Remembered between sweeps: a sweep is restarted often, and a blocklist that
# only lives in memory forgets the bad host on every restart.
BLOCKLIST = Blocklist()


def _sink_keys() -> set[str]:
    from launch.sync import list_keys

    with httpx.Client() as client:
        return set(list_keys(client, SINK_URL, TOKEN_FILE.read_text().strip()))


def _is_complete(key: str) -> bool:
    """Whether a sink object is a finished sweep rather than a level-by-level
    snapshot. Every level uploads, so the first object is not the answer."""
    with httpx.Client() as client:
        response = client.get(
            f"{SINK_URL}/{key}",
            headers={"Authorization": f"Bearer {TOKEN_FILE.read_text().strip()}"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("partial") is False


def run_one(
    run: Run,
    gppb_ref: str,
    ttl_minutes: int | None = None,
    sink_keys: Callable[[], set[str]] | None = None,
    is_complete: Callable[[str], bool] | None = None,
) -> str:
    """Rent one instance, wait for it to finish, and report how it ended.

    The instance destroys itself either way, so its disappearance says nothing
    about whether the run worked. What the run published to the sink does."""
    model = MODELS[run.model_key]
    tier = TIERS[run.tier_key]
    # Sized to this model, not to the anchor: the 27B downloads 55.6GB before
    # it serves a token.
    timers = timers_for(BASE_RUN_MINUTES * model.runtime_multiplier)
    if ttl_minutes is not None:
        timers = timers_for(ttl_minutes / TTL_SAFETY_FACTOR)

    offer = select_offer(
        search_offers(tier.key, 1), tier.key, 1,
        max_hourly=tier.typical_hourly_usd * 2.0,
        min_inet_down_mbps=MIN_INET_DOWN_MBPS,
        min_reliability=MIN_RELIABILITY,
        # The tier's declared VRAM is what feasible() reasons about, so the
        # rented card has to actually have it. A little tolerance because hosts
        # report 79-80GB for the same 80GB part.
        min_vram_gb=tier.vram_gb * 0.95,
        blocked=BLOCKLIST.machines(),
        min_cuda=MIN_CUDA,
    )
    env = build_env(
        model=model.hf_id, precision=model.precision, tp_size=1,
        run_index=run.run_index, hourly_usd=round(offer.hourly_usd, 4),
        ttl_minutes=timers.ttl, sink_url=SINK_URL, gppb_ref=gppb_ref,
        sink_token=TOKEN_FILE.read_text().strip(),
        backstop_minutes=timers.backstop,
    )
    keys = sink_keys or _sink_keys
    try:
        before = keys()
    except Exception:
        # The operator's own network is not evidence about the run.
        before = None
    created = launch_instance(offer, env, onstart_script(), disk_gb=120)
    instance_id = created.get("new_contract")
    print(f"    instance {instance_id} @ ${offer.hourly_usd:.4f}/hr "
          f"({offer.vram_gb:.0f}GB, ttl {timers.ttl}m)")

    def destroy() -> None:
        subprocess.run(
            [vastai_bin(), "destroy", "instance", str(instance_id), "-y"],
            capture_output=True, check=False,
        )

    # The pull is billed but is not the run. Time it separately so a slow host
    # is reported as a slow host, not as a benchmark that hung.
    pull_started = time.time()
    def verdict(outcome: str) -> str:
        """Record the host when the run bought nothing, so the next one goes
        somewhere else."""
        if outcome != "completed" and offer.machine_id is not None:
            BLOCKLIST.record(offer.machine_id, outcome)
        return outcome

    if not await_running(instance_id):
        destroy()
        return verdict("never-started")
    print(f"    running after {(time.time() - pull_started) / 60:.1f} min pull")

    # The container is supposed to destroy itself when the sweep ends. One
    # L40S did not, and a finished result sat in the sink for fifty minutes
    # while the meter ran. The result is the completion signal; the instance
    # disappearing is only a fallback.
    complete = is_complete or _is_complete
    deadline = time.time() + timers.run_timeout * 60
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        published = _published(before, keys, complete)
        if published:
            destroy()
            return verdict(published)
        if not any(i.get("id") == instance_id for i in _instances()):
            return verdict(_outcome(before, keys, complete))
    # Past the timeout the instance is hung and its own TTL has already failed
    # to stop it, so kill it here rather than trusting that timer twice.
    destroy()
    return verdict("timeout")


def _published(
    before: set[str] | None,
    keys: Callable[[], set[str]],
    complete: Callable[[str], bool],
) -> str | None:
    """The run's verdict if it has already published one, else None."""
    if before is None:
        return None
    try:
        new = keys() - before
    except Exception:
        return None
    if any(key.startswith(FAILURE_PREFIX) for key in new):
        return "failed"
    for key in new:
        try:
            if complete(key):
                return "completed"
        except Exception:
            continue
    return None


def _outcome(
    before: set[str] | None,
    keys: Callable[[], set[str]],
    complete: Callable[[str], bool],
) -> str:
    """What the instance left behind, once it is gone.

    A run that died at level 256 leaves eight levels in the sink. That is not
    a success — and calling it one would clear the failure streak that stops a
    systemic failure from spending the rest of the budget."""
    if before is None:
        return "completed"
    published = _published(before, keys, complete)
    if published:
        return published
    try:
        if keys() - before:
            return "no-result"
    except Exception:
        return "completed"
    return "no-result"


def run_matrix(
    gppb_ref: str = "main",
    limit: int | None = None,
    dry_run: bool = True,
    reserve_usd: float = 1.00,
    skip: int = 0,
    resume: bool = False,
    results_dir: Path | None = None,
) -> list[Run]:
    """Work the matrix until it is done or the budget says stop.

    `skip` drops the first N planned runs. A multi-hour paid sweep will be
    interrupted — this one was — and resuming from the start would buy the
    same results twice. The matrix is deterministic and ordered, so an offset
    is enough to resume from."""
    if resume:
        # After two interruptions and two dead hosts, "where was I" is a
        # question about which results exist, not about an offset into a plan
        # whose runs may each have failed.
        from report.generate import load_results

        planned = missing(load_results(results_dir or Path("results")))
    else:
        planned = build_matrix()[skip:]
    if limit is not None:
        planned = planned[:limit]
    if dry_run:
        return planned

    stranded = reap()
    if stranded:
        print(f"reaped before starting: {stranded}")

    blocked = preflight()
    if blocked:
        # Reported, not fatal: the other tiers are still worth collecting, and
        # a tier can become rentable again while the sweep is running.
        print(f"WARNING: no qualifying offers right now for {', '.join(blocked)}")

    gate = BudgetGate(current_credit(), reserve_usd=reserve_usd)
    print(f"credit ${gate.remaining:.2f}, reserve ${reserve_usd:.2f}")

    done: list[Run] = []
    streak = 0
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
        try:
            outcome = run_one(run, gppb_ref)
        except LookupError as exc:
            # The floors and the blocklist can between them empty a tier's
            # pool. Skip that run; the other tiers are still worth collecting.
            print(f"    skipped: {exc}")
            continue
        print(f"    {outcome}")
        done.append(run)
        streak = 0 if outcome == "completed" else streak + 1
        if streak >= MAX_CONSECUTIVE_FAILURES:
            print(f"stopping: {streak} runs in a row published nothing — "
                  f"read them with `python -m launch.sync --failures`")
            break

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
    parser.add_argument("--resume", action="store_true",
                        help="run only what results/ is still missing")
    parser.add_argument("--go", action="store_true", help="actually spend money")
    args = parser.parse_args()

    result = run_matrix(gppb_ref=args.ref, limit=args.limit,
                        dry_run=not args.go, skip=args.skip,
                        resume=args.resume)
    if not args.go:
        total = sum(
            estimate_run_usd(MODELS[r.model_key], TIERS[r.tier_key]) for r in result
        )
        print(f"planned {len(result)} runs, ~${total:.2f} — pass --go to run")
