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

# Generous relative to the measured 25-minute run, because the timer is a
# backstop against a hung instance, not a schedule.
RUN_TIMEOUT_MINUTES = 75
POLL_SECONDS = 60


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


def run_one(run: Run, gppb_ref: str, ttl_minutes: int = 60) -> str:
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
    )
    created = launch_instance(offer, env, onstart_script(), disk_gb=120)
    instance_id = created.get("new_contract")
    print(f"    instance {instance_id} @ ${offer.hourly_usd:.4f}/hr")

    deadline = time.time() + RUN_TIMEOUT_MINUTES * 60
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        if not any(i.get("id") == instance_id for i in _instances()):
            return "completed"
    # Past the timeout the instance is hung; the in-container TTL should have
    # fired, so kill it here rather than trusting it twice.
    subprocess.run(
        [vastai_bin(), "destroy", "instance", str(instance_id), "-y"],
        capture_output=True, check=False,
    )
    return "timeout"


def run_matrix(
    gppb_ref: str = "main",
    limit: int | None = None,
    dry_run: bool = True,
    reserve_usd: float = 1.00,
) -> list[Run]:
    """Work the matrix until it is done or the budget says stop."""
    planned = build_matrix()
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
    parser.add_argument("--go", action="store_true", help="actually spend money")
    args = parser.parse_args()

    result = run_matrix(gppb_ref=args.ref, limit=args.limit, dry_run=not args.go)
    if not args.go:
        total = sum(
            estimate_run_usd(MODELS[r.model_key], TIERS[r.tier_key]) for r in result
        )
        print(f"planned {len(result)} runs, ~${total:.2f} — pass --go to run")
