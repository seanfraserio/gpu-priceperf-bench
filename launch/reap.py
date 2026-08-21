"""Kill anything past its TTL. Run before and after every session."""
from __future__ import annotations

import json
import subprocess

from launch.vast import vastai_bin

DEFAULT_TTL_MINUTES = 60


def stale_instances(instances: list[dict], now_epoch: float) -> list[int]:
    stale = []
    for instance in instances:
        ttl = instance.get("ttl_minutes") or DEFAULT_TTL_MINUTES
        age_minutes = (now_epoch - instance["start_date"]) / 60
        if age_minutes > ttl:
            stale.append(instance["id"])
    return stale


def destroy_instance(instance_id: int) -> bool:
    """Destroy one instance. Returns whether it actually died.

    -y is mandatory: without it vastai prompts, aborts in a non-interactive
    context, and the caller happily reports a destruction that never happened
    while the GPU keeps billing."""
    done = subprocess.run(
        [vastai_bin(), "destroy", "instance", str(instance_id), "-y"],
        capture_output=True, text=True, check=False,
    )
    return done.returncode == 0


def reap() -> list[int]:
    import time
    raw = subprocess.run(
        [vastai_bin(), "show", "instances", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    destroyed: list[int] = []
    survivors: list[int] = []
    for instance_id in stale_instances(json.loads(raw), time.time()):
        if destroy_instance(instance_id):
            destroyed.append(instance_id)
            print(f"destroyed stale instance {instance_id}")
        else:
            survivors.append(instance_id)
            print(f"FAILED to destroy {instance_id} — still billing, destroy it by hand")
    if survivors:
        raise RuntimeError(f"instances still running: {survivors}")
    return destroyed


if __name__ == "__main__":
    print(f"reaped: {reap()}")
