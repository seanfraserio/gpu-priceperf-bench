"""Kill anything past its TTL. Run before and after every session."""
from __future__ import annotations

import json
import subprocess

DEFAULT_TTL_MINUTES = 60


def stale_instances(instances: list[dict], now_epoch: float) -> list[int]:
    stale = []
    for instance in instances:
        ttl = instance.get("ttl_minutes") or DEFAULT_TTL_MINUTES
        age_minutes = (now_epoch - instance["start_date"]) / 60
        if age_minutes > ttl:
            stale.append(instance["id"])
    return stale


def reap() -> list[int]:
    import time
    raw = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    victims = stale_instances(json.loads(raw), time.time())
    for instance_id in victims:
        subprocess.run(["vastai", "destroy", "instance", str(instance_id)], check=False)
        print(f"destroyed stale instance {instance_id}")
    return victims


if __name__ == "__main__":
    print(f"reaped: {reap()}")
