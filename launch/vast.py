"""Rent a GPU with a hard price ceiling and a mandatory TTL."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Stock upstream image — there is no custom build. The vLLM version pin is this
# tag; the harness itself is cloned at boot by onstart.sh.
IMAGE = "vllm/vllm-openai:v0.27.1"

# nccl-tests has no official binary distribution, so its runner uses the stock
# CUDA devel image and compiles at boot. Still no image to build or host.
NCCL_IMAGE = "nvidia/cuda:12.6.0-devel-ubuntu22.04"

_ROOT = Path(__file__).resolve().parent.parent
ONSTART_PATH = _ROOT / "runner-vllm" / "onstart.sh"
NCCL_ONSTART_PATH = _ROOT / "runner-nccl" / "onstart.sh"


def onstart_script() -> str:
    """The script Vast runs inside the stock image, passed verbatim."""
    return ONSTART_PATH.read_text()


def nccl_onstart_script() -> str:
    """Same contract for the multi-GPU interconnect run."""
    return NCCL_ONSTART_PATH.read_text()


@dataclass
class Offer:
    id: int
    gpu_name: str
    num_gpus: int
    hourly_usd: float


def select_offer(
    offers: list[Offer], gpu_name: str, num_gpus: int, max_hourly: float
) -> Offer:
    """Cheapest offer matching the request, or abort. Never silently upgrade."""
    matches = [
        o for o in offers
        if o.gpu_name == gpu_name and o.num_gpus == num_gpus and o.hourly_usd <= max_hourly
    ]
    if not matches:
        raise LookupError(
            f"no {num_gpus}x {gpu_name} at or under ${max_hourly}/hr — not renting"
        )
    return min(matches, key=lambda o: o.hourly_usd)


def build_env(
    model: str,
    precision: str,
    tp_size: int,
    run_index: int,
    hourly_usd: float,
    ttl_minutes: int,
    sink_url: str | None,
    gppb_ref: str = "main",
    sink_token: str | None = None,
) -> dict[str, str]:
    env = {
        "MODEL": model,
        "PRECISION": precision,
        "TP_SIZE": str(tp_size),
        "RUN_INDEX": str(run_index),
        "HOURLY_RATE_USD": str(hourly_usd),
        "TTL_MINUTES": str(ttl_minutes),
        "SWEEP": "1,2,4,8,16,32,64,128,256",
        "MAX_MODEL_LEN": "32768",
        # Pinned revision of the harness the instance clones at boot.
        "GPPB_REF": gppb_ref,
    }
    if sink_url:
        # The token is useless without the URL and the URL is useless without
        # the token — an instance that cannot upload burns money for nothing.
        env["SINK_URL"] = sink_url
        if sink_token:
            env["SINK_TOKEN"] = sink_token
    return env


def search_offers(gpu_name: str, num_gpus: int) -> list[Offer]:
    raw = subprocess.run(
        ["vastai", "search", "offers",
         f"gpu_name={gpu_name} num_gpus={num_gpus}", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [
        Offer(
            id=item["id"],
            gpu_name=item["gpu_name"],
            num_gpus=item["num_gpus"],
            hourly_usd=float(item["dph_total"]),
        )
        for item in json.loads(raw)
    ]
