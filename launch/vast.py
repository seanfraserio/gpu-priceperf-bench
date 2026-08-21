"""Rent a GPU with a hard price ceiling and a mandatory TTL."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
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


def vastai_bin() -> str:
    """Absolute path to the vastai CLI.

    It is usually installed into the same venv as this code, whose bin
    directory is not on PATH unless the venv is activated. The reaper depends
    on this and runs while a GPU is billing, so it resolves the interpreter's
    own directory first and says what to install rather than raising a bare
    FileNotFoundError."""
    beside_python = Path(sys.executable).parent / "vastai"
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which("vastai")
    if found:
        return found
    raise RuntimeError(
        "vastai CLI not found — pip install vastai, then `vastai set api-key <key>`"
    )


def onstart_script() -> str:
    """The script Vast runs inside the stock image, passed verbatim."""
    return ONSTART_PATH.read_text()


def nccl_onstart_script() -> str:
    """Same contract for the multi-GPU interconnect run."""
    return NCCL_ONSTART_PATH.read_text()


# Values that must never reach argv, which is world-readable through ps. They
# travel inside the onstart script body instead.
SECRET_ENV_KEYS = frozenset({"SINK_TOKEN"})


def _normalise_gpu_name(name: str) -> str:
    """Vast queries use underscores ("RTX_5090"), responses use spaces
    ("RTX 5090"). Compare on one form so searching and selecting agree."""
    return name.replace("_", " ").strip().casefold()


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
    wanted = _normalise_gpu_name(gpu_name)
    matches = [
        o for o in offers
        if _normalise_gpu_name(o.gpu_name) == wanted
        and o.num_gpus == num_gpus
        and o.hourly_usd <= max_hourly
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
        [vastai_bin(), "search", "offers",
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


def create_instance_command(
    offer_id: int,
    image: str,
    env: dict[str, str],
    onstart_path: Path,
    disk_gb: int,
) -> list[str]:
    """argv for `vastai create instance`.

    Secrets are deliberately absent — see SECRET_ENV_KEYS. Everything else is
    passed as Vast's '-e KEY=VAL' env string."""
    public = " ".join(
        f"-e {k}={v}" for k, v in sorted(env.items()) if k not in SECRET_ENV_KEYS
    )
    return [
        vastai_bin(), "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--onstart", str(onstart_path),
        "--env", public,
        "--ssh", "--direct",
        "--raw",
    ]


def render_onstart_with_secrets(script: str, env: dict[str, str]) -> str:
    """Inline secret exports at the top of the onstart body.

    The script is uploaded as a file, so this keeps the token out of argv while
    still reaching the instance."""
    secrets = {k: v for k, v in env.items() if k in SECRET_ENV_KEYS and v}
    if not secrets:
        return script
    exports = "\n".join(f"export {k}={v!r}" for k, v in sorted(secrets.items()))
    lines = script.splitlines(keepends=True)
    # Keep the shebang first; a shebang anywhere else is just a comment.
    if lines and lines[0].startswith("#!"):
        return lines[0] + exports + "\n" + "".join(lines[1:])
    return exports + "\n" + script


def launch_instance(
    offer: Offer,
    env: dict[str, str],
    script: str,
    image: str = IMAGE,
    disk_gb: int = 80,
) -> dict:
    """Rent `offer` and start the run. Returns Vast's created-instance payload."""
    directory = Path(tempfile.mkdtemp(prefix="gppb-onstart-"))
    onstart = directory / "onstart.sh"
    onstart.write_text(render_onstart_with_secrets(script, env))
    # The file holds a live token until the launch completes.
    onstart.chmod(stat.S_IRUSR | stat.S_IWUSR)
    try:
        out = subprocess.run(
            create_instance_command(offer.id, image, env, onstart, disk_gb),
            capture_output=True, text=True, check=True,
        ).stdout
    finally:
        onstart.unlink(missing_ok=True)
        os.rmdir(directory)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"raw": out.strip()}
