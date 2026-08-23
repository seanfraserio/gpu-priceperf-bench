"""Create the vast.ai templates, one per model.

Each carries the floors this benchmark learned by paying for their absence:
a disk floor (a 39GB host died mid-download), a bandwidth floor (the meter runs
during the pull), a CUDA floor (an old driver dies at startup with error 803),
and a VRAM floor sized to the model rather than to the card.

Never public. The onstart body is the harness's, and a public template invites
someone to run it against a sink they cannot write to."""
import math
import subprocess
import sys
from pathlib import Path

from launch.matrix import Model, MODELS
from launch.orchestrate import (
    DISK_GB, MIN_CUDA, MIN_INET_DOWN_MBPS, MIN_RELIABILITY, SINK_URL,
)
from launch.vast import IMAGE, vastai_bin

ROOT = Path(__file__).resolve().parent.parent
ONSTART = (ROOT / "runner-vllm" / "onstart.sh").read_text()
SWEEP = "1,2,4,8,16,32,64,128,256,512"

# Vast's gpu_ram *filter* takes decimal GB while its *replies* are in MB, and
# the two disagree by 7%. Filtering at 46 — the 27B's measured requirement —
# admits a 46,068MB L40S, which is 44.99 GiB and the one card proven to OOM on
# this model. So the floor is derived the same way feasible() decides rather
# than typed in beside it: two expressions of one rule drift apart, and the
# drift is only visible after paying for a boot that cannot finish.
def vram_floor_decimal_gb(model: Model) -> int:
    """The `gpu_ram>=` a card must clear to hold `model`, in the filter's units.

    vLLM reserves 10% of the card, so a model needing N GiB needs a card
    reporting N/0.9 GiB — then GiB to the decimal GB the filter speaks, rounded
    up so the boundary case is excluded rather than admitted."""
    reported_gib = model.required_vram_gb / 0.9
    return math.ceil(reported_gib * 1024 / 1000)


TEMPLATES = {
    "headline": dict(
        name="gppb · Qwen3.8-27B fp8 · serving sweep",
        desc=("Rents one GPU, serves Qwen3.8-27B at fp8 on vLLM 0.27.1, sweeps "
              "concurrency 1-512 at 1024 in / 256 out, uploads the result and "
              "destroys itself. Filters exclude cards the model does not fit."),
    ),
    "anchor": dict(
        name="gppb · Qwen3-8B bf16 · serving sweep",
        desc=("Same harness and workload as the 27B template, at bfloat16 on "
              "the smaller model. The anchor tier: it fits every card here, so "
              "it is what makes the GPUs comparable to each other."),
    ),
}

README = """## What this runs

The harness from github.com/seanfraserio/gpu-priceperf-bench. On boot it clones
the repo at `GPPB_REF`, serves `{model}` on vLLM, sweeps concurrency
{sweep}, and uploads one JSON result per level. Then it destroys itself.

A run takes roughly {minutes} minutes including the weight download.

## Before you launch

Nothing is required. The instance reads its own hourly rate from the
marketplace, so the cost figures are the price you are actually paying rather
than one typed in by hand.

To collect results automatically, add `-e SINK_TOKEN=...` to the Docker options
field. Without it the run still completes and still self-destructs; the result
is only in the log. The token is never stored in this template.

## The guards

* `TTL_MINUTES` destroys the instance on a timer, armed before anything that
  can fail or hang. A second, longer backstop covers a script that dies before
  the first one arms.
* Any non-zero exit uploads the log tail before destroying, so a failed run is
  diagnosable instead of just absent.
* `--max-num-seqs` is capped at the top sweep level. vLLM defaults it to 1024;
  this model is a hybrid, every decode sequence holds a Mamba cache block, and
  an 80GB card has ~823 of them — the default loses the whole boot to a failed
  CUDA graph capture before serving one token.

## The search filters, and why

* **disk >= {disk}GB** — the weights plus the image. A 39GB host was rented
  once and died partway through the download; the marketplace had accepted a
  120GB request against it without complaint.
* **inet_down >= {inet} Mbps** — the meter runs during the pull, so a slow link
  costs more overall than a dearer host that starts work sooner.
* **reliability >= {rel}** — a host that drops the instance mid-sweep bills for
  everything up to that point and returns nothing.
* **cuda >= {cuda}** — an older driver kills vLLM at startup with CUDA error
  803, after the pull and the weights are already paid for.
* **gpu_ram >= {vram}** — sized to this model, in the decimal GB this filter
  speaks (its replies are in MB, and the two disagree by 7%). Set from
  measurement, not the spec sheet: the 27B OOMs on a 45GiB L40S with 59MiB
  free, so the floor is high enough to exclude it.

Raising a floor is how a tier silently stops being rentable, so change these
knowing that an empty result list means no offers, not no GPUs.
"""


def search_params(vram_gb: int) -> str:
    return (
        f"num_gpus=1 rentable=true "
        f"gpu_ram>={vram_gb} "
        f"disk_space>={DISK_GB} "
        f"inet_down>={int(MIN_INET_DOWN_MBPS)} "
        f"reliability>{MIN_RELIABILITY} "
        f"cuda_max_good>={MIN_CUDA}"
    )


def env_field(model_key: str) -> str:
    model = MODELS[model_key]
    pairs = {
        "MODEL": model.hf_id,
        "PRECISION": model.precision,
        "TP_SIZE": "1",
        "SWEEP": SWEEP,
        "MAX_MODEL_LEN": "32768",
        "RUN_INDEX": "1",
        "TTL_MINUTES": "90",
        "TTL_BACKSTOP_MINUTES": "120",
        "GPPB_REF": "main",
        "SINK_URL": SINK_URL,
    }
    return " ".join(f"-e {k}={v}" for k, v in sorted(pairs.items()))


def create(model_key: str) -> dict:
    spec = TEMPLATES[model_key]
    model = MODELS[model_key]
    vram_gb = vram_floor_decimal_gb(model)
    readme = README.format(
        model=model.hf_id, sweep=SWEEP, disk=DISK_GB,
        inet=int(MIN_INET_DOWN_MBPS), rel=MIN_RELIABILITY, cuda=MIN_CUDA,
        vram=vram_gb,
        minutes=int(25 * model.runtime_multiplier),
    )
    argv = [
        vastai_bin(), "create", "template",
        "--name", spec["name"],
        "--image", IMAGE,
        "--desc", spec["desc"],
        "--readme", readme,
        "--env", env_field(model_key),
        "--onstart-cmd", ONSTART,
        "--search_params", search_params(vram_gb),
        "--disk_space", str(DISK_GB),
        "--repo", "https://github.com/seanfraserio/gpu-priceperf-bench",
        "--ssh", "--direct",
        "--raw",
    ]
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"create failed for {model_key}:\n{out.stdout}\n{out.stderr}")
    # `--raw` does not make this JSON: the CLI prints "New Template: {...}" with
    # a Python repr. Parsing it would be guessing; the created template is
    # confirmed by listing them afterwards instead.
    if "New Template" not in out.stdout:
        raise SystemExit(f"unexpected reply for {model_key}:\n{out.stdout[:400]}")
    return {"ok": True, "stdout_head": out.stdout[:120]}


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["headline", "anchor"]
    for key in wanted:
        create(key)
        print(f"created: {TEMPLATES[key]['name']}")
