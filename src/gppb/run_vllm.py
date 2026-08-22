"""Container entrypoint. Boots vLLM, sweeps, uploads after every level,
returns. The shell wrapper powers the machine off afterwards."""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
import uuid
from typing import Callable

import httpx

from gppb.models import BenchResult, Hardware, Pricing, Target, Timings, Workload
from gppb.sink import make_sink
from gppb.sweep import run_sweep

VLLM_VERSION_FLOOR = (0, 17, 0)


def assert_vllm_version(installed: str, floor: str = "0.17.0") -> None:
    """Qwen3.8-27B has day-0 support only at vLLM >= 0.17.0. Checked before
    any billable work begins."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed)
    if not match:
        raise RuntimeError(f"unparseable vLLM version: {installed}")
    parsed = tuple(int(g) for g in match.groups())
    required = tuple(int(p) for p in floor.split("."))
    if parsed < required:
        raise RuntimeError(f"vLLM {installed} is below the {floor} floor")


def parse_levels(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _nvidia_smi(query: str) -> str:
    """nvidia-smi output, or empty when there is no GPU to ask.

    The dry-run gate executes this module on a GPU-less machine, where the
    binary is missing entirely — that must degrade, never raise."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return ""
    return out.stdout.strip()


def _gpu_name() -> str:
    out = _nvidia_smi("name")
    return out.splitlines()[0] if out else "unknown"


def _peak_vram_bytes() -> int | None:
    out = _nvidia_smi("memory.used")
    try:
        return int(out.splitlines()[0]) * 1024 * 1024
    except (ValueError, IndexError):
        return None


def _server_alive() -> bool:
    """Whether the vLLM the entrypoint launched is still running.

    Absent a PID we assume alive: never turn a missing signal into a failure.
    """
    raw = os.environ.get("VLLM_PID")
    if not raw:
        return True
    try:
        os.kill(int(raw), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


async def _wait_healthy(
    base_url: str,
    timeout_s: float = 1800,
    is_alive: Callable[[], bool] | None = None,
) -> float:
    """Seconds from vLLM process start to first healthy response.

    Anchored to VLLM_START_EPOCH set by the entrypoint, so boot_seconds covers
    the whole server startup rather than only the part Python observed.

    A slow boot is not a dead boot — the 27B reads 55.6GB of weights before it
    serves anything — but polling for thirty minutes against a process that has
    already exited is thirty minutes of GPU rental for a known answer.
    """
    alive = is_alive or _server_alive
    start_epoch = float(os.environ.get("VLLM_START_EPOCH", time.time()))
    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while time.perf_counter() - started < timeout_s:
            try:
                if (await client.get(f"{base_url}/health", timeout=5.0)).status_code == 200:
                    return time.time() - start_epoch
            except httpx.HTTPError:
                pass
            if not alive():
                raise RuntimeError(
                    "vLLM exited before serving — check the server log for the "
                    "startup failure"
                )
            await asyncio.sleep(2.0)
    raise TimeoutError("vLLM never became healthy")


async def main() -> int:
    model = os.environ["MODEL"]
    precision = os.environ.get("PRECISION", "fp8")
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    levels = parse_levels(os.environ.get("SWEEP", "1,2,4,8,16,32,64,128,256"))
    run_index = int(os.environ.get("RUN_INDEX", "1"))
    hourly_rate = float(os.environ["HOURLY_RATE_USD"])
    download_seconds = float(os.environ.get("DOWNLOAD_SECONDS", "0"))
    base_url = "http://127.0.0.1:8000"

    # The dry-run gate executes this module on a machine with no vLLM and no
    # GPU. SKIP_VLLM_IMPORT is that escape hatch and nothing else — it is never
    # set on a rented instance.
    if os.environ.get("SKIP_VLLM_IMPORT") == "1":
        vllm_version = None
    else:
        import vllm
        assert_vllm_version(vllm.__version__)
        vllm_version = vllm.__version__

    boot_seconds = await _wait_healthy(base_url)

    sink = make_sink(os.environ.get("SINK_URL"))
    result = BenchResult(
        run_id=f"vllm-{_gpu_name().replace(' ', '-')}-tp{tp_size}-{uuid.uuid4().hex[:8]}",
        target=Target(kind="vllm", model=model, precision=precision, tp_size=tp_size),
        hardware=Hardware(
            gpu_name=_gpu_name(),
            gpu_count=tp_size,
            vllm_version=vllm_version,
            peak_vram_bytes=_peak_vram_bytes(),
        ),
        pricing=Pricing(hourly_rate_usd=hourly_rate),
        timings=Timings(download_seconds=download_seconds, boot_seconds=boot_seconds),
        workload=Workload(),
        run_index=run_index,
        partial=True,
    )

    async def on_step(steps):
        result.steps = list(steps)
        result.hardware.peak_vram_bytes = _peak_vram_bytes()
        await sink.put(result)

    await run_sweep(base_url, model, levels, Workload(), on_step)
    result.partial = False
    await sink.put(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
