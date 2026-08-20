"""Wrap parsed nccl-tests output in a BenchResult and upload it."""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

from gppb.models import BenchResult, Hardware, Pricing, Target, Timings, Workload
from gppb.nccl_parse import parse_nccl_output
from gppb.sink import make_sink


async def main(paths: list[str]) -> int:
    rows = []
    for path in paths:
        with open(path) as handle:
            rows.extend(parse_nccl_output(handle.read()))

    result = BenchResult(
        run_id=f"nccl-{uuid.uuid4().hex[:8]}",
        target=Target(kind="nccl", model="n/a"),
        hardware=Hardware(
            gpu_name=os.environ.get("GPU_NAME", "unknown"),
            gpu_count=int(os.environ.get("GPU_COUNT", "2")),
        ),
        pricing=Pricing(hourly_rate_usd=float(os.environ.get("HOURLY_RATE_USD", "0"))),
        timings=Timings(),
        workload=Workload(),
        nccl_rows=rows,
    )
    await make_sink(os.environ.get("SINK_URL")).put(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
