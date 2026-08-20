"""Result upload. Called repeatedly with the same run_id — each call replaces
the object with a more complete result, so a preempted run keeps whatever it
managed to publish."""
from __future__ import annotations

from pathlib import Path

import httpx

from gppb.models import BenchResult


class LocalSink:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def put(self, result: BenchResult) -> str:
        path = self.directory / f"{result.run_id}.json"
        path.write_text(result.model_dump_json(indent=2))
        return str(path)


class R2Sink:
    def __init__(self, put_url_prefix: str, transport: httpx.BaseTransport | None = None):
        self.put_url_prefix = put_url_prefix.rstrip("/")
        self._transport = transport

    async def put(self, result: BenchResult) -> str:
        url = f"{self.put_url_prefix}/{result.run_id}.json"
        async with httpx.AsyncClient(transport=self._transport) as client:
            response = await client.put(
                url,
                content=result.model_dump_json(indent=2).encode(),
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )
        if response.status_code >= 300:
            raise RuntimeError(f"sink upload failed: {response.status_code}")
        return url


def make_sink(sink_url: str | None, local_dir: Path = Path("results")):
    return R2Sink(sink_url) if sink_url else LocalSink(local_dir)
