"""Pull results out of the sink into results/.

The instances are destroyed the moment a run ends, so the sink is the only
copy. Fetching is deliberately a separate step from reporting: a result is
reviewed and committed once, and the report then reads a directory that is
under version control rather than a bucket that can change under it."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from gppb.models import BenchResult

SINK_URL = os.environ.get("SINK_URL", "https://gppb-sink.sfraser.workers.dev")
TOKEN_FILE = Path.home() / ".gppb-sink-token"


def list_keys(client: httpx.Client, sink_url: str, token: str) -> list[str]:
    response = client.get(
        f"{sink_url.rstrip('/')}/_list",
        headers={"Authorization": f"Bearer {token}"}, timeout=30.0,
    )
    response.raise_for_status()
    return [obj["key"] for obj in response.json()["objects"]]


def sync(
    directory: Path,
    sink_url: str = SINK_URL,
    token: str | None = None,
    client: httpx.Client | None = None,
    include_partial: bool = False,
) -> list[str]:
    """Download every complete result. Returns the run_ids written.

    Partial results are skipped by default: a run that uploaded four of nine
    levels is a preempted run, and writing it beside the finished ones invites
    exactly the blend that median_rows now refuses."""
    token = token or TOKEN_FILE.read_text().strip()
    directory.mkdir(parents=True, exist_ok=True)
    # Anything already recorded anywhere under results/ is left alone, which
    # includes results archived into a superseded/ subdirectory after being
    # found to have been measured wrongly. They stay in the sink as the record
    # of what ran; without this every sync would drag them back into the
    # directory the report reads.
    already = {path.name for path in directory.rglob("*.json")}
    owned = client or httpx.Client()
    written: list[str] = []
    try:
        for key in list_keys(owned, sink_url, token):
            if key in already:
                continue
            response = owned.get(
                f"{sink_url.rstrip('/')}/{key}",
                headers={"Authorization": f"Bearer {token}"}, timeout=60.0,
            )
            response.raise_for_status()
            # Validated before it lands: a malformed result should fail here,
            # not halfway through rendering the report.
            result = BenchResult.model_validate_json(response.text)
            if result.partial and not include_partial:
                continue
            (directory / key).write_text(response.text)
            written.append(result.run_id)
    finally:
        if client is None:
            owned.close()
    return written


if __name__ == "__main__":
    names = sync(Path("results"))
    print(f"synced {len(names)} complete results")
    for name in sorted(names):
        print(f"  {name}")
