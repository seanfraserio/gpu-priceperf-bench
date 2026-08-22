"""The sink is the only copy of a result — the instance is gone."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from launch.sync import sync

_BASE = {
    "target": {"kind": "vllm", "model": "Qwen/Qwen3-8B", "precision": "bfloat16", "tp_size": 1},
    "hardware": {"gpu_name": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    "pricing": {"hourly_rate_usd": 0.35},
    "timings": {"download_seconds": 400.0, "boot_seconds": 340.0},
    "workload": {},
    "run_index": 1,
    "steps": [{
        "concurrency": 1, "requests_completed": 4, "requests_failed": 0,
        "wall_seconds": 10.0, "output_tokens_total": 940, "output_tokens_per_sec": 94.0,
        "ttft_ms": {"p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0},
        "tpot_ms": {"p50": 1.0, "p90": 1.0, "min": 1.0, "max": 1.0},
    }],
}


def _result(run_id: str, partial: bool) -> str:
    return json.dumps({**_BASE, "run_id": run_id, "partial": partial})


def _transport() -> httpx.MockTransport:
    bodies = {
        "done.json": _result("done", partial=False),
        "half.json": _result("half", partial=True),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path == "/_list":
            return httpx.Response(200, json={
                "objects": [{"key": k, "size": len(v), "uploaded": "2026-08-21T19:00:00Z"}
                            for k, v in bodies.items()],
                "truncated": False,
            })
        return httpx.Response(200, text=bodies[request.url.path.lstrip("/")])

    return httpx.MockTransport(handler)


def test_sync_writes_complete_results(tmp_path: Path):
    with httpx.Client(transport=_transport()) as client:
        written = sync(tmp_path, sink_url="https://sink.test", token="secret", client=client)
    assert written == ["done"]
    assert (tmp_path / "done.json").exists()


def test_partial_results_are_left_in_the_sink(tmp_path: Path):
    """A run that uploaded four of nine levels is preempted, not finished.
    Writing it beside the complete runs invites the blend median_rows refuses."""
    with httpx.Client(transport=_transport()) as client:
        sync(tmp_path, sink_url="https://sink.test", token="secret", client=client)
    assert not (tmp_path / "half.json").exists()


def test_partials_can_be_fetched_deliberately(tmp_path: Path):
    """Salvaging a preempted run is a decision the operator makes explicitly."""
    with httpx.Client(transport=_transport()) as client:
        written = sync(tmp_path, sink_url="https://sink.test", token="secret",
                       client=client, include_partial=True)
    assert sorted(written) == ["done", "half"]


def test_sync_does_not_re_download_a_result_already_recorded(tmp_path: Path):
    """Including one filed under a superseded/ subdirectory.

    Results found to have been measured wrongly are archived rather than
    deleted — they are the record of what was actually run — but the sink still
    holds them, so without this every sync would drag them back into the
    directory the report reads."""
    archive = tmp_path / "superseded-client-pool"
    archive.mkdir()
    (archive / "done.json").write_text("{}")

    with httpx.Client(transport=_transport()) as client:
        written = sync(tmp_path, sink_url="https://sink.test", token="secret", client=client)

    assert written == [], "an archived result must not be pulled back in"
    assert not (tmp_path / "done.json").exists()
