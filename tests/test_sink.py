import json
import httpx
import pytest
from gppb.sink import LocalSink, R2Sink, make_sink
from gppb.models import BenchResult, Target, Hardware, Pricing, Timings, Workload


def _result(run_id="r1") -> BenchResult:
    return BenchResult(
        run_id=run_id,
        target=Target(kind="vllm", model="Qwen/Qwen3.8-27B", precision="fp8", tp_size=1),
        hardware=Hardware(gpu_name="H100"),
        pricing=Pricing(hourly_rate_usd=1.9),
        timings=Timings(),
        workload=Workload(),
    )


async def test_local_sink_writes_json_named_by_run_id(tmp_path):
    sink = LocalSink(tmp_path)
    path = await sink.put(_result("abc"))
    assert (tmp_path / "abc.json").exists()
    assert json.loads((tmp_path / "abc.json").read_text())["run_id"] == "abc"
    assert path.endswith("abc.json")


async def test_local_sink_overwrites_on_repeat_put(tmp_path):
    """Partial uploads reuse the run_id; later writes must replace earlier ones."""
    sink = LocalSink(tmp_path)
    r = _result("abc")
    await sink.put(r)
    r.partial = False
    r.steps = []
    await sink.put(r)
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_r2_sink_puts_to_prefixed_url(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200)

    sink = R2Sink("https://bucket.example.com/runs", transport=httpx.MockTransport(handler))
    await sink.put(_result("xyz"))
    assert seen["url"] == "https://bucket.example.com/runs/xyz.json"
    assert json.loads(seen["body"])["run_id"] == "xyz"


async def test_r2_sink_raises_on_upload_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    sink = R2Sink("https://bucket.example.com/runs", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="403"):
        await sink.put(_result())


def test_make_sink_falls_back_to_local_without_a_url(tmp_path):
    assert isinstance(make_sink(None, tmp_path), LocalSink)
    assert isinstance(make_sink("https://x/y", tmp_path), R2Sink)
