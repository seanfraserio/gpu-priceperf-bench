import json
from pathlib import Path
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


async def test_r2_sink_authenticates_the_upload():
    """The sink endpoint is write-authenticated — an unauthenticated PUT would
    mean anyone can forge published benchmark results."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200)

    sink = R2Sink(
        "https://sink.example.com/runs",
        token="s3cret",
        transport=httpx.MockTransport(handler),
    )
    await sink.put(_result("xyz"))
    assert seen["auth"] == "Bearer s3cret"


async def test_r2_sink_surfaces_a_rejected_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    sink = R2Sink("https://sink.example.com/runs", token="wrong",
                  transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="401"):
        await sink.put(_result())


def test_make_sink_reads_the_token_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SINK_TOKEN", "from-env")
    sink = make_sink("https://sink.example.com/runs", tmp_path)
    assert isinstance(sink, R2Sink)
    assert sink.token == "from-env"


def test_local_sink_directory_is_overridable_by_environment(monkeypatch, tmp_path):
    """The dry-run gate must not write mock results into the directory the
    published report reads, or fabricated rows appear beside real ones."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "dryrun"))
    sink = make_sink(None)
    assert isinstance(sink, LocalSink)
    assert sink.directory == tmp_path / "dryrun"


def test_local_sink_defaults_to_results(monkeypatch):
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    assert make_sink(None).directory == Path("results")
