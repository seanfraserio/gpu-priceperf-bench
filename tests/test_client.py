import json
import httpx
import pytest
from gppb.client import stream_one, RequestMetrics
from gppb.models import Workload


def _sse(chunks: list[str]) -> bytes:
    body = "".join(
        'data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % c for c in chunks
    )
    return (body + "data: [DONE]\n\n").encode()


async def test_stream_one_counts_output_tokens_from_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(["a", "b", "c", "d"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        m = await stream_one(client, "http://x", "m", "prompt", Workload())
    assert m.error is None
    assert m.output_tokens == 4


async def test_stream_one_reports_positive_ttft_and_tpot():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(["a", "b", "c"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        m = await stream_one(client, "http://x", "m", "prompt", Workload())
    assert m.ttft_ms > 0
    assert m.tpot_ms >= 0
    assert m.total_ms >= m.ttft_ms


async def test_stream_one_sends_the_workload_constraints_in_the_body():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse(["a"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await stream_one(client, "http://x", "mymodel", "prompt", Workload())

    assert seen["model"] == "mymodel"
    assert seen["stream"] is True
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 256
    assert seen["ignore_eos"] is True


async def test_stream_one_captures_http_errors_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"overloaded")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        m = await stream_one(client, "http://x", "m", "prompt", Workload())
    assert m.error is not None
    assert m.output_tokens == 0


async def test_single_token_response_has_zero_tpot():
    """TPOT is undefined with one token — must be 0.0, never a divide-by-zero."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(["only"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        m = await stream_one(client, "http://x", "m", "prompt", Workload())
    assert m.tpot_ms == 0.0


async def test_token_count_prefers_the_servers_usage_report():
    """Counting SSE chunks assumes one token per chunk. That holds for vLLM —
    verified against real runs, 4 requests x 256 = exactly 1024 — but nothing
    obliges a managed provider to stream one token at a time. A provider
    batching four tokens per chunk would have its throughput understated
    fourfold, biasing the comparison towards self-hosting."""
    chunks = [
        'data: {"choices":[{"delta":{"content":"alpha beta gamma delta"}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":4}}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join(chunks) + "\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        m = await stream_one(client, "http://t", "m", "p", Workload())

    assert m.output_tokens == 4, "must trust the server's own count"
    assert m.counted_from == "usage"


async def test_falls_back_to_chunk_counting_when_usage_is_absent():
    """Not every provider reports usage on a stream. Falling back is right;
    doing so silently is not, so the result records which count it used."""
    chunks = [
        'data: {"choices":[{"delta":{"content":"a"}}]}',
        'data: {"choices":[{"delta":{"content":"b"}}]}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join(chunks) + "\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        m = await stream_one(client, "http://t", "m", "p", Workload())

    assert m.output_tokens == 2
    assert m.counted_from == "chunks"


async def test_include_usage_is_requested():
    """The authoritative count only arrives if it is asked for."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await stream_one(client, "http://t", "m", "p", Workload())

    assert captured["stream_options"] == {"include_usage": True}


async def _stream(chunks: list[str]) -> RequestMetrics:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="".join(f"data: {c}\n\n" for c in chunks),
                              headers={"Content-Type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await stream_one(client, "http://x", "m", "p", Workload())


async def test_reasoning_tokens_count_as_output():
    """Qwen3.8-27B is a reasoning model. OpenRouter streams its thinking in
    delta.reasoning with delta.content empty, while self-hosted vLLM — run here
    without a reasoning parser — delivers the same tokens inside content.

    Counting only content meant the managed providers were scored on a
    fraction of what they produced and billed for, and a request that spent its
    whole budget reasoning was recorded as "no tokens received": a failure. At
    concurrency 1, three of every four buy-side requests failed this way."""
    metrics = await _stream([
        '{"choices":[{"delta":{"content":"","reasoning":"We"}}]}',
        '{"choices":[{"delta":{"content":"","reasoning":" need"}}]}',
        '{"choices":[{"delta":{"content":"","reasoning":" to"}}]}',
        "[DONE]",
    ])
    assert metrics.error is None
    assert metrics.output_tokens == 3
    assert metrics.ttft_ms > 0


async def test_a_stream_that_mixes_reasoning_and_content_counts_both():
    metrics = await _stream([
        '{"choices":[{"delta":{"reasoning":"think"}}]}',
        '{"choices":[{"delta":{"content":"answer"}}]}',
        "[DONE]",
    ])
    assert metrics.output_tokens == 2


async def test_a_stream_with_neither_is_still_a_failure():
    """Empty deltas are keepalives, not tokens — that failure must survive."""
    metrics = await _stream([
        '{"choices":[{"delta":{"content":""}}]}',
        "[DONE]",
    ])
    assert metrics.error == "no tokens received"
