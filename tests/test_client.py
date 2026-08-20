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
