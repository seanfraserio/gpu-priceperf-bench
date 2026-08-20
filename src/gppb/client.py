"""One streaming request, measured. The same code path serves vLLM and
OpenRouter — that identity is what makes the comparison honest."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from gppb.models import Workload


@dataclass
class RequestMetrics:
    ttft_ms: float
    tpot_ms: float
    output_tokens: int
    total_ms: float
    error: str | None = None


async def stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    workload: Workload,
    api_key: str | None = None,
    extra_body: dict | None = None,
) -> RequestMetrics:
    """Issue one streaming completion and time it.

    TTFT  = start -> first token-bearing chunk.
    TPOT  = mean gap across the remaining tokens (0.0 when only one arrives).
    """
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": workload.temperature,
        "max_tokens": workload.output_tokens,
        "ignore_eos": workload.ignore_eos,
    }
    if extra_body:
        body.update(extra_body)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    tokens = 0

    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=httpx.Timeout(300.0),
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestMetrics(
                    0.0, 0.0, 0, (time.perf_counter() - started) * 1000,
                    error=f"HTTP {response.status_code}",
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if not delta.get("content"):
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                tokens += 1
    except (httpx.HTTPError, OSError) as exc:
        return RequestMetrics(
            0.0, 0.0, 0, (time.perf_counter() - started) * 1000, error=str(exc)
        )

    total_ms = (time.perf_counter() - started) * 1000
    if first_token_at is None:
        return RequestMetrics(0.0, 0.0, 0, total_ms, error="no tokens received")

    ttft_ms = (first_token_at - started) * 1000
    if tokens > 1 and last_token_at is not None:
        tpot_ms = (last_token_at - first_token_at) * 1000 / (tokens - 1)
    else:
        tpot_ms = 0.0
    return RequestMetrics(ttft_ms, tpot_ms, tokens, total_ms)
