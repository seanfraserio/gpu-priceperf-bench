"""Buy-side baseline via OpenRouter, one result row per provider."""
from __future__ import annotations

import uuid

import httpx

from gppb.models import BenchResult, Hardware, Pricing, Target, Timings, Workload
from gppb.sweep import run_sweep

BASE_URL = "https://openrouter.ai/api"
MODEL = "qwen/qwen3.8-27b"

# The five providers serving Qwen3.8-27B as of 2026-08-19.
PROVIDERS = ["chutes", "akashml", "venice", "reka", "io-net"]


class SpendCapExceeded(RuntimeError):
    pass


class SpendCap:
    """Client-side hard budget. Enforced before requests are sent, because a
    server-side surprise is a surprise on your card."""

    def __init__(self, max_output_tokens: int):
        self.max_output_tokens = max_output_tokens
        self.spent = 0

    def charge(self, n: int) -> None:
        if self.spent + n > self.max_output_tokens:
            raise SpendCapExceeded(
                f"would spend {self.spent + n} output tokens, cap is {self.max_output_tokens}"
            )
        self.spent += n


def pin_provider_body(provider: str) -> dict:
    """Force one provider. A fallback would silently mislabel the result."""
    return {"provider": {"order": [provider], "allow_fallbacks": False}}


async def fetch_pricing(
    provider: str,
    model: str,
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> Pricing:
    """Read live per-token rates. Never hardcode — the post is only as honest
    as the prices it quotes."""
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get(
            f"{BASE_URL}/v1/models/{model}/endpoints",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()

    for entry in payload.get("data", []):
        endpoint = entry.get("endpoint", {})
        if endpoint.get("provider_name", "").lower().replace(".", "-") != provider:
            continue
        prices = endpoint["pricing"]
        return Pricing(
            input_per_mtok_usd=float(prices["prompt"]) * 1_000_000,
            output_per_mtok_usd=float(prices["completion"]) * 1_000_000,
        )
    raise LookupError(f"provider {provider} does not serve {model}")


async def run_openrouter(
    provider: str,
    api_key: str,
    levels: list[int],
    cap: SpendCap,
    sink,
    model: str = MODEL,
) -> BenchResult:
    workload = Workload()
    result = BenchResult(
        run_id=f"or-{provider}-{uuid.uuid4().hex[:8]}",
        target=Target(kind="openrouter", model=model, provider=provider),
        hardware=Hardware(gpu_name="managed", gpu_count=0),
        pricing=await fetch_pricing(provider, model, api_key),
        timings=Timings(),
        workload=workload,
        partial=True,
    )

    async def on_step(steps):
        cap.charge(steps[-1].output_tokens_total)
        result.steps = list(steps)
        await sink.put(result)

    await run_sweep(
        BASE_URL, model, levels, workload, on_step,
        api_key=api_key, extra_body=pin_provider_body(provider),
    )
    result.partial = False
    await sink.put(result)
    return result
