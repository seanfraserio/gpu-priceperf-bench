"""Buy-side baseline via OpenRouter, one result row per provider."""
from __future__ import annotations

import uuid

import httpx

from gppb.models import BenchResult, Hardware, Pricing, Target, Timings, Workload
from gppb.sweep import run_sweep

BASE_URL = "https://openrouter.ai/api"
MODEL = "qwen/qwen3.8-27b"

# Every provider serving Qwen3.8-27B, from OpenRouter's endpoint list on
# 2026-08-21. CoreWeave and Parasail were missing from the original five, and
# CoreWeave ties Chutes as the cheapest at $3.00 per 1M output tokens —
# leaving the cheapest buy-side option out would have overstated every
# self-host-wins claim in the writeup.
#
# Alibaba is excluded deliberately and not by oversight: it is the first-party
# vendor at a 1M context window, a different product from the resellers.
#
# AkashML serves bf16 where the rest serve fp8. That is not normalised away —
# it is what the endpoint actually is, and the precision belongs in the result.
PROVIDERS = ["coreweave", "chutes", "reka", "venice", "parasail", "akashml", "io-net"]


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


def project_level_tokens(concurrency: int, workload: Workload, requests_per_step: int = 0) -> int:
    """Output tokens one level will bill.

    `ignore_eos` forces every request to emit exactly `output_tokens`, so this
    is the real figure rather than an optimistic floor."""
    requests = requests_per_step or concurrency * 4
    return requests * workload.output_tokens


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

    async def before_step(level: int) -> None:
        # Refuse the level while refusing still prevents the spend.
        cap.charge(project_level_tokens(level, workload))

    async def on_step(steps):
        result.steps = list(steps)
        await sink.put(result)

    await run_sweep(
        BASE_URL, model, levels, workload, on_step,
        api_key=api_key, extra_body=pin_provider_body(provider),
        before_step=before_step,
    )
    result.partial = False
    await sink.put(result)
    return result
