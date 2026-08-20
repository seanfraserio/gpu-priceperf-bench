import httpx
import pytest
from gppb.openrouter import (
    PROVIDERS, pin_provider_body, SpendCap, SpendCapExceeded, fetch_pricing,
)


def test_provider_list_matches_the_five_that_serve_the_model():
    assert PROVIDERS == ["chutes", "akashml", "venice", "reka", "io-net"]


def test_pin_provider_body_disables_fallbacks():
    """A run that silently falls back to another vendor is mislabelled data."""
    body = pin_provider_body("chutes")
    assert body["provider"]["order"] == ["chutes"]
    assert body["provider"]["allow_fallbacks"] is False


def test_spend_cap_allows_spend_under_the_limit():
    cap = SpendCap(max_output_tokens=1000)
    cap.charge(600)
    cap.charge(300)
    assert cap.spent == 900


def test_spend_cap_raises_before_exceeding_the_limit():
    cap = SpendCap(max_output_tokens=1000)
    cap.charge(900)
    with pytest.raises(SpendCapExceeded):
        cap.charge(200)


async def test_fetch_pricing_reads_live_rates_not_hardcoded_ones():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{
            "id": "qwen/qwen3.8-27b",
            "endpoint": {"provider_name": "Chutes",
                         "pricing": {"prompt": "0.0000004", "completion": "0.000003"}},
        }]})

    pricing = await fetch_pricing(
        "chutes", "qwen/qwen3.8-27b", "key", transport=httpx.MockTransport(handler)
    )
    assert pricing.input_per_mtok_usd == pytest.approx(0.40)
    assert pricing.output_per_mtok_usd == pytest.approx(3.00)


async def test_fetch_pricing_errors_when_provider_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    with pytest.raises(LookupError):
        await fetch_pricing(
            "venice", "qwen/qwen3.8-27b", "key", transport=httpx.MockTransport(handler)
        )
