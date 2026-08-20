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


async def test_spend_cap_blocks_the_level_before_any_request_is_sent(monkeypatch):
    """The cap must refuse a level it cannot afford, not discover the overspend
    after the tokens are already billed."""
    from gppb import openrouter, sweep
    from gppb.client import RequestMetrics

    sent = {"n": 0}

    async def counting_stream_one(client, base_url, model, prompt, workload, api_key=None, extra_body=None):
        sent["n"] += 1
        return RequestMetrics(100.0, 10.0, 256, 2660.0)

    monkeypatch.setattr(sweep, "stream_one", counting_stream_one)

    class _Sink:
        async def put(self, result):
            return "x"

    async def fake_pricing(provider, model, api_key, transport=None):
        from gppb.models import Pricing
        return Pricing(input_per_mtok_usd=0.4, output_per_mtok_usd=3.0)

    monkeypatch.setattr(openrouter, "fetch_pricing", fake_pricing)

    # Level 1 projects 4 requests * 256 tokens = 1024, over the 500 cap.
    cap = openrouter.SpendCap(max_output_tokens=500)
    with pytest.raises(openrouter.SpendCapExceeded):
        await openrouter.run_openrouter("chutes", "key", [1], cap, _Sink())

    assert sent["n"] == 0, "no request may be sent once the cap is known to be exceeded"


def test_spend_cap_projects_a_level_from_the_workload():
    from gppb.openrouter import project_level_tokens
    from gppb.models import Workload
    # ignore_eos forces exactly output_tokens per request, so the projection is
    # exact rather than optimistic.
    assert project_level_tokens(concurrency=4, workload=Workload()) == 4 * 4 * 256
