import pytest
from launch.vast import Offer, select_offer, build_env
from launch.reap import stale_instances


def _offers():
    return [
        Offer(id=1, gpu_name="H100 80GB", num_gpus=1, hourly_usd=2.60),
        Offer(id=2, gpu_name="H100 80GB", num_gpus=1, hourly_usd=1.85),
        Offer(id=3, gpu_name="A100 80GB", num_gpus=1, hourly_usd=0.90),
        Offer(id=4, gpu_name="H100 80GB", num_gpus=2, hourly_usd=3.70),
    ]


def test_select_offer_picks_cheapest_match():
    assert select_offer(_offers(), "H100 80GB", 1, max_hourly=3.00).id == 2


def test_select_offer_respects_gpu_count():
    assert select_offer(_offers(), "H100 80GB", 2, max_hourly=5.00).id == 4


def test_select_offer_aborts_rather_than_overspend():
    """Never silently rent something pricier than the ceiling."""
    with pytest.raises(LookupError):
        select_offer(_offers(), "H100 80GB", 1, max_hourly=1.00)


def test_build_env_passes_the_price_actually_paid():
    env = build_env("Qwen/Qwen3.8-27B", "fp8", 1, 1, 1.85, 45, "https://r2/x")
    assert env["HOURLY_RATE_USD"] == "1.85"
    assert env["TTL_MINUTES"] == "45"
    assert env["MODEL"] == "Qwen/Qwen3.8-27B"


def test_build_env_always_sets_a_ttl():
    env = build_env("m", "fp8", 1, 1, 1.0, 45, None)
    assert int(env["TTL_MINUTES"]) > 0


def test_stale_instances_flags_past_ttl():
    now = 10_000.0
    instances = [
        {"id": 11, "start_date": now - 3600, "ttl_minutes": 45},  # 60min old, TTL 45
        {"id": 12, "start_date": now - 600, "ttl_minutes": 45},   # 10min old
    ]
    assert stale_instances(instances, now) == [11]


def test_stale_instances_defaults_ttl_when_missing():
    """An instance with no recorded TTL is treated as stale after 60 minutes —
    an untracked GPU is the expensive failure mode."""
    now = 10_000.0
    assert stale_instances([{"id": 13, "start_date": now - 4000}], now) == [13]
