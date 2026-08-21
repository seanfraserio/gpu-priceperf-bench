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


def test_onstart_runs_the_stock_pinned_vllm_image():
    """No custom image — the version pin lives in the upstream tag."""
    from launch.vast import IMAGE
    assert IMAGE == "vllm/vllm-openai:v0.27.1"


def test_onstart_arms_the_ttl_before_fetching_anything():
    """A hang in the clone or the weight download must still hit the TTL —
    so the self-destruct is armed before the first thing that can block."""
    from launch.vast import onstart_script
    script = onstart_script()
    ttl_at = script.index("poweroff -f")
    assert ttl_at < script.index("git clone")
    assert ttl_at < script.index("vllm serve")


def test_onstart_pins_the_harness_revision():
    """A rented GPU must never run whatever happens to be on the branch tip."""
    from launch.vast import onstart_script
    assert "GPPB_REF" in onstart_script()


def test_build_env_carries_the_harness_revision():
    env = build_env("m", "fp8", 1, 1, 1.0, 45, None, gppb_ref="abc1234")
    assert env["GPPB_REF"] == "abc1234"


def test_nccl_runs_a_stock_cuda_devel_image():
    """nccl-tests needs nvcc, so the base is the devel image — still stock."""
    from launch.vast import NCCL_IMAGE
    assert NCCL_IMAGE == "nvidia/cuda:12.6.0-devel-ubuntu22.04"


def test_nccl_onstart_arms_the_ttl_before_the_build():
    """Compiling nccl-tests is billed GPU time — a hung apt or make must still
    hit the self-destruct."""
    from launch.vast import nccl_onstart_script
    script = nccl_onstart_script()
    ttl_at = script.index("poweroff -f")
    assert ttl_at < script.index("apt-get")
    assert ttl_at < script.index("all_reduce_perf")


def test_nccl_onstart_pins_both_revisions():
    """The harness ref and nccl-tests itself are both pinned — an upstream
    change to the benchmark must never silently alter published numbers."""
    from launch.vast import nccl_onstart_script
    script = nccl_onstart_script()
    assert "GPPB_REF" in script
    assert "NCCL_TESTS_REF" in script


def test_build_env_carries_the_sink_credentials_together():
    """A URL without its token means the instance uploads nothing and the run
    is paid for twice — they travel as a pair."""
    env = build_env("m", "fp8", 1, 1, 1.0, 45, "https://sink.example.com",
                    sink_token="s3cret")
    assert env["SINK_URL"] == "https://sink.example.com"
    assert env["SINK_TOKEN"] == "s3cret"


def test_build_env_omits_the_token_when_there_is_no_sink():
    env = build_env("m", "fp8", 1, 1, 1.0, 45, None, sink_token="s3cret")
    assert "SINK_TOKEN" not in env


def test_vastai_binary_resolves_from_the_active_interpreter(monkeypatch, tmp_path):
    """The CLI usually lives in the venv alongside python, which is not on PATH
    when the venv is unactivated. The reaper is the money guard — it must never
    fail to find its own tool."""
    import shutil, sys
    from launch import vast

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "vastai").write_text("#!/bin/sh\n")
    (fake_bin / "vastai").chmod(0o755)

    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert vast.vastai_bin() == str(fake_bin / "vastai")


def test_vastai_binary_falls_back_to_path(monkeypatch):
    import shutil, sys
    from launch import vast

    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/vastai")
    assert vast.vastai_bin() == "/usr/local/bin/vastai"


def test_missing_vastai_names_the_install_command(monkeypatch):
    """A cryptic FileNotFoundError while a GPU bills by the second is the wrong
    failure — say what to install."""
    import shutil, sys
    from launch import vast

    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install vastai"):
        vast.vastai_bin()
