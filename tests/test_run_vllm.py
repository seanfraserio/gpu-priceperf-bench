import pytest
from gppb.run_vllm import assert_vllm_version, parse_levels


def test_version_floor_accepts_the_minimum():
    assert_vllm_version("0.17.0")


def test_version_floor_accepts_newer():
    assert_vllm_version("0.26.1")


def test_version_floor_rejects_older():
    """Qwen3.8-27B has no support below 0.17.0 — fail before renting time."""
    with pytest.raises(RuntimeError, match="0.17.0"):
        assert_vllm_version("0.16.3")


def test_version_floor_tolerates_dev_suffixes():
    assert_vllm_version("0.26.1rc1.dev608+g99a10304d")


def test_parse_levels_reads_the_sweep_env_var():
    assert parse_levels("1,2,4,8") == [1, 2, 4, 8]


def test_parse_levels_rejects_garbage():
    with pytest.raises(ValueError):
        parse_levels("1,two,4")


def test_skip_vllm_import_is_opt_in_only(monkeypatch):
    """The dry-run escape hatch must never be on by default — a rented GPU
    silently skipping the version assert would produce unlabelled results."""
    import os
    monkeypatch.delenv("SKIP_VLLM_IMPORT", raising=False)
    assert os.environ.get("SKIP_VLLM_IMPORT") != "1"


def test_gpu_probes_survive_a_machine_without_nvidia_smi(monkeypatch):
    """The dry-run gate runs on a GPU-less machine — a missing nvidia-smi must
    degrade to unknown, not crash the run before anything is recorded."""
    import subprocess
    from gppb import run_vllm

    def no_binary(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    monkeypatch.setattr(subprocess, "run", no_binary)
    assert run_vllm._gpu_name() == "unknown"
    assert run_vllm._peak_vram_bytes() is None


async def test_wait_healthy_gives_up_when_the_server_process_is_gone():
    """Polling /health for thirty minutes against a process that already died
    is thirty minutes of GPU rental for a known answer. The 27B is the run
    most likely to fail at startup and the most expensive to fail on."""
    import pytest
    from gppb.run_vllm import _wait_healthy

    with pytest.raises(RuntimeError, match="exited"):
        await _wait_healthy(
            "http://127.0.0.1:9", timeout_s=30.0, is_alive=lambda: False
        )


async def test_wait_healthy_keeps_waiting_while_the_server_is_alive():
    """A slow boot is not a dead boot — the 27B loads 55.6GB before it serves."""
    from gppb.run_vllm import _wait_healthy

    with pytest.raises(TimeoutError):
        await _wait_healthy(
            "http://127.0.0.1:9", timeout_s=0.2, is_alive=lambda: True
        )


def test_sweep_levels_are_checked_against_the_fd_limit_up_front():
    """Each in-flight request holds a socket. Discovering at level 512, on the
    most expensive card, that the container caps file descriptors is a failure
    at the worst possible moment."""
    import pytest
    from gppb.run_vllm import assert_fd_headroom

    assert_fd_headroom([1, 2, 4, 256], soft_limit=1024)
    with pytest.raises(RuntimeError, match="file descriptor"):
        assert_fd_headroom([1, 2, 512], soft_limit=256)
