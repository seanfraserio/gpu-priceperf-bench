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
