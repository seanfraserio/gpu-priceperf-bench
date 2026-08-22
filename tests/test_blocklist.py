"""select_offer takes the cheapest qualifying offer, and vast's cheapest RTX
5090 advertises 1508 Mbps and 0.996 reliability. It also failed three runs out
of four. Nothing in the selector remembered that, so every 5090 run went back
to the same machine."""
from __future__ import annotations

from pathlib import Path

import pytest

from launch.blocklist import Blocklist
from launch.vast import Offer, select_offer


def _offer(offer_id: int, price: float, machine: int) -> Offer:
    return Offer(id=offer_id, gpu_name="RTX 5090", num_gpus=1, hourly_usd=price,
                 inet_down_mbps=1500.0, reliability=0.99, vram_gb=32.0,
                 machine_id=machine)


def test_a_machine_that_failed_is_not_rented_again(tmp_path: Path):
    blocked = Blocklist(tmp_path / "b.json")
    blocked.record(machine_id=7, reason="never-started")
    chosen = select_offer(
        [_offer(1, 0.36, machine=7), _offer(2, 0.44, machine=8)],
        "RTX_5090", 1, max_hourly=1.0, blocked=blocked.machines(),
    )
    assert chosen.id == 2


def test_the_blocklist_survives_the_process_that_wrote_it(tmp_path: Path):
    """A sweep is restarted often — after a fix, after an interruption. A
    blocklist that only lives in memory forgets the bad host every time."""
    path = tmp_path / "b.json"
    Blocklist(path).record(machine_id=7, reason="no-result")
    assert Blocklist(path).machines() == {7}


def test_recording_the_same_machine_twice_counts_it_once(tmp_path: Path):
    path = tmp_path / "b.json"
    blocked = Blocklist(path)
    blocked.record(machine_id=7, reason="a")
    blocked.record(machine_id=7, reason="b")
    assert blocked.machines() == {7}


def test_an_absent_blocklist_blocks_nothing(tmp_path: Path):
    assert Blocklist(tmp_path / "missing.json").machines() == set()


def test_selection_is_unchanged_when_nothing_is_blocked():
    chosen = select_offer(
        [_offer(1, 0.36, machine=7), _offer(2, 0.44, machine=8)],
        "RTX_5090", 1, max_hourly=1.0,
    )
    assert chosen.id == 1


def test_offers_carry_the_machine_they_run_on(monkeypatch):
    """An offer id is a listing; the machine is the thing that keeps failing.
    Blocking the listing would let the same host come back under a new one."""
    import json
    import launch.vast as vast

    raw = json.dumps([{
        "id": 1, "machine_id": 42, "gpu_name": "RTX 5090", "num_gpus": 1,
        "dph_total": "0.36", "inet_down": 1500, "reliability2": 0.99,
        "gpu_ram": 32768,
    }])

    class _Done:
        stdout = raw

    # monkeypatch, not assignment: `subprocess` is the shared stdlib module,
    # and replacing run() on it leaks into every later test.
    monkeypatch.setattr(vast.subprocess, "run", lambda *a, **k: _Done())
    assert vast.search_offers("RTX_5090", 1)[0].machine_id == 42


def test_a_host_whose_driver_is_too_old_is_not_rented():
    """CUDA error 803: the image's runtime needs a newer driver than the host
    has. vast publishes cuda_max_good per offer, so this is a filter, not a
    failure to pay for."""
    old = Offer(id=1, gpu_name="RTX 5090", num_gpus=1, hourly_usd=0.33,
                inet_down_mbps=1500.0, reliability=0.99, vram_gb=32.0,
                machine_id=1, cuda_max_good=12.8)
    new = Offer(id=2, gpu_name="RTX 5090", num_gpus=1, hourly_usd=0.44,
                inet_down_mbps=1500.0, reliability=0.99, vram_gb=32.0,
                machine_id=2, cuda_max_good=13.0)
    assert select_offer([old, new], "RTX_5090", 1, max_hourly=1.0,
                        min_cuda=13.0).id == 2


def test_an_offer_without_a_cuda_version_is_not_assumed_good():
    unknown = Offer(id=1, gpu_name="RTX 5090", num_gpus=1, hourly_usd=0.33,
                    inet_down_mbps=1500.0, reliability=0.99, vram_gb=32.0,
                    machine_id=1)
    with pytest.raises(LookupError):
        select_offer([unknown], "RTX_5090", 1, max_hourly=1.0, min_cuda=13.0)


def test_offers_carry_the_drivers_cuda_ceiling(monkeypatch):
    import json
    import launch.vast as vast

    raw = json.dumps([{
        "id": 1, "machine_id": 42, "gpu_name": "RTX 5090", "num_gpus": 1,
        "dph_total": "0.36", "inet_down": 1500, "reliability2": 0.99,
        "gpu_ram": 32768, "cuda_max_good": 13.0,
    }])

    class _Done:
        stdout = raw

    monkeypatch.setattr(vast.subprocess, "run", lambda *a, **k: _Done())
    assert vast.search_offers("RTX_5090", 1)[0].cuda_max_good == 13.0
