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
