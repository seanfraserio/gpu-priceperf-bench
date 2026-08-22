"""'completed' meant 'the instance is gone'. Two runs died on boot and were
reported as successes, and the sweep would happily have spent the remaining
nineteen slots the same way. What counts as success is a published result."""
from __future__ import annotations

import types

from launch import orchestrate
from launch.matrix import Run


RUN = Run(tier_key="RTX_5090", model_key="anchor", run_index=1)


def _stub_launch(monkeypatch, keys_before: set[str], keys_after: set[str]):
    """Rent nothing; the instance appears, then disappears immediately."""
    from launch.vast import Offer

    offer = Offer(id=1, hourly_usd=0.35, gpu_name="RTX 5090", num_gpus=1,
                  inet_down_mbps=1000.0, reliability=0.99, vram_gb=32.0)
    monkeypatch.setattr(orchestrate, "search_offers", lambda *a, **k: [offer])
    monkeypatch.setattr(orchestrate, "select_offer", lambda *a, **k: offer)
    monkeypatch.setattr(orchestrate, "launch_instance",
                        lambda *a, **k: {"new_contract": 99})
    monkeypatch.setattr(orchestrate, "onstart_script", lambda: "#!/bin/sh\n")
    monkeypatch.setattr(orchestrate, "TOKEN_FILE",
                        types.SimpleNamespace(read_text=lambda: "t"))
    monkeypatch.setattr(orchestrate, "await_running", lambda *a, **k: True)
    monkeypatch.setattr(orchestrate, "_instances", lambda: [])
    monkeypatch.setattr(orchestrate.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def keys() -> set[str]:
        calls["n"] += 1
        return keys_before if calls["n"] == 1 else keys_after

    return keys


def test_an_instance_that_published_nothing_is_not_completed(monkeypatch):
    keys = _stub_launch(monkeypatch, {"old.json"}, {"old.json"})
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys) == "no-result"


def test_a_published_result_is_what_completed_means(monkeypatch):
    keys = _stub_launch(monkeypatch, {"old.json"}, {"old.json", "vllm-x-tp1-a.json"})
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys) == "completed"


def test_an_uploaded_failure_log_is_reported_as_a_failure(monkeypatch):
    keys = _stub_launch(monkeypatch, {"old.json"}, {"old.json", "fail-123-4.json"})
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys) == "failed"


def test_a_sink_that_cannot_be_listed_does_not_invent_a_failure(monkeypatch):
    """A network blip on the operator's side is not evidence about the run."""
    def keys() -> set[str]:
        raise RuntimeError("sink unreachable")

    _stub_launch(monkeypatch, set(), set())
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys) == "completed"


def test_the_sweep_stops_once_the_failures_are_clearly_systemic(monkeypatch):
    """Every run in the matrix runs the same code on the same kind of host. A
    third consecutive failure is a bug, not bad luck, and spending the rest of
    the budget proving it again buys nothing."""
    monkeypatch.setattr(orchestrate, "reap", lambda: [])
    monkeypatch.setattr(orchestrate, "preflight", lambda: [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 40.0)
    attempts = {"n": 0}

    def always_fails(run, ref, **kwargs):
        attempts["n"] += 1
        return "no-result"

    monkeypatch.setattr(orchestrate, "run_one", always_fails)
    orchestrate.run_matrix(dry_run=False, resume=False)
    assert attempts["n"] == orchestrate.MAX_CONSECUTIVE_FAILURES


def test_a_success_clears_the_failure_streak(monkeypatch):
    monkeypatch.setattr(orchestrate, "reap", lambda: [])
    monkeypatch.setattr(orchestrate, "preflight", lambda: [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 40.0)
    outcomes = iter(["no-result", "no-result", "completed", "no-result"])
    attempts = {"n": 0}

    def scripted(run, ref, **kwargs):
        attempts["n"] += 1
        return next(outcomes, "completed")

    monkeypatch.setattr(orchestrate, "run_one", scripted)
    orchestrate.run_matrix(dry_run=False, resume=False, limit=4)
    assert attempts["n"] == 4
