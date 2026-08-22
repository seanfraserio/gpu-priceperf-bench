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
                  inet_down_mbps=1000.0, reliability=0.99, vram_gb=32.0,
                  machine_id=4242)
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
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys,
                               is_complete=lambda key: True) == "completed"


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


def _alive_forever(monkeypatch, keys_before, keys_after):
    """The instance never goes away on its own — which is what happened: a
    complete result sat in the sink for fifty minutes while the container
    failed to destroy itself and the meter ran at $0.80/hr."""
    keys = _stub_launch(monkeypatch, keys_before, keys_after)
    monkeypatch.setattr(orchestrate, "_instances", lambda: [{"id": 99}])
    destroyed = []
    monkeypatch.setattr(orchestrate.subprocess, "run",
                        lambda *a, **k: destroyed.append(a) or None)
    return keys, destroyed


def test_a_complete_result_ends_the_run_without_waiting_for_the_container(monkeypatch):
    keys, destroyed = _alive_forever(monkeypatch, {"old.json"},
                                     {"old.json", "vllm-x-tp1-a.json"})
    outcome = orchestrate.run_one(RUN, "ref", sink_keys=keys,
                                  is_complete=lambda key: True)
    assert outcome == "completed"
    assert destroyed, "the instance must be destroyed once its result is banked"


def test_a_partial_result_is_not_a_finished_run(monkeypatch):
    """Every level uploads. Stopping at the first object would truncate the
    sweep at concurrency 1."""
    keys, destroyed = _alive_forever(monkeypatch, {"old.json"},
                                     {"old.json", "vllm-x-tp1-a.json"})
    monkeypatch.setattr(orchestrate, "timers_for",
                        lambda minutes: orchestrate.Timers(1, 0, 2))
    outcome = orchestrate.run_one(RUN, "ref", sink_keys=keys,
                                  is_complete=lambda key: False)
    assert outcome == "timeout"


def test_an_uploaded_failure_log_ends_the_run_immediately(monkeypatch):
    keys, destroyed = _alive_forever(monkeypatch, {"old.json"},
                                     {"old.json", "fail-1-2.json"})
    outcome = orchestrate.run_one(RUN, "ref", sink_keys=keys,
                                  is_complete=lambda key: True)
    assert outcome == "failed"
    assert destroyed


def test_a_partial_upload_left_by_a_dead_instance_is_not_a_success(monkeypatch):
    """A run that died at level 256 leaves eight levels in the sink. Calling
    that 'completed' resets the failure streak, which is the guard that stops
    a systemic failure from spending the rest of the budget."""
    keys = _stub_launch(monkeypatch, {"old.json"},
                        {"old.json", "vllm-x-tp1-a.json"})
    outcome = orchestrate.run_one(RUN, "ref", sink_keys=keys,
                                  is_complete=lambda key: False)
    assert outcome == "no-result"


def test_a_host_that_returns_nothing_is_not_rented_again(monkeypatch, tmp_path):
    """Three of four RTX 5090 runs went to one machine and came back empty,
    because it was the cheapest and nothing remembered the last time."""
    from launch.blocklist import Blocklist

    blocked = Blocklist(tmp_path / "b.json")
    keys = _stub_launch(monkeypatch, {"old.json"}, {"old.json"})
    monkeypatch.setattr(orchestrate, "BLOCKLIST", blocked)
    assert orchestrate.run_one(RUN, "ref", sink_keys=keys) == "no-result"
    assert blocked.machines() == {4242}


def test_a_host_that_delivered_stays_available(monkeypatch, tmp_path):
    from launch.blocklist import Blocklist

    blocked = Blocklist(tmp_path / "b.json")
    keys = _stub_launch(monkeypatch, {"old.json"}, {"old.json", "vllm-x-tp1-a.json"})
    monkeypatch.setattr(orchestrate, "BLOCKLIST", blocked)
    orchestrate.run_one(RUN, "ref", sink_keys=keys, is_complete=lambda key: True)
    assert blocked.machines() == set()


def test_a_tier_with_no_rentable_host_skips_instead_of_killing_the_sweep(monkeypatch):
    """The blocklist and the floors can between them empty a tier's pool. That
    is a reason to move to the next run, not to abandon sixteen paid runs with
    an unhandled LookupError."""
    monkeypatch.setattr(orchestrate, "reap", lambda: [])
    monkeypatch.setattr(orchestrate, "preflight", lambda: [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 40.0)
    seen = []

    def sometimes_starved(run, ref, **kwargs):
        seen.append(run.tier_key)
        if run.tier_key == "L40S":
            raise LookupError("no 1x L40S — not renting")
        return "completed"

    monkeypatch.setattr(orchestrate, "run_one", sometimes_starved)
    orchestrate.run_matrix(dry_run=False, resume=False, limit=6)
    assert len(seen) == 6, "a starved tier must not end the sweep"


def test_preflight_uses_the_same_floors_the_sweep_will(monkeypatch):
    """A preflight that checks looser floors than run_one reports a tier as
    rentable and then the sweep cannot rent it — which is the report being
    worse than useless, because it was consulted instead of the truth."""
    import inspect

    source = inspect.getsource(orchestrate.preflight)
    for floor in ("min_inet_down_mbps", "min_reliability", "min_vram_gb",
                  "min_cuda", "blocked"):
        assert floor in source, floor
