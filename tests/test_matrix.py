import pytest
from launch.matrix import (
    TIERS, MODELS, build_matrix, feasible, BudgetGate, BudgetExhausted,
    estimate_run_usd,
)


def test_matrix_covers_every_feasible_model_and_tier_three_times():
    runs = build_matrix()
    expected = {
        (m, t) for m in MODELS for t in TIERS if feasible(MODELS[m], TIERS[t])
    }
    assert {(r.model_key, r.tier_key) for r in runs} == expected
    for key in expected:
        assert len([r for r in runs if (r.model_key, r.tier_key) == key]) == 3


def test_run_indices_are_one_two_three():
    runs = build_matrix()
    first = [r for r in runs if r.model_key == "anchor" and r.tier_key == "RTX_5090"]
    assert sorted(r.run_index for r in first) == [1, 2, 3]


def test_a_model_too_large_for_a_tier_is_infeasible():
    """Qwen3.8-27B in FP8 needs ~27GB of weights plus KV cache; a 32GB RTX 5090
    cannot hold it at 0.9 utilisation. Renting one to watch it OOM is money
    spent for a known answer."""
    assert feasible(MODELS["headline"], TIERS["RTX_5090"]) is False
    assert feasible(MODELS["headline"], TIERS["H100_SXM"]) is True
    assert feasible(MODELS["anchor"], TIERS["RTX_5090"]) is True


def test_infeasible_cells_are_excluded_from_the_matrix():
    runs = build_matrix()
    assert not [r for r in runs
                if r.model_key == "headline" and r.tier_key == "RTX_5090"]


def test_cheapest_tier_runs_first():
    """Order by price so a budget that runs out mid-matrix has still bought the
    widest coverage."""
    runs = build_matrix()
    prices = [TIERS[r.tier_key].typical_hourly_usd for r in runs]
    assert prices == sorted(prices)


def test_budget_gate_blocks_a_run_it_cannot_afford():
    gate = BudgetGate(credit_usd=1.00, reserve_usd=0.50)
    with pytest.raises(BudgetExhausted):
        gate.check(estimated_usd=0.80)


def test_budget_gate_allows_a_run_within_the_reserve():
    gate = BudgetGate(credit_usd=5.00, reserve_usd=0.50)
    gate.check(estimated_usd=1.00)


def test_budget_gate_keeps_a_reserve_for_the_unexpected():
    """Spending to zero leaves nothing to rent a replacement if a run is lost."""
    gate = BudgetGate(credit_usd=1.00, reserve_usd=0.50)
    gate.check(estimated_usd=0.40)
    with pytest.raises(BudgetExhausted):
        gate.check(estimated_usd=0.40)


def test_run_estimate_scales_with_price_and_model_size():
    cheap = estimate_run_usd(MODELS["anchor"], TIERS["RTX_5090"])
    dear = estimate_run_usd(MODELS["headline"], TIERS["H100_SXM"])
    assert dear > cheap > 0


def test_orchestrator_reaps_before_it_starts(monkeypatch):
    """A leftover instance from an earlier session is spending money that this
    sweep is about to budget for."""
    from launch import orchestrate
    order = []
    monkeypatch.setattr(orchestrate, "reap", lambda: order.append("reap") or [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 100.0)
    monkeypatch.setattr(orchestrate, "run_one", lambda *a, **k: order.append("run") or "ok")
    orchestrate.run_matrix(limit=1, dry_run=False)
    assert order[0] == "reap"


def test_orchestrator_dry_run_spends_nothing(monkeypatch):
    from launch import orchestrate
    launched = []
    monkeypatch.setattr(orchestrate, "reap", lambda: [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 100.0)
    monkeypatch.setattr(orchestrate, "run_one", lambda *a, **k: launched.append(1))
    plan = orchestrate.run_matrix(dry_run=True)
    assert launched == []
    assert len(plan) == 21


def test_orchestrator_stops_when_the_budget_gate_trips(monkeypatch):
    """Running out mid-sweep must end the sweep, not keep trying."""
    from launch import orchestrate
    monkeypatch.setattr(orchestrate, "reap", lambda: [])
    monkeypatch.setattr(orchestrate, "current_credit", lambda: 1.20)
    calls = []
    monkeypatch.setattr(orchestrate, "run_one", lambda *a, **k: calls.append(1))
    done = orchestrate.run_matrix(dry_run=False)
    assert len(calls) < 21, "the gate must stop the sweep"
    assert len(done) == len(calls)


def test_declared_vram_matches_what_hosts_actually_report():
    """The VRAM floor is checked against the host's reported gpu_ram, so a
    tier declaring more than any real host reports rents nothing at all. These
    are the values observed live on Vast."""
    observed = {"RTX_5090": 32.0, "A100_SXM4": 80.0, "L40S": 45.0, "H100_SXM": 80.0}
    for key, reported in observed.items():
        assert TIERS[key].vram_gb <= reported, (
            f"{key} declares {TIERS[key].vram_gb}GB but hosts report {reported}GB"
        )
        assert TIERS[key].vram_gb * 0.95 <= reported


def test_the_headline_model_still_fits_where_it_should():
    """Correcting the L40S size must not silently drop it from the matrix."""
    assert feasible(MODELS["headline"], TIERS["L40S"]) is True
    assert feasible(MODELS["headline"], TIERS["RTX_5090"]) is False
