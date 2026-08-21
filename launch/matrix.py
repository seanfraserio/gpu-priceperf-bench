"""The serving sweep matrix: which model on which GPU, how many times, in what
order, and when to stop spending.

Ordering and feasibility are not cosmetic. A budget that runs out halfway
through should have bought the widest coverage it could, and a cell that is
known to OOM should never be rented at all."""
from __future__ import annotations

from dataclasses import dataclass

# Wall-clock estimate per run, from the measured validation run: 123s weight
# download plus 331s boot on an 8B, then the nine-level sweep. The 27B is
# slower on every axis, so it carries its own multiplier.
BASE_RUN_MINUTES = 25.0

RUNS_PER_CONFIG = 3


@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    precision: str
    # Weights plus KV cache and activations at --max-model-len 32768. Used to
    # reject a tier before renting it, not to size anything at runtime.
    required_vram_gb: float
    runtime_multiplier: float


@dataclass(frozen=True)
class Tier:
    key: str
    vram_gb: float
    typical_hourly_usd: float


@dataclass(frozen=True)
class Run:
    model_key: str
    tier_key: str
    run_index: int


MODELS: dict[str, Model] = {
    "anchor": Model("anchor", "Qwen/Qwen3-8B", "bfloat16", 20.0, 1.0),
    "headline": Model("headline", "Qwen/Qwen3.8-27B", "fp8", 34.0, 1.6),
}

# Prices are indicative, for ordering and budgeting only. The rate actually
# paid is whatever the accepted offer charges, and that is what lands in the
# result — never this table.
TIERS: dict[str, Tier] = {
    "RTX_5090": Tier("RTX_5090", 32.0, 0.34),
    "A100_SXM4": Tier("A100_SXM4", 80.0, 0.60),
    "L40S": Tier("L40S", 48.0, 0.80),
    "H100_SXM": Tier("H100_SXM", 80.0, 1.99),
}


class BudgetExhausted(RuntimeError):
    pass


def feasible(model: Model, tier: Tier) -> bool:
    """Whether the model fits the card with vLLM's default 0.9 utilisation."""
    return model.required_vram_gb <= tier.vram_gb * 0.9


def estimate_run_usd(model: Model, tier: Tier) -> float:
    hours = BASE_RUN_MINUTES * model.runtime_multiplier / 60.0
    return hours * tier.typical_hourly_usd


def build_matrix() -> list[Run]:
    """Every feasible (model, tier) three times, cheapest tier first."""
    runs: list[Run] = []
    for tier in sorted(TIERS.values(), key=lambda t: t.typical_hourly_usd):
        for model in MODELS.values():
            if not feasible(model, tier):
                continue
            for index in range(1, RUNS_PER_CONFIG + 1):
                runs.append(Run(model.key, tier.key, index))
    return runs


class BudgetGate:
    """Refuses a run the remaining credit cannot cover.

    A reserve is held back deliberately: spending to zero leaves nothing to
    re-rent a lost run, and a rental that fails for lack of funds mid-sweep is
    the worst outcome — a partial result set and possibly a live instance."""

    def __init__(self, credit_usd: float, reserve_usd: float = 1.00):
        self.remaining = credit_usd
        self.reserve = reserve_usd
        self.committed = 0.0

    def check(self, estimated_usd: float) -> None:
        if self.remaining - estimated_usd < self.reserve:
            raise BudgetExhausted(
                f"run needs ~${estimated_usd:.2f}, only "
                f"${self.remaining - self.reserve:.2f} spendable above the "
                f"${self.reserve:.2f} reserve"
            )
        self.remaining -= estimated_usd
        self.committed += estimated_usd


def matrix_cost_estimate() -> float:
    return sum(
        estimate_run_usd(MODELS[r.model_key], TIERS[r.tier_key])
        for r in build_matrix()
    )
