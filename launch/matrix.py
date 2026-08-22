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
    # 46GB, not the 34 first estimated. Measured: the fp8 weights are 28.06GiB
    # and CUDA-graph capture needs more headroom than a 44.40GiB L40S has —
    # three rentals died in profile_cudagraph_memory with 59.31MiB free, and
    # the low estimate is what let the sweep keep paying to rediscover it.
    # See docs/l40s-27b-does-not-fit.md.
    "headline": Model("headline", "Qwen/Qwen3.8-27B", "fp8", 46.0, 1.6),
}

# Prices are indicative, for ordering and budgeting only. The rate actually
# paid is whatever the accepted offer charges, and that is what lands in the
# result — never this table.
#
# Observed on Vast 2026-08-21 as the cheapest offer meeting the bandwidth,
# reliability and VRAM floors. The earlier A100 and H100 figures were roughly
# half the real price, which made the budget gate optimistic by ~60% across
# the matrix — it would have approved runs the credit could not cover.
TIERS: dict[str, Tier] = {
    "RTX_5090": Tier("RTX_5090", 32.0, 0.34),
    "A100_SXM4": Tier("A100_SXM4", 80.0, 1.04),
    # The L40S is a 48GB part but every Vast host reports 45GB usable, and the
    # VRAM floor is checked against what the host reports. Declaring 48 here
    # rejected every L40S offer and failed the tier outright.
    "L40S": Tier("L40S", 45.0, 0.74),
    "H100_SXM": Tier("H100_SXM", 80.0, 2.93),
    # Blackwell server part, 96GB GDDR7 — every Vast host reports 95GB, so
    # that is what is declared. Eleven single-card offers on 2026-08-22 ran
    # $1.06-$1.81/hr; the median is what the budget reasons about.
    "RTX_PRO_6000_S": Tier("RTX_PRO_6000_S", 95.0, 1.45),
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
