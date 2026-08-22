"""What the matrix actually has, versus what it planned.

Runs fail — two of the first four hosts never pulled the image — and the sweep
does not go back for them. Counting coverage by hand across four tiers and two
models is how a two-sample median ends up presented as three."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

from gppb.cost import saturated
from gppb.models import BenchResult

from launch.matrix import MODELS, RUNS_PER_CONFIG, Run, build_matrix

# A result records what nvidia-smi reported, which is not the string Vast was
# asked for: "H100_SXM" comes back as "NVIDIA H100 80GB HBM3". Each tier is
# identified by tokens that must be present and tokens that rule it out — an
# H100 PCIe is a different card with different interconnect, and quietly filing
# it under the SXM heading would corrupt the comparison the project exists for.
TIER_MARKERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "RTX_5090": (("RTX", "5090"), ()),
    # Half the VRAM is a different machine here: it changes what fits and how
    # much KV cache there is. The first live A100 run rented a 40GB card.
    "A100_SXM4": (("A100", "SXM4", "80GB"), ()),
    "L40S": (("L40S",), ()),
    "H100_SXM": (("H100",), ("PCIE", "NVL")),
}


def tier_for(gpu_name: str) -> str | None:
    """The tier a reported GPU name belongs to, or None when it is ambiguous.

    Returning None is deliberate: an unrecognised card means something was
    rented that the matrix did not ask for, and that deserves a report rather
    than a guess."""
    upper = gpu_name.upper().replace("-", " ").replace("_", " ")
    for tier, (required, forbidden) in TIER_MARKERS.items():
        if all(token in upper for token in required) and not any(
            token in upper for token in forbidden
        ):
            return tier
    return None


def _model_for(hf_id: str) -> str | None:
    for key, model in MODELS.items():
        if model.hf_id == hf_id:
            return key
    return None


def usable(result: BenchResult) -> bool:
    """Whether a result may count as one of a cell's samples.

    A partial upload is a preempted run, and a sweep still climbing at its last
    level never found the ceiling the headline number is quoted at. Neither is
    a sample."""
    return (
        result.valid
        and not result.partial
        and bool(result.steps)
        and saturated(result.steps)
    )


def coverage(results: list[BenchResult]) -> dict[tuple[str, str], int]:
    """Usable sample count per (tier, model) cell."""
    counts: Counter[tuple[str, str]] = Counter()
    for result in results:
        if not usable(result):
            continue
        tier = tier_for(result.hardware.gpu_name)
        model = _model_for(result.target.model)
        if tier is None or model is None:
            continue
        counts[(tier, model)] += 1
    return dict(counts)


def missing(
    results: list[BenchResult], runs_per_config: int = RUNS_PER_CONFIG
) -> list[Run]:
    """The planned runs still owed, in the matrix's own cheapest-first order."""
    have = coverage(results)
    owed: list[Run] = []
    for run in build_matrix():
        cell = (run.tier_key, run.model_key)
        banked = have.get(cell, 0) + sum(
            1 for r in owed if (r.tier_key, r.model_key) == cell
        )
        if banked < runs_per_config:
            # Numbered after what is already banked, not by position in the
            # matrix: a resumed sweep that plans a second "run 1" leaves two
            # results both claiming to be the first of three.
            owed.append(replace(run, run_index=banked + 1))
    return owed


if __name__ == "__main__":
    from pathlib import Path

    from report.generate import load_results

    found = load_results(Path("results"))
    have = coverage(found)
    print(f"{len(found)} results in results/")
    for run in build_matrix():
        cell = (run.tier_key, run.model_key)
        if run.run_index == 1:
            print(f"  {cell[0]:<11} {MODELS[cell[1]].hf_id:<22} "
                  f"{have.get(cell, 0)}/{RUNS_PER_CONFIG}")
    owed = missing(found)
    print(f"\n{len(owed)} runs still owed")
