"""Charts and tables from results/. Pure function of committed data."""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gppb.cost import (  # noqa: E402
    api_usd_per_mtok, coldstart_usd, saturated, selfhost_usd_per_mtok,
    throughput_knee,
)
from gppb.models import BenchResult  # noqa: E402


@dataclass
class CostRow:
    label: str
    usd_per_mtok: float
    tokens_per_sec: float
    coldstart_usd: float | None
    # The concurrency levels this run actually swept. Two runs that swept
    # different levels are different measurements and must never be averaged.
    levels: tuple[int, ...] = ()
    # False when the curve was still climbing at the last level tested.
    saturated: bool = True


def load_results(directory: Path) -> list[BenchResult]:
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        result = BenchResult.model_validate_json(path.read_text())
        if result.valid and result.steps:
            results.append(result)
    return results


def short_model(model: str) -> str:
    """The model name without its org prefix — "Qwen/Qwen3.8-27B" -> "Qwen3.8-27B"."""
    return model.rsplit("/", 1)[-1]


def row_label(result: BenchResult) -> str:
    """What was served, and on what.

    The label was the hardware alone, which made an A100 running the 8B and an
    A100 running the 27B one group: `median_rows` keys on the label, so the two
    were reduced to a single $/1M describing neither. The model is half the
    measurement and belongs in the identity of the row.
    """
    if result.target.kind == "openrouter":
        return f"{result.target.provider or 'unknown'} {short_model(result.target.model)}"
    return (
        f"{result.hardware.gpu_name} TP{result.target.tp_size} "
        f"{short_model(result.target.model)}"
    )


def cost_rows(results: list[BenchResult]) -> list[CostRow]:
    rows: list[CostRow] = []
    for result in results:
        knee = throughput_knee(result.steps)
        levels = tuple(step.concurrency for step in result.steps)
        is_saturated = saturated(result.steps)
        if result.target.kind == "openrouter":
            rows.append(CostRow(
                label=row_label(result),
                usd_per_mtok=api_usd_per_mtok(
                    result.pricing.input_per_mtok_usd or 0.0,
                    result.pricing.output_per_mtok_usd or 0.0,
                    result.workload.input_tokens,
                    result.workload.output_tokens,
                ),
                tokens_per_sec=knee.output_tokens_per_sec,
                coldstart_usd=None,
                levels=levels,
                saturated=is_saturated,
            ))
        else:
            rate = result.pricing.hourly_rate_usd or 0.0
            rows.append(CostRow(
                label=row_label(result),
                usd_per_mtok=selfhost_usd_per_mtok(rate, knee.output_tokens_per_sec),
                tokens_per_sec=knee.output_tokens_per_sec,
                coldstart_usd=coldstart_usd(result.timings.boot_seconds or 0.0, rate),
                levels=levels,
                saturated=is_saturated,
            ))
    return rows


def median_rows(rows: list[CostRow]) -> list[CostRow]:
    """Collapse the 3 runs per config to their median."""
    # Grouped by whether the run found its ceiling, not by the exact levels it
    # swept. What makes two runs incomparable is that one is a measurement and
    # the other an upper bound — and since the sweep stops two levels past its
    # own peak, repeat runs of one config legitimately stop at different
    # concurrencies. Keying on the level list would split those and quietly
    # produce a single-sample "median" of three runs.
    grouped: dict[tuple[str, bool], list[CostRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.label, row.saturated)].append(row)

    collapsed = []
    for (label, is_saturated), group in grouped.items():
        # An upper bound is always marked, whether or not a fuller run exists
        # to compare it against. The ceiling quoted is the lowest any run in
        # the group reached, since that is the weakest claim they jointly
        # support.
        levels = min((r.levels for r in group if r.levels), default=())
        if not is_saturated and levels:
            label = f"{label} (\u2264{levels[-1]})"
        collapsed.append(CostRow(
            label=label,
            levels=levels,
            saturated=is_saturated,
            usd_per_mtok=statistics.median(r.usd_per_mtok for r in group),
            tokens_per_sec=statistics.median(r.tokens_per_sec for r in group),
            coldstart_usd=(
                statistics.median(r.coldstart_usd for r in group)
                if group[0].coldstart_usd is not None else None
            ),
        ))
    return sorted(collapsed, key=lambda r: r.usd_per_mtok)


def throughput_curves(
    results: list[BenchResult],
) -> dict[str, list[tuple[int, float]]]:
    """Tokens/sec against concurrency, one curve per tier, median across runs.

    Levels where every request failed are dropped rather than plotted as zero:
    a failed level is missing data, and drawing it as a cliff invents a
    performance characteristic the hardware does not have."""
    by_label: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        label = row_label(result)
        for step in result.steps:
            if step.requests_completed == 0:
                continue
            by_label[label][step.concurrency].append(step.output_tokens_per_sec)

    curves: dict[str, list[tuple[int, float]]] = {}
    for label, levels in by_label.items():
        curves[label] = [
            (concurrency, statistics.median(samples))
            for concurrency, samples in sorted(levels.items())
        ]
    return curves


def render_throughput_chart(
    curves: dict[str, list[tuple[int, float]]], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, points in sorted(curves.items()):
        xs = [c for c, _ in points]
        ys = [t for _, t in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("output tokens/sec")
    ax.set_title("Throughput vs concurrency — the knee is where cost is quoted")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def render_markdown_table(rows: list[CostRow]) -> str:
    lines = [
        "| Target | $/1M output tokens | tokens/sec | cold start $ |",
        "|---|---|---|---|",
    ]
    for row in rows:
        cold = f"${row.coldstart_usd:.4f}" if row.coldstart_usd is not None else "n/a"
        lines.append(
            f"| {row.label} | ${row.usd_per_mtok:.4f} | {row.tokens_per_sec:.0f} | {cold} |"
        )
    return "\n".join(lines)


def render_cost_chart(rows: list[CostRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r.label for r in rows]
    values = [r.usd_per_mtok for r in rows]
    colors = ["#2563eb" if r.coldstart_usd is not None else "#dc2626" for r in rows]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("USD per 1M output tokens (lower is better)")
    ax.set_title("Qwen3.8-27B: self-hosted (blue) vs managed API (red)")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


if __name__ == "__main__":
    results = load_results(Path("results"))
    rows = median_rows(cost_rows(results))
    Path("report/out").mkdir(parents=True, exist_ok=True)
    render_cost_chart(rows, Path("report/out/cost.svg"))
    render_throughput_chart(throughput_curves(results), Path("report/out/throughput.svg"))
    Path("report/out/table.md").write_text(render_markdown_table(rows))

    # The thread is a build artefact like the charts, not a document someone
    # edits afterwards — regenerating it is the only way its numbers stay
    # consistent with the results directory.
    from report.thread import build_thread, render_thread
    Path("report/out/thread.txt").write_text(render_thread(build_thread(results)))

    print(render_markdown_table(rows))
    print("\nwrote report/out/{cost.svg,throughput.svg,table.md,thread.txt}")
