"""Generate the X thread from results.

Nothing here composes a number by hand. Every figure that reaches the thread is
formatted through `Post.num`, which records it as a source, and the test suite
asserts that no decimal appears in the rendered text without a matching source.
That is the whole design: a benchmark thread is widely read and rarely checked,
so the only durable defence against a number that drifted in the author's
favour is that the author never typed one."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from gppb.models import BenchResult
from report.generate import CostRow, cost_rows, median_rows

# Room for the numbering prefix the renderer adds.
MAX_POST_CHARS = 280

_DECIMAL = re.compile(r"\d+\.\d+")


@dataclass
class Post:
    text: str = ""
    sources: set[str] = field(default_factory=set)

    def num(self, value: float, places: int = 2) -> str:
        """Format a measured value and record it as traceable."""
        token = f"{value:.{places}f}"
        self.sources.add(token)
        return token

    def borrow(self, text: str) -> str:
        """Embed a string that came from a result (a GPU name, a model id, a
        vLLM version). Its digits are traceable too — they were reported by the
        run, not chosen here."""
        self.sources.update(_DECIMAL.findall(text))
        return text


def _split(rows: list[CostRow]) -> tuple[list[CostRow], list[CostRow]]:
    """Self-hosted rows carry a cold-start cost; API rows cannot."""
    selfhost = [r for r in rows if r.coldstart_usd is not None]
    api = [r for r in rows if r.coldstart_usd is None]
    return selfhost, api


def build_thread(results: list[BenchResult]) -> list[Post]:
    """The thread, as a list of posts. Raises rather than inventing content."""
    rows = median_rows(cost_rows(results))
    if not rows:
        raise ValueError("no results to write a thread from — refusing to draft one")

    selfhost, api = _split(rows)
    model = results[0].target.model
    posts: list[Post] = []

    head = Post()
    best = rows[0]
    head.text = (
        f"I rented GPUs and measured what it actually costs to serve "
        f"{head.borrow(model)}.\n\n"
        f"Cheapest: {head.borrow(best.label)} at "
        f"${head.num(best.usd_per_mtok, 4)} per 1M output tokens"
    )
    if len(rows) > 1:
        dearest = rows[-1]
        ratio = dearest.usd_per_mtok / best.usd_per_mtok if best.usd_per_mtok else 0.0
        head.text += (
            f".\nDearest: {head.borrow(dearest.label)}, "
            f"{head.num(ratio, 1)}x more."
        )
    posts.append(head)

    table = Post()
    lines = [f"$/1M output tokens, {table.borrow(model)}:"]
    for row in rows[:5]:
        lines.append(
            f"{table.borrow(row.label)} — ${table.num(row.usd_per_mtok, 4)} "
            f"@ {table.num(row.tokens_per_sec, 0)} tok/s"
        )
    table.text = "\n".join(lines)
    posts.append(table)

    if selfhost:
        cold = Post()
        cheapest_cold = min(selfhost, key=lambda r: r.coldstart_usd or 0.0)
        cold.text = (
            "Self-hosting has a cost the API bill never shows: you pay for the "
            "boot.\n\n"
            f"{cold.borrow(cheapest_cold.label)} costs "
            f"${cold.num(cheapest_cold.coldstart_usd or 0.0, 4)} every cold "
            "start, before a single token is served."
        )
        posts.append(cold)

    if selfhost and api:
        verdict = Post()
        best_self = min(selfhost, key=lambda r: r.usd_per_mtok)
        best_api = min(api, key=lambda r: r.usd_per_mtok)
        cheaper = best_self.usd_per_mtok < best_api.usd_per_mtok
        gap = (
            best_api.usd_per_mtok / best_self.usd_per_mtok if cheaper
            else best_self.usd_per_mtok / best_api.usd_per_mtok
        ) if best_self.usd_per_mtok and best_api.usd_per_mtok else 0.0
        winner, loser = (
            (best_self, best_api) if cheaper else (best_api, best_self)
        )
        verdict.text = (
            f"Self-host vs managed API, same model, same workload:\n\n"
            f"{verdict.borrow(winner.label)} "
            f"${verdict.num(winner.usd_per_mtok, 4)}\n"
            f"{verdict.borrow(loser.label)} "
            f"${verdict.num(loser.usd_per_mtok, 4)}\n\n"
            f"{verdict.num(gap, 1)}x apart — and that ignores the hours you "
            "spend running it."
        )
        posts.append(verdict)

    method = Post()
    workload = results[0].workload
    method.text = (
        "Method: vLLM on rented GPUs, "
        f"{method.num(workload.input_tokens, 0)} in / "
        f"{method.num(workload.output_tokens, 0)} out, prefix caching off, "
        "distinct prompts, cost taken at the throughput knee.\n\n"
        "Rented-GPU pricing moves hourly and run-to-run variance is real. "
        "Numbers, code and raw results are public — check them."
    )
    posts.append(method)

    for post in posts:
        if len(post.text) > MAX_POST_CHARS:
            raise ValueError(
                f"generated post is {len(post.text)} chars, over the "
                f"{MAX_POST_CHARS} limit:\n{post.text}"
            )
    return posts


def render_thread(posts: list[Post]) -> str:
    total = len(posts)
    return "\n\n---\n\n".join(
        f"{i}/{total}\n{post.text}" for i, post in enumerate(posts, start=1)
    )
