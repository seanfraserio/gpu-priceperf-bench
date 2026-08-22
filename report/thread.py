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
from report.generate import CostRow, cost_rows, median_rows, short_model

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


def compact(label: str) -> str:
    """A row label short enough for a 280-character post.

    The full label carries the vendor prefix and a TP1 that is true of every
    single-GPU row; neither distinguishes anything, and with the model name now
    in the label the table post ran 42 characters over the limit and stopped
    rendering. Multi-GPU rows keep their TP marker, since there it is the point.
    """
    return label.replace("NVIDIA ", "").replace(" TP1 ", " ")


def _split(rows: list[CostRow]) -> tuple[list[CostRow], list[CostRow]]:
    """Self-hosted rows carry a cold-start cost; API rows cannot."""
    selfhost = [r for r in rows if r.coldstart_usd is not None]
    api = [r for r in rows if r.coldstart_usd is None]
    return selfhost, api


def build_thread(results: list[BenchResult]) -> list[Post]:
    """The thread, as a list of posts. Raises rather than inventing content."""
    # Only saturated runs may be quoted. A run whose throughput was still
    # climbing at its last level measured the sweep, not the hardware — on real
    # data the generator otherwise compared a 5090 against the same 5090 swept
    # less deeply and announced a 5.0x gap.
    # Saturation is a sell-side requirement. A GPU row that never found its
    # ceiling reports a floor on the hardware, so quoting its cost would
    # understate what the card can do — but a managed endpoint's $/1M is its
    # published rate card, exact whether or not the sweep found a plateau, and
    # none of them reach one before they rate-limit. Holding APIs to the GPU
    # rule deleted every API row and with it the verdict post.
    rows = [
        row for row in median_rows(cost_rows(results))
        if row.saturated or row.coldstart_usd is None
    ]
    if not rows:
        raise ValueError(
            "no run found its throughput ceiling — refusing to quote an upper "
            "bound as a measurement"
        )

    selfhost, api = _split(rows)
    model = results[0].target.model
    posts: list[Post] = []

    head = Post()
    # Cheapest and dearest are only comparable within one model. Ranked across
    # the whole matrix the pair was an 8B against a 27B, and the ratio between
    # them — announced as "8.0x more" — measured the difference between two
    # model sizes while reading as a difference between two rentals.
    # Case-folded: vLLM echoes the HuggingFace id ("Qwen/Qwen3.8-27B") and
    # OpenRouter lowercases its own, so whichever result happens to be first
    # decides the casing of the needle but not of the labels.
    needle = short_model(model).lower()
    headline_rows = [row for row in rows if needle in row.label.lower()] or rows
    best = headline_rows[0]
    head.text = (
        f"I rented GPUs and measured what it actually costs to serve "
        f"{head.borrow(model)}.\n\n"
        f"Cheapest: {head.borrow(compact(best.label))} at "
        f"${head.num(best.usd_per_mtok, 4)} per 1M output tokens"
    )
    if len(headline_rows) > 1:
        dearest = headline_rows[-1]
        ratio = dearest.usd_per_mtok / best.usd_per_mtok if best.usd_per_mtok else 0.0
        head.text += (
            f".\nDearest: {head.borrow(compact(dearest.label))}, "
            f"{head.num(ratio, 1)}x more."
        )
    posts.append(head)

    table = Post()
    # Deliberately not "$/1M, <model>:". The rows span every model measured,
    # and the header used to name whichever one happened to be first in the
    # results directory — a caption that was false for most of the table.
    lines = ["$/1M output tokens, cheapest first:"]
    for row in rows[:5]:
        lines.append(
            f"{table.borrow(compact(row.label))} — ${table.num(row.usd_per_mtok, 4)} "
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
            f"{cold.borrow(compact(cheapest_cold.label))} costs "
            f"${cold.num(cheapest_cold.coldstart_usd or 0.0, 4)} every cold "
            "start, before a single token is served."
        )
        posts.append(cold)

    # Both sides of the verdict must serve the same model. The post opens with
    # "same model, same workload", and it was pairing the cheapest self-host
    # row in the matrix — an 8B — against a 27B endpoint, announcing 71.7x
    # directly beneath a sentence promising that is not what it is.
    self_headline = [r for r in selfhost if needle in r.label.lower()] or selfhost
    api_headline = [r for r in api if needle in r.label.lower()] or api

    if self_headline and api_headline:
        verdict = Post()
        best_self = min(self_headline, key=lambda r: r.usd_per_mtok)
        best_api = min(api_headline, key=lambda r: r.usd_per_mtok)
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
            f"{verdict.borrow(compact(winner.label))} "
            f"${verdict.num(winner.usd_per_mtok, 4)}\n"
            f"{verdict.borrow(compact(loser.label))} "
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
