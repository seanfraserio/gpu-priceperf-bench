"""Deterministic prompt corpus. Same bytes on every GPU and every provider."""
from __future__ import annotations

import random

# A small closed vocabulary keeps tokenisation stable across tokenisers, and
# every word in it is a single token under Qwen's BPE. That is what lets
# `input_tokens` be quoted honestly: measured against Qwen/Qwen3-8B's
# tokenizer.json on 2026-08-21, a corpus configured for 1024 encodes to 1029
# tokens — a 1.00x ratio. Adding a multi-token word here would break that, and
# the published method note would start overstating the input length.
_VOCAB = [
    "system", "network", "latency", "kernel", "memory", "throughput", "buffer",
    "queue", "packet", "cluster", "tensor", "gradient", "storage", "compute",
    "pipeline", "scheduler", "cache", "bandwidth", "socket", "thread",
]


def build_corpus(n: int, input_tokens: int, seed: int = 1337) -> list[str]:
    """Build `n` distinct prompts of roughly `input_tokens` whitespace tokens.

    Distinct by construction — identical prompts would be served from residual
    caches and inflate measured throughput.
    """
    rng = random.Random(seed)
    prompts: list[str] = []
    for i in range(n):
        words = [rng.choice(_VOCAB) for _ in range(input_tokens - 4)]
        # Unique prefix guarantees distinctness without disturbing length much.
        prompts.append(f"Document {i}: " + " ".join(words) + " . Summarize.")
    return prompts
