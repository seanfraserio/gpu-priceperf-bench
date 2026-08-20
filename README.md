# gpu-priceperf-bench

Price/performance measurements for **Qwen3.8-27B** — self-hosted on rented
NVIDIA GPUs versus the managed API providers serving it.

## The number

    $/1M output tokens = hourly_rate_usd / (output_tokens_per_sec * 3600) * 1e6

For metered APIs, input cost is folded in and normalised to output tokens so
both sit on one axis:

    $/1M output = (in_tok/1e6 * in_rate + out_tok/1e6 * out_rate) / out_tok * 1e6

## Method

- Fixed shape: 1024 input / 256 output tokens, `temperature=0`, `ignore_eos=true`.
- Concurrency swept 1 -> 256. Headline cost quoted at the throughput knee, never
  at concurrency 1.
- `--max-model-len 32768` everywhere, so no GPU wins on KV headroom alone.
- No speculative decoding, no prefix caching. Both are real production wins and
  both break cross-provider comparability.
- 3 runs per config, median reported with min/max.
- Cold start reported as its own cost line: `boot_seconds * $/hr`.

## Reproduce

    pip install -e ".[dev]"
    ./scripts/dryrun.sh        # free, proves the path
    python -m report.generate  # rebuild charts from committed results/

Raw results live in `results/`, append-only. Bad runs are marked
`"valid": false` with a reason rather than deleted.
