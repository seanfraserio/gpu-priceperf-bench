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

## Running a benchmark

Instances run the **stock** `vllm/vllm-openai` image — there is no image to
build or host. `runner-vllm/onstart.sh` is passed to the instance verbatim: it
arms a TTL self-destruct, clones this repo at a pinned revision, then serves
and sweeps. `GPPB_REF` selects the revision; a rented GPU never runs branch tip.

The multi-GPU interconnect run works the same way on the stock
`nvidia/cuda` devel image, compiling nccl-tests at a pinned tag on the box —
roughly 2-3 minutes of billed time, and nothing to host.

    pip install vastai
    vastai set api-key <key>      # from https://cloud.vast.ai/account/
    python -m launch.reap         # before and after every session

Instances power off the moment a run ends, so results only survive if they are
uploaded. `SINK_URL` + `SINK_TOKEN` point at the write-authenticated sink
Worker in `sink-worker/`, which stores each `BenchResult` in R2. Without them
the run writes to a container filesystem that is about to disappear.

Raw results live in `results/`, append-only. Bad runs are marked
`"valid": false` with a reason rather than deleted.
